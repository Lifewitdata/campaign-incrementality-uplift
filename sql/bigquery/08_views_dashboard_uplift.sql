-- =============================================================================
-- Stage 08 - views_dashboard_uplift (decile rollup Tableau actually reads)
-- Purpose: turn per-customer uplift scores into a per-decile treated/control
--          read: conversion counts and rates, cumulative uplift and a
--          Qini-style cumulative gain, so a dashboard can show whether ranking
--          customers by predicted uplift beats mailing everyone.
-- Inputs:  campaign_incrementality.uplift_customer_scores (stage 06),
--          campaign_incrementality.curated_campaign_customers (stage 03).
-- Outputs: campaign_incrementality.dashboard_uplift_deciles - one row per
--          comparison x split x decile (decile 1 = strongest predicted uplift).
--          campaign_incrementality.dashboard_uplift_summary - one row per
--          comparison x split.
-- Assumptions: scores are joined on record_id, a row-position load key and NOT a
--          customer key (the Hillstrom file ships no identifier). Only the
--          treated arm of each comparison and the shared control arm enter a
--          decile, so control customers are reused across comparisons but never
--          across splits. THIS IS THE AGGREGATE LAYER: no per-customer PII is
--          shipped to the dashboard - no record_id and no individual score
--          leaves this view, only decile-level counts and score bounds.
--          cumulative_qini_gain_conversions uses the same accumulation as the
--          Python targeting curve: cum_treated_conversions - cum_control_
--          conversions * (cum_treated_customers / cum_control_customers).
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
-- =============================================================================

CREATE OR REPLACE VIEW `campaign_incrementality.dashboard_uplift_deciles` AS
WITH scored AS (
  SELECT
    s.comparison, s.split, s.decile, s.uplift_score, s.propensity_score,
    c.visit, c.conversion, c.spend,
    (c.arm_code = CASE s.comparison WHEN 'mens_vs_control' THEN 'mens'
                                    WHEN 'womens_vs_control' THEN 'womens' END) AS is_treated,
    (c.arm_code = 'control') AS is_control
  FROM `campaign_incrementality.uplift_customer_scores` AS s
  JOIN `campaign_incrementality.curated_campaign_customers` AS c ON c.record_id = s.record_id
  WHERE s.comparison IN ('mens_vs_control', 'womens_vs_control') AND s.decile BETWEEN 1 AND 10
),
decile_stats AS (
  SELECT
    comparison, split, decile,
    COUNT(*) AS scored_customers,
    MIN(uplift_score) AS uplift_score_min, MAX(uplift_score) AS uplift_score_max,
    AVG(uplift_score) AS uplift_score_avg, AVG(propensity_score) AS propensity_score_avg,
    SUM(IF(is_treated, 1, 0)) AS treated_customers,
    SUM(IF(is_control, 1, 0)) AS control_customers,
    SUM(IF(is_treated, conversion, 0)) AS treated_conversions,
    SUM(IF(is_control, conversion, 0)) AS control_conversions,
    SUM(IF(is_treated, visit, 0)) AS treated_visits,
    SUM(IF(is_control, visit, 0)) AS control_visits,
    SUM(IF(is_treated, spend, 0.0)) AS treated_spend,
    SUM(IF(is_control, spend, 0.0)) AS control_spend
  FROM scored
  WHERE is_treated OR is_control
  GROUP BY comparison, split, decile
),
cumulative AS (
  SELECT
    comparison, split, decile, scored_customers, uplift_score_min, uplift_score_max,
    uplift_score_avg, propensity_score_avg, treated_customers, control_customers,
    treated_conversions, control_conversions, treated_visits, control_visits,
    treated_spend, control_spend,
    SUM(treated_customers) OVER (PARTITION BY comparison, split ORDER BY decile ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_treated_customers,
    SUM(control_customers) OVER (PARTITION BY comparison, split ORDER BY decile ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_control_customers,
    SUM(treated_conversions) OVER (PARTITION BY comparison, split ORDER BY decile ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_treated_conversions,
    SUM(control_conversions) OVER (PARTITION BY comparison, split ORDER BY decile ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_control_conversions,
    SUM(treated_spend) OVER (PARTITION BY comparison, split ORDER BY decile ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_treated_spend,
    SUM(control_spend) OVER (PARTITION BY comparison, split ORDER BY decile ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_control_spend,
    SUM(scored_customers) OVER (PARTITION BY comparison, split) AS comparison_scored_customers
  FROM decile_stats
)
SELECT
  comparison, split, decile, scored_customers, uplift_score_min, uplift_score_max,
  uplift_score_avg, propensity_score_avg,
  treated_customers, control_customers, treated_conversions, control_conversions,
  SAFE_DIVIDE(treated_conversions, treated_customers) AS treated_conversion_rate,
  SAFE_DIVIDE(control_conversions, control_customers) AS control_conversion_rate,
  SAFE_DIVIDE(treated_conversions, treated_customers)
    - SAFE_DIVIDE(control_conversions, control_customers) AS decile_conversion_uplift,
  SAFE_DIVIDE(treated_visits, treated_customers) AS treated_visit_rate,
  SAFE_DIVIDE(control_visits, control_customers) AS control_visit_rate,
  cum_treated_customers, cum_control_customers, cum_treated_conversions, cum_control_conversions,
  SAFE_DIVIDE(cum_treated_conversions, cum_treated_customers) AS cumulative_treated_conversion_rate,
  SAFE_DIVIDE(cum_control_conversions, cum_control_customers) AS cumulative_control_conversion_rate,
  SAFE_DIVIDE(cum_treated_conversions, cum_treated_customers)
    - SAFE_DIVIDE(cum_control_conversions, cum_control_customers) AS cumulative_uplift,
  SAFE_DIVIDE(cum_treated_customers, cum_control_customers) AS control_scaling_factor,
  cum_treated_conversions
    - cum_control_conversions * SAFE_DIVIDE(cum_treated_customers, cum_control_customers) AS cumulative_qini_gain_conversions,
  cum_treated_spend
    - cum_control_spend * SAFE_DIVIDE(cum_treated_customers, cum_control_customers) AS cumulative_qini_gain_spend,
  SAFE_DIVIDE(cum_treated_customers + cum_control_customers, comparison_scored_customers) AS cumulative_targeted_share,
  comparison_scored_customers,
  CAST(2 AS INT64) AS outcome_window_weeks
FROM cumulative;

-- Headline read for the uplift page: does ranking beat mailing everyone? The
-- random-targeting benchmark is 10% of the full-population Qini gain (decile 1
-- is 10% of customers by construction), so qini_vs_random_targeting_ratio
-- above 1 means the score is genuinely sorting on incrementality.
CREATE OR REPLACE VIEW `campaign_incrementality.dashboard_uplift_summary` AS
SELECT
  comparison, split,
  MAX(IF(decile = 10, comparison_scored_customers, NULL)) AS scored_customers,
  MAX(IF(decile = 1, cumulative_targeted_share, NULL)) AS top_decile_targeted_share,
  MAX(IF(decile = 1, uplift_score_min, NULL)) AS top_decile_score_cutoff,
  MAX(IF(decile = 1, decile_conversion_uplift, NULL)) AS top_decile_conversion_uplift,
  MAX(IF(decile = 1, cumulative_uplift, NULL)) AS cumulative_uplift_at_top_decile,
  MAX(IF(decile = 1, cumulative_qini_gain_conversions, NULL)) AS qini_gain_conversions_top_decile,
  MAX(IF(decile = 10, cumulative_qini_gain_conversions, NULL)) AS qini_gain_conversions_all_customers,
  SAFE_DIVIDE(MAX(IF(decile = 1, cumulative_qini_gain_conversions, NULL)),
              NULLIF(MAX(IF(decile = 10, cumulative_qini_gain_conversions, NULL)), 0)) AS qini_gain_captured_by_top_decile_share,
  SAFE_DIVIDE(MAX(IF(decile = 1, cumulative_qini_gain_conversions, NULL)),
              NULLIF(0.1 * MAX(IF(decile = 10, cumulative_qini_gain_conversions, NULL)), 0)) AS qini_vs_random_targeting_ratio,
  MAX(IF(decile = 1, cumulative_qini_gain_spend, NULL)) AS qini_gain_spend_top_decile,
  MAX(IF(decile = 10, cumulative_qini_gain_spend, NULL)) AS qini_gain_spend_all_customers
FROM `campaign_incrementality.dashboard_uplift_deciles`
GROUP BY comparison, split;
