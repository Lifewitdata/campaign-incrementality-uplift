-- =============================================================================
-- Stage 09 - views_dashboard_economics (targeting level x cost/margin scenario)
-- Purpose: show what targeting the top k% by uplift score would be worth under
--          each modelled cost and margin assumption, crossed with the seven
--          budget levels from configs/experiment.yaml (5, 10, 20, 30, 50, 75,
--          100% of customers mailed).
-- Inputs:  campaign_incrementality.uplift_customer_scores (stage 06),
--          campaign_incrementality.curated_campaign_customers (stage 03),
--          campaign_incrementality.reference_scenarios (stage 05).
-- Outputs: campaign_incrementality.dashboard_targeting_scenario_economics -
--          one row per comparison x split x targeting level x scenario.
-- Assumptions: THESE ARE MODELLED SCENARIOS, NOT REALISED ROI. The Hillstrom
--          experiment records no e-mail cost and no product margin, so every
--          dollar column below is incremental revenue x an assumed margin minus
--          assumed sends x an assumed cost. Incremental revenue and conversions
--          use the same control-scaled counterfactual as the Qini gain in stage
--          08, measured inside one split, then scaled to the mailed share; they
--          are not extrapolated to the full customer base. Ranking ties use exported
--          seeded Python ranks. Costs and revenue both
--          refer to the historical treated subset, not the entire selected list.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
-- =============================================================================

CREATE OR REPLACE VIEW `campaign_incrementality.dashboard_targeting_scenario_economics` AS
WITH scored AS (
  SELECT
    s.comparison, s.split, s.uplift_score, s.record_id, s.rank_position,
    (c.arm_code = CASE s.comparison WHEN 'mens_vs_control' THEN 'mens'
                                    WHEN 'womens_vs_control' THEN 'womens' END) AS is_treated,
    (c.arm_code = 'control') AS is_control,
    c.conversion, c.spend
  FROM `campaign_incrementality.uplift_customer_scores` AS s
  JOIN `campaign_incrementality.curated_campaign_customers` AS c ON c.record_id = s.record_id
  WHERE s.comparison IN ('mens_vs_control', 'womens_vs_control') AND s.uplift_score IS NOT NULL
),
ranked AS (
  -- The pool that gets ranked (and mailed) is the treated arm of this
  -- comparison plus the shared control arm, within one split.
  SELECT
    comparison, split, is_treated, is_control, conversion, spend,
    COUNT(*) OVER (PARTITION BY comparison, split) AS comparison_pool_customers,
    rank_position
  FROM scored
  WHERE is_treated OR is_control
),
budget_levels AS (
  SELECT targeting_share, CAST(ROUND(targeting_share * 100) AS INT64) AS targeting_pct
  FROM UNNEST([0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]) AS targeting_share
),
targeted AS (
  SELECT
    levels.targeting_share, levels.targeting_pct, pool.comparison, pool.split,
    pool.is_treated, pool.is_control, pool.conversion, pool.spend, pool.comparison_pool_customers
  FROM ranked AS pool
  CROSS JOIN budget_levels AS levels
  WHERE pool.rank_position <= CAST(ROUND(CAST(levels.targeting_share AS NUMERIC) * pool.comparison_pool_customers, 0, 'ROUND_HALF_EVEN') AS INT64)
),
outcomes AS (
  SELECT
    comparison, split, targeting_share, targeting_pct,
    COUNT(*) AS targeted_customers_in_slice,
    MAX(comparison_pool_customers) AS comparison_pool_customers,
    SUM(IF(is_treated, 1, 0)) AS treated_targeted_customers,
    SUM(IF(is_control, 1, 0)) AS control_targeted_customers,
    SUM(IF(is_treated, conversion, 0)) AS treated_targeted_conversions,
    SUM(IF(is_control, conversion, 0)) AS control_targeted_conversions,
    SUM(IF(is_treated, spend, 0.0)) AS treated_targeted_spend,
    SUM(IF(is_control, spend, 0.0)) AS control_targeted_spend
  FROM targeted
  GROUP BY comparison, split, targeting_share, targeting_pct
),
counterfactual AS (
  SELECT
    comparison, split, targeting_share, targeting_pct, targeted_customers_in_slice,
    comparison_pool_customers, treated_targeted_customers, control_targeted_customers,
    treated_targeted_conversions, control_targeted_conversions,
    treated_targeted_spend - control_targeted_spend
      * SAFE_DIVIDE(treated_targeted_customers, control_targeted_customers) AS modelled_incremental_revenue,
    treated_targeted_conversions - control_targeted_conversions
      * SAFE_DIVIDE(treated_targeted_customers, control_targeted_customers) AS modelled_incremental_conversions,
    treated_targeted_customers AS modelled_emails_sent
  FROM outcomes
),
scenario_applied AS (
  SELECT
    base.comparison, base.split, base.targeting_pct, base.targeting_share,
    CONCAT('top_', CAST(base.targeting_pct AS STRING), 'pct_by_uplift_score') AS targeting_level,
    scen.scenario_id, scen.scenario_label, scen.email_cost_per_send, scen.gross_margin_rate,
    scen.is_base_case, base.modelled_emails_sent, base.modelled_incremental_conversions,
    base.modelled_incremental_revenue, base.comparison_pool_customers,
    base.targeted_customers_in_slice, base.treated_targeted_customers,
    base.control_targeted_customers,
    base.modelled_incremental_revenue * scen.gross_margin_rate AS modelled_incremental_margin,
    CAST(base.modelled_emails_sent AS FLOAT64) * scen.email_cost_per_send AS modelled_campaign_cost
  FROM counterfactual AS base
  CROSS JOIN `campaign_incrementality.reference_scenarios` AS scen
)
SELECT
  comparison, split, targeting_pct, targeting_share, targeting_level,
  scenario_id, scenario_label, is_base_case, email_cost_per_send, gross_margin_rate,
  comparison_pool_customers, targeted_customers_in_slice,
  treated_targeted_customers, control_targeted_customers,
  modelled_emails_sent, modelled_incremental_conversions, modelled_incremental_revenue,
  modelled_incremental_margin, modelled_campaign_cost,
  modelled_incremental_margin - modelled_campaign_cost AS modelled_net_contribution_scenario,
  SAFE_DIVIDE(modelled_incremental_margin, modelled_campaign_cost) AS modelled_margin_cover_ratio_scenario,
  SAFE_DIVIDE(modelled_campaign_cost, modelled_incremental_conversions) AS modelled_cost_per_incremental_conversion,
  SAFE_DIVIDE(modelled_incremental_margin, CAST(modelled_emails_sent AS FLOAT64)) AS modelled_break_even_cost_per_send,
  'Modelled scenario, not realised ROI: the Hillstrom data carries no e-mail cost and no product margin.' AS scenario_caveat
FROM scenario_applied;
