-- =============================================================================
-- Stage 07 - views_dashboard_campaign_incrementality (headline A/B read model)
-- Purpose: what the campaign did on average, per arm, and the incrementality of
--          each e-mail arm against the no-e-mail control with 95% intervals.
--          This is the untargeted "average treatment effect" layer.
-- Inputs:  campaign_incrementality.curated_campaign_customers (stage 03),
--          campaign_incrementality.reference_arms (stage 01).
-- Outputs: campaign_incrementality.dashboard_campaign_incrementality - one row
--          per arm: counts, visit rate, conversion rate, mean spend per
--          customer, mean spend per visitor.
--          campaign_incrementality.dashboard_campaign_incrementality_effects -
--          one row per email arm vs control with difference, standard
--          normal-approximation 95% CI and significance flag.
-- Assumptions: the Hillstrom arms are randomised, so the no-e-mail control is a
--          valid counterfactual. Proportion CI uses the unpooled SE
--          SQRT(p1(1-p1)/n1 + p2(1-p2)/n2); mean-spend CI uses the Welch
--          (unpooled) SE SQRT(var1/n1 + var2/n2), both with z = 1.96. Spend is
--          zero-inflated, so the spend interval is approximate - the bootstrap
--          intervals in the Python layer are the ones to quote in the write-up.
--          Outcomes are a two-week window only.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
-- =============================================================================

CREATE OR REPLACE VIEW `campaign_incrementality.dashboard_campaign_incrementality` AS
WITH arm_stats AS (
  SELECT
    c.arm_code,
    c.segment AS arm_label,
    COUNT(*) AS customers,
    SUM(c.visit) AS visits,
    SUM(c.conversion) AS conversions,
    COUNTIF(c.spend > 0) AS paying_customers,
    SUM(c.spend) AS spend_total,
    AVG(c.visit) AS visit_rate,
    AVG(c.conversion) AS conversion_rate,
    AVG(c.spend) AS mean_spend_per_customer,
    VAR_SAMP(c.spend) AS spend_variance,
    SAFE_DIVIDE(SUM(c.spend), SUM(c.visit)) AS mean_spend_per_visitor,
    MAX(c.campaign_weeks) AS outcome_window_weeks
  FROM `campaign_incrementality.curated_campaign_customers` AS c
  GROUP BY c.arm_code, c.segment
)
SELECT
  stats.arm_code, stats.arm_label, arms.role, arms.arm_rank,
  stats.customers, stats.visits, stats.conversions, stats.paying_customers,
  stats.visit_rate, stats.conversion_rate, stats.spend_total,
  stats.mean_spend_per_customer, stats.mean_spend_per_visitor,
  stats.spend_variance, stats.outcome_window_weeks
FROM arm_stats AS stats
LEFT JOIN `campaign_incrementality.reference_arms` AS arms ON arms.arm_code = stats.arm_code;

CREATE OR REPLACE VIEW `campaign_incrementality.dashboard_campaign_incrementality_effects` AS
WITH arm_stats AS (
  SELECT
    c.arm_code,
    c.segment AS arm_label,
    COUNT(*) AS customers,
    AVG(c.visit) AS visit_rate,
    AVG(c.conversion) AS conversion_rate,
    AVG(c.spend) AS mean_spend_per_customer,
    VAR_SAMP(c.spend) AS spend_variance
  FROM `campaign_incrementality.curated_campaign_customers` AS c
  GROUP BY c.arm_code, c.segment
),
effects AS (
  SELECT
    CONCAT(treated.arm_code, '_vs_control') AS comparison,
    treated.arm_code AS treatment_arm_code,
    treated.arm_label AS treatment_arm_label,
    control.arm_code AS control_arm_code,
    control.arm_label AS control_arm_label,
    treated.customers AS treated_customers,
    control.customers AS control_customers,
    treated.visit_rate AS treated_visit_rate,
    control.visit_rate AS control_visit_rate,
    treated.conversion_rate AS treated_conversion_rate,
    control.conversion_rate AS control_conversion_rate,
    treated.mean_spend_per_customer AS treated_mean_spend_per_customer,
    control.mean_spend_per_customer AS control_mean_spend_per_customer,
    treated.visit_rate - control.visit_rate AS visit_rate_difference,
    1.96 * SQRT(treated.visit_rate * (1 - treated.visit_rate) / treated.customers
              + control.visit_rate * (1 - control.visit_rate) / control.customers) AS visit_rate_ci_half_width,
    treated.conversion_rate - control.conversion_rate AS conversion_rate_difference,
    1.96 * SQRT(treated.conversion_rate * (1 - treated.conversion_rate) / treated.customers
              + control.conversion_rate * (1 - control.conversion_rate) / control.customers) AS conversion_rate_ci_half_width,
    treated.mean_spend_per_customer - control.mean_spend_per_customer AS mean_spend_difference,
    1.96 * SQRT(treated.spend_variance / treated.customers
              + control.spend_variance / control.customers) AS mean_spend_ci_half_width
  FROM arm_stats AS treated
  JOIN arm_stats AS control ON control.arm_code = 'control'
  WHERE treated.arm_code <> 'control'
)
SELECT
  comparison, treatment_arm_code, treatment_arm_label, control_arm_code, control_arm_label,
  treated_customers, control_customers,
  treated_visit_rate, control_visit_rate, visit_rate_difference,
  visit_rate_ci_half_width,
  visit_rate_difference - visit_rate_ci_half_width AS visit_rate_ci_low_95pct,
  visit_rate_difference + visit_rate_ci_half_width AS visit_rate_ci_high_95pct,
  ABS(visit_rate_difference) > visit_rate_ci_half_width AS visit_difference_significant_95pct,
  treated_conversion_rate, control_conversion_rate, conversion_rate_difference,
  conversion_rate_ci_half_width,
  conversion_rate_difference - conversion_rate_ci_half_width AS conversion_rate_ci_low_95pct,
  conversion_rate_difference + conversion_rate_ci_half_width AS conversion_rate_ci_high_95pct,
  ABS(conversion_rate_difference) > conversion_rate_ci_half_width AS conversion_difference_significant_95pct,
  SAFE_DIVIDE(conversion_rate_difference, control_conversion_rate) AS conversion_relative_lift,
  treated_mean_spend_per_customer, control_mean_spend_per_customer, mean_spend_difference,
  mean_spend_ci_half_width,
  mean_spend_difference - mean_spend_ci_half_width AS mean_spend_ci_low_95pct,
  mean_spend_difference + mean_spend_ci_half_width AS mean_spend_ci_high_95pct,
  ABS(mean_spend_difference) > mean_spend_ci_half_width AS mean_spend_difference_significant_95pct,
  SAFE_DIVIDE(mean_spend_difference, control_mean_spend_per_customer) AS mean_spend_relative_lift
FROM effects;
