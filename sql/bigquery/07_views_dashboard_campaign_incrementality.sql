-- Purpose: Hillstrom arm summaries and authoritative Python effect evidence.
-- Inputs: curated_campaign_customers, reference_arms, evidence_campaign_incrementality_effects.
-- Outputs: dashboard_campaign_incrementality and dashboard_campaign_incrementality_effects.
-- Assumptions: the imported evidence retains Python methods, Holm p-values and
-- family-adjusted intervals. SQL does not independently recalculate inference.
-- NOT EXECUTED against live BigQuery: retain the deployment receipt after execution.

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

-- Use the exact Python effect estimates, adjusted intervals and Holm p-values.
-- This avoids presenting a different confidence convention in Tableau.
CREATE OR REPLACE VIEW `campaign_incrementality.dashboard_campaign_incrementality_effects` AS
SELECT comparison, outcome, measure, effect, effect_type, mean_treatment,
       mean_control, n_treatment, n_control, ci_low_fwer, ci_high_fwer,
       ci_level_fwer, p_holm, reject_holm, method, confirmatory
FROM `campaign_incrementality.evidence_campaign_incrementality_effects`;
