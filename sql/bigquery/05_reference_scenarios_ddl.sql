-- =============================================================================
-- Stage 05 - reference_scenarios DDL and seed
-- Purpose: hold the cost and margin assumptions the economics view crosses
--          against targeting levels, so scenarios are data (auditable, easy to
--          extend) instead of numbers buried in a view definition.
-- Inputs:  none - values come from configs/experiment.yaml (economics block).
-- Outputs: campaign_incrementality.reference_scenarios
-- Assumptions: THE HILLSTROM DATASET CARRIES NO COST AND NO MARGIN DATA. Every
--          row here is a stated modelling assumption, not an observed unit
--          economics, and nothing downstream of it is realised ROI. Base case
--          is $0.005 per send at 45% gross margin; the grid is the three cost
--          and three margin scenarios in the config.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
--          Idempotent: transactional DELETE/INSERT replaces the grid.
-- =============================================================================

CREATE TABLE IF NOT EXISTS `campaign_incrementality.reference_scenarios` (
  scenario_id         STRING,   -- stable key, e.g. cost_0.005_margin_0.45
  scenario_label      STRING,   -- human label for the dashboard filter
  email_cost_per_send FLOAT64,  -- modelled cost of one campaign e-mail
  gross_margin_rate   FLOAT64,  -- modelled margin on incremental revenue
  is_base_case        BOOL,     -- the scenario the write-up quotes
  assumption_note     STRING    -- repeats that these are modelled, not realised
)
OPTIONS(
  description = 'Modelled cost and margin assumptions for scenario analysis. The Hillstrom experiment records no spend on e-mail and no product margin, so these are assumptions, never realised economics.'
);

BEGIN TRANSACTION;
DELETE FROM `campaign_incrementality.reference_scenarios` WHERE TRUE;
INSERT INTO `campaign_incrementality.reference_scenarios` (
  scenario_id, scenario_label, email_cost_per_send, gross_margin_rate, is_base_case, assumption_note
)
SELECT
  CONCAT('cost_', CAST(CAST(ROUND(cost * 1000) AS INT64) AS STRING), 'm_margin_',
         CAST(CAST(ROUND(margin * 100) AS INT64) AS STRING), 'pct') AS scenario_id,
  CONCAT('$', CAST(ROUND(cost, 3) AS STRING), ' per send x ',
         CAST(CAST(ROUND(margin * 100) AS INT64) AS STRING), '% margin') AS scenario_label,
  cost AS email_cost_per_send, margin AS gross_margin_rate,
  (cost = 0.005 AND margin = 0.45) AS is_base_case,
  'Modelled assumption: no cost or margin data exists in the Hillstrom file, so this is not realised ROI.'
FROM UNNEST([0.002, 0.005, 0.010]) AS cost
CROSS JOIN UNNEST([0.30, 0.45, 0.60]) AS margin;

COMMIT TRANSACTION;
