-- =============================================================================
-- Stage 01 - reference_arms dimension DDL and seed
-- Purpose: one small hand-maintained dimension describing the three randomised
--          arms, so dashboards join for labels and roles instead of hard-coding
--          strings in every view.
-- Inputs:  none - values are transcribed from the published Hillstrom design and
--          from configs/experiment.yaml (experiment.arms).
-- Outputs: campaign_incrementality.reference_arms
-- Assumptions: 'Womens E-Mail' and 'Mens E-Mail' name the MERCHANDISE LINE in
--          the creative, not the customer's gender. The dataset has no gender
--          field at all, and the mens/womens flags on the fact table record
--          pre-campaign purchase behaviour. Arm labels are never people
--          segments; do not title a chart 'men' or 'women'.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
--          Idempotent by design: INSERT OVERWRITE replaces the three rows.
-- =============================================================================

CREATE TABLE IF NOT EXISTS `campaign_incrementality.reference_arms` (
  arm_code    STRING,  -- control | mens | womens
  arm_label   STRING,  -- label exactly as published in the segment column
  role        STRING,  -- control | treatment
  arm_rank    INT64,   -- stable display order for the dashboard
  description STRING
)
OPTIONS(
  description = 'Randomised arm dimension for the Hillstrom e-mail experiment. Arm names are product lines, not customer genders.'
);

INSERT OVERWRITE `campaign_incrementality.reference_arms` (
  arm_code, arm_label, role, arm_rank, description
)
SELECT
  'control' AS arm_code, 'No E-Mail' AS arm_label, 'control' AS role,
  CAST(1 AS INT64) AS arm_rank,
  'Randomised control: no campaign e-mail was sent. Baseline for every incrementality comparison; 21,306 customers.'
UNION ALL
SELECT
  'mens', 'Mens E-Mail', 'treatment', CAST(2 AS INT64),
  'Treatment: customers sent an e-mail featuring the mens merchandise LINE. The name describes the creative, not the recipient; 21,307 customers.'
UNION ALL
SELECT
  'womens', 'Womens E-Mail', 'treatment', CAST(3 AS INT64),
  'Treatment: customers sent an e-mail featuring the womens merchandise LINE. The name describes the creative, not the recipient; 21,387 customers.';
