-- =============================================================================
-- Stage 06 - uplift_customer_scores DDL (contract for the scored table)
-- Purpose: define the table the modelling job writes per-customer uplift scores
--          into, because the uplift and economics dashboards read it. This is
--          the contract, not a data load: scores arrive by `bq load` or a
--          python client write (README).
-- Inputs:  model output for the Hillstrom arms: one row per scored customer per
--          comparison per split.
-- Outputs: campaign_incrementality.uplift_customer_scores
-- Assumptions: record_id joins to curated_campaign_customers and is a row
--          position in the published file, NOT a customer key - the dataset
--          ships no identifier, so a score can never be attached to a person.
--          decile is 1-10 within (comparison, split), decile 1 = highest
--          predicted uplift. comparison values match configs/experiment.yaml:
--          mens_vs_control (primary), womens_vs_control. propensity_score is
--          P(treated | features) and is carried for diagnostics only.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
--
-- Column mapping from the modelling job (campaign_ml.uplift.score_frame emits
-- record_id, comparison, learner, uplift_score, response_score for the held-out
-- split only). To land it here, pick one learner per (comparison, split) and
-- carry: uplift_score -> uplift_score, the propensity learner's response_score
-- -> propensity_score, split -> the split that was scored ('test' today), and
-- decile = NTILE(10) over uplift_score descending within (comparison, split),
-- matching campaign_ml.evaluation.assign_deciles (decile 1 = strongest uplift).
-- A warehouse view must never invent scores: if this table is empty, stages 08
-- and 09 return zero rows, which is the correct honest dashboard state.
-- =============================================================================

CREATE TABLE IF NOT EXISTS `campaign_incrementality.uplift_customer_scores` (
  record_id        INT64,   -- join key to the curated table; NOT a customer key
  comparison       STRING,  -- mens_vs_control | womens_vs_control
  uplift_score     FLOAT64, -- predicted incremental conversion probability
  propensity_score FLOAT64, -- predicted P(in the treatment arm | features)
  decile           INT64,   -- 1-10 within comparison+split, 1 = strongest uplift
  split            STRING   -- train | validate | test; dashboards filter on it
)
CLUSTER BY comparison
OPTIONS(
  description = 'Per-customer uplift scores for the Hillstrom arms. Row-level scores stay in this table: dashboards read only the aggregate decile rollups, so no per-customer PII or score is shipped to Tableau.'
);
