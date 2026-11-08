-- Purpose: typed contract for the selected Hillstrom holdout learner.
-- Inputs: prepare_bigquery.py exports; exactly one learner per comparison/split.
-- Outputs: uplift_customer_scores used by aggregate dashboard views.
-- Assumptions: record_id is a file row key, not a customer key. PII is not present.
-- response_score is purchase response probability, not treatment assignment.
-- Deciles and rank_position are computed in Python with the evaluation seed.
-- NOT EXECUTED against live BigQuery: use deploy_bigquery.py and retain its receipt.
CREATE TABLE IF NOT EXISTS `campaign_incrementality.uplift_customer_scores` (
  record_id        INT64,   -- join key to the curated table; NOT a customer key
  comparison       STRING,  -- mens_vs_control | womens_vs_control
  uplift_score     FLOAT64, -- predicted incremental conversion probability
  response_score FLOAT64, -- predicted P(conversion | email, features); NOT assignment propensity
  decile           INT64,   -- 1-10 within comparison+split, 1 = strongest uplift
  rank_position    INT64,   -- seeded Python tie order; never rerank in SQL
  learner          STRING,  -- selected uplift learner (default t_learner)
  split            STRING   -- train | validate | test; dashboards filter on it
)
CLUSTER BY comparison
OPTIONS(
  description = 'Per-customer uplift scores for the Hillstrom arms. Row-level scores stay in this table: dashboards read only the aggregate decile rollups, so no per-customer PII or score is shipped to Streamlit.'
);
