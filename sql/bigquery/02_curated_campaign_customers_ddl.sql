-- =============================================================================
-- Stage 02 - curated_campaign_customers DDL
-- Purpose: define the one curated customer-level table the warehouse serves,
--          matching data/processed/campaign_customers.csv (16 published
--          columns, 64,000 rows, no nulls) plus two warehouse audit columns.
-- Inputs:  campaign_incrementality.landing_hillstrom_raw (stage 00).
-- Outputs: campaign_incrementality.curated_campaign_customers
-- Assumptions: the first 16 columns are the published contract and are loaded
--          in CSV order. zip_code_raw and data_loaded_utc are warehouse-only
--          audit columns, so the byte-comparable extract lives in the
--          campaign_customers_csv view created by stage 03. record_id is the
--          row position in the published source file: the Hillstrom dataset
--          ships NO customer identifier, so record_id is a stable join/load
--          key, NOT a customer key - it cannot follow a person across
--          campaigns or across years of sends.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
-- =============================================================================

CREATE TABLE IF NOT EXISTS `campaign_incrementality.curated_campaign_customers` (
  record_id       INT64,     -- row position in the published file; load key, NOT a customer key
  recency         INT64,     -- months since last purchase, 1-12 (pre-campaign)
  history_segment STRING,    -- published spend band, trimmed
  history         FLOAT64,   -- dollars spent in the 12 months pre-campaign (29.99-3345.93)
  mens            INT64,     -- 1/0 purchased mens merchandise pre-campaign; NOT gender
  womens          INT64,     -- 1/0 purchased womens merchandise pre-campaign; NOT gender
  zip_code        STRING,    -- cleaned geography: Suburban | Rural | Urban
  zip_code_raw    STRING,    -- source spelling kept so the Surburban->Suburban fix is auditable
  newbie          INT64,     -- 1/0 first-time customer in the 12 months before sampling
  channel         STRING,    -- past purchase channel: Web | Phone | Multichannel
  segment         STRING,    -- randomised arm label as published
  visit           INT64,     -- OUTCOME (post-campaign): visited within 2 weeks, 1/0
  conversion      INT64,     -- OUTCOME (post-campaign): purchased within 2 weeks, 1/0
  spend           FLOAT64,   -- OUTCOME (post-campaign): dollars spent within 2 weeks, 0-499.00
  arm_code        STRING,    -- control | mens | womens, derived from segment
  history_band    STRING,    -- band re-derived from history; must equal history_segment
  campaign_weeks  INT64,     -- observation window in weeks after the send; constant 2
  data_loaded_utc TIMESTAMP  -- warehouse-only: when this curated refresh ran
)
CLUSTER BY arm_code
OPTIONS(
  description = 'Curated Hillstrom experiment table: one row per sampled customer, pre-campaign features, randomised arm and two-week post-campaign outcomes. One row per record_id; no PII beyond these aggregate-safe attributes.',
  labels = [('layer', 'curated'), ('source', 'hillstrom_2008')]
);
