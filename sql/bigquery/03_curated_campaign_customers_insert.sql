-- =============================================================================
-- Stage 03 - curated_campaign_customers load (landing -> curated)
-- Purpose: land the raw Hillstrom landing table into the curated table: trim
--          labels, fix the zip_code typo while keeping the raw value, derive
--          arm_code and history_band, and stamp the load time.
-- Inputs:  campaign_incrementality.landing_hillstrom_raw (stage 00).
-- Outputs: campaign_incrementality.curated_campaign_customers (stage 02 DDL)
--          campaign_incrementality.campaign_customers_csv - the 16-column
--          extract that matches data/processed/campaign_customers.csv exactly.
-- Assumptions: re-running this file replaces the table wholesale, so it is
--          idempotent. The CASE arms are the only place arm and band logic is
--          written; views must read the columns, never re-derive them.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
-- =============================================================================

INSERT OVERWRITE `campaign_incrementality.curated_campaign_customers` (
  record_id, recency, history_segment, history, mens, womens, zip_code, zip_code_raw,
  newbie, channel, segment, visit, conversion, spend, arm_code, history_band,
  campaign_weeks, data_loaded_utc
)
WITH staging AS (
  SELECT
    -- record_id is the row position in the published file. The dataset ships no
    -- customer identifier, so this is a join/load key, not a customer key.
    -- BigQuery does not preserve CSV row order across loads, so the ordering
    -- below is a deterministic surrogate for that position; README option B
    -- (load the pipeline-published curated CSV) is the only way to reproduce
    -- the exact record_id values the Python artefacts carry.
    ROW_NUMBER() OVER (
      ORDER BY recency, history, history_segment, mens, womens, zip_code, newbie,
               channel, segment, visit, conversion, spend
    ) AS record_id,
    recency,
    TRIM(history_segment) AS history_segment,
    history,
    mens,
    womens,
    TRIM(zip_code) AS zip_code_raw,
    -- Source typo 'Surburban' -> 'Suburban'. Unknown spellings fall through to
    -- NULL on purpose so stage 04 fails the null gate instead of guessing.
    CASE TRIM(zip_code)
      WHEN 'Surburban' THEN 'Suburban'
      WHEN 'Rural' THEN 'Rural'
      WHEN 'Urban' THEN 'Urban'
      ELSE CAST(NULL AS STRING)
    END AS zip_code,
    newbie,
    TRIM(channel) AS channel,
    TRIM(segment) AS segment,
    visit,
    conversion,
    spend,
    CASE TRIM(segment)
      WHEN 'No E-Mail' THEN 'control'
      WHEN 'Mens E-Mail' THEN 'mens'
      WHEN 'Womens E-Mail' THEN 'womens'
      ELSE CAST(NULL AS STRING)
    END AS arm_code,
    -- Band edges 0/100/200/350/500/750/1000 reproduce the published
    -- history_segment labels; agreement with it is checked in stage 04.
    CASE
      WHEN history < 100.0 THEN '1) $0 - $100'
      WHEN history < 200.0 THEN '2) $100 - $200'
      WHEN history < 350.0 THEN '3) $200 - $350'
      WHEN history < 500.0 THEN '4) $350 - $500'
      WHEN history < 750.0 THEN '5) $500 - $750'
      WHEN history < 1000.0 THEN '6) $750 - $1,000'
      ELSE '7) $1,000 +'
    END AS history_band,
    CAST(2 AS INT64) AS campaign_weeks
  FROM `campaign_incrementality.landing_hillstrom_raw`
)
SELECT
  staging.record_id, staging.recency, staging.history_segment, staging.history,
  staging.mens, staging.womens, staging.zip_code, staging.zip_code_raw,
  staging.newbie, staging.channel, staging.segment, staging.visit,
  staging.conversion, staging.spend, staging.arm_code, staging.history_band,
  staging.campaign_weeks, CURRENT_TIMESTAMP() AS data_loaded_utc
FROM staging;

-- Published contract: the same 16 columns, in the same order, as
-- data/processed/campaign_customers.csv. Unload this, not the base table, if a
-- notebook or test needs to diff the warehouse against the repo CSV.
CREATE OR REPLACE VIEW `campaign_incrementality.campaign_customers_csv` AS
SELECT
  record_id, recency, history_segment, history, mens, womens, zip_code, newbie,
  channel, segment, visit, conversion, spend, arm_code, history_band,
  campaign_weeks
FROM `campaign_incrementality.curated_campaign_customers`;
