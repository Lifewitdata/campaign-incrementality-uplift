-- =============================================================================
-- Stage 00 - Dataset and raw landing table DDL
-- Purpose: create the warehouse dataset plus a landing table that mirrors the
--          published Hillstrom MineThatData e-mail experiment CSV column for
--          column, so loading stays a dumb file-to-table copy with no logic.
-- Inputs:  data/raw/hillstrom.csv - 64,000 rows, 12 columns, 1 header row.
-- Outputs: campaign_incrementality.landing_hillstrom_raw (12 source columns,
--          no derived columns, no constraints).
-- Assumptions: the CSV itself is loaded with `bq load` (see README), never by
--          this DDL. The published file ships NO customer identifier, so there
--          is no primary key available at this layer. Landing keeps the source
--          spelling verbatim, including the `Surburban` typo, because label
--          fixes belong in the curated layer where they can be audited.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
--          Two statements - run with `bq query --batch` or one at a time.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS `campaign_incrementality`
OPTIONS(
  description = 'Campaign incrementality and uplift analysis of the Hillstrom (2008) randomised e-mail experiment.'
);

-- Every column is nullable on purpose: stage 04 is the gate that proves
-- completeness, so a truncated or mis-typed load fails a check rather than
-- silently rejecting rows here.
CREATE TABLE IF NOT EXISTS `campaign_incrementality.landing_hillstrom_raw` (
  recency         INT64,   -- months since last purchase, 1-12
  history_segment STRING,  -- published spend band, e.g. '2) $100 - $200'
  history         FLOAT64, -- dollars spent in the 12 months before the campaign
  mens            INT64,   -- 1/0 bought mens merchandise pre-campaign (NOT a gender label)
  womens          INT64,   -- 1/0 bought womens merchandise pre-campaign (NOT a gender label)
  zip_code        STRING,  -- 'Surburban' (source typo), 'Rural', 'Urban'
  newbie          INT64,   -- 1/0 first purchase within 12 months before sampling
  channel         STRING,  -- past purchase channel: Web, Phone, Multichannel
  segment         STRING,  -- randomised arm: Womens E-Mail / Mens E-Mail / No E-Mail
  visit           INT64,   -- OUTCOME: visited in the two weeks after the send, 1/0
  conversion      INT64,   -- OUTCOME: purchased in the two weeks after the send, 1/0
  spend           FLOAT64  -- OUTCOME: dollars spent in the two weeks after the send
)
CLUSTER BY segment
OPTIONS(
  description = 'Raw Hillstrom e-mail experiment file, loaded verbatim. visit/conversion/spend are post-campaign outcomes and must never be used as model features.',
  labels = [('layer', 'landing'), ('source', 'hillstrom_2008')]
);
