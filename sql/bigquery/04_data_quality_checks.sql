-- =============================================================================
-- Stage 04 - data_quality_checks (post-load gate on the curated table)
-- Purpose: prove in SQL what the Python pipeline asserts, so a warehouse load
--          that was never through the repo pipeline is still gated. One UNION
--          ALL query, one row per check: check_name, passed, detail,
--          metric_value.
-- Inputs:  campaign_incrementality.curated_campaign_customers (stage 03).
-- Outputs: a result set; materialise it with
--          `bq query --destination_table=campaign_incrementality.data_quality_results`
--          (README) and alert on any row where passed IS FALSE.
-- Assumptions: expected volumes and arm counts are the published Hillstrom
--          numbers (64,000 rows; control 21,306 / mens 21,307 / womens 21,387)
--          and the design is balanced 1/3 allocation. The chi-square p-value
--          the Python gate reports is approximated here by the arm-share
--          deviation, which is the part SQL should own.
-- NOT EXECUTED: parsed by sqlglot only; never run against a live project.
-- =============================================================================

WITH agg AS (
  SELECT
    COUNT(*) AS n_rows, COUNT(DISTINCT record_id) AS n_record_ids,
    COUNTIF(record_id IS NULL) AS record_id_nulls, COUNTIF(recency IS NULL) AS recency_nulls,
    COUNTIF(history_segment IS NULL) AS history_segment_nulls, COUNTIF(history IS NULL) AS history_nulls,
    COUNTIF(mens IS NULL) AS mens_nulls, COUNTIF(womens IS NULL) AS womens_nulls,
    COUNTIF(zip_code IS NULL) AS zip_code_nulls, COUNTIF(zip_code_raw IS NULL) AS zip_code_raw_nulls,
    COUNTIF(newbie IS NULL) AS newbie_nulls, COUNTIF(channel IS NULL) AS channel_nulls,
    COUNTIF(segment IS NULL) AS segment_nulls, COUNTIF(visit IS NULL) AS visit_nulls,
    COUNTIF(conversion IS NULL) AS conversion_nulls, COUNTIF(spend IS NULL) AS spend_nulls,
    COUNTIF(arm_code IS NULL) AS arm_code_nulls, COUNTIF(history_band IS NULL) AS history_band_nulls,
    COUNTIF(campaign_weeks IS NULL) AS campaign_weeks_nulls, COUNTIF(data_loaded_utc IS NULL) AS data_loaded_utc_nulls,
    COUNTIF(TRIM(segment) = '' OR TRIM(zip_code) = '' OR TRIM(channel) = '' OR TRIM(history_segment) = '') AS blank_label_rows,
    COUNTIF(visit NOT IN (0, 1)) AS bad_visit_rows,
    COUNTIF(conversion NOT IN (0, 1)) AS bad_conversion_rows,
    COUNTIF(spend < 0) AS negative_spend_rows, MIN(spend) AS min_spend,
    COUNTIF(spend > 0 AND conversion = 0) AS spend_without_conversion_rows,
    COUNTIF(conversion = 1 AND visit = 0) AS conversion_without_visit_rows,
    COUNTIF(IFNULL(history_band, '~') <> IFNULL(history_segment, '~')) AS band_mismatch_rows,
    SUM(IF(segment = 'No E-Mail', 1, 0)) AS control_rows,
    SUM(IF(segment = 'Mens E-Mail', 1, 0)) AS mens_rows,
    SUM(IF(segment = 'Womens E-Mail', 1, 0)) AS womens_rows
  FROM `campaign_incrementality.curated_campaign_customers`
),
nulls_long AS (
  SELECT column_name, null_count
  FROM agg
  UNPIVOT (null_count FOR column_name IN (
    record_id_nulls, recency_nulls, history_segment_nulls, history_nulls, mens_nulls, womens_nulls,
    zip_code_nulls, zip_code_raw_nulls, newbie_nulls, channel_nulls, segment_nulls, visit_nulls,
    conversion_nulls, spend_nulls, arm_code_nulls, history_band_nulls, campaign_weeks_nulls,
    data_loaded_utc_nulls))
),
null_gate AS (
  SELECT CAST(COALESCE(SUM(null_count), 0) AS INT64) AS null_cells,
         STRING_AGG(CASE WHEN null_count > 0 THEN column_name END, ', ' ORDER BY column_name) AS offending_columns
  FROM nulls_long
),
arms AS (
  SELECT 'No E-Mail' AS arm_label, CAST(21306 AS INT64) AS expected_rows, MAX(control_rows) AS observed_rows FROM agg
  UNION ALL SELECT 'Mens E-Mail', 21307, MAX(mens_rows) FROM agg
  UNION ALL SELECT 'Womens E-Mail', 21387, MAX(womens_rows) FROM agg
),
arms_gate AS (
  SELECT COUNTIF(expected_rows <> observed_rows) AS mismatched_arms, COUNT(*) AS arms_checked,
         STRING_AGG(CONCAT(arm_label, ' observed ', CAST(observed_rows AS STRING), ' vs expected ', CAST(expected_rows AS STRING)), '; ' ORDER BY arm_label) AS arm_detail
  FROM arms
),
arm_shares AS (
  SELECT arm_code, COUNT(*) / SUM(COUNT(*)) OVER () AS share
  FROM `campaign_incrementality.curated_campaign_customers`
  GROUP BY arm_code
),
srm_gate AS (
  SELECT COALESCE(MAX(ABS(share - 1.0 / 3.0) / (1.0 / 3.0)), 1.0) AS worst_relative_deviation,
         COUNT(*) AS arms_in_share, STRING_AGG(CONCAT(arm_code, ' ', CAST(ROUND(share, 5) AS STRING)), ', ') AS share_detail
  FROM arm_shares
)
SELECT 'volume.row_count' AS check_name, n_rows = 64000 AS passed,
       CONCAT('rows: ', CAST(n_rows AS STRING), ' vs expected 64000') AS detail, CAST(n_rows AS FLOAT64) AS metric_value FROM agg
UNION ALL
SELECT 'completeness.no_nulls', null_cells = 0,
       CONCAT('null cells: ', CAST(null_cells AS STRING), '; columns: ', IFNULL(offending_columns, 'none')), CAST(null_cells AS FLOAT64) FROM null_gate
UNION ALL
SELECT 'completeness.no_blank_labels', blank_label_rows = 0,
       CONCAT('blank label rows: ', CAST(blank_label_rows AS STRING)), CAST(blank_label_rows AS FLOAT64) FROM agg
UNION ALL
SELECT 'uniqueness.record_id_no_duplicates', n_record_ids = n_rows,
       CONCAT('distinct record_id: ', CAST(n_record_ids AS STRING), ' of ', CAST(n_rows AS STRING), ' rows'), CAST(n_rows - n_record_ids AS FLOAT64) FROM agg
UNION ALL
SELECT 'treatment.arm_labels_present', arms_checked = 3,
       CONCAT('distinct published arm labels: ', CAST(arms_checked AS STRING)), CAST(arms_checked AS FLOAT64) FROM arms_gate
UNION ALL
SELECT 'treatment.arm_counts_match_published', mismatched_arms = 0,
       arm_detail, CAST(mismatched_arms AS FLOAT64) FROM arms_gate
UNION ALL
SELECT 'binary.visit_in_zero_one', bad_visit_rows = 0,
       CONCAT('visit rows outside {0,1}: ', CAST(bad_visit_rows AS STRING)), CAST(bad_visit_rows AS FLOAT64) FROM agg
UNION ALL
SELECT 'binary.conversion_in_zero_one', bad_conversion_rows = 0,
       CONCAT('conversion rows outside {0,1}: ', CAST(bad_conversion_rows AS STRING)), CAST(bad_conversion_rows AS FLOAT64) FROM agg
UNION ALL
SELECT 'consistency.spend_non_negative', negative_spend_rows = 0,
       CONCAT('negative spend rows: ', CAST(negative_spend_rows AS STRING), '; min spend ', CAST(IFNULL(min_spend, -1.0) AS STRING)), CAST(IFNULL(min_spend, -1.0) AS FLOAT64) FROM agg
UNION ALL
SELECT 'consistency.spend_implies_conversion', spend_without_conversion_rows = 0,
       CONCAT('rows with spend > 0 and conversion = 0: ', CAST(spend_without_conversion_rows AS STRING)), CAST(spend_without_conversion_rows AS FLOAT64) FROM agg
UNION ALL
SELECT 'consistency.conversion_implies_visit', conversion_without_visit_rows = 0,
       CONCAT('rows with conversion = 1 and visit = 0: ', CAST(conversion_without_visit_rows AS STRING)), CAST(conversion_without_visit_rows AS FLOAT64) FROM agg
UNION ALL
SELECT 'consistency.history_band_matches_segment', band_mismatch_rows = 0,
       CONCAT('rows where history_band disagrees with history_segment: ', CAST(band_mismatch_rows AS STRING)), CAST(band_mismatch_rows AS FLOAT64) FROM agg
UNION ALL
SELECT 'design.srm_arm_share_within_1pct', worst_relative_deviation <= 0.01,
       CONCAT('arm shares vs the 1/3 design: ', share_detail), worst_relative_deviation FROM srm_gate;
