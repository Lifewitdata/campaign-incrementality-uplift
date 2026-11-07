# BigQuery SQL layer — campaign incrementality and uplift

Layered BigQuery standard SQL for the Hillstrom MineThatData e-mail experiment:
64,000 customers randomised into Womens E-Mail / Mens E-Mail / No E-Mail, with
`visit`, `conversion` and `spend` observed for two weeks after the send.

> ## ⚠ NOT EXECUTED AGAINST A LIVE BIGQUERY PROJECT
>
> **These scripts have never been run on BigQuery.** The author has no cloud
> credentials, so no dataset, table, view or load in this folder has ever been
> created on a real project.
>
> What *is* verified: every file parses under `sqlglot` with
> `dialect="bigquery"` and is checked for structure and for the contracts below
> by `tests/test_sql_bigquery.py`. That is a parser and a lint, **not** an
> execution. Nothing here has returned a row, been cost-estimated, or been
> compared to a `bq` schema.
>
> **Expect to fix dialect details on your project.** Most likely candidates:
> multi-statement files (split on `;` if your `bq`/SDK version rejects them),
> `UNPIVOT` availability, `STRING_AGG(... ORDER BY ...)`, `CLUSTER BY` on tables
> this small, two-part versus three-part (project-qualified) names, `OPTIONS()`
> keys, and whether your region requires `--location`. Treat every object
> definition as a reviewed draft, run them one at a time, and read the errors.

## Load order

Files are prefixed so the dependency order is the filename order. Run top to
bottom; each stage only reads objects from stages above it.

| # | File | Creates / writes | Reads |
|---|------|------------------|-------|
| 00 | `00_dataset_and_landing_ddl.sql` | dataset + `landing_hillstrom_raw` (12 source columns, verbatim) | — |
| 01 | `01_reference_arms_ddl.sql` | `reference_arms` (3 rows) | — |
| 02 | `02_curated_campaign_customers_ddl.sql` | `curated_campaign_customers` (18 columns) | — |
| 03 | `03_curated_campaign_customers_insert.sql` | `curated_campaign_customers` (INSERT OVERWRITE), `campaign_customers_csv` view | landing, curated DDL |
| 04 | `04_data_quality_checks.sql` | result set → `data_quality_results` | curated |
| 05 | `05_reference_scenarios_ddl.sql` | `reference_scenarios` (9 cost × margin rows) | — |
| 06 | `06_uplift_customer_scores_ddl.sql` | `uplift_customer_scores` (loaded, not generated) | — |
| 07 | `07_views_dashboard_campaign_incrementality.sql` | `dashboard_campaign_incrementality`, `dashboard_campaign_incrementality_effects` | curated, reference_arms |
| 08 | `08_views_dashboard_uplift.sql` | `dashboard_uplift_deciles`, `dashboard_uplift_summary` | scores, curated |
| 09 | `09_views_dashboard_economics.sql` | `dashboard_targeting_scenario_economics` | scores, curated, reference_scenarios |

## Prerequisites

```bash
# One-time: authenticate the CLI against your own project.
gcloud auth application-default login
export PROJECT_ID="your-gcp-project"      # <-- the only value you must change
export REGION="US"                        # match your dataset location
export DATASET="campaign_incrementality"  # keep, or see "Renaming" below
```

## 1. Create the dataset, then the landing table

`00_dataset_and_landing_ddl.sql` contains both `CREATE SCHEMA` and
`CREATE TABLE`; run it once. `bq mk` and the DDL do the same job — pick one, and
if you use `bq mk` the `CREATE SCHEMA IF NOT EXISTS` in stage 00 is a no-op.

```bash
bq --project_id="$PROJECT_ID" --location="$REGION" mk --dataset \
  --description "Hillstrom campaign incrementality and uplift warehouse" \
  "${PROJECT_ID}:${DATASET}"

bq --project_id="$PROJECT_ID" --location="$REGION" query \
  --use_legacy_sql=false --batch \
  "$(cat sql/bigquery/00_dataset_and_landing_ddl.sql)"
```

## 2. Load the raw source CSV into `landing_hillstrom_raw`

The landing table already exists with the right schema, so load without an
inline schema (the destination table's schema is used). The published file has
no customer identifier and no `record_id` column — 12 columns, in this order.

```bash
bq --project_id="$PROJECT_ID" load --source_format=CSV --skip_leading_rows=1 \
  "${PROJECT_ID}:${DATASET}.landing_hillstrom_raw" \
  data/raw/hillstrom.csv \
  'recency:INTEGER,history_segment:STRING,history:FLOAT,mens:INTEGER,womens:INTEGER,zip_code:STRING,newbie:INTEGER,channel:STRING,segment:STRING,visit:INTEGER,conversion:INTEGER,spend:FLOAT'

bq --project_id="$PROJECT_ID" query --use_legacy_sql=false \
  "SELECT COUNT(*) AS landing_rows FROM \`${DATASET}.landing_hillstrom_raw\`"
# expect 64000 (and sha256 of the local file: see data/raw/hillstrom.csv.sha256)
```

## 3. Run the transforms

Stages 01, 02, 05 and 06 are DDL (01 and 05 also seed themselves). Stages 03 and
04 are the load and the gate. Run in order:

```bash
for stage in 01_reference_arms_ddl 02_curated_campaign_customers_ddl \
             03_curated_campaign_customers_insert 05_reference_scenarios_ddl \
             06_uplift_customer_scores_ddl; do
  bq --project_id="$PROJECT_ID" query --use_legacy_sql=false --batch \
    "$(cat "sql/bigquery/${stage}.sql")"
done
```

`--batch` (multiple statements per job) is available in current `bq` versions.
If yours rejects it, feed one statement at a time — each file is split at `;`
and the statement boundaries are documented in its header.

### Option B: identical `record_id` values

`record_id` in stage 03 is a deterministic surrogate for the source file's row
position, because a BigQuery load does not preserve CSV row order. If you need
`record_id` to match `data/processed/campaign_customers.csv` byte for byte (for
joining warehouse output back to the Python artefacts), skip the landing→curated
transform and load the curated CSV instead:

```bash
bq --project_id="$PROJECT_ID" load --source_format=CSV --skip_leading_rows=1 \
  --replace \
  "${PROJECT_ID}:${DATASET}.curated_campaign_customers" \
  data/processed/campaign_customers.csv \
  'record_id:INTEGER,recency:INTEGER,history_segment:STRING,history:FLOAT,mens:INTEGER,womens:INTEGER,zip_code:STRING,newbie:INTEGER,channel:STRING,segment:STRING,visit:INTEGER,conversion:INTEGER,spend:FLOAT,arm_code:STRING,history_band:STRING,campaign_weeks:INTEGER'
```

That path gives you the 16 published columns and leaves `zip_code_raw` and
`data_loaded_utc` NULL — stage 04 flags those two as failing checks, which is
the honest signal that the warehouse row is not the transform row.

## 4. Data-quality gate

Single UNION ALL query returning `check_name, passed, detail, metric_value`.
Nothing else in this layer should be trusted until it returns zero failing rows.

```bash
bq --project_id="$PROJECT_ID" query --use_legacy_sql=false --replace \
  --destination_table="${PROJECT_ID}:${DATASET}.data_quality_results" \
  "$(cat sql/bigquery/04_data_quality_checks.sql)"

bq --project_id="$PROJECT_ID" query --use_legacy_sql=false \
  "SELECT check_name, passed, detail, metric_value
   FROM \`${DATASET}.data_quality_results\` WHERE NOT passed
   ORDER BY check_name"
```

It mirrors the Python gates: 64,000 rows, zero nulls in every column, no blank
labels, unique `record_id`, the three published arm labels with their exact
counts (control 21,306 / mens 21,307 / womens 21,387), `visit` and `conversion`
in {0,1}, `spend >= 0`, no `spend > 0` without `conversion = 1`, no
`conversion = 1` without `visit = 1`, `history_band` agreeing with
`history_segment`, and an arm-share-vs-1/3 sample-ratio-mismatch check.

## 5. Load the scored table

The uplift and economics dashboards read `uplift_customer_scores`, written by
the modelling job (not by this repo's SQL, and not yet published as a file in
this repo — export it from your scoring run with exactly these six columns).
`decile` is 1–10 within `(comparison, split)`, decile 1 = strongest predicted
uplift; `comparison` values must be `mens_vs_control` / `womens_vs_control`.
The mapping from what `campaign_ml.uplift.score_frame` emits today
(`record_id, comparison, learner, uplift_score, response_score`) to this
contract is documented in the header of
`06_uplift_customer_scores_ddl.sql` — it is a load-time reshape, and the SQL
layer never invents scores: with this table empty, stages 08 and 09 return zero
rows.

```bash
bq --project_id="$PROJECT_ID" load --source_format=CSV --skip_leading_rows=1 \
  --replace \
  "${PROJECT_ID}:${DATASET}.uplift_customer_scores" \
  outputs/tables/uplift_customer_scores.csv \
  'record_id:INTEGER,comparison:STRING,uplift_score:FLOAT,propensity_score:FLOAT,decile:INTEGER,split:STRING'
```

## 6. Build the dashboard views

Order matters: 07 reads curated only, 08 reads scores + curated, 09 reads both
plus scenarios.

```bash
for stage in 07_views_dashboard_campaign_incrementality 08_views_dashboard_uplift \
             09_views_dashboard_economics; do
  bq --project_id="$PROJECT_ID" query --use_legacy_sql=false --batch \
    "$(cat "sql/bigquery/${stage}.sql")"
done

bq --project_id="$PROJECT_ID" ls --views --dataset="${PROJECT_ID}:${DATASET}"
```

## Renaming the project or dataset

Every object is written as a two-part backtick-quoted name
(\`campaign_incrementality.curated_campaign_customers\`), which resolves against
the project your CLI or SDK defaults to. If your dataset has a different name,
rewrite the files (single quotes matter here — a backtick inside double quotes
runs a shell command):

```bash
cp -r sql/bigquery sql/bigquery.bak
sed -i 's/`campaign_incrementality\./`my-project.my_dataset./g' sql/bigquery/*.sql
sed -i 's/CREATE SCHEMA IF NOT EXISTS `campaign_incrementality`/CREATE SCHEMA IF NOT EXISTS `my-project.my_dataset`/' sql/bigquery/00_dataset_and_landing_ddl.sql
grep -c 'my-project.my_dataset' sql/bigquery/*.sql   # sanity check, then delete the .bak
```

If one project hosts several copies of this dataset, or you want fully
qualified names, substitute the three-part form \`your-project.campaign_incrementality.table\`
and keep the `CREATE SCHEMA` target as \`your-project.campaign_incrementality\`.

## What the dashboard should show

- `dashboard_campaign_incrementality` — one row per arm: counts, visit rate,
  conversion rate, mean spend per customer, mean spend per visitor.
- `dashboard_campaign_incrementality_effects` — each e-mail arm vs the control,
  with a normal-approximation 95% CI for the difference in proportions
  (`SQRT(p1(1-p1)/n1 + p2(1-p2)/n2)`) and a Welch (unpooled) CI for the
  difference in mean spend (`SQRT(var1/n1 + var2/n2)`). These CIs are
  closed-form approximations; the headline numbers in the write-up come from the
  Python layer's tests and bootstrap.
- `dashboard_uplift_deciles` / `dashboard_uplift_summary` — aggregate decile
  rollups only. Per-customer scores stay in `uplift_customer_scores`; ship no
  row-level table to Tableau.
- `dashboard_targeting_scenario_economics` — targeting level × cost/margin
  scenario. Every money column is prefixed `modelled_` because the source data
  carries **no** e-mail cost and **no** margin: this is scenario arithmetic, not
  realised ROI.

Arm labels are product lines in the creative, not customer genders — the dataset
has no gender field, and `mens` / `womens` on the fact table are pre-campaign
purchases. `record_id` is a load/join key, never a customer key.

## Verifying this folder locally

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_sql_bigquery.py -q
```

The test parses every file with sqlglot's BigQuery dialect and checks the object
contracts, header documentation, explicit column lists, and that this README
states the not-executed caveat. It cannot tell you whether a query returns the
rows you expect — only a real project can.
