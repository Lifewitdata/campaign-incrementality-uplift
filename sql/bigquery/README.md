# BigQuery deployment and reconciliation

**Status: not executed against a live BigQuery project in this development environment.**
Local tests and sqlglot parsing do not prove cloud execution. A successful live run
writes `outputs/bigquery/deployment_receipt.json` with job IDs and verification results.
Do not describe this as deployed until that receipt reports verification passed.

## Supported workflow (Windows PowerShell)

Run from the repository folder, not `C:\Windows\System32`. Use a dedicated dataset;
execution replaces this project's indexed landing, score, reference, curated and evidence
tables. Other datasets are not touched. Use your actual Google Cloud project ID.

```powershell
python -m pip install -e ".[dev,warehouse]"
python scripts/ingest_data.py
python scripts/prepare_bigquery.py
python scripts/deploy_bigquery.py --project YOUR_PROJECT_ID
```

The first command installs dependencies; ingestion verifies the pinned original CSV.
Preparation checks every score ID, restricts scores to the correct two experiment arms,
reproduces seeded tie ranks and reconciles deciles/policies with the saved evidence.
The deployment command above is an **offline preview**, not a BigQuery dry-run job:
it makes no cloud calls and needs no credentials.

If preparation reports missing or inconsistent analysis outputs, run the analysis
pipeline described in the root README. Do not skip checksum or reconciliation checks.
Default learner is `t_learner`; `--learner s_learner` is also supported by preparation.
This is a reporting selection, not a claim of a validated targeting winner.

When the preview identifies the intended project and dataset:

```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
python scripts/deploy_bigquery.py --project YOUR_PROJECT_ID --execute
```

BigQuery API and billing must be available. The account needs job creation in the
project, dataset creation if it does not exist, and write access in the target dataset.
No service-account key needs to be downloaded or committed. Location defaults to `US`;
set `--location` to the actual existing dataset location if different.
`--dataset campaign_incrementality` is the default. The dataset is fully qualified with
the supplied project ID. Query jobs have a default 1 GB per-job maximum bytes billed;
this is not a total cost cap. The dataset is small, but normal cloud billing applies.

## What deployment executes

1. Verify exported files against the manifest hashes.
2. Create the dedicated dataset and typed tables; seed arms and cost assumptions.
3. Load NDJSON with explicit schemas, avoiding CSV header/autodetection mistakes.
4. Build curated records using the **exported zero-based record ID**. Never infer
   original row order inside BigQuery. Identical source rows must not be deduplicated.
5. Fail if any SQL data-quality check fails; compare every curated row with Python.
6. Create dashboard views; reconcile SQL targeting economics with Python policy values.
7. Save job IDs, checks and success/failure in the local deployment receipt.

Individual replacements are idempotent, but the whole deployment is not an atomic
multi-table release. A failed run may leave some tables refreshed. Resolve the error
and rerun; do not refresh a dashboard until the receipt reports all checks passed.
The receipt and exports are ignored by Git because they include run-specific evidence
and row-level data. Keep a copy locally for your portfolio verification.

## Data contracts

| Object | Grain / purpose |
|---|---|
| `landing_hillstrom_indexed` | Original 12 fields plus zero-based `record_id`; 64,000 rows |
| `curated_campaign_customers` | One record per original row, corrected labels and audit fields |
| `uplift_customer_scores` | One chosen learner per comparison/test split; 8,523 rows for default Mens comparison |
| `evidence_*` | Exact existing Streamlit extracts, preserving all models and Python uncertainty estimates |
| `dashboard_campaign_incrementality` | Arm-level SQL descriptive metrics |
| `dashboard_campaign_incrementality_effects` | Python effect estimates, adjusted intervals and Holm p-values |
| `dashboard_uplift_deciles` / `dashboard_uplift_summary` | Descriptive rollups for the chosen learner |
| `dashboard_targeting_scenario_economics` | Chosen learner x budget x assumed cost/margin; historical treated subset |

`response_score` means **P(conversion given email and features)**, not probability of
assignment to treatment. Decile and rank are exported from the same seeded Python
ranking used in evaluation; SQL must not rerank ties. Costs and revenue both use the
selected **historical treated count**, not a mix of that count and the entire selected
pool. These are scenarios, not realised ROI. Decile gains alone do not establish
statistical superiority; use the paired-bootstrap evidence tables for that question.

`landing_hillstrom_raw` and `scripts/load_bigquery.py` remain a legacy raw-only loader
for compatibility. They do **not** feed the supported curated/scoring deployment.
Do not follow the old raw-only path to build dashboards.

## Streamlit reporting

Run `python -m streamlit run streamlit_app.py` from the repository root.
The app reads the canonical aggregate CSVs in `outputs/tables/`; it does not query
BigQuery and needs no cloud credentials. `outputs/dashboard/` holds warehouse
reporting exports. Keep the deployment receipt as separate cloud verification.
A successful warehouse run does not automatically refresh or host the app.
