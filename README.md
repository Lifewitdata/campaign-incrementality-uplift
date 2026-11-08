# Campaign Incrementality & Customer Uplift Modeling

**A real-data experiment analysis with an interactive Streamlit decision dashboard, Python machine learning and a reconciled BigQuery batch workflow.**

Should a retailer send a campaign to everyone, target customers likely to purchase, or target customers whose purchase probability changes because of the campaign? This project separates those questions using Kevin Hillstrom's randomized email experiment: **64,000 customers**, three campaign arms and two weeks of observed visits, purchases and spending.

## Open the dashboard locally

The app runs immediately from the aggregate results included in this repository. No raw-data download, model training, Google Cloud login or API key is required to view it.

From the project folder in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Using the virtual environment's Python directly avoids PowerShell activation-policy issues.

On macOS/Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

## What the dashboard answers

| Page | Decision supported | Interactive evidence |
|---|---|---|
| Campaign impact | Did either email change behaviour? | Conversion, visit and spend selectors; campaign comparisons; adjusted effect intervals; exploratory customer segments |
| Uplift targeting | Does causal targeting improve on purchase-response targeting? | S-learner, T-learner and response rankings; decile effects; Qini gain above random; paired policy comparison by budget |
| Campaign economics | Under which cost and margin assumptions could a policy be worthwhile? | Policy, targeting share, cost and margin controls; sensitivity heatmap; net contribution and conditional intervals |
| Evidence & planning | Can the results be checked, and how should a new experiment be sized? | Quality checks, randomization diagnostics, baseline balance, power planning and downloadable aggregate tables |

Controls are scoped to their sections. For example, the policy head-to-head comparison is explicitly T-learner versus purchase-response, even when another decile model is selected. The economics heatmap shows the full assumption grid; KPI cards reflect the chosen cost and margin.

## Findings from this experiment

| Question | Measured result |
|---|---|
| Conversion lift versus no email | Mens email **+0.68 percentage points**; Womens email **+0.31 percentage points** |
| Incremental spend per customer | Approximately **+$0.77** and **+$0.42**, respectively |
| Does uplift ranking outperform the baseline? | The holdout does **not establish superiority**; conversion Qini coefficients are negative and paired policy intervals include zero at the reported small budgets |
| Are there segment differences? | One of twelve Holm-adjusted interaction tests survives: Womens email by new-customer status. Treat this as a hypothesis for a new test |
| Is the scenario value realised profit? | No. Email costs and margins are assumed, not observed |

The larger Mens point estimate is not itself a direct statistical test against Womens. Campaign names and `mens`/`womens` purchase flags refer to merchandise lines, not customer gender.

Read the [experiment brief](reports/experiment_brief.md), [model card](reports/model_card.md), [power analysis](reports/power_analysis.md) and [business recommendations](reports/business_recommendations.md) for the detailed evidence.

## Statistical and ML approach

- **Data validation:** source SHA-256 pin, row counts, treatment labels, binary outcome domains and cross-field consistency checks.
- **Experimentation:** randomization diagnostics, treatment-control contrasts, confidence intervals, Holm multiplicity correction, bootstrap analysis for skewed spend and subgroup interaction tests.
- **Planning:** conversion/spend sample-size calculations, minimum detectable effects and simulated power checks.
- **Machine learning:** LightGBM S-learner, T-learner and a purchase-response baseline. Preprocessing is fitted on training data; post-campaign outcomes are excluded from features.
- **Evaluation:** held-out deciles, cumulative gain/Qini, response diagnostics, fixed-budget targeting comparisons and paired-bootstrap uncertainty.
- **Economics:** conditional scenario contribution from incremental spending, assumed margin and assumed send costs. Costs and revenue use the same historical treated subset.

The train/validation/test split spans all 64,000 records; **8,523 test records** belong to the Mens-versus-control comparison. Individual causal effects are not observed. Re-running this historical holdout is reproducibility, not a new independent validation.

## Reproduce the full analysis

For training and statistical analysis, install the project dependencies rather than only the lightweight app environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,tracking,notebook,dashboard,warehouse]"
.\.venv\Scripts\python.exe scripts/ingest_data.py
.\.venv\Scripts\python.exe scripts/run_experiment_analysis.py
.\.venv\Scripts\python.exe scripts/run_power_analysis.py
.\.venv\Scripts\python.exe scripts/train_uplift_models.py
.\.venv\Scripts\python.exe scripts/build_economics_scenarios.py
.\.venv\Scripts\python.exe -m pytest
```

The [executed analysis notebook](notebooks/01_results_walkthrough.ipynb) also walks through EDA, inference, modelling, evaluation and recommendations from raw data. Choose the same Python environment as its kernel. Model training can record runs in the local MLflow store; the dashboard does not require MLflow.

## BigQuery workflow

BigQuery is the warehouse and reconciliation layer. The Streamlit app deliberately reads the committed aggregate evidence so it remains portable and does not expose cloud credentials. It is **not a live BigQuery connection**.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_bigquery.py
.\.venv\Scripts\python.exe scripts/deploy_bigquery.py --project YOUR_PROJECT_ID
```

The second command previews offline. To execute after configuring application-default credentials:

```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
.\.venv\Scripts\python.exe scripts/deploy_bigquery.py --project YOUR_PROJECT_ID --execute
```

Use a dedicated dataset, default `campaign_incrementality`. Execution replaces project-owned tables there. It preserves original record IDs, validates scores and seeded ranks, runs SQL quality checks, compares every curated row with Python, and reconciles policy economics. Normal Google Cloud billing and permissions apply.

See the [BigQuery runbook](sql/bigquery/README.md). A live run writes `outputs/bigquery/deployment_receipt.json`; its job IDs and verification status are the evidence of that deployment. This repository does not include the user's local receipt and does not independently certify a cloud run. Never commit credentials.

## Host the app on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Create an app from repository `Lifewitdata/campaign-incrementality-uplift`.
3. Select branch `main` and entrypoint **`streamlit_app.py`**.
4. Use Python **3.12**, matching the app validation environment, and deploy.

The root `requirements.txt` installs the minimal app dependencies. No secrets are needed for the included aggregate-data app. A hosted URL should be added only after the actual deployment succeeds; none is claimed here. See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Project layout

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Four-page decision dashboard |
| `requirements.txt`, `.streamlit/config.toml` | App dependencies and theme |
| `src/campaign_ml/` | Statistical, ML, validation and warehouse logic |
| `scripts/` | Ingestion, analysis, training, economics and BigQuery entrypoints |
| `configs/experiment.yaml` | Experiment, split, feature and scenario definitions |
| `outputs/tables/` | Primary evidence read by the app |
| `outputs/dashboard/` | Derived reporting exports used by warehouse preparation |
| `sql/bigquery/` | Warehouse tables, quality checks and views |
| `notebooks/`, `reports/` | Analysis walkthrough, model card and decision documentation |
| `tests/` | Statistical, contract and Streamlit interaction tests |

## Validation and scope

Validation for this update: **271 non-slow tests passed**, including four Streamlit interaction tests; 20 slow tests were deselected. Warehouse preparation also reconciled the pinned dataset and holdout results after the reporting-folder rename.

Run the dashboard tests with `python -m pytest tests/test_streamlit_app.py`. They check page navigation, outcome/segment/model filters, the 64,000 and 8,523 record totals, the paired policy result and scenario arithmetic. Run `python -m pytest -m "not slow"` for the broader fast suite. Full real-data tests require the pinned raw file from ingestion.

This is a completed portfolio analysis and runnable dashboard with a BigQuery deployment path. It is not a scheduled production service. Public app hosting is a separate account-level deployment step. The experiment covers one retailer in 2008 and two-week outcomes; it does not support claims about current campaign profitability, long-term CLV or production targeting gains. No observational matching or difference-in-differences is needed for the randomized treatment comparison.

## Data attribution

Kevin Hillstrom (2008), **MineThatData E-Mail Analytics and Data Mining Challenge**. See [data provenance](data/README.md) for the original source, checksum and interpretation. Synthetic data appears only in tests that check whether the algorithms can recover known signals; the project findings use the real experiment.
