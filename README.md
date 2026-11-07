# Campaign Incrementality & Customer Uplift Modeling

This project measures whether two e-mail campaigns changed customer behaviour, and tests
whether uplift modelling picks better targets than a purchase-propensity list. It uses the
public **Hillstrom MineThatData** experiment: 64,000 customers randomised to Mens E-Mail,
Womens E-Mail or No E-Mail, with visits, conversions and spend tracked for two weeks.

## Answers

| Question | Answer |
|---|---|
| Did either e-mail raise conversion? | Yes. Mens +0.68 pp (97.5% FWER CI +0.47 to +0.89); Womens +0.31 pp (+0.13 to +0.50). Holm-adjusted for the shared control. |
| How much incremental spend? | Mens +$0.77 per customer, Womens +$0.42 (bootstrap CIs, because spend is ~99% zeros). Nearly all of it comes from additional buyers. |
| Which groups respond more? | Only one of 12 Holm-adjusted interactions survives: the Womens e-mail works better on new customers. Everything else is noise. |
| Does uplift targeting beat propensity? | **No, not on this data.** Holdout Qini coefficients are all negative, and the paired-bootstrap difference vs propensity and vs random straddles zero. |

Details: [experiment brief](reports/experiment_brief.md), [model card](reports/model_card.md),
[power analysis](reports/power_analysis.md), [recommendations](reports/business_recommendations.md).

## What this does *not* claim

- Randomisation identifies group-average effects. No customer's individual treatment effect
  is observed, so uplift scores are estimates, judged only on the holdout.
- Profitability numbers use assumed margins and send costs. They are **scenarios, not realised
  ROI**.
- `mens` / `womens` are pre-campaign purchase flags, not gender.
- **BigQuery SQL (`sql/bigquery/`) is parse-verified with sqlglot but has never been run against
  a live project.** `scripts/load_bigquery.py` defaults to `--dry-run`.

## Run it

```bash
python -m venv .venv
# Windows Git Bash (Linux/macOS: source .venv/bin/activate)
source .venv/Scripts/activate
python -m pip install -e ".[dev,tracking]"
python scripts/ingest_data.py              # download, SHA-256 pin, validate, clean
python scripts/run_experiment_analysis.py  # effects, multiplicity, spend skew, subgroups
python scripts/run_power_analysis.py       # sample size / MDE for the next test
python scripts/train_uplift_models.py      # propensity, S-, T-learner + holdout eval + MLflow
python scripts/build_economics_scenarios.py
pytest                                     # add -m "not slow" to skip real-data checks
```

Outputs: `outputs/tables/` (evidence CSV/JSON) and `outputs/tableau/` (dashboard extracts, see
[dashboards/README.md](dashboards/README.md)).

## Complete Python analysis notebook

[Open the end-to-end analysis](notebooks/01_results_walkthrough.ipynb). It works from the
raw dataset rather than loading published scores: customer EDA, randomization diagnostics,
bootstrap effects and multiplicity, exploratory interactions, power planning, training-only
preprocessing, explicit LightGBM propensity/S-/T-learner fits, response diagnostics, holdout
deciles and Qini curves, paired targeting comparisons, feature importance and scenario economics.

```bash
python -m pip install -e ".[notebook,tracking]"
python scripts/ingest_data.py
```

Select this environment's Python kernel in your notebook editor and run all cells in order.
The notebook does not overwrite pipeline evidence CSVs. Reusing the historical holdout is a
reproducibility check, not fresh independent validation.

## Layout

```
configs/experiment.yaml   arms, comparisons, outcomes, split, features, scenarios
src/campaign_ml/          ingestion, validation, experimentation, power, features, uplift, evaluation
scripts/                  pipeline entry points
sql/bigquery/             landing, curated table, DQ checks, dashboard views (not executed)
tests/                    unit tests + slow integration tests that recompute published tables
reports/ dashboards/ notebooks/ data/README.md
```

Data: Kevin Hillstrom (2008), MineThatData E-Mail Analytics and Data Mining Challenge. See
[data/README.md](data/README.md).
