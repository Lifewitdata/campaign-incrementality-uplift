# Tableau Public dashboards

Built by hand in Tableau Public from the CSV extracts in `outputs/tableau/` (regenerate them
with the pipeline). No workbook is committed until it is published.

| Dashboard | Upload these CSVs | Key views |
|---|---|---|
| 1. Campaign Incrementality | `campaign_incrementality_effects.csv`, `campaign_incrementality_arms.csv`, `campaign_incrementality_subgroups.csv`, `campaign_incrementality_heterogeneity.csv` | Effect + family-wise CI by arm and outcome; subgroup forest plot (labelled exploratory) |
| 2. Customer Uplift & Targeting | `uplift_deciles.csv`, `uplift_curves.csv`, `policy_head_to_head.csv`, `uplift_customer_scores.csv` | Observed uplift by decile; Qini curve vs random line; uplift vs propensity by budget |
| 3. Campaign Economics & Reliability | `campaign_economics_scenarios.csv`, `targeting_policy_values.csv` | Net-value scenario grid (margin x cost); CI bands. **Title must say "Scenario - not realised ROI".** |
