# Model card - uplift models, Mens E-Mail vs No E-Mail

**Purpose.** Rank customers by *estimated* incremental conversion probability from the Mens
e-mail, and test whether that ranking beats a purchase-propensity list.

**Data.** One predefined comparison (Mens vs control). Stratified split by arm x conversion:
train 38,399 / validate 12,801 / test 12,800 rows, with arm shares preserved in every split.
Preprocessing (imputation, scaling, one-hot) is fitted on training rows only. Features are
pre-campaign only (recency, history, purchase flags, zip type, newbie, channel); `visit`,
`conversion`, `spend` and the arm label are blocked by config and by a runtime check.

**Models.** LightGBM with validation early stopping.
- *Propensity baseline*: P(convert | emailed), trained on the emailed arm.
- *S-learner*: one model with treatment as a feature.
- *T-learner*: separate treated / control models; uplift = difference.

Runs are tracked in MLflow (SQLite backend).

**Holdout results (12,800 test rows; 8,523 in the two compared arms).**

| Learner | Validation AUC | Qini coefficient | Top-decile uplift | Bottom-decile uplift |
|---|---|---|---|---|
| Propensity | 0.568 | -0.64 | +1.53 pp | +1.92 pp |
| S-learner | 0.617 | -0.06 | +0.55 pp | +1.43 pp |
| T-learner | 0.568 | -0.75 | +1.54 pp | +2.17 pp |

Early stopping halts after 1-6 trees. **No evidence of better targeting on this holdout**:
every conversion Qini coefficient is negative and no decile ordering is monotonic.

T-learner vs propensity list, paired bootstrap of the difference in incremental conversions:
-0.73 [-3.66, +0.73] at 5% targeted, +0.01 [-0.16, +0.21] at 10%. T-learner vs random at 10%:
+0.30 [-8.75, +9.13].

**Can the code find heterogeneity when it exists?** Yes. `tests/test_uplift.py` plants
"sure things" and "persuadables" in synthetic data, and the T-learner recovers them where the
propensity model does not. The null result here reflects the data (rare outcome, about 0.7 pp
average effect, few pre-campaign features), not a pipeline that cannot detect a signal.

**Do not use** these scores to claim individual treatment effects, or to target before a
model shows positive holdout Qini on new data.
