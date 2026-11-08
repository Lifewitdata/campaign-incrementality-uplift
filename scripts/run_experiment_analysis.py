"""Phase 2 entrypoint: allocation checks, balance, and treatment effects.

    python scripts/run_experiment_analysis.py

Writes the tables behind the Campaign Incrementality dashboard to
``outputs/tables/`` plus the Streamlit-ready extract in ``outputs/dashboard/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from campaign_ml.config import load_config, resolve
from campaign_ml.experimentation import (
    arm_summary,
    balance_table,
    distribution_report,
    heterogeneity_test,
    holm_adjust,
    regression_adjusted_effect,
    run_experiment_analysis,
    sample_ratio_test,
    spend_decomposition,
    subgroup_effects,
)
from campaign_ml.validation import FEATURE_COLUMNS, load_raw, prepare_clean_table

TABLEAU_DIR = "outputs/dashboard"

COVARIATES = [
    "recency", "history", "history_segment", "zip_code", "newbie", "channel", "mens", "womens",
]


def main() -> int:
    config = load_config()
    exp = config["experiment"]
    df = prepare_clean_table(load_raw(resolve(config["data"]["raw_path"])))
    tables = resolve("outputs/tables")
    tables.mkdir(parents=True, exist_ok=True)
    dashboard = resolve(TABLEAU_DIR)
    dashboard.mkdir(parents=True, exist_ok=True)

    srm = sample_ratio_test(df, expected_share=float(exp["srm"]["expected_shares"]))
    pd.DataFrame([srm]).to_csv(tables / "design_checks.csv", index=False)
    print(f"[design] allocation chi2={srm['chi2']:.3f} on {srm['degrees_of_freedom']:.0f} df, "
          f"p={srm['p_value']:.4f} -> "
          f"{'randomisation looks intact' if srm['p_value'] > 0.001 else 'SAMPLE RATIO MISMATCH'}")

    summary = arm_summary(df, confidence_level=float(exp["confidence_level"]))
    summary.to_csv(tables / "arm_summary.csv", index=False)
    print("[arms] " + " | ".join(
        f"{row['arm']}: conv {100 * row['conversion_rate']:.3f}%, "
        f"visit {100 * row['visit_rate']:.2f}%, spend ${row['spend_per_customer']:.3f}"
        for _, row in summary.iterrows()))

    balance = balance_table(
        df, features=FEATURE_COLUMNS,
        categorical=config["modelling"]["features"]["categorical"],
    )
    balance.to_csv(tables / "baseline_balance.csv", index=False)
    worst = balance["std_mean_difference"].abs().max()
    print(f"[balance] worst |standardised difference| = {worst:.4f} "
          f"({ 'pass' if worst < 0.1 else 'FAIL' } at the 0.1 rule of thumb)")

    distribution = pd.DataFrame([
        {"scope": arm if arm else "pooled",
         **distribution_report(df["spend"] if not arm else df.loc[df["segment"] == arm, "spend"])}
        for arm in [None, *df["segment"].unique()]
    ])
    distribution.to_csv(tables / "spend_distribution.csv", index=False)
    print(f"[spend] pooled skewness {distribution.loc[0, 'skewness']:.2f}, "
          f"{100 * distribution.loc[0, 'share_zero']:.1f}% of customers spent nothing")

    effects = run_experiment_analysis(df, config)
    effects.to_csv(tables / "experiment_effects.csv", index=False)

    decomposition = pd.DataFrame([
        {"comparison": c["name"], "treatment": c["treatment"],
         **spend_decomposition(df, c["treatment"], c["control"])}
        for c in exp["comparisons"]
    ])
    decomposition.to_csv(tables / "spend_decomposition.csv", index=False)
    for _, row in decomposition.iterrows():
        print(f"[spend split] {row['comparison']}: "
              f"{100 * row['conversion_driven_share']:.1f}% of the incremental spend is extra "
              f"buyers (basket difference p={row['basket_p_value']:.3f}, "
              f"identity residual {row['identity_residual']:.2e})")

    primary = exp["outcomes"]["primary"]
    specs = [(primary, "ols"), (primary, "logit"), ("spend", "ols")]
    adjusted = pd.DataFrame([
        {"comparison": c["name"],
         **regression_adjusted_effect(df, outcome=outcome, treatment_label=c["treatment"],
                                      covariates=COVARIATES, control_label=c["control"],
                                      model=model)}
        for c in exp["comparisons"] for outcome, model in specs
    ])
    adjusted.to_csv(tables / "covariate_adjusted_effects.csv", index=False)
    print("[adjusted] unadjusted vs covariate-adjusted on the same scale:")
    for _, row in adjusted[adjusted["model"] != "logit"].iterrows():
        print(f"  {row['comparison']:18s} {row['outcome']:10s} "
              f"unadjusted {row['unadjusted_effect_same_scale']:+.5f} -> "
              f"adjusted {row['adjusted_effect']:+.5f} "
              f"({100 * row['relative_change_vs_unadjusted']:+.1f}% on {row['effect_scale']})")

    alpha = float(exp["alpha"])
    sub_columns = list(exp["subgroups"]["variables"])
    sub_outcome = str(exp["subgroups"]["outcome"])
    subgroup_frames, heterogeneity_rows = [], []
    for comparison in exp["comparisons"]:
        table = subgroup_effects(df, sub_columns, comparison["treatment"],
                                 comparison["control"], sub_outcome, alpha=alpha)
        table.insert(0, "comparison", comparison["name"])
        subgroup_frames.append(table)
        for column in sub_columns:
            heterogeneity_rows.append(
                {"comparison": comparison["name"],
                 **heterogeneity_test(df, column, comparison["treatment"],
                                      comparison["control"], sub_outcome)}
            )
    subgroups = pd.concat(subgroup_frames, ignore_index=True)
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    adjusted_family = holm_adjust(heterogeneity["p_value"], alpha)
    heterogeneity["p_holm"] = np.minimum(adjusted_family.p_adjusted.to_numpy(), 1.0)
    heterogeneity["reject_holm"] = (adjusted_family.reject_adjusted.to_numpy()
                                    .astype(bool))
    heterogeneity["n_tests_in_family"] = len(heterogeneity)
    heterogeneity["interpretation"] = np.where(
        heterogeneity["reject_holm"],
        "interaction survives multiplicity control",
        "no evidence that the effect differs between these levels",
    )
    subgroups.to_csv(tables / "subgroup_effects.csv", index=False)
    heterogeneity.to_csv(tables / "heterogeneity_tests.csv", index=False)
    print(f"[subgroups] {len(subgroups)} exploratory group effects on {sub_outcome}; "
          f"the widest and narrowest differ by "
          f"{subgroups['effect'].max() - subgroups['effect'].min():+.5f}")
    for _, row in heterogeneity.sort_values("p_value").iterrows():
        print(f"  {row['comparison']:18s} by {row['subgroup_variable']:16s} "
              f"LR chi2={row['chi2']:.2f} on {row['degrees_of_freedom']:.0f} df "
              f"p={row['p_value']:.4g} p_holm={row['p_holm']:.4g} -> {row['interpretation']}")

    key = effects[(effects["measure"] != "relative_lift")
                  & (effects["method"].str.startswith(("Wald", "percentile bootstrap")))]
    key.to_csv(dashboard / "campaign_incrementality_effects.csv", index=False)
    summary.to_csv(dashboard / "campaign_incrementality_arms.csv", index=False)
    subgroups.to_csv(dashboard / "campaign_incrementality_subgroups.csv", index=False)
    heterogeneity.to_csv(dashboard / "campaign_incrementality_heterogeneity.csv", index=False)
    print(f"[exports] {len(effects)} effect rows, {len(key)} headline rows -> {dashboard}")

    for comparison in exp["comparisons"]:
        for outcome in [exp["outcomes"]["primary"], *exp["outcomes"]["secondary"]]:
            pick = key[(key["comparison"] == comparison["name"]) & (key["outcome"] == outcome)]
            if pick.empty:
                continue
            row = pick.iloc[0]
            print(f"  {comparison['name']:18s} {outcome:10s} "
                  f"effect {row['effect']:+.5f} [{row['ci_low']:.5f}, {row['ci_high']:.5f}] "
                  f"p_holm={row['p_holm']:.4g} sig={bool(row['reject_holm'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
