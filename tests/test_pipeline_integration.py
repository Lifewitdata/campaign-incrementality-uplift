"""End-to-end reconciliation of the published artefacts against the pinned dataset.

Marked ``slow`` because it reads the 64k-row file. It does not retrain anything:
it takes the tables the pipeline published and recomputes them from source, which
is the check that catches a stale ``outputs/`` directory - the most likely way this
repo could ship a number its own data contradicts.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from campaign_ml import evaluation as ev
from campaign_ml import experimentation as ex
from campaign_ml import features as ft
from campaign_ml.config import load_config, resolve
from campaign_ml.validation import prepare_clean_table

pytestmark = pytest.mark.slow

TABLES = resolve("outputs/tables")
TABLEAU = resolve("outputs/dashboard")


def table(name: str) -> pd.DataFrame:
    path = TABLES / f"{name}.csv"
    assert path.exists(), f"{path} missing - run the scripts in the documented order"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def raw_real():
    from campaign_ml.validation import load_raw

    path = resolve("data/raw/hillstrom.csv")
    if not path.exists():
        pytest.skip("data/raw/hillstrom.csv not present; run scripts/ingest_data.py")
    return load_raw(path)


@pytest.fixture(scope="module")
def real(raw_real, config):
    clean = prepare_clean_table(raw_real)
    clean.attrs["sha256"] = hashlib.sha256(
        resolve("data/raw/hillstrom.csv").read_bytes()).hexdigest()
    return clean


@pytest.fixture(scope="module")
def test_split(real, config):
    return ft.make_splits(real, config)["test"]


def test_every_published_quality_gate_passed():
    checks = table("data_quality_checks")
    assert checks["passed"].all(), checks.loc[~checks["passed"], ["name", "detail"]]
    assert {"schema.columns", "volume.row_count", "design.srm_balance",
            "leakage.no_post_campaign_features",
            "consistency.spend_implies_conversion"} <= set(checks["name"])


def test_the_lineage_record_matches_the_bytes_on_disk(real, raw_real, config):
    lineage = json.loads((TABLES / "data_lineage.json").read_text(encoding="utf-8"))
    assert lineage["sha256"] == real.attrs["sha256"]
    assert lineage["sha256"] == config["data"]["expected_sha256"]
    assert lineage["rows"] == len(raw_real) == config["data"]["expected_rows"]
    assert len(raw_real.columns) == config["data"]["expected_columns"]
    assert lineage["columns"] == list(real.columns)
    assert sum(lineage["arm_counts"].values()) == lineage["rows"]
    assert lineage["source_url"].endswith(".csv")


def test_arm_summary_reconciles_with_the_clean_table(real):
    published = table("arm_summary").set_index("arm")
    for arm, group in real.groupby("segment", observed=True):
        row = published.loc[arm]
        assert row["n_customers"] == len(group)
        assert row["conversions"] == int(group["conversion"].sum())
        assert row["conversion_rate"] == pytest.approx(float(group["conversion"].mean()))
        assert row["spend_total"] == pytest.approx(float(group["spend"].sum()))
        assert row["spend_per_customer"] == pytest.approx(float(group["spend"].mean()))
        assert row["spend_skewness"] > 10
        assert 0.98 < row["share_zero_spend"] < 1.0
    assert published["share_of_sample"].sum() == pytest.approx(1.0)


def test_headline_effects_are_reproducible_from_the_clean_table(real, config):
    published = table("experiment_effects")
    for _, row in published[published["confirmatory"]].iterrows():
        treated = real.loc[real["segment"] == row["treatment_label"], row["outcome"]]
        control = real.loc[real["segment"] == row["control_label"], row["outcome"]]
        assert row["n_treatment"] == len(treated)
        assert row["n_control"] == len(control)
        assert row["mean_treatment"] == pytest.approx(float(treated.mean()), abs=1e-12)
        recomputed = (
            ex.two_proportion_effect(treated.to_numpy(float), control.to_numpy(float), 0.05)
            if row["outcome"] in ex.PROPORTION_OUTCOMES else
            ex.bootstrap_mean_difference(treated.to_numpy(float), control.to_numpy(float),
                                         replicates=int(row["n_replicates"]),
                                         seed=int(config["experiment"]["bootstrap"]["seed"]))
        )
        assert row["effect"] == pytest.approx(recomputed["effect"], abs=1e-12)
        assert row["ci_low"] == pytest.approx(recomputed["ci_low"], abs=1e-12)
        assert row["ci_high"] == pytest.approx(recomputed["ci_high"], abs=1e-12)


def test_exactly_one_confirmatory_test_per_comparison_and_outcome(config):
    published = table("experiment_effects")
    assert len(published) == 12
    counts = published.groupby(["comparison", "outcome"])["confirmatory"].sum()
    assert (counts == 1).all(), counts
    assert set(published.loc[~published["confirmatory"], "measure"]) == {
        "relative_lift", "mean_difference"}
    assert published["n_comparisons_in_family"].unique().tolist() == [2]
    assert (published.loc[published["confirmatory"], "p_holm"]
            >= published.loc[published["confirmatory"], "p_value"]).all()
    assert published.loc[~published["confirmatory"], "p_holm"].isna().all()
    assert published.loc[~published["confirmatory"], "reject_holm"].isna().all()


def test_the_two_emails_are_measured_against_one_shared_control(real, config):
    published = table("experiment_effects")
    control_rows = published[published["control_label"] != published["treatment_label"]]
    assert set(control_rows["control_label"]) == {config["experiment"]["arms"]["control"]}
    shared = real[real["segment"] == config["experiment"]["arms"]["control"]]
    for _, row in published[published["confirmatory"]].iterrows():
        if row["control_label"] == shared["segment"].iloc[0]:
            assert row["n_control"] == len(shared)


def test_newcombe_crosscheck_lands_on_the_same_estimate():
    published = table("experiment_effects")
    rows = published[published["crosscheck_method"].notna()]
    assert len(rows) == 4
    for _, row in rows.iterrows():
        assert row["ci_low_crosscheck"] <= row["effect"] <= row["ci_high_crosscheck"]
        midpoint = (row["ci_low_crosscheck"] + row["ci_high_crosscheck"]) / 2
        assert abs(midpoint - row["effect"]) < 0.0005
        assert row["ci_low_fwer"] <= row["ci_low"] and row["ci_high_fwer"] >= row["ci_high"]


def test_family_wise_intervals_widen_by_the_declared_number_of_tests(config):
    published = table("experiment_effects")
    alpha = float(config["experiment"]["alpha"])
    for row in published[published["confirmatory"]].itertuples():
        assert row.ci_high - row.ci_low < row.ci_high_fwer - row.ci_low_fwer
        # Bonferroni-dual: the family-wise interval for m tests is the pointwise
        # interval run at alpha/m, so its level must be 1 - alpha/m, not 1 - alpha.
        assert row.ci_level_fwer == pytest.approx(1 - alpha / row.n_comparisons_in_family)
        assert (row.ci_low_fwer <= row.ci_low <= row.ci_high <= row.ci_high_fwer)


def test_spend_skew_and_zero_share_are_reported_where_the_bootstrap_is_justified():
    published = table("spend_distribution")
    assert (published["skewness"] > 10).all()
    assert (published["share_zero"] > 0.9).all()
    pooled = published[published["scope"] == "pooled"].iloc[0]
    assert pooled["max"] > 200
    # The 99th percentile is still a zero: a Wald/t interval on this column would
    # rest entirely on the handful of large orders, which is why spend is bootstrapped.
    assert pooled["p99"] == 0.0
    assert pooled["mean"] < 0.02 * pooled["positive_mean"]
    assert (published["skewness"] > published["positive_skewness"]).all()


def test_subgroup_tables_reconcile_and_the_heterogeneity_family_is_adjusted(real, config):
    effects = table("subgroup_effects")
    tests = table("heterogeneity_tests")
    variables = config["experiment"]["subgroups"]["variables"]
    comparison = config["experiment"]["comparisons"][0]
    recomputed = ex.subgroup_effects(real, variables, comparison["treatment"],
                                    comparison["control"], "conversion",
                                    alpha=float(config["experiment"]["alpha"]))
    published = effects[effects["comparison"] == comparison["name"]].drop(
        columns="comparison").reset_index(drop=True)
    pd.testing.assert_frame_equal(published, recomputed, check_dtype=False, atol=1e-12)

    assert len(tests) == 2 * len(variables)
    assert (tests["p_holm"] >= tests["p_value"]).all()
    assert (tests["degrees_of_freedom"] > 0).all()
    assert tests["n_tests_in_family"].unique().tolist() == [2 * len(variables)]
    assert tests["interpretation"].isin([
        "interaction survives multiplicity control",
        "no evidence that the effect differs between these levels"]).all()
    assert (tests["reject_holm"] == (tests["p_holm"] < 0.05)).all()


def test_the_newbie_interaction_is_the_one_surviving_claim_of_real_heterogeneity():
    """Pinned because it is the project's answer to business question 3."""
    tests = table("heterogeneity_tests")
    survivors = tests[tests["reject_holm"]]
    assert len(survivors) == 1
    row = survivors.iloc[0]
    assert row["comparison"] == "womens_vs_control"
    assert row["subgroup_variable"] == "newbie"
    assert row["p_holm"] < 0.05


def test_published_holdout_deciles_keep_both_arms_in_every_bin(test_split):
    deciles = table("uplift_deciles_holdout")
    scores = table("holdout_customer_scores")
    for learner, group in deciles.groupby("learner"):
        share = group["n_treated"] / group["n_customers"]
        assert share.between(0.35, 0.65).all(), (learner, share.round(3).to_list())
        assert group["n_customers"].sum() == pytest.approx(
            int((test_split["segment"] != "Womens E-Mail").sum()), abs=30)
        assert (group["conversions_treated"] + group["conversions_control"]
                <= group["n_customers"]).all()
        assert group["uplift_ci_low"].lt(group["uplift_ci_high"]).all()
        # The propensity baseline ranks on response, so its uplift column is
        # deliberately empty; each learner must still publish the score used.
        column = "response_score" if learner == "propensity" else "uplift_score"
        assert scores[scores["learner"] == learner][column].notna().all()


def test_the_decile_table_can_be_rebuilt_from_the_published_scores(test_split):
    scores = table("holdout_customer_scores")
    published = table("uplift_deciles_holdout")
    comparison = scores["comparison"].unique().item()
    holdout = ev.prepare_holdout(test_split, "Mens E-Mail", "No E-Mail", comparison)
    learner = "t_learner"
    series = pd.Series(
        scores.loc[scores["learner"] == learner, "uplift_score"].to_numpy(),
        index=scores.loc[scores["learner"] == learner, "record_id"].to_numpy(),
    )
    rebuilt = ev.decile_table(holdout, series, n_bins=10, seed=20261107)
    original = published[published["learner"] == learner].drop(
        columns="learner").reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, original, check_dtype=False, atol=1e-12)


def test_published_qini_coefficients_follow_from_the_published_curves():
    curves = table("uplift_curves_holdout")
    ranking = table("uplift_ranking_summary").set_index("learner")
    conversion = curves[curves["outcome"] == "conversion"]
    spend = curves[curves["outcome"] == "spend"]
    assert set(conversion["decile"]) == set(range(0, 11))
    for learner in ranking.index:
        assert ranking.loc[learner, "holdout_qini_coefficient"] == pytest.approx(
            ev.qini_coefficient(conversion[conversion["learner"] == learner]), abs=1e-9)
        assert ranking.loc[learner, "holdout_qini_coefficient_spend"] == pytest.approx(
            ev.qini_coefficient(spend[spend["learner"] == learner]), abs=1e-6)
        assert conversion[conversion["learner"] == learner]["share_targeted"].iloc[-1] \
            == pytest.approx(1.0)


def test_model_summary_is_strict_json_and_agrees_with_the_head_to_head_table():
    text = (TABLES / "model_summary.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    summary = json.loads(text)
    head_to_head = table("policy_head_to_head")
    assert summary["comparison"] == "mens_vs_control"
    assert summary["outcome"] == "conversion"
    deciles = table("uplift_deciles_holdout")
    assert deciles.groupby("learner")["n_customers"].sum().eq(summary["holdout_rows"]).all()
    published = summary["targeting_vs_propensity"]
    assert len(published) == len(head_to_head)
    for row, (_, csv_row) in zip(published, head_to_head.iterrows()):
        assert row["share_targeted"] == pytest.approx(csv_row["share_targeted"])
        assert row["incremental_conversions_difference"] == pytest.approx(
            csv_row["incremental_conversions_difference"])
        assert row["difference_ci_low"] == pytest.approx(csv_row["difference_ci_low"])
    ranking = pd.DataFrame(summary["ranking"]).set_index("learner")
    pd.testing.assert_frame_equal(
        ranking.reset_index(), table("uplift_ranking_summary"), check_dtype=False, atol=1e-12)


def test_the_head_to_head_difference_is_the_difference_of_the_two_lists():
    head = table("policy_head_to_head")
    assert head["uplift_minus_propensity_conversions"].to_numpy() == pytest.approx(
        (head["uplift_conversions"] - head["propensity_conversions"]).to_numpy(), abs=1e-9)
    assert head["uplift_minus_propensity_spend_usd"].to_numpy() == pytest.approx(
        (head["uplift_spend_usd"] - head["propensity_spend_usd"]).to_numpy(), abs=1e-6)
    assert (head["difference_ci_low"] <= head["uplift_minus_propensity_conversions"]
            ).all()
    assert (head["uplift_minus_propensity_conversions"] <= head["difference_ci_high"]
            ).all()
    assert head["n_selected"].iloc[0] < head["n_selected"].iloc[-1]


def test_policy_values_table_spreads_every_budget_and_strategy(test_split):
    policies = table("targeting_policy_values")
    assert set(policies["strategy"]) == {"uplift_t_learner", "uplift_s_learner",
                                         "propensity_response_model", "random"}
    budgets = sorted(policies["share_targeted"].unique())
    assert budgets[0] == 0.05 and budgets[-1] == 1.0
    assert len(policies) == len(policies["strategy"].unique()) * len(budgets)
    for _, group in policies.groupby("strategy"):
        ordered = group.sort_values("share_targeted")
        assert ordered["n_selected"].is_monotonic_increasing
        assert (ordered["n_selected"]
                == (ordered["share_targeted"] * ordered["n_selected"].iloc[-1])
                .round().astype(int)).all()
        assert (ordered["n_selected_treated"] + ordered["n_selected_control"]
                == ordered["n_selected"]).all()
    full = policies[policies["share_targeted"] == 1.0]
    assert len(full) == policies["strategy"].nunique()
    # Emailing the whole holdout is the holdout ATE by definition: whatever it
    # was ranked by, every customer is selected, so all four strategies must
    # collapse onto the same number - and that number is the test-split effect,
    # not the full-sample one quoted in the experiment table.
    treated = test_split.loc[test_split["segment"] == "Mens E-Mail", "conversion"]
    control = test_split.loc[test_split["segment"] == "No E-Mail", "conversion"]
    holdout_ate = float(treated.mean() - control.mean())
    assert (full["uplift_conversion_per_customer"] - holdout_ate).abs().max() < 1e-12
    assert full["n_selected"].nunique() == 1


def test_reports_quote_figures_that_are_recomputed_not_typed(real, config):
    from campaign_ml.power import assumptions_from_control

    report = (resolve("reports") / "power_analysis.md").read_text(encoding="utf-8")
    control = assumptions_from_control(
        real, control_label=config["experiment"]["arms"]["control"])
    assert f"{100 * control.baseline_conversion:.4f}%" in report
    reality = table("design_reality_check").iloc[0]
    assert f"{100 * reality['mde_conversion_absolute']:.4f} pp" in report
    assert "0.5728" not in report


def test_dashboard_extracts_are_the_same_numbers_with_their_columns_intact():
    effects = table("experiment_effects")
    published = pd.read_csv(TABLEAU / "campaign_incrementality_effects.csv")
    assert set(published.columns) == set(effects.columns)
    assert len(published) < len(effects)
    assert published["measure"].eq("risk_difference").any()
    for name, twin in [("uplift_deciles", "uplift_deciles_holdout"),
                       ("uplift_curves", "uplift_curves_holdout"),
                       ("policy_head_to_head", "policy_head_to_head"),
                       ("targeting_policy_values", "targeting_policy_values"),
                       ("uplift_customer_scores", "holdout_customer_scores")]:
        left = pd.read_csv(TABLEAU / f"{name}.csv")
        right = table(twin)
        assert left.shape == right.shape, name
        numeric = left.select_dtypes("number").columns
        np.testing.assert_allclose(left[numeric].to_numpy(float),
                                   right[numeric].to_numpy(float), rtol=0, atol=0,
                                   equal_nan=True)


def test_the_holdout_scores_cover_every_test_row_for_every_learner(test_split):
    scores = table("holdout_customer_scores")
    learners = scores["learner"].unique().tolist()
    assert len(learners) == 3
    assert scores["comparison"].nunique() == 1
    for learner in learners:
        ids = scores.loc[scores["learner"] == learner, "record_id"]
        assert set(ids) == set(test_split.index)
        assert len(ids) == len(test_split) == 12800
    propensity = scores[scores["learner"] == "propensity"]
    assert propensity["uplift_score"].isna().all()
    assert propensity["response_score"].between(0, 1).all()
