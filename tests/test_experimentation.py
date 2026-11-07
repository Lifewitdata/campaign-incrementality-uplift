"""Statistical core: hand-checkable values, invariants, and design checks.

Each test either pins a number that can be recomputed by hand, or asserts a
property the estimator must have (a midpoint, an identity, a monotone widening).
Nothing here re-derives the implementation and calls that a test.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from campaign_ml import experimentation as ex
from campaign_ml.config import load_config

ALPHA = 0.05


@pytest.fixture(scope="module")
def hand_case():
    """Ten per arm: 2 conversions vs 1, so every figure is arithmetic."""
    treatment = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    control = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    return treatment, control


def test_risk_difference_matches_hand_arithmetic(hand_case):
    treatment, control = hand_case
    out = ex.two_proportion_effect(treatment, control, ALPHA)
    assert out["effect"] == pytest.approx(0.1)
    assert out["mean_treatment"] == pytest.approx(0.2)
    assert out["mean_control"] == pytest.approx(0.1)
    assert out["relative_effect"] == pytest.approx(1.0)
    # unpooled SE: sqrt(0.2*0.8/10 + 0.1*0.9/10)
    assert out["se"] == pytest.approx(np.sqrt(0.025))
    z_crit = stats.norm.ppf(0.975)
    assert out["ci_low"] == pytest.approx(0.1 - z_crit * np.sqrt(0.025))
    assert out["ci_high"] == pytest.approx(0.1 + z_crit * np.sqrt(0.025))
    # pooled SE for the test statistic: sqrt(0.15*0.85*0.2)
    assert out["test_statistic"] == pytest.approx(0.1 / np.sqrt(0.15 * 0.85 * 0.2))


def test_risk_difference_ci_is_centred_on_the_effect(synth_clean):
    out = ex.two_proportion_effect(
        synth_clean.loc[synth_clean["segment"] == ex.MENS, "conversion"].to_numpy(float),
        synth_clean.loc[synth_clean["segment"] == ex.CONTROL, "conversion"].to_numpy(float),
        ALPHA,
    )
    assert (out["ci_low"] + out["ci_high"]) / 2 == pytest.approx(out["effect"])


def test_wilson_interval_matches_the_score_formula():
    lo, hi = ex.wilson_interval(267, 21307, ALPHA)
    z = stats.norm.ppf(0.975)
    p = 267 / 21307
    centre = (p + z**2 / (2 * 21307)) / (1 + z**2 / 21307)
    spread = (z / (1 + z**2 / 21307)) * np.sqrt(p * (1 - p) / 21307 + z**2 / (4 * 21307**2))
    assert lo == pytest.approx(centre - spread)
    assert hi == pytest.approx(centre + spread)
    assert lo < p < hi


def test_wilson_interval_of_zero_events_starts_at_zero():
    lo, hi = ex.wilson_interval(0, 500, ALPHA)
    assert lo == pytest.approx(0.0)
    assert hi > 0


def test_newcombe_interval_brackets_and_is_asymmetric(synth_clean):
    t = synth_clean.loc[synth_clean["segment"] == ex.MENS, "conversion"]
    c = synth_clean.loc[synth_clean["segment"] == ex.CONTROL, "conversion"]
    lo, hi = ex.newcombe_interval(t.mean(), len(t), c.mean(), len(c), ALPHA)
    rd = ex.two_proportion_effect(t.to_numpy(float), c.to_numpy(float), ALPHA)
    assert lo < rd["effect"] < hi
    assert abs(rd["effect"] - lo) != pytest.approx(abs(hi - rd["effect"]), abs=1e-6)


def test_welch_uses_t_not_normal_and_agrees_with_scipy():
    rng = np.random.default_rng(7)
    treatment = rng.lognormal(4, 1, 400)
    control = rng.lognormal(3.8, 1, 350)
    out = ex.welch_mean_effect(treatment, control, ALPHA)
    reference = stats.ttest_ind(treatment, control, equal_var=False)
    assert out["p_value"] == pytest.approx(reference.pvalue)
    assert out["test_statistic"] == pytest.approx(reference.statistic)
    assert out["effect"] == pytest.approx(treatment.mean() - control.mean())


def test_bootstrap_p_value_cannot_be_below_its_own_floor():
    rng = np.random.default_rng(3)
    treatment = rng.lognormal(4.2, 1.0, 300)
    control = rng.lognormal(3.0, 1.0, 300)
    out = ex.bootstrap_mean_difference(treatment, control, replicates=500, seed=11, alpha=ALPHA)
    assert out["p_value"] >= out["p_value_floor"] - 1e-12
    assert out["p_value_floor"] == pytest.approx(2 / 501)
    assert out["effect"] == pytest.approx(treatment.mean() - control.mean())


def test_bootstrap_is_reproducible_for_a_given_seed():
    rng = np.random.default_rng(5)
    a, b = rng.random(200), rng.random(200)
    first = ex.bootstrap_mean_difference(a, b, replicates=300, seed=99, alpha=ALPHA)
    second = ex.bootstrap_mean_difference(a, b, replicates=300, seed=99, alpha=ALPHA)
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]


def test_bootstrap_and_welch_agree_on_the_point_estimate(synth_clean):
    t = synth_clean.loc[synth_clean["segment"] == ex.MENS, "spend"].to_numpy(float)
    c = synth_clean.loc[synth_clean["segment"] == ex.CONTROL, "spend"].to_numpy(float)
    boot = ex.bootstrap_mean_difference(t, c, replicates=800, seed=1, alpha=ALPHA)
    welch = ex.welch_mean_effect(t, c, ALPHA)
    assert boot["effect"] == pytest.approx(welch["effect"])
    assert boot["ci_low"] < welch["effect"] < boot["ci_high"]


def test_skewed_spend_needs_a_non_normal_interval(synth_clean):
    spend = synth_clean["spend"].to_numpy(float)
    report = ex.distribution_report(spend)
    assert report["share_zero"] > 0.9
    assert report["skewness"] > 5
    assert report["max"] > 10 * report["mean"]


def test_spend_decomposition_is_exact_by_construction(synth_clean):
    out = ex.spend_decomposition(synth_clean, ex.MENS, ex.CONTROL, ALPHA)
    assert out["identity_residual"] == pytest.approx(0.0, abs=1e-12)
    assert (out["conversion_driven_part"] + out["basket_driven_part"]
           == pytest.approx(out["total_spend_effect"]))
    assert out["conversion_driven_share"] == pytest.approx(
        out["conversion_driven_part"] / out["total_spend_effect"]
    )


def test_spend_decomposition_is_all_buyers_when_baskets_are_identical():
    """Synthetic case: same basket in both arms, so only conversion can move spend."""
    df = pd.DataFrame({
        "segment": ["A"] * 1000 + ["B"] * 1000,
        "conversion": [1] * 200 + [0] * 800 + [1] * 100 + [0] * 900,
        "spend": [50.0] * 200 + [0.0] * 800 + [50.0] * 100 + [0.0] * 900,
    })
    out = ex.spend_decomposition(df, "A", "B", ALPHA)
    assert out["basket_driven_part"] == pytest.approx(0.0)
    assert out["conversion_driven_part"] == pytest.approx(out["total_spend_effect"])
    assert out["identity_residual"] == pytest.approx(0.0, abs=1e-15)


def test_sample_ratio_test_is_zero_for_a_balanced_allocation():
    df = pd.DataFrame({"segment": ["No E-Mail"] * 100 + ["Mens E-Mail"] * 100
                              + ["Womens E-Mail"] * 100})
    out = ex.sample_ratio_test(df, "segment", 1 / 3)
    assert out["chi2"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["degrees_of_freedom"] == pytest.approx(2.0)


def test_holm_adjustment_matches_the_textbook_steps():
    out = ex.holm_adjust([0.01, 0.02, 0.03], alpha=0.05)
    assert list(out.p_adjusted.round(6)) == [0.03, 0.04, 0.04]
    assert list(out.reject_adjusted) == [True, True, True]


def test_holm_never_inflates_a_p_value_past_one_or_below_raw():
    p = [0.001, 0.2, 0.45, 0.8, 0.99]
    out = ex.holm_adjust(p, alpha=0.05)
    adj = out.p_adjusted.to_numpy()
    assert (adj >= np.asarray(p) - 1e-15).all()
    assert (adj <= 1.0).all()
    assert (np.diff(adj[np.argsort(p)]) >= -1e-15).all()
    assert list(out.reject_adjusted) == [True, False, False, False, False]


def test_familywise_interval_widens_with_each_extra_comparison():
    base = ex.familywise_bounds(0.1, 0.02, 0.0604, 0.1396, 0.95, 1)
    twice = ex.familywise_bounds(0.1, 0.02, 0.0604, 0.1396, 0.95, 2)
    four = ex.familywise_bounds(0.1, 0.02, 0.0604, 0.1396, 0.95, 4)
    assert base[2] == pytest.approx(0.95)
    assert twice[2] == pytest.approx(0.975)
    assert four[0] < twice[0] < base[0] and four[1] > twice[1] > base[1]


def test_familywise_interval_keeps_an_asymmetric_shape():
    low, high, level = ex.familywise_bounds(0.5, 0.1, 0.35, 0.75, 0.95, 2, asymmetric=True)
    assert level == pytest.approx(0.975)
    assert abs(low - 0.5) > abs(0.35 - 0.5) and abs(high - 0.5) > abs(0.75 - 0.5)
    # scaled about the original centre, not about the effect
    assert (low + high) / 2 == pytest.approx((0.35 + 0.75) / 2)


@pytest.fixture(scope="module")
def analysis(synth_clean):
    config = copy.deepcopy(load_config())
    config["experiment"]["bootstrap"]["replicates"] = 400
    config["experiment"]["bootstrap"]["replicates_small"] = 200
    return ex.run_experiment_analysis(synth_clean, config)


def test_analysis_covers_both_comparisons_and_three_outcomes(analysis):
    assert set(analysis["comparison"]) == {"mens_vs_control", "womens_vs_control"}
    assert set(analysis["outcome"]) == {"conversion", "visit", "spend"}
    assert len(analysis) == 12


def test_analysis_recovers_the_injected_uplift(analysis):
    conversion = analysis[(analysis["outcome"] == "conversion")
                          & (analysis["measure"] == "risk_difference")
                          & (analysis["confirmatory"])]
    # fixture injects a 2 pp uplift on a 6% control base
    assert conversion["effect"].mean() == pytest.approx(0.02, abs=0.012)
    assert (conversion["ci_low"] < 0.02).all() and (conversion["ci_high"] > 0.02).all()


def test_exactly_one_confirmatory_estimate_per_comparison_and_outcome(analysis):
    grouped = analysis.groupby(["comparison", "outcome"])["confirmatory"].sum()
    assert (grouped == 1).all(), grouped
    assert set(analysis["n_comparisons_in_family"]) == {2}


def test_relative_lift_and_welch_are_labelled_supplementary(analysis):
    supplementary = analysis[~analysis["confirmatory"]]
    assert set(supplementary["measure"]) == {"relative_lift", "mean_difference"}
    assert supplementary["p_holm"].isna().all()
    assert supplementary["reject_holm"].isna().all()
    assert analysis[analysis["confirmatory"]]["p_holm"].notna().all()


def test_holm_multiplier_is_applied_within_an_outcome(analysis):
    spend = analysis[(analysis["outcome"] == "spend") & (analysis["confirmatory"])]
    smallest = spend["p_value"].idxmin()
    assert spend.loc[smallest, "p_holm"] == pytest.approx(
        2 * spend.loc[smallest, "p_value"]
    )


def test_incremental_totals_scale_the_effect_by_contacts(analysis):
    spend = analysis[(analysis["outcome"] == "spend") & (analysis["measure"] == "risk_difference")]
    assert spend.empty
    mens = analysis[(analysis["comparison"] == "mens_vs_control")
                    & (analysis["outcome"] == "spend")].iloc[0]
    assert mens["incremental_total"] == pytest.approx(mens["effect"] * mens["n_contacted"])
    assert mens["incremental_unit"] == "USD"


def test_covariate_adjustment_agrees_with_the_unadjusted_contrast(synth_clean):
    covariates = ["recency", "history", "newbie", "mens", "womens", "zip_code", "channel"]
    out = ex.regression_adjusted_effect(synth_clean, "conversion", ex.MENS, covariates,
                                        ex.CONTROL, model="ols")
    unadjusted = ex.two_proportion_effect(
        synth_clean.loc[synth_clean["segment"] == ex.MENS, "conversion"].to_numpy(float),
        synth_clean.loc[synth_clean["segment"] == ex.CONTROL, "conversion"].to_numpy(float),
        ALPHA,
    )
    assert out["effect_scale"] == "risk difference"
    assert out["adjusted_effect"] == pytest.approx(unadjusted["effect"], abs=0.01)
    assert abs(out["relative_change_vs_unadjusted"]) < 25.0


def test_logit_adjustment_is_reported_on_its_own_scale(synth_clean):
    out = ex.regression_adjusted_effect(synth_clean, "conversion", ex.MENS,
                                        ["recency", "history"], ex.CONTROL, model="logit")
    assert out["effect_scale"] == "log odds ratio"
    assert out["adjusted_odds_ratio"] > 1
    assert out["adjusted_effect"] != pytest.approx(out["adjusted_odds_ratio"])


def test_balance_table_clears_the_0_1_threshold(synth_clean):
    features = ["recency", "history", "newbie", "mens", "womens"]
    categorical = ["zip_code", "channel", "history_segment"]
    table = ex.balance_table(synth_clean, features, categorical, ex.CONTROL)
    assert (table["std_mean_difference"].abs() < 0.1).all()
    assert table["within_0_1"].all()
    assert set(table["arm"]) == {ex.MENS, ex.WOMENS}


def test_heterogeneity_test_returns_a_non_negative_statistic(synth_clean):
    out = ex.heterogeneity_test(synth_clean, "zip_code", ex.MENS, ex.CONTROL, "conversion")
    assert out["chi2"] >= 0
    assert 0 <= out["p_value"] <= 1
    # three zip levels, one is the reference, so two interaction terms are tested
    assert out["degrees_of_freedom"] == 2.0
    assert out["p_value"] == pytest.approx(stats.chi2.sf(out["chi2"], out["degrees_of_freedom"]))


def test_heterogeneity_test_does_not_claim_structure_from_a_uniform_effect(synth_clean):
    """The synthetic uplift is constant by construction, so the interaction must not be significant."""
    out = ex.heterogeneity_test(synth_clean, "channel", ex.MENS, ex.CONTROL, "conversion")
    assert out["p_value"] > 0.05


def test_subgroup_effects_keep_every_level_and_its_width(synth_clean):
    table = ex.subgroup_effects(synth_clean, ["zip_code"], ex.MENS, ex.CONTROL, "conversion")
    assert len(table) == synth_clean["zip_code"].nunique()
    assert (table["ci_high"] >= table["ci_low"]).all()


def test_welch_handles_two_constant_arms_without_dividing_by_zero():
    """A degenerate basket comparison must return an exact difference, not crash."""
    out = ex.welch_mean_effect(np.full(50, 50.0), np.full(50, 50.0), ALPHA)
    assert out["effect"] == pytest.approx(0.0)
    assert out["se"] == 0.0
    assert out["ci_low"] == out["ci_high"] == 0.0
    shifted = ex.welch_mean_effect(np.full(50, 60.0), np.full(50, 50.0), ALPHA)
    assert shifted["effect"] == pytest.approx(10.0)
    assert shifted["se"] == 0.0


def test_arm_summary_reconciles_with_the_raw_columns(synth_clean):
    summary = ex.arm_summary(synth_clean, confidence_level=0.95).set_index("arm")
    for arm in summary.index:
        rows = synth_clean[synth_clean["segment"] == arm]
        assert summary.loc[arm, "n_customers"] == len(rows)
        assert summary.loc[arm, "conversions"] == int(rows["conversion"].sum())
        assert summary.loc[arm, "spend_total"] == pytest.approx(rows["spend"].sum())
        assert summary.loc[arm, "conversion_rate_ci_low"] \
            <= summary.loc[arm, "conversion_rate"] \
            <= summary.loc[arm, "conversion_rate_ci_high"]
