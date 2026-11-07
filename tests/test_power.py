"""Power planning: closed-form sizes validated by simulation, not by re-running the formula.

The interesting property of a power calculator is that its answer actually
delivers the promised power when you simulate the experiment it planned. Each
test here either pins a hand-checkable number or checks a monotonicity that a
wrong formula would break.
"""

from __future__ import annotations

import numpy as np
import pytest

from campaign_ml import power as pw

RARE_CONTROL = 0.005728
MENS_CONVERSION = 0.012535
NEAR_NULL_CONVERSION = 0.006813


def test_sample_size_for_a_common_outcome_matches_the_pinned_number():
    plan = pw.sample_size_two_proportions(0.10, 0.15)
    assert plan["n_control"] == 686.0
    assert plan["n_total"] == 1372.0
    assert plan["effect_absolute"] == pytest.approx(0.05)
    assert plan["effect_relative"] == pytest.approx(0.5)


def test_pinned_sample_size_actually_delivers_80_percent_power():
    """The anchor: simulate the experiment the formula planned."""
    n = int(pw.sample_size_two_proportions(0.10, 0.15)["n_control"])
    check = pw.monte_carlo_power(0.10, 0.15, n)
    assert abs(check["simulated_power"] - 0.80) < 4 * check["standard_error_of_power"]


def test_rare_outcome_needs_two_orders_more_contact_than_a_common_one():
    """Why the 2008 design was well powered for a lift but not for a nudge."""
    rare = pw.sample_size_two_proportions(RARE_CONTROL, NEAR_NULL_CONVERSION)["n_control"]
    common = pw.sample_size_two_proportions(0.10, 0.15)["n_control"]
    assert rare > 50 * common
    check = pw.monte_carlo_power(RARE_CONTROL, NEAR_NULL_CONVERSION, int(rare))
    assert abs(check["simulated_power"] - 0.80) < 0.03


def test_larger_effect_and_higher_power_move_n_in_opposite_directions():
    small_effect = pw.sample_size_two_proportions(0.10, 0.12)["n_control"]
    big_effect = pw.sample_size_two_proportions(0.10, 0.15)["n_control"]
    assert big_effect < small_effect
    assert pw.sample_size_two_proportions(0.10, 0.15, power=0.95)["n_control"] > big_effect


def test_strict_alpha_costs_contact_and_still_delivers_the_planned_power():
    per_comparison = pw.sample_size_two_proportions(0.10, 0.15, alpha=0.05)["n_control"]
    family = pw.sample_size_two_proportions(0.10, 0.15, alpha=0.025)["n_control"]
    assert family > per_comparison
    assert pw.achieved_power_two_proportions(0.10, 0.15, int(family), alpha=0.025) \
        == pytest.approx(0.80, abs=0.01)


def test_unequal_allocation_scales_the_treatment_arm_only():
    plan = pw.sample_size_two_proportions(0.10, 0.15, allocation_ratio=0.25)
    assert plan["n_treatment"] == np.ceil(0.25 * plan["n_control"])
    assert plan["n_total"] == plan["n_control"] + plan["n_treatment"]


def test_sample_size_refuses_a_null_effect_and_an_impossible_rate():
    with pytest.raises(ValueError, match="no effect"):
        pw.sample_size_two_proportions(0.1, 0.1)
    with pytest.raises(ValueError, match="strictly between"):
        pw.sample_size_two_proportions(0.0, 0.1)
    with pytest.raises(ValueError, match="strictly between"):
        pw.sample_size_two_proportions(0.1, 1.0)


def test_mde_and_power_are_inverses_of_each_other():
    """Bisection is only right if the effect it returns detects at the target power."""
    n_per_arm = 21330
    mde = pw.mde_two_proportions(RARE_CONTROL, n_per_arm,
                                 alpha=pw.alpha_for_family(0.05, 2), power=0.80)
    assert mde == pytest.approx(0.00248, abs=1e-5)
    assert pw.achieved_power_two_proportions(RARE_CONTROL, RARE_CONTROL + mde,
                                             n_per_arm, alpha=0.025) == pytest.approx(0.80, abs=1e-3)


def test_the_2008_design_could_detect_the_effect_it_found_but_not_a_nudge():
    """Both halves of the honest statement about how lucky this experiment was."""
    n_per_arm = 21330
    observed = pw.achieved_power_two_proportions(RARE_CONTROL, MENS_CONVERSION, n_per_arm)
    nudge = pw.achieved_power_two_proportions(RARE_CONTROL, NEAR_NULL_CONVERSION, n_per_arm)
    assert observed > 0.99
    assert nudge < 0.35


def test_mde_shrinks_as_the_sample_grows():
    small = pw.mde_two_proportions(RARE_CONTROL, 5_000)
    large = pw.mde_two_proportions(RARE_CONTROL, 200_000)
    assert 0 < large < small


def test_mde_is_reported_as_undetectable_when_the_range_cannot_reach_power():
    """Say so plainly rather than returning the edge of the bracket."""
    assert np.isnan(pw.mde_two_proportions(RARE_CONTROL, 200, max_effect=0.001))


def test_sample_size_mean_agrees_with_the_normal_approximation_and_round_trips():
    delta, sd = 0.30, 9.0
    plan = pw.sample_size_mean(delta, sd)
    closed = 2 * (pw.normal_quantile(0.975) + pw.normal_quantile(0.80)) ** 2 * sd ** 2 / delta ** 2
    assert plan["n_control"] == pytest.approx(closed, rel=0.01)
    assert plan["n_total"] == 2 * plan["n_control"]
    assert pw.mde_mean(int(plan["n_control"]), sd) == pytest.approx(delta, rel=0.01)


def test_sample_size_mean_needs_the_whole_heavy_tailed_sd_not_the_buyers_only_sd():
    """A design that planned on basket size alone would be wildly under-powered."""
    all_customers = pw.sample_size_mean(0.30, sd=9.0)["n_control"]
    buyers_only = pw.sample_size_mean(0.30, sd=1.0)["n_control"]
    assert all_customers > 50 * buyers_only
    with pytest.raises(ValueError, match="positive"):
        pw.sample_size_mean(0.30, sd=0.0)


def test_monte_carlo_power_is_deterministic_and_reports_its_own_error():
    a = pw.monte_carlo_power(0.10, 0.12, 5000, simulations=500)
    b = pw.monte_carlo_power(0.10, 0.12, 5000, simulations=500)
    assert a == b
    assert a["standard_error_of_power"] > 0
    other_seed = pw.monte_carlo_power(0.10, 0.12, 5000, simulations=500, seed=99)
    assert other_seed["simulated_power"] != a["simulated_power"]


def test_monte_carlo_power_is_low_when_the_null_is_true():
    """A plan cannot claim power for an effect that does not exist."""
    check = pw.monte_carlo_power(0.10, 0.10, 2000, simulations=2000)
    assert check["simulated_power"] == pytest.approx(0.05, abs=0.02)


def test_alpha_split_for_a_shared_control_is_bonferroni_and_never_divides_by_zero():
    assert pw.alpha_for_family(0.05, 2) == 0.025
    assert pw.alpha_for_family(0.05, 1) == 0.05
    assert pw.alpha_for_family(0.05, 0) == 0.05


def test_assumptions_are_taken_from_the_control_arm_alone(synth_clean):
    """Planning on the pooled sample would hide the lift inside the baseline."""
    assumptions = pw.assumptions_from_control(synth_clean)
    control = synth_clean.loc[synth_clean["segment"] == "No E-Mail"]
    assert assumptions.baseline_conversion == pytest.approx(float(control["conversion"].mean()))
    assert assumptions.baseline_visit == pytest.approx(float(control["visit"].mean()))
    assert assumptions.spend_sd == pytest.approx(float(control["spend"].std(ddof=1)))
    assert assumptions.baseline_conversion < float(synth_clean["conversion"].mean())
    assert assumptions.n_comparisons == 2
    assert "comparisons against a shared control" in assumptions.describe()


def test_plan_holds_family_wise_sizes_at_least_as_large_as_per_comparison_sizes():
    assumptions = pw.PlanningAssumptions(baseline_conversion=0.006, baseline_visit=0.10,
                                        spend_sd=9.0, spend_mean=0.65)
    rows = pw.build_plan(assumptions)
    conversion = [r for r in rows if r["outcome"] == "conversion"]
    spend = [r for r in rows if r["outcome"] == "spend"]
    assert len(conversion) == 12 and len(spend) == 10
    for per, family in zip(conversion[0::2], conversion[1::2]):
        assert family["n_per_arm"] >= per["n_per_arm"]
        assert family["alpha_used"] == pytest.approx(0.025)
    # three arms: one control plus one emailed arm per available treatment
    for row in rows:
        assert row["n_total_three_arm"] == pytest.approx(
            row["n_per_arm"] + 2 * row["n_per_arm"])
        assert row["n_total_two_arm"] == 2 * row["n_per_arm"]
    # the spend rows simulate a mean, not a proportion, so no Monte-Carlo column
    assert all(np.isnan(r["monte_carlo_power"]) for r in spend)
    assert all(not np.isnan(r["monte_carlo_power"]) for r in conversion)


def test_conversion_sizes_grow_monotonically_as_the_targeted_lift_shrinks():
    assumptions = pw.PlanningAssumptions(baseline_conversion=0.006, baseline_visit=0.10,
                                        spend_sd=9.0, spend_mean=0.65)
    rows = [r for r in pw.build_plan(assumptions)
            if r["outcome"] == "conversion" and r["alpha_convention"] == "per comparison"]
    sizes = [r["n_per_arm"] for r in rows]
    assert sizes == sorted(sizes, reverse=True)


def test_realised_design_note_uses_the_rate_it_is_given(synth_clean):
    assumptions = pw.assumptions_from_control(synth_clean)
    observed = float(
        synth_clean.loc[synth_clean["segment"] == "Mens E-Mail", "conversion"].mean())
    note = pw.realised_design_note(3000, assumptions, observed)
    assert set(note) == {
        "n_per_arm", "observed_treatment_rate", "mde_conversion_absolute",
        "mde_conversion_relative", "mde_visit_absolute", "mde_spend_usd",
        "achieved_power_at_observed_conversion_lift",
    }
    assert note["observed_treatment_rate"] == observed
    assert 0 < note["mde_conversion_absolute"] < 0.05
    assert note["mde_conversion_relative"] == pytest.approx(
        note["mde_conversion_absolute"] / assumptions.baseline_conversion)
    assert note["mde_visit_absolute"] > note["mde_conversion_absolute"]
    assert note["mde_spend_usd"] > 0
    assert note["achieved_power_at_observed_conversion_lift"] > 0.8
    # the same effect measured on a hundredth of the list is not measurable
    tiny = pw.realised_design_note(200, assumptions, observed)
    assert tiny["achieved_power_at_observed_conversion_lift"] < 0.5
