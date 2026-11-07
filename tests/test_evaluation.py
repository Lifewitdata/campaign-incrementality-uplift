"""Holdout evaluation: curves, deciles and policy value, checked against hand arithmetic.

These functions turn a score into a business claim, so every one of them is tested
on a holdout whose truth is known by construction. The tie-breaking tests are
regression tests for a real defect found while building this project: a coarsely
ranked score plus row-order tie-breaking sorted customers by arm and invented
targeting skill out of nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campaign_ml.evaluation import (
    HoldoutData,
    assign_deciles,
    area_under_curve,
    cumulative_targeting_curve,
    decile_spread,
    decile_table,
    policy_table,
    policy_value,
    prepare_holdout,
    qini_coefficient,
    random_strategy,
    targeting_comparison,
    _descending_rank,
    _slice_incremental,
)

MENS, CONTROL = "Mens E-Mail", "No E-Mail"
BOOTSTRAP = {"replicates": 400, "replicates_small": 200, "seed": 20261107}
ALPHA = 0.05


def build_holdout(treated: np.ndarray, conversion: np.ndarray,
                  spend: np.ndarray | None = None,
                  visit: np.ndarray | None = None) -> HoldoutData:
    """A two-arm holdout with the columns ``prepare_holdout`` would have produced."""
    treated = np.asarray(treated, dtype=int)
    conversion = np.asarray(conversion, dtype=int)
    n = len(treated)
    frame = pd.DataFrame({
        "treated": treated,
        "visit": conversion if visit is None else np.asarray(visit, dtype=int),
        "conversion": conversion,
        "spend": np.where(conversion == 1, 50.0, 0.0) if spend is None
        else np.asarray(spend, dtype=float),
        "segment": np.where(treated == 1, MENS, CONTROL),
    }, index=pd.RangeIndex(1, n + 1, name="record_id"))
    return HoldoutData("mens_vs_control", MENS, CONTROL, frame)


@pytest.fixture(scope="module")
def balanced_flat_holdout() -> HoldoutData:
    """No heterogeneity at all: every slice must return the same answer."""
    rng = np.random.default_rng(2026)
    n = 4000
    treated = np.tile([1, 0], n // 2)
    conversion = (rng.random(n) < np.where(treated == 1, 0.12, 0.10)).astype(int)
    return build_holdout(treated, conversion)


@pytest.fixture(scope="module")
def stepped_holdout() -> HoldoutData:
    """Four known uplift levels, so a correct ranking has a checkable shape."""
    rng = np.random.default_rng(4242)
    levels = [(0.10, 0.10), (0.05, 0.08), (0.00, 0.12), (-0.03, 0.06)]
    frames = []
    for group, (uplift, base) in enumerate(levels):
        for arm in (1, 0):
            rows = 600
            probability = base + (uplift if arm == 1 else 0.0)
            frames.append(pd.DataFrame({
                "treated": arm,
                "visit": (rng.random(rows) < 0.30).astype(int),
                "conversion": (rng.random(rows) < probability).astype(int),
                "spend": 0.0,
                "group": group,
                "score": uplift + 0.001 * rng.random(rows),
            }))
    frame = pd.concat(frames, ignore_index=True)
    frame["segment"] = np.where(frame["treated"] == 1, MENS, CONTROL)
    frame["spend"] = np.where(frame["conversion"] == 1,
                              np.round(rng.lognormal(3.0, 0.6, len(frame)), 2), 0.0)
    holdout = HoldoutData("mens_vs_control", MENS, CONTROL,
                          frame[["treated", "visit", "conversion", "spend", "segment"]])
    return holdout, pd.Series(frame["score"].to_numpy(), index=holdout.frame.index)


def test_prepare_holdout_keeps_only_the_two_arms_of_the_predefined_comparison(synth_clean):
    y_test = synth_clean[["visit", "conversion", "spend", "segment"]].copy()
    holdout = prepare_holdout(y_test, MENS, CONTROL, "mens_vs_control")
    assert set(holdout.frame["segment"]) == {MENS, CONTROL}
    assert holdout.n == int(y_test["segment"].isin([MENS, CONTROL]).sum())
    assert holdout.n_treated == int((y_test["segment"] == MENS).sum())
    assert holdout.frame["treated"].equals(
        (holdout.frame["segment"] == MENS).astype(int))
    assert list(holdout.frame.columns) == ["treated", "visit", "conversion", "spend", "segment"]


def test_prepare_holdout_refuses_a_frame_without_the_outcome_it_must_measure():
    y_test = pd.DataFrame({"visit": [1, 0], "segment": [MENS, CONTROL]})
    with pytest.raises(KeyError, match="missing outcome column"):
        prepare_holdout(y_test, MENS, CONTROL, "mens_vs_control")


def test_descending_rank_is_a_permutation_and_orders_by_value():
    scores = pd.Series([0.5, 0.5, 0.5, 0.1, 0.9])
    ranks = _descending_rank(scores, seed=7)
    assert sorted(ranks) == [1, 2, 3, 4, 5]
    assert ranks.idxmin() == 4  # rank 1 goes to the highest score
    assert ranks.idxmax() == 3  # and the last place to the lowest
    assert _descending_rank(scores, seed=7).equals(ranks)


def test_ties_are_broken_by_a_draw_that_travels_with_the_customer():
    """With nothing but ties, no seed may systematically prefer the early rows."""
    scores = pd.Series(np.zeros(200))
    first = _descending_rank(scores, seed=7)
    second = _descending_rank(scores, seed=8)
    assert sorted(first.tolist()) == list(range(1, 201))
    assert first.idxmin() != second.idxmin()
    assert first.tolist() != second.tolist()
    assert abs(float(first.iloc[:100].mean()) - float(second.iloc[:100].mean())) < 12


def test_equal_sized_bins_whenever_the_scores_are_exactly_tied():
    """LightGBM stopped after two rounds really does hand out a handful of values."""
    scores = pd.Series(np.tile([0.001, 0.002], 500))
    deciles = assign_deciles(scores, n_bins=10, seed=3)
    counts = deciles.value_counts().sort_index()
    assert counts.min() == 100 and counts.max() == 100
    assert sorted(deciles.unique()) == list(range(1, 11))


def test_tie_breaking_does_not_sort_customers_by_arm(balanced_flat_holdout):
    """Regression test: a constant score must produce arm-balanced bins, not arm-pure bins.

    The holdout is stored treated-then-control alternating, so a positional
    tie-break would still be fine here; the defect appeared because the real test
    split is assembled stratum by stratum. Order it that way and check.
    """
    frame = balanced_flat_holdout.frame.sort_values("treated", ascending=False)
    ordered = HoldoutData("mens_vs_control", MENS, CONTROL, frame)
    constant = pd.Series(0.0068, index=frame.index)
    table = decile_table(ordered, constant, n_bins=10, seed=BOOTSTRAP["seed"])
    share = table["n_treated"] / table["n_customers"]
    assert share.between(0.35, 0.65).all(), share.to_list()
    # a score with no signal must not generate a gradient out of row position
    spread = decile_spread(table)
    assert abs(spread["top_minus_bottom"]) < 0.10
    gradient = np.corrcoef(table["decile"], table["observed_uplift_conversion"])[0, 1]
    assert abs(gradient) < 0.45, gradient


def test_decile_table_reconciles_with_the_rows_it_was_given(stepped_holdout):
    holdout, scores = stepped_holdout
    table = decile_table(holdout, scores, n_bins=6, seed=1, alpha=ALPHA)
    assert table["n_customers"].sum() == holdout.n
    assert (table["n_treated"] + table["n_control"] == table["n_customers"]).all()
    assert table["conversions_treated"].sum() == float(
        holdout.frame.loc[holdout.frame["treated"] == 1, "conversion"].sum())
    assert (table["uplift_ci_high"] >= table["uplift_ci_low"]).all()
    assert (table["mean_predicted_score"].diff().dropna() <= 0).all()


def test_a_correct_ranking_separates_the_groups_it_was_built_from(stepped_holdout):
    holdout, scores = stepped_holdout
    table = decile_table(holdout, scores, n_bins=10, seed=1, alpha=ALPHA)
    top = table.head(3)["observed_uplift_conversion"].mean()
    bottom = table.tail(3)["observed_uplift_conversion"].mean()
    assert top > 0.04
    assert bottom < 0.02
    assert top > bottom
    spread = decile_spread(table)
    assert spread["top_minus_bottom"] > 0.03
    assert spread["top_decile_uplift_conversion"] == pytest.approx(
        float(table.iloc[0]["observed_uplift_conversion"]))


def test_decile_spread_flags_a_flat_model_as_flat(balanced_flat_holdout):
    scores = pd.Series(np.random.default_rng(5).random(balanced_flat_holdout.n),
                       index=balanced_flat_holdout.frame.index)
    table = decile_table(balanced_flat_holdout, scores, seed=1)
    spread = decile_spread(table)
    assert abs(spread["top_decile_uplift_conversion"] - 0.02) < 0.03
    assert abs(spread["top_minus_bottom"]) < 0.06


def test_targeting_curve_starts_at_zero_and_ends_at_the_whole_list(stepped_holdout):
    holdout, scores = stepped_holdout
    curve = cumulative_targeting_curve(decile_table(holdout, scores, seed=1))
    assert curve.iloc[0]["decile"] == 0
    assert curve.iloc[0]["qini_gain"] == 0.0
    assert curve.iloc[0]["share_targeted"] == 0.0
    assert curve["share_targeted"].iloc[-1] == pytest.approx(1.0)
    assert curve["n_targeted"].iloc[-1] == holdout.n
    assert curve["outcome"].eq("conversion").all()


def test_the_curve_endpoint_is_the_full_population_incremental_conversion(stepped_holdout):
    holdout, scores = stepped_holdout
    curve = cumulative_targeting_curve(decile_table(holdout, scores, seed=1))
    treated = holdout.frame[holdout.frame["treated"] == 1]
    control = holdout.frame[holdout.frame["treated"] == 0]
    expected = float(treated["conversion"].sum()
                     - control["conversion"].sum() * len(treated) / len(control))
    assert curve["qini_gain"].iloc[-1] == pytest.approx(expected)
    assert curve["random_line_gain"].iloc[-1] == pytest.approx(expected)


def test_the_random_line_is_strictly_proportional_to_share_targeted(stepped_holdout):
    holdout, scores = stepped_holdout
    curve = cumulative_targeting_curve(decile_table(holdout, scores, seed=1))
    endpoint = curve["random_line_gain"].iloc[-1]
    np.testing.assert_allclose(curve["random_line_gain"].to_numpy(),
                              curve["share_targeted"].to_numpy() * endpoint, atol=1e-9)


def test_a_perfectly_linear_curve_scores_zero_qini_coefficient():
    """The comparator: if gains grow exactly with share, the model is worth nothing."""
    deciles = pd.DataFrame({
        "decile": range(1, 11),
        "n_treated": [100] * 10,
        "n_control": [100] * 10,
        "conversions_treated": [10.0] * 10,
        "conversions_control": [8.0] * 10,
        "spend_total_treated": [500.0] * 10,
        "spend_total_control": [400.0] * 10,
    })
    curve = cumulative_targeting_curve(deciles)
    assert qini_coefficient(curve) == pytest.approx(0.0, abs=1e-9)
    assert area_under_curve(curve) == pytest.approx(
        area_under_curve(curve, y="random_line_gain"), abs=1e-9)


def test_a_good_ranking_beats_a_reversed_one_on_area_under_the_curve(stepped_holdout):
    holdout, scores = stepped_holdout
    ordered = decile_table(holdout, scores, seed=1)
    reversed_scores = -scores
    reversed_table = decile_table(holdout, reversed_scores, seed=1)
    good = qini_coefficient(cumulative_targeting_curve(ordered))
    bad = qini_coefficient(cumulative_targeting_curve(reversed_table))
    assert good > 0
    assert bad < 0
    assert good > bad


def test_the_spend_curve_uses_dollars_and_shares_the_same_shape(stepped_holdout):
    holdout, scores = stepped_holdout
    table = decile_table(holdout, scores, seed=1)
    conversion_curve = cumulative_targeting_curve(table, outcome="conversion")
    spend_curve = cumulative_targeting_curve(table, outcome="spend")
    assert spend_curve["outcome"].eq("spend").all()
    np.testing.assert_allclose(conversion_curve["share_targeted"].to_numpy(),
                              spend_curve["share_targeted"].to_numpy())
    assert spend_curve["qini_gain"].iloc[-1] != 0.0


def test_targeting_curve_rejects_an_outcome_it_has_no_columns_for(stepped_holdout):
    holdout, scores = stepped_holdout
    with pytest.raises(ValueError, match="unknown outcome"):
        cumulative_targeting_curve(decile_table(holdout, scores, seed=1), outcome="recency")


def test_policy_value_selects_exactly_the_requested_number_of_customers(balanced_flat_holdout):
    scores = pd.Series(np.random.default_rng(9).random(balanced_flat_holdout.n),
                       index=balanced_flat_holdout.frame.index)
    result = policy_value(balanced_flat_holdout, scores, 0.10, BOOTSTRAP, ALPHA)
    assert result["n_selected"] == 400
    assert result["n_selected_treated"] + result["n_selected_control"] == 400
    assert 0.3 < result["arm_balance_in_slice"] < 0.7
    assert result["incremental_conversions_in_slice"] == pytest.approx(
        result["uplift_conversion_per_customer"] * result["n_selected_treated"])
    assert result["incremental_spend_scaled_to_list_usd"] == pytest.approx(
        result["uplift_spend_per_customer"] * balanced_flat_holdout.n_treated)
    assert result["uplift_conversion_ci_low"] <= result["uplift_conversion_per_customer"] \
        <= result["uplift_conversion_ci_high"]


def test_policy_value_refuses_a_slice_it_could_not_measure(balanced_flat_holdout):
    scores = pd.Series(np.random.default_rng(9).random(balanced_flat_holdout.n),
                       index=balanced_flat_holdout.frame.index)
    with pytest.raises(ValueError, match="leaves only"):
        policy_value(balanced_flat_holdout, scores, 0.001, BOOTSTRAP, ALPHA)


def test_policy_value_refuses_a_rule_that_would_email_only_one_arm(balanced_flat_holdout):
    """A score that is just the arm label is not a targeting strategy; say so."""
    arm_as_score = balanced_flat_holdout.frame["treated"].astype(float)
    with pytest.raises(ValueError, match="lost one of the two randomised arms"):
        policy_value(balanced_flat_holdout, arm_as_score, 0.10, BOOTSTRAP, ALPHA)


def test_policy_table_crosses_every_strategy_with_every_budget(balanced_flat_holdout):
    n = balanced_flat_holdout.n
    rng = np.random.default_rng(11)
    strategies = {
        "t_learner": pd.Series(rng.random(n), index=balanced_flat_holdout.frame.index),
        "propensity": pd.Series(rng.random(n), index=balanced_flat_holdout.frame.index),
    }
    table = policy_table(balanced_flat_holdout, strategies, [0.05, 0.10, 0.20], BOOTSTRAP, ALPHA)
    assert len(table) == 6
    assert list(table.columns[:6]) == ["strategy", "comparison", "share_targeted",
                                      "n_selected", "uplift_conversion_per_customer",
                                      "uplift_spend_per_customer"]
    assert set(table["strategy"]) == set(strategies)
    assert table["n_selected"].tolist() == [200, 400, 800] * 2
    assert table["comparison"].eq("mens_vs_control").all()


def test_random_strategy_is_a_shuffled_sequence_number_and_is_reproducible(balanced_flat_holdout):
    first = random_strategy(balanced_flat_holdout, seed=4)
    assert first.equals(random_strategy(balanced_flat_holdout, seed=4))
    np.testing.assert_array_equal(np.sort(first.to_numpy()), np.arange(balanced_flat_holdout.n))
    result = policy_value(balanced_flat_holdout, first, 0.25, BOOTSTRAP, ALPHA)
    # a random quarter of the list reproduces the experiment-wide effect, give or take noise
    assert abs(result["uplift_conversion_per_customer"] - 0.02) < 0.04
    assert result["uplift_conversion_ci_low"] < 0.02 < result["uplift_conversion_ci_high"]
    assert result["arm_balance_in_slice"] == pytest.approx(0.5, abs=0.05)


def test_slice_incremental_matches_a_hand_count():
    treated = np.array([1, 1, 0, 0, 0, 0])
    conversion = np.array([1, 0, 1, 0, 0, 0])
    spend = np.array([60.0, 0.0, 30.0, 0.0, 0.0, 0.0])
    conv, cash = _slice_incremental(conversion, spend, treated, np.array([0, 1, 2, 3, 4, 5]))
    # 2 emailed vs 4 control: the control total is rescaled by 2/4 before subtraction
    assert conv == pytest.approx(1 - 1 * (2 / 4))
    assert cash == pytest.approx(60 - 30 * (2 / 4))
    # one emailed buyer against one control buyer, both converted: no incremental gain
    conv, _ = _slice_incremental(conversion, spend, treated, np.array([0, 2]))
    assert conv == pytest.approx(0.0)


def test_slice_incremental_is_undefined_when_a_slice_loses_an_arm():
    treated = np.array([1, 1, 0, 0])
    conversion = np.array([1, 0, 1, 0])
    conv, cash = _slice_incremental(conversion, conversion * 10.0, treated, np.array([0, 1]))
    assert np.isnan(conv) and np.isnan(cash)


def test_comparing_a_list_with_itself_is_exactly_zero_noise(stepped_holdout):
    holdout, scores = stepped_holdout
    out = targeting_comparison(holdout, scores, scores, 0.20, BOOTSTRAP, ALPHA)
    assert out["incremental_conversions_difference"] == pytest.approx(0.0)
    assert out["difference_ci_low"] == pytest.approx(out["difference_ci_high"], abs=1e-12)
    assert out["incremental_conversions_a"] == pytest.approx(out["incremental_conversions_b"])


def test_head_to_head_is_antisymmetric_and_reports_the_draws_it_used(stepped_holdout):
    holdout, scores = stepped_holdout
    forward = targeting_comparison(holdout, scores, -scores, 0.20, BOOTSTRAP, ALPHA)
    reverse = targeting_comparison(holdout, -scores, scores, 0.20, BOOTSTRAP, ALPHA)
    assert forward["incremental_conversions_difference"] == pytest.approx(
        -reverse["incremental_conversions_difference"])
    assert forward["difference_ci_low"] == pytest.approx(
        -reverse["difference_ci_high"], abs=1e-9)
    assert forward["n_selected"] == int(round(0.20 * holdout.n))
    assert 0 < forward["bootstrap_draws_used"] <= BOOTSTRAP["replicates"]


def test_a_genuine_targeting_advantage_survives_its_own_interval(stepped_holdout):
    """If this fails, the paired bootstrap is not pricing the difference at all."""
    holdout, scores = stepped_holdout
    out = targeting_comparison(holdout, scores, -scores, 0.20, BOOTSTRAP, ALPHA)
    assert out["incremental_conversions_difference"] > 0
    assert out["difference_ci_low"] > 0
    assert out["incremental_spend_difference_usd"] == pytest.approx(
        out["incremental_spend_a_usd"] - out["incremental_spend_b_usd"])


def test_head_to_head_is_reproducible_from_the_seed(stepped_holdout):
    holdout, scores = stepped_holdout
    first = targeting_comparison(holdout, scores, -scores, 0.10, BOOTSTRAP, ALPHA)
    second = targeting_comparison(holdout, scores, -scores, 0.10, BOOTSTRAP, ALPHA)
    assert first == second


def test_head_to_head_refuses_a_budget_it_cannot_measure(stepped_holdout):
    holdout, scores = stepped_holdout
    with pytest.raises(ValueError, match="cannot be measured"):
        targeting_comparison(holdout, scores, -scores, 0.001, BOOTSTRAP, ALPHA)
    with pytest.raises(ValueError, match="cannot be measured"):
        targeting_comparison(holdout, scores, -scores, 1.5, BOOTSTRAP, ALPHA)
