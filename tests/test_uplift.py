"""Learner behaviour: what each score means, who each model is allowed to see, and whether the
machinery can find heterogeneity when heterogeneity actually exists.

The last file of tests in this suite matters most: a null uplift result is only credible if the
same code recovers an injected, unevenly-distributed effect.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_experiment

from campaign_ml import features as ft
from campaign_ml import uplift as up
from campaign_ml.config import load_config
from campaign_ml.validation import ARMS, prepare_clean_table

MENS, CONTROL = "Mens E-Mail", "No E-Mail"


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def comparison(config):
    return next(c for c in config["experiment"]["comparisons"] if c["treatment"] == MENS)


@pytest.fixture(scope="session")
def bundle(synth_clean, config):
    return ft.build_features(ft.make_splits(synth_clean, config), config)


@pytest.fixture(scope="session")
def models(bundle, config, comparison):
    return up.train_comparison(bundle, config, comparison, "conversion")


def test_lightgbm_parameters_keep_the_experiment_settings_and_force_binary_objective(config):
    params = up.lightgbm_params(config, seed=123)
    assert params["objective"] == "binary"
    assert params["random_state"] == 123
    assert params["verbosity"] == -1
    assert params["importance_type"] == "gain"
    assert params["n_jobs"] == -1
    for key, value in config["modelling"]["lightgbm"].items():
        if key not in {"objective", "random_state", "verbosity", "importance_type"}:
            assert params[key] == value
    assert up.lightgbm_params(config, 123, extra={"num_leaves": 7})["num_leaves"] == 7


def test_an_arm_that_collapsed_to_one_class_is_refused_rather_than_fitted():
    x = pd.DataFrame({"recency": np.arange(50, dtype=float)})
    params = {"objective": "binary", "verbose": -1, "n_estimators": 5}
    with pytest.raises(ValueError, match="single outcome class"):
        up._fit_lgbm(params, x, np.zeros(50), x, np.zeros(50), 3)


def test_every_learner_returns_the_two_scores_it_claims_to(models, bundle):
    for name, entry in models.items():
        scores = entry["scores_test"]
        assert list(scores.columns) == up.UPLIFT_COLUMNS
        assert len(scores) == len(bundle.x_test)
        assert scores.index.equals(bundle.x_test.index)
        assert scores["response_score"].between(0.0, 1.0).all()


def test_the_propensity_baseline_publishes_no_uplift_score(models):
    """It never saw a control arm, so it cannot claim to know the incremental effect."""
    scores = models["propensity"]["scores_test"]
    assert scores["uplift_score"].isna().all()
    assert scores["response_score"].notna().all()


def test_both_learners_score_the_control_rows_they_were_never_fitted_on(models, bundle):
    arms = bundle.y_test["segment"]
    for name in ("s_learner", "t_learner"):
        scores = models[name]["scores_test"]
        uplift = scores.loc[arms.to_numpy() == CONTROL, "uplift_score"]
        assert uplift.notna().all() and uplift.abs().sum() > 0


def test_the_s_learners_uplift_is_a_difference_of_two_predictions_not_a_probability(models):
    scores = models["t_learner"]["scores_test"]
    assert np.isfinite(scores["uplift_score"]).all()
    assert (scores["uplift_score"] <= scores["response_score"]).all()


def test_propensity_is_fitted_on_emailed_rows_only(models, bundle):
    n_treated = int((bundle.y_train["segment"] == MENS).sum())
    n_validate = int((bundle.y_validate["segment"] == MENS).sum())
    metrics = models["propensity"]["metrics"]
    assert metrics["train_n"] == float(n_treated)
    assert metrics["validation_n"] == float(n_validate)
    assert metrics["validation_scope"] == "emailed arm only"


def test_s_learner_pools_both_arms_and_says_so(models, bundle):
    n_treated = int((bundle.y_train["segment"] == MENS).sum())
    n_control = int((bundle.y_train["segment"] == CONTROL).sum())
    metrics = models["s_learner"]["metrics"]
    assert metrics["train_n"] == n_treated + n_control
    assert metrics["n_treated_rows"] == float(n_treated)
    assert metrics["n_control_rows"] == float(n_control)
    assert metrics["validation_scope"] == "both arms pooled"


def test_t_learner_keeps_the_two_surfaces_separate_and_labels_the_headline(models, bundle):
    metrics = models["t_learner"]["metrics"]
    assert {"treated_train_auc", "control_train_auc", "validation_auc",
            "control_validation_auc", "best_iteration_treated",
            "best_iteration_control"} <= set(metrics)
    assert metrics["validation_scope"] == "emailed arm only"
    # the unprefixed validation numbers are the emailed arm's, by construction
    assert metrics["validation_n"] == float((bundle.y_validate["segment"] == MENS).sum())
    assert metrics["control_validation_n"] == float((bundle.y_validate["segment"] == CONTROL).sum())
    assert metrics["treated_train_n"] == metrics["control_train_n"]


def test_the_test_split_cannot_influence_anything_measured_during_fit(bundle, config, comparison):
    """Scrambling the holdout matrix must leave every fitted metric identical."""
    baseline = up.train_comparison(bundle, config, comparison, "conversion")
    scrambled = copy.deepcopy(bundle)
    order = np.random.default_rng(7).permutation(len(scrambled.x_test))
    scrambled.x_test = scrambled.x_test.iloc[order]
    permuted = up.train_comparison(scrambled, config, comparison, "conversion")
    for name in baseline:
        assert baseline[name]["metrics"] == permuted[name]["metrics"]
    assert not np.allclose(baseline["t_learner"]["scores_test"]["uplift_score"].to_numpy(),
                           permuted["t_learner"]["scores_test"]["uplift_score"].to_numpy())


def test_training_is_reproducible_from_the_seed(bundle, config, comparison):
    first = up.train_comparison(bundle, config, comparison, "conversion")
    second = up.train_comparison(bundle, config, comparison, "conversion")
    for name in first:
        pd.testing.assert_frame_equal(first[name]["scores_test"], second[name]["scores_test"])
        assert first[name]["metrics"] == second[name]["metrics"]


def test_scores_from_all_three_learners_stack_into_one_evaluable_table(models, bundle):
    frame = up.score_frame(models)
    assert list(frame.columns) == ["record_id", "comparison", "learner",
                                   "uplift_score", "response_score"]
    assert len(frame) == 3 * len(bundle.x_test)
    assert set(frame["learner"]) == set(models)
    assert frame["comparison"].nunique() == 1
    assert frame.groupby("learner")["record_id"].nunique().eq(len(bundle.x_test)).all()


def test_feature_importance_reports_gain_for_every_fitted_surface(models):
    table = up.feature_importance(models, top=200)
    assert list(table.columns) == ["learner", "feature", "gain"]
    assert set(table["learner"]) == set(models)
    assert (table["gain"] >= 0).all()
    assert table["feature"].map(str.islower).all()
    top = up.feature_importance(models, top=3)
    assert len(top) <= len(table)
    assert top["gain"].is_monotonic_decreasing


def test_the_s_learner_uses_the_treatment_indicator_it_was_handed(models):
    table = up.feature_importance(models, top=200)
    s_columns = set(table.loc[table["learner"] == "s_learner", "feature"])
    assert "treated" in s_columns
    assert "treated" not in set(table.loc[table["learner"] == "propensity", "feature"])


def test_with_treatment_never_mutates_the_matrix_it_reads():
    x = pd.DataFrame({"recency": [1.0, 2.0]})
    out = up._with_treatment(x, 1, "treated")
    assert list(x.columns) == ["recency"]
    assert list(out.columns) == ["recency", "treated"]


def test_arm_masks_split_a_frame_into_the_two_arms_of_one_comparison(bundle):
    treated, control = up.arm_masks(bundle.y_train, MENS, CONTROL)
    segment = bundle.y_train["segment"]
    assert (treated & control).sum() == 0
    assert treated.sum() == int((segment == MENS).sum())
    assert control.sum() == int((segment == CONTROL).sum())
    # the third arm of the experiment is not silently folded into either side
    assert int((treated | control).sum()) < len(bundle.y_train)
    assert not segment[treated | control].eq("Womens E-Mail").any()


def _heterogeneous_frame(n_per_arm: int = 12000, seed: int = 4242) -> pd.DataFrame:
    """Sure things versus persuadables: the pattern a propensity list gets backwards.

    Experienced customers already convert at 20% and an email moves them to 22%
    (a 2 point nudge). New customers sit at 4% untouched and 12% emailed - an 8
    point effect. So the group with the *higher* response rate under email is the
    group with the *lower* incremental effect, which is exactly the case uplift
    modelling exists to catch and a propensity list is guaranteed to miss.
    """
    frame = synthetic_experiment(n_per_arm=n_per_arm, seed=seed, uplift=0.0)
    rng = np.random.default_rng(seed + 1)
    emailed = frame["segment"].to_numpy() != ARMS[0]
    new = frame["newbie"].to_numpy() == 1
    probability = np.where(new, 0.04, 0.20) + np.where(new, 0.08, 0.02) * emailed
    conversion = (rng.random(len(frame)) < probability).astype(int)
    visit = np.maximum(conversion, (rng.random(len(frame)) < 0.30).astype(int))
    spend = np.where(conversion == 1, np.round(rng.lognormal(3.4, 1.0, len(frame)), 2), 0.0)
    out = frame.copy()
    out["conversion"], out["visit"], out["spend"] = conversion, visit, spend
    return out


@pytest.fixture(scope="session")
def heterogeneous(config):
    clean = prepare_clean_table(_heterogeneous_frame())
    splits = ft.make_splits(clean, config)
    return ft.build_features(splits, config), splits["test"]


def test_the_machinery_finds_an_effect_that_only_exists_for_one_group(
        heterogeneous, config, comparison):
    """A negative uplift result on real data means something only if this test passes."""
    bundle, test_frame = heterogeneous
    models = up.train_comparison(bundle, config, comparison, "conversion")
    scores = models["t_learner"]["scores_test"]
    new = (test_frame.loc[scores.index, "newbie"] == 1).to_numpy()

    persuadable = float(scores["uplift_score"].loc[new].mean())
    sure_thing = float(scores["uplift_score"].loc[~new].mean())
    assert persuadable > 0.04
    assert sure_thing < 0.04
    assert persuadable > 2 * sure_thing


def test_propensity_would_have_ranked_the_same_experiment_backwards(
        heterogeneous, config, comparison):
    """The comparator is not a weaker uplift model; it answers a different question."""
    bundle, test_frame = heterogeneous
    models = up.train_comparison(bundle, config, comparison, "conversion")
    scores = models["propensity"]["scores_test"]
    response = scores["response_score"].to_numpy()
    uplift = models["t_learner"]["scores_test"]["uplift_score"].to_numpy()
    new = (test_frame.loc[scores.index, "newbie"] == 1).to_numpy()
    decile = len(new) // 10

    share_new_by_response = new[np.argsort(-response)[:decile]].mean()
    share_new_by_uplift = new[np.argsort(-uplift)[:decile]].mean()
    assert share_new_by_response < 0.40
    assert share_new_by_uplift > 0.75
    # the two lists disagree about most customers, which is the whole argument
    top_response = set(np.argsort(-response)[:decile])
    assert len(top_response & set(np.argsort(-uplift)[:decile])) < decile * 0.6


def test_a_uniformly_injected_uplift_is_recovered_as_a_uniform_score(models, synth_clean):
    """With no heterogeneity in the data, the estimated uplift should not be heterogeneous."""
    scores = models["t_learner"]["scores_test"]
    overall = float(scores["uplift_score"].mean())
    assert 0.002 < overall < 0.05
    by_group = scores["uplift_score"].groupby(
        synth_clean.loc[scores.index, "zip_code"]).mean()
    assert (by_group - overall).abs().max() < 0.03
