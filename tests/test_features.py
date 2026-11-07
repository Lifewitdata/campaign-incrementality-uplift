"""Feature preparation: split integrity, fit-state discipline, and leakage refusal.

The three rules this module exists to enforce are (1) no post-campaign column
reaches a model matrix, (2) preprocessing is fitted on training data only, and
(3) every split keeps all three arms. Each test attacks one of those directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campaign_ml import features as ft
from campaign_ml.config import load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def splits(synth_clean, config):
    return ft.make_splits(synth_clean, config)


def test_configured_features_exclude_every_post_campaign_column(config):
    names = ft.feature_columns(config)
    configured = set(names["numeric"]) | set(names["categorical"])
    assert configured.isdisjoint(ft.LEAKAGE_BLOCKLIST)
    assert {"recency", "history", "newbie", "mens", "womens"} <= set(names["numeric"])


def test_a_configured_leaking_column_is_rejected_not_silently_dropped(config):
    bad = {"modelling": {"features": {
        "numeric": ["recency", "spend"], "categorical": ["zip_code"], "forbidden": ["spend"],
    }}}
    with pytest.raises(ValueError, match="post-campaign column"):
        ft.feature_columns(bad)

    without_declaration = {"modelling": {"features": {
        "numeric": ["recency", "conversion"], "categorical": ["zip_code"],
    }}}
    with pytest.raises(ValueError, match="post-campaign column"):
        ft.feature_columns(without_declaration)


def test_a_column_cannot_play_two_roles_at_once(config):
    bad = {"modelling": {"features": {
        "numeric": ["recency"], "categorical": ["recency"], "forbidden": [],
    }}}
    with pytest.raises(ValueError, match="both numeric and categorical"):
        ft.feature_columns(bad)


def test_leakage_guard_fires_on_a_model_matrix_that_already_has_the_outcome():
    matrix = pd.DataFrame({"recency": [1, 2], "conversion": [0, 1]})
    with pytest.raises(ValueError, match="leakage: test matrix"):
        ft.assert_no_leakage(matrix, "test")
    ft.assert_no_leakage(pd.DataFrame({"recency": [1, 2], "zip_code": ["Urban", "Rural"]}), "test")


def test_arm_code_counts_as_leakage_because_it_is_the_label_wearing_a_hat():
    with pytest.raises(ValueError, match="arm_code"):
        ft.assert_no_leakage(pd.DataFrame({"arm_code": [0, 1]}), "train")


def test_stratum_key_refuses_a_column_that_is_not_in_the_frame(synth_clean, config):
    with pytest.raises(KeyError, match="absent column"):
        ft.stratum_key(synth_clean.drop(columns=["conversion"]), config)


def test_stratum_key_pairs_arm_with_outcome(synth_clean, config):
    key = ft.stratum_key(synth_clean, config)
    assert key.iloc[0].count("|") == len(config["modelling"]["split"]["stratify_on"]) - 1
    assert set(key.unique()) == {
        f"{arm}|{outcome}"
        for arm in synth_clean["segment"].unique()
        for outcome in (0, 1)
    }


def test_splits_are_the_configured_sizes_disjoint_and_exhaustive(synth_clean, splits, config):
    shares = config["modelling"]["split"]
    total = len(synth_clean)
    assert len(splits["train"]) == pytest.approx(total * float(shares["train"]), abs=6)
    assert len(splits["validate"]) == pytest.approx(total * float(shares["validate"]), abs=6)
    assert len(splits["test"]) == total - len(splits["train"]) - len(splits["validate"])
    indices = [set(split.index) for split in splits.values()]
    assert indices[0] & indices[1] == set() and indices[1] & indices[2] == set()
    assert indices[0] | indices[1] | indices[2] == set(synth_clean.index)


def test_every_split_keeps_all_three_arms_in_the_design_proportion(synth_clean, splits):
    overall = synth_clean["segment"].value_counts(normalize=True)
    for name, split in splits.items():
        assert set(split["segment"].unique()) == set(synth_clean["segment"].unique()), name
        share = split["segment"].value_counts(normalize=True)
        assert (share - overall).abs().max() < 0.01, f"{name} drifted from the arm mix"


def test_every_split_keeps_the_conversion_rate_of_every_arm(synth_clean, splits):
    """Treatment *and* outcome representation - otherwise the holdout is not comparable."""
    for arm, group in synth_clean.groupby("segment", observed=True):
        reference = float(group["conversion"].mean())
        for name, split in splits.items():
            observed = float(split.loc[split["segment"] == arm, "conversion"].mean())
            assert abs(observed - reference) < 0.01, f"{name}/{arm}: {observed} vs {reference}"


def test_splits_are_reproducible_from_the_seed(synth_clean, config):
    first = ft.make_splits(synth_clean, config)
    second = ft.make_splits(synth_clean, config)
    for name in first:
        assert list(first[name].index) == list(second[name].index)


def test_build_features_returns_matrices_with_no_blocked_column(splits, config):
    bundle = ft.build_features(splits, config)
    for matrix in (bundle.x_train, bundle.x_validate, bundle.x_test):
        assert set(matrix.columns).isdisjoint(ft.LEAKAGE_BLOCKLIST)
        assert not matrix.isna().any().any()
        assert np.isfinite(matrix.to_numpy()).all()
    assert bundle.feature_names == list(bundle.x_train.columns)
    assert bundle.split_sizes == {k: len(v) for k, v in splits.items()}


def test_outcome_frames_stay_aligned_row_for_row_with_the_matrices(splits, config):
    bundle = ft.build_features(splits, config)
    for split_name in ("train", "validate", "test"):
        x = getattr(bundle, f"x_{split_name}")
        y = getattr(bundle, f"y_{split_name}")
        assert list(x.index) == list(y.index)
        assert list(y["segment"]) == list(splits[split_name]["segment"])


def test_scaler_state_comes_from_training_rows_only(splits, config):
    """A transformer re-fitted on validation would re-centre validation on zero."""
    bundle = ft.build_features(splits, config)
    names = ft.feature_columns(config)
    columns = names["numeric"] + names["categorical"]

    assert bundle.x_train["history"].mean() == pytest.approx(0.0, abs=1e-12)
    assert bundle.x_train["history"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)

    shifted = splits["validate"].copy()
    shifted["history"] = shifted["history"] * 100
    transformed = bundle.preprocessor.transform(shifted[columns])
    assert abs(float(transformed["history"].mean())) > 1.0
    assert abs(float(transformed["history"].mean())) > abs(float(bundle.x_validate["history"].mean()))


def test_numeric_imputation_uses_the_training_median(splits, config):
    damaged = {k: v.copy() for k, v in splits.items()}
    train = damaged["train"]
    train.loc[train.index[:25], "history"] = np.nan
    bundle = ft.build_features(damaged, config)
    imputer = bundle.preprocessor.transformers_[0][1].named_steps["imputer"]
    median_position = list(config["modelling"]["features"]["numeric"]).index("history")
    assert imputer.statistics_[median_position] == pytest.approx(
        float(train["history"].median()))
    assert not bundle.x_train.isna().any().any()


def test_an_unseen_category_in_the_test_split_degrades_to_an_all_zero_row(splits, config):
    names = ft.feature_columns(config)
    preprocessor = ft.make_preprocessor(names["numeric"], names["categorical"])
    train = splits["train"][names["numeric"] + names["categorical"]]
    preprocessor.fit(train)
    unseen = train.head(1).copy()
    unseen["zip_code"] = "Metropolitan"  # a level the training split never saw
    encoded = preprocessor.transform(unseen)
    zip_columns = [c for c in encoded.columns if c.startswith("zip_code")]
    assert len(zip_columns) == train["zip_code"].nunique()
    assert encoded[zip_columns].to_numpy().sum() == 0.0


def test_published_spend_band_labels_become_warehouse_safe_names():
    band = "6) $750 - $1,000"
    assert ft.sanitize_feature_names([band]) == ["6_750_1_000"]
    assert all(",()$- " not in name for name in ft.sanitize_feature_names([band]))


def test_distinct_labels_that_slug_alike_keep_both_rows():
    assert ft.sanitize_feature_names(["a b", "a/b", "a b"]) == ["a_b", "a_b_2", "a_b_3"]


def test_a_label_with_no_word_characters_is_refused_rather_than_becoming_blank():
    with pytest.raises(ValueError, match="no usable characters"):
        ft.sanitize_feature_names(["---"])


def test_name_mapping_pairs_every_source_label_with_its_model_column():
    table = ft.name_mapping(["6) $750 - $1,000", "1) Under 100"], ["6_750_1_000", "1_under_100"])
    assert list(table.columns) == ["source_label", "model_column"]
    assert table["model_column"].is_unique


def test_slearner_treatment_column_is_added_without_touching_the_original_matrix(splits, config):
    bundle = ft.build_features(splits, config)
    treated = (bundle.y_train["segment"] != "No E-Mail").astype(int).to_numpy()
    with_arm = ft.add_treatment(bundle.x_train, treated)
    assert "treated" not in bundle.x_train.columns
    assert list(with_arm.columns) == list(bundle.x_train.columns) + ["treated"]
    assert with_arm["treated"].dtype == np.float64
    assert with_arm["treated"].sum() == treated.sum()
    assert with_arm.iloc[:, : len(bundle.x_train.columns)].equals(bundle.x_train)


def test_arms_per_split_reconciles_with_the_split_sizes(splits, config):
    bundle = ft.build_features(splits, config)
    arms = bundle.arms_per_split(splits)
    assert list(arms.index) == ["train", "validate", "test"]
    assert (arms.sum(axis=1) == pd.Series(bundle.split_sizes)).all()
    assert (arms > 0).all().all()
