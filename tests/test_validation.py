"""Data-quality gates: each one must pass on clean data and fail on its own defect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campaign_ml.validation import (
    ARMS,
    SOURCE_COLUMNS,
    DataQualityError,
    ValidationReport,
    prepare_clean_table,
    run_all_checks,
    validate_binaries,
    validate_domains,
    validate_history_band_consistency,
    validate_missing_values,
    validate_no_leakage,
    validate_outcome_consistency,
    validate_randomisation,
    validate_row_count,
    validate_schema,
)


def names(report: ValidationReport) -> set[str]:
    return {check.name for check in report.checks}


def failing(report: ValidationReport) -> set[str]:
    return {check.name for check in report.failed}


def test_clean_synthetic_passes_every_gate(synth_raw):
    report = run_all_checks(synth_raw, expected_rows=len(synth_raw))
    assert report.failed == [], report.to_frame().to_string()


def test_schema_gate_rejects_reordered_columns(synth_raw):
    shuffled = synth_raw[list(reversed(SOURCE_COLUMNS))]
    assert "schema.columns" in failing(validate_schema(shuffled))


def test_row_count_gate_rejects_a_missing_row(synth_raw):
    assert "volume.row_count" in failing(
        validate_row_count(synth_raw.iloc[:-1], expected_rows=len(synth_raw))
    )


def test_missing_value_gate_finds_a_blank(synth_raw):
    broken = synth_raw.copy()
    broken.loc[3, "recency"] = np.nan
    assert "completeness.no_nulls" in failing(validate_missing_values(broken))


@pytest.mark.parametrize("column,value,expected", [
    ("segment", "Both E-Mail", "treatment.labels_exact"),
    ("zip_code", "Region 4", "domain.zip_code"),
    ("channel", "Carrier Pigeon", "domain.channel"),
    ("history_segment", "8) $5,000 +", "domain.history_segment"),
    ("recency", 99, "range.numeric_bounds"),
])
def test_domain_gate_rejects_each_out_of_domain_value(synth_raw, column, value, expected):
    broken = synth_raw.copy()
    broken.loc[0, column] = value
    assert expected in failing(validate_domains(broken))


def test_binary_gate_rejects_a_non_zero_one_flag(synth_raw):
    broken = synth_raw.copy()
    broken.loc[1, "conversion"] = 2
    assert "binary.conversion" in failing(validate_binaries(broken))


def test_conversion_without_visit_is_impossible_and_gated(synth_raw):
    broken = synth_raw.copy()
    broken.loc[broken["conversion"] == 1, "visit"] = 0
    assert "consistency.conversion_implies_visit" in failing(
        validate_outcome_consistency(broken)
    )


def test_spend_without_a_conversion_is_gated(synth_raw):
    broken = synth_raw.copy()
    row = broken.index[(broken["conversion"] == 0) & (broken["spend"] == 0)][0]
    broken.loc[row, "spend"] = 12.5
    assert "consistency.spend_implies_conversion" in failing(
        validate_outcome_consistency(broken)
    )


def test_negative_spend_is_gated(synth_raw):
    broken = synth_raw.copy()
    broken.loc[broken["spend"] > 0, "spend"] = -0.01
    assert "consistency.spend_non_negative" in failing(validate_outcome_consistency(broken))


def test_history_band_gate_catches_a_mislabelled_band(synth_raw):
    broken = synth_raw.copy()
    high = broken["history"].idxmax()
    broken.loc[high, "history_segment"] = "1) $0 - $100"
    assert "consistency.history_band_matches_segment" in failing(
        validate_history_band_consistency(broken)
    )


def test_randomisation_gate_tolerates_the_real_allocation(synth_raw):
    assert validate_randomisation(synth_raw).passed


def test_randomisation_gate_rejects_a_broken_allocation(synth_raw):
    broken = synth_raw.copy()
    broken.loc[broken["segment"] == ARMS[2], "segment"] = ARMS[0]
    failed = failing(validate_randomisation(broken))
    assert "design.srm_balance" in failed
    assert "design.arm_share_within_1pct" in failed


def test_leakage_gate_names_a_post_campaign_feature():
    report = validate_no_leakage(SOURCE_COLUMNS, ["segment", "visit", "conversion", "spend"])
    assert "leakage.no_post_campaign_features" in failing(report)


def test_leakage_gate_passes_a_clean_feature_list():
    clean = ["recency", "history", "zip_code", "channel", "newbie"]
    assert validate_no_leakage(clean, ["segment", "visit", "conversion", "spend"]).passed


def test_clean_table_adds_keys_and_normalises_labels(synth_raw):
    clean = prepare_clean_table(synth_raw)
    assert clean["record_id"].tolist() == list(range(len(clean)))
    assert set(clean["zip_code"].unique()) == {"Rural", "Suburban", "Urban"}
    assert set(clean["arm_code"].unique()) == {"control", "mens", "womens"}
    assert clean["campaign_weeks"].nunique() == 1
    assert len(clean) == len(synth_raw)
    assert clean.columns.tolist()[:2] == ["record_id", "recency"]


def test_clean_table_keeps_the_published_row_order(synth_raw):
    clean = prepare_clean_table(synth_raw)
    pd.testing.assert_series_equal(clean["segment"].reset_index(drop=True),
                                   synth_raw["segment"].reset_index(drop=True))


def test_clean_table_rejects_an_unknown_arm(synth_raw):
    broken = synth_raw.copy()
    broken.loc[0, "segment"] = "Telegram"
    with pytest.raises(DataQualityError, match="unmapped treatment arm"):
        prepare_clean_table(broken)


def test_clean_table_rejects_an_unmapped_zip_label(synth_raw):
    broken = synth_raw.copy()
    broken.loc[0, "zip_code"] = "Spaceport"
    with pytest.raises(DataQualityError, match="zip_code labels"):
        prepare_clean_table(broken)


def test_surburban_typo_is_valid_raw_and_normalised_on_cleaning(synth_raw):
    """The 2008 file spells one region wrong; the gate accepts it, cleaning fixes it."""
    broken = synth_raw.copy()
    broken.loc[broken.index[:50], "zip_code"] = "Surburban"
    assert validate_domains(broken).passed
    cleaned = prepare_clean_table(broken)
    assert "Surburban" not in set(cleaned["zip_code"].unique())
    assert set(cleaned["zip_code"].unique()) == {"Rural", "Suburban", "Urban"}


def test_history_band_is_an_exact_function_of_history(synth_raw):
    clean = prepare_clean_table(synth_raw)
    derived = clean["history"].pipe(
        lambda s: pd.cut(s, bins=[0.0, 100.0, 200.0, 350.0, 500.0, 750.0, 1000.0, np.inf],
                        labels=False, right=False)
    )
    published = clean["history_segment"].astype("category").cat.codes
    assert (derived.to_numpy() == published.to_numpy()).all()
