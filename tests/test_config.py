"""The configuration is the pre-registration. These tests hold it to what it says.

Business-question order, the shared control, the primary outcome, the scenario-only
economics: all of it is decided in ``configs/experiment.yaml`` before any script
runs, so the file deserves the same scrutiny as the code.
"""

from __future__ import annotations

import copy

import pytest

from campaign_ml.config import (
    PROJECT_ROOT,
    REQUIRED_KEYS,
    Paths,
    load_config,
    resolve,
    validate_config,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_the_pinned_source_is_the_published_hillstrom_file(config):
    data = config["data"]
    assert data["expected_rows"] == 64000
    assert data["expected_columns"] == 12
    assert len(data["expected_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in data["expected_sha256"])
    assert data["raw_path"] == "data/raw/hillstrom.csv"
    assert "Hillstrom" in data["source_name"]
    assert data["attribution"].startswith("Kevin Hillstrom")


def test_every_declared_section_is_present(config):
    assert set(REQUIRED_KEYS) <= set(config)


def test_conversion_is_the_primary_outcome_and_the_others_are_secondary(config):
    outcomes = config["experiment"]["outcomes"]
    assert outcomes["primary"] == "conversion"
    assert set(outcomes["secondary"]) == {"visit", "spend"}
    assert outcomes["primary"] not in outcomes["secondary"]


def test_both_comparisons_read_against_the_same_control_arm(config):
    comparisons = config["experiment"]["comparisons"]
    assert len(comparisons) == 2
    control = config["experiment"]["arms"]["control"]
    assert {c["control"] for c in comparisons} == {control}
    assert {c["treatment"] for c in comparisons} == {
        config["experiment"]["arms"]["mens"], config["experiment"]["arms"]["womens"]}
    assert {c["role"] for c in comparisons} == {"primary", "secondary"}
    assert len({c["name"] for c in comparisons}) == 2


def test_the_arm_labels_are_the_published_campaign_categories(config):
    assert set(config["experiment"]["arms"].values()) == {
        "No E-Mail", "Mens E-Mail", "Womens E-Mail"}


def test_the_modelling_comparison_is_one_of_the_declared_ones(config):
    assert config["modelling"]["primary_comparison"] == "mens_vs_control"
    assert config["modelling"]["primary_comparison"] == \
        config["experiment"]["comparisons"][0]["name"]


def test_the_holdout_split_is_reserved_and_stratified_on_the_design(config):
    split = config["modelling"]["split"]
    assert split["train"] + split["validate"] + split["test"] == pytest.approx(1.0)
    assert split["test"] >= 0.10
    assert set(split["stratify_on"]) == {"segment", "conversion"}
    assert isinstance(split["seed"], int)


def test_no_feature_or_subgroup_variable_is_measured_after_the_send(config):
    features = config["modelling"]["features"]
    forbidden = set(features["forbidden"])
    assert {"segment", "visit", "conversion", "spend"} <= forbidden
    assert forbidden.isdisjoint(features["numeric"])
    assert forbidden.isdisjoint(features["categorical"])
    assert not (set(features["numeric"]) & set(features["categorical"]))
    assert forbidden.isdisjoint(set(config["experiment"]["subgroups"]["variables"]))


def test_subgroup_variables_are_all_pre_campaign_customer_attributes(config):
    declared = set(config["modelling"]["features"]["numeric"]) | set(
        config["modelling"]["features"]["categorical"])
    assert set(config["experiment"]["subgroups"]["variables"]) <= declared
    assert config["experiment"]["subgroups"]["outcome"] == "conversion"


def test_bootstrap_settings_are_ordered_and_seeded(config):
    bootstrap = config["experiment"]["bootstrap"]
    assert bootstrap["replicates"] > bootstrap["replicates_small"] >= 200
    assert isinstance(bootstrap["seed"], int)


def test_budget_levels_span_the_list_from_a_small_slice_to_everyone(config):
    budgets = config["modelling"]["targeting"]["budget_levels"]
    assert budgets[0] > 0.0
    assert budgets[-1] == 1.0
    assert budgets == sorted(budgets)
    assert config["modelling"]["targeting"]["deciles"] == 10


def test_multiplicity_is_declared_as_holm_within_an_outcome_family(config):
    assert config["experiment"]["multiplicity"]["method"] == "holm"
    assert config["experiment"]["alpha"] == 0.05
    assert config["experiment"]["confidence_level"] == 0.95


def test_costs_and_margins_are_configured_as_scenarios_only(config):
    economics = config["economics"]
    assert len(economics["cost_scenarios"]) == 3
    assert len(economics["margin_scenarios"]) == 3
    assert economics["email_cost_per_send"] in economics["cost_scenarios"]
    assert economics["gross_margin_rate"] in economics["margin_scenarios"]
    assert 0 < min(economics["cost_scenarios"]) < max(economics["cost_scenarios"]) < 1
    assert not any("roi" in key.lower() or "profit" in key.lower() for key in economics)


def test_the_sample_ratio_check_expects_the_balanced_design(config):
    srm = config["experiment"]["srm"]
    assert srm["expected_shares"] == pytest.approx(1 / 3, abs=1e-6)
    assert srm["min_p_value"] < config["experiment"]["alpha"]


def test_validate_config_reports_a_missing_section(config):
    broken = copy.deepcopy(config)
    del broken["economics"]
    with pytest.raises(KeyError, match="missing section"):
        validate_config(broken)


def test_validate_config_rejects_a_primary_that_was_never_declared(config):
    broken = copy.deepcopy(config)
    broken["modelling"]["primary_comparison"] = "womens_vs_mens"
    with pytest.raises(KeyError, match="not a defined comparison"):
        validate_config(broken)


def test_validate_config_rejects_split_shares_that_do_not_add_up(config):
    broken = copy.deepcopy(config)
    broken["modelling"]["split"]["train"] = 0.75
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_config(broken)


def test_validate_config_rejects_an_arm_that_no_comparison_uses(config):
    broken = copy.deepcopy(config)
    broken["experiment"]["arms"]["control"] = "Holdout"
    with pytest.raises(ValueError, match="must appear in the defined comparisons"):
        validate_config(broken)


def test_validate_config_refuses_a_subgroup_defined_by_the_outcome_itself(config):
    broken = copy.deepcopy(config)
    broken["experiment"]["subgroups"]["variables"].append("conversion")
    with pytest.raises(ValueError, match="measured after the send"):
        validate_config(broken)


def test_resolve_is_root_relative_and_passes_absolute_paths_through():
    assert resolve("data/raw/hillstrom.csv") == PROJECT_ROOT / "data/raw/hillstrom.csv"
    absolute = PROJECT_ROOT / "configs" / "experiment.yaml"
    assert resolve(absolute) == absolute
    assert resolve("outputs/tables").is_absolute()


def test_paths_and_config_agree_on_where_the_raw_file_lives(config):
    assert Paths().raw_csv == resolve(config["data"]["raw_path"])
    assert Paths().clean_csv.parts[-3:] == ("data", "processed", "campaign_customers.csv")
    for path in (Paths().tables, Paths().figures, Paths().models):
        assert path.is_relative_to(PROJECT_ROOT)
