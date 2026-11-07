"""Configuration and repo path helpers for the campaign incrementality project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


@dataclass(frozen=True)
class Paths:
    """Repo-relative locations that the pipeline reads from and writes to."""

    root: Path = PROJECT_ROOT

    @property
    def raw_csv(self) -> Path:
        return self.root / "data" / "raw" / "hillstrom.csv"

    @property
    def clean_csv(self) -> Path:
        return self.root / "data" / "processed" / "campaign_customers.csv"

    @property
    def checksum_file(self) -> Path:
        return self.root / "data" / "raw" / "hillstrom.csv.sha256"

    @property
    def tables(self) -> Path:
        return self.root / "outputs" / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "outputs" / "figures"

    @property
    def models(self) -> Path:
        return self.root / "outputs" / "models"


PATHS = Paths()


def resolve(relative: Path | str) -> Path:
    """Resolve a repo-relative path such as ``data/raw/hillstrom.csv``."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


REQUIRED_KEYS = ("data", "experiment", "modelling", "economics", "mlflow")


def validate_config(config: dict[str, Any]) -> None:
    """Fail early on a typo in ``configs/experiment.yaml`` rather than mid-run."""
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise KeyError(f"configs/experiment.yaml is missing section(s): {missing}")
    arms = config["experiment"]["arms"]
    comparisons = config["experiment"]["comparisons"]
    named = {c["name"] for c in comparisons}
    primary = config["modelling"]["primary_comparison"]
    if primary not in named:
        raise KeyError(f"modelling.primary_comparison {primary!r} is not a defined comparison")
    split = config["modelling"]["split"]
    total = split["train"] + split["validate"] + split["test"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split shares must sum to 1.0, found {total}")
    labels = {c["treatment"] for c in comparisons} | {c["control"] for c in comparisons}
    if not set(arms.values()) <= labels:
        raise ValueError("experiment.arms must appear in the defined comparisons")
    forbidden = set(config["modelling"]["features"]["forbidden"])
    offenders = sorted(set(config["experiment"].get("subgroups", {}).get("variables", []))
                       & forbidden)
    if offenders:
        raise ValueError(
            f"subgroup variable(s) measured after the send: {offenders}; "
            "a group defined by the outcome cannot be used to ask who responds"
        )
