"""Data-quality gates for the Hillstrom experiment table.

Every check returns a named, machine-readable result so the same assertions can
be raised as a hard failure in the pipeline, reported in the README, or shown on
the reliability dashboard. A failed gate stops the run: an incrementality
analysis built on a mis-labelled treatment arm is worse than no analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

SOURCE_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]

# Source-file typo: the retail geography label is "Suburban".
ZIP_CODE_LABELS = {"Surburban": "Suburban", "Rural": "Rural", "Urban": "Urban"}
HISTORY_SEGMENTS = [
    "1) $0 - $100",
    "2) $100 - $200",
    "3) $200 - $350",
    "4) $350 - $500",
    "5) $500 - $750",
    "6) $750 - $1,000",
    "7) $1,000 +",
]
CHANNELS = ["Web", "Phone", "Multichannel"]

# Lower bound (inclusive) of each published spend band.
HISTORY_BAND_EDGES = {
    "1) $0 - $100": 0.0,
    "2) $100 - $200": 100.0,
    "3) $200 - $350": 200.0,
    "4) $350 - $500": 350.0,
    "5) $500 - $750": 500.0,
    "6) $750 - $1,000": 750.0,
    "7) $1,000 +": 1000.0,
}

ARMS = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
ARM_CODES = {"No E-Mail": "control", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}
ARM_ORDER = ["control", "mens", "womens"]

NUMERIC_BOUNDS = {
    "recency": (1, 12),
    "history": (0.0, 5000.0),
    "newbie": (0, 1),
    "mens": (0, 1),
    "womens": (0, 1),
    "visit": (0, 1),
    "conversion": (0, 1),
    "spend": (0.0, 1000.0),
}

# Columns that only exist because the campaign already ran. Never model inputs.
POST_CAMPAIGN_COLUMNS = ("segment", "visit", "conversion", "spend")
FEATURE_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]


class DataQualityError(RuntimeError):
    """Raised when a hard data-quality gate fails."""


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    value: Any = None

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if isinstance(row["value"], (np.integer, np.floating)):
            row["value"] = row["value"].item()
        elif isinstance(row["value"], (list, tuple)):
            row["value"] = ", ".join(str(v) for v in row["value"])
        elif isinstance(row["value"], float) and not np.isfinite(row["value"]):
            row["value"] = None
        return row


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def add(self, name: str, passed: bool, detail: str, value: Any = None) -> Check:
        check = Check(name, bool(passed), detail, value)
        self.checks.append(check)
        return check

    def extend(self, other: "ValidationReport") -> None:
        self.checks.extend(other.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([check.as_dict() for check in self.checks])

    def raise_if_failed(self) -> None:
        if self.failed:
            lines = "\n".join(f"  - {check.name}: {check.detail}" for check in self.failed)
            raise DataQualityError(f"{len(self.failed)} data-quality check(s) failed:\n{lines}")


def load_raw(path: Path | str) -> pd.DataFrame:
    """Read the source CSV with the exact column order the 2008 file publishes."""
    df = pd.read_csv(path)
    if list(df.columns) != SOURCE_COLUMNS:
        raise DataQualityError(
            f"Unexpected columns in {path}. Expected {SOURCE_COLUMNS}, found {list(df.columns)}."
        )
    return df


def validate_schema(df: pd.DataFrame, report: ValidationReport | None = None) -> ValidationReport:
    report = report or ValidationReport()
    report.add("schema.columns", list(df.columns) == SOURCE_COLUMNS,
               f"{len(df.columns)} columns", len(df.columns))
    numeric = ["recency", "history", "mens", "womens", "newbie", "visit", "conversion", "spend"]
    bad = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    report.add("schema.numeric_dtypes", not bad, f"non-numeric numeric columns: {bad}", bad)
    categorical = ["history_segment", "zip_code", "channel", "segment"]
    bad_text = [c for c in categorical if pd.api.types.is_numeric_dtype(df[c])]
    report.add("schema.text_dtypes", not bad_text, f"numeric text columns: {bad_text}", bad_text)
    return report


def validate_row_count(df: pd.DataFrame, expected_rows: int,
                        report: ValidationReport | None = None) -> ValidationReport:
    report = report or ValidationReport()
    ok = len(df) == expected_rows
    report.add("volume.row_count", ok, f"{len(df)} rows vs expected {expected_rows}", len(df))
    return report


def validate_missing_values(df: pd.DataFrame,
                            report: ValidationReport | None = None) -> ValidationReport:
    report = report or ValidationReport()
    nulls = df.isna().sum()
    offenders = nulls[nulls > 0]
    report.add("completeness.no_nulls", offenders.empty,
               f"columns with nulls: {dict(offenders)}", int(nulls.sum()))
    blank_text = {
        c: int((df[c].astype("string").str.strip() == "").sum())
        for c in ["history_segment", "zip_code", "channel", "segment"]
    }
    offenders_text = {c: n for c, n in blank_text.items() if n}
    report.add("completeness.no_blank_labels", not offenders_text,
               f"blank label counts: {offenders_text}", sum(offenders_text.values()))
    return report


def validate_domains(df: pd.DataFrame, arms: Iterable[str] = ARMS,
                     report: ValidationReport | None = None) -> ValidationReport:
    """Treatment labels must match the published arms exactly, spelling included."""
    report = report or ValidationReport()
    observed_arms = sorted(df["segment"].astype(str).str.strip().unique().tolist())
    report.add("treatment.labels_exact", observed_arms == sorted(arms),
               f"observed {observed_arms} vs expected {sorted(arms)}", observed_arms)

    levels = {
        "history_segment": sorted(df["history_segment"].astype(str).unique().tolist()),
        "zip_code": sorted(df["zip_code"].astype(str).unique().tolist()),
        "channel": sorted(df["channel"].astype(str).unique().tolist()),
    }
    report.add("domain.history_segment", set(levels["history_segment"]) <= set(HISTORY_SEGMENTS),
               f"history_segment levels {levels['history_segment']}", levels["history_segment"])
    report.add("domain.zip_code", set(levels["zip_code"]) <= set(ZIP_CODE_LABELS),
               f"zip_code levels {levels['zip_code']}", levels["zip_code"])
    report.add("domain.channel", set(levels["channel"]) <= set(CHANNELS),
               f"channel levels {levels['channel']}", levels["channel"])

    offenders = {c: df.loc[~df[c].between(lo, hi)].shape[0]
                 for c, (lo, hi) in NUMERIC_BOUNDS.items()}
    out_of_range = {c: n for c, n in offenders.items() if n}
    report.add("range.numeric_bounds", not out_of_range,
               f"rows outside documented ranges: {out_of_range}", sum(out_of_range.values()))
    return report


def validate_binaries(df: pd.DataFrame,
                      report: ValidationReport | None = None) -> ValidationReport:
    report = report or ValidationReport()
    for column in ["visit", "conversion", "newbie", "mens", "womens"]:
        values = sorted(df[column].unique().tolist())
        report.add(f"binary.{column}", values == [0, 1], f"{column} values {values}", values)
    return report


def validate_outcome_consistency(df: pd.DataFrame,
                                 report: ValidationReport | None = None) -> ValidationReport:
    """Spend requires a conversion; a conversion requires a visit in this file."""
    report = report or ValidationReport()
    spend_no_conversion = int(((df["spend"] > 0) & (df["conversion"] == 0)).sum())
    report.add(
        "consistency.spend_implies_conversion",
        spend_no_conversion == 0,
        f"{spend_no_conversion} rows spend > 0 without conversion=1",
        spend_no_conversion,
    )
    conversion_no_visit = int(((df["conversion"] == 1) & (df["visit"] == 0)).sum())
    report.add(
        "consistency.conversion_implies_visit",
        conversion_no_visit == 0,
        f"{conversion_no_visit} rows converted without visiting",
        conversion_no_visit,
    )
    report.add("consistency.spend_non_negative", bool((df["spend"] >= 0).all()),
               f"min spend {float(df['spend'].min())}", float(df["spend"].min()))
    return report


def validate_randomisation(df: pd.DataFrame, expected_share: float = 1 / 3,
                           min_p_value: float = 0.001,
                           report: ValidationReport | None = None) -> ValidationReport:
    """Sample-ratio-mismatch test against the balanced design Hillstrom published."""
    report = report or ValidationReport()
    counts = df["segment"].value_counts().reindex(list(ARMS)).fillna(0)
    total = int(counts.sum())
    expected = np.full(len(counts), total * expected_share)
    chi2, p_value = stats.chisquare(counts.to_numpy(), f_exp=expected, ddof=0)
    report.add(
        "design.srm_balance",
        p_value >= min_p_value,
        f"arm counts {counts.to_dict()}; chi-square p={p_value:.4g} vs threshold {min_p_value}",
        float(p_value),
    )
    shares = (counts / total).to_dict()
    worst_deviation = max(abs(v - expected_share) / expected_share for v in shares.values())
    report.add(
        "design.arm_share_within_1pct",
        worst_deviation <= 0.01,
        f"arm shares { {k: round(v, 5) for k, v in shares.items()} }; "
        f"worst relative deviation {worst_deviation:.4%}",
        float(worst_deviation),
    )
    return report


def validate_no_leakage(feature_columns: Iterable[str],
                        forbidden: Iterable[str] = POST_CAMPAIGN_COLUMNS,
                        report: ValidationReport | None = None) -> ValidationReport:
    """Guard rail: post-campaign columns must never reach a model matrix."""
    report = report or ValidationReport()
    forbidden = set(forbidden)
    offenders = sorted(set(feature_columns) & forbidden)
    report.add("leakage.no_post_campaign_features", not offenders,
               f"forbidden columns present: {offenders}", offenders)
    return report


def prepare_clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise labels, cast types and add the warehouse keys.

    ``record_id`` is the row position in the published file. The dataset ships no
    customer identifier, so this is a stable join key for the warehouse - it is
    not a customer key and cannot be used to follow a person across campaigns.
    """
    out = df.copy()
    out.insert(0, "record_id", np.arange(len(out), dtype="int64"))
    out["segment"] = out["segment"].astype(str).str.strip()
    out["zip_code"] = out["zip_code"].astype(str).str.strip().map(ZIP_CODE_LABELS)
    unmapped = out["zip_code"].isna().sum()
    if unmapped:
        raise DataQualityError(f"{unmapped} zip_code labels did not map to a known region")
    out["arm_code"] = out["segment"].map(ARM_CODES)
    if out["arm_code"].isna().any():
        raise DataQualityError("unmapped treatment arm found in segment")
    for column in ["history_segment", "channel"]:
        out[column] = out[column].astype(str).str.strip().astype("category")
    out["zip_code"] = out["zip_code"].astype("category")
    out["arm_code"] = pd.Categorical(out["arm_code"], categories=ARM_ORDER, ordered=True)
    out["recency"] = out["recency"].astype("int64")
    out["history"] = out["history"].astype("float64")
    out["spend"] = out["spend"].astype("float64")
    for column in ["newbie", "mens", "womens", "visit", "conversion"]:
        out[column] = out[column].astype("int64")
    out["history_band"] = _history_band(out["history"])
    out["campaign_weeks"] = 2
    return out


def _history_band(history: pd.Series) -> pd.Categorical:
    """Reproduce the published ``history_segment`` bands from raw ``history``."""
    bins = [0.0, 100.0, 200.0, 350.0, 500.0, 750.0, 1000.0, np.inf]
    return pd.cut(history, bins=bins, labels=HISTORY_SEGMENTS, right=False,
                  ordered=True)


def validate_history_band_consistency(df: pd.DataFrame,
                                      report: ValidationReport | None = None) -> ValidationReport:
    """``history_segment`` must be exactly the band of ``history`` it labels."""
    report = report or ValidationReport()
    derived = _history_band(df["history"])
    mismatched = int((derived.astype(str) != df["history_segment"].astype(str)).sum())
    report.add("consistency.history_band_matches_segment", mismatched == 0,
               f"{mismatched} rows where the band label disagrees with history", mismatched)
    return report


def run_all_checks(df: pd.DataFrame, expected_rows: int = 64000,
                   expected_share: float = 1 / 3) -> ValidationReport:
    report = ValidationReport()
    for step in (validate_schema, validate_missing_values, validate_binaries):
        step(df, report)
    validate_row_count(df, expected_rows, report)
    validate_domains(df, report=report)
    validate_outcome_consistency(df, report)
    validate_history_band_consistency(df, report)
    validate_randomisation(df, expected_share=expected_share, report=report)
    validate_no_leakage(
        [c for c in FEATURE_COLUMNS if c in df.columns], POST_CAMPAIGN_COLUMNS, report
    )
    return report
