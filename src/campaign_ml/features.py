"""Pre-campaign feature preparation with the leakage guards stated, not implied.

Three rules the rest of the project depends on:

1. Only variables that existed **before the send** may enter a model matrix.
   ``segment``, ``visit``, ``conversion``, ``spend`` and the derived ``arm_code``
   are post-campaign and rejected on sight.
2. Every transformer is fitted on the training split only. Validation and test
   data are transformed through it, never re-fit.
3. Splits preserve treatment representation, because an uplift model evaluated
   on a test set that lost its control arm cannot be evaluated at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Derived from segment, so it is the treatment label wearing a different hat.
DERIVED_TREATMENT_COLUMNS = ("arm_code",)
KEY_COLUMNS = ("record_id",)
LEAKAGE_BLOCKLIST = ("segment", "visit", "conversion", "spend", *DERIVED_TREATMENT_COLUMNS)


@dataclass
class FeatureBundle:
    """Model matrices plus the fitted preprocessing, so fit state travels with data."""

    x_train: pd.DataFrame
    x_validate: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_validate: pd.DataFrame
    y_test: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    feature_name_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    preprocessor: ColumnTransformer | None = None
    split_sizes: dict[str, int] = field(default_factory=dict)

    def arms_per_split(self, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat({name: split["segment"].value_counts() for name, split in splits.items()},
                      axis=1)
            .fillna(0)
            .astype(int)
            .T
        )


def feature_columns(config: dict) -> dict[str, list[str]]:
    """Numeric and categorical feature names, checked against the blocklist."""
    cfg = config["modelling"]["features"]
    forbidden = set(cfg.get("forbidden", [])) | set(LEAKAGE_BLOCKLIST)
    numeric = list(cfg["numeric"])
    categorical = list(cfg["categorical"])
    offenders = sorted((set(numeric) | set(categorical)) & forbidden)
    if offenders:
        raise ValueError(f"post-campaign column(s) configured as features: {offenders}")
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ValueError(f"column(s) configured as both numeric and categorical: {overlap}")
    return {"numeric": numeric, "categorical": categorical}


def assert_no_leakage(matrix: pd.DataFrame, where: str) -> None:
    """Hard stop if a blocked column ever reaches a model matrix."""
    offenders = sorted(set(matrix.columns) & set(LEAKAGE_BLOCKLIST))
    if offenders:
        raise ValueError(f"leakage: {where} matrix contains post-campaign column(s) {offenders}")


def stratum_key(df: pd.DataFrame, config: dict) -> pd.Series:
    """Composite stratum so each split keeps arm mix and outcome mix."""
    columns = config["modelling"]["split"]["stratify_on"]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"cannot stratify on absent column(s): {missing}")
    key = df[columns[0]].astype(str)
    for column in columns[1:]:
        key = key + "|" + df[column].astype(str)
    return key


def make_splits(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """60/20/20 train/validate/test, stratified on treatment and primary outcome.

    Deterministic given the seed in ``configs/experiment.yaml``. The test split is
    set aside before any model sees it and is used only for the holdout readout.
    """
    split_cfg = config["modelling"]["split"]
    seed = int(split_cfg["seed"])
    train_share, validate_share, test_share = (
        float(split_cfg["train"]), float(split_cfg["validate"]), float(split_cfg["test"])
    )
    key = stratum_key(df, config)

    holdout_share = validate_share + test_share
    train, holdout = _stratified_partition(df, key, train_share, 1 - holdout_share, seed)
    validate, test = _stratified_partition(
        holdout, key.loc[holdout.index], 0.5, 0.5, seed + 1
    )
    splits = {"train": train, "validate": validate, "test": test}
    for name, split in splits.items():
        missing_arms = set(df["segment"].unique()) - set(split["segment"].unique())
        if missing_arms:
            raise ValueError(f"{name} split lost arm(s): {sorted(missing_arms)}")
    return splits


def _stratified_partition(df: pd.DataFrame, key: pd.Series, keep_a: float, keep_b: float,
                          seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic two-way stratified split on a composite key.

    Written by hand rather than with ``train_test_split`` because the rare
    conversion stratum has only ~120 members in an arm: this puts whole strata
    into one side or the other in proportion to their size, which keeps the
    treatment-by-outcome cells integral instead of rounding them to nothing.
    """
    rng = np.random.default_rng(seed)
    index_a: list = []
    index_b: list = []
    for _, group in df.groupby(key, observed=True, sort=False):
        positions = group.index.to_numpy(copy=True)
        rng.shuffle(positions)
        n_a = int(round(keep_a * len(positions)))
        index_a.extend(positions[:n_a].tolist())
        index_b.extend(positions[n_a:].tolist())
    return df.loc[index_a], df.loc[index_b]


def make_preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    """Median-impute + standardise numerics, one-hot categoricals.

    The scaler is not required by the gradient-boosted learners but costs them
    nothing, and it keeps the logistic comparison honest. ``handle_unknown`` is
    ``ignore`` so a level that only appears in the test split degrades to an all
    zero row rather than an exception at evaluation time.
    """
    blocks = []
    if len(numeric):
        blocks.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            list(numeric),
        ))
    if len(categorical):
        blocks.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            list(categorical),
        ))
    transformer = ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)
    # Pandas output is not optional here: downstream code names columns after the
    # transform and LightGBM needs feature names, so set it once at construction.
    transformer.set_output(transform="pandas")
    return transformer


def build_features(splits: dict[str, pd.DataFrame], config: dict) -> FeatureBundle:
    """Fit preprocessing on train only, then transform every split."""
    names = feature_columns(config)
    columns = names["numeric"] + names["categorical"]
    for split in splits.values():
        assert_no_leakage(split[columns], "feature")

    preprocessor = make_preprocessor(names["numeric"], names["categorical"])
    x_train = _transform(preprocessor, splits["train"], columns, fit=True)
    x_validate = _transform(preprocessor, splits["validate"], columns)
    x_test = _transform(preprocessor, splits["test"], columns)

    outcomes = ["visit", "conversion", "spend"]
    original_names = list(preprocessor.get_feature_names_out())
    return FeatureBundle(
        x_train=x_train,
        x_validate=x_validate,
        x_test=x_test,
        y_train=splits["train"][outcomes + ["segment"]],
        y_validate=splits["validate"][outcomes + ["segment"]],
        y_test=splits["test"][outcomes + ["segment"]],
        feature_names=list(x_train.columns),
        feature_name_map=name_mapping(original_names, list(x_train.columns)),
        preprocessor=preprocessor,
        split_sizes={k: len(v) for k, v in splits.items()},
    )


def _transform(preprocessor: ColumnTransformer, frame: pd.DataFrame,
               columns: Sequence[str], fit: bool = False) -> pd.DataFrame:
    subset = frame[list(columns)]
    if fit:
        preprocessor.fit(subset)
    matrix = preprocessor.transform(subset)
    matrix.index = frame.index
    matrix.columns = sanitize_feature_names(matrix.columns)
    assert_no_leakage(matrix, "model matrix")
    return matrix


UNSAFE_CHARACTERS = re.compile(r"[^0-9a-z_]+")


def sanitize_feature_names(names: Sequence[str]) -> list[str]:
    """Model-safe, warehouse-safe column names.

    The spend bands ship as labels like ``6) $750 - $1,000``. LightGBM refuses
    commas and other JSON-special characters in feature names, and the same
    labels would need quoting in every SQL query. They are slugged here once,
    at the only place raw source labels become model columns, and the mapping is
    written out by the pipeline so a reviewer can trace a name back to its band.
    """
    safe: list[str] = []
    seen: dict[str, int] = {}
    for name in names:
        slug = UNSAFE_CHARACTERS.sub("_", str(name).lower()).strip("_")
        if not slug:
            raise ValueError(f"feature name {name!r} has no usable characters")
        seen[slug] = seen.get(slug, 0) + 1
        safe.append(slug if seen[slug] == 1 else f"{slug}_{seen[slug]}")
    return safe


def name_mapping(original: Sequence[str], safe: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({"source_label": list(original), "model_column": list(safe)})


def add_treatment(x: pd.DataFrame, treated: Sequence[int] | np.ndarray) -> pd.DataFrame:
    """Append the experimental indicator for the S-learner only.

    This is the one place the arm label enters a model, and it enters as a
    *design* variable, never as a customer attribute.
    """
    out = x.copy()
    out["treated"] = np.asarray(treated, dtype=float)
    return out
