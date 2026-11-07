"""Uplift learners: a response baseline, an S-learner and a T-learner.

All three are trained on the **training split only**, tuned on the **validation
split only** (early stopping), and are never shown the test split. Each returns
a scores frame with two columns:

* ``response_score`` - predicted probability of converting if emailed. This is
  what a purchase-propensity list would rank on.
* ``uplift_score`` - predicted *difference* in conversion probability between
  being emailed and not. Undefined for the response baseline, so it is NaN
  rather than a number pretending to be a causal estimate.

Neither column is an individual treatment effect. Randomisation lets us
estimate effects for groups; an individual's uplift is not observable, and the
scores here are conditional expectations that only earn trust on the holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

UPLIFT_COLUMNS = ["uplift_score", "response_score"]


def lightgbm_params(config: dict, seed: int, extra: dict | None = None) -> dict:
    params = dict(config["modelling"]["lightgbm"])
    params.update({
        "objective": "binary",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
        "importance_type": "gain",
    })
    if extra:
        params.update(extra)
    return params


def _fit_lgbm(params: dict, x: pd.DataFrame, y: np.ndarray, x_valid: pd.DataFrame,
              y_valid: np.ndarray, early_stopping_rounds: int) -> lgb.LGBMClassifier:
    """One gradient-boosted binary model with validation-based early stopping."""
    if len(np.unique(y)) < 2:
        raise ValueError("a training arm collapsed to a single outcome class")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x, y,
        eval_X=x_valid, eval_y=y_valid,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    return model


def _response_metrics(model: lgb.LGBMClassifier, x: pd.DataFrame, y: np.ndarray,
                      label: str) -> dict[str, float]:
    p = model.predict_proba(x, num_iteration=model.best_iteration_)[:, 1]
    prefix = "train" if label == "train" else "validation"
    return {
        f"{prefix}_log_loss": float(log_loss(y, p, labels=[0, 1])),
        f"{prefix}_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        f"{prefix}_n": float(len(y)),
    }


@dataclass
class PropensityModel:
    """Purchase-propensity baseline: P(convert | emailed), no control arm at all.

    This is the strategy the business already runs - mail the customers who look
    like buyers - and it is the honest comparator for an uplift list.
    """

    name: str = "propensity"
    model: lgb.LGBMClassifier | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def fit(self, x_treated: pd.DataFrame, y_treated: np.ndarray,
            x_valid: pd.DataFrame, y_valid: np.ndarray, params: dict,
            early_stopping_rounds: int) -> "PropensityModel":
        self.model = _fit_lgbm(params, x_treated, y_treated, x_valid, y_valid,
                               early_stopping_rounds)
        self.metrics = {
            **_response_metrics(self.model, x_treated, y_treated, "train"),
            **_response_metrics(self.model, x_valid, y_valid, "validation"),
            # This model only ever sees emailed customers, so its validation
            # frame is the emailed arm - stated here because the T-learner's
            # headline AUC comes from the same arm and the S-learner's does not.
            "validation_scope": "emailed arm only",
            "best_iteration": float(self.model.best_iteration_ or 0),
        }
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        p = self.model.predict_proba(x, num_iteration=self.model.best_iteration_)[:, 1]
        return pd.DataFrame({"uplift_score": np.nan, "response_score": p}, index=x.index)


@dataclass
class SLearner:
    """One model on both arms with ``treated`` as an input feature."""

    name: str = "s_learner"
    model: lgb.LGBMClassifier | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    treatment_column: str = "treated"

    def fit(self, x_treated: pd.DataFrame, y_treated: np.ndarray,
            x_control: pd.DataFrame, y_control: np.ndarray,
            x_valid: pd.DataFrame, y_valid: np.ndarray, treated_valid: np.ndarray,
            params: dict, early_stopping_rounds: int) -> "SLearner":
        x = pd.concat([
            _with_treatment(x_treated, 1, self.treatment_column),
            _with_treatment(x_control, 0, self.treatment_column),
        ])
        y = np.concatenate([y_treated, y_control])
        x_validated = _with_treatment(x_valid, treated_valid, self.treatment_column)
        self.model = _fit_lgbm(params, x, y, x_validated, y_valid, early_stopping_rounds)
        self.metrics = {
            **_response_metrics(self.model, x, y, "train"),
            **_response_metrics(self.model, x_validated, y_valid, "validation"),
            "validation_scope": "both arms pooled",
            "n_treated_rows": float(len(y_treated)),
            "n_control_rows": float(len(y_control)),
            "best_iteration": float(self.model.best_iteration_ or 0),
        }
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        p1 = self.model.predict_proba(
            _with_treatment(x, 1, self.treatment_column),
            num_iteration=self.model.best_iteration_)[:, 1]
        p0 = self.model.predict_proba(
            _with_treatment(x, 0, self.treatment_column),
            num_iteration=self.model.best_iteration_)[:, 1]
        return pd.DataFrame({"uplift_score": p1 - p0, "response_score": p1}, index=x.index)


@dataclass
class TLearner:
    """Two independent response surfaces; uplift is their difference."""

    name: str = "t_learner"
    model_treated: lgb.LGBMClassifier | None = None
    model_control: lgb.LGBMClassifier | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def fit(self, x_treated: pd.DataFrame, y_treated: np.ndarray,
            x_control: pd.DataFrame, y_control: np.ndarray,
            x_valid_treated: pd.DataFrame, y_valid_treated: np.ndarray,
            x_valid_control: pd.DataFrame, y_valid_control: np.ndarray,
            params: dict, early_stopping_rounds: int) -> "TLearner":
        self.model_treated = _fit_lgbm(params, x_treated, y_treated,
                                       x_valid_treated, y_valid_treated,
                                       early_stopping_rounds)
        self.model_control = _fit_lgbm(params, x_control, y_control,
                                       x_valid_control, y_valid_control,
                                       early_stopping_rounds)
        t_train = _response_metrics(self.model_treated, x_treated, y_treated, "train")
        c_train = _response_metrics(self.model_control, x_control, y_control, "train")
        t_valid = _response_metrics(self.model_treated, x_valid_treated,
                                    y_valid_treated, "validation")
        c_valid = _response_metrics(self.model_control, x_valid_control,
                                    y_valid_control, "validation")
        self.metrics = {
            **{f"treated_{k}": v for k, v in t_train.items()},
            **{f"control_{k}": v for k, v in c_train.items()},
            **{f"control_{k}": v for k, v in c_valid.items()},
            # unprefixed validation_* is the emailed arm, because that is the
            # surface the targeted list is actually ranked on.
            **t_valid,
            "validation_scope": "emailed arm only",
            "n_treated_rows": float(len(y_treated)),
            "n_control_rows": float(len(y_control)),
            "best_iteration_treated": float(self.model_treated.best_iteration_ or 0),
            "best_iteration_control": float(self.model_control.best_iteration_ or 0),
        }
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        p1 = self.model_treated.predict_proba(
            x, num_iteration=self.model_treated.best_iteration_)[:, 1]
        p0 = self.model_control.predict_proba(
            x, num_iteration=self.model_control.best_iteration_)[:, 1]
        return pd.DataFrame({"uplift_score": p1 - p0, "response_score": p1}, index=x.index)


def _with_treatment(x: pd.DataFrame, value, column: str) -> pd.DataFrame:
    out = x.copy()
    out[column] = value
    return out


def arm_masks(y: pd.DataFrame, treatment_label: str,
              control_label: str) -> tuple[pd.Series, pd.Series]:
    """Row masks for the two arms of one comparison."""
    return (y["segment"] == treatment_label), (y["segment"] == control_label)


def train_comparison(bundle, config: dict, comparison: dict, outcome: str
                     ) -> dict[str, dict[str, object]]:
    """Fit all three learners for one email-versus-control comparison.

    ``bundle`` is the :class:`campaign_ml.features.FeatureBundle`. The validation
    split tunes early stopping; the test split is never touched here.
    """
    exp = config["experiment"]
    treatment_label = comparison["treatment"]
    control_label = comparison["control"]
    seed = int(config["modelling"]["split"]["seed"])
    params = lightgbm_params(config, seed)
    patience = int(config["modelling"]["early_stopping_rounds"])

    x_train, y_train = bundle.x_train, bundle.y_train
    x_val, y_val = bundle.x_validate, bundle.y_validate

    train_treated, train_control = arm_masks(y_train, treatment_label, control_label)
    valid_treated, valid_control = arm_masks(y_val, treatment_label, control_label)

    xt, yt = x_train[train_treated], y_train.loc[train_treated, outcome]
    xc, yc = x_train[train_control], y_train.loc[train_control, outcome]
    vt, vvt = x_val[valid_treated], y_val.loc[valid_treated, outcome]
    vc, vvc = x_val[valid_control], y_val.loc[valid_control, outcome]

    models: dict[str, dict[str, object]] = {}

    propensity = PropensityModel().fit(xt, yt.to_numpy(), vt, vvt.to_numpy(), params, patience)
    models[propensity.name] = {"model": propensity, "metrics": propensity.metrics}

    s_learner = SLearner().fit(
        xt, yt.to_numpy(), xc, yc.to_numpy(),
        pd.concat([vt, vc]), np.concatenate([vvt.to_numpy(), vvc.to_numpy()]),
        np.concatenate([np.ones(len(vt)), np.zeros(len(vc))]),
        params, patience,
    )
    models[s_learner.name] = {"model": s_learner, "metrics": s_learner.metrics}

    t_learner = TLearner().fit(
        xt, yt.to_numpy(), xc, yc.to_numpy(), vt, vvt.to_numpy(), vc, vvc.to_numpy(),
        params, patience,
    )
    models[t_learner.name] = {"model": t_learner, "metrics": t_learner.metrics}

    for entry in models.values():
        entry["scores_test"] = entry["model"].predict(bundle.x_test)  # type: ignore[union-attr]
        entry["scores_validate"] = entry["model"].predict(bundle.x_validate)  # type: ignore[union-attr]
        entry["comparison"] = comparison["name"]
        entry["outcome"] = outcome
    return models


def score_frame(models: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Stack every learner's test scores into one evaluatable table."""
    frames = []
    for name, entry in models.items():
        scores: pd.DataFrame = entry["scores_test"].copy()  # type: ignore[assignment]
        scores.insert(0, "learner", name)
        scores.insert(0, "comparison", entry["comparison"])
        frames.append(scores)
    return pd.concat(frames).rename_axis("record_id").reset_index()


def feature_importance(models: dict[str, dict[str, object]], top: int = 15) -> pd.DataFrame:
    """Gain-based importances, for the model card and the interpretation notebook."""
    rows = []
    for name, entry in models.items():
        model = entry["model"]
        for attribute in ("model", "model_treated", "model_control"):
            fitted = getattr(model, attribute, None)
            if fitted is None or not hasattr(fitted, "feature_importances_"):
                continue
            importance = pd.DataFrame({
                "learner": name,
                "sub_model": attribute,
                "feature": fitted.feature_name_,
                "gain": fitted.feature_importances_,
            })
            rows.append(importance)
    table = pd.concat(rows, ignore_index=True)
    return (table.groupby(["learner", "feature"], as_index=False)["gain"].mean()
            .sort_values("gain", ascending=False).head(top).reset_index(drop=True))
