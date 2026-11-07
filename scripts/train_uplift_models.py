"""Phases 4 and 5: fit the learners, then judge them on an untouched holdout.

    python scripts/train_uplift_models.py

One predefined email-versus-control comparison is modelled
(``modelling.primary_comparison`` in ``configs/experiment.yaml``). Training data
tunes, validation data stops the boosting early, and the test split is opened
once - for the ranking and policy numbers that this repository publishes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from campaign_ml.config import load_config, resolve
from campaign_ml.evaluation import (
    cumulative_targeting_curve,
    decile_spread,
    decile_table,
    policy_table,
    prepare_holdout,
    qini_coefficient,
    random_strategy,
    targeting_comparison,
)
from campaign_ml.features import build_features, make_splits
from campaign_ml.tracking import Tracker, recent_runs
from campaign_ml.uplift import feature_importance, score_frame, train_comparison
from campaign_ml.validation import load_raw, prepare_clean_table

OUTCOME = "conversion"


def main() -> int:
    config = load_config()
    modelling = config["modelling"]
    exp = config["experiment"]
    seed = int(modelling["split"]["seed"])

    df = prepare_clean_table(load_raw(resolve(config["data"]["raw_path"])))
    splits = make_splits(df, config)
    bundle = build_features(splits, config)

    primary_name = modelling["primary_comparison"]
    comparison = next(c for c in exp["comparisons"] if c["name"] == primary_name)
    print(f"[split] {bundle.split_sizes} | {len(bundle.feature_names)} model features | "
          f"comparison {comparison['treatment']} vs {comparison['control']}")
    arms = pd.DataFrame({
        name: split["segment"].value_counts() for name, split in splits.items()
    }).fillna(0).astype(int)
    print("[split arms]\n" + arms.to_string())

    models = train_comparison(bundle, config, comparison, OUTCOME)
    tracker = Tracker.from_config(config)

    holdout = prepare_holdout(bundle.y_test, comparison["treatment"],
                              comparison["control"], comparison["name"])
    scores = score_frame(models).set_index("record_id")
    test_frame = df.loc[scores.index].set_index("record_id")

    tables = resolve("outputs/tables")
    tables.mkdir(parents=True, exist_ok=True)
    tableau = resolve("outputs/tableau")
    tableau.mkdir(parents=True, exist_ok=True)

    deciles_rows, curves_rows, spread_rows, run_records = [], [], [], []
    for name, entry in models.items():
        learner_scores = scores.loc[scores["learner"] == name]
        uplift = learner_scores["uplift_score"]
        response = learner_scores["response_score"]
        # A response model has no causal ranking to offer; ranking it by
        # predicted purchase probability is exactly the incumbent strategy.
        ranking = uplift if np.isfinite(uplift).all() else response

        deciles = decile_table(holdout, ranking, n_bins=int(modelling["targeting"]["deciles"]),
                               alpha=1 - float(exp["confidence_level"]),
                               seed=int(exp["bootstrap"]["seed"]))
        deciles.insert(0, "learner", name)
        deciles_rows.append(deciles)

        curve_conversion = cumulative_targeting_curve(deciles, outcome=OUTCOME)
        curve_spend = cumulative_targeting_curve(deciles, outcome="spend")
        curve_conversion.insert(0, "learner", name)
        curve_spend.insert(0, "learner", name)
        curves_rows.extend([curve_conversion, curve_spend])

        spread = {"learner": name, **decile_spread(deciles)}
        # How coarse the ranking actually is: with a stopped-early tree most
        # customers share a score, so the deciles are arbitrary within big ties.
        spread["distinct_scores_in_holdout"] = int(ranking.loc[
            holdout.frame.index].nunique())
        spread["holdout_qini_coefficient"] = qini_coefficient(curve_conversion)
        spread["holdout_qini_coefficient_spend"] = qini_coefficient(curve_spend)
        # AUC is reported on the frame each learner early-stopped on; the scope
        # column says what that frame was, so the three stay comparable.
        spread["validation_auc"] = entry["metrics"].get("validation_auc")
        spread["validation_auc_scope"] = entry["metrics"].get("validation_scope")
        spread_rows.append(spread)

        handle = tracker.start(f"{comparison['name']}-{name}", {
            "learner": name,
            "outcome": OUTCOME,
            "comparison": comparison["name"],
            "treatment": comparison["treatment"],
            "control": comparison["control"],
            "seed": seed,
            "split": modelling["split"],
            "n_model_features": len(bundle.feature_names),
            "n_train": bundle.split_sizes["train"],
            "n_validate": bundle.split_sizes["validate"],
            "n_test": bundle.split_sizes["test"],
        })
        numeric = {k: float(v) for k, v in entry["metrics"].items()
                   if isinstance(v, (int, float, np.integer, np.floating))}
        descriptors = {k: str(v) for k, v in entry["metrics"].items()
                       if not isinstance(v, (int, float, np.integer, np.floating))}
        numeric.update({
            "holdout_qini_coefficient": spread["holdout_qini_coefficient"],
            "holdout_qini_coefficient_spend": spread["holdout_qini_coefficient_spend"],
            "top_decile_observed_uplift": spread["top_decile_uplift_conversion"],
            "bottom_decile_observed_uplift": spread["bottom_decile_uplift_conversion"],
            "distinct_scores_in_holdout": spread["distinct_scores_in_holdout"],
            "test_arm_mix": holdout.n_treated / holdout.n,
        })
        handle.log_metrics(numeric)
        handle.log_tags({
            **descriptors,
            "dataset_sha256_pinned": str(config["data"]["expected_sha256"])[:12],
            "is_primary_learner": "yes" if name == "t_learner" else "no",
        })
        handle.finish()
        run_records.append({"learner": name, **numeric})

    deciles_all = pd.concat(deciles_rows, ignore_index=True)
    curves_all = pd.concat(curves_rows, ignore_index=True)
    spread_table = pd.DataFrame(spread_rows)
    deciles_all.to_csv(tables / "uplift_deciles_holdout.csv", index=False)
    curves_all.to_csv(tables / "uplift_curves_holdout.csv", index=False)
    spread_table.to_csv(tables / "uplift_ranking_summary.csv", index=False)

    strategies = {
        "uplift_t_learner": scores.loc[scores["learner"] == "t_learner", "uplift_score"],
        "uplift_s_learner": scores.loc[scores["learner"] == "s_learner", "uplift_score"],
        "propensity_response_model": scores.loc[scores["learner"] == "propensity",
                                                "response_score"],
        "random": random_strategy(holdout, seed),
    }
    budgets = [float(b) for b in modelling["targeting"]["budget_levels"]]
    alpha = 1 - float(exp["confidence_level"])
    policies = policy_table(holdout, strategies, budgets, exp["bootstrap"], alpha)
    policies.to_csv(tables / "targeting_policy_values.csv", index=False)

    uplift_run = policies[policies["strategy"] == "uplift_t_learner"].set_index("share_targeted")
    propensity_run = policies[policies["strategy"] == "propensity_response_model"].set_index(
        "share_targeted"
    )
    head_to_head = pd.DataFrame({
        "uplift_conversions": uplift_run["incremental_conversions_in_slice"],
        "propensity_conversions": propensity_run["incremental_conversions_in_slice"],
        "uplift_spend_usd": uplift_run["incremental_spend_in_slice_usd"],
        "propensity_spend_usd": propensity_run["incremental_spend_in_slice_usd"],
        "n_selected": uplift_run["n_selected"],
        "uplift_conversion_ci_low": uplift_run["uplift_conversion_ci_low"],
        "uplift_conversion_ci_high": uplift_run["uplift_conversion_ci_high"],
        "propensity_conversion_ci_low": propensity_run["uplift_conversion_ci_low"],
        "propensity_conversion_ci_high": propensity_run["uplift_conversion_ci_high"],
    }).reset_index()

    # The two lists select mostly the same customers, so their own intervals say
    # nothing about the gap between them. Price the gap directly instead.
    paired_cfg = {**exp["bootstrap"],
                  "replicates": int(exp["bootstrap"]["replicates_small"])}
    paired = [targeting_comparison(holdout, strategies["uplift_t_learner"],
                                   strategies["propensity_response_model"], budget,
                                   paired_cfg, alpha) for budget in budgets]
    paired_t = targeting_comparison(holdout, strategies["uplift_t_learner"],
                                    strategies["random"], budgets[1], paired_cfg, alpha)
    head_to_head = head_to_head.merge(
        pd.DataFrame(paired)[["share_targeted", "incremental_conversions_difference",
                              "difference_ci_low", "difference_ci_high",
                              "incremental_spend_difference_usd", "spend_difference_ci_low",
                              "spend_difference_ci_high"]],
        on="share_targeted", validate="one_to_one",
    )
    head_to_head["uplift_minus_propensity_conversions"] = (
        head_to_head["incremental_conversions_difference"])
    head_to_head["uplift_minus_propensity_spend_usd"] = (
        head_to_head["incremental_spend_difference_usd"])
    head_to_head.to_csv(tables / "policy_head_to_head.csv", index=False)
    head_to_head.to_csv(tableau / "policy_head_to_head.csv", index=False)
    print(f"[paired] t-learner vs propensity at {budgets[0]:.0%} targeted: "
          f"{paired[0]['incremental_conversions_difference']:+.2f} conversions "
          f"[{paired[0]['difference_ci_low']:+.2f}, "
          f"{paired[0]['difference_ci_high']:+.2f}], "
          f"{paired[0]['incremental_spend_difference_usd']:+,.0f} USD "
          f"[{paired[0]['spend_difference_ci_low']:+,.0f}, "
          f"{paired[0]['spend_difference_ci_high']:+,.0f}]")
    print(f"[paired] t-learner vs random at {budgets[1]:.0%}: "
          f"{paired_t['incremental_conversions_difference']:+.2f} conversions "
          f"[{paired_t['difference_ci_low']:+.2f}, {paired_t['difference_ci_high']:+.2f}]")

    scores.reset_index().to_csv(tables / "holdout_customer_scores.csv", index=False)
    scores.reset_index().to_csv(tableau / "uplift_customer_scores.csv", index=False)
    deciles_all.to_csv(tableau / "uplift_deciles.csv", index=False)
    curves_all.to_csv(tableau / "uplift_curves.csv", index=False)
    policies.to_csv(tableau / "targeting_policy_values.csv", index=False)

    importance = feature_importance(models)
    importance.to_csv(tables / "feature_importance.csv", index=False)
    bundle.feature_name_map.to_csv(tables / "feature_name_map.csv", index=False)

    model_dir = resolve("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in models.items():
        joblib.dump(entry["model"], model_dir / f"{comparison['name']}--{name}.joblib")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comparison": comparison["name"],
        "outcome": OUTCOME,
        "split_sizes": bundle.split_sizes,
        "n_model_features": len(bundle.feature_names),
        "holdout_rows": holdout.n,
        "holdout_treated_share": holdout.n_treated / holdout.n,
        "learners": {name: dict(entry["metrics"]) for name, entry in models.items()},
        "ranking": spread_table,
        "targeting_vs_propensity": head_to_head,
        "mlflow_runs": recent_runs(tracker),
        "mlflow_enabled": tracker.enabled,
    }
    resolve("outputs/tables/model_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )

    print("[ranking]")
    for row in spread_table.to_dict("records"):
        print(f"  {row['learner']:12s} top-decile uplift "
              f"{100 * row['top_decile_uplift_conversion']:+.3f} pp, "
              f"bottom-decile {100 * row['bottom_decile_uplift_conversion']:+.3f} pp, "
              f"Qini coefficient {row['holdout_qini_coefficient']:+.2f} conversions")
    print(f"[exports] {len(deciles_all)} decile rows, {len(policies)} policy rows -> {tableau}")
    return 0


def _clean(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value if value is None or isinstance(value, (int, float, str, bool)) else str(value)


def _json_safe(obj):
    """Strict JSON has no NaN: a published artefact must parse in every reader."""
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict("records"))
    return _clean(obj)


if __name__ == "__main__":
    raise SystemExit(main())
