"""Scenario economics for the Campaign Economics & Reliability dashboard.

The dataset has no cost or margin data. Every figure here multiplies a measured
holdout increment by an *assumed* margin and send cost from configs/experiment.yaml,
so the output is a scenario grid, never realised ROI.
"""

from __future__ import annotations

import itertools

import pandas as pd

from campaign_ml.config import PATHS, load_config, resolve


def main() -> None:
    config = load_config()
    econ = config["economics"]
    policies = pd.read_csv(PATHS.tables / "targeting_policy_values.csv")
    rows = []
    for margin, cost in itertools.product(econ["margin_scenarios"], econ["cost_scenarios"]):
        frame = policies[["strategy", "comparison", "share_targeted", "n_selected",
                          "incremental_spend_in_slice_usd", "uplift_spend_ci_low",
                          "uplift_spend_ci_high", "n_selected_treated"]].copy()
        frame["assumed_gross_margin_rate"] = margin
        frame["assumed_cost_per_send_usd"] = cost
        frame["scenario_send_cost_usd"] = frame["n_selected_treated"] * cost
        frame["scenario_incremental_margin_usd"] = frame["incremental_spend_in_slice_usd"] * margin
        frame["scenario_net_value_usd"] = (frame["scenario_incremental_margin_usd"]
                                           - frame["scenario_send_cost_usd"])
        frame["scenario_net_value_ci_low_usd"] = (
            frame["uplift_spend_ci_low"] * frame["n_selected_treated"] * margin
            - frame["scenario_send_cost_usd"])
        frame["scenario_net_value_ci_high_usd"] = (
            frame["uplift_spend_ci_high"] * frame["n_selected_treated"] * margin
            - frame["scenario_send_cost_usd"])
        frame["is_base_case"] = (margin == econ["gross_margin_rate"]
                                 and cost == econ["email_cost_per_send"])
        frame["label"] = "SCENARIO - assumed margin and cost, not realised ROI"
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(PATHS.tables / "economics_scenarios.csv", index=False)
    dashboard = resolve("outputs/dashboard")
    out.to_csv(dashboard / "campaign_economics_scenarios.csv", index=False)
    base = out[out["is_base_case"] & (out["share_targeted"] == 1.0)]
    print(f"[economics] {len(out)} scenario rows; base case, full list:")
    print(base[["strategy", "scenario_net_value_usd", "scenario_net_value_ci_low_usd",
                "scenario_net_value_ci_high_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
