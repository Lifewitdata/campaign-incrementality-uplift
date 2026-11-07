"""Holdout evaluation of uplift rankings and targeting policies.

Everything here runs on the **test split only**, and every number is computed
from realised outcomes inside randomised arms - not from model predictions. That
is what makes the comparison defensible:

* within any subgroup of a randomised experiment, the difference in observed
  conversion between the emailed and the never-emailed customers is an unbiased
  estimate of the average effect *for that subgroup*;
* so a list built by ranking on a predicted uplift score can be audited by
  measuring what actually happened to the customers it selected;
* and a model that ranks well but predicts the wrong level still shows up here,
  because the y-axis is measured, not asserted.

The counterfactual for a given customer is still never observed. These are group
averages used to judge a targeting rule, not claims about individuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from campaign_ml.experimentation import bootstrap_mean_difference, two_proportion_effect


@dataclass
class HoldoutData:
    """One comparison's test rows, with the treatment flag made explicit."""

    comparison: str
    treatment_label: str
    control_label: str
    frame: pd.DataFrame  # index = record_id; columns: treated, conversion, visit, spend

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def n_treated(self) -> int:
        return int(self.frame["treated"].sum())


def prepare_holdout(y_test: pd.DataFrame, treatment_label: str, control_label: str,
                   comparison: str) -> HoldoutData:
    """Keep only the two arms of the predefined comparison."""
    keep = y_test[y_test["segment"].isin([treatment_label, control_label])].copy()
    keep["treated"] = (keep["segment"] == treatment_label).astype(int)
    missing = {"conversion", "visit", "spend"} - set(keep.columns)
    if missing:
        raise KeyError(f"test frame is missing outcome column(s): {sorted(missing)}")
    return HoldoutData(
        comparison=comparison,
        treatment_label=treatment_label,
        control_label=control_label,
        frame=keep[["treated", "visit", "conversion", "spend", "segment"]],
    )


def _descending_rank(scores: pd.Series, seed: int) -> pd.Series:
    """Rank highest-first, breaking ties at random rather than by row order.

    A boosted model stopped after two rounds hands thousands of customers the
    *same* predicted probability. The splits are assembled stratum by stratum, so
    row position correlates with the arm a customer was randomised into -
    positional tie-breaking would quietly sort customers by arm and manufacture
    differences that look like targeting skill. A seeded coin-flip keeps the bins
    equal-sized, arm-independent and reproducible.
    """
    values = scores.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(values)), -values))
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return pd.Series(ranks, index=scores.index, name="rank")


def assign_deciles(scores: pd.Series, n_bins: int = 10, seed: int = 0) -> pd.Series:
    """Rank into equal-sized deciles, 1 = highest predicted uplift.

    Ties are broken by seeded random draw rather than by value, which keeps the
    bins exactly equal-sized. That matters for a LightGBM score on a rare
    outcome: a large block of customers share an identical predicted probability,
    and value-based binning would fold whole deciles together.
    """
    ranks = _descending_rank(scores, seed)
    decile = np.floor((ranks.to_numpy() - 1) / len(scores) * n_bins).astype(int) + 1
    return pd.Series(np.clip(decile, 1, n_bins), index=scores.index, name="decile")


def decile_table(holdout: HoldoutData, scores: pd.Series, n_bins: int = 10,
                 alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Observed treatment effect inside each predicted-uplift decile."""
    frame = holdout.frame.join(scores.rename("score"))
    frame["decile"] = assign_deciles(frame["score"], n_bins, seed)
    rows = []
    for decile, group in frame.groupby("decile"):
        treated = group[group["treated"] == 1]
        control = group[group["treated"] == 0]
        effect = two_proportion_effect(
            treated["conversion"].to_numpy(dtype=float),
            control["conversion"].to_numpy(dtype=float),
            alpha,
        )
        spend_t = treated["spend"].to_numpy(dtype=float)
        spend_c = control["spend"].to_numpy(dtype=float)
        spend_effect = float(spend_t.mean() - spend_c.mean()) if len(spend_t) and len(spend_c) \
            else float("nan")
        spend_se = float(np.sqrt(
            (spend_t.var(ddof=1) / len(spend_t)) + (spend_c.var(ddof=1) / len(spend_c))
        )) if len(spend_t) > 1 and len(spend_c) > 1 else float("nan")
        rows.append({
            "decile": int(decile),
            "n_customers": int(len(group)),
            "n_treated": int(len(treated)),
            "n_control": int(len(control)),
            "mean_predicted_score": float(group["score"].mean()),
            "conversion_rate_treated": effect["mean_treatment"],
            "conversion_rate_control": effect["mean_control"],
            "observed_uplift_conversion": effect["effect"],
            "uplift_ci_low": effect["ci_low"],
            "uplift_ci_high": effect["ci_high"],
            "uplift_p_value": effect["p_value"],
            "observed_uplift_spend": spend_effect,
            "spend_uplift_se": spend_se,
            "visit_rate_treated": float(treated["visit"].mean()) if len(treated) else float("nan"),
            "visit_rate_control": float(control["visit"].mean()) if len(control) else float("nan"),
            "conversions_treated": float(treated["conversion"].sum()),
            "conversions_control": float(control["conversion"].sum()),
            "spend_total_treated": float(treated["spend"].sum()),
            "spend_total_control": float(control["spend"].sum()),
        })
    table = pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
    return table


def cumulative_targeting_curve(deciles: pd.DataFrame, outcome: str = "conversion") -> pd.DataFrame:
    """Uplift and Qini curves as a function of how many customers are emailed.

    Built on the classic Qini accumulation: within the top-k deciles, take the
    realised outcome in the emailed group and subtract the never-emailed group's
    outcome rescaled to the same number of people. Doing the rescaling with the
    running treated/control counts - rather than assuming the two groups are
    interchangeable - is what keeps the curve honest when the selected slice has
    an unbalanced arm mix.

    ``outcome="conversion"`` counts extra purchases; ``outcome="spend"`` is the
    same curve in dollars, which is the version the economics dashboard reads.
    """
    columns = {
        "conversion": ("conversions_treated", "conversions_control"),
        "spend": ("spend_total_treated", "spend_total_control"),
    }
    if outcome not in columns:
        raise ValueError(f"unknown outcome {outcome!r}")
    treated_column, control_column = columns[outcome]

    work = deciles.sort_values("decile").reset_index(drop=True)
    cumulative_treated = work[treated_column].cumsum()
    cumulative_control = work[control_column].cumsum()
    n_treated = work["n_treated"].cumsum()
    n_control = work["n_control"].cumsum()
    n_total = int(work["n_treated"].sum() + work["n_control"].sum())
    share = (n_treated + n_control) / n_total

    qini_gain = cumulative_treated - cumulative_control * (n_treated / n_control)
    endpoint = float(qini_gain.iloc[-1])
    curve = pd.DataFrame({
        "decile": work["decile"].astype(int),
        "share_targeted": share.to_numpy(),
        "n_targeted": (n_treated + n_control).to_numpy().astype(int),
        "cumulative_effect_per_contacted_customer": (
            cumulative_treated / n_treated - cumulative_control / n_control
        ).to_numpy(),
        "qini_gain": qini_gain.to_numpy(),
        "random_line_gain": (share * endpoint).to_numpy(),
        "outcome": outcome,
    })
    # The curve has to start at "nobody emailed, no gain" or the area under it
    # integrates only the segment between the first decile and the whole list.
    origin = pd.DataFrame([{
        "decile": 0, "share_targeted": 0.0, "n_targeted": 0,
        "cumulative_effect_per_contacted_customer": 0.0,
        "qini_gain": 0.0, "random_line_gain": 0.0, "outcome": outcome,
    }])
    return pd.concat([origin, curve], ignore_index=True)


def area_under_curve(curve: pd.DataFrame, x: str = "share_targeted",
                     y: str = "qini_gain") -> float:
    """Trapezoidal area under a targeting curve, in outcome units."""
    return float(np.trapezoid(curve[y].to_numpy(), curve[x].to_numpy()))


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's Qini curve and the random-targeting diagonal."""
    model_area = area_under_curve(curve)
    random_area = area_under_curve(curve, y="random_line_gain")
    return model_area - random_area


def policy_value(holdout: HoldoutData, score: pd.Series, budget: float,
                 bootstrap_cfg: dict, alpha: float = 0.05) -> dict[str, float]:
    """Measured value of emailing the top ``budget`` share, ranked by ``score``.

    Returns the observed incremental conversions and spending for exactly the
    customers the rule would have selected, with bootstrap intervals on the
    spending figure. This is the number a marketing budget decision turns on.
    """
    frame = holdout.frame.join(score.rename("score"))
    n_targeted = int(round(budget * len(frame)))
    if n_targeted < 20:
        raise ValueError(f"budget {budget} leaves only {n_targeted} test customers to measure on")
    ranks = _descending_rank(frame["score"], int(bootstrap_cfg["seed"]))
    selected = frame.loc[ranks.sort_values().index[:n_targeted]]
    treated = selected.loc[selected["treated"] == 1]
    control = selected.loc[selected["treated"] == 0]
    if treated.empty or control.empty:
        raise ValueError("the selected holdout slice lost one of the two randomised arms")

    conversion = two_proportion_effect(treated["conversion"].to_numpy(dtype=float),
                                       control["conversion"].to_numpy(dtype=float), alpha)
    spend = bootstrap_mean_difference(
        treated["spend"].to_numpy(dtype=float), control["spend"].to_numpy(dtype=float),
        replicates=int(bootstrap_cfg["replicates_small"]), seed=int(bootstrap_cfg["seed"]),
        alpha=alpha,
    )
    n_selected = len(selected)
    return {
        "share_targeted": budget,
        "n_selected": n_selected,
        "n_selected_treated": int(len(treated)),
        "n_selected_control": int(len(control)),
        "conversion_rate_treated": conversion["mean_treatment"],
        "conversion_rate_control": conversion["mean_control"],
        "uplift_conversion_per_customer": conversion["effect"],
        "uplift_conversion_ci_low": conversion["ci_low"],
        "uplift_conversion_ci_high": conversion["ci_high"],
        "uplift_conversion_p_value": conversion["p_value"],
        "uplift_spend_per_customer": spend["effect"],
        "uplift_spend_ci_low": spend["ci_low"],
        "uplift_spend_ci_high": spend["ci_high"],
        "incremental_conversions_in_slice": conversion["effect"] * len(treated),
        "incremental_spend_in_slice_usd": spend["effect"] * len(treated),
        "incremental_spend_scaled_to_list_usd": spend["effect"] * holdout.n_treated,
        "mean_predicted_uplift_in_slice": float(selected["score"].mean()),
        "arm_balance_in_slice": float(len(treated) / n_selected),
    }


def policy_table(holdout: HoldoutData, strategies: dict[str, pd.Series],
                 budgets: Sequence[float], bootstrap_cfg: dict,
                 alpha: float = 0.05) -> pd.DataFrame:
    """One row per (strategy, budget): what emailing that list actually returns."""
    rows = []
    for name, score in strategies.items():
        for budget in budgets:
            result = policy_value(holdout, score, budget, bootstrap_cfg, alpha)
            result["strategy"] = name
            result["comparison"] = holdout.comparison
            rows.append(result)
    table = pd.DataFrame(rows)
    ordered = ["strategy", "comparison", "share_targeted", "n_selected",
               "uplift_conversion_per_customer", "uplift_spend_per_customer"]
    return table[ordered + [c for c in table.columns if c not in ordered]]


def random_strategy(holdout: HoldoutData, seed: int) -> pd.Series:
    """The control arm of strategy comparison: mail a random slice."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.permutation(len(holdout.frame)).astype(float), index=holdout.frame.index)


def _slice_incremental(conversion: np.ndarray, spend: np.ndarray, treated: np.ndarray,
                      selected: np.ndarray) -> tuple[float, float]:
    """Incremental conversions and dollars for one selected slice of a randomised frame."""
    mask = np.zeros(len(treated), dtype=bool)
    mask[selected] = True
    in_slice = mask & (treated == 1)
    counterfactual = mask & (treated == 0)
    n_t, n_c = int(in_slice.sum()), int(counterfactual.sum())
    if n_t == 0 or n_c == 0:
        return float("nan"), float("nan")
    conv = float(conversion[in_slice].sum() - conversion[counterfactual].sum() * n_t / n_c)
    cash = float(spend[in_slice].sum() - spend[counterfactual].sum() * n_t / n_c)
    return conv, cash


def targeting_comparison(holdout: HoldoutData, score_a: pd.Series, score_b: pd.Series,
                         budget: float, bootstrap_cfg: dict,
                         alpha: float = 0.05) -> dict[str, float]:
    """Paired bootstrap of one targeting list *against another* on the same holdout.

    Two lists built from the same customers select heavily-overlapping slices, so
    each list's own confidence interval says nothing about whether one beats the
    other. Resampling customers together and re-ranking inside each resample
    prices the sampling noise in the *difference*, which is the quantity business
    question 4 actually turns on.
    """
    frame = holdout.frame.join(score_a.rename("a")).join(score_b.rename("b"))
    conversion = frame["conversion"].to_numpy(dtype=float)
    spend = frame["spend"].to_numpy(dtype=float)
    treated = frame["treated"].to_numpy(dtype=int)
    a = frame["a"].to_numpy(dtype=float)
    b = frame["b"].to_numpy(dtype=float)
    n = len(frame)
    k = int(round(budget * n))
    if k < 20 or k > n:
        raise ValueError(f"budget {budget} cannot be measured on a {n}-row holdout")

    seed = int(bootstrap_cfg["seed"])
    # Share tie-break draws so identical scores select identical customers.
    jit = np.random.default_rng(seed).random(n)

    def _top(score: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return np.lexsort((jit[rows], -score[rows]))[:k]

    all_rows = np.arange(n)
    top_a = _top(a, all_rows)
    top_b = _top(b, all_rows)
    conv_a, spend_a = _slice_incremental(conversion, spend, treated, top_a)
    conv_b, spend_b = _slice_incremental(conversion, spend, treated, top_b)

    rng = np.random.default_rng(seed + 1)
    replicates = int(bootstrap_cfg["replicates"])
    diff_conv = np.full(replicates, np.nan)
    diff_spend = np.full(replicates, np.nan)
    for draw in range(replicates):
        idx = rng.integers(0, n, size=n)
        c_s, s_s = conversion[idx], spend[idx]
        t_s = treated[idx]
        pick_a = _top(a, idx)
        pick_b = _top(b, idx)
        ca, sa = _slice_incremental(c_s, s_s, t_s, pick_a)
        cb, sb = _slice_incremental(c_s, s_s, t_s, pick_b)
        # A resample can leave one arm out of the slice; that draw has no
        # defined difference, so it stays NaN and is left out below.
        if not (np.isfinite(ca) and np.isfinite(cb) and np.isfinite(sa) and np.isfinite(sb)):
            continue
        diff_conv[draw] = ca - cb
        diff_spend[draw] = sa - sb
    kept = np.isfinite(diff_conv)
    if kept.sum() < replicates * 0.5:
        raise ValueError("too many bootstrap resamples lost an arm inside the selected slice")
    valid_conv, valid_spend = diff_conv[kept], diff_spend[kept]
    low, high = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {
        "share_targeted": budget,
        "n_selected": k,
        "incremental_conversions_difference": conv_a - conv_b,
        "difference_ci_low": float(np.percentile(valid_conv, low)),
        "difference_ci_high": float(np.percentile(valid_conv, high)),
        "incremental_spend_difference_usd": spend_a - spend_b,
        "spend_difference_ci_low": float(np.percentile(valid_spend, low)),
        "spend_difference_ci_high": float(np.percentile(valid_spend, high)),
        "bootstrap_draws_used": int(valid_conv.size),
        "incremental_conversions_a": conv_a,
        "incremental_conversions_b": conv_b,
        "incremental_spend_a_usd": spend_a,
        "incremental_spend_b_usd": spend_b,
    }


def decile_spread(deciles: pd.DataFrame) -> dict[str, float]:
    """How much the model separates the population, in measured uplift terms."""
    top = deciles.iloc[0]
    bottom = deciles.iloc[-1]
    return {
        "top_decile_uplift_conversion": float(top["observed_uplift_conversion"]),
        "bottom_decile_uplift_conversion": float(bottom["observed_uplift_conversion"]),
        "top_minus_bottom": float(top["observed_uplift_conversion"]
                                  - bottom["observed_uplift_conversion"]),
        "top_decile_uplift_spend": float(top["observed_uplift_spend"]),
        "bottom_decile_uplift_spend": float(bottom["observed_uplift_spend"]),
        "monotonic_by_decile": bool(
            np.all(np.diff(deciles["observed_uplift_conversion"].to_numpy()) <= 0.0005)
        ),
    }
