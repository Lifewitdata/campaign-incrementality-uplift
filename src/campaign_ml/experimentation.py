"""Experiment estimates: effects, uncertainty, multiplicity and balance.

The design is a three-arm randomised experiment with a shared control, so two
comparisons read against the same baseline. Two things follow and are handled
here rather than hand-waved:

1. **Multiplicity.** Each outcome is tested twice (mens vs control, womens vs
   control). p-values are Holm-adjusted *within* an outcome and a conservative
   family-wise interval is reported next to the comparison-wise one.
2. **Skew in spending.** ``spend`` is zero-inflated and heavily right-skewed, so
   the primary interval is a percentile bootstrap, cross-checked against the
   Welch normal approximation. At ~21k customers per arm the two agree, which is
   the evidence that the normal approximation is doing its job here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

CONTROL, MENS, WOMENS = "No E-Mail", "Mens E-Mail", "Womens E-Mail"
PROPORTION_OUTCOMES = ("visit", "conversion")
CONTINUOUS_OUTCOMES = ("spend",)


@dataclass
class EffectResult:
    """One treatment-versus-control estimate on one outcome."""

    comparison: str
    outcome: str
    measure: str
    treatment_label: str
    control_label: str
    n_treatment: int
    n_control: int
    mean_treatment: float
    mean_control: float
    effect: float
    effect_type: str
    relative_effect: float | None
    se: float | None
    ci_low: float
    ci_high: float
    ci_level: float
    p_value: float
    method: str
    test_statistic: float | None = None
    n_replicates: int | None = None
    ci_low_crosscheck: float | None = None
    ci_high_crosscheck: float | None = None
    crosscheck_method: str | None = None
    # Exactly one estimate per (comparison, outcome) is the confirmatory test
    # that multiplicity control is applied to. The others are re-expressions on
    # another scale, or a second method on the same question, and adjusting them
    # again would double-count the same chance of a false positive.
    confirmatory: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AdjustmentResult:
    """The same family of tests after multiplicity control."""

    p_raw: pd.Series
    p_adjusted: pd.Series
    reject_raw: pd.Series
    reject_adjusted: pd.Series
    method: str = "holm"
    family: str = ""


def arm_counts(df: pd.DataFrame, arm_column: str = "segment") -> pd.Series:
    return df[arm_column].value_counts().reindex([CONTROL, MENS, WOMENS]).fillna(0).astype(int)


def sample_ratio_test(df: pd.DataFrame, arm_column: str = "segment",
                      expected_share: float = 1 / 3) -> dict[str, float]:
    """Chi-square goodness-of-fit of the realised allocation against 1/3 each."""
    counts = arm_counts(df, arm_column).to_numpy(dtype=float)
    total = counts.sum()
    expected = np.full(counts.size, total * expected_share)
    chi2, p_value = stats.chisquare(counts, f_exp=expected, ddof=0)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": float(counts.size - 1),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
    }


def two_proportion_effect(treatment: np.ndarray, control: np.ndarray,
                          alpha: float = 0.05) -> dict[str, float]:
    """Risk difference with an unpooled Wald interval and a pooled-variance z test."""
    n1, n2 = len(treatment), len(control)
    p1, p2 = float(treatment.mean()), float(control.mean())
    diff = p1 - p2
    se = float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_pooled = float(np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "n_treatment": n1,
        "n_control": n2,
        "mean_treatment": p1,
        "mean_control": p2,
        "effect": diff,
        "relative_effect": diff / p2 if p2 > 0 else float("nan"),
        "se": se,
        "ci_low": diff - z_crit * se,
        "ci_high": diff + z_crit * se,
        "p_value": p_value,
        "test_statistic": float(z),
    }


def welch_mean_effect(treatment: np.ndarray, control: np.ndarray,
                      alpha: float = 0.05) -> dict[str, float]:
    """Difference in means with Welch's unequal-variance t test."""
    n1, n2 = len(treatment), len(control)
    var1, var2 = float(treatment.var(ddof=1)), float(control.var(ddof=1))
    se = float(np.sqrt(var1 / n1 + var2 / n2))
    if se == 0.0:
        # Both arms are constant: the difference is exact and has no sampling
        # error, so the t test has no denominator and scipy only warns.
        diff = float(treatment.mean() - control.mean())
        return {
            "n_treatment": n1, "n_control": n2,
            "mean_treatment": float(treatment.mean()), "mean_control": float(control.mean()),
            "effect": diff,
            "relative_effect": diff / float(control.mean()) if control.mean() != 0 else float("nan"),
            "se": 0.0, "ci_low": diff, "ci_high": diff,
            "p_value": float("nan") if diff == 0 else 0.0,
            "test_statistic": float("nan"), "degrees_of_freedom": float("nan"),
        }
    stat = stats.ttest_ind(treatment, control, equal_var=False)
    df_welch = se**4 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df_welch)
    diff = float(treatment.mean() - control.mean())
    return {
        "n_treatment": n1,
        "n_control": n2,
        "mean_treatment": float(treatment.mean()),
        "mean_control": float(control.mean()),
        "effect": diff,
        "relative_effect": diff / float(control.mean()) if control.mean() != 0 else float("nan"),
        "se": se,
        "ci_low": diff - t_crit * se,
        "ci_high": diff + t_crit * se,
        "p_value": float(stat.pvalue),
        "test_statistic": float(stat.statistic),
        "degrees_of_freedom": float(df_welch),
    }


def bootstrap_mean_difference(treatment: np.ndarray, control: np.ndarray,
                              replicates: int = 10000, seed: int = 20261107,
                              alpha: float = 0.05) -> dict[str, float]:
    """Percentile bootstrap for the difference in means.

    Uses Poisson (exponential-weight style) resampling: each replicate draws
    integer weights ~ Poisson(1) per observation, which is asymptotically the
    same as resampling rows with replacement but avoids materialising a
    ``replicates x n`` index matrix for 21k-row arms.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    diffs = np.empty(replicates, dtype=float)
    for i in range(replicates):
        wt = rng.poisson(1.0, size=t.size)
        wc = rng.poisson(1.0, size=c.size)
        diffs[i] = (wt @ t) / wt.sum() - (wc @ c) / wc.sum()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # A percentile bootstrap cannot resolve a p-value below 1 / replicates, so the
    # add-one convention is used and the bound is stated in the method label.
    tail = min((diffs <= 0).sum(), (diffs >= 0).sum()) + 1
    p_value = min(1.0, 2 * tail / (replicates + 1))
    return {
        "n_treatment": t.size,
        "n_control": c.size,
        "mean_treatment": float(t.mean()),
        "mean_control": float(c.mean()),
        "effect": float(t.mean() - c.mean()),
        "relative_effect": float((t.mean() - c.mean()) / c.mean()) if c.mean() else float("nan"),
        "se": float(diffs.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_value),
        "test_statistic": None,
        "n_replicates": int(replicates),
        "p_value_floor": float(2 / (replicates + 1)),
    }


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Score (Wilson) interval for a single proportion - safe when events are rare."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def newcombe_interval(p1: float, n1: int, p2: float, n2: int,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe hybrid-score interval for a difference of two proportions.

    Included because the primary outcome converts at well under 1%: the Wald
    interval on a difference of small proportions under-covers, while this
    construction stays close to its nominal level. It is reported as a
    cross-check column, not as a second bite at significance.
    """
    l1, u1 = wilson_interval(p1 * n1, n1, alpha)
    l2, u2 = wilson_interval(p2 * n2, n2, alpha)
    diff = p1 - p2
    return float(diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), \
        float(diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def arm_summary(df: pd.DataFrame, arms: Sequence[str] = (CONTROL, MENS, WOMENS),
                confidence_level: float = 0.95) -> pd.DataFrame:
    """Per-arm descriptive table: sizes, rates with Wilson intervals, spend."""
    alpha = 1 - confidence_level
    rows = []
    for arm in arms:
        sub = df[df["segment"] == arm]
        n = len(sub)
        visits = int(sub["visit"].sum())
        conversions = int(sub["conversion"].sum())
        visit_lo, visit_hi = wilson_interval(visits, n, alpha)
        conv_lo, conv_hi = wilson_interval(conversions, n, alpha)
        spend_se = float(sub["spend"].std(ddof=1) / np.sqrt(n))
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        spend_mean = float(sub["spend"].mean())
        converters = sub.loc[sub["conversion"] == 1, "spend"]
        rows.append({
            "arm": arm,
            "n_customers": n,
            "share_of_sample": n / len(df),
            "visits": visits,
            "visit_rate": visits / n,
            "visit_rate_ci_low": visit_lo,
            "visit_rate_ci_high": visit_hi,
            "conversions": conversions,
            "conversion_rate": conversions / n,
            "conversion_rate_ci_low": conv_lo,
            "conversion_rate_ci_high": conv_hi,
            "conversion_rate_per_1k": 1000 * conversions / n,
            "visit_to_conversion_rate": conversions / visits if visits else float("nan"),
            "spend_total": float(sub["spend"].sum()),
            "spend_per_customer": spend_mean,
            "spend_ci_low": spend_mean - t_crit * spend_se,
            "spend_ci_high": spend_mean + t_crit * spend_se,
            "spend_per_converter": float(converters.mean()) if len(converters) else 0.0,
            "spend_sd": float(sub["spend"].std(ddof=1)),
            "spend_skewness": float(stats.skew(sub["spend"].to_numpy(), bias=False)),
            "share_zero_spend": float((sub["spend"] == 0).mean()),
        })
    return pd.DataFrame(rows)


def distribution_report(values: np.ndarray) -> dict[str, float]:
    """Skew and zero-mass evidence for an outcome, computed on the pooled sample."""
    v = np.asarray(values, dtype=float)
    nz = v[v > 0]
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "share_zero": float((v == 0).mean()),
        "share_positive": float((v > 0).mean()),
        "skewness": float(stats.skew(v, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(v, bias=False)),
        "p99": float(np.quantile(v, 0.99)),
        "max": float(v.max()),
        "positive_mean": float(nz.mean()) if nz.size else 0.0,
        "positive_skewness": float(stats.skew(nz, bias=False)) if nz.size > 2 else float("nan"),
    }


def spend_decomposition(df: pd.DataFrame, treatment_label: str, control_label: str = CONTROL,
                        alpha: float = 0.05) -> dict[str, float]:
    """Split the incremental spend into "more buyers" versus "bigger baskets".

    ``E[spend] = P(convert) x E[spend | convert]``, so the difference in mean
    spend decomposes into a conversion-driven and a basket-driven term. With a
    99%+ zero rate on the outcome, this is the difference between the email
    bringing in extra buyers and the email simply sending existing buyers to a
    more expensive product line.
    """
    t = df[df["segment"] == treatment_label]
    c = df[df["segment"] == control_label]
    conv_t, conv_c = float(t["conversion"].mean()), float(c["conversion"].mean())
    spend_t_pos = t.loc[t["conversion"] == 1, "spend"]
    spend_c_pos = c.loc[c["conversion"] == 1, "spend"]
    basket_t = float(spend_t_pos.mean()) if len(spend_t_pos) else 0.0
    basket_c = float(spend_c_pos.mean()) if len(spend_c_pos) else 0.0
    mean_total_t, mean_total_c = float(t["spend"].mean()), float(c["spend"].mean())
    total_effect = mean_total_t - mean_total_c
    conversion_term = (conv_t - conv_c) * basket_c
    basket_term = conv_t * (basket_t - basket_c)
    # Exact algebra: the two terms sum to the total by construction, so this
    # residual is a self-check on the decomposition rather than a third effect.
    identity_residual = total_effect - conversion_term - basket_term
    welch_basket = welch_mean_effect(spend_t_pos.to_numpy(), spend_c_pos.to_numpy(), alpha)
    return {
        "conversion_rate_treatment": conv_t,
        "conversion_rate_control": conv_c,
        "mean_spend_per_converter_treatment": basket_t,
        "mean_spend_per_converter_control": basket_c,
        "n_converters_treatment": int(len(spend_t_pos)),
        "n_converters_control": int(len(spend_c_pos)),
        "total_spend_effect": total_effect,
        "conversion_driven_part": conversion_term,
        "basket_driven_part": basket_term,
        "identity_residual": identity_residual,
        "conversion_driven_share": conversion_term / total_effect if total_effect else float("nan"),
        "basket_p_value": welch_basket["p_value"],
        "basket_ci_low": welch_basket["ci_low"],
        "basket_ci_high": welch_basket["ci_high"],
    }


def analyse_outcome(df: pd.DataFrame, outcome: str, comparison_name: str,
                    treatment_label: str, control_label: str, alpha: float,
                    confidence_level: float, bootstrap_replicates: int,
                    seed: int, bootstrap_for_spend: bool = True) -> list[EffectResult]:
    """All effect measures for one (outcome, comparison) pair."""
    t = df.loc[df["segment"] == treatment_label, outcome].to_numpy(dtype=float)
    c = df.loc[df["segment"] == control_label, outcome].to_numpy(dtype=float)
    alpha_local = 1 - confidence_level
    results: list[EffectResult] = []

    if outcome in PROPORTION_OUTCOMES:
        stats_map = two_proportion_effect(t, c, alpha_local)
        newcombe = newcombe_interval(stats_map["mean_treatment"], stats_map["n_treatment"],
                                     stats_map["mean_control"], stats_map["n_control"],
                                     alpha_local)
        stats_map["ci_low_newcombe"] = newcombe[0]
        stats_map["ci_high_newcombe"] = newcombe[1]
        results.append(_build(comparison_name, outcome, "risk_difference", treatment_label,
                              control_label, stats_map, "pp",
                              method="Wald z test, pooled variance",
                              confidence_level=confidence_level))

        # Relative lift is reported on the risk-ratio scale with a log-scale
        # interval: conversion here is a rare event (~0.6% in the control arm),
        # so a symmetric normal interval on a ratio would be the wrong shape.
        p1, p2 = stats_map["mean_treatment"], stats_map["mean_control"]
        rr = p1 / p2
        var_log = ((1 - p1) / (p1 * stats_map["n_treatment"])
                   + (1 - p2) / (p2 * stats_map["n_control"]))
        z_crit = stats.norm.ppf(1 - alpha_local / 2)
        rel_stats = dict(stats_map)
        rel_stats.pop("ci_low_newcombe", None)
        rel_stats.pop("ci_high_newcombe", None)
        rel_stats.update({
            "effect": rr - 1,
            "se": rr * float(np.sqrt(var_log)),
            "ci_low": float(np.exp(np.log(rr) - z_crit * np.sqrt(var_log))) - 1,
            "ci_high": float(np.exp(np.log(rr) + z_crit * np.sqrt(var_log))) - 1,
            "relative_effect": rr - 1,
            "p_value": float(2 * stats.norm.sf(abs(np.log(rr)) / np.sqrt(var_log))),
        })
        results.append(_build(comparison_name, outcome, "relative_lift", treatment_label,
                              control_label, rel_stats, "ratio",
                              method="log-scale delta-method interval on the risk ratio",
                              confidence_level=confidence_level, confirmatory=False))
    elif outcome in CONTINUOUS_OUTCOMES:
        welch = welch_mean_effect(t, c, alpha_local)
        results.append(_build(comparison_name, outcome, "mean_difference", treatment_label,
                              control_label, welch, "currency",
                              method="Welch t test (unequal variance)",
                              confidence_level=confidence_level, confirmatory=False))
        if bootstrap_for_spend:
            boot = bootstrap_mean_difference(t, c, replicates=bootstrap_replicates,
                                             seed=seed, alpha=alpha_local)
            results.append(_build(comparison_name, outcome, "mean_difference", treatment_label,
                                  control_label, boot, "currency",
                                  method=f"percentile bootstrap, {bootstrap_replicates} "
                                         f"replicates (p floored at 2/(reps+1))",
                                  n_replicates=bootstrap_replicates,
                                  confidence_level=confidence_level))
    else:
        raise ValueError(f"unknown outcome {outcome!r}")

    if not results:  # pragma: no cover - defensive
        raise RuntimeError("no effects produced")
    return results


def _build(comparison: str, outcome: str, measure: str, treatment_label: str,
           control_label: str, values: dict[str, float], effect_type: str, method: str,
           n_replicates: int | None = None, confidence_level: float = 0.95,
           confirmatory: bool = True) -> EffectResult:
    return EffectResult(
        comparison=comparison,
        outcome=outcome,
        measure=measure,
        treatment_label=treatment_label,
        control_label=control_label,
        n_treatment=int(values["n_treatment"]),
        n_control=int(values["n_control"]),
        mean_treatment=float(values["mean_treatment"]),
        mean_control=float(values["mean_control"]),
        effect=float(values["effect"]),
        effect_type=effect_type,
        relative_effect=float(values.get("relative_effect", float("nan"))),
        se=float(values["se"]) if values.get("se") is not None else None,
        ci_low=float(values["ci_low"]),
        ci_high=float(values["ci_high"]),
        ci_level=float(confidence_level),
        p_value=float(values["p_value"]),
        method=method,
        test_statistic=float(values["test_statistic"]) if values.get("test_statistic") is not None else None,
        n_replicates=n_replicates,
        ci_low_crosscheck=float(values["ci_low_newcombe"]) if "ci_low_newcombe" in values else None,
        ci_high_crosscheck=float(values["ci_high_newcombe"]) if "ci_high_newcombe" in values else None,
        crosscheck_method="Newcombe hybrid score" if "ci_low_newcombe" in values else None,
        confirmatory=confirmatory,
    )


def holm_adjust(p_values: Sequence[float], alpha: float = 0.05) -> AdjustmentResult:
    """Holm-Bonferroni step-down adjustment, ordered from smallest p upward."""
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    m = p.size
    ranked = p[order]
    adjusted_sorted = ranked * np.arange(m, 0, -1)
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    reject_raw = p < alpha
    reject_adjusted = adjusted < alpha
    return AdjustmentResult(pd.Series(p), pd.Series(adjusted),
                            pd.Series(reject_raw), pd.Series(reject_adjusted))


def familywise_bounds(effect: float, se: float | None, ci_low: float, ci_high: float,
                      confidence_level: float, n_comparisons: int,
                      asymmetric: bool = False) -> tuple[float, float, float]:
    """Bonferroni-conservative interval: the dual of the first Holm step.

    Returns ``(low, high, level)``. Two arms read against one control, so a
    family-wise 5% error rate means each interval has to be built at
    ``1 - alpha/2``. Intervals that were not built from a symmetric normal
    standard error (percentile bootstrap, log-scale risk ratio) are widened by
    the critical-value ratio rather than re-derived, which preserves their shape.
    """
    alpha = 1 - confidence_level
    alpha_family = alpha / max(n_comparisons, 1)
    level = 1 - alpha_family
    z_family = stats.norm.ppf(1 - alpha_family / 2)

    if not asymmetric:
        return effect - z_family * se, effect + z_family * se, level
    z_comparison = stats.norm.ppf(1 - alpha / 2)
    centre = (ci_low + ci_high) / 2
    half_width = (centre - ci_low) * (z_family / z_comparison)
    return centre - half_width, centre + half_width, level


def run_experiment_analysis(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Everything the incrementality dashboard needs, in one tidy table."""
    exp = config["experiment"]
    alpha = float(exp["alpha"])
    confidence_level = float(exp["confidence_level"])
    boot_cfg = exp["bootstrap"]
    rows: list[dict[str, object]] = []
    for comparison in exp["comparisons"]:
        for outcome in [exp["outcomes"]["primary"], *exp["outcomes"]["secondary"]]:
            effects = analyse_outcome(
                df,
                outcome=outcome,
                comparison_name=comparison["name"],
                treatment_label=comparison["treatment"],
                control_label=comparison["control"],
                alpha=alpha,
                confidence_level=confidence_level,
                bootstrap_replicates=int(boot_cfg["replicates"]),
                seed=int(boot_cfg["seed"]),
            )
            rows.extend(effect.as_dict() for effect in effects)
    results = pd.DataFrame(rows)

    # One family = one outcome tested against the shared control by both emails,
    # counted on the confirmatory estimate only (configs/experiment.yaml states
    # this same rule). Relative lift and the Welch cross-check describe the same
    # question again, so they carry no extra multiplicity and are not adjusted.
    results["n_comparisons_in_family"] = (
        results.groupby("outcome")["confirmatory"].transform("sum").astype(int)
    )
    results["multiplicity_role"] = np.where(
        results["confirmatory"], "confirmatory (Holm within outcome)",
        "supplementary view, not adjusted",
    )
    results["p_holm"] = np.nan
    # Nullable so a supplementary row reads as "not adjusted" rather than
    # "tested and not significant".
    results["reject_holm"] = pd.array([pd.NA] * len(results), dtype="boolean")
    for outcome, idx in results.groupby("outcome", sort=False).groups.items():
        family = results.loc[idx]
        family = family[family["confirmatory"]]
        adjustment = holm_adjust(family["p_value"], alpha)
        results.loc[family.index, "p_holm"] = adjustment.p_adjusted.to_numpy()
        results.loc[family.index, "reject_holm"] = pd.array(
            adjustment.reject_adjusted.to_numpy(), dtype="boolean"
        )

    bounds = [
        familywise_bounds(
            row["effect"], row["se"], row["ci_low"], row["ci_high"],
            confidence_level, int(row["n_comparisons_in_family"]),
            asymmetric=(str(row["method"]).startswith("percentile bootstrap")
                        or row["measure"] == "relative_lift"),
        )
        for row in results.to_dict("records")
    ]
    results["ci_low_fwer"] = [b[0] for b in bounds]
    results["ci_high_fwer"] = [b[1] for b in bounds]
    results["ci_level_fwer"] = [b[2] for b in bounds]

    return results.merge(
        incremental_totals(results, df),
        left_on=["comparison", "outcome", "measure"], right_index=True, how="left",
    )


def incremental_totals(results: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Scale each per-customer effect to the number of customers contacted."""
    rows: dict[tuple[str, str, str], dict[str, object]] = {}
    for _, row in results.iterrows():
        n_contacted = int((df["segment"] == row["treatment_label"]).sum())
        if row["measure"] == "relative_lift":
            total, unit = float("nan"), "not additive"
        elif row["outcome"] in PROPORTION_OUTCOMES:
            total, unit = float(row["effect"]) * n_contacted, "extra conversions"
        else:
            total, unit = float(row["effect"]) * n_contacted, "USD"
        rows[(row["comparison"], row["outcome"], row["measure"])] = {
            "n_contacted": n_contacted,
            "incremental_total": total,
            "incremental_unit": unit,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def subgroup_effects(df: pd.DataFrame, subgroup_columns: Sequence[str],
                     treatment_label: str, control_label: str = CONTROL,
                     outcome: str = "conversion", alpha: float = 0.05) -> pd.DataFrame:
    """Estimated effect inside each pre-campaign customer group.

    **Exploratory by design.** With this many subgroups, some will clear a 95%
    interval by chance alone, so the table reports interval widths rather than
    stars, and the heterogeneity test asks the sharper question: is the effect
    genuinely different between these groups, or is one group just noisier
    because the outcome is rare?
    """
    keep = df[df["segment"].isin([treatment_label, control_label])]
    rows = []
    for column in subgroup_columns:
        if column not in keep.columns:
            raise KeyError(f"subgroup column {column!r} is not in the frame")
        for level, group in keep.groupby(column, observed=True):
            treated = group.loc[group["segment"] == treatment_label, outcome].to_numpy(dtype=float)
            control = group.loc[group["segment"] == control_label, outcome].to_numpy(dtype=float)
            if len(treated) < 30 or len(control) < 30:
                continue
            effect = two_proportion_effect(treated, control, alpha) if outcome in PROPORTION_OUTCOMES \
                else welch_mean_effect(treated, control, alpha)
            rows.append({
                "subgroup_variable": column,
                "subgroup": str(level),
                "outcome": outcome,
                "n_treatment": int(len(treated)),
                "n_control": int(len(control)),
                "rate_treatment": effect["mean_treatment"],
                "rate_control": effect["mean_control"],
                "effect": effect["effect"],
                "ci_low": effect["ci_low"],
                "ci_high": effect["ci_high"],
                "p_value_unadjusted": effect["p_value"],
                "events_treatment": int(treated.sum()),
                "events_control": int(control.sum()),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    overall = two_proportion_effect(
        keep.loc[keep["segment"] == treatment_label, outcome].to_numpy(dtype=float),
        keep.loc[keep["segment"] == control_label, outcome].to_numpy(dtype=float), alpha,
    )
    table["effect_minus_overall"] = table["effect"] - overall["effect"]
    table["interval_contains_overall"] = ((table["ci_low"] <= overall["effect"])
                                          & (table["ci_high"] >= overall["effect"]))
    return table.sort_values("effect", ascending=False).reset_index(drop=True)


def heterogeneity_test(df: pd.DataFrame, column: str, treatment_label: str,
                       control_label: str = CONTROL,
                       outcome: str = "conversion") -> dict[str, float]:
    """Likelihood-ratio test for treatment-by-group interaction.

    The honest version of "this segment responds better": it is only a
    conclusion if the interaction block as a whole beats the additive model.
    """
    keep = df[df["segment"].isin([treatment_label, control_label])].copy()
    keep["treated"] = (keep["segment"] == treatment_label).astype(float)
    levels = pd.get_dummies(keep[column].astype(str), drop_first=True, dtype=float)
    base_design = sm.add_constant(pd.concat([keep[["treated"]], levels], axis=1))
    interaction = levels.mul(keep["treated"], axis=0)
    interaction.columns = [f"{c}:treated" for c in interaction.columns]
    full_design = sm.add_constant(pd.concat([keep[["treated"]], levels, interaction], axis=1))
    y = keep[outcome].astype(float)
    base = sm.GLM(y, base_design, family=sm.families.Binomial()).fit()
    full = sm.GLM(y, full_design, family=sm.families.Binomial()).fit()
    deviance_difference = float(base.deviance - full.deviance)
    degrees = float(full.df_model - base.df_model)
    result = {
        "chi2": deviance_difference,
        "p_value": float(stats.chi2.sf(deviance_difference, degrees)) if degrees > 0 else float("nan"),
        "degrees_of_freedom": max(degrees, 0.0),
        "n_levels_tested": max(degrees, 0.0),
        "outcome": outcome,
        "subgroup_variable": column,
    }
    if degrees <= 0 or deviance_difference < 0.0:
        # A negative gap means the fuller design was rank-deficient, not that the
        # interaction helped; report it as no evidence rather than an impossible statistic.
        result.update({"chi2": 0.0, "p_value": 1.0 if degrees > 0 else float("nan")})
    return result


def balance_table(df: pd.DataFrame, features: Sequence[str], categorical: Sequence[str],
                  control_label: str = CONTROL) -> pd.DataFrame:
    """Pre-campaign covariate balance for each email arm against the control.

    Standardised mean differences below 0.1 in absolute value are the usual
    empirical all-clear for randomisation. This is a design check, not a
    variable-selection step.
    """
    base = df[df["segment"] == control_label]
    rows = []
    for arm in [a for a in df["segment"].unique() if a != control_label]:
        treated = df[df["segment"] == arm]
        for feature in features:
            if feature in categorical:
                levels = sorted(set(base[feature].astype(str)) | set(treated[feature].astype(str)))
                for level in levels:
                    p_t = float((treated[feature].astype(str) == level).mean())
                    p_c = float((base[feature].astype(str) == level).mean())
                    pooled = (p_t + p_c) / 2
                    sd = np.sqrt(pooled * (1 - pooled)) if 0 < pooled < 1 else np.nan
                    rows.append(_balance_row(arm, feature, level, p_t, p_c,
                                             (p_t - p_c) / sd if sd and sd > 0 else float("nan")))
            else:
                mean_t = float(treated[feature].mean())
                mean_c = float(base[feature].mean())
                sd = np.sqrt((treated[feature].var(ddof=1) + base[feature].var(ddof=1)) / 2)
                rows.append(_balance_row(arm, feature, None, mean_t, mean_c,
                                         (mean_t - mean_c) / sd if sd > 0 else float("nan")))
    table = pd.DataFrame(rows)
    table["within_0_1"] = table["std_mean_difference"].abs() < 0.1
    return table


def _balance_row(arm: str, feature: str, level: str | None, treated: float, control: float,
                 smd: float) -> dict[str, object]:
    return {
        "arm": arm,
        "feature": feature,
        "level": level,
        "treated_mean": treated,
        "control_mean": control,
        "difference": treated - control,
        "std_mean_difference": smd,
    }


def regression_adjusted_effect(df: pd.DataFrame, outcome: str, treatment_label: str,
                               covariates: Sequence[str], control_label: str = CONTROL,
                               model: str = "ols") -> dict[str, float | str]:
    """Covariate-adjusted treatment estimate on a stated scale.

    ``model="ols"`` is a linear-probability / mean-shift fit with HC1 standard
    errors, so its coefficient is on the same scale as the unadjusted difference
    and the two can be compared directly. ``model="logit"`` reports a log odds
    ratio for the binary outcome, which is a different question and is labelled
    as such rather than blended into the headline table.

    Randomisation already makes the unadjusted contrast valid. Adjustment is a
    precision and robustness check: if the adjusted and unadjusted estimates
    disagreed materially, that would be evidence the assignment was not clean.
    """
    keep = df[df["segment"].isin([treatment_label, control_label])].copy()
    keep["treated"] = (keep["segment"] == treatment_label).astype(float)
    design = pd.get_dummies(keep[list(covariates)], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([keep[["treated"]], design], axis=1))
    y = keep[outcome].astype(float)
    if model == "logit":
        fitted = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        scale = "log odds ratio"
    elif model == "ols":
        fitted = sm.OLS(y, X).fit(cov_type="HC1")
        scale = "risk difference" if outcome in PROPORTION_OUTCOMES else "USD per customer"
    else:
        raise ValueError(f"unknown model {model!r}")
    ci = fitted.conf_int().loc["treated"]
    effect = float(fitted.params["treated"])
    unadjusted = float(
        keep.loc[keep["treated"] == 1, outcome].mean()
        - keep.loc[keep["treated"] == 0, outcome].mean()
    )
    return {
        "outcome": outcome,
        "treatment": treatment_label,
        "control": control_label,
        "model": model,
        "effect_scale": scale,
        "adjusted_effect": effect,
        "adjusted_odds_ratio": float(np.exp(effect)) if model == "logit" else float("nan"),
        "adjusted_se": float(fitted.bse["treated"]),
        "adjusted_ci_low": float(ci[0]),
        "adjusted_ci_high": float(ci[1]),
        "adjusted_p_value": float(fitted.pvalues["treated"]),
        "unadjusted_effect_same_scale": unadjusted,
        "adjusted_minus_unadjusted": effect - unadjusted,
        "relative_change_vs_unadjusted": (effect - unadjusted) / unadjusted if unadjusted else float("nan"),
        "n_obs": int(len(keep)),
        "n_covariates": int(X.shape[1] - 2),
    }
