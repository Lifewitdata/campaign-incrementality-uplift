"""Sample size and minimum detectable effect for a future campaign.

Two things make this more than a calculator lookup:

* the design has **two email arms reading against one control**, so the
  per-comparison alpha has to be split if the plan is to keep a family-wise
  error rate; and
* the primary outcome here is rare (control conversion is well under 1%), so
  the normal-approximation answer is checked against a Monte-Carlo run of the
  same experiment rather than trusted on its own.

Formulas are written out rather than called from a library so that the
arithmetic can be read and unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


def normal_quantile(probability: float) -> float:
    return float(stats.norm.ppf(probability))


def sample_size_two_proportions(p_control: float, p_treatment: float, alpha: float = 0.05,
                                power: float = 0.80, allocation_ratio: float = 1.0) -> dict[str, float]:
    """Per-arm n for a two-sided test of two proportions (pooled-variance form).

    ``allocation_ratio`` is n_treatment / n_control, so an unequal split - a
    small holdout, say - can be planned without rewriting the formula.
    """
    if not 0 < p_control < 1 or not 0 < p_treatment < 1:
        raise ValueError("both proportions must lie strictly between 0 and 1")
    if p_control == p_treatment:
        raise ValueError("no effect to detect: p_treatment must differ from p_control")
    z_alpha = normal_quantile(1 - alpha / 2)
    z_beta = normal_quantile(power)
    p_bar = (p_control + allocation_ratio * p_treatment) / (1 + allocation_ratio)
    effect = abs(p_treatment - p_control)
    common = (z_alpha * np.sqrt((1 + 1 / allocation_ratio) * p_bar * (1 - p_bar))
              + z_beta * np.sqrt(p_control * (1 - p_control)
                                 + p_treatment * (1 - p_treatment) / allocation_ratio))
    n_control = float(np.ceil(common ** 2 / effect ** 2))
    n_treatment = float(np.ceil(allocation_ratio * n_control))
    return {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "n_total": n_control + n_treatment,
        "effect_absolute": effect,
        "effect_relative": effect / p_control,
        "alpha": alpha,
        "power": power,
        "allocation_ratio": allocation_ratio,
    }


def achieved_power_two_proportions(p_control: float, p_treatment: float, n_per_arm: int,
                                   alpha: float = 0.05) -> float:
    """Normal-approximation power of a two-sided pooled z test at a fixed n."""
    effect = abs(p_treatment - p_control)
    z_alpha = normal_quantile(1 - alpha / 2)
    se_alt = np.sqrt((p_control * (1 - p_control) + p_treatment * (1 - p_treatment)) / n_per_arm)
    return float(stats.norm.sf(z_alpha - effect / se_alt) + stats.norm.cdf(-z_alpha - effect / se_alt))


def mde_two_proportions(p_control: float, n_per_arm: int, alpha: float = 0.05,
                        power: float = 0.80, tol: float = 1e-7,
                        max_effect: float = 0.5) -> float:
    """Smallest absolute conversion lift detectable at a given per-arm n.

    Solved by bisection because the pooled-variance term makes the equation
    implicit in the effect size.
    """
    lo, hi = tol, min(max_effect, 1 - p_control - tol)
    if achieved_power_two_proportions(p_control, p_control + hi, n_per_arm, alpha) < power:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        candidate = achieved_power_two_proportions(p_control, p_control + mid, n_per_arm, alpha)
        if candidate > power:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return float((lo + hi) / 2)


def sample_size_mean(delta: float, sd: float, alpha: float = 0.05, power: float = 0.80,
                     k_timesteps: int = 8) -> dict[str, float]:
    """Per-arm n for a two-sample t test on a continuous outcome.

    Starts from the normal approximation and iterates on the t critical value
    so the answer is not optimistic for the modest sample sizes a small holdout
    would give.
    """
    if sd <= 0:
        raise ValueError("standard deviation must be positive")
    n = 2 * (normal_quantile(1 - alpha / 2) + normal_quantile(power)) ** 2 * sd ** 2 / delta ** 2
    for _ in range(k_timesteps):
        n = max(4, int(np.ceil(n)))
        df = 2 * n - 2
        t_alpha = float(stats.t.ppf(1 - alpha / 2, df))
        n_next = 2 * (t_alpha + normal_quantile(power)) ** 2 * sd ** 2 / delta ** 2
        if abs(n_next - n) < 1:
            break
        n = n_next
    n = float(np.ceil(n))
    return {
        "n_control": n,
        "n_treatment": n,
        "n_total": 2 * n,
        "effect_absolute": abs(delta),
        "alpha": alpha,
        "power": power,
        "sd": sd,
    }


def mde_mean(n_per_arm: int, sd: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest detectable difference in a mean at a given per-arm n."""
    df = max(2 * n_per_arm - 2, 1)
    t_alpha = float(stats.t.ppf(1 - alpha / 2, df))
    return float((t_alpha + normal_quantile(power)) * sd * np.sqrt(2 / n_per_arm))


MONTE_CARLO_SIMULATIONS = 4000


def monte_carlo_power(p_control: float, p_treatment: float, n_per_arm: int,
                      alpha: float = 0.05, simulations: int = MONTE_CARLO_SIMULATIONS,
                      seed: int = 20261107) -> dict[str, float]:
    """Empirical power by simulating the experiment itself.

    Draws binomial outcomes per arm and runs the same pooled two-proportion z
    test the analysis uses. This is the check on the closed-form answer for a
    rare outcome, where the normal approximation is doing real work.
    """
    rng = np.random.default_rng(seed)
    control = rng.binomial(n_per_arm, p_control, size=simulations)
    treated = rng.binomial(n_per_arm, p_treatment, size=simulations)
    p_hat_c, p_hat_t = control / n_per_arm, treated / n_per_arm
    pooled = (control + treated) / (2 * n_per_arm)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n_per_arm))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(se > 0, (p_hat_t - p_hat_c) / se, 0.0)
    reject = np.abs(z) > normal_quantile(1 - alpha / 2)
    return {
        "simulated_power": float(reject.mean()),
        "simulations": float(simulations),
        "n_per_arm": float(n_per_arm),
        "standard_error_of_power": float(np.sqrt(reject.mean() * (1 - reject.mean()) / simulations)),
    }


def alpha_for_family(alpha: float, n_comparisons: int) -> float:
    """Bonferroni-split per-comparison alpha for a shared-control design."""
    return alpha / max(n_comparisons, 1)


@dataclass
class PlanningAssumptions:
    """Everything a power answer depends on, stated in one place."""

    baseline_conversion: float
    baseline_visit: float
    spend_sd: float
    spend_mean: float
    alpha: float = 0.05
    power: float = 0.80
    n_comparisons: int = 2
    arms_available: Sequence[str] = ("mens", "womens")

    def describe(self) -> str:
        return (
            f"baseline conversion {self.baseline_conversion:.4%}, baseline visit "
            f"{self.baseline_visit:.4%}, spend SD ${self.spend_sd:,.2f} "
            f"(mean ${self.spend_mean:,.2f}); alpha={self.alpha}, power={self.power:.0%}, "
            f"{self.n_comparisons} comparisons against a shared control"
        )


def assumptions_from_control(df, control_label: str = "No E-Mail", alpha: float = 0.05,
                             power: float = 0.80, n_comparisons: int = 2) -> PlanningAssumptions:
    """Take the baselines from the control arm that was actually observed.

    Planning on the control arm - not the pooled sample - is the point: the
    no-email customers are the counterfactual a future campaign is compared
    against, and using a rate contaminated by the emails would understate how
    many sends are needed to detect the same lift again.
    """
    control = df[df["segment"] == control_label]
    return PlanningAssumptions(
        baseline_conversion=float(control["conversion"].mean()),
        baseline_visit=float(control["visit"].mean()),
        spend_sd=float(control["spend"].std(ddof=1)),
        spend_mean=float(control["spend"].mean()),
        alpha=alpha,
        power=power,
        n_comparisons=n_comparisons,
    )


def build_plan(assumptions: PlanningAssumptions,
               relative_lifts: Sequence[float] = (0.10, 0.20, 0.30, 0.50, 0.75, 1.00),
               absolute_lifts_usd: Sequence[float] = (0.10, 0.20, 0.30, 0.50, 0.75)) -> list[dict[str, object]]:
    """Sample sizes for a future campaign, with and without multiplicity control."""
    alpha_family = alpha_for_family(assumptions.alpha, assumptions.n_comparisons)
    rows: list[dict[str, object]] = []
    for lift in relative_lifts:
        p_treatment = assumptions.baseline_conversion * (1 + lift)
        for label, alpha in (("per comparison", assumptions.alpha),
                             ("family-wise (Bonferroni)", alpha_family)):
            plan = sample_size_two_proportions(assumptions.baseline_conversion, p_treatment,
                                               alpha=alpha, power=assumptions.power)
            check = monte_carlo_power(assumptions.baseline_conversion, p_treatment,
                                      int(plan["n_control"]), alpha=alpha)
            rows.append({
                "outcome": "conversion",
                "scenario": f"{lift:.0%} relative lift",
                "assumed_treatment_rate": p_treatment,
                "alpha_used": alpha,
                "alpha_convention": label,
                "n_per_arm": plan["n_control"],
                "n_total_two_arm": plan["n_total"],
                "n_total_three_arm": plan["n_control"] + plan["n_treatment"] * len(assumptions.arms_available),
                "closed_form_power": assumptions.power,
                "monte_carlo_power": check["simulated_power"],
            })
    for lift in absolute_lifts_usd:
        for label, alpha in (("per comparison", assumptions.alpha),
                             ("family-wise (Bonferroni)", alpha_family)):
            plan = sample_size_mean(lift, assumptions.spend_sd, alpha=alpha,
                                    power=assumptions.power)
            rows.append({
                "outcome": "spend",
                "scenario": f"${lift:.2f} per customer",
                "assumed_treatment_rate": assumptions.spend_mean + lift,
                "alpha_used": alpha,
                "alpha_convention": label,
                "n_per_arm": plan["n_control"],
                "n_total_two_arm": plan["n_total"],
                "n_total_three_arm": plan["n_control"] + plan["n_treatment"] * len(assumptions.arms_available),
                "closed_form_power": assumptions.power,
                "monte_carlo_power": float("nan"),
            })
    return rows


def realised_design_note(n_per_arm: int, assumptions: PlanningAssumptions,
                         observed_treatment_rate: float) -> dict[str, float]:
    """What the 2008 design could actually detect - the honest post-hoc framing.

    ``observed_treatment_rate`` is passed in rather than assumed, so the power
    number is always attached to a rate somebody can trace back to the data.
    """
    mde_conversion = mde_two_proportions(assumptions.baseline_conversion, n_per_arm,
                                         alpha=alpha_for_family(assumptions.alpha,
                                                                 assumptions.n_comparisons),
                                         power=assumptions.power)
    return {
        "n_per_arm": n_per_arm,
        "observed_treatment_rate": observed_treatment_rate,
        "mde_conversion_absolute": mde_conversion,
        "mde_conversion_relative": mde_conversion / assumptions.baseline_conversion,
        "mde_visit_absolute": mde_two_proportions(assumptions.baseline_visit, n_per_arm,
                                                 alpha=assumptions.alpha, power=assumptions.power),
        "mde_spend_usd": mde_mean(n_per_arm, assumptions.spend_sd, alpha=assumptions.alpha,
                                  power=assumptions.power),
        "achieved_power_at_observed_conversion_lift": achieved_power_two_proportions(
            assumptions.baseline_conversion, observed_treatment_rate, n_per_arm,
            alpha=assumptions.alpha,
        ),
    }
