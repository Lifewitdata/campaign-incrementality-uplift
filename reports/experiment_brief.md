# Experiment brief - Hillstrom e-mail test

**Design.** 64,000 customers randomised to Mens E-Mail (21,307), Womens E-Mail (21,387) and
No E-Mail (21,306); outcomes observed over two weeks. Sample-ratio check passes; pre-campaign
covariates are balanced (`outputs/tables/baseline_balance.csv`).

**Analysis outcomes specified in the project brief.** Conversion is primary; visit and spend are secondary. Both
campaigns are compared with the *same* control, so each outcome family holds two tests and is
Holm-adjusted; family-wise intervals are Bonferroni-dual (97.5% per test).

| Comparison | Outcome | Effect | 97.5% family-wise CI | Holm p |
|---|---|---|---|---|
| Mens vs control | Conversion | +0.681 pp | [+0.474, +0.887] | 3.0e-13 |
| Mens vs control | Visit | +7.66 pp | [+6.90, +8.42] | <1e-100 |
| Mens vs control | Spend / customer | +$0.770 | [+$0.443, +$1.093] | 4.0e-4 |
| Womens vs control | Conversion | +0.311 pp | [+0.127, +0.496] | 1.6e-4 |
| Womens vs control | Visit | +4.52 pp | [+3.80, +5.25] | <1e-40 |
| Womens vs control | Spend / customer | +$0.424 | [+$0.135, +$0.720] | 1.0e-3 |

**Spend is extremely skewed** (99% of customers spend $0, skewness ~20, max $499), so spend
intervals use a percentile bootstrap rather than a normal approximation. Decomposing
`E[spend] = P(convert) x E[spend | convert]` shows nearly all incremental spend comes from
additional buyers, not larger baskets (`spend_decomposition.csv`).

**Who responds (exploratory).** Twelve treatment x subgroup likelihood-ratio tests (six
pre-campaign variables x two campaigns) were Holm-adjusted as one family. Exactly one survives:
the Womens e-mail works better on **new customers** (`newbie`, Holm p = 0.015). No other
subgroup difference is distinguishable from noise. `mens`/`womens` are pre-campaign purchase
flags, not gender labels.

**Limits.** Effects are average group effects. No customer's individual treatment effect is
observed. Randomisation already identifies these effects, so no matching or
difference-in-differences was used.
