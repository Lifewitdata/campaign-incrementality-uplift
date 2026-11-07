# Business recommendations

1. **Both campaigns work; Mens works more.** Mens e-mail lifts conversion by +0.68 pp and spend
   by +$0.77 per customer sent. Womens e-mail lifts them by +0.31 pp and +$0.42. Both results
   survive multiplicity control. If only one campaign can run, run Mens.
2. **Don't build a targeting program on these models yet.** Uplift ranking did not beat a
   propensity list or a random list on the holdout. Emailing the whole list captures the
   measured effect without model risk.
3. **One segment hypothesis is worth re-testing:** the Womens e-mail does better on new
   customers (+0.53 pp vs +0.09 pp for existing customers). It is the only interaction out of
   twelve that survives Holm adjustment. Confirm it in a new randomised test before acting on it.
4. **Economics are scenarios only.** The dataset has no costs or margins. At the assumed 45%
   margin and $0.005 per send, e-mailing the full Mens holdout list returns about $1,490 net
   (bootstrap range roughly $310-$2,740) (`outputs/tables/economics_scenarios.csv`). That is
   **not realised ROI**.
5. **Next test design:** detecting a 10%-relative conversion lift needs far more customers than
   this test had. See `reports/power_analysis.md` (this design's MDE is 0.25 pp under
   Bonferroni).
