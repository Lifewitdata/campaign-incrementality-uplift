# Data

## Source

| | |
|---|---|
| Dataset | Hillstrom MineThatData E-Mail Analytics And Data Mining Challenge (2008) |
| Author | Kevin Hillstrom, MineThatData |
| Description | A real randomised field experiment run by a women's apparel retailer. 64,000 customers were randomly assigned to one of three cells: a women's-merchandise email, a men's-merchandise email, or no email at all. Site visits, conversions and spending were then observed for the following two weeks. |
| Original file | `Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv` |
| URL | <http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv> |
| Rows x columns | 64,000 x 12 (source), 64,000 x 16 (clean table) |
| Retrieved | Pinned by SHA-256, see below |

**Attribution and licensing.** The CSV was published by Kevin Hillstrom for a public
data-mining challenge. The file itself carries no machine-readable licence statement,
so it is used here with visible credit for portfolio and educational purposes. Confirm
the terms with the source before any commercial use. No customer-identifying data is
present in the file.

**SHA-256 of the exact bytes used for every number in this repository:**

```
0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece  hillstrom.csv
3,964,977 bytes
```

`data/raw/` is git-ignored (the file is re-downloadable and byte-identical). Run
`python scripts/ingest_data.py` to fetch it and verify the checksum; a mismatch is a
hard error, not a silent re-download, because the published results are tied to this
exact file.

## What this data is *not*

- `mens` and `womens` are **pre-campaign purchase flags**, not customer gender. The
  three campaign names describe which product line the email promoted.
- There are no loyalty tiers, membership details, unsubscribe or delivery-metadata
  fields. Nothing in this repository invents them.
- There is no customer ID. `record_id` in the clean table is the row position in the
  published file - a load/join key, not a person.
- `visit`, `conversion` and `spend` are **post-campaign outcomes**. They are labels for
  the experiment, and forbidden as model features.

## Clean table

`scripts/ingest_data.py` writes `data/processed/campaign_customers.csv` with the source
columns plus four derived ones: `arm_code`, `history_band`, `campaign_weeks`,
`record_id`. The source spelling `Surburban` is normalised to `Suburban`; the
normalisation is reproducible and asserted by a data-quality gate.

## Data dictionary

### Treatment assignment

| Column | Type | Values | Definition |
|---|---|---|---|
| `segment` | string | `Womens E-Mail` (21,387), `Mens E-Mail` (21,307), `No E-Mail` (21,306) | Randomised arm. `No E-Mail` is the control: those customers received nothing. |
| `arm_code` *(derived)* | string | `womens`, `mens`, `control` | Machine-readable arm label used by the code and the warehouse. |
| `campaign_weeks` *(derived)* | int | `2` (constant) | Outcome observation window after the send, in weeks. Recorded so the window is explicit rather than assumed. |

### Pre-campaign customer features (safe to model)

| Column | Type | Range / levels | Definition |
|---|---|---|---|
| `recency` | int | 1-12 | Months since the customer's last purchase before sampling. |
| `history` | float | $29.99 - $3,345.93 | Dollars spent with the retailer in the 12 months before the campaign. |
| `history_segment` | string | `1) $0 - $100` (22,970), `2) $100 - $200` (14,254), `3) $200 - $350` (12,289), `4) $350 - $500` (6,409), `5) $500 - $750` (4,911), `6) $750 - $1,000` (1,859), `7) $1,000 +` (1,308) | Ordered band of `history`. Verified to be an exact function of `history` on all 64,000 rows (`outputs/tables/data_quality_checks.csv`, gate `consistency.history_band_matches_segment`), so it carries no information beyond `history` other than the discretisation. |
| `history_band` *(derived)* | string | same seven levels | Band re-derived from `history` by the pipeline; used as the integrity check on `history_segment`. |
| `mens` | int | 0 / 1 | Purchased men's merchandise in the pre-campaign year. Product affinity, not gender. |
| `womens` | int | 0 / 1 | Purchased women's merchandise in the pre-campaign year. Product affinity, not gender. |
| `zip_code` | string | `Surburban` (28,776), `Urban` (25,661), `Rural` (9,563) | Retail trade area of the mailing address. `Surburban` is a typo in the published file; normalised to `Suburban` downstream. |
| `newbie` | int | 0 / 1 | First-time buyer within the 12 months before sampling. |
| `channel` | string | `Web` (28,217), `Phone` (28,021), `Multichannel` (7,762) | Predominant pre-campaign purchase channel. |

### Post-campaign outcomes (never model inputs)

| Column | Type | Range | Definition |
|---|---|---|---|
| `visit` | int | 0 / 1 | Visited the site during the two-week window. |
| `conversion` | int | 0 / 1 | Made a purchase during the two-week window. **Primary outcome**, fixed before analysis. |
| `spend` | float | $0.00 - $499.00 | Dollars spent during the two-week window; 0 for non-buyers. Secondary outcome. |

Observed logical constraints, both verified on all 64,000 rows with zero violations:
`spend > 0` implies `conversion = 1`, and `conversion = 1` implies `visit = 1`. There
are no nulls, no blank labels and no duplicate `record_id` values.

## Outcomes chosen before analysis

Primary outcome: `conversion`. Secondary outcomes: `visit` and `spend`. This ordering
was fixed in `configs/experiment.yaml` before any treatment comparison was computed, so
the primary readout is not the outcome that happened to look best.

## Column order in the source file

```
recency, history_segment, history, mens, womens, zip_code, newbie, channel,
segment, visit, conversion, spend
```

`scripts/../src/campaign_ml/validation.py` fails the run if this order changes, because
the file has no header-independent contract and a silently reordered load would corrupt
every downstream estimate.
