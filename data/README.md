# Data

Every workbook used by the notebooks is committed here. Nothing is downloaded at
runtime and no model is fetched.

```
data/
├── indic/     ← IndicMT Eval, five Indic languages (the primary corpus)
└── latin/     ← WMT24 ENG-DEU / ENG-SPA (Latin-script controls)
```

See each subfolder's README for dataset credit, citation and instructions for
obtaining the raw annotations.

---

## Integrity

`scripts/check_data.py` verifies presence and digests. `make data` runs it in
strict mode and fails the build on any mismatch.

```bash
python3 scripts/check_data.py --strict
```

| SHA-256 | Bytes | File |
|---|---|---|
| `fa4a7a7dc64583235159db3c5c292a76b50f61dc0b8f0fac4e30b581ceb2fbc0` | 9,188,987 | `indic/indic_parity_xlmr.xlsx` |
| `d6a183f45142a2e82f2fb35d9e35fc890fe0f90986f87ea4871facf5b1eef110` | 19,762,733 | `indic/indic_parity_multi_tokenizer.xlsx` |
| `bcf5f584e7fdf19139a6c1700c39484b8aa44bb007ae5e37b0c92a7dba914bcc` | 10,740,018 | `latin/wmt24_ende_enes_metrics.xlsx` |

---

## What the workbooks contain

The Indic workbooks hold both **raw corpus text** (source, reference,
translation, their romanised forms, and per-token strings) and **derived numeric
quantities** (COMET, BLEURT, BERTScore, BLEU, chrF, TER, token counts, TP, IP)
alongside the MQM error-type and severity labels.

The analysis in this repository reads **only the derived numeric columns and the
MQM labels**. The text columns are retained so that the workbooks remain a
faithful, inspectable record of the corpus each score was computed from, and so
that a reader can trace any individual score back to the segment that produced
it.

The text originates from IndicMT Eval and WMT24 and is redistributed under those
projects' own licences, cited in the subfolder READMEs. It is not the authors'
own text and carries no annotator identities.

---

## Provenance

| Workbook | Produced by | Contents |
|---|---|---|
| `indic/indic_parity_xlmr.xlsx` | Upstream scoring pipeline: COMET / BLEURT / BERTScore / sacreBLEU scoring, IndicXlit romanisation, XLM-R tokenisation, BLOOM-560M NLL for IP | 5 sheets × 1,400 rows × 52–53 columns |
| `indic/indic_parity_multi_tokenizer.xlsx` | The same pipeline, extended with mBERT, GPT-2 and ByT5 tokenisation | 5 sheets × 1,400 rows × 94–95 columns |
| `latin/wmt24_ende_enes_metrics.xlsx` | WMT24 data via `mt-metrics-eval`, scored with the same metric stack | 2 sheets (German, Spanish), 6,006 and 4,651 rows |

The multi-tokenizer workbook is a **strict superset** of the primary one: notebook
08 asserts that all seven shared numeric columns are bit-identical across all five
sheets before using it. Both are committed because the primary workbook is
half the size and is what notebooks 01–07 and 09 read.

### Known data-hygiene quirks

These are present in the source workbooks and are handled explicitly rather than
cleaned away, so that the committed data stays byte-identical to what produced
the reported numbers:

- One Malayalam row has `Human_scores` = `` `19 `` (a stray backtick). Coercing
  to numeric turns it into `NaN`, which is what removes it from the 6,995
  working set. Notebook 01 prints it.
- Four further rows (1 Tamil, 3 Marathi) have a blank `Human_scores`.
- The Malayalam sheet has a `Chrf` column where the others have `chrF`, plus an
  `Unnamed: 18` column. Neither is read by any analysis.
- The Tamil sheet's ID column header carries stray appended text. The column is
  unused (`ID (ignore_column)`).
- Hindi, Malayalam and Tamil carry an extra `comet_qe_romanized` column that
  Gujarati and Marathi lack. It is unused.

---

## Reproducing the workbooks from raw corpora

The workbooks are the *output* of the upstream scoring pipeline, not its input.
Regenerating them from scratch requires the raw corpora and the neural metric
stack, neither of which is vendored here:

```bash
# Indic: obtain the MQM annotations
#   https://github.com/AI4Bharat/IndicMT-Eval/tree/master/Dataset

# Latin: fetch WMT24 via mt-metrics-eval (~2 GB into ~/.mt-metrics-eval/)
pip install mt-metrics-eval
python3 -c "from mt_metrics_eval import data; data.Download()"
```

Then run COMET, BLEURT, BERTScore and sacreBLEU over both the native and the
IndicXlit-romanised targets, and compute XLM-R token counts and BLOOM-560M NLL
for TP and IP. The resulting columns are what these workbooks hold.

Because that path depends on GPU-scored neural metrics whose outputs vary with
library and checkpoint versions, it will not reproduce the committed workbooks
bit-for-bit. The digests above are the authoritative reference for the numbers
reported in this repository.
