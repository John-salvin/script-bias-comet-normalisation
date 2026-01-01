# Latin-Script Data (WMT24)

Data for the two Latin-script language pairs used as cross-linguistic controls:
**German (ENG-DEU)** and **Spanish (ENG-SPA)**, from the WMT24 General MT Shared
Task.

---

## Dataset credit and attribution

Derived from the **WMT24 General MT Shared Task**, accessed via Google Research's
publicly available [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval)
toolkit.

| Property | Details |
|---|---|
| **Source repository** | [github.com/google-research/mt-metrics-eval](https://github.com/google-research/mt-metrics-eval) |
| **Paper** | Kocmi et al. (2024). Findings of the WMT24 General MT Shared Task. *Proceedings of WMT 2024*, pp. 1–46. ACL. |
| **ACL Anthology** | [aclanthology.org/2024.wmt-1.1](https://aclanthology.org/2024.wmt-1.1/) |

### Statistics

| Language pair | Rows | MT systems | Sentences | Annotations |
|---|---|---|---|---|
| English → German (ENG-DEU) | 6,006 | 18 | 339 | MQM |
| English → Spanish (ENG-ESP) | 4,651 | 14 | 339 | MQM + ESA |

---

## Files

| File | Description |
|---|---|
| `wmt24_ende_enes_metrics.xlsx` | Two sheets, `German` and `Spanish`. COMET, COMET-QE, BLEURT, BERTScore, BLEU, chrF, TER, XLM-R token counts, TP and IP for source, target and reference, plus MQM rater, category, severity and score. |

The analysis reads one column: `target_xlmr_IP`, from which IPI = |mean IP − 1.0|
is computed. `target_xlmr_TP` is additionally used for the descriptive SBI
figures in notebook 03.

---

## Why these two controls

They separate two explanations for the Indic results.

- **ENG-SPA** sits in the Parity zone (IPI = 0.009), which anchors the scale and
  shows what an encoder handling a language well actually looks like.
- **ENG-DEU** uses the Latin alphabet, yet lands in the Burden zone
  (IPI = 0.465) — inside the Indic native-script range of 0.364–0.562. If the
  Indic burden were caused by non-Latin script surface, German would not be
  there. It is caused by how the encoder's vocabulary is allocated.

Both values are recomputed from the committed workbook in notebook 03 and
plotted as the control markers in Figure 1.

---

## How to reproduce the raw data

The raw WMT24 human evaluation data is not committed here. To fetch it:

```bash
pip install mt-metrics-eval sacrebleu pandas openpyxl
python3 -c "from mt_metrics_eval import data; data.Download()"
```

The download step fetches roughly 2 GB into `~/.mt-metrics-eval/`. Scoring it
with the same metric stack described in [`../README.md`](../README.md) produces
the columns in this workbook.
