# Indic Data

Data for the five Indic language pairs used in the experiments:
**Gujarati, Hindi, Malayalam, Marathi, Tamil** (English → Indic, MQM-annotated).

---

## Dataset credit and attribution

This project uses the **IndicMT Eval** MQM dataset created by the
[AI4Bharat](https://github.com/AI4Bharat) team.

| Property | Details |
|---|---|
| **Repository** | [github.com/AI4Bharat/IndicMT-Eval](https://github.com/AI4Bharat/IndicMT-Eval) |
| **Paper** | IndicMT Eval: A Dataset to Meta-Evaluate Machine Translation Metrics for Indian Languages (ACL 2023) |
| **Languages** | Gujarati, Hindi, Malayalam, Marathi, Tamil |
| **MT systems** | 7 |
| **Error types** | 11 MQM error categories, 3 severity levels |
| **Annotations** | Human MQM annotations by language experts |
| **Segments** | 1,400 per language, 7,000 total |

Please cite both papers when using this dataset:

```bibtex
@article{DBLP:journals/corr/abs-2212-10180,
  author    = {Ananya B. Sai and Tanay Dixit and Vignesh Nagarajan and
               Anoop Kunchukuttan and Pratyush Kumar and
               Mitesh M. Khapra and Raj Dabre},
  title     = {IndicMT Eval: {A} Dataset to Meta-Evaluate Machine Translation
               metrics for Indian Languages},
  journal   = {CoRR},
  volume    = {abs/2212.10180},
  year      = {2022},
  url       = {https://arxiv.org/abs/2212.10180}
}

@article{singh2024good,
  title   = {How Good is Zero-Shot MT Evaluation for Low Resource Indian Languages?},
  author  = {Singh, Anushka and Sai, Ananya B and Dabre, Raj and Puduppully, Ratish
             and Kunchukuttan, Anoop and Khapra, Mitesh M},
  journal = {arXiv preprint arXiv:2406.03893},
  year    = {2024}
}
```

Romanisation uses **IndicXlit** (Madhani et al., 2023, *Aksharantar*, EMNLP
Findings). Information Parity follows Tsvetkov & Kipnis (2024); Tokenization
Parity follows Petrov et al. (2023).

---

## Files

| File | Description |
|---|---|
| `indic_parity_xlmr.xlsx` | Primary workbook. 5 sheets (one per language) × 1,400 rows. COMET, BLEURT, BERTScore, BLEU, chrF, TER and COMET-QE for both native and romanised conditions; XLM-R token counts, TP and IP for source, reference and translation in both conditions; MQM error types, severities and human scores. |
| `indic_parity_multi_tokenizer.xlsx` | Superset of the above, adding mBERT, GPT-2 and ByT5 token counts and TP for every text column. Read only by notebook 08. |

Sheet names carry the upstream spacing exactly and must not be normalised:

| Language | Sheet |
|---|---|
| Gujarati | `Indic_mt _for_analysis - Gujara` |
| Hindi | `Indic_mt _for_analysis - Hindi_` |
| Malayalam | `Indic_mt _for_analysis - Malaya` |
| Marathi | `Indic_mt _for_analysis - Marath` |
| Tamil | `Indic_mt _for_analysis - Tamil_` |

### The columns the analysis reads

| Column | Meaning |
|---|---|
| `COMET` / `COMET_romanized` | COMET, native and romanised target |
| `Translation_xlmr_TP` / `Translation_Transliteration_romanized_xlmr_TP` | Tokenization Parity |
| `Translation_xlmr_IP` / `Translation_Transliteration_romanized_xlmr_IP` | Information Parity |
| `Human_scores` | MQM-derived human score |
| `Error1_Severity` / `Error1_Type` | Primary annotated error |

Everything else in the workbooks — including all raw text and per-token strings —
is retained for traceability but is not read by any analysis.

---

## How to get the raw data

The raw MQM annotation files are not included here. Download them from
[AI4Bharat/IndicMT-Eval](https://github.com/AI4Bharat/IndicMT-Eval/tree/master/Dataset).
See [`../README.md`](../README.md) for what would then have to be run to
regenerate these workbooks, and why that path will not reproduce them
bit-for-bit.
