# Notebooks

Twelve notebooks reproduce every result in the paper. **Run them in order** — each
writes outputs the next reads, and notebook 12 checks all of them at once.

```bash
# from the repository root
make notebooks

# or interactively
jupyter lab      # then open 01_load_and_sample_sizes.ipynb
```

Install dependencies from `requirements.txt` at the **root of the repository**:

```bash
pip install -r requirements.txt          # from the repo root
pip install -r ../requirements.txt       # from inside notebooks/
```

---

## Execution order

| # | Notebook | What it does | Writes |
|---|---|---|---|
| 01 | `01_load_and_sample_sizes.ipynb` | Establishes the three reporting bases (7,000 / 6,995 / 1,258); itemises the five dropped segments; confirms COMET is complete | `sample_sizes.csv` |
| 02 | `02_romanisation_effect.ipynb` | Tables 2 and 13 (mean COMET, TP and IP), Table 10 (Spearman ρ for all six metrics), Table 11 (Meng, Steiger and a 5,000-round paired permutation test, the ANOVA on both bases, and the HIN–GUJ Welch *t*) | `table2_comet_tp_ip.csv`, `table10_correlation_tests.csv`, `anova_and_hin_guj.csv` |
| 03 | `03_diagnostics_sbi_ipi_tax.ipynb` | Tables 12, 13 and 14 — SBI, IPI, zone classification, and the Computational Tax decomposition into LP × EP; the two WMT24 Latin-script controls | `diagnostics_sbi_ipi_tax.csv`, `latin_controls.csv` |
| 04 | `04_severity_inversion.ipynb` | Table 3 — Marathi severity inversion: mean ΔCOMET by MQM severity bucket, sentence-level Spearman ρ, Wilcoxon signed-rank, and the worked 36.6 → 62.7 example | `severity_inversion.csv` |
| 05 | `05_corrector_impossibility.ipynb` | Table 5 — mean shift, affine, quantile map and isotonic correctors against the native ceiling, on a 50/50 within-language split | `table5_corrector_impossibility.csv` |
| 06 | `06_comet_qn.ipynb` | Tables 6 and 7 — Eq. 3 worked through five segments, then COMET-QN over the ten (language, script) cells; rank-invariance check; pooled COMET~TP and COMET~IP slopes | `table7_comet_qn.csv` |
| 07 | `07_lolo_calibration.ipynb` | Table 8 — leave-one-language-out parity-feature calibration with a gradient-boosted regressor | `table8_lolo_calibration.csv` |
| 08 | `08_tokenizer_robustness.ipynb` | Table 4 — re-measures TP through mBERT, GPT-2 and ByT5 to test whether the XLM-R effect is tokenizer-specific | `tokenizer_robustness.csv` |
| 09 | `09_corpus_corroboration.ipynb` | Tables 13, 15 and 17 — MATTR (w = 500), byte premium, dependent-vowel rate, word-count vs XLM-R token-count Spearman, and sentence-level IP–COMET Pearson | `corpus_corroboration.csv` |
| 10 | `10_latin_controls.ipynb` | Table 16 and Appendix E — ENG-DEU / ENG-SPA IP, TP, SBI, IPI and zone placement, plus the DEU SBI reversal (Kruskal–Wallis H = 135.0) | `latin_controls_full.csv`, `deu_sbi_severity.csv` |
| 11 | `11_figures.ipynb` | Figure 2 (IPI zone transitions) and Figure 1 (COMET-QN before/after), both recomputed from data rather than redrawn | `results/figures/fig_ipi_zones.{png,pdf}`, `fig_comet_qn.{png,pdf}` |
| 12 | `12_verify_paper_numbers.ipynb` | Runs `scripts/verify_paper_numbers.py` over the whole registry and asserts a clean exit | `results/logs/verification.md` |

Runtime for the full sequence is about six minutes on an 8-core CPU. Notebooks
02 and 07 dominate it.

---

## Conventions

Each notebook is **self-contained**: it restates its own configuration block
rather than importing shared code, so it can be read and run on its own. Every
path is relative — nothing in this repository refers to an absolute filesystem
location.

Every notebook follows the same shape:

1. A title cell naming what it produces, with its inputs and outputs.
2. `## Step 0 — Configuration` — paths, sheet names, column names, seeds.
3. Numbered `## Step N` sections, each a markdown explanation followed by one code cell.
4. **Cross-verification cells** that assert computed values against the reported
   values and emit a `✓` line for each. These are not decorative: a notebook
   whose assertions fail aborts `make notebooks`.
5. An output manifest, then a references cell.

Notebooks are committed **output-stripped** with `execution_count: null`.
`make notebooks` writes executed copies with outputs to
`results/logs/executed/`, which is git-ignored, so running the pipeline never
dirties the working tree.

---

## Seeds

| Seed | Value | Used by |
|---|---|---|
| Split | 42 | 50/50 within-language split, `numpy.RandomState(42).choice` consumed one language at a time in alphabetical order (notebook 05). |
| GBM | 0 | `GradientBoostingRegressor` (notebook 07) |
| Permutation | 0 | 5,000-round paired sign-flip test (notebook 02) |

Results depending on these are registered with status `script` in
`../paper_numbers.yaml` and are reproducible only under these seeds.

---

## References

- Sai et al. (2023). *IndicMT Eval.* ACL 2023. https://aclanthology.org/2023.acl-long.795
- Kocmi et al. (2024). *Findings of WMT24.* WMT 2024. https://aclanthology.org/2024.wmt-1.1
- Madhani et al. (2023). *Aksharantar.* EMNLP Findings 2023.
- Conneau et al. (2020). *XLM-RoBERTa.* ACL 2020.
- Rei et al. (2020). *COMET.* EMNLP 2020.
- Petrov et al. (2023). *Tokenizer unfairness.* NeurIPS 36.
- Tsvetkov & Kipnis (2024). *Information Parity.* EMNLP Findings 2024.
- Meng, Rosenthal & Rubin (1992). *Comparing correlated correlation coefficients.* Psychological Bulletin 111(1).
- Steiger (1980). *Tests for comparing elements of a correlation matrix.* Psychological Bulletin 87(2).
