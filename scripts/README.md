# Scripts

Command-line entry points. The notebooks are the readable pipeline; these are the
pieces that need to run headlessly, plus a single-file reference implementation.

| Script | Purpose |
|---|---|
| `check_data.py` | Confirms the committed workbooks are present and prints their SHA-256 digests. `--strict` compares against the digests recorded in `data/README.md` and exits non-zero on mismatch. Invoked by `make data`. |
| `verify_paper_numbers.py` | Recomputes every entry in `paper_numbers.yaml` **independently of the notebooks** and reports `[PASS]` / `[FAIL]` / `[SKIP]` per entry to stdout and to `results/logs/verification.md`. Exits non-zero on any failure. Invoked by `make verify` and by notebook 12. |
| `reproduce_all.py` | Single-file reference implementation covering the baseline table, the corrector classes, the leave-one-language-out calibration, COMET-QN and the pooled diagnostic slopes. Independent of the notebook pipeline. |

```bash
python3 scripts/check_data.py --strict
python3 scripts/verify_paper_numbers.py
python3 scripts/reproduce_all.py --xlsx data/indic/indic_parity_xlmr.xlsx
```

`verify_paper_numbers.py` re-derives each quantity from the raw columns rather
than importing the notebooks, so agreement means two separate implementations
concur. It takes about 80 seconds, dominated by the permutation test and the ten
gradient-boosted fits. Its `compute_all()` is also what `tests/` builds on, via a
session-scoped fixture in `tests/conftest.py`.

`reproduce_all.py` uses its own internal alphabetical language ordering
(Gujarati, Hindi, Malayalam, Marathi, Tamil) rather than the display order
(GUJ, TAM, MAL, MAR, HIN) used everywhere else. The 50/50 split in Table 7
depends on that iteration order, so it is preserved deliberately.
