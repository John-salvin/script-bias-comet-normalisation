#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_all.py
================
Single-file reproduction of every number reported for the COMET-QN /
parity-feature calibration analysis in "Lost in Tokenization" (journal
extension).

Data source (single source of truth):
    Information_parity_outputs_all.xlsx
    5 Indic sheets, 1400 annotated segments each.

What it reproduces
------------------
  [0] Baseline table  : per-language COMET native/romanised means, gaps,
                        and Spearman(nat/rom, human).            (§1 / Table: baseline)
  [1] Corrector table : raw / mean-shift / affine / quantile / isotonic
                        / native-ceiling, with mean|gap|, lang-avg Spearman,
                        pooled Spearman.                          (Table 8-ish)
  [2] LOLO table      : leave-one-language-out parity-feature GBM,
                        raw/cal/native Spearman + % recovered.    (Table 9)
  [3] COMET-QN        : quantile renormalisation to a common reference,
                        gap->0, pooled corr up, within-lang rank invariant.
  [4] Diagnostic slopes: pooled COMET~TP and COMET~IP slopes.

Determinism
-----------
  numpy seed = 42 everywhere a split is drawn.
  GBM random_state = 0.

Usage
-----
  python3 reproduce_all.py --xlsx /path/to/Information_parity_outputs_all.xlsx
  (defaults to ./Information_parity_outputs_all.xlsx if --xlsx omitted)

Dependencies: pandas, numpy, scipy, scikit-learn, openpyxl
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats

# scikit-learn is only needed for [1:isotonic] and [2:GBM]
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42

SHEET_MAP = {
    "Gujarati":  "Indic_mt _for_analysis - Gujara",
    "Hindi":     "Indic_mt _for_analysis - Hindi_",
    "Malayalam": "Indic_mt _for_analysis - Malaya",
    "Marathi":   "Indic_mt _for_analysis - Marath",
    "Tamil":     "Indic_mt _for_analysis - Tamil_",
}
LANGS = list(SHEET_MAP)

# The seven numeric columns every analysis depends on.
NUM = [
    "COMET",
    "COMET_romanized",
    "Human_scores",
    "Translation_xlmr_TP",
    "Translation_Transliteration_romanized_xlmr_TP",
    "Translation_xlmr_IP",
    "Translation_Transliteration_romanized_xlmr_IP",
]


# --------------------------------------------------------------------------- #
# Data loading  (shared by all analyses)
# --------------------------------------------------------------------------- #
def load(xlsx_path):
    r"""
    Load all five sheets, coerce the seven required columns to numeric
    (this silently converts the stray Malayalam string cell `\`19` -> NaN),
    drop Unnamed columns, drop any row missing a required value, and stack.

    Returns the pooled DataFrame D with derived columns:
        dTP, dIP  : native-minus-romanised parity deltas
        gap       : COMET - COMET_romanized  (per-segment)
    r"""
    xl = pd.ExcelFile(xlsx_path)
    frames = []
    dropped = {}
    for lang, sheet in SHEET_MAP.items():
        df = pd.read_excel(xl, sheet_name=sheet)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        for col in NUM:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        before = len(df)
        df = df.dropna(subset=NUM).copy()
        dropped[lang] = before - len(df)
        df["Language"] = lang
        frames.append(df[NUM + ["Language"]])

    D = pd.concat(frames, ignore_index=True)
    D["dTP"] = D["Translation_xlmr_TP"] - D["Translation_Transliteration_romanized_xlmr_TP"]
    D["dIP"] = D["Translation_xlmr_IP"] - D["Translation_Transliteration_romanized_xlmr_IP"]
    D["gap"] = D["COMET"] - D["COMET_romanized"]

    print("Rows dropped per language (missing/non-numeric):", dropped)
    print("Total clean rows:", len(D))
    return D


# --------------------------------------------------------------------------- #
# [0] Baseline
# --------------------------------------------------------------------------- #
def analysis_baseline(D):
    print("\n" + "=" * 78)
    print("[0] BASELINE  — per-language means, gap, Spearman(nat/rom, human)")
    print("=" * 78)
    print(f"{'Lang':10s} {'n':>5s} {'nat':>8s} {'rom':>8s} {'gap':>8s} "
          f"{'Sp_nat':>8s} {'Sp_rom':>8s}")
    for l in LANGS:
        s = D[D.Language == l]
        c, cr, h = s.COMET, s.COMET_romanized, s.Human_scores
        print(f"{l:10s} {len(s):5d} {c.mean():8.3f} {cr.mean():8.3f} "
              f"{(c-cr).mean():8.3f} "
              f"{stats.spearmanr(c,h)[0]:8.4f} {stats.spearmanr(cr,h)[0]:8.4f}")
    c, cr, h = D.COMET, D.COMET_romanized, D.Human_scores
    print("-" * 60)
    print(f"{'POOLED':10s} {len(D):5d} {c.mean():8.3f} {cr.mean():8.3f} "
          f"{(c-cr).mean():8.3f} "
          f"{stats.spearmanr(c,h)[0]:8.4f} {stats.spearmanr(cr,h)[0]:8.4f}")


# --------------------------------------------------------------------------- #
# [1] Corrector-class table
# --------------------------------------------------------------------------- #
def _score_block(te, corrected_by_lang):
    """Given test frame and dict lang->corrected romanised vector (aligned to
    te[te.Language==lang] order), return (mean|gap|, lang-avg Spearman,
    pooled Spearman)."""
    gaps, sp_lang, pooled_x, pooled_h = [], [], [], []
    for l in LANGS:
        sub = te[te.Language == l]
        x = np.asarray(corrected_by_lang[l], dtype=float)
        gaps.append(abs(sub.COMET.mean() - x.mean()))
        sp_lang.append(stats.spearmanr(x, sub.Human_scores)[0])
        pooled_x.append(pd.Series(x, index=sub.index))
        pooled_h.append(sub.Human_scores)
    px = pd.concat(pooled_x)
    ph = pd.concat(pooled_h)
    return np.mean(gaps), np.mean(sp_lang), stats.spearmanr(px, ph)[0]


def analysis_correctors(D):
    print("\n" + "=" * 78)
    print("[1] CORRECTOR CLASSES — 50/50 within-language split (seed=42)")
    print("    columns: mean|gap|   Spearman(lang-avg)   Spearman(pooled)")
    print("=" * 78)

    rng = np.random.RandomState(SEED)
    D = D.copy()
    D["fold"] = 0
    for l in LANGS:
        idx = D.index[D.Language == l].to_numpy()
        test = rng.choice(idx, size=len(idx) // 2, replace=False)
        D.loc[test, "fold"] = 1
    tr, te = D[D.fold == 0], D[D.fold == 1]

    rows = []

    # raw romanised (identity)
    rows.append(("Raw romanised (identity)",
                 {l: te[te.Language == l].COMET_romanized.values for l in LANGS}))

    # native ceiling
    native = {l: te[te.Language == l].COMET.values for l in LANGS}

    # A: per-language mean shift  (fit shift on train)
    shift = {l: tr[tr.Language == l].COMET.mean()
                - tr[tr.Language == l].COMET_romanized.mean() for l in LANGS}
    rows.append(("Per-lang mean shift (location)",
                 {l: te[te.Language == l].COMET_romanized.values + shift[l] for l in LANGS}))

    # B: per-language affine z-match
    def affine(l, x):
        t = tr[tr.Language == l]
        mu_r, sd_r = t.COMET_romanized.mean(), t.COMET_romanized.std()
        mu_n, sd_n = t.COMET.mean(), t.COMET.std()
        return (x - mu_r) / sd_r * sd_n + mu_n
    rows.append(("Per-lang affine / z (loc.+scale)",
                 {l: affine(l, te[te.Language == l].COMET_romanized.values) for l in LANGS}))

    # C: per-language quantile mapping (train romanised -> train native quantiles)
    def qmap(l, x):
        t = tr[tr.Language == l]
        r = np.searchsorted(np.sort(t.COMET_romanized.values), x) / len(t)
        r = np.clip(r, 0.001, 0.999)
        return np.quantile(t.COMET.values, r)
    rows.append(("Per-lang quantile map (monotone)",
                 {l: qmap(l, te[te.Language == l].COMET_romanized.values) for l in LANGS}))

    # D: per-language paired isotonic  (romanised -> native)
    def iso(l, x):
        t = tr[tr.Language == l]
        ir = IsotonicRegression(out_of_bounds="clip").fit(
            t.COMET_romanized.values, t.COMET.values)
        return ir.predict(x)
    rows.append(("Per-lang isotonic (monotone)",
                 {l: iso(l, te[te.Language == l].COMET_romanized.values) for l in LANGS}))

    # print
    print(f"{'Corrector':36s} {'|gap|':>7s} {'rho_lang':>9s} {'rho_pool':>9s}")
    for name, corr in rows:
        g, sl, sp = _score_block(te, corr)
        print(f"{name:36s} {g:7.2f} {sl:9.3f} {sp:9.3f}")
    g, sl, sp = _score_block(te, native)
    print(f"{'Native score (ceiling)':36s} {g:7.2f} {sl:9.3f} {sp:9.3f}")


# --------------------------------------------------------------------------- #
# [2] LOLO parity-feature calibration  (Table 9)
# --------------------------------------------------------------------------- #
def analysis_lolo(D):
    print("\n" + "=" * 78)
    print("[2] LOLO PARITY-FEATURE CALIBRATION (Table 9)")
    print("    GBM: 300 trees, depth 3, lr 0.05, random_state=0")
    print("    fit on 4 languages -> predict native COMET on held-out 5th")
    print("=" * 78)

    F = ["COMET_romanized", "dTP", "dIP",
         "Translation_Transliteration_romanized_xlmr_TP",
         "Translation_Transliteration_romanized_xlmr_IP",
         "Translation_xlmr_TP", "Translation_xlmr_IP"]

    print(f"{'Held-out':10s} {'rho_raw':>8s} {'rho_cal':>8s} {'rho_nat':>8s} {'recovered':>10s}")
    recs = []
    for hold in LANGS:
        tr = D[D.Language != hold]
        te = D[D.Language == hold]
        gbm = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0
        ).fit(tr[F], tr.COMET)
        pred = gbm.predict(te[F])
        h = te.Human_scores
        raw = stats.spearmanr(te.COMET_romanized, h)[0]
        cal = stats.spearmanr(pred, h)[0]
        nat = stats.spearmanr(te.COMET, h)[0]
        rec = (cal - raw) / (nat - raw) if nat != raw else float("nan")
        recs.append(rec)
        print(f"{hold:10s} {raw:8.3f} {cal:8.3f} {nat:8.3f} {rec*100:9.1f}%")
    print("-" * 48)
    print(f"{'MEAN':10s} {'':8s} {'':8s} {'':8s} {np.mean(recs)*100:9.1f}%")


# --------------------------------------------------------------------------- #
# [3] COMET-QN — quantile renormalisation to a common reference
# --------------------------------------------------------------------------- #
def analysis_qn(D):
    print("\n" + "=" * 78)
    print("[3] COMET-QN — quantile renormalisation to pooled-native reference")
    print("=" * 78)

    ref = np.sort(D.COMET.values)  # common reference distribution

    def qnorm(x):
        x = np.asarray(x, dtype=float)
        ranks = stats.rankdata(x) / (len(x) + 1)
        return np.quantile(ref, ranks)

    before_x, after_x, H = [], [], []
    gaps_b, gaps_a = [], []
    for l in LANGS:
        s = D[D.Language == l]
        nb, rb = s.COMET.values, s.COMET_romanized.values
        na, ra = qnorm(nb), qnorm(rb)
        before_x += [nb, rb]
        after_x += [na, ra]
        H += [s.Human_scores.values, s.Human_scores.values]
        gaps_b.append((nb - rb).mean())
        gaps_a.append((na - ra).mean())

    B = np.concatenate(before_x)
    A = np.concatenate(after_x)
    Hc = np.concatenate(H)
    print(f"Pooled 10 cells, n={len(B)}")
    print(f"  BEFORE: Pearson={stats.pearsonr(B,Hc)[0]:.4f}  Spearman={stats.spearmanr(B,Hc)[0]:.4f}")
    print(f"  AFTER : Pearson={stats.pearsonr(A,Hc)[0]:.4f}  Spearman={stats.spearmanr(A,Hc)[0]:.4f}")
    print(f"  mean|per-lang gap| BEFORE={np.mean(np.abs(gaps_b)):.3f}  AFTER={np.mean(np.abs(gaps_a)):.3f}")
    print("  per-language gap BEFORE:", [f"{g:+.2f}" for g in gaps_b])
    print("  per-language gap AFTER :", [f"{g:+.2f}" for g in gaps_a])

    print("\n  Within-language Spearman(native, human): unchanged by QN")
    print(f"  {'lang':10s} {'before':>8s} {'after':>8s}")
    for l in LANGS:
        s = D[D.Language == l]
        b = stats.spearmanr(s.COMET, s.Human_scores)[0]
        a = stats.spearmanr(qnorm(s.COMET.values), s.Human_scores)[0]
        print(f"  {l:10s} {b:8.4f} {a:8.4f}")


# --------------------------------------------------------------------------- #
# [4] Diagnostic slopes
# --------------------------------------------------------------------------- #
def analysis_slopes(D):
    print("\n" + "=" * 78)
    print("[4] DIAGNOSTIC SLOPES — pooled COMET ~ TP and COMET ~ IP")
    print("=" * 78)
    X_tp = np.concatenate([D.Translation_xlmr_TP.values,
                           D.Translation_Transliteration_romanized_xlmr_TP.values])
    X_ip = np.concatenate([D.Translation_xlmr_IP.values,
                           D.Translation_Transliteration_romanized_xlmr_IP.values])
    Y = np.concatenate([D.COMET.values, D.COMET_romanized.values])
    b_tp = np.polyfit(X_tp, Y, 1)[0]
    b_ip = np.polyfit(X_ip, Y, 1)[0]
    print(f"  slope COMET~TP = {b_tp:.3f} COMET points per unit TP")
    print(f"  slope COMET~IP = {b_ip:.3f} COMET points per unit IP")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="Information_parity_outputs_all.xlsx")
    args = ap.parse_args()

    print("Loading:", args.xlsx)
    D = load(args.xlsx)

    analysis_baseline(D)
    analysis_correctors(D)
    analysis_lolo(D)
    analysis_qn(D)
    analysis_slopes(D)

    print("\nDone. All numbers above are computed from the xlsx with seed",
          SEED, "(GBM random_state=0).")


if __name__ == "__main__":
    main()
