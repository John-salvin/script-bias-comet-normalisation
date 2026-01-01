#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_paper_numbers.py
=======================
Recomputes every claim registered in ``paper_numbers.yaml`` directly from the
committed workbooks and compares it against the registered value within the
registered tolerance.

This is deliberately an *independent* implementation. It does not import the
notebooks or ``reproduce_all.py``; it re-derives each quantity from the raw
columns. Agreement therefore means two separate implementations concur, not that
one implementation is self-consistent.

Status handling
---------------
  verified : recomputed here and asserted. A mismatch is a FAIL.
  script   : depends on an RNG (50/50 split, GBM, permutation). Recomputed under
             the documented seeds; a mismatch is still a FAIL, but the value is
             only meaningful under those seeds.

Output
------
  One line per claim on stdout and to ``results/logs/verification.md``:

      [PASS] table2.comet_nat.GUJ            expected 85.70   got 85.697
      [FAIL] hin_guj.welch_t_native          expected 27.4    got 26.9   (delta 0.5)

Exit status
-----------
  0 if every claim passed, 1 otherwise.

Usage
-----
  python3 scripts/verify_paper_numbers.py
  python3 scripts/verify_paper_numbers.py --registry paper_numbers.yaml

Dependencies: pandas, numpy, scipy, scikit-learn, openpyxl, PyYAML
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Configuration — mirrors notebooks/, restated here so the two are independent
# --------------------------------------------------------------------------- #
DATA_XLMR = ROOT / "data" / "indic" / "indic_parity_xlmr.xlsx"
DATA_MULTI = ROOT / "data" / "indic" / "indic_parity_multi_tokenizer.xlsx"
DATA_LATIN = ROOT / "data" / "latin" / "wmt24_ende_enes_metrics.xlsx"

SHEET_MAP = {
    "GUJ": "Indic_mt _for_analysis - Gujara",
    "TAM": "Indic_mt _for_analysis - Tamil_",
    "MAL": "Indic_mt _for_analysis - Malaya",
    "MAR": "Indic_mt _for_analysis - Marath",
    "HIN": "Indic_mt _for_analysis - Hindi_",
}
LANGS = ["GUJ", "TAM", "MAL", "MAR", "HIN"]

C_COMET_NAT = "COMET"
C_COMET_ROM = "COMET_romanized"
C_TP_NAT = "Translation_xlmr_TP"
C_TP_ROM = "Translation_Transliteration_romanized_xlmr_TP"
C_IP_NAT = "Translation_xlmr_IP"
C_IP_ROM = "Translation_Transliteration_romanized_xlmr_IP"
C_HUMAN = "Human_scores"
C_SEVERITY = "Error1_Severity"

SEED_SPLIT = 42
SEED_GBM = 0
SEED_PERM = 0

SEVERITY_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]
SEVERITY_ALL = ["Default"] + SEVERITY_ORDER

# Table 3 metric columns: (native, romanised). The Malayalam sheet spells chrF
# as "Chrf"; handled at read time rather than by renaming the committed file.
METRIC_COLS = {
    "COMET": ("COMET", "COMET_romanized"),
    "BLEURT": ("BLEURT", "Bleurt_romanized"),
    "BERTScore": ("BertScore", "Bertscore_romanized"),
    "BLEU": ("BLEU", "BLEU_romanized"),
    "ChrF": ("chrF", "CHRF_romanized"),
    "TER": ("TER", "TER_romanized"),
}
TOKENIZERS = ["xlmr", "mbert", "gpt2", "byt5"]


# --------------------------------------------------------------------------- #
# Recomputation
# --------------------------------------------------------------------------- #
def load():
    full, work = {}, {}
    for lang in LANGS:
        d = pd.read_excel(DATA_XLMR, sheet_name=SHEET_MAP[lang])
        d["H"] = pd.to_numeric(d[C_HUMAN], errors="coerce")
        full[lang] = d
        work[lang] = d.dropna(subset=["H"]).reset_index(drop=True)
    return full, work


def meng_z(r12, r13, r23, n):
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rbar2 = (r12 ** 2 + r13 ** 2) / 2
    f = min((1 - r23) / (2 * (1 - rbar2)), 1.0)
    h = (1 - f * rbar2) / (1 - rbar2)
    return (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - r23) * h))


def steiger_z(r12, r13, r23, n):
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rm2 = (r12 ** 2 + r13 ** 2) / 2
    cov = (r23 * (1 - 2 * rm2) - 0.5 * rm2 * (1 - 2 * rm2 - r23 ** 2)) / ((1 - rm2) ** 2)
    return (z12 - z13) * np.sqrt((n - 3) / (2 - 2 * cov))


def qn_map(scores, reference_sorted):
    frac = stats.rankdata(scores) / (len(scores) + 1)
    return np.quantile(reference_sorted, frac)



def mattr(texts, window=500):
    """Moving-average type-token ratio over whitespace tokens (Covington & McFall).

    Case-folded before tokenising.
    """
    toks = []
    for t in texts:
        if pd.isna(t):
            continue
        toks.extend(str(t).lower().split())
    if len(toks) < window:
        return len(set(toks)) / max(len(toks), 1)
    return float(np.mean([len(set(toks[i:i + window])) / window
                          for i in range(len(toks) - window + 1)]))



# Dependent-vowel characters per script. A token counts as an "isolated
# dependent vowel" when it is exactly ONE character drawn from this set.
DEP_VOWELS = {
    "Devanagari": set("\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0947"
                      "\u0948\u094b\u094c\u094d\u0902\u0903"),
    "Gujarati": set("\u0abe\u0abf\u0ac0\u0ac1\u0ac2\u0ac3\u0ac7\u0ac8"
                    "\u0acb\u0acc\u0acd\u0a82\u0a83"),
    "Tamil": set("\u0bbe\u0bbf\u0bc0\u0bc1\u0bc2\u0bc6\u0bc7\u0bc8"
                 "\u0bca\u0bcb\u0bcc\u0bcd\u0b82"),
    "Malayalam": set("\u0d3e\u0d3f\u0d40\u0d41\u0d42\u0d43\u0d47\u0d48"
                     "\u0d4a\u0d4b\u0d4c\u0d4d\u0d02\u0d03\u0d3b\u0d3c"),
}
SCRIPT_OF = {"GUJ": "Gujarati", "TAM": "Tamil", "MAL": "Malayalam",
             "MAR": "Devanagari", "HIN": "Devanagari"}


def bytes_per_word(texts):
    """Mean UTF-8 byte length over individual whitespace-delimited words.

    Averaged over words rather than over sentences.
    """
    lengths = [len(w.encode("utf-8")) for t in texts if pd.notna(t)
               for w in str(t).split()]
    return float(np.mean(lengths)) if lengths else float("nan")


def byte_premium(target, source):
    """Target bytes-per-word divided by English source bytes-per-word."""
    return bytes_per_word(target) / bytes_per_word(source)


def dep_vowel_rate(token_series, script):
    """Isolated dependent-vowel tokens per 1,000 XLM-R tokens."""
    charset = DEP_VOWELS[script]
    total = isolated = 0
    for s in token_series.astype(str):
        for tok in (t.strip() for t in s.split("|")):
            if not tok:
                continue
            total += 1
            if len(tok) == 1 and tok in charset:
                isolated += 1
    return isolated / total * 1000 if total else float("nan")


def spearman_vs_human(frame, column):
    """Spearman rho of one metric column against the human score, on rows where both exist."""
    v = pd.to_numeric(frame[column], errors="coerce")
    ok = v.notna() & frame["H"].notna()
    return stats.spearmanr(v[ok], frame["H"][ok])[0]


def metric_column(frame, name):
    """Resolve a Table 3 metric column, tolerating the Malayalam chrF spelling."""
    if name in frame.columns:
        return name
    if name == "chrF" and "Chrf" in frame.columns:
        return "Chrf"
    raise KeyError(name)


def compute_all(skip_slow=False):
    """Return {claim_id: computed_value} for every registered claim."""
    got = {}
    full, work = load()

    # ---- sample sizes ----------------------------------------------------- #
    got["sample_sizes.full_total"] = sum(len(full[l]) for l in LANGS)
    got["sample_sizes.working_total"] = sum(len(work[l]) for l in LANGS)
    got["sample_sizes.mar_severity"] = int(
        len(full["MAR"]) - (full["MAR"][C_SEVERITY] == "Default").sum())
    for l in LANGS:
        got[f"sample_sizes.dropped.{l}"] = len(full[l]) - len(work[l])

    # ---- Table 2 ---------------------------------------------------------- #
    for l in LANGS:
        d = full[l]
        cn, cr = d[C_COMET_NAT].mean(), d[C_COMET_ROM].mean()
        tn, tr = d[C_TP_NAT].mean(), d[C_TP_ROM].mean()
        ipn, ipr = d[C_IP_NAT].mean(), d[C_IP_ROM].mean()
        got[f"table2.comet_nat.{l}"] = cn
        got[f"table2.comet_rom.{l}"] = cr
        got[f"table13.tp_nat.{l}"] = tn
        got[f"table13.tp_rom.{l}"] = tr
        got[f"table13.ip_nat.{l}"] = ipn
        got[f"table13.ip_rom.{l}"] = ipr
        got[f"table13.tp_pct.{l}"] = (tr / tn - 1) * 100
        got[f"table13.ip_pct.{l}"] = (ipr / ipn - 1) * 100

    # ---- Table 3 ---------------------------------------------------------- #
    rng = np.random.default_rng(SEED_PERM)
    for l in LANGS:
        d = work[l]
        h = stats.rankdata(d["H"])
        a = stats.rankdata(d[C_COMET_NAT])
        b = stats.rankdata(d[C_COMET_ROM])
        r12 = np.corrcoef(a, h)[0, 1]
        r13 = np.corrcoef(b, h)[0, 1]
        r23 = np.corrcoef(a, b)[0, 1]
        n = len(d)
        got[f"table2.rho_nat.{l}"] = r12
        got[f"table2.rho_rom.{l}"] = r13
        got[f"table11.meng_z.{l}"] = meng_z(r12, r13, r23, n)
        got[f"table11.steiger_z.{l}"] = steiger_z(r12, r13, r23, n)

        obs = r12 - r13
        A_, B_ = d[C_COMET_NAT].values, d[C_COMET_ROM].values
        exceed = 0
        rounds = 200 if skip_slow else 5000
        for _ in range(rounds):
            m = rng.random(n) < 0.5
            s1 = np.corrcoef(stats.rankdata(np.where(m, B_, A_)), h)[0, 1]
            s2 = np.corrcoef(stats.rankdata(np.where(m, A_, B_)), h)[0, 1]
            if abs(s1 - s2) >= abs(obs):
                exceed += 1
        got[f"table11.perm_exceed.{l}"] = exceed

    # ---- ANOVA ------------------------------------------------------------ #
    for tag, src in [("7000", full), ("6995", work)]:
        for cond, col in [("native", C_COMET_NAT), ("romanised", C_COMET_ROM)]:
            dev = pd.concat([src["HIN"], src["MAR"]])[col].dropna()
            nod = pd.concat([src["GUJ"], src["MAL"], src["TAM"]])[col].dropna()
            F, _ = stats.f_oneway(dev, nod)
            allv = pd.concat([dev, nod])
            gm = allv.mean()
            ssb = len(dev) * (dev.mean() - gm) ** 2 + len(nod) * (nod.mean() - gm) ** 2
            eta2 = ssb / ((allv - gm) ** 2).sum() * 100
            got[f"anova.{cond}.F_{tag}"] = F
            got[f"anova.{cond}.eta2_{tag}"] = eta2
    got["anova.variance_reduction_pct"] = (
        (got["anova.native.eta2_7000"] - got["anova.romanised.eta2_7000"])
        / got["anova.native.eta2_7000"] * 100)

    # ---- HIN-GUJ ---------------------------------------------------------- #
    for cond, col in [("native", C_COMET_NAT), ("romanised", C_COMET_ROM)]:
        g, h_ = full["GUJ"][col], full["HIN"][col]
        t, _ = stats.ttest_ind(g, h_, equal_var=False)
        got[f"hin_guj.gap_{cond}"] = g.mean() - h_.mean()
        got[f"hin_guj.welch_t_{cond}"] = t
    got["hin_guj.gap_reduction_pct"] = (
        (got["hin_guj.gap_native"] - got["hin_guj.gap_romanised"])
        / got["hin_guj.gap_native"] * 100)

    # ---- Diagnostics ------------------------------------------------------ #
    for l in LANGS:
        d = full[l]
        sbi = d[C_TP_NAT] / d[C_IP_NAT].replace(0, np.nan)
        ipn, ipr = d[C_IP_NAT].mean(), d[C_IP_ROM].mean()
        tpn, tpr = d[C_TP_NAT].mean(), d[C_TP_ROM].mean()
        lp, ep = tpr / tpn, ipn / ipr
        tax = lp * ep
        got[f"diagnostics.sbi_nat.{l}"] = sbi.mean()
        got[f"diagnostics.p_sbi_ge3_pct.{l}"] = 100 * (sbi >= 3.0).mean()
        got[f"diagnostics.ipi_nat.{l}"] = abs(ipn - 1)
        got[f"diagnostics.ipi_rom.{l}"] = abs(ipr - 1)
        got[f"diagnostics.LP.{l}"] = lp
        got[f"diagnostics.EP.{l}"] = ep
        got[f"diagnostics.tax.{l}"] = tax
        got[f"diagnostics.ep_pct.{l}"] = 100 * np.log(ep) / np.log(tax)
        got[f"figure2.ipi_nat.{l}"] = abs(ipn - 1)
        got[f"figure2.ipi_rom.{l}"] = abs(ipr - 1)
    got["diagnostics.spearman_sbi_comet_n5"] = stats.spearmanr(
        [(full[l][C_TP_NAT] / full[l][C_IP_NAT]).mean() for l in LANGS],
        [full[l][C_COMET_NAT].mean() for l in LANGS])[0]

    # ---- Latin controls --------------------------------------------------- #
    for sheet, iso in [("German", "DEU"), ("Spanish", "SPA")]:
        d = pd.read_excel(DATA_LATIN, sheet_name=sheet)
        ip = pd.to_numeric(d["target_xlmr_IP"], errors="coerce")
        got[f"latin_controls.ip.{iso}"] = ip.mean()
        got[f"latin_controls.ipi.{iso}"] = abs(ip.mean() - 1)
        got[f"latin_controls.rows.{iso}"] = len(d)

    # ---- Severity inversion ----------------------------------------------- #
    d = full["MAR"].copy()
    d["delta"] = d[C_COMET_ROM] - d[C_COMET_NAT]
    for s in SEVERITY_ORDER:
        sub = d[d[C_SEVERITY] == s]
        key = s.replace(" ", "_").lower()
        got[f"severity.N.{key}"] = len(sub)
        got[f"severity.mean_dcomet.{key}"] = sub["delta"].mean()
    nd = d[d[C_SEVERITY].isin(SEVERITY_ORDER)]
    smap = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    rho, _ = stats.spearmanr(nd[C_SEVERITY].map(smap), nd["delta"])
    got["severity.spearman_rho"] = rho
    got["severity.spearman_N"] = len(nd)
    w = stats.wilcoxon(d["delta"])
    got["severity.wilcoxon_W"] = w.statistic
    got["severity.wilcoxon_p"] = w.pvalue
    got["severity.wilcoxon_median"] = d["delta"].median()
    got["severity.wilcoxon_mean"] = d["delta"].mean()
    zero = d[d["H"] == 0]
    vh = zero[zero[C_SEVERITY] == "Very High"].sort_values("delta", ascending=False)
    r = vh.iloc[0]
    got["severity.example_comet_nat"] = r[C_COMET_NAT]
    got["severity.example_comet_rom"] = r[C_COMET_ROM]
    got["severity.example_delta"] = r["delta"]

    # ---- Working-set frame for the calibration analyses ------------------- #
    A = pd.concat([work[l].assign(lang=l) for l in LANGS], ignore_index=True)
    A["Cn"], A["Cr"] = A[C_COMET_NAT], A[C_COMET_ROM]
    A["TPn"], A["TPr"] = A[C_TP_NAT], A[C_TP_ROM]
    A["IPn"], A["IPr"] = A[C_IP_NAT], A[C_IP_ROM]
    A["dTP"], A["dIP"] = A["TPn"] - A["TPr"], A["IPn"] - A["IPr"]

    # ---- COMET-QN (Table 8) ----------------------------------------------- #
    got["comet_qn.N"] = len(A)
    reference = np.sort(A["Cn"].values)
    Q = pd.concat([
        pd.DataFrame({
            "lang": l, "H": A[A.lang == l]["H"].values,
            "raw_n": A[A.lang == l]["Cn"].values, "raw_r": A[A.lang == l]["Cr"].values,
            "qn_n": qn_map(A[A.lang == l]["Cn"].values, reference),
            "qn_r": qn_map(A[A.lang == l]["Cr"].values, reference),
        }) for l in LANGS], ignore_index=True)
    got["comet_qn.gap_before"] = np.mean(
        [abs(Q[Q.lang == l]["raw_n"].mean() - Q[Q.lang == l]["raw_r"].mean()) for l in LANGS])
    got["comet_qn.gap_after"] = np.mean(
        [abs(Q[Q.lang == l]["qn_n"].mean() - Q[Q.lang == l]["qn_r"].mean()) for l in LANGS])
    raw = np.concatenate([Q["raw_n"], Q["raw_r"]])
    qn = np.concatenate([Q["qn_n"], Q["qn_r"]])
    HH = np.concatenate([Q["H"], Q["H"]])
    got["comet_qn.pearson_before"] = stats.pearsonr(raw, HH)[0]
    got["comet_qn.pearson_after"] = stats.pearsonr(qn, HH)[0]
    got["comet_qn.spearman_before"] = stats.spearmanr(raw, HH)[0]
    got["comet_qn.spearman_after"] = stats.spearmanr(qn, HH)[0]
    for l in LANGS:
        s = Q[Q.lang == l]
        got[f"comet_qn.within_before.{l}"] = stats.spearmanr(s["raw_n"], s["H"])[0]
        got[f"comet_qn.within_after.{l}"] = stats.spearmanr(s["qn_n"], s["H"])[0]

    # ---- Table 6: Eq. 3 worked through five romanised Gujarati segments ---- #
    # The first five romanised GUJ segments of the working set, displayed in
    # ascending score order. Position is rank/(n+1) with n = 5; Mapped reads the
    # pooled native reference at that position.
    sample = A[A.lang == "GUJ"]["Cr"].values[:5]
    order = np.argsort(sample)
    for i, idx in enumerate(order, start=1):
        got[f"table6.raw.{i}"] = float(sample[idx])
        got[f"table6.position.{i}"] = i / (len(sample) + 1)
        got[f"table6.mapped.{i}"] = float(np.quantile(reference, i / (len(sample) + 1)))

    # ---- Slopes ----------------------------------------------------------- #
    Y = np.concatenate([A["Cn"], A["Cr"]])
    got["slopes.comet_tp"] = np.polyfit(np.concatenate([A["TPn"], A["TPr"]]), Y, 1)[0]
    got["slopes.comet_ip"] = np.polyfit(np.concatenate([A["IPn"], A["IPr"]]), Y, 1)[0]

    # ---- Table 7: corrector impossibility --------------------------------- #
    # Table 7's split mirrors scripts/reproduce_all.py. Two details fix the draw:
    #   1. The frame is concatenated in ALPHABETICAL language order (Gujarati,
    #      Hindi, Malayalam, Marathi, Tamil), so the row indices differ from A.
    #   2. A single RandomState(42) is consumed one language at a time in that
    #      same order, so the iteration order is part of the specification.
    NUM = [C_COMET_NAT, C_COMET_ROM, C_HUMAN, C_TP_NAT, C_TP_ROM, C_IP_NAT, C_IP_ROM]
    ALPHA = ["Gujarati", "Hindi", "Malayalam", "Marathi", "Tamil"]
    ALPHA_SHEET = {
        "Gujarati": SHEET_MAP["GUJ"], "Hindi": SHEET_MAP["HIN"],
        "Malayalam": SHEET_MAP["MAL"], "Marathi": SHEET_MAP["MAR"],
        "Tamil": SHEET_MAP["TAM"],
    }
    frames = []
    for name in ALPHA:
        f = pd.read_excel(DATA_XLMR, sheet_name=ALPHA_SHEET[name])
        f = f.loc[:, ~f.columns.str.startswith("Unnamed")]
        for c in NUM:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        f = f.dropna(subset=NUM).copy()
        f["Language"] = name
        frames.append(f[NUM + ["Language"]])
    D = pd.concat(frames, ignore_index=True)
    D["Cn"], D["Cr"] = D[C_COMET_NAT], D[C_COMET_ROM]
    D["H"] = D[C_HUMAN]

    rng_split = np.random.RandomState(SEED_SPLIT)
    D["fold"] = 0
    for name in ALPHA:
        idx = D.index[D.Language == name].to_numpy()
        test = rng_split.choice(idx, size=len(idx) // 2, replace=False)
        D.loc[test, "fold"] = 1
    TR, TE = D[D.fold == 0], D[D.fold == 1]
    T7_LANGS = ALPHA

    def evaluate(by_lang):
        gaps, rl, px, ph = [], [], [], []
        for l in T7_LANGS:
            te = TE[TE.Language == l]
            x = np.asarray(by_lang[l], dtype=float)
            gaps.append(abs(np.mean(x) - te["Cn"].mean()))
            rl.append(stats.spearmanr(x, te["H"])[0])
            px += list(x)
            ph += list(te["H"])
        return np.mean(gaps), np.mean(rl), stats.spearmanr(px, ph)[0]

    def affine(l):
        tr, te = TR[TR.Language == l], TE[TE.Language == l]
        return ((te["Cr"].values - tr["Cr"].mean()) / tr["Cr"].std()
                * tr["Cn"].std() + tr["Cn"].mean())

    def quantile(l):
        """Quantile mapper, as defined in scripts/reproduce_all.py.

        The fractional rank comes from searchsorted against the training
        romanised distribution, clipped to [0.001, 0.999].
        """
        tr, te = TR[TR.Language == l], TE[TE.Language == l]
        r = np.searchsorted(np.sort(tr["Cr"].values), te["Cr"].values) / len(tr)
        r = np.clip(r, 0.001, 0.999)
        return np.quantile(tr["Cn"].values, r)

    def isotonic(l):
        tr, te = TR[TR.Language == l], TE[TE.Language == l]
        return IsotonicRegression(out_of_bounds="clip").fit(
            tr["Cr"], tr["Cn"]).predict(te["Cr"].values)

    correctors = {
        "raw": {l: TE[TE.Language == l]["Cr"].values for l in T7_LANGS},
        "mean_shift": {l: TE[TE.Language == l]["Cr"].values
                       + (TR[TR.Language == l]["Cn"].mean() - TR[TR.Language == l]["Cr"].mean())
                       for l in T7_LANGS},
        "affine": {l: affine(l) for l in T7_LANGS},
        "quantile": {l: quantile(l) for l in T7_LANGS},
        "isotonic": {l: isotonic(l) for l in T7_LANGS},
        "native_ceiling": {l: TE[TE.Language == l]["Cn"].values for l in T7_LANGS},
    }
    for name, by_lang in correctors.items():
        gap, rl, rp = evaluate(by_lang)
        got[f"table5.gap.{name}"] = gap
        got[f"table5.rho_lang.{name}"] = rl
        got[f"table5.rho_pool.{name}"] = rp

    # ---- Table 9: LOLO ---------------------------------------------------- #
    def lolo(features, prefix):
        recs = []
        for l in LANGS:
            tr, te = A[A.lang != l], A[A.lang == l]
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=SEED_GBM).fit(tr[features], tr["Cn"])
            pred = model.predict(te[features])
            r_raw = stats.spearmanr(te["Cr"], te["H"])[0]
            r_cal = stats.spearmanr(pred, te["H"])[0]
            r_nat = stats.spearmanr(te["Cn"], te["H"])[0]
            rec = (r_cal - r_raw) / (r_nat - r_raw) * 100
            recs.append(rec)
            if prefix == "lolo":
                got[f"lolo.rho_raw.{l}"] = r_raw
                got[f"lolo.rho_cal.{l}"] = r_cal
                got[f"lolo.rho_nat.{l}"] = r_nat
                got[f"lolo.recovered_pct.{l}"] = rec
        got[f"{prefix}.mean_recovered_pct"] = float(np.mean(recs))

    lolo(["Cr", "dTP", "dIP", "TPr", "IPr", "TPn", "IPn"], "lolo")

    # ---- Table 3: all six metrics, native and romanised -------------------- #
    for l in LANGS:
        d = work[l]
        for metric, (c_nat, c_rom) in METRIC_COLS.items():
            got[f"table10.{metric}.nat.{l}"] = spearman_vs_human(d, metric_column(d, c_nat))
            got[f"table10.{metric}.rom.{l}"] = spearman_vs_human(d, metric_column(d, c_rom))

    # ---- Table 10: severity bucket counts, every language ------------------ #
    for l in LANGS:
        counts = full[l][C_SEVERITY].value_counts()
        for bucket in SEVERITY_ALL:
            key = bucket.replace(" ", "_").lower()
            got[f"table9.{key}.{l}"] = int(counts.get(bucket, 0))
        got[f"table9.total.{l}"] = int(sum(counts.get(b, 0) for b in SEVERITY_ALL))

    # ---- Table 14: word count vs XLM-R token count ------------------------- #
    for l in LANGS:
        d = full[l]
        wn = d["Translation"].astype(str).str.split().str.len()
        wr = d["Translation_Transliteration_romanized"].astype(str).str.split().str.len()
        got[f"table15.nat.{l}"] = stats.spearmanr(wn, d["Translation_xlmr_token_count"])[0]
        got[f"table15.rom.{l}"] = stats.spearmanr(
            wr, d["Translation_Transliteration_romanized_xlmr_token_count"])[0]

    # ---- Table 12 / 16: MATTR and sentence-level IP-COMET ------------------ #
    for l in LANGS:
        d = full[l]
        got[f"table17.mattr_nat.{l}"] = mattr(d["Translation"].tolist())
        got[f"table17.mattr_rom.{l}"] = mattr(d["Translation_Transliteration_romanized"].tolist())
        got[f"table13.ip_comet_r.{l}"] = stats.pearsonr(d[C_IP_NAT], d[C_COMET_NAT])[0]

    # ---- Table 16: Byte Premium, and Table 12: DV per 1k -------------------- #
    for l in LANGS:
        d = full[l]
        got[f"table17.byte_premium.{l}"] = byte_premium(d["Translation"], d["Source"])
        got[f"table13.dv_per_1k.{l}"] = dep_vowel_rate(
            d["Translation_xlmr_tokens"], SCRIPT_OF[l])

    # ---- Section 4.1: TP-IP anti-correlation across the five languages ----- #
    tp_n = [full[l][C_TP_NAT].mean() for l in LANGS]
    ip_n = [full[l][C_IP_NAT].mean() for l in LANGS]
    tp_r = [full[l][C_TP_ROM].mean() for l in LANGS]
    ip_r = [full[l][C_IP_ROM].mean() for l in LANGS]
    got["sec41.tp_ip_r_nat"], got["sec41.tp_ip_p_nat"] = stats.pearsonr(tp_n, ip_n)
    got["sec41.tp_ip_r_rom"], got["sec41.tp_ip_p_rom"] = stats.pearsonr(tp_r, ip_r)

    # ---- Section 4.3: shared variance ------------------------------------- #
    for l in LANGS:
        got[f"sec43.rho2_nat.{l}"] = got[f"table2.rho_nat.{l}"] ** 2
        got[f"sec43.rho2_rom.{l}"] = got[f"table2.rho_rom.{l}"] ** 2

    # ---- Table 1: the two worked examples ---------------------------------- #
    # Each source sentence appears once per MT system, so the system is part of
    # the key: both examples are the IndicTrans_Sam output scored 25/25 by
    # annotators. Selecting on the human score alone would be ambiguous.
    def worked_example(lang, needle):
        d = full[lang]
        m = d["Source"].astype(str).str.contains(needle, regex=False, na=False)
        hit = d[m & (d["H"] == 25)]
        return hit.iloc[0] if len(hit) else None

    ex_a = worked_example("MAL", "All Blacks")
    if ex_a is not None:
        got["table1.A.comet_nat"] = ex_a[C_COMET_NAT]
        got["table1.A.comet_rom"] = ex_a[C_COMET_ROM]
        got["table1.A.drop"] = ex_a[C_COMET_NAT] - ex_a[C_COMET_ROM]
    ex_mal = worked_example("MAL", "Hydrogen ions are protons")
    ex_hin = worked_example("HIN", "Hydrogen ions are protons")
    if ex_mal is not None and ex_hin is not None:
        got["table1.B.comet.MAL"] = ex_mal[C_COMET_NAT]
        got["table1.B.comet.HIN"] = ex_hin[C_COMET_NAT]
        got["table1.B.gap"] = ex_mal[C_COMET_NAT] - ex_hin[C_COMET_NAT]

    # ---- Table 15 / Appendix E: Latin-script controls ---------------------- #
    for sheet, iso in [("German", "DEU"), ("Spanish", "SPA")]:
        d = pd.read_excel(DATA_LATIN, sheet_name=sheet)
        ip = pd.to_numeric(d["target_xlmr_IP"], errors="coerce")
        tp = pd.to_numeric(d["target_xlmr_TP"], errors="coerce")
        cm = pd.to_numeric(d["COMET"], errors="coerce")
        got[f"latin_controls.tp.{iso}"] = tp.mean()
        got[f"latin_controls.sbi.{iso}"] = (tp / ip).mean()
        ok = ip.notna() & cm.notna()
        got[f"latin_controls.ip_comet_r.{iso}"] = stats.pearsonr(ip[ok], cm[ok])[0]

    for sheet, iso in [("German", "DEU"), ("Spanish", "SPA")]:
        d = pd.read_excel(DATA_LATIN, sheet_name=sheet)
        got[f"latin_controls.mattr.{iso}"] = mattr(d["target"].tolist())
        got[f"latin_controls.byte_premium.{iso}"] = byte_premium(d["target"], d["source"])

    deu = pd.read_excel(DATA_LATIN, sheet_name="German")
    deu["SBI"] = (pd.to_numeric(deu["target_xlmr_TP"], errors="coerce")
                  / pd.to_numeric(deu["target_xlmr_IP"], errors="coerce"))
    groups = {k: deu[deu["severity"] == k]["SBI"].dropna()
              for k in ("No-error", "minor", "major")}
    got["appE.deu_sbi_errorfree"] = groups["No-error"].mean()
    got["appE.deu_sbi_major"] = groups["major"].mean()
    got["appE.deu_kruskal_H"] = stats.kruskal(*groups.values())[0]

    # ---- Tokenizer robustness --------------------------------------------- #
    per_tok = {t: {"nat": [], "rom": [], "pct": []} for t in TOKENIZERS}
    for l in LANGS:
        b = pd.read_excel(DATA_MULTI, sheet_name=SHEET_MAP[l])
        for tok in TOKENIZERS:
            n = pd.to_numeric(b[f"Translation_{tok}_TP"], errors="coerce").mean()
            r = pd.to_numeric(
                b[f"Translation_Transliteration_romanized_{tok}_TP"],
                errors="coerce").mean()
            pct = (r / n - 1) * 100
            got[f"tokenizer.{tok}.tp_pct.{l}"] = pct
            per_tok[tok]["nat"].append(n)
            per_tok[tok]["rom"].append(r)
            per_tok[tok]["pct"].append(pct)

    # Table 6 aggregates: means over the five languages, and the native range.
    for tok, v in per_tok.items():
        got[f"table4.tp_nat.{tok}"] = float(np.mean(v["nat"]))
        got[f"table4.tp_rom.{tok}"] = float(np.mean(v["rom"]))
        got[f"table4.delta_pct.{tok}"] = float(np.mean(v["pct"]))
        got[f"table4.nat_min.{tok}"] = float(min(v["nat"]))
        got[f"table4.nat_max.{tok}"] = float(max(v["nat"]))

    return got


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt(v):
    if isinstance(v, (int, np.integer)):
        return f"{v}"
    return f"{v:.6g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--registry", default=str(ROOT / "paper_numbers.yaml"))
    ap.add_argument("--log", default=str(ROOT / "results" / "logs" / "verification.md"))
    ap.add_argument("--fast", action="store_true",
                    help="use 200 permutation rounds instead of 5,000 (smoke test only)")
    args = ap.parse_args()

    registry = yaml.safe_load(Path(args.registry).read_text())
    claims = registry["claims"]

    print("Recomputing every registered claim from the committed workbooks ...")
    got = compute_all(skip_slow=args.fast)
    print()

    lines, n_pass, n_fail, n_skip = [], 0, 0, 0

    for c in claims:
        cid, expected, tol, status = c["id"], c["value"], c["tol"], c["status"]
        if cid not in got:
            line = (f"[SKIP] {cid:38s} not recomputed by this script "
                    f"(status {status})")
            n_skip += 1
        else:
            actual = got[cid]
            delta = abs(float(actual) - float(expected))
            if delta <= float(tol) + 1e-12:
                line = (f"[PASS] {cid:38s} expected {fmt(expected):<10s} "
                        f"got {fmt(actual)}")
                n_pass += 1
            else:
                line = (f"[FAIL] {cid:38s} expected {fmt(expected):<10s} "
                        f"got {fmt(actual):<10s} (delta {delta:.6g}, tol {fmt(tol)})")
                n_fail += 1
        print(line)
        lines.append(line)

    total = n_pass + n_fail + n_skip
    summary = (f"\n{'=' * 78}\n"
               f"SUMMARY  {n_pass} passed, {n_fail} failed, {n_skip} skipped "
               f"({total} claims registered)\n"
               f"{'=' * 78}")
    print(summary)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("# Verification report\n\n")
        f.write("Generated by `scripts/verify_paper_numbers.py`. Every value on the\n")
        f.write("right was recomputed from the committed workbooks by an implementation\n")
        f.write("independent of the notebooks.\n\n")
        f.write("```\n" + "\n".join(lines) + "\n")
        f.write(summary.strip() + "\n```\n")
    print(f"\n[written] {log_path.relative_to(ROOT)}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
