# -*- coding: utf-8 -*-
"""Behavioural tests for the diagnostics: TP, IP, SBI, IPI, LP, EP, Tax.

These test the definitions and their invariants, not the specific values —
the values are covered by test_paper_numbers.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_paper_numbers as V  # noqa: E402

LANGS = V.LANGS


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #
def sbi(tp, ip):
    return np.mean(np.asarray(tp) / np.asarray(ip))


def ipi(ip_mean):
    return abs(ip_mean - 1.0)


def zone(value):
    if value < 0.05:
        return "Parity"
    if value > 0.70:
        return "Paradox"
    return "Burden"


def tax(tp_nat, tp_rom, ip_nat, ip_rom):
    lp = tp_rom / tp_nat
    ep = ip_nat / ip_rom
    return lp, ep, lp * ep


# --------------------------------------------------------------------------- #
# Unit behaviour
# --------------------------------------------------------------------------- #
def test_ipi_is_zero_at_perfect_parity():
    assert ipi(1.0) == 0.0


def test_ipi_is_symmetric_about_parity():
    assert ipi(0.8) == pytest.approx(ipi(1.2))


def test_zone_boundaries():
    assert zone(0.049) == "Parity"
    assert zone(0.05) == "Burden"
    assert zone(0.70) == "Burden"
    assert zone(0.701) == "Paradox"


def test_sbi_is_one_when_tp_equals_ip():
    assert sbi([1.5, 2.0, 3.0], [1.5, 2.0, 3.0]) == pytest.approx(1.0)


def test_sbi_is_a_mean_of_ratios_not_a_ratio_of_means():
    """The two differ; the paper uses the mean of per-sentence ratios."""
    tp, ip = [1.0, 3.0], [1.0, 1.0]
    assert sbi(tp, ip) == pytest.approx(2.0)
    ratio_of_means = np.mean(tp) / np.mean(ip)
    assert ratio_of_means == pytest.approx(2.0)
    tp, ip = [1.0, 4.0], [1.0, 2.0]
    assert sbi(tp, ip) == pytest.approx(1.5)
    assert np.mean(tp) / np.mean(ip) == pytest.approx(5 / 3)
    assert sbi(tp, ip) != pytest.approx(np.mean(tp) / np.mean(ip))


def test_tax_is_one_when_romanisation_changes_nothing():
    lp, ep, t = tax(1.2, 1.2, 0.5, 0.5)
    assert (lp, ep, t) == (1.0, 1.0, 1.0)


def test_tax_factorises_in_log_space():
    lp, ep, t = tax(1.2, 1.8, 0.6, 0.3)
    assert np.log(t) == pytest.approx(np.log(lp) + np.log(ep))


def test_ep_share_of_log_tax_is_a_fraction():
    lp, ep, t = tax(1.2, 1.8, 0.6, 0.3)
    share = np.log(ep) / np.log(t)
    assert 0.0 <= share <= 1.0
    lp_share = np.log(lp) / np.log(t)
    assert share + lp_share == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Behaviour on the real data
# --------------------------------------------------------------------------- #
def test_tp_inflates_and_ip_collapses_for_every_language(sheets):
    full, _ = sheets
    for lang in LANGS:
        d = full[lang]
        assert d[V.C_TP_ROM].mean() > d[V.C_TP_NAT].mean(), f"{lang}: TP did not inflate"
        assert d[V.C_IP_ROM].mean() < d[V.C_IP_NAT].mean(), f"{lang}: IP did not collapse"


def test_every_language_moves_from_burden_to_paradox(sheets):
    full, _ = sheets
    for lang in LANGS:
        d = full[lang]
        assert zone(ipi(d[V.C_IP_NAT].mean())) == "Burden"
        assert zone(ipi(d[V.C_IP_ROM].mean())) == "Paradox"


def test_entropy_penalty_dominates_length_penalty(sheets):
    """More than half the Computational Tax is information loss, not length."""
    full, _ = sheets
    for lang in LANGS:
        d = full[lang]
        lp, ep, t = tax(d[V.C_TP_NAT].mean(), d[V.C_TP_ROM].mean(),
                        d[V.C_IP_NAT].mean(), d[V.C_IP_ROM].mean())
        assert ep > lp, f"{lang}: LP exceeds EP"
        assert np.log(ep) / np.log(t) > 0.5, f"{lang}: EP is not the dominant share"


def test_latin_controls_bracket_the_indic_range(sheets):
    """ENG-SPA anchors Parity; ENG-DEU lands in Burden despite Latin script."""
    import pandas as pd
    full, _ = sheets
    indic = [ipi(full[l][V.C_IP_NAT].mean()) for l in LANGS]
    values = {}
    for sheet, iso in [("German", "DEU"), ("Spanish", "SPA")]:
        ip = pd.to_numeric(
            pd.read_excel(V.DATA_LATIN, sheet_name=sheet)["target_xlmr_IP"],
            errors="coerce")
        values[iso] = ipi(ip.mean())
    assert zone(values["SPA"]) == "Parity"
    assert zone(values["DEU"]) == "Burden"
    assert min(indic) <= values["DEU"] <= max(indic), (
        "ENG-DEU should sit inside the Indic native-script range")
