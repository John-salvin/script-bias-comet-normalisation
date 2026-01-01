# -*- coding: utf-8 -*-
"""Assert every claim in paper_numbers.yaml.

Claims with status ``verified`` are recomputed independently and asserted.
Claims with status ``script`` are RNG-dependent and are asserted only under the
documented seeds.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_paper_numbers as V  # noqa: E402

REGISTRY = __import__("yaml").safe_load((ROOT / "paper_numbers.yaml").read_text())
CLAIMS = REGISTRY["claims"]

VERIFIED = [c for c in CLAIMS if c["status"] == "verified"]
SCRIPT = [c for c in CLAIMS if c["status"] == "script"]


def _check(claim, computed):
    cid = claim["id"]
    assert cid in computed, f"{cid} is registered but not recomputed by the verifier"
    actual = float(computed[cid])
    expected = float(claim["value"])
    tol = float(claim["tol"])
    delta = abs(actual - expected)
    assert delta <= tol + 1e-12, (
        f"{cid}: expected {expected}, got {actual} (delta {delta:.6g}, tol {tol})"
    )


@pytest.mark.parametrize("claim", VERIFIED, ids=lambda c: c["id"])
def test_verified_claim(claim, computed):
    """Recomputed from the committed data; a mismatch is a real failure."""
    _check(claim, computed)


@pytest.mark.parametrize("claim", SCRIPT, ids=lambda c: c["id"])
def test_script_claim(claim, computed):
    """RNG-dependent. Reproducible only under the seeds recorded in meta.seeds."""
    _check(claim, computed)


# --------------------------------------------------------------------------- #
# Registry hygiene
# --------------------------------------------------------------------------- #
def test_registry_ids_are_unique():
    ids = [c["id"] for c in CLAIMS]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate claim ids: {sorted(duplicates)}"


def test_every_claim_has_a_source_notebook():
    for c in CLAIMS:
        assert (ROOT / c["source"]).exists(), f"{c['id']} points at a missing {c['source']}"


def test_language_order_is_the_paper_order():
    assert REGISTRY["meta"]["language_order"] == ["GUJ", "TAM", "MAL", "MAR", "HIN"]
    assert V.LANGS == ["GUJ", "TAM", "MAL", "MAR", "HIN"]


def test_the_three_sample_size_bases():
    bases = REGISTRY["meta"]["bases"]
    assert bases["full"] == 7000
    assert bases["working"] == 6995
    assert bases["mar_severity"] == 1258


# --------------------------------------------------------------------------- #
# Headline quantities, pinned explicitly so a regression is loud rather than
# subtle.
# --------------------------------------------------------------------------- #
def test_hin_guj_welch_t(computed):
    assert abs(computed["hin_guj.welch_t_native"] - 27.4) < 0.05
    assert abs(computed["hin_guj.welch_t_romanised"] - 6.97) < 0.05


def test_lolo_mean_recovery(computed):
    mean = computed["lolo.mean_recovered_pct"]
    assert abs(mean - 17.1) < 0.1, f"mean LOLO recovery must be 17.1%, got {mean:.1f}%"


def test_anova_f_on_the_full_base(computed):
    assert abs(computed["anova.native.F_7000"] - 2080.8) < 0.1


def test_anova_reported_on_both_bases(computed):
    """The ANOVA is reported on both the 7,000 and 6,995 bases.

    COMET is complete, so the 7,000-segment base is the applicable one: the five
    segments lacking a human score are irrelevant to a test that does not use the
    human score. Both values are pinned here.
    """
    assert abs(computed["anova.native.F_7000"] - 2080.8) < 0.1
    assert abs(computed["anova.native.F_6995"] - 2080.2) < 0.1
