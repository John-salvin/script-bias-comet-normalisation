# -*- coding: utf-8 -*-
"""Behavioural tests for COMET-QN and the corrector classes.

The load-bearing property throughout is that Spearman rho is invariant under any
strictly monotone transform. It is what makes post-hoc correction useless for
ranking (notebook 05) and what makes quantile renormalisation safe (notebook 06).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_paper_numbers as V  # noqa: E402

LANGS = V.LANGS
qn_map = V.qn_map


# --------------------------------------------------------------------------- #
# The renormalisation map
# --------------------------------------------------------------------------- #
def test_qn_maps_onto_the_reference_range():
    rng = np.random.default_rng(0)
    scores = rng.normal(50, 10, 500)
    reference = np.sort(rng.normal(80, 5, 2000))
    out = qn_map(scores, reference)
    assert out.min() >= reference.min()
    assert out.max() <= reference.max()


def test_qn_preserves_ordering_exactly():
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 1, 300)
    reference = np.sort(rng.normal(10, 3, 1000))
    out = qn_map(scores, reference)
    assert np.array_equal(stats.rankdata(scores), stats.rankdata(out))


def test_qn_is_idempotent_against_its_own_reference():
    rng = np.random.default_rng(2)
    scores = rng.normal(0, 1, 400)
    reference = np.sort(scores)
    once = qn_map(scores, reference)
    twice = qn_map(once, reference)
    assert stats.spearmanr(once, twice)[0] == pytest.approx(1.0)


def test_qn_leaves_spearman_untouched():
    rng = np.random.default_rng(3)
    human = rng.normal(0, 1, 500)
    scores = human * 0.6 + rng.normal(0, 1, 500)
    reference = np.sort(rng.normal(70, 12, 3000))
    before = stats.spearmanr(scores, human)[0]
    after = stats.spearmanr(qn_map(scores, reference), human)[0]
    assert after == pytest.approx(before, abs=1e-12)


# --------------------------------------------------------------------------- #
# Monotone correctors cannot change within-language ranking
# --------------------------------------------------------------------------- #
def _correctors(train_x, train_y, test_x):
    shift = train_y.mean() - train_x.mean()
    affine = ((test_x - train_x.mean()) / train_x.std() * train_y.std()
              + train_y.mean())
    quantile = qn_map(test_x, np.sort(train_y))
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        train_x, train_y).predict(test_x)
    return {
        "mean_shift": test_x + shift,
        "affine": affine,
        "quantile": quantile,
        "isotonic": isotonic,
    }


def test_every_corrector_is_rank_preserving_on_synthetic_data():
    rng = np.random.default_rng(4)
    train_x = rng.normal(60, 8, 700)
    train_y = train_x * 1.1 + 12 + rng.normal(0, 2, 700)
    test_x = rng.normal(60, 8, 700)
    human = test_x * 0.5 + rng.normal(0, 5, 700)

    baseline = stats.spearmanr(test_x, human)[0]
    for name, corrected in _correctors(train_x, train_y, test_x).items():
        rho = stats.spearmanr(corrected, human)[0]
        # Isotonic regression can tie values, which perturbs rho very slightly.
        assert rho == pytest.approx(baseline, abs=5e-3), f"{name} changed the ranking"


def test_correctors_close_the_gap_they_are_meant_to_close():
    rng = np.random.default_rng(5)
    train_x = rng.normal(60, 8, 700)
    train_y = train_x + 15 + rng.normal(0, 2, 700)
    test_x = rng.normal(60, 8, 700)

    raw_gap = abs(test_x.mean() - train_y.mean())
    assert raw_gap > 10
    for name, corrected in _correctors(train_x, train_y, test_x).items():
        assert abs(corrected.mean() - train_y.mean()) < 1.5, f"{name} left a gap"


# --------------------------------------------------------------------------- #
# On the real data
# --------------------------------------------------------------------------- #
def test_comet_qn_does_not_move_within_language_correlation(computed):
    for lang in LANGS:
        before = computed[f"comet_qn.within_before.{lang}"]
        after = computed[f"comet_qn.within_after.{lang}"]
        assert after == pytest.approx(before, abs=1e-9), (
            f"{lang}: COMET-QN changed within-language rho")


def test_comet_qn_removes_the_cross_language_gap(computed):
    assert computed["comet_qn.gap_before"] > 8.0
    assert computed["comet_qn.gap_after"] < 0.01


def test_comet_qn_improves_pooled_correlation(computed):
    assert computed["comet_qn.spearman_after"] > computed["comet_qn.spearman_before"]
    assert computed["comet_qn.pearson_after"] > computed["comet_qn.pearson_before"]


def test_rho_lang_is_frozen_across_monotone_correctors(computed):
    """The impossibility result, on the real data."""
    monotone = ["raw", "mean_shift", "affine", "quantile", "isotonic"]
    values = [computed[f"table5.rho_lang.{k}"] for k in monotone]
    assert max(values) - min(values) < 0.005, (
        f"rho_lang varied across monotone correctors: {values}")

    ceiling = computed["table5.rho_lang.native_ceiling"]
    assert ceiling - max(values) > 0.2, (
        "the native ceiling should be far above every corrector")


def test_correctors_close_the_gap_on_the_real_data(computed):
    # Table 5's reported raw gap, pinned exactly rather than by a loose bound.
    assert computed["table5.gap.raw"] == pytest.approx(7.88, abs=0.005)
    for k in ["mean_shift", "affine", "quantile", "isotonic"]:
        assert computed[f"table5.gap.{k}"] < 1.0, f"{k} left a gap"


def test_parity_features_recover_only_a_fraction_of_the_native_signal(computed):
    """LOLO calibration helps a little on average and hurts Marathi outright."""
    assert 0 < computed["lolo.mean_recovered_pct"] < 25
    assert computed["lolo.recovered_pct.MAR"] < 0
