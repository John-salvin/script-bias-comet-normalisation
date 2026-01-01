# -*- coding: utf-8 -*-
"""Shared fixtures.

``compute_all`` reads three workbooks and fits ten gradient-boosted models, so it
is computed once per session and shared by every test that needs it.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_paper_numbers as V  # noqa: E402


@pytest.fixture(scope="session")
def computed():
    """Every recomputed claim value, keyed by claim id."""
    return V.compute_all()


@pytest.fixture(scope="session")
def registry():
    import yaml
    return yaml.safe_load((ROOT / "paper_numbers.yaml").read_text())


@pytest.fixture(scope="session")
def sheets():
    """(full, work) dicts keyed by ISO code."""
    return V.load()
