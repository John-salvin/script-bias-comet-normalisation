#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_data.py
=============
Confirms the committed workbooks are present and prints their SHA-256 digests,
so that a fork or a partial clone fails loudly rather than silently reproducing
different numbers.

The expected digests are recorded in ``data/README.md``. Passing ``--strict``
compares against them and exits non-zero on any mismatch.

Usage
-----
  python3 scripts/check_data.py
  python3 scripts/check_data.py --strict

Dependencies: none beyond the standard library.
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# SHA-256 of each committed workbook. Regenerate with:
#   python3 scripts/check_data.py
EXPECTED = {
    "data/indic/indic_parity_xlmr.xlsx":
        "fa4a7a7dc64583235159db3c5c292a76b50f61dc0b8f0fac4e30b581ceb2fbc0",
    "data/indic/indic_parity_multi_tokenizer.xlsx":
        "d6a183f45142a2e82f2fb35d9e35fc890fe0f90986f87ea4871facf5b1eef110",
    "data/latin/wmt24_ende_enes_metrics.xlsx":
        "bcf5f584e7fdf19139a6c1700c39484b8aa44bb007ae5e37b0c92a7dba914bcc",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="fail if a digest does not match the recorded value")
    args = ap.parse_args()

    missing, mismatched = [], []
    print(f"{'SHA-256':64s}  {'size':>12s}  file")
    print("-" * 100)
    for rel, expected in EXPECTED.items():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            print(f"{'MISSING':64s}  {'-':>12s}  {rel}")
            continue
        digest = sha256(path)
        print(f"{digest:64s}  {path.stat().st_size:>12,}  {rel}")
        if args.strict and digest != expected:
            mismatched.append((rel, expected, digest))

    if missing:
        print("\nMissing workbook(s):")
        for rel in missing:
            print(f"  {rel}")
        print("\nSee data/README.md for how to obtain them.")
        return 1

    if mismatched:
        print("\nDigest mismatch:")
        for rel, exp, got in mismatched:
            print(f"  {rel}\n    recorded  {exp}\n    on disk   {got}")
        return 1

    print("\nAll workbooks present.")
    if not args.strict:
        print("Run with --strict to compare against the digests recorded in data/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
