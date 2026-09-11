"""Fetch and verify the datasets this project uses.

What it can and cannot do
-------------------------
Two of the datasets are behind logins and **cannot** be downloaded by a script:

  * **Elliptic** -- Kaggle, needs an account and accepted terms
  * **Elliptic++** -- a Google Drive folder, needs a browser

The rest are public and this script fetches them automatically. For the two that are not,
it prints exact instructions and then *verifies whatever you downloaded* -- checking row
counts, headers and class coding, so a wrong or truncated file is caught here rather than
three modules downstream.

Run::

    python scripts/fetch_data.py            # fetch public data, verify everything present
    python scripts/fetch_data.py --verify   # verification only, no network
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import DATA_RAW  # noqa: E402
from ml.ingest.elliptic import (  # noqa: E402
    EXPECTED_ELLIPTIC_ROWS,
    N_ELLIPTIC_FEATURE_COLS,
    SchemaError,
    describe_capabilities,
    find_dataset,
    load_elliptic,
    load_elliptic_plus,
)

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, WARN = f"{GREEN}ok{RESET}", f"{RED}missing{RESET}", f"{YELLOW}warn{RESET}"


def fetch_public() -> dict[str, bool]:
    """Fetch the datasets that need no login. Safe to re-run; skips what exists."""
    from ml.ingest.real import fetch_ofac, fetch_ransomwhere, fetch_recent_block_addresses

    results = {}
    for name, fn in (
        ("Ransomwhere (ransom payment addresses)", fetch_ransomwhere),
        ("OFAC sanctioned addresses", fetch_ofac),
        ("Recent-block control addresses", fetch_recent_block_addresses),
    ):
        try:
            data = fn()
            n = len(data)
            print(f"  [{OK}] {name}: {n:,} records")
            results[name] = True
        except Exception as e:  # network is the normal failure here, not a bug
            print(f"  [{FAIL}] {name}: {type(e).__name__}: {str(e)[:70]}")
            results[name] = False
    return results


def verify_real_dataset(raw_dir: Path) -> dict:
    """Check whichever real dataset is present and report exactly what it supports."""
    paths = find_dataset(raw_dir)
    if paths is None:
        print(f"  [{FAIL}] No real dataset found in {raw_dir}")
        return {"present": False}

    print(f"  [{OK}] Found {paths.kind} at {paths.root}")
    for k, v in paths.files.items():
        size = Path(v).stat().st_size / 1e6
        print(f"        {DIM}{k:10} {size:8.1f} MB  {Path(v).name}{RESET}")

    try:
        if paths.kind == "elliptic_plus":
            entities, edges, meta = load_elliptic_plus(paths)
        else:
            entities, edges, meta = load_elliptic(paths)
    except SchemaError as e:
        print(f"  [{RED}schema error{RESET}] {e}")
        return {"present": True, "valid": False, "error": str(e)}

    counts = entities["label"].value_counts().to_dict()
    total = len(entities)
    print(f"  [{OK}] Loaded {total:,} entities, {len(edges):,} edges")
    print(f"        {DIM}labels: " + "  ".join(
        f"{k}={v:,} ({100*v/total:.1f}%)" for k, v in counts.items()) + RESET)

    # These are the numbers that end up on a slide, so check them rather than trusting a
    # paper abstract -- the dataset has been revised since publication.
    if paths.kind == "elliptic":
        if total != EXPECTED_ELLIPTIC_ROWS:
            print(f"  [{WARN}] Expected {EXPECTED_ELLIPTIC_ROWS:,} transactions, got "
                  f"{total:,}. Quote the number printed here, not the published one.")
        illicit_pct = 100 * counts.get("illicit", 0) / total
        if not (1.0 <= illicit_pct <= 4.0):
            print(f"  [{WARN}] Illicit share is {illicit_pct:.2f}%, outside the expected "
                  f"~2%. Check the class column coding (1=illicit, 2=licit).")

    caps = meta["capabilities"]
    print(f"\n  {caps['dataset']}")
    print(f"    unit ................ {caps['unit']}")
    print(f"    entity resolution ... {GREEN if caps['entity_resolution'] else YELLOW}"
          f"{caps['entity_resolution']}{RESET}")
    print(f"    typology rules ...... {caps.get('typology_rules', '?')}")
    for c in caps.get("caveats", []):
        print(f"    {DIM}caveat: {c}{RESET}")

    return {"present": True, "valid": True, "kind": paths.kind, "n_entities": total,
            "labels": counts, "capabilities": caps}


INSTRUCTIONS = f"""
{YELLOW}These two need a login, so fetch them by hand:{RESET}

  {GREEN}Elliptic++{RESET}  (preferred -- has wallet addresses, supports entity resolution)
    1. Open https://github.com/git-disl/EllipticPlusPlus
    2. Follow the Google Drive link in its README
    3. Download the "Actors Dataset" CSVs
    4. Put them in  data/raw/elliptic_plus/

  {GREEN}Elliptic{RESET}  (the literature benchmark -- transactions only, no addresses)
    1. Open https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
    2. Sign in, accept the terms, download
    3. Put these three in  data/raw/elliptic/ :
         elliptic_txs_features.csv
         elliptic_txs_classes.csv
         elliptic_txs_edgelist.csv

Then run this script again to verify, and build with:
    python -m ml.ingest.build --real && python -m ml.train.baseline

{DIM}Licence note: Elliptic is CC BY-NC-SA 4.0 (non-commercial). Put that on the data slide.{RESET}
""".strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="verify only, no network")
    args = ap.parse_args()

    print("=" * 74)
    print("  ChainTrace dataset fetch & verify")
    print("=" * 74)

    if not args.verify:
        print("\nPublic datasets (no login needed):")
        fetch_public()

    print("\nReal labelled dataset:")
    result = verify_real_dataset(DATA_RAW)

    if not result.get("valid"):
        print("\n" + INSTRUCTIONS)
        print(f"\n{DIM}The project runs fully without these -- `python run.py` uses a "
              f"synthetic\nBitcoin world. But any number you report from synthetic data "
              f"must say so.{RESET}")
        return 1

    print(f"\n{GREEN}Ready.{RESET} Build with:  python -m ml.ingest.build --real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
