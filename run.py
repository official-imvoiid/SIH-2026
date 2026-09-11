#!/usr/bin/env python
"""One command to get ChainTrace running from a fresh clone.

    python run.py

Builds the dataset if it is missing, trains the model if it is missing, then serves the
API and the investigator UI at http://localhost:8000.

Everything runs offline. There is no download, no API key, and no network call at any
point -- which is a hard requirement for the sponsoring organisation and also removes the
most common way a live demo dies.

Useful flags::

    python run.py --rebuild     regenerate data and retrain from scratch
    python run.py --no-serve    build and train only
    python run.py --port 8080   serve somewhere else
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_DATA = ROOT / "data" / "processed" / "entities.parquet"
REQUIRED_MODEL = ROOT / "ml" / "artifacts" / "model.joblib"

BANNER = r"""
   ___ _         _    _____
  / __| |_  __ _(_)_ |_   _| _ __ _ __ ___
 | (__| ' \/ _` | | ' \| || '_/ _` / _/ -_)
  \___|_||_\__,_|_|_||_|_||_| \__,_\__\___|

  Bitcoin entity forensics - offline, explainable
"""


def _check_deps() -> list[str]:
    missing = []
    for mod, pkg in [
        ("pandas", "pandas"), ("numpy", "numpy"), ("pyarrow", "pyarrow"),
        ("sklearn", "scikit-learn"), ("joblib", "joblib"),
        ("networkx", "networkx"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def _step(label: str, module: str) -> None:
    print(f"\n>> {label}")
    print("-" * 70)
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n!! {module} failed. Fix the error above and re-run.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="regenerate data and retrain")
    ap.add_argument("--no-serve", action="store_true", help="build and train only")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    print(BANNER)

    missing = _check_deps()
    if missing:
        sys.exit(
            "Missing dependencies: " + ", ".join(missing)
            + "\n\nInstall them with:\n    pip install -r requirements.txt"
        )

    t0 = time.time()
    if args.rebuild or not REQUIRED_DATA.exists():
        _step("Building entity graph (synthetic world, no download)", "ml.ingest.build")
    else:
        print(f"\n>> Entity graph already built ({REQUIRED_DATA.name}); --rebuild to redo")

    if args.rebuild or not REQUIRED_MODEL.exists():
        _step("Training risk model", "ml.train.baseline")
    else:
        print(f">> Model already trained ({REQUIRED_MODEL.name}); --rebuild to redo")

    print(f"\n   setup complete in {time.time()-t0:.1f}s")

    if args.no_serve:
        return

    print("\n" + "=" * 70)
    print(f"   Investigator UI   http://{args.host}:{args.port}")
    print(f"   API browser       http://{args.host}:{args.port}/docs")
    print("   Ctrl-C to stop")
    print("=" * 70 + "\n")

    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
