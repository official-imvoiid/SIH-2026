"""Recalibrate typology thresholds against real Bitcoin, honestly.

The failure this addresses
--------------------------
``ml/experiments/real_vs_control.py`` showed the rules firing on ~72% of both ransomware
and ordinary money -- no separation at all. The diagnosis was that thresholds tuned against
our own synthetic generator are far too permissive for real Bitcoin, where ordinary wallets
routinely touch hundreds of counterparties.

This module sweeps each threshold against real data and reports what, if anything, actually
separates.

Two methodological fixes over the first experiment
--------------------------------------------------
**1. Cleaner labels.** The first run treated *every entity in the ransomware crawl* as a
positive. That is wrong: crawling one hop out from a ransom wallet also collects the
victims who paid in and the exchanges the money was cashed out to. Most of that
neighbourhood is innocent, so the "positive" class was heavily contaminated and no rule
could have separated it.

Here, positives are **only entities containing a Ransomwhere seed address** -- wallets
independently confirmed to have received ransom payments. Negatives are **only entities
containing a control seed**. Everything else in both crawls is treated as unlabelled and
excluded. Fewer examples, but they mean something.

**2. Held-out evaluation, split by crew.** Thresholds are chosen on a calibration set and
scored on a held-out set containing **ransomware families the calibration never saw**. A
random split would let the same crew's habits appear on both sides and would report tuned
numbers as if they generalised. If the held-out result is much worse than the calibration
result, the thresholds are memorising crews rather than finding laundering.

Scoring metric
--------------
**Youden's J = TPR - FPR**: the share of ransom wallets a rule catches, minus the share of
ordinary wallets it wrongly flags. Insensitive to the artificial 50/50 group sizing, which
precision would not be. J <= 0 means the rule is useless or actively misleading.

Run::

    python -m ml.typology.calibrate --size 500
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import networkx as nx

from ml.config import ARTIFACTS, RANDOM_SEED
from ml.experiments.real_vs_control import build_graph, observed_entities
from ml.ingest.real import (
    crawl,
    fetch_ransomwhere,
    fetch_recent_block_addresses,
    select_seeds,
)
from ml.typology import rules as R

# Threshold grids. Each entry maps a rule to the keyword arguments to sweep and the values
# to try. Ranges deliberately extend well past the synthetic defaults, because the whole
# finding is that the defaults are far too permissive for real data.
GRIDS: dict[str, dict[str, list]] = {
    "detect_fan_in": {"min_counterparties": [10, 15, 20, 30, 40, 60, 90, 130]},
    "detect_fan_out": {"min_counterparties": [10, 20, 30, 50, 80, 120, 180]},
    "detect_rapid_passthrough": {
        "min_fwd_ratio": [0.80, 0.90, 0.95, 0.98, 0.995],
        "max_hold_ts": [0, 1, 2],
    },
    "detect_peel_chain": {
        "min_hops": [2, 3, 4, 6, 8],
        "max_peel_ratio": [0.10, 0.25, 0.40],
    },
    "detect_structuring": {
        "min_transfers": [4, 6, 8, 12, 20],
        "max_cv": [0.03, 0.05, 0.10, 0.20],
    },
    "detect_dormant_burst": {
        "min_gap_ts": [3, 6, 12, 18, 30],
        "min_post_txs": [2, 4, 8],
    },
}


def _param_combos(grid: dict[str, list]) -> Iterable[dict]:
    keys = sorted(grid)
    if not keys:
        yield {}
        return

    def rec(i: int, acc: dict):
        if i == len(keys):
            yield dict(acc)
            return
        for v in grid[keys[i]]:
            acc[keys[i]] = v
            yield from rec(i + 1, acc)

    yield from rec(0, {})


def fire_rate(
    rule: Callable,
    g: nx.DiGraph,
    nodes: Iterable[str],
    params: dict,
    suppress_services: bool = False,
) -> tuple[int, int]:
    """How many of ``nodes`` the rule fires on. Returns ``(n_fired, n_total)``."""
    fired = total = 0
    for node in nodes:
        total += 1
        if suppress_services and R.is_service_like(g, node):
            continue
        try:
            if rule(g, node, **params) is not None:
                fired += 1
        except (KeyError, ValueError, ZeroDivisionError):
            continue
    return fired, total


def youden_j(tp_rate: float, fp_rate: float) -> float:
    return tp_rate - fp_rate


def seed_entities(addr_to_entity: dict[str, str], seeds: Iterable[str]) -> set[str]:
    """Entities containing at least one seed address -- our only trustworthy labels."""
    return {addr_to_entity[a] for a in seeds if a in addr_to_entity}


def split_seeds_by_family(
    seeds: list[dict], holdout_frac: float = 0.4, seed: int = RANDOM_SEED
) -> tuple[list[dict], list[dict]]:
    """Split ransomware seeds so that whole *families* are held out.

    Splitting individual addresses would put the same crew on both sides: their wallets
    share habits, so a threshold fitted to one would trivially fit the other and the
    held-out score would be meaningless.
    """
    rng = random.Random(seed)
    families = sorted({s["family"] for s in seeds})
    rng.shuffle(families)
    n_hold = max(1, int(round(len(families) * holdout_frac)))
    holdout_fams = set(families[:n_hold])
    holdout = [s for s in seeds if s["family"] in holdout_fams]
    calib = [s for s in seeds if s["family"] not in holdout_fams]
    return calib, holdout


def evaluate_set(
    rule: Callable,
    params: dict,
    pos_graph: nx.DiGraph,
    pos_nodes: set[str],
    neg_graph: nx.DiGraph,
    neg_nodes: set[str],
    suppress_services: bool = False,
) -> dict[str, Any]:
    tp, n_pos = fire_rate(rule, pos_graph, pos_nodes, params, suppress_services)
    fp, n_neg = fire_rate(rule, neg_graph, neg_nodes, params, suppress_services)
    tpr = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    return {
        "params": dict(params),
        "suppress_services": suppress_services,
        "tp": tp,
        "n_pos": n_pos,
        "tpr": round(tpr, 4),
        "fp": fp,
        "n_neg": n_neg,
        "fpr": round(fpr, 4),
        "youden_j": round(youden_j(tpr, fpr), 4),
    }


def run(size: int = 500, hops: int = 1, holdout_frac: float = 0.4) -> dict:
    print("=" * 76)
    print("  THRESHOLD RECALIBRATION AGAINST REAL BITCOIN")
    print("=" * 76)

    rng = random.Random(RANDOM_SEED)
    ransomware = fetch_ransomwhere()
    seeds = select_seeds(ransomware, n=max(8, size // 12))

    calib_seeds, holdout_seeds = split_seeds_by_family(seeds, holdout_frac)
    print(f"\n[1/4] {len(seeds)} ransom seeds across "
          f"{len({s['family'] for s in seeds})} families")
    print(f"      calibration: {len(calib_seeds)} seeds / "
          f"{len({s['family'] for s in calib_seeds})} families")
    print(f"      held out:    {len(holdout_seeds)} seeds / "
          f"{len({s['family'] for s in holdout_seeds})} families "
          f"({', '.join(sorted({s['family'] for s in holdout_seeds}))})")

    print(f"\n[2/4] loading crawls (cached) ...")
    rw_records, rw_report = crawl([s["address"] for s in seeds], max_addresses=size, hops=hops)
    control_pool = fetch_recent_block_addresses()
    control_seeds = rng.sample(control_pool, min(len(seeds), len(control_pool)))
    ctl_records, ctl_report = crawl(control_seeds, max_addresses=size, hops=hops)

    rw_graph, rw_map = build_graph(rw_records)
    ctl_graph, ctl_map = build_graph(ctl_records)

    rw_fetched = set(rw_report["fetched_addresses"])
    ctl_fetched = set(ctl_report["fetched_addresses"])

    # Positives: only entities that ARE a confirmed ransom-receiving wallet.
    pos_cal = seed_entities(rw_map, [s["address"] for s in calib_seeds]) & observed_entities(rw_map, rw_fetched)
    pos_hold = seed_entities(rw_map, [s["address"] for s in holdout_seeds]) & observed_entities(rw_map, rw_fetched)

    # Negatives: control seeds, split the same way so both sides are held out together.
    ctl_obs = list(sorted(seed_entities(ctl_map, control_seeds) & observed_entities(ctl_map, ctl_fetched)))
    rng.shuffle(ctl_obs)
    n_hold_neg = max(1, int(round(len(ctl_obs) * holdout_frac)))
    neg_hold, neg_cal = set(ctl_obs[:n_hold_neg]), set(ctl_obs[n_hold_neg:])

    print(f"\n[3/4] labelled entities (seed wallets only, not their neighbourhoods)")
    print(f"      calibration: {len(pos_cal):>3} ransom  vs {len(neg_cal):>3} ordinary")
    print(f"      held out:    {len(pos_hold):>3} ransom  vs {len(neg_hold):>3} ordinary")

    if len(pos_cal) < 3 or len(neg_cal) < 3:
        print("\n  Too few labelled entities to calibrate. Run with a larger --size.")
        return {}

    print(f"\n[4/4] sweeping thresholds ...\n")
    results = {}
    for rule in R.RULES:
        name = rule.__name__
        grid = GRIDS.get(name)
        if grid is None:
            continue

        candidates = []
        service_options = [False, True] if name == "detect_fan_out" else [False]
        for suppress in service_options:
            for params in _param_combos(grid):
                candidates.append(
                    evaluate_set(rule, params, rw_graph, pos_cal, ctl_graph, neg_cal, suppress)
                )

        best = max(candidates, key=lambda c: (c["youden_j"], -sum(
            v if isinstance(v, (int, float)) else 0 for v in c["params"].values())))
        default = evaluate_set(rule, {}, rw_graph, pos_cal, ctl_graph, neg_cal)

        # The number that matters: does the tuned threshold survive unseen crews?
        held = evaluate_set(
            rule, best["params"], rw_graph, pos_hold, ctl_graph, neg_hold,
            best["suppress_services"],
        )

        default_held = evaluate_set(
            rule, {}, rw_graph, pos_hold, ctl_graph, neg_hold
        )
        results[name] = {
            "default_on_calibration": default,
            "best_on_calibration": best,
            "best_on_holdout": held,
            "default_on_holdout": default_held,
            # If the untuned rule scores at least as well on unseen crews, the sweep found
            # nothing real and we keep the simpler, unfitted thresholds.
            "tuning_helped": held["youden_j"] > default_held["youden_j"] + 0.02,
        }

        tag = " +service-guard" if best["suppress_services"] else ""
        print(f"  {name}")
        print(f"    default   J={default['youden_j']:+.3f}  "
              f"(catches {default['tpr']:.0%} of ransom, flags {default['fpr']:.0%} of ordinary)")
        print(f"    tuned     J={best['youden_j']:+.3f}  {best['params']}{tag}")
        print(f"    HELD OUT  tuned   J={held['youden_j']:+.3f}  "
              f"(catches {held['tpr']:.0%} of ransom, flags {held['fpr']:.0%} of ordinary)")
        print(f"    HELD OUT  default J={default_held['youden_j']:+.3f}  "
              f"(catches {default_held['tpr']:.0%}, flags {default_held['fpr']:.0%})"
              + ("   <- tuning did NOT help" if not results[name]["tuning_helped"] else ""))
        print()

    def best_holdout_j(r: dict) -> float:
        return max(r["best_on_holdout"]["youden_j"], r["default_on_holdout"]["youden_j"])

    survivors = {n: r for n, r in results.items() if best_holdout_j(r) > 0.15}

    out = {
        "note": (
            "Positives are entities containing a Ransomwhere-confirmed ransom address. "
            "Negatives are entities containing an address sampled from recent blocks. "
            "Thresholds chosen on calibration families, scored on held-out families."
        ),
        "params": {"size": size, "hops": hops, "holdout_frac": holdout_frac,
                   "seed": RANDOM_SEED},
        "n_labelled": {
            "pos_calibration": len(pos_cal), "neg_calibration": len(neg_cal),
            "pos_holdout": len(pos_hold), "neg_holdout": len(neg_hold),
        },
        "per_rule": results,
        "survivors": sorted(survivors),
        # Adopt tuned thresholds only where tuning demonstrably helped on unseen crews;
        # otherwise keep the defaults, which are simpler and not fitted to this sample.
        "calibrated_thresholds": {
            n: (
                {**r["best_on_calibration"]["params"],
                 "suppress_services": r["best_on_calibration"]["suppress_services"]}
                if r["tuning_helped"]
                else {"_keep_defaults": True}
            )
            for n, r in survivors.items()
        },
    }

    print("=" * 76)
    if survivors:
        print(f"  {len(survivors)} rule(s) separate real ransom money on UNSEEN crews:")
        for n in sorted(survivors):
            r = results[n]
            h = r["best_on_holdout"] if r["tuning_helped"] else r["default_on_holdout"]
            which = "tuned" if r["tuning_helped"] else "default"
            print(f"    {n:28} J={h['youden_j']:+.3f}  "
                  f"TPR {h['tpr']:.0%}  FPR {h['fpr']:.0%}   ({which})")
        print("\n  These thresholds are worth adopting. The rest are not.")
    else:
        print("  NO rule separated real ransom money from ordinary money on held-out")
        print("  crews, at any threshold in the grid. Structure alone is not enough here.")
        print("  Report this as the finding -- it is a real result, not a bug.")
    print("=" * 76)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "calibrated_thresholds.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  written to {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--holdout-frac", type=float, default=0.4)
    args = ap.parse_args()
    run(size=args.size, hops=args.hops, holdout_frac=args.holdout_frac)
