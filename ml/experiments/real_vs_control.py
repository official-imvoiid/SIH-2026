"""The experiment that makes this project's claims non-circular.

The problem it solves
---------------------
The synthetic world in ``ml/data/synth.py`` has laundering patterns planted in it by hand.
Running our detectors over it and reporting that they fire is close to a tautology: we
wrote the generator, then wrote a detector for what the generator makes. Any precision
figure from that setup is unearned.

This experiment removes the generator entirely.

Design
------
Two groups, drawn from the **real Bitcoin blockchain**, treated identically:

* **Ransomware group** -- seeded from addresses in the Ransomwhere dataset that really
  received ransom payments, labelled with the family that collected them.
* **Control group** -- seeded from addresses appearing in recent Bitcoin blocks: ordinary
  users, exchanges, merchants, miners going about their business.

Both groups get the same crawl depth, the same address budget, the same clustering, and
the same typology rules. The only difference is where the seeds came from. So if the rules
fire substantially more often on the ransomware group, that difference is a property of
criminal money, not of our code.

What a negative result would mean
---------------------------------
If the rules fire at the same rate on both groups, the rule engine does not work on real
data and we should say so. That would be a genuine and publishable finding, and a far
better submission than a fabricated success. The script reports whatever it finds.

Run::

    python -m ml.experiments.real_vs_control                 # default: 120 addresses/group
    python -m ml.experiments.real_vs_control --size 300      # bigger, slower, better

First run fetches from the network and caches to ``data/raw/``; later runs are offline.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import networkx as nx

from ml.config import DATA_PROCESSED, RANDOM_SEED
from ml.features.clustering import cluster_addresses
from ml.features.graph import build_entity_graph
from ml.ingest.real import (
    crawl,
    fetch_ofac,
    fetch_ransomwhere,
    fetch_recent_block_addresses,
    select_seeds,
)
from ml.typology.rules import RULES, detect_all, structural


def observed_entities(addr_to_entity: dict[str, str], fetched: set[str]) -> set[str]:
    """Entities containing at least one address whose transaction history we fetched."""
    return {addr_to_entity[a] for a in fetched if a in addr_to_entity}


def build_graph(records: list) -> tuple[nx.DiGraph, dict[str, str]]:
    """Cluster real addresses, then build the real entity money-flow graph."""
    addr_to_entity = cluster_addresses(
        (r[0], r[1], r[2]) for r in records
    )
    next_id = len(set(addr_to_entity.values())) + 1
    for _txid, _ins, _outs, _ts, outputs in records:
        for addr, _v in outputs:
            if addr not in addr_to_entity:
                addr_to_entity[addr] = f"E-{next_id:05d}"
                next_id += 1

    edges = []
    for _txid, ins, _outs, ts, outputs in records:
        if not ins:
            continue
        src = addr_to_entity.get(ins[0])
        if src is None:
            continue
        for addr, value in outputs:
            dst = addr_to_entity.get(addr)
            if dst and dst != src:
                edges.append({"src": src, "dst": dst, "ts": int(ts), "value_btc": float(value)})

    g = build_entity_graph(edges)
    for node in g.nodes():
        g.nodes[node]["label"] = "unknown"
    return g, addr_to_entity


def measure(g: nx.DiGraph, observed: set[str] | None = None) -> dict:
    """Fire rate of every rule, over the entities we actually have history for.

    ``observed`` is critical and its absence was a real bug in the first run of this
    experiment. Crawling 220 addresses pulls in ~50,000 *other* addresses that merely
    appear as transaction outputs. Each becomes a singleton entity with one edge and no
    fetched history, so it can never fire any rule.

    Measuring across all of them diluted both groups to ~1% and made the comparison
    meaningless -- it was mostly measuring "what fraction of this graph is singletons",
    which is a property of the crawl, not of the money. We can only judge the behaviour of
    entities whose transactions we actually retrieved, so that is the population.
    """
    nodes = [n for n in g.nodes() if observed is None or n in observed]
    n = max(len(nodes), 1)
    per_rule = Counter()
    with_1 = with_2 = 0

    for node in nodes:
        hits = detect_all(g, node)
        for rule in RULES:
            if rule(g, node) is not None:
                per_rule[rule.__name__] += 1
        s = len(structural(hits))
        if s >= 1:
            with_1 += 1
        if s >= 2:
            with_2 += 1

    return {
        "n_entities": len(nodes),
        "n_entities_in_graph": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "per_rule_pct": {k: round(100 * v / n, 2) for k, v in sorted(per_rule.items())},
        "per_rule_count": dict(sorted(per_rule.items())),
        "pct_with_1_structural": round(100 * with_1 / n, 2),
        "pct_with_2_structural": round(100 * with_2 / n, 2),
        "n_with_1_structural": with_1,
        "n_with_2_structural": with_2,
    }


def run(size: int = 120, hops: int = 1) -> dict:
    rng = random.Random(RANDOM_SEED)

    print("=" * 74)
    print("  REAL BITCOIN EXPERIMENT -- ransomware money vs ordinary money")
    print("=" * 74)

    ransomware = fetch_ransomwhere()
    ofac = fetch_ofac()
    seeds = select_seeds(ransomware, n=max(8, size // 12))
    fam_counts = Counter(s["family"] for s in seeds)
    print(f"\n[1/4] ransomware seeds: {len(seeds)} addresses across {len(fam_counts)} families")
    for fam, c in fam_counts.most_common():
        print(f"        {c:>2}  {fam}")

    print(f"\n[2/4] crawling REAL blockchain for ransomware group (budget {size}) ...")
    rw_records, rw_report = crawl(
        [s["address"] for s in seeds], max_addresses=size, hops=hops
    )
    print(f"      {len(rw_records):,} real transactions   "
          f"{ {k: v for k, v in rw_report.items() if k != 'fetched_addresses'} }")

    print(f"\n[3/4] crawling REAL blockchain for control group (budget {size}) ...")
    control_pool = fetch_recent_block_addresses()
    control_seeds = rng.sample(control_pool, min(len(seeds), len(control_pool)))
    ctl_records, ctl_report = crawl(control_seeds, max_addresses=size, hops=hops)
    print(f"      {len(ctl_records):,} real transactions   "
          f"{ {k: v for k, v in ctl_report.items() if k != 'fetched_addresses'} }")

    print("\n[4/4] clustering and running typology rules on both ...")
    rw_graph, rw_map = build_graph(rw_records)
    ctl_graph, ctl_map = build_graph(ctl_records)

    # Restrict to entities we actually retrieved history for -- see measure().
    rw_obs = observed_entities(rw_map, set(rw_report["fetched_addresses"]))
    ctl_obs = observed_entities(ctl_map, set(ctl_report["fetched_addresses"]))
    print(f"      observed entities: ransomware {len(rw_obs):,}, control {len(ctl_obs):,}")

    rw_stats = measure(rw_graph, rw_obs)
    ctl_stats = measure(ctl_graph, ctl_obs)

    # Did any crawled address turn out to be sanctioned? A real cross-reference hit.
    ofac_hits = sorted((set(rw_map) | set(ctl_map)) & ofac)

    result = {
        "note": (
            "Both groups are real Bitcoin addresses crawled from the live chain with "
            "identical parameters. The only difference is seed selection."
        ),
        "params": {"size": size, "hops": hops, "seed": RANDOM_SEED},
        "ransomware": {
            "seeds": len(seeds),
            "families": dict(fam_counts),
            "addresses_clustered": len(rw_map),
            "crawl": {k: v for k, v in rw_report.items() if k != "fetched_addresses"},
            **rw_stats,
        },
        "control": {
            "seeds": len(control_seeds),
            "addresses_clustered": len(ctl_map),
            "crawl": {k: v for k, v in ctl_report.items() if k != "fetched_addresses"},
            **ctl_stats,
        },
        "ofac_matches": ofac_hits,
    }

    _report(result)
    out = DATA_PROCESSED / "real_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  written to {out}")
    return result


def _report(r: dict) -> None:
    rw, ctl = r["ransomware"], r["control"]

    print("\n" + "=" * 74)
    print("  RESULTS -- real Bitcoin, both groups treated identically")
    print("=" * 74)
    print(f"{'':34}{'RANSOMWARE':>16}{'CONTROL':>16}")
    print(f"  {'entities':32}{rw['n_entities']:>16,}{ctl['n_entities']:>16,}")
    print(f"  {'edges':32}{rw['n_edges']:>16,}{ctl['n_edges']:>16,}")
    print("-" * 74)

    def row(label: str, a: float, b: float) -> None:
        ratio = (a / b) if b > 0 else float("inf") if a > 0 else 1.0
        arrow = f"{ratio:>6.1f}x" if ratio not in (float("inf"),) else "    inf"
        print(f"  {label:32}{a:>15.1f}%{b:>15.1f}%   {arrow}")

    row("has >=1 structural typology", rw["pct_with_1_structural"], ctl["pct_with_1_structural"])
    row("has >=2 structural typologies", rw["pct_with_2_structural"], ctl["pct_with_2_structural"])
    print("-" * 74)
    print("  per-rule fire rate:")
    all_rules = sorted(set(rw["per_rule_pct"]) | set(ctl["per_rule_pct"]))
    for rule in all_rules:
        a = rw["per_rule_pct"].get(rule, 0.0)
        b = ctl["per_rule_pct"].get(rule, 0.0)
        row("  " + rule.replace("detect_", ""), a, b)

    print("=" * 74)
    if r["ofac_matches"]:
        print(f"  OFAC SANCTIONED ADDRESSES FOUND IN CRAWL: {len(r['ofac_matches'])}")
        for a in r["ofac_matches"][:5]:
            print(f"    {a}")
    else:
        print("  No OFAC-sanctioned addresses appeared in this crawl.")

    lift = rw["pct_with_1_structural"] - ctl["pct_with_1_structural"]
    print("-" * 74)
    if lift > 10:
        print(f"  FINDING: the rules fire {lift:.1f} percentage points more often on real")
        print("  ransomware money than on ordinary Bitcoin activity. The signal is real.")
    elif lift > 3:
        print(f"  FINDING: modest separation ({lift:.1f} pp). Real but weak -- report it as such")
        print("  and tune thresholds in ml/typology/rules.py before claiming more.")
    else:
        print(f"  FINDING: little or no separation ({lift:.1f} pp). On this sample the rules do")
        print("  NOT distinguish ransomware money from ordinary activity. Say so honestly;")
        print("  a negative result reported straight is worth more than an invented one.")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=120, help="address budget per group")
    ap.add_argument("--hops", type=int, default=1)
    args = ap.parse_args()
    run(size=args.size, hops=args.hops)
