"""Generate the synthetic entity graph every workstream develops against.

Run this once on a fresh clone and you have working data in ten seconds, with no
dataset download and no ML dependencies -- stdlib only. That is the whole point: nobody
on the team is ever blocked waiting for role 1's ingest pipeline to be finished.

    python -m backend.tests.fixtures.make_fixtures

Writes to backend/tests/fixtures/:
    entities.json   one record per entity  -- mirrors entities.parquet
    edges.json      one record per flow    -- mirrors edges.parquet
    api/*.json      canned API responses   -- what the frontend binds to

The graph has laundering patterns deliberately planted in it, so the typology rules in
ml/typology/rules.py visibly fire.

Not to be confused with ml/data/synth.py. The two exist for different jobs:

    make_fixtures.py    ~90 entities, stdlib only, instant, committed to git.
                        Small enough to eyeball and diff. Frontend and API development
                        run against this without pandas, sklearn or a build step.

    ml/data/synth.py    ~17,000 entities, the real pipeline input. Realistic class
                        balance, CoinJoins, and a behavioural regime change at time
                        step 43. This is what `python run.py` builds and trains on.

Use this file when you want something tiny and readable; use synth.py when you want
something the model can actually learn from.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent
API_DIR = FIXTURE_DIR / "api"
SEED = 42

# Entity id blocks, so a reader of the JSON can tell what they are looking at.
VICTIMS = [f"E-{i:05d}" for i in range(1, 21)]  # ransom payers
COLLECTOR = "E-00100"  # sweeps victim payments
PEEL = [f"E-{100 + i:05d}" for i in range(1, 9)]  # E-00101..E-00108 peel chain
PEEL_OUTS = [f"E-{200 + i:05d}" for i in range(1, 9)]  # cash-out at each hop
MIXER = "E-00300"
MIXER_OUTS = [f"E-{300 + i:05d}" for i in range(1, 16)]
MULE = "E-00400"
MULE_SINK = "E-00401"
SMURF = "E-00500"
SMURF_OUTS = [f"E-{500 + i:05d}" for i in range(1, 11)]
SLEEPER = "E-00600"
SLEEPER_OUTS = [f"E-{600 + i:05d}" for i in range(1, 5)]
EXCHANGE = "E-00900"
MERCHANTS = [f"E-{900 + i:05d}" for i in range(1, 16)]


def build() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    edges: list[dict] = []
    labels: dict[str, str] = {}

    def flow(src: str, dst: str, ts: int, btc: float, n: int = 1) -> None:
        edges.append(
            {
                "src_entity": src,
                "dst_entity": dst,
                "ts": ts,
                "value_btc": round(btc, 8),
                "n_txs": n,
            }
        )

    # --- Pattern 1: fan-in. Twenty victims pay a ransom into one collector wallet in
    # the same time window. This is the shape a real ransomware campaign leaves behind.
    for v in VICTIMS:
        flow(v, COLLECTOR, ts=10, btc=rng.uniform(0.15, 0.35))
        labels[v] = "licit"
    labels[COLLECTOR] = "illicit"

    # --- Pattern 2: peel chain. The collector forwards the bulk down a chain, shaving a
    # spendable slice off at each hop.
    remaining = sum(e["value_btc"] for e in edges if e["dst_entity"] == COLLECTOR)
    chain = [COLLECTOR, *PEEL]
    for hop in range(len(PEEL)):
        peel_amt = remaining * 0.09
        remaining -= peel_amt
        flow(chain[hop], PEEL_OUTS[hop], ts=11 + hop, btc=peel_amt)
        flow(chain[hop], chain[hop + 1], ts=11 + hop, btc=remaining)
    for e in PEEL:
        labels.setdefault(e, "unknown")
    labels[PEEL[-1]] = "illicit"

    # --- Pattern 3: fan-out. The chain terminates in a dispersal step.
    flow(PEEL[-1], MIXER, ts=20, btc=remaining)
    for r in MIXER_OUTS:
        flow(MIXER, r, ts=21, btc=remaining / len(MIXER_OUTS) * rng.uniform(0.9, 1.1))
        labels.setdefault(r, "unknown")
    labels[MIXER] = "illicit"

    # --- Pattern 4: rapid pass-through. In and straight back out, nothing retained.
    flow("E-00050", MULE, ts=25, btc=5.0)
    flow(MULE, MULE_SINK, ts=25, btc=4.96)
    labels[MULE] = "unknown"
    labels["E-00050"] = "unknown"
    labels[MULE_SINK] = "unknown"

    # --- Pattern 5: structuring. Ten near-identical outbound transfers.
    flow("E-00051", SMURF, ts=30, btc=5.2)
    for i, d in enumerate(SMURF_OUTS):
        flow(SMURF, d, ts=30 + i, btc=0.500 + rng.uniform(-0.002, 0.002))
        labels.setdefault(d, "unknown")
    labels[SMURF] = "unknown"
    labels["E-00051"] = "licit"

    # --- Pattern 6: dormant, then burst.
    flow("E-00052", SLEEPER, ts=2, btc=40.0)
    flow("E-00053", SLEEPER, ts=3, btc=2.0)
    for i, d in enumerate(SLEEPER_OUTS):
        flow(SLEEPER, d, ts=44 + i, btc=10.0)
        labels.setdefault(d, "unknown")
    labels[SLEEPER] = "unknown"
    labels["E-00052"] = "licit"
    labels["E-00053"] = "licit"

    # --- Background: organic exchange traffic, so the graph is not only laundering and
    # a model trained on it has something to say "licit" about.
    labels[EXCHANGE] = "licit"
    for m in MERCHANTS:
        labels[m] = "licit"
        for _ in range(rng.randint(2, 6)):
            flow(EXCHANGE, m, ts=rng.randint(1, 49), btc=rng.uniform(0.01, 2.5))
            flow(m, EXCHANGE, ts=rng.randint(1, 49), btc=rng.uniform(0.01, 1.2))

    # --- Roll edges up into entity records.
    ids = sorted({e["src_entity"] for e in edges} | {e["dst_entity"] for e in edges})
    entities: list[dict] = []
    for eid in ids:
        ins = [e for e in edges if e["dst_entity"] == eid]
        outs = [e for e in edges if e["src_entity"] == eid]
        stamps = [e["ts"] for e in ins + outs]
        entities.append(
            {
                "entity_id": eid,
                "n_addresses": rng.randint(1, 40),
                "first_seen_ts": min(stamps),
                "last_seen_ts": max(stamps),
                "total_in_btc": round(sum(e["value_btc"] for e in ins), 8),
                "total_out_btc": round(sum(e["value_btc"] for e in outs), 8),
                "in_degree": len(ins),
                "out_degree": len(outs),
                "label": labels.get(eid, "unknown"),
            }
        )
    return entities, edges


def to_graph(entities: list[dict], edges: list[dict]):
    """Build the networkx DiGraph that ml/typology/rules.py expects."""
    import networkx as nx

    g = nx.DiGraph()
    for e in entities:
        g.add_node(e["entity_id"], **e)
    for e in edges:
        g.add_edge(
            e["src_entity"],
            e["dst_entity"],
            ts=e["ts"],
            value_btc=e["value_btc"],
            n_txs=e["n_txs"],
        )
    return g


def _stub_risk(entity: dict) -> float:
    """Placeholder risk score until role 2's real model lands.

    Deliberately crude and clearly named so nobody mistakes it for a trained model. The
    frontend needs *a* number in the 0-1 range to build against; this provides one.
    """
    return {"illicit": 0.93, "licit": 0.07, "unknown": 0.45}[entity["label"]]


def write_api_fixtures(entities: list[dict], edges: list[dict]) -> None:
    """Canned API responses. The frontend binds to these on day one."""
    from ml.typology.rules import detect_all, narrate

    API_DIR.mkdir(parents=True, exist_ok=True)
    g = to_graph(entities, edges)
    by_id = {e["entity_id"]: e for e in entities}

    # One full entity-detail response for the collector -- the most interesting node.
    ent = by_id[COLLECTOR]
    risk = _stub_risk(ent)
    hits = detect_all(g, COLLECTOR)
    detail = {
        "id": COLLECTOR,
        "risk": risk,
        "label": ent["label"],
        "n_addresses": ent["n_addresses"],
        "first_seen_ts": ent["first_seen_ts"],
        "last_seen_ts": ent["last_seen_ts"],
        "total_in_btc": ent["total_in_btc"],
        "total_out_btc": ent["total_out_btc"],
        "features": {
            "in_degree": float(ent["in_degree"]),
            "out_degree": float(ent["out_degree"]),
            "fwd_ratio": round(ent["total_out_btc"] / max(ent["total_in_btc"], 1e-9), 4),
            "lifetime_ts": float(ent["last_seen_ts"] - ent["first_seen_ts"]),
        },
        # Placeholder until role 4 wires real SHAP. Shape is final; values are not.
        "shap": [
            {"feature": "in_degree", "value": 20.0, "contribution": 0.31},
            {"feature": "fwd_ratio", "value": 0.98, "contribution": 0.22},
            {"feature": "lifetime_ts", "value": 8.0, "contribution": -0.11},
        ],
        "typologies": [t.to_dict() for t in hits],
        "neighbors": [
            {
                "id": e["src_entity"],
                "risk": _stub_risk(by_id[e["src_entity"]]),
                "label": by_id[e["src_entity"]]["label"],
                "direction": "in",
                "value_btc": e["value_btc"],
                "n_txs": e["n_txs"],
            }
            for e in edges
            if e["dst_entity"] == COLLECTOR
        ][:20],
        "narrative": narrate(COLLECTOR, risk, hits),
        "model_version": "0.0.0-fixture",
        "clustering_confidence": 0.82,
    }
    (API_DIR / "entity_detail.json").write_text(json.dumps(detail, indent=2))

    # A 2-hop subgraph around the collector, capped and honest about truncation.
    import networkx as nx

    near = nx.single_source_shortest_path_length(g.to_undirected(as_view=True), COLLECTOR, cutoff=2)
    keep = set(near)
    sub = {
        "seed_id": COLLECTOR,
        "hops": 2,
        "nodes": [
            {
                "id": n,
                "risk": _stub_risk(by_id[n]),
                "label": by_id[n]["label"],
                "size": round(by_id[n]["total_in_btc"] + by_id[n]["total_out_btc"], 4),
                "is_seed": n == COLLECTOR,
            }
            for n in sorted(keep)
        ],
        "edges": [
            {
                "source": e["src_entity"],
                "target": e["dst_entity"],
                "value_btc": e["value_btc"],
                "n_txs": e["n_txs"],
                "ts": e["ts"],
            }
            for e in edges
            if e["src_entity"] in keep and e["dst_entity"] in keep
        ],
        "truncated": False,
        "total_available": len(keep),
        "ranking": "risk_x_value",
    }
    (API_DIR / "subgraph.json").write_text(json.dumps(sub, indent=2))

    # Timeline buckets for the scrubber.
    timeline = []
    for ts in range(1, 50):
        at_ts = [e for e in edges if e["ts"] == ts]
        timeline.append(
            {
                "ts": ts,
                "n_txs": sum(e["n_txs"] for e in at_ts),
                "volume_btc": round(sum(e["value_btc"] for e in at_ts), 6),
                "n_illicit": sum(
                    1 for e in at_ts if by_id[e["dst_entity"]]["label"] == "illicit"
                ),
            }
        )
    (API_DIR / "timeline.json").write_text(json.dumps(timeline, indent=2))


def main() -> None:
    entities, edges = build()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "entities.json").write_text(json.dumps(entities, indent=2))
    (FIXTURE_DIR / "edges.json").write_text(json.dumps(edges, indent=2))
    write_api_fixtures(entities, edges)

    n_illicit = sum(1 for e in entities if e["label"] == "illicit")
    print(f"entities        {len(entities)}")
    print(f"edges           {len(edges)}")
    print(f"illicit         {n_illicit}")
    print(f"total volume    {sum(e['value_btc'] for e in edges):.4f} BTC")
    print(f"written to      {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
