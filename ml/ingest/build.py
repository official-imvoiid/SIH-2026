"""Build the canonical entity store that everything downstream reads.

Pipeline: transactions -> address clustering -> entity graph -> features -> parquet.

Run it with::

    python -m ml.ingest.build            # synthetic world, no download needed
    python -m ml.ingest.build --real     # real Elliptic++, once you have the CSVs

Outputs (the frozen contract from docs/TEAM.md):

    data/processed/entities.parquet     one row per entity, with features and label
    data/processed/edges.parquet        entity-to-entity money flows
    data/processed/address_map.parquet  address -> entity_id
    data/processed/build_report.json    counts, clustering quality, provenance
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from ml.config import DATA_PROCESSED, RANDOM_SEED, TRAIN_TS_MAX
from ml.features.clustering import cluster_addresses, evaluate_clustering
from ml.features.graph import (
    build_entity_graph,
    label_proximity_features,
    structural_features,
)
from ml.ingest.elliptic import (
    DOWNLOAD_HELP,
    find_dataset,
    load_elliptic,
    load_elliptic_plus,
)


def _load_synthetic():
    from ml.data.synth import generate

    world = generate(seed=RANDOM_SEED)
    txs = [
        (tx.txid, list(tx.inputs), [v for _a, v in tx.outputs], tx.ts, tx.outputs)
        for tx in world.transactions
    ]
    truth_owner = dict(world.owner)
    truth_label = {eid: e.label for eid, e in world.entities.items()}
    return txs, truth_owner, truth_label, "synthetic-v1"


def build_from_real(raw_dir: Path, out_dir: Path) -> dict:
    """Build the entity store from a real dataset already on disk.

    Differs from the synthetic path in one structural way: the real datasets arrive as
    entity/edge tables, not as raw transactions, so there is nothing to cluster. For
    Elliptic++ the addresses *are* the entities; for classic Elliptic the transactions are.
    Either way clustering is skipped and the build report says so, because presenting a
    transaction-level build as entity resolution would be the central dishonesty this
    project exists to avoid.
    """
    paths = find_dataset(raw_dir)
    if paths is None:
        raise SystemExit(DOWNLOAD_HELP)

    print(f"[1/5] loading {paths} ...")
    if paths.kind == "elliptic_plus":
        entities, edges, meta = load_elliptic_plus(paths)
    else:
        entities, edges, meta = load_elliptic(paths)

    caps = meta["capabilities"]
    print(f"      {meta['n_entities']:,} entities | {meta['n_edges']:,} edges")
    print(f"      unit: {caps['unit']}   entity resolution: {caps['entity_resolution']}")
    for c in caps.get("caveats", []):
        print(f"      caveat: {c}")

    print("[2/5] building entity money-flow graph ...")
    g = build_entity_graph(
        {"src": r.src_entity, "dst": r.dst_entity, "ts": r.ts, "value_btc": r.value_btc}
        for r in edges.itertuples()
    )
    print(f"      {g.number_of_nodes():,} nodes | {g.number_of_edges():,} edges")

    print("[3/5] computing graph features ...")
    struct = structural_features(g)

    entity_label = dict(zip(entities["entity_id"], entities["label"]))
    train_illicit = {
        eid
        for eid, lab in entity_label.items()
        if lab == "illicit" and eid in struct and struct[eid]["last_ts"] <= TRAIN_TS_MAX
    }
    print(
        f"      {len(train_illicit)} illicit entities inside the training window "
        f"(ts <= {TRAIN_TS_MAX}) used for proximity features"
    )
    prox = label_proximity_features(g, train_illicit, max_hops=3)

    print("[4/5] merging dataset features with graph features ...")
    graph_rows = []
    for eid in g.nodes():
        row = {"entity_id": eid}
        row.update(struct[eid])
        row.update(prox.get(eid, {}))
        graph_rows.append(row)
    graph_df = pd.DataFrame(graph_rows)

    # Entities with no edges still deserve a row: they carry dataset features and a label,
    # and dropping them would silently shrink the evaluation set.
    merged = entities.merge(graph_df, on="entity_id", how="left", suffixes=("", "_graph"))
    for col in merged.columns:
        if merged[col].dtype.kind in "fc":
            merged[col] = merged[col].fillna(0.0)

    print("[5/5] writing parquet ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = merged.sort_values("entity_id").reset_index(drop=True)
    merged.to_parquet(out_dir / "entities.parquet", index=False)
    edges.to_parquet(out_dir / "edges.parquet", index=False)
    pd.DataFrame({"address": merged["entity_id"], "entity_id": merged["entity_id"]}).to_parquet(
        out_dir / "address_map.parquet", index=False
    )

    label_counts = merged["label"].value_counts().to_dict()
    total = len(merged)
    report = {
        **meta,
        "seed": RANDOM_SEED,
        "n_entities": total,
        "n_edges": len(edges),
        "labels": label_counts,
        "label_pct": {k: round(100 * v / total, 2) for k, v in label_counts.items()},
        "clustering_quality": {
            "skipped": True,
            "reason": (
                "Elliptic++ addresses are already entities; classic Elliptic has no "
                "addresses at all. Co-spend clustering applies to neither."
            ),
        },
        "train_illicit_used_for_proximity": len(train_illicit),
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"  {caps['dataset']}")
    print(f"  entities {total:,}   edges {len(edges):,}")
    print(
        "  labels:  "
        + "  ".join(f"{k}={v:,} ({report['label_pct'][k]}%)" for k, v in label_counts.items())
    )
    if not caps["entity_resolution"]:
        print("  NOTE: this build does NOT do entity resolution -- see capabilities.")
    print("=" * 66)
    return report


def build(use_real: bool = False, out_dir: Path = DATA_PROCESSED) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_real:
        return build_from_real(Path("data/raw"), out_dir)

    print("[1/6] loading transactions ...")
    txs, truth_owner, truth_label, provenance = _load_synthetic()
    print(f"      {len(txs):,} transactions  ({provenance})")

    print("[2/6] clustering addresses into entities (co-spend heuristic) ...")
    addr_to_entity = cluster_addresses(
        ((txid, ins, outs) for txid, ins, outs, _ts, _o in txs), skip_coinjoins=True
    )
    # Addresses that only ever received (never spent) are never seen by the clustering
    # step. Each becomes its own singleton entity -- correct, since we have no evidence
    # linking them to anything.
    next_id = len(set(addr_to_entity.values())) + 1
    for _txid, _ins, _outs, _ts, outputs in txs:
        for addr, _v in outputs:
            if addr not in addr_to_entity:
                addr_to_entity[addr] = f"E-{next_id:05d}"
                next_id += 1
    print(
        f"      {len(addr_to_entity):,} addresses -> "
        f"{len(set(addr_to_entity.values())):,} entities"
    )

    clustering_quality = {}
    if truth_owner:
        clustering_quality = evaluate_clustering(addr_to_entity, truth_owner)
        print(
            f"      pair precision {clustering_quality['pair_precision']:.4f} | "
            f"recall {clustering_quality['pair_recall']:.4f} | "
            f"false merges {clustering_quality['false_merges']}"
        )

    print("[3/6] building entity money-flow graph ...")
    edge_rows: list[dict] = []
    for _txid, ins, _outs, ts, outputs in txs:
        if not ins:
            continue
        src = addr_to_entity.get(ins[0])
        if src is None:
            continue
        for addr, value in outputs:
            dst = addr_to_entity.get(addr)
            if dst is None or dst == src:  # self-change, not a real flow
                continue
            edge_rows.append({"src": src, "dst": dst, "ts": int(ts), "value_btc": float(value)})

    g = build_entity_graph(edge_rows)
    print(f"      {g.number_of_nodes():,} nodes | {g.number_of_edges():,} edges")

    print("[4/6] propagating labels to clusters ...")
    # A predicted cluster inherits the majority label of the true owners it contains.
    cluster_true_labels: dict[str, Counter] = defaultdict(Counter)
    for addr, eid in addr_to_entity.items():
        true_owner = truth_owner.get(addr)
        if true_owner is not None:
            cluster_true_labels[eid][truth_label.get(true_owner, "unknown")] += 1

    entity_label: dict[str, str] = {}
    for eid in set(addr_to_entity.values()):
        counts = cluster_true_labels.get(eid)
        if not counts:
            entity_label[eid] = "unknown"
            continue
        # "unknown" never outvotes a real label -- a single confirmed illicit address
        # taints the cluster, which is how an investigator would treat it too.
        for lab in ("illicit", "licit"):
            if counts.get(lab):
                entity_label[eid] = lab
                break
        else:
            entity_label[eid] = "unknown"

    print("[5/6] computing features ...")
    struct = structural_features(g)

    # Label-derived features use ONLY entities whose activity ends inside the training
    # window. Using every illicit label here would hand the model test-period answers.
    train_illicit = {
        eid
        for eid, lab in entity_label.items()
        if lab == "illicit"
        and eid in struct
        and struct[eid]["last_ts"] <= TRAIN_TS_MAX
    }
    print(
        f"      {len(train_illicit)} illicit entities inside the training window "
        f"(ts <= {TRAIN_TS_MAX}) used for proximity features"
    )
    prox = label_proximity_features(g, train_illicit, max_hops=3)

    addrs_per_entity = Counter(addr_to_entity.values())

    rows = []
    for eid in g.nodes():
        feat = dict(struct[eid])
        feat.update(prox.get(eid, {}))
        feat["n_addresses"] = float(addrs_per_entity.get(eid, 1))
        rows.append(
            {
                "entity_id": eid,
                "label": entity_label.get(eid, "unknown"),
                **feat,
            }
        )

    entities = pd.DataFrame(rows).sort_values("entity_id").reset_index(drop=True)
    edges = pd.DataFrame(
        [
            {
                "src_entity": u,
                "dst_entity": v,
                "ts": d["ts"],
                "value_btc": round(d["value_btc"], 8),
                "n_txs": d["n_txs"],
            }
            for u, v, d in g.edges(data=True)
        ]
    ).sort_values(["src_entity", "dst_entity"]).reset_index(drop=True)

    addr_map = pd.DataFrame(
        sorted(addr_to_entity.items()), columns=["address", "entity_id"]
    )

    print("[6/6] writing parquet ...")
    entities.to_parquet(out_dir / "entities.parquet", index=False)
    edges.to_parquet(out_dir / "edges.parquet", index=False)
    addr_map.to_parquet(out_dir / "address_map.parquet", index=False)

    label_counts = entities["label"].value_counts().to_dict()
    total = len(entities)
    report = {
        "provenance": provenance,
        "seed": RANDOM_SEED,
        "n_transactions": len(txs),
        "n_addresses": len(addr_to_entity),
        "n_entities": total,
        "n_edges": len(edges),
        "labels": label_counts,
        "label_pct": {k: round(100 * v / total, 2) for k, v in label_counts.items()},
        "clustering_quality": clustering_quality,
        "train_illicit_used_for_proximity": len(train_illicit),
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"  entities {total:,}   edges {len(edges):,}   addresses {len(addr_map):,}")
    print(
        "  labels:  "
        + "  ".join(
            f"{k}={v:,} ({report['label_pct'][k]}%)" for k, v in label_counts.items()
        )
    )
    print("=" * 66)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true", help="use the real Elliptic++ CSVs")
    args = ap.parse_args()
    build(use_real=args.real)
