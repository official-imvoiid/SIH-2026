"""Does graph structure actually add anything over hand-crafted features?

The question
------------
Weber et al. (2019) found that on Elliptic, a Random Forest on raw features (~0.79 illicit
F1) beat a Graph Convolutional Network (~0.65), and that their best result came from
concatenating GNN node embeddings onto the raw features rather than from the GNN alone.
That finding is the reason this project ships a forest and treats graph learning as an
upgrade to be *measured*, not assumed.

This module runs that comparison directly. Four models, identical temporal split, identical
forest, identical metrics -- the only thing that varies is what goes in:

  A. **tabular**    hand-crafted features only (volume, timing, degree, PageRank...)
  B. **embeddings** learned graph representation only
  C. **combined**   both, concatenated -- the Weber et al. "AF+NE" recipe
  D. **no-graph**   tabular minus every graph-derived column, as a floor

D matters because A already contains PageRank and hop-distance features. Without D you
cannot tell whether "graph structure helps" or just "our hand-crafted graph features help".

What this is NOT
----------------
**These are spectral embeddings (truncated SVD of the adjacency matrix), not a GNN.** They
are a well-established shallow graph representation, they need no GPU and no PyTorch, and
they answer the "does learned structure help?" question honestly. They do not answer "would
a GraphSAGE beat this?" -- for that you need PyTorch Geometric and, more importantly, the
real Elliptic dataset, because a GNN comparison on our own generated world would measure
the generator rather than the method.

Run::

    python -m ml.train.benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

from ml.config import ARTIFACTS, DATA_PROCESSED, RANDOM_SEED
from ml.train.baseline import NON_FEATURES, evaluate, load_features, temporal_split

# Two distinct families, separated because they turn out to behave very differently.
#
# STRUCTURE: derived from the graph shape alone. Label-free, so they cannot leak.
GRAPH_STRUCTURAL = {
    "pagerank",
    "clustering_coef",
    "in_degree",
    "out_degree",
    "total_degree",
    "degree_ratio",
}
# ASSOCIATION: derived from *other entities' labels* -- guilt by association. These are the
# ones the project already demotes on ethical grounds ("behaviour beats association"), and
# the benchmark exists partly to check whether that principle also costs accuracy.
GRAPH_ASSOCIATION = {
    "hops_to_illicit",
    "illicit_neighbours",
    "illicit_neighbour_frac",
    "near_illicit_2hop",
}
GRAPH_DERIVED = GRAPH_STRUCTURAL | GRAPH_ASSOCIATION

EMBED_DIM = 32


def build_embeddings(
    entity_ids: list[str], edges: pd.DataFrame, dim: int = EMBED_DIM
) -> np.ndarray:
    """Spectral node embeddings: truncated SVD of the symmetrised adjacency matrix.

    Each entity gets a dense vector summarising who it transacts with. Two entities with
    similar neighbourhoods land near each other, which is the same intuition a GNN exploits
    -- learned here in one linear algebra step instead of by gradient descent.

    Symmetrised on purpose: direction matters for typologies (fan-in is not fan-out) but
    for *representation* the undirected structure is denser and better conditioned, and the
    directional signal is already carried by the tabular degree features.
    """
    index = {eid: i for i, eid in enumerate(entity_ids)}
    n = len(entity_ids)

    rows, cols, vals = [], [], []
    for src, dst, w in zip(edges["src_entity"], edges["dst_entity"], edges["value_btc"]):
        i, j = index.get(src), index.get(dst)
        if i is None or j is None:
            continue
        # log1p on value: raw BTC amounts span many orders of magnitude and would let a
        # handful of huge transfers dominate every singular vector.
        weight = float(np.log1p(max(w, 0.0))) + 1e-3
        rows += [i, j]
        cols += [j, i]
        vals += [weight, weight]

    adjacency = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)

    k = min(dim, max(1, min(adjacency.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=RANDOM_SEED, algorithm="randomized")
    emb = svd.fit_transform(adjacency)

    if emb.shape[1] < dim:  # pad so downstream shapes stay fixed
        emb = np.hstack([emb, np.zeros((n, dim - emb.shape[1]), dtype=emb.dtype)])
    return emb.astype(np.float32), float(svd.explained_variance_ratio_.sum())


def _forest() -> RandomForestClassifier:
    """Identical model in every arm -- only the inputs differ."""
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def _run_arm(name: str, X_tr, y_tr, X_te, y_te) -> dict:
    t0 = time.time()
    model = _forest().fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    res = evaluate(y_te, proba)
    res["arm"] = name
    res["n_input_features"] = int(X_tr.shape[1])
    res["fit_seconds"] = round(time.time() - t0, 2)
    return res


def run(out_dir: Path = ARTIFACTS) -> dict:
    df, feature_cols = load_features()
    edges = pd.read_parquet(DATA_PROCESSED / "edges.parquet")

    print("building spectral graph embeddings ...")
    entity_ids = df["entity_id"].tolist()
    emb, explained = build_embeddings(entity_ids, edges)
    print(f"  {emb.shape[1]} dimensions, {explained:.1%} of variance captured")

    emb_cols = [f"emb_{i}" for i in range(emb.shape[1])]
    df_emb = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(emb, columns=emb_cols)], axis=1
    )

    train_df, _val_df, test_df = temporal_split(df_emb)
    if train_df["y"].sum() < 5 or test_df["y"].sum() < 3:
        raise SystemExit("Not enough labelled illicit entities to benchmark.")

    tabular = [c for c in feature_cols if c not in NON_FEATURES]
    no_graph = [c for c in tabular if c not in GRAPH_DERIVED]
    # The decomposition that matters: drop only the guilt-by-association features and keep
    # the label-free structural ones. If this arm beats full tabular, association features
    # are actively harming generalisation, not merely redundant.
    no_assoc = [c for c in tabular if c not in GRAPH_ASSOCIATION]

    arms = {
        "no-graph (behaviour only)": no_graph,
        "no-association (behaviour + structure)": no_assoc,
        "tabular (everything hand-crafted)": tabular,
        "embeddings (learned structure only)": emb_cols,
        "combined (tabular + embeddings)": tabular + emb_cols,
    }

    y_tr = train_df["y"].to_numpy()
    y_te = test_df["y"].to_numpy()

    results = []
    for name, cols in arms.items():
        res = _run_arm(
            name,
            train_df[cols].to_numpy(dtype=np.float32),
            y_tr,
            test_df[cols].to_numpy(dtype=np.float32),
            y_te,
        )
        results.append(res)
        print(
            f"  {name:46} F1={res['illicit_f1']:.3f}  "
            f"P={res['illicit_precision']:.3f}  R={res['illicit_recall']:.3f}  "
            f"AP={res['average_precision']}"
        )

    best = max(results, key=lambda r: r["illicit_f1"])
    tab = next(r for r in results if r["arm"].startswith("tabular"))
    noassoc = next(r for r in results if r["arm"].startswith("no-association"))
    embed = next(r for r in results if r["arm"].startswith("embeddings"))
    nograph = next(r for r in results if r["arm"].startswith("no-graph"))
    combined = next(r for r in results if r["arm"].startswith("combined"))

    verdict = []
    if noassoc["illicit_f1"] > tab["illicit_f1"] + 0.01:
        verdict.append(
            f"GUILT-BY-ASSOCIATION FEATURES HURT: dropping them raises F1 from "
            f"{tab['illicit_f1']:.3f} to {noassoc['illicit_f1']:.3f}. They are fitted to "
            f"the training period's illicit population, which changes by test time -- so "
            f"they generalise worse than behaviour does. The project's "
            f"'behaviour beats association' rule is therefore not only an ethical "
            f"position; it also scores better."
        )
    elif tab["illicit_f1"] > noassoc["illicit_f1"] + 0.01:
        verdict.append(
            f"Association features help accuracy ({noassoc['illicit_f1']:.3f} -> "
            f"{tab['illicit_f1']:.3f}). They stay demoted in explanations on ethical "
            f"grounds, but they earn their place in the model."
        )
    else:
        verdict.append(
            f"Association features are roughly neutral ({noassoc['illicit_f1']:.3f} vs "
            f"{tab['illicit_f1']:.3f})."
        )

    if noassoc["illicit_f1"] > nograph["illicit_f1"] + 0.01:
        verdict.append(
            f"Label-free graph structure helps: F1 {nograph['illicit_f1']:.3f} -> "
            f"{noassoc['illicit_f1']:.3f}."
        )
    else:
        verdict.append(
            f"Label-free graph structure adds little on this data: "
            f"{nograph['illicit_f1']:.3f} -> {noassoc['illicit_f1']:.3f}."
        )

    if embed["illicit_f1"] > tab["illicit_f1"]:
        verdict.append(
            f"Learned embeddings alone BEAT hand-crafted features "
            f"({embed['illicit_f1']:.3f} vs {tab['illicit_f1']:.3f})."
        )
    else:
        verdict.append(
            f"Learned embeddings alone lose to hand-crafted features "
            f"({embed['illicit_f1']:.3f} vs {tab['illicit_f1']:.3f}) -- consistent with "
            f"Weber et al. finding trees beat a GCN on this problem."
        )

    if combined["illicit_f1"] > tab["illicit_f1"] + 0.01:
        verdict.append(
            f"Concatenating both is best ({combined['illicit_f1']:.3f}), which is the "
            f"Weber et al. 'AF+NE' result reproduced."
        )
    else:
        verdict.append(
            f"Concatenating embeddings does not improve on tabular alone "
            f"({combined['illicit_f1']:.3f} vs {tab['illicit_f1']:.3f}); the simpler model "
            f"is the one to ship."
        )

    out = {
        "note": (
            "Spectral embeddings (truncated SVD of the adjacency matrix), NOT a GNN. "
            "Answers whether learned graph structure helps; does not answer whether a "
            "GraphSAGE would win. A true GNN comparison needs the real Elliptic dataset, "
            "since benchmarking on a generated world measures the generator."
        ),
        "seed": RANDOM_SEED,
        "embedding_dim": int(emb.shape[1]),
        "explained_variance": round(explained, 4),
        "provenance": json.loads(
            (DATA_PROCESSED / "build_report.json").read_text(encoding="utf-8")
        ).get("provenance", "unknown"),
        "arms": results,
        "best_arm": best["arm"],
        "verdict": verdict,
    }

    print("\n" + "=" * 74)
    print(f"  BEST: {best['arm']}  (F1 {best['illicit_f1']:.3f})")
    for v in verdict:
        print(f"  - {v}")
    print("=" * 74)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  written to {path}")
    return out


if __name__ == "__main__":
    run()
