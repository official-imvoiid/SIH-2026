"""Entity-level features derived from the money-flow graph.

Two families of feature live here, and the distinction between them is the difference
between an honest result and a leaky one:

**Structural features** (degree, volume, timing, PageRank) use only the graph shape. They
are safe to compute over the whole graph.

**Label-derived features** (proximity to known-illicit entities, neighbourhood illicit
density) use ground-truth labels, so they can leak the answer. A model that knows "one of
my neighbours is labelled illicit *in the test period*" has been handed information no
real investigator would have at prediction time. Every such feature here therefore takes
an explicit ``known_illicit`` set that the caller must restrict to the training window.
There is no default that silently uses all labels.

The label-derived family is worth the care: distance-to-known-illicit is the single
strongest cheap feature on this kind of graph (the "GuiltyWalker" idea), and it needs no
deep learning at all.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import networkx as nx

__all__ = [
    "structural_features",
    "label_proximity_features",
    "build_entity_graph",
    "FEATURE_NAMES",
]

# Sentinel for "no labelled illicit entity is reachable". Chosen rather than infinity so
# the value stays usable by tree models, and large enough to sit clearly outside any real
# hop distance.
UNREACHABLE = 99


def build_entity_graph(edges: Iterable[dict]) -> nx.DiGraph:
    """Build the entity-to-entity money-flow graph.

    Each edge record is ``{src, dst, ts, value_btc}``. Parallel edges between the same
    pair are collapsed into one edge carrying aggregate weight, plus the per-transfer
    detail the typology rules need to reason about timing.
    """
    g = nx.DiGraph()
    for e in edges:
        src, dst = e["src"], e["dst"]
        ts, value = int(e["ts"]), float(e["value_btc"])
        if g.has_edge(src, dst):
            d = g[src][dst]
            d["value_btc"] += value
            d["n_txs"] += 1
            d["ts"] = min(d["ts"], ts)
            d["ts_list"].append(ts)
        else:
            g.add_edge(src, dst, value_btc=value, n_txs=1, ts=ts, ts_list=[ts])
    return g


def _cv(values: list[float]) -> float:
    """Coefficient of variation -- how uniform a set of transfer sizes is.

    Near zero means near-identical amounts, which is the structuring signature. Scale-free,
    so it compares a wallet moving 0.01 BTC with one moving 100 BTC.
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) / mean


def structural_features(g: nx.DiGraph) -> dict[str, dict[str, float]]:
    """Compute label-free features for every entity. Safe on the full graph."""
    pagerank = nx.pagerank(g, alpha=0.85, max_iter=100, tol=1e-6) if g.number_of_edges() else {}
    undirected = g.to_undirected(as_view=True)

    out: dict[str, dict[str, float]] = {}
    for node in g.nodes():
        in_edges = list(g.in_edges(node, data=True))
        out_edges = list(g.out_edges(node, data=True))

        in_vals = [d["value_btc"] for _u, _v, d in in_edges]
        out_vals = [d["value_btc"] for _u, _v, d in out_edges]
        total_in = sum(in_vals)
        total_out = sum(out_vals)

        all_ts = [t for _u, _v, d in in_edges + out_edges for t in d["ts_list"]]
        first_ts = min(all_ts) if all_ts else 0
        last_ts = max(all_ts) if all_ts else 0

        # Burst: the most counterparties seen in any single time step. This is what
        # separates a collection wallet from a merchant with the same total degree.
        in_by_ts: dict[int, set] = {}
        for u, _v, d in in_edges:
            for t in d["ts_list"]:
                in_by_ts.setdefault(t, set()).add(u)
        out_by_ts: dict[int, set] = {}
        for _u, v, d in out_edges:
            for t in d["ts_list"]:
                out_by_ts.setdefault(t, set()).add(v)

        n_tx_in = sum(d["n_txs"] for _u, _v, d in in_edges)
        n_tx_out = sum(d["n_txs"] for _u, _v, d in out_edges)

        # Longest run of consecutive inactive time steps, which is what the
        # dormant-then-burst typology keys on.
        uniq_ts = sorted(set(all_ts))
        max_gap = max(
            (b - a for a, b in zip(uniq_ts, uniq_ts[1:])), default=0
        )

        out[node] = {
            "in_degree": float(len(in_edges)),
            "out_degree": float(len(out_edges)),
            "total_degree": float(len(in_edges) + len(out_edges)),
            "degree_ratio": len(out_edges) / max(len(in_edges), 1),
            "total_in_btc": total_in,
            "total_out_btc": total_out,
            "net_btc": total_in - total_out,
            # The pass-through signature: near 1.0 means the wallet retained nothing.
            "fwd_ratio": (total_out / total_in) if total_in > 0 else 0.0,
            "mean_in_btc": (total_in / len(in_vals)) if in_vals else 0.0,
            "mean_out_btc": (total_out / len(out_vals)) if out_vals else 0.0,
            "max_in_btc": max(in_vals, default=0.0),
            "max_out_btc": max(out_vals, default=0.0),
            "out_value_cv": _cv(out_vals),
            "in_value_cv": _cv(in_vals),
            "n_tx_in": float(n_tx_in),
            "n_tx_out": float(n_tx_out),
            "first_ts": float(first_ts),
            "last_ts": float(last_ts),
            "lifetime_ts": float(last_ts - first_ts),
            "max_dormancy_ts": float(max_gap),
            "max_in_burst": float(max((len(s) for s in in_by_ts.values()), default=0)),
            "max_out_burst": float(max((len(s) for s in out_by_ts.values()), default=0)),
            "pagerank": pagerank.get(node, 0.0) * 1e4,
            "clustering_coef": nx.clustering(undirected, node),
        }
    return out


def _bfs_hop_distance(
    g: nx.Graph, sources: set[str], max_hops: int
) -> dict[str, int]:
    """Multi-source BFS: hop distance from every node to the nearest source.

    Run once from all known-illicit entities at the same time, rather than once per node,
    so this stays linear in the graph size instead of quadratic.
    """
    dist: dict[str, int] = {s: 0 for s in sources if s in g}
    queue = deque(dist)
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= max_hops:
            continue
        for nbr in g.neighbors(node):
            if nbr not in dist:
                dist[nbr] = d + 1
                queue.append(nbr)
    return dist


def label_proximity_features(
    g: nx.DiGraph,
    known_illicit: set[str],
    max_hops: int = 3,
) -> dict[str, dict[str, float]]:
    """Features describing how close an entity sits to *known* illicit activity.

    Parameters
    ----------
    known_illicit
        Entities labelled illicit **within the training window only**. Passing the full
        label set here leaks the test answers and will inflate your F1 by a wide margin.
        The caller owns this restriction; this function deliberately has no default.

    Notes
    -----
    Guilt by association is weak evidence on its own -- an exchange sits two hops from
    everything -- but combined with a structural flag it is exactly what an analyst
    triages on, and it is cheap.
    """
    undirected = g.to_undirected(as_view=True)
    sources = {e for e in known_illicit if e in g}
    dist = _bfs_hop_distance(undirected, sources, max_hops)

    out: dict[str, dict[str, float]] = {}
    for node in g.nodes():
        own_hops = dist.get(node, UNREACHABLE)
        # An entity that is itself a known-illicit source must not read its own label back
        # as a feature: report its distance to the nearest *other* illicit entity.
        if node in sources:
            neighbour_hops = [
                dist.get(n, UNREACHABLE)
                for n in undirected.neighbors(node)
                if n in dist
            ]
            own_hops = 1 + min(neighbour_hops, default=UNREACHABLE - 1)
            own_hops = min(own_hops, UNREACHABLE)

        n_illicit_1hop = sum(
            1 for n in undirected.neighbors(node) if n in sources and n != node
        )
        deg = undirected.degree(node) or 1

        out[node] = {
            "hops_to_illicit": float(own_hops),
            "illicit_neighbours": float(n_illicit_1hop),
            "illicit_neighbour_frac": n_illicit_1hop / deg,
            "near_illicit_2hop": 1.0 if own_hops <= 2 else 0.0,
        }
    return out


FEATURE_NAMES: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "total_degree",
    "degree_ratio",
    "total_in_btc",
    "total_out_btc",
    "net_btc",
    "fwd_ratio",
    "mean_in_btc",
    "mean_out_btc",
    "max_in_btc",
    "max_out_btc",
    "out_value_cv",
    "in_value_cv",
    "n_tx_in",
    "n_tx_out",
    "first_ts",
    "last_ts",
    "lifetime_ts",
    "max_dormancy_ts",
    "max_in_burst",
    "max_out_burst",
    "pagerank",
    "clustering_coef",
    "n_addresses",
    "hops_to_illicit",
    "illicit_neighbours",
    "illicit_neighbour_frac",
    "near_illicit_2hop",
)

# Plain-English names for the UI and the case pack. A feature an investigator cannot read
# is not an explanation, and Elliptic's own columns are anonymised precisely because
# nobody bothered with this.
FEATURE_LABELS: dict[str, str] = {
    "in_degree": "number of distinct senders",
    "out_degree": "number of distinct recipients",
    "total_degree": "total counterparties",
    "degree_ratio": "recipients per sender",
    "total_in_btc": "total BTC received",
    "total_out_btc": "total BTC sent",
    "net_btc": "BTC retained",
    "fwd_ratio": "share of received funds forwarded on",
    "mean_in_btc": "average incoming transfer",
    "mean_out_btc": "average outgoing transfer",
    "max_in_btc": "largest incoming transfer",
    "max_out_btc": "largest outgoing transfer",
    "out_value_cv": "uniformity of outgoing amounts",
    "in_value_cv": "uniformity of incoming amounts",
    "n_tx_in": "incoming transaction count",
    "n_tx_out": "outgoing transaction count",
    "first_ts": "first activity time step",
    "last_ts": "last activity time step",
    "lifetime_ts": "active lifespan",
    "max_dormancy_ts": "longest dormant period",
    "max_in_burst": "most senders in one time step",
    "max_out_burst": "most recipients in one time step",
    "pagerank": "centrality in the money-flow graph",
    "clustering_coef": "how interconnected its counterparties are",
    "n_addresses": "addresses controlled",
    "hops_to_illicit": "hops to nearest known illicit entity",
    "illicit_neighbours": "directly connected known illicit entities",
    "illicit_neighbour_frac": "share of counterparties known illicit",
    "near_illicit_2hop": "within two hops of known illicit activity",
}
