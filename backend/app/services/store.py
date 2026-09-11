"""In-memory store loaded once at startup.

Design rule from docs/ARCHITECTURE.md: **no model inference in the request path**. The
model runs once over the whole graph during training and writes ``scores.parquet``; this
store loads that, builds the graph, and answers every request from memory. Two reasons:
p95 latency stays flat, and a slow or broken model cannot hang a live demo.

Typology detection is the exception -- it is fast enough to run per request, and running
it lazily means new entities work without a rebuild.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import pandas as pd

from ml.config import ARTIFACTS, DATA_PROCESSED, MAX_SUBGRAPH_NODES, MODEL_VERSION
from ml.explain.attribution import attribution_sentence, explain_entity
from ml.features.graph import build_entity_graph
from ml.typology.rules import detect_all, narrate, structural
from ml.typology.validation import annotate, validation_summary

# Risk bands. These are presentation thresholds, not statistical ones -- an analyst reads
# a colour before they read a number, so the boundaries are documented in one place
# rather than scattered through the UI.
RISK_HIGH = 0.70
RISK_MEDIUM = 0.40


def risk_band(risk: float) -> str:
    if risk >= RISK_HIGH:
        return "high"
    if risk >= RISK_MEDIUM:
        return "medium"
    return "low"


class Store:
    """Everything the API serves, held in memory."""

    def __init__(self, processed: Path = DATA_PROCESSED, artifacts: Path = ARTIFACTS):
        self.processed = processed
        self.artifacts = artifacts
        self.ready = False
        self.error: str | None = None

        self.entities: pd.DataFrame = pd.DataFrame()
        self.edges: pd.DataFrame = pd.DataFrame()
        self.scores: dict[str, float] = {}
        self.labels: dict[str, str] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self.address_map: dict[str, str] = {}
        self.feature_cols: list[str] = []
        self.forest = None
        self.metrics: dict[str, Any] = {}
        self.build_report: dict[str, Any] = {}

    # -- loading -----------------------------------------------------------------------

    def load(self) -> None:
        try:
            self._load()
            self.ready = True
            self.error = None
        except Exception as exc:  # surfaced via /api/health rather than crashing the app
            self.ready = False
            self.error = str(exc)

    def _load(self) -> None:
        ent_path = self.processed / "entities.parquet"
        if not ent_path.exists():
            raise FileNotFoundError(
                f"{ent_path} missing. Run:  python -m ml.ingest.build"
            )

        self.entities = pd.read_parquet(ent_path).set_index("entity_id", drop=False)
        self.edges = pd.read_parquet(self.processed / "edges.parquet")
        self.labels = self.entities["label"].to_dict()

        addr_path = self.processed / "address_map.parquet"
        if addr_path.exists():
            am = pd.read_parquet(addr_path)
            self.address_map = dict(zip(am["address"], am["entity_id"]))

        report_path = self.processed / "build_report.json"
        if report_path.exists():
            self.build_report = json.loads(report_path.read_text(encoding="utf-8"))

        score_path = self.artifacts / "scores.parquet"
        if score_path.exists():
            sc = pd.read_parquet(score_path)
            self.scores = dict(zip(sc["entity_id"], sc["risk"].astype(float)))
        else:
            # The UI must still work before anyone has trained a model. Unscored entities
            # read as 0.0 and the health endpoint reports model_loaded=false.
            self.scores = {}

        model_path = self.artifacts / "model.joblib"
        if model_path.exists():
            bundle = joblib.load(model_path)
            self.forest = bundle.get("forest")
            self.feature_cols = list(bundle.get("feature_cols", []))

        metrics_path = self.artifacts / "metrics.json"
        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.graph = build_entity_graph(
            self.edges.rename(
                columns={"src_entity": "src", "dst_entity": "dst"}
            ).to_dict("records")
        )
        # Labels on nodes let the proximity typology rule work without a second lookup.
        for node in self.graph.nodes():
            self.graph.nodes[node]["label"] = self.labels.get(node, "unknown")

    # -- lookups -----------------------------------------------------------------------

    def risk(self, eid: str) -> float:
        return float(self.scores.get(eid, 0.0))

    def exists(self, eid: str) -> bool:
        return eid in self.entities.index

    def features(self, eid: str) -> dict[str, float]:
        if not self.exists(eid):
            return {}
        row = self.entities.loc[eid]
        return {
            c: float(row[c])
            for c in self.entities.columns
            if c not in ("entity_id", "label") and isinstance(row[c], (int, float))
            and not (isinstance(row[c], float) and math.isnan(row[c]))
        }

    @lru_cache(maxsize=4096)
    def _typologies_cached(self, eid: str) -> tuple:
        if eid not in self.graph:
            return ()
        return tuple(t.to_dict() for t in detect_all(self.graph, eid))

    def typologies(self, eid: str) -> list[dict]:
        """Typologies for one entity, each annotated with whether it has been shown to
        work on real Bitcoin. An untested indicator is still shown -- it is a genuine
        observation about behaviour -- but it is labelled, so nobody reads it as
        equivalent to a validated one."""
        return [annotate(t) for t in self._typologies_cached(eid)]

    def explain(self, eid: str, top_k: int = 8) -> tuple[list[dict], str, str]:
        """Returns ``(attributions, method, sentence)``."""
        if self.forest is None or not self.feature_cols or not self.exists(eid):
            return [], "unavailable", ""
        feats = self.features(eid)
        attrs, method = explain_entity(self.forest, self.feature_cols, feats, top_k=top_k)
        return [a.to_dict() for a in attrs], method, attribution_sentence(attrs)

    def narrative(self, eid: str) -> str:
        if eid not in self.graph:
            return ""
        hits = detect_all(self.graph, eid)
        base = narrate(eid, self.risk(eid), hits)
        _attrs, _method, sentence = self.explain(eid, top_k=6)
        return f"{base} {sentence}".strip() if sentence else base

    def neighbors(self, eid: str, limit: int = 60) -> list[dict]:
        if eid not in self.graph:
            return []
        out: list[dict] = []
        for u, _v, d in self.graph.in_edges(eid, data=True):
            out.append(
                {
                    "id": u,
                    "risk": self.risk(u),
                    "label": self.labels.get(u, "unknown"),
                    "direction": "in",
                    "value_btc": round(d["value_btc"], 8),
                    "n_txs": d["n_txs"],
                }
            )
        for _u, v, d in self.graph.out_edges(eid, data=True):
            out.append(
                {
                    "id": v,
                    "risk": self.risk(v),
                    "label": self.labels.get(v, "unknown"),
                    "direction": "out",
                    "value_btc": round(d["value_btc"], 8),
                    "n_txs": d["n_txs"],
                }
            )
        # Rank by risk first, then by value: an analyst wants the dangerous counterparty
        # before the merely large one.
        out.sort(key=lambda n: (-n["risk"], -n["value_btc"]))
        return out[:limit]

    def subgraph(
        self, seed: str, hops: int = 2, max_nodes: int = MAX_SUBGRAPH_NODES
    ) -> dict:
        """Extract a bounded k-hop neighbourhood around ``seed``.

        The cap is not cosmetic. A hub entity's real 2-hop neighbourhood runs to tens of
        thousands of nodes, which freezes the browser and tells the analyst nothing. We
        expand breadth-first, keep the highest-value/highest-risk frontier, and always
        report ``truncated`` and ``total_available`` so a partial view is never mistaken
        for the whole picture.
        """
        if seed not in self.graph:
            return {
                "nodes": [], "edges": [], "seed_id": seed, "hops": hops,
                "truncated": False, "total_available": 0, "ranking": "risk_x_value",
            }

        undirected = self.graph.to_undirected(as_view=True)
        selected = {seed}
        frontier = {seed}
        total_reachable = {seed}

        for _ in range(max(hops, 0)):
            candidates: dict[str, float] = {}
            for node in frontier:
                for nbr in undirected.neighbors(node):
                    total_reachable.add(nbr)
                    if nbr in selected:
                        continue
                    weight = 0.0
                    if self.graph.has_edge(node, nbr):
                        weight += self.graph[node][nbr]["value_btc"]
                    if self.graph.has_edge(nbr, node):
                        weight += self.graph[nbr][node]["value_btc"]
                    # Score = risk x log-value. Risk dominates, value breaks ties, and the
                    # log stops one whale edge from crowding out everything else.
                    score = (0.1 + self.risk(nbr)) * math.log1p(weight)
                    candidates[nbr] = max(candidates.get(nbr, 0.0), score)

            room = max_nodes - len(selected)
            if room <= 0:
                break
            ranked = sorted(candidates.items(), key=lambda kv: -kv[1])[:room]
            frontier = {n for n, _s in ranked}
            selected |= frontier
            if not frontier:
                break

        sub = self.graph.subgraph(selected)
        nodes = [
            {
                "id": n,
                "risk": self.risk(n),
                "label": self.labels.get(n, "unknown"),
                "band": risk_band(self.risk(n)),
                "size": float(
                    self.entities.loc[n]["total_in_btc"] if n in self.entities.index else 1.0
                ),
                "n_addresses": int(
                    self.entities.loc[n]["n_addresses"] if n in self.entities.index else 1
                ),
                "is_seed": n == seed,
            }
            for n in sub.nodes()
        ]
        edges = [
            {
                "source": u,
                "target": v,
                "value_btc": round(d["value_btc"], 8),
                "n_txs": d["n_txs"],
                "ts": d["ts"],
            }
            for u, v, d in sub.edges(data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "seed_id": seed,
            "hops": hops,
            "truncated": len(total_reachable) > len(selected),
            "total_available": len(total_reachable),
            "ranking": "risk_x_log_value",
        }

    def search(self, q: str, limit: int = 25) -> list[dict]:
        """Search by entity ID or by Bitcoin address.

        Address search is what an investigator actually starts from -- they have a wallet
        from a victim report, not an internal cluster ID.
        """
        q = q.strip()
        if not q:
            return []
        results: list[dict] = []

        if q in self.address_map:
            eid = self.address_map[q]
            results.append(
                {
                    "id": eid,
                    "risk": self.risk(eid),
                    "label": self.labels.get(eid, "unknown"),
                    "n_addresses": int(self.entities.loc[eid]["n_addresses"])
                    if eid in self.entities.index
                    else 1,
                    "matched_address": q,
                }
            )

        ql = q.lower()
        for eid in self.entities.index:
            if len(results) >= limit:
                break
            if ql in eid.lower() and not any(r["id"] == eid for r in results):
                results.append(
                    {
                        "id": eid,
                        "risk": self.risk(eid),
                        "label": self.labels.get(eid, "unknown"),
                        "n_addresses": int(self.entities.loc[eid]["n_addresses"]),
                        "matched_address": None,
                    }
                )
        return results[:limit]

    def top_risk(self, limit: int = 50, min_risk: float = 0.0) -> list[dict]:
        """The triage queue: highest-risk entities, which is how the tool is really used."""
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        out = []
        for eid, risk in ranked:
            if risk < min_risk or len(out) >= limit:
                break
            if eid not in self.entities.index:
                continue
            row = self.entities.loc[eid]
            hits = self.typologies(eid)
            out.append(
                {
                    "id": eid,
                    "risk": risk,
                    "band": risk_band(risk),
                    "label": self.labels.get(eid, "unknown"),
                    "n_addresses": int(row["n_addresses"]),
                    "total_in_btc": round(float(row["total_in_btc"]), 8),
                    "first_ts": int(row["first_ts"]),
                    "last_ts": int(row["last_ts"]),
                    "n_typologies": len(hits),
                    # Structural = evidence about this entity's own conduct. Counted
                    # separately so the UI can filter out pure guilt-by-association,
                    # which is never a sufficient basis for a lead.
                    "n_structural": sum(1 for h in hits if not h.get("corroborating")),
                    "top_typology": next(
                        (h["name"] for h in hits if not h.get("corroborating")),
                        hits[0]["name"] if hits else None,
                    ),
                }
            )
        return out

    def timeline(self) -> list[dict]:
        if self.edges.empty:
            return []
        grouped = self.edges.groupby("ts").agg(
            n_txs=("n_txs", "sum"), volume_btc=("value_btc", "sum")
        )
        illicit_ids = {e for e, lab in self.labels.items() if lab == "illicit"}
        illicit_by_ts = (
            self.edges[self.edges["src_entity"].isin(illicit_ids)]
            .groupby("ts")["n_txs"].sum()
            .to_dict()
        )
        return [
            {
                "ts": int(ts),
                "n_txs": int(row["n_txs"]),
                "volume_btc": round(float(row["volume_btc"]), 4),
                "n_illicit": int(illicit_by_ts.get(ts, 0)),
            }
            for ts, row in grouped.iterrows()
        ]

    def stats(self) -> dict:
        bands = {"high": 0, "medium": 0, "low": 0}
        for r in self.scores.values():
            bands[risk_band(r)] += 1
        return {
            "n_entities": int(len(self.entities)),
            "n_edges": int(len(self.edges)),
            "n_addresses": len(self.address_map),
            "labels": self.entities["label"].value_counts().to_dict()
            if not self.entities.empty
            else {},
            "risk_bands": bands,
            "model_version": MODEL_VERSION,
            "model_loaded": self.forest is not None,
            "provenance": self.build_report.get("provenance", "unknown"),
            "clustering_quality": self.build_report.get("clustering_quality", {}),
        }


store = Store()
