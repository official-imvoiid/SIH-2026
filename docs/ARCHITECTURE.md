# Architecture

## Pipeline

```
  data/raw/                    ml/ingest/build.py  ml/features/clustering.py
  Elliptic + Elliptic++  ──▶   normalise      ──▶  co-spend union-find
  + BABD (optional)            to parquet          addresses -> entities
                                                          │
                                                          ▼
                                                  data/processed/
                                                  entities.parquet
                                                  edges.parquet
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────┐
                        ▼                                 ▼                         ▼
              ml/features/tabular.py           ml/features/graph.py        ml/typology/rules.py
              volume, timing, ratios           PageRank, k-hop illicit     FATF red flags
                        │                       density, centrality        (no training needed)
                        └────────────┬──────────────────┘                         │
                                     ▼                                            │
                        ml/train/baseline.py (GBDT)                                │
                        ml/train/gnn.py      (GraphSAGE, optional ensemble)        │
                                     │                                            │
                                     ▼                                            │
                        ml/explain/ (SHAP, GNNExplainer)                           │
                                     │                                            │
                                     └────────────────┬───────────────────────────┘
                                                      ▼
                                          ml/artifacts/  (model + cached scores)
                                                      │
                                                      ▼
                                          backend/  FastAPI, precomputed
                                                      │
                                                      ▼
                                          frontend/  React + Cytoscape
```

## Design decisions worth defending

**Entities, not transactions, are the unit of analysis.** Published work on Elliptic almost always
classifies transactions. An investigator does not arrest a transaction. Clustering addresses into
entities first is the step that makes the output actionable, and it is where our novelty sits.

**Scores are precomputed at startup, never inferred per request.** The model runs once over the
whole graph and writes `ml/artifacts/scores.parquet`. The API is a lookup. This keeps p95 latency
under 200 ms and means a slow or broken model cannot hang the demo.

**Subgraph size is capped server-side.** `GET /api/subgraph/{id}` returns at most ~300 nodes,
ranked by risk × edge value, with `truncated` and `total_available` always set. Rendering a hub
entity's real 2-hop neighbourhood would freeze the browser, and silently truncating would mislead
the analyst.

**The typology engine has no dependency on the model.** `ml/typology/rules.py` imports only
networkx. If the model, the GNN, or the whole ML stack fails on demo day, the graph view and the
red-flag explanations still work. This is deliberate demo insurance, and it is also the honest
answer to concept drift.

**Explanations are template-generated, not LLM-generated.** `narrate()` is deterministic: same
input, same sentence, every time. An LLM polish pass is an optional enhancement layered on top,
never a runtime dependency. Do not put a model download on the critical path of a live demo.

## Module ownership

See [`TEAM.md`](TEAM.md). Every directory has exactly one owner who reviews changes to it.

## Testing strategy

| Layer | What is tested | Runs without data? |
|---|---|---|
| `ml/typology/` | Every rule against hand-built graphs with known patterns | **Yes** |
| `ml/features/` | Clustering correctness on small synthetic co-spend sets | **Yes** |
| `backend/` | Endpoint shapes against `backend/tests/fixtures/` | **Yes** |
| `ml/train/` | Seeded metrics regression — F1 must not drop between commits | No |

Everything except model training runs on a clean clone with no dataset. That is what keeps the
team unblocked in sprint 1 and what makes the repo credible to a stranger on GitHub.
