# Team Plan — 6 people, 5 sprints

## Current status

A working end-to-end implementation is already in the repo: synthetic data generator,
co-spend clustering, entity features, trained model, attribution, typology engine, API and
UI. `python run.py` builds and serves all of it in about a minute.

That changes what these roles mean. Nobody is starting from a blank file -- each role now
owns **improving and replacing** a working baseline, which is a much better position to
start a hackathon from. The highest-value open work, in order:

1. **Wire the real Elliptic++ loader** (`ml/ingest/build.py::_load_real`) -- one function;
   everything downstream already works unchanged. This is the single biggest win.
2. **Tune the typology thresholds** on real labels (`ml/typology/rules.py`, top of file).
3. **Add the GNN benchmark** and report honestly whether it beats the forest.
4. **PDF case-pack export** -- the deliverable an investigator actually keeps.
5. **OFAC / Ransomwhere tag cross-reference** -- turns a cluster into a named lead.

## Design principle: contracts before code

The single biggest failure mode for a 6-person hackathon team is everyone waiting on everyone
else. We avoid it by **freezing the data contracts on day 1** and having everyone code against
fixtures, not against each other's unfinished modules.

On day 1 we commit `data/processed/SCHEMA.md` and `backend/app/models/schemas.py` with the exact
column names and JSON shapes below. After that, all six workstreams run in parallel. Nobody is
blocked; if an upstream module is late, you work against the fixture and swap it later.

### The three frozen contracts

```
entities.parquet     entity_id, n_addresses, first_seen_ts, last_seen_ts,
                     total_in_btc, total_out_btc, label {illicit|licit|unknown}

edges.parquet        src_entity, dst_entity, ts, value_btc, n_txs

GET /api/entity/{id} -> { id, risk, label, features,
                          shap: [{feature, value, contribution}],
                          typologies: [{code, name, severity, evidence}],
                          neighbors: [{id, risk, direction, value_btc}] }
```

Fixture files (~200 rows of realistic fake data) land in `backend/tests/fixtures/` on day 1 so the
frontend can build the whole UI before the model exists.

---

## Role assignments

Each role owns directories, not tasks. You are the reviewer for anything touching your
directories. Nobody merges to `main` without one review from the owner of the touched area.

### 1 — Data & Entity Resolution · owns `ml/ingest/`, `ml/features/clustering.py`, `scripts/`

The hardest, least glamorous, most load-bearing role. Give it to whoever is most stubborn.

- Download, checksum, and version the three datasets
- Elliptic / Elliptic++ / BABD joiners into one canonical parquet store (DuckDB)
- **Co-spend union-find clustering** — addresses into entities. This is the project's core IP
- CoinJoin detection and exclusion (equal-output-value heuristic) to limit false merges
- Publishes `entities.parquet` and `edges.parquet`. Everyone downstream depends on these
- **Done when:** `python run.py --rebuild` runs clean from a fresh clone in under 10 minutes

### 2 — Tabular ML · owns `ml/train/baseline.py`, `ml/features/tabular.py`

This role produces the number that goes on the slide. Weber et al. got 0.79 illicit F1; beat it.

- LightGBM / XGBoost / RandomForest on entity-level features
- **Temporal split, never random split.** Train on early timesteps, test on later ones. A random
  split leaks the future and inflates F1 by roughly ten points; a judge who knows the field
  will ask which split you used
- Class-imbalance handling (`scale_pos_weight`, focal loss), probability calibration
- Produces the before/after-timestep-43 drift table for the pitch
- **Done when:** reproducible seeded metrics land in `ml/artifacts/metrics.json`

### 3 — Graph ML · owns `ml/train/gnn.py`, `ml/features/graph.py`

- Derived graph features first: k-hop illicit density, PageRank, betweenness, degree over a time
  window, distance-to-nearest-known-illicit (the GuiltyWalker idea)
- **These features alone, fed to role 2's GBDT, may beat the GNN.** Ship them before the GNN
- Then PyTorch Geometric GraphSAGE / GAT on the entity graph
- Ensemble: GNN embeddings concatenated onto tabular features (the Weber et al. "AF+NE" recipe)
- **Done when:** if the GNN does not beat the GBDT by sprint 4, we ship the GBDT and say so in
  the README. That is a legitimate, publishable result — do not fake it

### 4 — Explainability & Typologies · owns `ml/explain/`, `ml/typology/`

The differentiator role. This is what makes it a forensics tool instead of a classifier.

- SHAP over the tree model; cache attributions so the API stays fast
- **Rule engine for FATF red-flag typologies** — peel chain, layering, fan-in / fan-out, rapid
  pass-through, structuring, dormant-then-burst. Pure graph algorithms, no ML, always works
- GNNExplainer for the subgraph highlight
- **Natural-language narrative generator.** Template-based, deterministic, no LLM required.
  Optional stretch: local Ollama for prose polish — but the templates must work standalone,
  because demo-day GPU availability is not a risk worth taking
- **Done when:** every flagged entity returns at least one human-readable sentence with evidence

### 5 — Backend · owns `backend/`

- FastAPI: entity lookup, k-hop subgraph extraction, search, timeline, case CRUD
- **Subgraph extraction with a hard node cap.** A 2-hop neighbourhood on a hub entity is 50k nodes
  and will freeze the browser. Cap at ~300, rank by risk times edge value, paginate the rest
- Precompute and cache scores at startup — no model inference in the request path
- PDF case-pack export (Jinja2 + WeasyPrint)
- **Done when:** p95 latency under 200 ms on every endpoint with the full dataset loaded

### 6 — Frontend & Delivery · owns `frontend/`, `docs/`, demo

- React + Vite + TypeScript, Tailwind, shadcn/ui
- **Graph canvas: Cytoscape.js** for the 300-node investigation subgraph — easy, good layouts.
  Only reach for sigma.js + graphology if you truly need to render more than 5k nodes
- Time scrubber, risk filter, entity search, evidence side panel, case builder
- Owns the README screenshots, the demo script, and the pitch deck
- **Done when:** a stranger can clone the repo and get a working UI from the README alone

---

## Sprint plan

| Sprint | Shared goal | What must be true at the end |
|---|---|---|
| **1** | Contracts, fixtures, skeletons | Frontend renders a fake graph; API returns fixture JSON; datasets downloaded |
| **2** | Real data flowing | Entity clustering works; GBDT baseline has a real F1 number |
| **3** | Intelligence layer | SHAP and typology rules firing; graph features in the model |
| **4** | Integration | Click a node, get a real score and a real explanation. **Feature freeze** |
| **5** | Polish and pitch | Drift demo rehearsed; PDF export works; README complete; demo run 5x on the demo laptop |

**Sprint 4 feature freeze is non-negotiable.** Teams lose on a broken demo, never on a missing
feature. Sprint 5 exists to make what you already have work perfectly.

---

## Working agreements

- **Branches:** `feat/<role-number>-<short-name>`. PRs into `main`, one review from the owner of
  the touched area.
- **No notebooks on `main`.** Explore in `notebooks/`, but anything that runs in the demo is a
  module in `ml/` or `backend/` with a test.
- **Seeds fixed everywhere** (`RANDOM_SEED = 42` in `ml/config.py`). Unreproducible metrics are
  not metrics.
- **Daily 15-minute standup**, three questions: what is done, what is blocked, what contract
  changed. A changed contract is announced to the whole team the moment it changes.
- **The demo laptop is chosen in sprint 4** and everything is tested on it, not on your machine.
  Model artifacts travel via Git LFS or a USB drive. Zero network calls on demo day.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| GNN underperforms the GBDT | **High** — the literature says so | Ship the GBDT. Report the GNN result honestly; it is a finding |
| Browser freezes rendering the graph | High | Hard 300-node cap enforced server-side, from day 1 |
| Clustering false-merges via CoinJoin | Medium | Detect equal-output-value transactions and exclude; document the residual error |
| Dataset download fails on demo day | Medium | Everything cached locally by sprint 4; no network in the demo path |
| Judges challenge label quality | **Certain** | Prepared answer: heuristic vendor labels, we predict risk not guilt. Say it before they ask |
| One member's module blocks everyone | Medium | Fixtures from day 1 mean nobody is ever hard-blocked |
