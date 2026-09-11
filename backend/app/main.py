"""ChainTrace API.

Run::

    uvicorn backend.app.main:app --reload

Then open http://localhost:8000 for the investigator UI, or /docs for the API browser.

Every endpoint answers from memory (see ``services/store.py``) -- no model inference
happens in a request, so latency is flat and a broken model degrades the tool rather than
hanging it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ml.config import DEFAULT_HOPS, MAX_SUBGRAPH_NODES, MODEL_VERSION
from ml.typology.validation import validation_summary
from backend.app.services.casepack import render as render_casepack
from backend.app.services.store import store

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
FRONTEND_PUBLIC = Path(__file__).resolve().parent.parent.parent / "frontend" / "public"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    if store.ready:
        s = store.stats()
        print(
            f"  ChainTrace ready: {s['n_entities']:,} entities, {s['n_edges']:,} edges, "
            f"model_loaded={s['model_loaded']}"
        )
    else:
        print(f"  ChainTrace started WITHOUT data: {store.error}")
        print("  Run:  python -m ml.ingest.build  &&  python -m ml.train.baseline")
    yield


app = FastAPI(
    title="ChainTrace API",
    description=(
        "Entity-level illicit actor detection on the Bitcoin transaction graph. "
        "Risk scores indicate investigative priority, not proof of criminal conduct."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# The frontend dev server runs on a different port during development. Tightened for a
# real deployment; permissive here because everything is local and offline.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _require_ready() -> None:
    if not store.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Data not loaded: {store.error}. "
                "Run: python -m ml.ingest.build && python -m ml.train.baseline"
            ),
        )


@app.get("/api/health", tags=["meta"])
def health():
    """Liveness plus a full description of what is actually loaded."""
    if not store.ready:
        return JSONResponse(
            status_code=503,
            content={"status": "no_data", "error": store.error, "hint":
                     "python -m ml.ingest.build && python -m ml.train.baseline"},
        )
    # rule_validation tells the client which indicators have real-data evidence behind
    # them, so the interface can label flags rather than presenting all seven as equal.
    return {"status": "ok", **store.stats(), "rule_validation": validation_summary()}


@app.get("/api/metrics", tags=["meta"])
def metrics():
    """Model evaluation, including the concept-drift breakdown that drives the demo."""
    _require_ready()
    if not store.metrics:
        raise HTTPException(404, "No metrics.json -- run: python -m ml.train.baseline")
    return store.metrics


@app.get("/api/entity/{entity_id}", tags=["investigate"])
def entity(entity_id: str):
    """Full dossier for one entity: score, features, attributions, typologies, narrative."""
    _require_ready()
    if not store.exists(entity_id):
        raise HTTPException(404, f"Unknown entity {entity_id}")

    row = store.entities.loc[entity_id]
    attributions, method, _sentence = store.explain(entity_id)
    risk = store.risk(entity_id)

    return {
        "id": entity_id,
        "risk": risk,
        "band": ("high" if risk >= 0.7 else "medium" if risk >= 0.4 else "low"),
        "label": store.labels.get(entity_id, "unknown"),
        "n_addresses": int(row["n_addresses"]),
        "first_ts": int(row["first_ts"]),
        "last_ts": int(row["last_ts"]),
        "total_in_btc": round(float(row["total_in_btc"]), 8),
        "total_out_btc": round(float(row["total_out_btc"]), 8),
        "features": store.features(entity_id),
        "attributions": attributions,
        "attribution_method": method,
        "typologies": store.typologies(entity_id),
        "neighbors": store.neighbors(entity_id),
        "narrative": store.narrative(entity_id),
        "model_version": MODEL_VERSION,
        # Surfaced in the UI rather than buried: co-spend clustering under-merges, and an
        # analyst needs to know the entity boundary is an inference, not a fact.
        "clustering_note": (
            "Entity membership is inferred from the common-input-ownership heuristic. "
            "Addresses of the same owner that never co-spent may be missing."
        ),
    }


@app.get("/api/subgraph/{entity_id}", tags=["investigate"])
def subgraph(
    entity_id: str,
    hops: int = Query(DEFAULT_HOPS, ge=1, le=4),
    max_nodes: int = Query(MAX_SUBGRAPH_NODES, ge=10, le=1000),
):
    """Bounded k-hop money-flow neighbourhood, ranked when it must be truncated."""
    _require_ready()
    if not store.exists(entity_id):
        raise HTTPException(404, f"Unknown entity {entity_id}")
    return store.subgraph(entity_id, hops=hops, max_nodes=max_nodes)


@app.get("/api/search", tags=["investigate"])
def search(q: str = Query(..., min_length=1), limit: int = Query(25, ge=1, le=100)):
    """Search by entity ID or Bitcoin address."""
    _require_ready()
    return {"query": q, "results": store.search(q, limit=limit)}


@app.get("/api/top", tags=["investigate"])
def top(limit: int = Query(50, ge=1, le=500), min_risk: float = Query(0.0, ge=0.0, le=1.0)):
    """The triage queue -- highest-risk entities first."""
    _require_ready()
    return {"results": store.top_risk(limit=limit, min_risk=min_risk)}


@app.get("/api/case/{entity_id}", tags=["investigate"], response_class=HTMLResponse)
def case_pack(entity_id: str):
    """Full evidence document for one entity, printable to PDF from the browser.

    Returned as HTML rather than a generated PDF on purpose: every browser has a PDF
    writer, and a hackathon cannot afford a rendering dependency that fails to install on
    the demo machine.
    """
    _require_ready()
    if not store.exists(entity_id):
        raise HTTPException(status_code=404, detail=f"Unknown entity {entity_id}")
    return HTMLResponse(render_casepack(store, entity_id))


@app.get("/api/timeline", tags=["investigate"])
def timeline():
    """Per-time-step activity, for the scrubber and the drift demo."""
    _require_ready()
    return {"buckets": store.timeline()}


# -- static frontend -------------------------------------------------------------------
# Serves the built React app if it exists, else the zero-build HTML UI in frontend/public.
# The tool therefore works with no npm install at all, which matters on demo day.

_static_root = FRONTEND_DIST if (FRONTEND_DIST / "index.html").exists() else FRONTEND_PUBLIC

if (_static_root / "index.html").exists():
    if (_static_root / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_static_root / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_static_root / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        candidate = _static_root / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_static_root / "index.html")
