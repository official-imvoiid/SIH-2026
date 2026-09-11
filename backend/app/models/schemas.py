"""Frozen API contract.

This file is the interface between the six workstreams (see docs/TEAM.md). Changing any
field here means announcing it to the whole team the same day -- the frontend, the model
layer, and the report generator all bind to these shapes.

Everything here is representable from fixtures, so the frontend can be built end to end
before a single model has been trained.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Label(str, Enum):
    ILLICIT = "illicit"
    LICIT = "licit"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Direction(str, Enum):
    IN = "in"
    OUT = "out"


class ShapContribution(BaseModel):
    """One feature's contribution to the risk score, from the tree model."""

    feature: str
    value: float = Field(description="The raw feature value for this entity")
    contribution: float = Field(
        description="Signed SHAP value; positive pushes toward illicit"
    )


class TypologyHit(BaseModel):
    """A fired FATF red-flag indicator. Mirrors ml.typology.rules.Typology."""

    code: str = Field(examples=["VA-PEELCHAIN"])
    name: str = Field(examples=["Peel chain"])
    severity: Severity
    description: str = Field(description="Human-readable; goes verbatim into the case pack")
    evidence: dict = Field(default_factory=dict)


class NeighborRef(BaseModel):
    id: str
    risk: float = Field(ge=0.0, le=1.0)
    label: Label = Label.UNKNOWN
    direction: Direction
    value_btc: float
    n_txs: int = 1


class EntityDetail(BaseModel):
    """Response for GET /api/entity/{id} -- the core object of the whole application."""

    id: str
    risk: float = Field(ge=0.0, le=1.0, description="Calibrated illicit probability")
    label: Label = Field(description="Ground-truth dataset label, if any")
    n_addresses: int = Field(description="Addresses merged into this entity by clustering")
    first_seen_ts: int
    last_seen_ts: int
    total_in_btc: float
    total_out_btc: float

    features: dict[str, float] = Field(default_factory=dict)
    shap: list[ShapContribution] = Field(default_factory=list)
    typologies: list[TypologyHit] = Field(default_factory=list)
    neighbors: list[NeighborRef] = Field(default_factory=list)
    narrative: str = Field(
        default="", description="Deterministic plain-language summary for investigators"
    )

    # Honesty fields. These are rendered in the UI, not hidden in a log.
    model_version: str = ""
    clustering_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Lower when CoinJoin-like patterns make a false merge plausible",
    )


class GraphNode(BaseModel):
    id: str
    risk: float = Field(ge=0.0, le=1.0)
    label: Label = Label.UNKNOWN
    size: float = Field(default=1.0, description="Render hint: scaled by total volume")
    is_seed: bool = False


class GraphEdge(BaseModel):
    source: str
    target: str
    value_btc: float
    n_txs: int = 1
    ts: int


class SubgraphResponse(BaseModel):
    """Response for GET /api/subgraph/{id}.

    ``truncated`` and ``total_available`` are mandatory, not optional. A hub entity has a
    50k-node 2-hop neighbourhood; the server caps what it returns and the UI must be able
    to tell the analyst that it is showing a ranked subset rather than the whole picture.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    seed_id: str
    hops: int
    truncated: bool = False
    total_available: int = 0
    ranking: str = Field(
        default="risk_x_value",
        description="How the returned subset was chosen when truncated",
    )


class SearchResult(BaseModel):
    id: str
    risk: float
    label: Label
    n_addresses: int
    matched_address: str | None = None


class TimelineBucket(BaseModel):
    ts: int
    n_txs: int
    volume_btc: float
    n_illicit: int


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str
    n_entities: int
    n_edges: int
    dataset_version: str
