"""FATF red-flag typology detection over the entity graph.

These are deterministic graph algorithms, not machine learning. That matters for three
reasons:

1. They keep working when the statistical model hits concept drift (see docs/RESEARCH.md
   section 4 -- the timestep-43 collapse). Structural laundering patterns survive when
   learned feature distributions do not.
2. Every hit maps to a *named, published* FATF indicator, so a flag reads as investigative
   evidence rather than an opaque score.
3. They need no training data, so they work on entities the model has never seen.

Reference: FATF (2020), "Virtual Assets Red Flag Indicators of Money Laundering and
Terrorist Financing."

Graph contract (see docs/TEAM.md): a networkx.DiGraph where every edge carries
``ts`` (int timestep) and ``value_btc`` (float).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

__all__ = ["Typology", "detect_all", "RULES"]


@dataclass(frozen=True)
class Typology:
    """One fired red-flag indicator against one entity."""

    code: str
    name: str
    severity: str  # "low" | "medium" | "high"
    description: str  # human-readable, goes straight into the case pack
    evidence: dict[str, Any] = field(default_factory=dict)
    # Corroborating indicators describe an entity's *surroundings*, not its own conduct.
    # They support a case that already stands on structural evidence; they must never be
    # the reason a flag is raised, so they are always ranked last and never lead the
    # narrative. Guilt by association is not evidence.
    corroborating: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "corroborating": self.corroborating,
        }


# --------------------------------------------------------------------------------------
# Thresholds. Tune these on the labelled set -- do not leave them at these defaults for
# the final submission. Role 4 owns calibrating them (see docs/TEAM.md).
# --------------------------------------------------------------------------------------

FAN_MIN_COUNTERPARTIES = 10
FAN_WINDOW_TS = 1
PASSTHROUGH_MIN_FWD_RATIO = 0.90
PASSTHROUGH_MAX_HOLD_TS = 1
PEEL_MIN_HOPS = 4
PEEL_MAX_PEEL_RATIO = 0.25
STRUCTURING_MIN_TRANSFERS = 8
STRUCTURING_MAX_CV = 0.10  # coefficient of variation across transfer sizes
DORMANCY_MIN_GAP_TS = 18
DORMANCY_MIN_POST_TXS = 4


def _in_edges(g: nx.DiGraph, node: Any) -> list[tuple]:
    return [(u, v, d) for u, v, d in g.in_edges(node, data=True)]


def _out_edges(g: nx.DiGraph, node: Any) -> list[tuple]:
    return [(u, v, d) for u, v, d in g.out_edges(node, data=True)]


def detect_fan_in(
    g: nx.DiGraph,
    node: Any,
    min_counterparties: int | None = None,
    window_ts: int | None = None,
) -> Typology | None:
    """Many distinct senders converging on one entity in a short window.

    The classic consolidation step: victim payments, mule accounts, or scam proceeds being
    swept into a single collection wallet. FATF indicator group: structuring / consolidation.

    Thresholds default to the module constants but can be overridden, which is how
    ``ml.typology.calibrate`` sweeps them against real data without editing this file.
    """
    min_counterparties = (
        FAN_MIN_COUNTERPARTIES if min_counterparties is None else min_counterparties
    )
    window_ts = FAN_WINDOW_TS if window_ts is None else window_ts

    edges = _in_edges(g, node)
    if len(edges) < min_counterparties:
        return None

    by_window: dict[int, set] = {}
    value_by_window: dict[int, float] = {}
    for u, _v, d in edges:
        w = d["ts"] // max(window_ts, 1)
        by_window.setdefault(w, set()).add(u)
        value_by_window[w] = value_by_window.get(w, 0.0) + d.get("value_btc", 0.0)

    window, senders = max(by_window.items(), key=lambda kv: len(kv[1]))
    if len(senders) < min_counterparties:
        return None

    return Typology(
        code="VA-CONSOLIDATION",
        name="Fan-in / consolidation",
        severity="high" if len(senders) >= 2 * min_counterparties else "medium",
        description=(
            f"Received funds from {len(senders)} distinct entities within a single time "
            f"window, totalling {value_by_window[window]:.4f} BTC. Consolidation from many "
            f"unrelated counterparties is a recognised collection-wallet pattern."
        ),
        evidence={
            "distinct_senders": len(senders),
            "window_ts": window * max(window_ts, 1),
            "total_btc": round(value_by_window[window], 8),
            "sample_senders": sorted(senders, key=str)[:10],
        },
    )


def detect_fan_out(
    g: nx.DiGraph,
    node: Any,
    min_counterparties: int | None = None,
    window_ts: int | None = None,
) -> Typology | None:
    """One entity dispersing to many distinct recipients in a short window.

    The distribution step of layering -- breaking a lump sum into many smaller flows to
    frustrate tracing.

    Caveat established by measurement: on real Bitcoin this fires *more* on ordinary money
    than on ransomware, because exchanges legitimately pay out to thousands. Pair it with
    :func:`is_service_like` before treating a hit as suspicious.
    """
    min_counterparties = (
        FAN_MIN_COUNTERPARTIES if min_counterparties is None else min_counterparties
    )
    window_ts = FAN_WINDOW_TS if window_ts is None else window_ts

    edges = _out_edges(g, node)
    if len(edges) < min_counterparties:
        return None

    by_window: dict[int, set] = {}
    value_by_window: dict[int, float] = {}
    for _u, v, d in edges:
        w = d["ts"] // max(window_ts, 1)
        by_window.setdefault(w, set()).add(v)
        value_by_window[w] = value_by_window.get(w, 0.0) + d.get("value_btc", 0.0)

    window, recipients = max(by_window.items(), key=lambda kv: len(kv[1]))
    if len(recipients) < min_counterparties:
        return None

    return Typology(
        code="VA-DISPERSION",
        name="Fan-out / dispersion",
        severity="high" if len(recipients) >= 2 * min_counterparties else "medium",
        description=(
            f"Dispersed {value_by_window[window]:.4f} BTC to {len(recipients)} distinct "
            f"entities within a single time window. Rapid dispersion to many counterparties "
            f"is a layering indicator."
        ),
        evidence={
            "distinct_recipients": len(recipients),
            "window_ts": window * max(window_ts, 1),
            "total_btc": round(value_by_window[window], 8),
            "sample_recipients": sorted(recipients, key=str)[:10],
        },
    )


def detect_rapid_passthrough(
    g: nx.DiGraph,
    node: Any,
    min_fwd_ratio: float | None = None,
    max_hold_ts: int | None = None,
) -> Typology | None:
    """Funds arrive and leave almost immediately, with almost nothing retained.

    A pass-through wallet holds no economic position -- it exists only to add a hop. This
    is one of the strongest single indicators of a laundering intermediary.
    """
    min_fwd_ratio = (
        PASSTHROUGH_MIN_FWD_RATIO if min_fwd_ratio is None else min_fwd_ratio
    )
    max_hold_ts = PASSTHROUGH_MAX_HOLD_TS if max_hold_ts is None else max_hold_ts

    ins, outs = _in_edges(g, node), _out_edges(g, node)
    if not ins or not outs:
        return None

    total_in = sum(d.get("value_btc", 0.0) for *_x, d in ins)
    total_out = sum(d.get("value_btc", 0.0) for *_x, d in outs)
    if total_in <= 0:
        return None

    fwd_ratio = total_out / total_in
    if fwd_ratio < min_fwd_ratio:
        return None

    first_in = min(d["ts"] for *_x, d in ins)
    last_out = max(d["ts"] for *_x, d in outs)
    hold = last_out - first_in
    if hold > max_hold_ts or hold < 0:
        return None

    return Typology(
        code="VA-PASSTHROUGH",
        name="Rapid pass-through",
        severity="high",
        description=(
            f"Forwarded {fwd_ratio:.1%} of {total_in:.4f} BTC received, within {hold} time "
            f"step(s) of receipt, retaining effectively no balance. Wallets that hold no "
            f"position are characteristic laundering intermediaries rather than end users."
        ),
        evidence={
            "forward_ratio": round(fwd_ratio, 4),
            "hold_duration_ts": hold,
            "total_in_btc": round(total_in, 8),
            "total_out_btc": round(total_out, 8),
        },
    )


def detect_peel_chain(
    g: nx.DiGraph,
    node: Any,
    max_depth: int = 12,
    min_hops: int | None = None,
    max_peel_ratio: float | None = None,
) -> Typology | None:
    """A repeated peel: small amount split off, large remainder forwarded onward.

    Peel chains launder a large sum by shaving off spendable amounts over many hops while
    the bulk keeps moving. Detected by walking the dominant-value path and counting
    consecutive hops that fit the peel shape.
    """
    min_hops = PEEL_MIN_HOPS if min_hops is None else min_hops
    max_peel_ratio = PEEL_MAX_PEEL_RATIO if max_peel_ratio is None else max_peel_ratio

    chain = [node]
    current = node
    peels: list[float] = []

    for _ in range(max_depth):
        outs = _out_edges(g, current)
        if len(outs) != 2:  # a canonical peel has exactly a payment and a change output
            break

        outs.sort(key=lambda e: e[2].get("value_btc", 0.0))
        small, large = outs[0], outs[1]
        s_val = small[2].get("value_btc", 0.0)
        l_val = large[2].get("value_btc", 0.0)
        total = s_val + l_val
        if total <= 0 or (s_val / total) > max_peel_ratio:
            break

        peels.append(s_val)
        current = large[1]
        if current in chain:  # cycle guard
            break
        chain.append(current)

    hops = len(peels)
    if hops < min_hops:
        return None

    return Typology(
        code="VA-PEELCHAIN",
        name="Peel chain",
        severity="high",
        description=(
            f"Origin of a {hops}-hop peel chain: at each hop a small amount "
            f"(mean {sum(peels) / hops:.6f} BTC) was split off while the remainder was "
            f"forwarded to a fresh entity. Peel chains are used to launder a large sum in "
            f"spendable increments while keeping the bulk in motion."
        ),
        evidence={
            "chain_length": hops,
            "total_peeled_btc": round(sum(peels), 8),
            "mean_peel_btc": round(sum(peels) / hops, 8),
            "chain_head": [str(n) for n in chain[:6]],
        },
    )


def detect_structuring(
    g: nx.DiGraph,
    node: Any,
    min_transfers: int | None = None,
    max_cv: float | None = None,
) -> Typology | None:
    """Many near-identical transfer amounts -- smurfing to stay under reporting limits."""
    min_transfers = STRUCTURING_MIN_TRANSFERS if min_transfers is None else min_transfers
    max_cv = STRUCTURING_MAX_CV if max_cv is None else max_cv

    outs = _out_edges(g, node)
    values = [d.get("value_btc", 0.0) for *_x, d in outs if d.get("value_btc", 0.0) > 0]
    if len(values) < min_transfers:
        return None

    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    cv = (variance**0.5) / mean
    if cv > max_cv:
        return None

    return Typology(
        code="VA-STRUCTURING",
        name="Structuring / smurfing",
        severity="medium",
        description=(
            f"Sent {len(values)} outbound transfers of near-identical size "
            f"(mean {mean:.6f} BTC, coefficient of variation {cv:.3f}). Uniform transfer "
            f"sizing suggests deliberate splitting rather than organic payment activity."
        ),
        evidence={
            "n_transfers": len(values),
            "mean_btc": round(mean, 8),
            "coefficient_of_variation": round(cv, 4),
        },
    )


def detect_dormant_burst(
    g: nx.DiGraph,
    node: Any,
    min_gap_ts: int | None = None,
    min_post_txs: int | None = None,
) -> Typology | None:
    """Long inactivity followed by sudden high-value movement.

    Typical of a wallet reactivated to move proceeds after a cooling-off period.
    """
    min_gap_ts = DORMANCY_MIN_GAP_TS if min_gap_ts is None else min_gap_ts
    min_post_txs = DORMANCY_MIN_POST_TXS if min_post_txs is None else min_post_txs

    stamps = sorted(
        d["ts"] for *_x, d in list(_in_edges(g, node)) + list(_out_edges(g, node))
    )
    if len(stamps) < 3:
        return None

    gaps = [(stamps[i + 1] - stamps[i], i) for i in range(len(stamps) - 1)]
    max_gap, idx = max(gaps)
    if max_gap < min_gap_ts:
        return None

    after = [s for s in stamps if s > stamps[idx]]
    if len(after) < min_post_txs:
        return None

    return Typology(
        code="VA-DORMANCY",
        name="Dormant-then-burst",
        severity="medium",
        description=(
            f"Inactive for {max_gap} time steps, then produced {len(after)} transactions "
            f"after reactivation. Reactivation after prolonged dormancy is a recognised "
            f"indicator of proceeds being moved after a cooling-off period."
        ),
        evidence={
            "dormancy_ts": max_gap,
            "reactivation_ts": stamps[idx + 1],
            "post_reactivation_txs": len(after),
        },
    )


def detect_illicit_proximity(g: nx.DiGraph, node: Any, hops: int = 1) -> Typology | None:
    """Known-illicit entities among the entity's own direct counterparties.

    Deliberately restricted to **one hop**. An earlier two-hop version fired on roughly
    69% of all entities: in a small-world payment graph almost everything sits two hops
    from almost everything else, so the indicator carried no information while looking
    authoritative. One hop means "this entity transacted directly with a known illicit
    party", which is a fact about its own conduct rather than about the network's shape.

    Always marked corroborating, and capped at medium severity no matter how many hits:
    an exchange transacts with everyone, and would otherwise be permanently flagged.
    """
    if node not in g:
        return None

    counterparties = set(g.predecessors(node)) | set(g.successors(node))
    counterparties.discard(node)
    illicit = [n for n in counterparties if g.nodes[n].get("label") == "illicit"]
    if not illicit:
        return None

    share = len(illicit) / max(len(counterparties), 1)
    # A single illicit counterparty out of hundreds is a coincidence; a meaningful share
    # of a small counterparty set is not.
    if len(illicit) < 2 and share < 0.25:
        return None

    return Typology(
        code="VA-PROXIMITY",
        name="Direct exposure to known illicit entities",
        severity="medium" if (len(illicit) >= 3 or share >= 0.5) else "low",
        description=(
            f"Transacted directly with {len(illicit)} entity/entities independently "
            f"labelled illicit ({share:.0%} of its {len(counterparties)} counterparties). "
            f"Corroborating indicator only -- it does not establish the subject's own "
            f"status, and high-volume services legitimately touch illicit parties."
        ),
        evidence={
            "n_illicit_counterparties": len(illicit),
            "n_counterparties": len(counterparties),
            "illicit_share": round(share, 4),
            "sample": [str(n) for n in sorted(illicit, key=str)[:10]],
        },
        corroborating=True,
    )


# Thresholds for the service guard below. Deliberately generous: the cost of wrongly
# calling a laundering hop a "service" is a missed flag, while the cost of calling an
# exchange a launderer is an innocent business in a police case file.
SERVICE_MIN_DEGREE = 40
SERVICE_MIN_ACTIVE_WINDOWS = 6


def is_service_like(
    g: nx.DiGraph,
    node: Any,
    min_degree: int | None = None,
    min_active_windows: int | None = None,
) -> bool:
    """Is this entity a high-volume service rather than a laundering intermediary?

    Why this exists
    ---------------
    Measured against real Bitcoin, ``detect_fan_out`` fired **three times more often on
    ordinary money than on ransomware money**. The reason is structural and not fixable by
    tightening the fan-out threshold alone: an exchange, mining pool or payment processor
    genuinely does receive from thousands of counterparties and pay out to thousands. On
    shape alone it is indistinguishable from layering.

    What separates them is *persistence*. A laundering hop is transient -- it appears,
    moves value, and goes quiet. A service operates continuously across the whole
    observation window. So we use sustained high degree as a proxy for "this is somebody's
    business", and let the caller suppress dispersion flags on those entities.

    This is a heuristic standing in for what commercial tools do with a curated exchange
    tag list. It is weaker than a real tag list and should be replaced by one if the team
    can obtain or build it.
    """
    min_degree = SERVICE_MIN_DEGREE if min_degree is None else min_degree
    min_active_windows = (
        SERVICE_MIN_ACTIVE_WINDOWS if min_active_windows is None else min_active_windows
    )

    degree = g.in_degree(node) + g.out_degree(node)
    if degree < min_degree:
        return False

    stamps = {
        t
        for *_x, d in list(_in_edges(g, node)) + list(_out_edges(g, node))
        for t in d.get("ts_list", [d["ts"]])
    }
    return len(stamps) >= min_active_windows


RULES = (
    detect_fan_in,
    detect_fan_out,
    detect_rapid_passthrough,
    detect_peel_chain,
    detect_structuring,
    detect_dormant_burst,
    detect_illicit_proximity,
)


def detect_all(g: nx.DiGraph, node: Any) -> list[Typology]:
    """Run every rule against one entity.

    Ordered structural-before-corroborating, then by severity. The ordering is not
    cosmetic: the first entry is what the UI shows as the primary indicator and what
    ``narrate`` builds its opening sentence from, so a weak contextual signal must never
    reach that slot ahead of evidence about the entity's own behaviour.
    """
    order = {"high": 0, "medium": 1, "low": 2}
    hits = [t for rule in RULES if (t := rule(g, node)) is not None]
    return sorted(hits, key=lambda t: (t.corroborating, order.get(t.severity, 3)))


def structural(typologies: list[Typology]) -> list[Typology]:
    """Only the indicators describing the entity's own conduct."""
    return [t for t in typologies if not t.corroborating]


def narrate(node: Any, risk: float, typologies: list[Typology]) -> str:
    """Deterministic natural-language summary for the case pack.

    Template-based on purpose: no model call, no GPU, identical output every run. An LLM
    pass over this text is an optional polish step, never a dependency.

    Three cases, because the strength of the claim differs sharply between them:

    * **structural evidence** -- the entity's own conduct matches a laundering pattern.
      This leads.
    * **corroborating only** -- we know something about its neighbours, not about it. Said
      explicitly, so nobody reads a context signal as a finding of conduct.
    * **nothing matched** -- the score rests on statistics alone; the weakest lead.

    Every branch ends with the same caveat. A risk score must never travel without its
    qualification, so this is enforced by a test rather than left to care.
    """
    CAVEAT = (
        "Risk scores reflect heuristic dataset labels and structural pattern matching; "
        "they indicate investigative priority, not proof of criminal conduct."
    )

    structural_hits = [t for t in typologies if not t.corroborating]
    corroborating_hits = [t for t in typologies if t.corroborating]
    opening = f"Entity {node} carries a model risk score of {risk:.2f}."

    if structural_hits:
        lead = structural_hits[0]
        parts = [
            f"Entity {node} is flagged with a model risk score of {risk:.2f}.",
            f"Primary indicator -- {lead.name} ({lead.code}): {lead.description}",
        ]
        others = structural_hits[1:]
        if others:
            parts.append(
                "Additional conduct indicators: "
                + "; ".join(f"{t.name} ({t.code})" for t in others)
                + "."
            )
        if corroborating_hits:
            parts.append(
                "Corroborating context: "
                + "; ".join(f"{t.name} ({t.code})" for t in corroborating_hits)
                + "."
            )
        parts.append(CAVEAT)
        return " ".join(parts)

    if corroborating_hits:
        return " ".join(
            [
                opening,
                "No laundering typology matched this entity's own conduct.",
                "Only corroborating context applies: "
                + "; ".join(f"{t.name} ({t.code})" for t in corroborating_hits)
                + ".",
                "Association with flagged counterparties is not itself evidence of "
                "wrongdoing, so this is a weak lead requiring human review.",
                CAVEAT,
            ]
        )

    return " ".join(
        [
            opening,
            "No structural laundering typology matched; the score rests on statistical "
            "features alone and should be treated as a low-confidence lead.",
            CAVEAT,
        ]
    )
