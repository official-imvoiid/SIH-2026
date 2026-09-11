"""API contract tests.

These assert the *shapes* the frontend binds to (docs/TEAM.md, "frozen contracts"). If a
test here fails, someone changed a contract and owes the team an announcement.

Two of them encode ethical requirements rather than technical ones -- that a truncated
graph always says so, and that a risk score always ships with its caveat. Those are as
much a part of the product as the model.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ready(client):
    if client.get("/api/health").status_code != 200:
        pytest.skip("No data built. Run: python -m ml.ingest.build")
    return True


@pytest.fixture(scope="module")
def some_entity(client, ready):
    return client.get("/api/top?limit=1").json()["results"][0]["id"]


def test_health_reports_what_is_loaded(client, ready):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    for key in ("n_entities", "n_edges", "risk_bands", "model_version", "provenance"):
        assert key in h
    assert h["n_entities"] > 0


def test_health_degrades_rather_than_crashing(client):
    """503 with a fix-it hint beats a stack trace when data is missing."""
    r = client.get("/api/health")
    assert r.status_code in (200, 503)
    if r.status_code == 503:
        assert "hint" in r.json()


def test_entity_returns_the_full_contract(client, some_entity):
    e = client.get(f"/api/entity/{some_entity}").json()
    for key in (
        "id", "risk", "band", "label", "n_addresses", "total_in_btc", "total_out_btc",
        "features", "attributions", "typologies", "neighbors", "narrative",
        "model_version", "clustering_note",
    ):
        assert key in e, f"missing contract field: {key}"
    assert 0.0 <= e["risk"] <= 1.0
    assert e["band"] in ("high", "medium", "low")
    assert e["label"] in ("illicit", "licit", "unknown")


def test_unknown_entity_is_404(client, ready):
    assert client.get("/api/entity/E-DOES-NOT-EXIST").status_code == 404


def test_attributions_are_signed_and_readable(client, some_entity):
    """Negative contributions must survive to the UI -- exculpatory evidence counts."""
    e = client.get(f"/api/entity/{some_entity}").json()
    for a in e["attributions"]:
        assert {"feature", "readable", "value", "contribution"} <= a.keys()
        assert isinstance(a["contribution"], (int, float))
        assert a["readable"], "every feature needs a plain-English name"


def test_narrative_carries_the_not_proof_caveat(client, some_entity):
    """A risk score must never travel without its qualification."""
    e = client.get(f"/api/entity/{some_entity}").json()
    assert e["narrative"]
    assert "not proof" in e["narrative"].lower()


def test_typologies_declare_whether_they_are_corroborating(client, some_entity):
    e = client.get(f"/api/entity/{some_entity}").json()
    for t in e["typologies"]:
        assert {"code", "name", "severity", "description", "corroborating"} <= t.keys()
        assert t["severity"] in ("high", "medium", "low")


def test_corroborating_typologies_never_lead(client, ready):
    """Guilt by association must not be the headline indicator on any entity."""
    for row in client.get("/api/top?limit=40").json()["results"]:
        e = client.get(f"/api/entity/{row['id']}").json()
        t = e["typologies"]
        if len(t) > 1:
            structural_after_corroborating = any(
                not t[i]["corroborating"] and t[i - 1]["corroborating"]
                for i in range(1, len(t))
            )
            assert not structural_after_corroborating, f"{row['id']} ordered wrongly"


def test_subgraph_is_capped_and_declares_truncation(client, some_entity):
    sg = client.get(f"/api/subgraph/{some_entity}?hops=2&max_nodes=50").json()
    assert len(sg["nodes"]) <= 50, "server-side cap not enforced"
    assert isinstance(sg["truncated"], bool)
    assert sg["total_available"] >= len(sg["nodes"])
    if sg["truncated"]:
        assert sg["ranking"], "a truncated view must say how it was ranked"


def test_subgraph_always_contains_its_seed(client, some_entity):
    sg = client.get(f"/api/subgraph/{some_entity}").json()
    seeds = [n for n in sg["nodes"] if n["is_seed"]]
    assert len(seeds) == 1
    assert seeds[0]["id"] == some_entity


def test_subgraph_edges_reference_returned_nodes_only(client, some_entity):
    """A dangling edge would crash any renderer."""
    sg = client.get(f"/api/subgraph/{some_entity}").json()
    ids = {n["id"] for n in sg["nodes"]}
    for e in sg["edges"]:
        assert e["source"] in ids and e["target"] in ids


def test_hops_parameter_is_bounded(client, some_entity):
    assert client.get(f"/api/subgraph/{some_entity}?hops=99").status_code == 422


def test_search_finds_an_entity_by_id(client, some_entity):
    r = client.get(f"/api/search?q={some_entity}").json()
    assert any(x["id"] == some_entity for x in r["results"])


def test_search_requires_a_query(client, ready):
    assert client.get("/api/search?q=").status_code == 422


def test_top_is_sorted_by_descending_risk(client, ready):
    rows = client.get("/api/top?limit=30").json()["results"]
    risks = [r["risk"] for r in rows]
    assert risks == sorted(risks, reverse=True)
    for r in rows:
        assert "n_structural" in r, "UI filters on this"


def test_top_respects_min_risk(client, ready):
    rows = client.get("/api/top?limit=50&min_risk=0.8").json()["results"]
    assert all(r["risk"] >= 0.8 for r in rows)


def test_timeline_is_ordered_and_complete(client, ready):
    buckets = client.get("/api/timeline").json()["buckets"]
    assert buckets
    ts = [b["ts"] for b in buckets]
    assert ts == sorted(ts)
    for b in buckets:
        assert {"ts", "n_txs", "volume_btc", "n_illicit"} <= b.keys()


def test_metrics_expose_the_drift_breakdown(client, ready):
    r = client.get("/api/metrics")
    if r.status_code == 404:
        pytest.skip("model not trained")
    m = r.json()
    assert "drift" in m
    assert "before_drift" in m["drift"] and "after_drift" in m["drift"]
    # Accuracy must never appear -- at a ~2% base rate it is actively misleading.
    assert "accuracy" not in m.get("test", {})
    assert "illicit_f1" in m["test"]
