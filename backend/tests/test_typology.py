"""Tests for the typology rule engine.

These run with no dataset and no trained model -- which is the point. The rule engine is
the part of the system that must work on demo day regardless of what else is broken.
"""

import networkx as nx
import pytest

from ml.typology.rules import (
    detect_all,
    detect_dormant_burst,
    detect_fan_in,
    detect_fan_out,
    detect_peel_chain,
    detect_rapid_passthrough,
    detect_structuring,
    narrate,
)


def _edge(g, u, v, ts, value):
    g.add_edge(u, v, ts=ts, value_btc=value)


def test_fan_in_fires_on_many_senders_one_window():
    g = nx.DiGraph()
    for i in range(15):
        _edge(g, f"S{i}", "COLLECTOR", ts=5, value=0.1)

    hit = detect_fan_in(g, "COLLECTOR")
    assert hit is not None
    assert hit.code == "VA-CONSOLIDATION"
    assert hit.evidence["distinct_senders"] == 15
    assert hit.severity == "medium"


def test_fan_in_ignores_senders_spread_over_time():
    g = nx.DiGraph()
    for i in range(15):
        _edge(g, f"S{i}", "MERCHANT", ts=i * 3, value=0.1)

    assert detect_fan_in(g, "MERCHANT") is None


def test_fan_out_fires_on_dispersion():
    g = nx.DiGraph()
    _edge(g, "SRC", "HUB", ts=1, value=10.0)
    for i in range(12):
        _edge(g, "HUB", f"R{i}", ts=2, value=0.8)

    hit = detect_fan_out(g, "HUB")
    assert hit is not None
    assert hit.evidence["distinct_recipients"] == 12


def test_rapid_passthrough_fires_when_nothing_retained():
    g = nx.DiGraph()
    _edge(g, "A", "MULE", ts=10, value=5.0)
    _edge(g, "MULE", "B", ts=10, value=4.95)

    hit = detect_rapid_passthrough(g, "MULE")
    assert hit is not None
    assert hit.severity == "high"
    assert hit.evidence["forward_ratio"] == pytest.approx(0.99)


def test_rapid_passthrough_ignores_wallet_that_retains_balance():
    g = nx.DiGraph()
    _edge(g, "A", "SAVER", ts=10, value=5.0)
    _edge(g, "SAVER", "B", ts=10, value=1.0)

    assert detect_rapid_passthrough(g, "SAVER") is None


def test_peel_chain_detects_repeated_peeling():
    g = nx.DiGraph()
    remaining = 100.0
    for hop in range(6):
        peel = remaining * 0.10
        remaining -= peel
        _edge(g, f"P{hop}", f"PEEL_OUT{hop}", ts=hop, value=peel)
        _edge(g, f"P{hop}", f"P{hop + 1}", ts=hop, value=remaining)

    hit = detect_peel_chain(g, "P0")
    assert hit is not None
    assert hit.code == "VA-PEELCHAIN"
    assert hit.evidence["chain_length"] >= 4


def test_peel_chain_ignores_ordinary_two_output_spend():
    """A normal payment plus change is two outputs but not a repeated chain."""
    g = nx.DiGraph()
    _edge(g, "WALLET", "SHOP", ts=1, value=0.2)
    _edge(g, "WALLET", "CHANGE", ts=1, value=0.8)

    assert detect_peel_chain(g, "WALLET") is None


def test_structuring_fires_on_uniform_amounts():
    g = nx.DiGraph()
    for i in range(10):
        _edge(g, "SMURF", f"D{i}", ts=i, value=0.500 + (i % 2) * 0.001)

    hit = detect_structuring(g, "SMURF")
    assert hit is not None
    assert hit.evidence["n_transfers"] == 10


def test_structuring_ignores_varied_amounts():
    g = nx.DiGraph()
    for i in range(10):
        _edge(g, "NORMAL", f"D{i}", ts=i, value=0.1 * (i + 1))

    assert detect_structuring(g, "NORMAL") is None


def test_dormant_burst_fires_after_long_gap():
    g = nx.DiGraph()
    _edge(g, "X", "SLEEPER", ts=1, value=50.0)
    _edge(g, "Y", "SLEEPER", ts=2, value=1.0)
    for i in range(4):
        _edge(g, "SLEEPER", f"OUT{i}", ts=40 + i, value=10.0)

    hit = detect_dormant_burst(g, "SLEEPER")
    assert hit is not None
    assert hit.evidence["dormancy_ts"] >= 10


def test_detect_all_orders_by_severity():
    g = nx.DiGraph()
    for i in range(15):
        _edge(g, f"S{i}", "MULE", ts=5, value=1.0)
    _edge(g, "MULE", "SINK", ts=5, value=14.9)

    hits = detect_all(g, "MULE")
    severities = [h.severity for h in hits]
    assert "high" in severities
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


def test_narrate_is_deterministic_and_carries_a_caveat():
    g = nx.DiGraph()
    _edge(g, "A", "MULE", ts=1, value=5.0)
    _edge(g, "MULE", "B", ts=1, value=4.99)

    hits = detect_all(g, "MULE")
    first = narrate("MULE", 0.93, hits)
    second = narrate("MULE", 0.93, hits)

    assert first == second
    assert "0.93" in first
    assert "not proof" in first


def test_narrate_handles_no_typologies():
    text = narrate("E-1", 0.55, [])
    assert "low-confidence lead" in text
