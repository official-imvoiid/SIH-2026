"""Tests for threshold recalibration and the service guard.

The most important test here is the leakage one. A calibration that puts the same
ransomware crew on both sides of the split will report a tuned threshold as if it
generalises, when it has only memorised one gang's habits. That failure is invisible in
the output -- the numbers look good -- so it has to be caught structurally.
"""

import networkx as nx
import pytest

from ml.typology import rules as R
from ml.typology.calibrate import (
    _param_combos,
    evaluate_set,
    fire_rate,
    seed_entities,
    split_seeds_by_family,
    youden_j,
)


def _edge(g, u, v, ts, value):
    if g.has_edge(u, v):
        g[u][v]["value_btc"] += value
        g[u][v]["n_txs"] += 1
        g[u][v]["ts_list"].append(ts)
    else:
        g.add_edge(u, v, ts=ts, value_btc=value, n_txs=1, ts_list=[ts])


# --------------------------------------------------------------------------------------
# Split integrity -- the leakage guard
# --------------------------------------------------------------------------------------


def test_split_never_puts_a_family_on_both_sides():
    seeds = [{"address": f"a{i}", "family": f"F{i % 5}"} for i in range(40)]
    calib, hold = split_seeds_by_family(seeds, holdout_frac=0.4)

    calib_fams = {s["family"] for s in calib}
    hold_fams = {s["family"] for s in hold}
    assert calib_fams & hold_fams == set(), "a ransomware family leaked across the split"
    assert calib_fams | hold_fams == {f"F{i}" for i in range(5)}


def test_split_is_deterministic():
    seeds = [{"address": f"a{i}", "family": f"F{i % 6}"} for i in range(30)]
    a = split_seeds_by_family(seeds)
    b = split_seeds_by_family(seeds)
    assert [s["address"] for s in a[0]] == [s["address"] for s in b[0]]
    assert [s["address"] for s in a[1]] == [s["address"] for s in b[1]]


def test_split_always_holds_out_at_least_one_family():
    seeds = [{"address": "a", "family": "OnlyOne"}, {"address": "b", "family": "OnlyOne"}]
    _calib, hold = split_seeds_by_family(seeds, holdout_frac=0.4)
    assert len(hold) >= 1


def test_seed_entities_maps_only_known_addresses():
    mapping = {"addr1": "E-1", "addr2": "E-1", "addr3": "E-2"}
    assert seed_entities(mapping, ["addr1", "addr3", "missing"]) == {"E-1", "E-2"}


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def test_youden_j_is_zero_when_rule_cannot_discriminate():
    assert youden_j(0.72, 0.72) == pytest.approx(0.0)


def test_youden_j_is_negative_when_rule_is_backwards():
    assert youden_j(0.10, 0.40) < 0


def test_param_combos_covers_the_full_grid():
    combos = list(_param_combos({"a": [1, 2], "b": [10, 20, 30]}))
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos
    assert {"a": 2, "b": 30} in combos


def test_param_combos_with_empty_grid_yields_one_empty_dict():
    assert list(_param_combos({})) == [{}]


def test_fire_rate_counts_correctly():
    g = nx.DiGraph()
    _edge(g, "A", "MULE", ts=1, value=10.0)
    _edge(g, "MULE", "B", ts=1, value=9.95)
    _edge(g, "X", "SAVER", ts=1, value=10.0)
    _edge(g, "SAVER", "Y", ts=1, value=1.0)

    fired, total = fire_rate(R.detect_rapid_passthrough, g, ["MULE", "SAVER"], {})
    assert (fired, total) == (1, 2)


def test_evaluate_set_reports_separation():
    pos = nx.DiGraph()
    _edge(pos, "A", "MULE", ts=1, value=10.0)
    _edge(pos, "MULE", "B", ts=1, value=9.99)

    neg = nx.DiGraph()
    _edge(neg, "X", "SAVER", ts=1, value=10.0)
    _edge(neg, "SAVER", "Y", ts=1, value=2.0)

    res = evaluate_set(R.detect_rapid_passthrough, {}, pos, {"MULE"}, neg, {"SAVER"})
    assert res["tpr"] == 1.0
    assert res["fpr"] == 0.0
    assert res["youden_j"] == 1.0


# --------------------------------------------------------------------------------------
# The service guard -- the fix for "exchanges look like launderers"
# --------------------------------------------------------------------------------------


def test_service_guard_flags_a_sustained_high_degree_hub():
    """An exchange: many counterparties, sustained across many time steps."""
    g = nx.DiGraph()
    for i in range(30):
        _edge(g, f"U{i}", "EXCHANGE", ts=i % 12, value=1.0)
    for i in range(30):
        _edge(g, "EXCHANGE", f"W{i}", ts=i % 12, value=0.9)

    assert R.is_service_like(g, "EXCHANGE") is True


def test_service_guard_ignores_a_transient_burst():
    """A laundering hop: high degree, but all inside one short window."""
    g = nx.DiGraph()
    for i in range(30):
        _edge(g, f"U{i}", "MIXER", ts=5, value=1.0)
    for i in range(30):
        _edge(g, "MIXER", f"W{i}", ts=5, value=0.9)

    assert R.is_service_like(g, "MIXER") is False


def test_service_guard_ignores_low_degree_entities():
    g = nx.DiGraph()
    for ts in range(20):
        _edge(g, "A", "SMALL", ts=ts, value=1.0)
    assert R.is_service_like(g, "SMALL") is False


# --------------------------------------------------------------------------------------
# Tunability -- rules must honour overrides, or the whole sweep is a no-op
# --------------------------------------------------------------------------------------


def test_fan_in_threshold_override_actually_changes_behaviour():
    g = nx.DiGraph()
    for i in range(15):
        _edge(g, f"S{i}", "HUB", ts=3, value=1.0)

    assert R.detect_fan_in(g, "HUB") is not None                      # default 10
    assert R.detect_fan_in(g, "HUB", min_counterparties=20) is None    # tightened


def test_passthrough_threshold_override_actually_changes_behaviour():
    g = nx.DiGraph()
    _edge(g, "A", "M", ts=1, value=10.0)
    _edge(g, "M", "B", ts=1, value=9.2)  # 92% forwarded

    assert R.detect_rapid_passthrough(g, "M") is not None                    # default .90
    assert R.detect_rapid_passthrough(g, "M", min_fwd_ratio=0.99) is None    # tightened


def test_structuring_threshold_override_actually_changes_behaviour():
    g = nx.DiGraph()
    for i in range(10):
        _edge(g, "S", f"D{i}", ts=i, value=0.5)

    assert R.detect_structuring(g, "S") is not None
    assert R.detect_structuring(g, "S", min_transfers=25) is None


def test_dormant_burst_threshold_override_actually_changes_behaviour():
    g = nx.DiGraph()
    _edge(g, "X", "S", ts=1, value=5.0)
    _edge(g, "Y", "S", ts=2, value=1.0)
    for i in range(5):
        _edge(g, "S", f"O{i}", ts=40 + i, value=1.0)

    assert R.detect_dormant_burst(g, "S") is not None
    assert R.detect_dormant_burst(g, "S", min_gap_ts=90) is None


def test_overrides_do_not_change_the_module_defaults():
    """A sweep must not leave global state mutated for the next caller."""
    before = (R.FAN_MIN_COUNTERPARTIES, R.PASSTHROUGH_MIN_FWD_RATIO)
    g = nx.DiGraph()
    for i in range(15):
        _edge(g, f"S{i}", "HUB", ts=1, value=1.0)
    R.detect_fan_in(g, "HUB", min_counterparties=99)
    assert (R.FAN_MIN_COUNTERPARTIES, R.PASSTHROUGH_MIN_FWD_RATIO) == before
