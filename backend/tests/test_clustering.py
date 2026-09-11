"""Tests for address clustering -- the project's core IP.

The critical test here is ``test_coinjoin_is_not_merged``. A missed CoinJoin merges
unrelated people into one entity, and because union-find is transitive, a single bad merge
can cascade until a large part of the graph collapses into one meaningless blob. Every
downstream result would then be wrong in a way that looks plausible.
"""

import pytest

from ml.features.clustering import (
    UnionFind,
    cluster_addresses,
    evaluate_clustering,
    is_likely_coinjoin,
)


# ---------------------------------------------------------------------------- union-find


def test_union_find_merges_transitively():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert len(uf.groups()) == 1


def test_union_find_keeps_separate_sets_apart():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("x", "y")
    assert uf.find("a") != uf.find("x")
    assert len(uf.groups()) == 2


def test_union_reports_whether_it_merged():
    uf = UnionFind()
    assert uf.union("a", "b") is True
    assert uf.union("a", "b") is False


# ------------------------------------------------------------------------ coinjoin guard


def test_coinjoin_detected_from_equal_outputs():
    inputs = [f"addr{i}" for i in range(8)]
    outputs = [0.5] * 8 + [0.031, 0.017]  # 8 equal denominations plus change
    assert is_likely_coinjoin(inputs, outputs) is True


def test_ordinary_payment_is_not_a_coinjoin():
    """Two inputs, a payment and change. The overwhelmingly common case."""
    assert is_likely_coinjoin(["a", "b"], [1.5, 0.34]) is False


def test_exchange_batch_payout_is_not_a_coinjoin():
    """Many inputs but scattered output values -- fails the equal-output condition.

    This is the false positive that matters: wrongly screening exchange batches would
    silently stop the heuristic clustering the largest, most important entities.
    """
    inputs = [f"hot{i}" for i in range(40)]
    outputs = [0.13, 2.7, 0.0041, 11.5, 0.66, 3.02, 0.917, 0.4]
    assert is_likely_coinjoin(inputs, outputs) is False


def test_few_inputs_never_counts_as_coinjoin():
    assert is_likely_coinjoin(["a", "b"], [0.5, 0.5, 0.5, 0.5]) is False


def test_empty_outputs_handled():
    assert is_likely_coinjoin([f"a{i}" for i in range(9)], []) is False


# --------------------------------------------------------------------------- clustering


def test_co_spent_addresses_join_one_entity():
    txs = [("t1", ["a", "b"], [1.0, 0.2]), ("t2", ["b", "c"], [0.5, 0.1])]
    m = cluster_addresses(txs)
    assert m["a"] == m["b"] == m["c"]


def test_unrelated_addresses_stay_separate():
    txs = [("t1", ["a", "b"], [1.0]), ("t2", ["x", "y"], [1.0])]
    m = cluster_addresses(txs)
    assert m["a"] == m["b"]
    assert m["x"] == m["y"]
    assert m["a"] != m["x"]


def test_coinjoin_is_not_merged():
    """The guard in action, and the single most important test in the suite.

    Two genuinely separate owners (a/b and x/y) also take part in one CoinJoin. Without
    screening, that transaction would fuse them permanently.
    """
    participants = ["a", "x", "p", "q", "r", "s", "t", "u"]
    txs = [
        ("own1", ["a", "b"], [1.0, 0.3]),
        ("own2", ["x", "y"], [1.0, 0.3]),
        ("mix", participants, [0.5] * 8 + [0.02]),
    ]
    m = cluster_addresses(txs, skip_coinjoins=True)
    assert m["a"] == m["b"]
    assert m["x"] == m["y"]
    assert m["a"] != m["x"], "CoinJoin merged two unrelated owners"


def test_disabling_the_guard_does_cause_the_false_merge():
    """Confirms the guard is what prevents the merge, not luck in the fixture."""
    participants = ["a", "x", "p", "q", "r", "s", "t", "u"]
    txs = [
        ("own1", ["a", "b"], [1.0, 0.3]),
        ("own2", ["x", "y"], [1.0, 0.3]),
        ("mix", participants, [0.5] * 8 + [0.02]),
    ]
    m = cluster_addresses(txs, skip_coinjoins=False)
    assert m["a"] == m["x"], "expected the unguarded heuristic to over-merge"


def test_entity_ids_are_ordered_by_cluster_size():
    txs = [
        ("big", ["a", "b", "c", "d"], [1.0]),
        ("small", ["x", "y"], [1.0]),
    ]
    m = cluster_addresses(txs)
    assert m["a"] == "E-00001"
    assert m["x"] == "E-00002"


def test_clustering_is_deterministic():
    """An investigator must be able to re-derive the same entity from the same evidence."""
    txs = [
        ("t1", ["a", "b"], [1.0]),
        ("t2", ["c", "d"], [1.0]),
        ("t3", ["e", "f", "g"], [1.0]),
    ]
    assert cluster_addresses(txs) == cluster_addresses(txs)


def test_single_input_transactions_produce_singletons():
    m = cluster_addresses([("t1", ["lonely"], [1.0])])
    assert m["lonely"].startswith("E-")
    assert len(set(m.values())) == 1


# ---------------------------------------------------------------------------- evaluation


def test_evaluation_reports_a_false_merge():
    predicted = {"a": "E-1", "b": "E-1"}
    truth = {"a": "OWNER-1", "b": "OWNER-2"}
    res = evaluate_clustering(predicted, truth)
    assert res["false_merges"] == 1
    assert res["pair_precision"] == 0.0


def test_evaluation_reports_a_split_not_a_merge():
    """Under-merging costs recall but keeps precision perfect -- the correct trade."""
    predicted = {"a": "E-1", "b": "E-2"}
    truth = {"a": "OWNER-1", "b": "OWNER-1"}
    res = evaluate_clustering(predicted, truth)
    assert res["false_merges"] == 0
    assert res["split_true_entities"] == 1
    assert res["pair_recall"] == 0.0


def test_perfect_clustering_scores_one():
    predicted = {"a": "E-1", "b": "E-1", "x": "E-2"}
    truth = {"a": "O-1", "b": "O-1", "x": "O-2"}
    res = evaluate_clustering(predicted, truth)
    assert res["pair_precision"] == 1.0
    assert res["pair_recall"] == 1.0
    assert res["false_merges"] == 0


# ------------------------------------------------------- end-to-end on the synthetic world


@pytest.mark.slow
def test_synthetic_world_clusters_without_false_merges():
    """The property that must hold at scale: high precision, zero invented links.

    Recall is deliberately not asserted to be high. Co-spend clustering under-merges by
    construction, and pretending otherwise would be the dishonest version of this test.
    """
    from ml.data.synth import generate, iter_transactions

    world = generate()
    predicted = cluster_addresses(iter_transactions(world), skip_coinjoins=True)
    res = evaluate_clustering(predicted, world.owner)

    assert res["false_merges"] == 0, "CoinJoin guard failed at scale"
    assert res["pair_precision"] >= 0.99
    assert 0.15 < res["pair_recall"] < 1.0, "under-merging is expected, total failure is not"
