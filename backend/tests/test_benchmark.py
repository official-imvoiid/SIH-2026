"""Tests for the graph-learning benchmark and the association-feature exclusion.

The exclusion is the load-bearing part. It was not a design preference -- it is the result
of a measurement (dropping guilt-by-association features raised illicit F1 from 0.755 to
0.838 under the same temporal split). If a later refactor quietly puts those features back
into the model, the project loses both the accuracy and the ethical argument, and nothing
would visibly break. Hence these tests.
"""

import numpy as np
import pandas as pd
import pytest

from ml.train.baseline import (
    ASSOCIATION_FEATURES,
    EXCLUDE_ASSOCIATION_FEATURES,
    NON_FEATURES,
)
from ml.train.benchmark import (
    EMBED_DIM,
    GRAPH_ASSOCIATION,
    GRAPH_DERIVED,
    GRAPH_STRUCTURAL,
    build_embeddings,
)


# --------------------------------------------------------------------------------------
# The exclusion
# --------------------------------------------------------------------------------------


def test_association_features_are_excluded_from_the_model():
    """Guarded because re-including them silently costs ~8 F1 points."""
    assert EXCLUDE_ASSOCIATION_FEATURES is True


def test_association_feature_set_matches_the_benchmark_definition():
    """Two modules define this set; they must not drift apart."""
    assert ASSOCIATION_FEATURES == GRAPH_ASSOCIATION


def test_association_features_are_actually_dropped_by_the_loader(tmp_path):
    from ml.train.baseline import load_features

    df = pd.DataFrame(
        {
            "entity_id": ["E-1", "E-2"],
            "label": ["illicit", "licit"],
            "first_ts": [1.0, 2.0],
            "last_ts": [3.0, 4.0],
            "total_in_btc": [1.0, 2.0],
            "pagerank": [0.1, 0.2],
            **{f: [0.0, 1.0] for f in ASSOCIATION_FEATURES},
        }
    )
    path = tmp_path / "entities.parquet"
    df.to_parquet(path, index=False)

    _df, feature_cols = load_features(path)
    assert set(feature_cols).isdisjoint(ASSOCIATION_FEATURES)
    # Label-free structural features must survive -- only association is dropped.
    assert "pagerank" in feature_cols
    assert "total_in_btc" in feature_cols


def test_structural_and_association_families_are_disjoint():
    assert GRAPH_STRUCTURAL & GRAPH_ASSOCIATION == set()
    assert GRAPH_DERIVED == GRAPH_STRUCTURAL | GRAPH_ASSOCIATION


def test_association_features_are_never_confused_with_identifiers():
    assert ASSOCIATION_FEATURES & NON_FEATURES == set()


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------


def _edges(pairs, value=1.0):
    return pd.DataFrame(
        {
            "src_entity": [a for a, _b in pairs],
            "dst_entity": [b for _a, b in pairs],
            "value_btc": [value] * len(pairs),
            "n_txs": [1] * len(pairs),
            "ts": [1] * len(pairs),
        }
    )


def test_embeddings_have_one_row_per_entity_and_fixed_width():
    ids = [f"E-{i}" for i in range(60)]
    edges = _edges([(f"E-{i}", f"E-{i+1}") for i in range(59)])
    emb, explained = build_embeddings(ids, edges)

    assert emb.shape == (60, EMBED_DIM)
    assert 0.0 <= explained <= 1.0
    assert np.isfinite(emb).all()


def test_embeddings_are_deterministic():
    ids = [f"E-{i}" for i in range(40)]
    edges = _edges([(f"E-{i}", f"E-{(i * 7) % 40}") for i in range(40)])
    a, _ = build_embeddings(ids, edges)
    b, _ = build_embeddings(ids, edges)
    assert np.allclose(a, b)


def test_embeddings_pad_when_graph_is_tiny():
    """A graph smaller than the requested dimensionality must not crash the benchmark."""
    ids = ["E-1", "E-2", "E-3"]
    emb, _ = build_embeddings(ids, _edges([("E-1", "E-2")]))
    assert emb.shape == (3, EMBED_DIM)


def test_isolated_entities_get_zero_embeddings_not_errors():
    ids = ["E-1", "E-2", "E-lonely"]
    emb, _ = build_embeddings(ids, _edges([("E-1", "E-2")]))
    assert np.isfinite(emb).all()
    assert np.allclose(emb[2], 0.0)


def test_edges_referencing_unknown_entities_are_skipped():
    """Real edge lists reference nodes outside the entity table; that must not crash."""
    ids = ["E-1", "E-2"]
    edges = _edges([("E-1", "E-2"), ("E-1", "E-ghost"), ("E-nobody", "E-2")])
    emb, _ = build_embeddings(ids, edges)
    assert emb.shape == (2, EMBED_DIM)


def test_log_weighting_compresses_orders_of_magnitude():
    """log1p exists so one enormous transfer cannot swamp the whole decomposition.

    Comparing singular *vectors* here would be meaningless: a path graph has degenerate
    singular values, and any rotation within a degenerate eigenspace is an equally valid
    basis. What is well defined is scale. Raw BTC weighting would make a 1e9x larger
    transfer produce a 1e9x larger embedding; log1p should keep the growth to a small
    multiple.
    """
    ids = [f"E-{i}" for i in range(30)]
    pairs = [(f"E-{i}", f"E-{i + 1}") for i in range(29)]

    small = build_embeddings(ids, _edges(pairs, value=1.0))[0]
    huge = build_embeddings(ids, _edges(pairs, value=1e9))[0]

    ratio = np.linalg.norm(huge) / (np.linalg.norm(small) + 1e-12)
    assert ratio < 100, f"embedding scaled {ratio:.0f}x -- log weighting is not applied"
    assert ratio > 1.0, "a larger transfer should still register as larger"


def test_benchmark_note_disclaims_being_a_gnn():
    """The write-up must not let a reader believe a GNN was benchmarked."""
    import ml.train.benchmark as b

    assert "NOT a GNN" in b.__doc__ or "not a GNN" in b.__doc__
