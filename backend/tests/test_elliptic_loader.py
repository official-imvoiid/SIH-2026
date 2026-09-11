"""Tests for the real Elliptic / Elliptic++ loaders.

Neither dataset can be downloaded automatically, so these tests build CSVs in the *exact*
published format and run the loaders over them. That means the drop-in is verified before
the real files ever arrive -- which matters, because the alternative is discovering a
header bug at 2am with the real 400 MB download half-finished.

The format details encoded here are the ones that actually trip people up:

  * the features file has **no header row** (pandas will eat transaction #1 as column names)
  * the class coding is **1 = illicit, 2 = licit**, which is the reverse of most guesses
  * `unknown` is a literal string in the class column, not a blank or NaN
"""

import csv
import gzip
import json

import pandas as pd
import pytest

from ml.ingest.elliptic import (
    ELLIPTIC_CLASS_MAP,
    N_ELLIPTIC_FEATURE_COLS,
    SchemaError,
    describe_capabilities,
    find_dataset,
    load_elliptic,
    load_elliptic_plus,
)


# --------------------------------------------------------------------------------------
# Fixture builders -- these write the real published format
# --------------------------------------------------------------------------------------


def write_elliptic(root, n_tx=40, n_timesteps=49, gzipped=False, n_cols=None):
    d = root / "elliptic"
    d.mkdir(parents=True, exist_ok=True)
    n_cols = n_cols or N_ELLIPTIC_FEATURE_COLS

    suffix = ".gz" if gzipped else ""
    opener = (lambda p: gzip.open(p, "wt", newline="", encoding="utf-8")) if gzipped else (
        lambda p: open(p, "w", newline="", encoding="utf-8")
    )

    # features: NO header row. col0 = txId, col1 = time step, rest = anonymised features.
    with opener(d / f"elliptic_txs_features.csv{suffix}") as fh:
        w = csv.writer(fh)
        for i in range(n_tx):
            txid = 230000000 + i
            ts = (i % n_timesteps) + 1
            w.writerow([txid, ts] + [round(0.01 * (i + j), 4) for j in range(n_cols - 2)])

    # classes: header, and 1 = illicit / 2 = licit / "unknown"
    with opener(d / f"elliptic_txs_classes.csv{suffix}") as fh:
        w = csv.writer(fh)
        w.writerow(["txId", "class"])
        for i in range(n_tx):
            cls = "1" if i % 10 == 0 else ("2" if i % 3 == 0 else "unknown")
            w.writerow([230000000 + i, cls])

    with opener(d / f"elliptic_txs_edgelist.csv{suffix}") as fh:
        w = csv.writer(fh)
        w.writerow(["txId1", "txId2"])
        for i in range(n_tx - 1):
            w.writerow([230000000 + i, 230000000 + i + 1])
    return d


def write_elliptic_plus(root, n_addr=30):
    d = root / "elliptic_plus"
    d.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_addr):
        rows.append(
            {
                "address": f"1Addr{i:05d}",
                "Time step": (i % 49) + 1,
                "class": "1" if i % 8 == 0 else ("2" if i % 3 == 0 else "unknown"),
                "total_txs": i + 1,
                "btc_transacted_total": round(0.5 * i, 4),
                "num_addr_transacted_multiple": i % 5,
            }
        )
    pd.DataFrame(rows).to_csv(d / "wallets_features_classes_combined.csv", index=False)

    pd.DataFrame(
        {"input_address": [f"1Addr{i:05d}" for i in range(n_addr - 1)],
         "txId": [900000 + i for i in range(n_addr - 1)]}
    ).to_csv(d / "AddrTx_edgelist.csv", index=False)

    pd.DataFrame(
        {"input_address": [f"1Addr{i:05d}" for i in range(n_addr - 1)],
         "output_address": [f"1Addr{i+1:05d}" for i in range(n_addr - 1)]}
    ).to_csv(d / "AddrAddr_edgelist.csv", index=False)
    return d


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def test_find_dataset_returns_none_when_nothing_present(tmp_path):
    assert find_dataset(tmp_path) is None


def test_find_dataset_locates_classic_elliptic(tmp_path):
    write_elliptic(tmp_path)
    paths = find_dataset(tmp_path)
    assert paths is not None and paths.kind == "elliptic"


def test_elliptic_plus_is_preferred_when_both_present(tmp_path):
    """Elliptic++ is the only one that supports entity resolution, so it must win."""
    write_elliptic(tmp_path)
    write_elliptic_plus(tmp_path)
    assert find_dataset(tmp_path).kind == "elliptic_plus"


def test_find_dataset_handles_gzipped_files(tmp_path):
    write_elliptic(tmp_path, gzipped=True)
    assert find_dataset(tmp_path).kind == "elliptic"


# --------------------------------------------------------------------------------------
# Classic Elliptic
# --------------------------------------------------------------------------------------


def test_load_elliptic_produces_the_frozen_contract(tmp_path):
    write_elliptic(tmp_path, n_tx=40)
    entities, edges, meta = load_elliptic(find_dataset(tmp_path))

    for col in ("entity_id", "label", "first_ts", "last_ts", "n_addresses"):
        assert col in entities.columns
    for col in ("src_entity", "dst_entity", "value_btc", "n_txs", "ts"):
        assert col in edges.columns
    assert len(entities) == 40
    assert len(edges) == 39


def test_class_coding_is_1_illicit_2_licit(tmp_path):
    """The single most likely thing to get backwards, so it gets its own test."""
    assert ELLIPTIC_CLASS_MAP["1"] == "illicit"
    assert ELLIPTIC_CLASS_MAP["2"] == "licit"

    write_elliptic(tmp_path, n_tx=30)
    entities, _edges, _meta = load_elliptic(find_dataset(tmp_path))
    # index 0, 10, 20 -> class "1" -> illicit
    illicit = set(entities.loc[entities["label"] == "illicit", "entity_id"])
    assert "230000000" in illicit
    assert "230000010" in illicit


def test_unknown_labels_are_preserved_not_dropped(tmp_path):
    write_elliptic(tmp_path, n_tx=40)
    entities, _e, _m = load_elliptic(find_dataset(tmp_path))
    assert (entities["label"] == "unknown").sum() > 0
    assert set(entities["label"]) <= {"illicit", "licit", "unknown"}


def test_features_file_read_without_header(tmp_path):
    """If the loader assumed a header, transaction #1 would vanish into column names."""
    write_elliptic(tmp_path, n_tx=25)
    entities, _e, _m = load_elliptic(find_dataset(tmp_path))
    assert len(entities) == 25
    assert "230000000" in set(entities["entity_id"])


def test_feature_columns_are_named_local_and_agg(tmp_path):
    write_elliptic(tmp_path, n_tx=10)
    entities, _e, _m = load_elliptic(find_dataset(tmp_path))
    assert sum(c.startswith("local_") for c in entities.columns) == 93
    assert sum(c.startswith("agg_") for c in entities.columns) == 72


def test_gzipped_elliptic_loads_identically(tmp_path):
    write_elliptic(tmp_path, n_tx=20, gzipped=True)
    entities, edges, _m = load_elliptic(find_dataset(tmp_path))
    assert len(entities) == 20 and len(edges) == 19


def test_wrong_column_count_raises_a_useful_error(tmp_path):
    write_elliptic(tmp_path, n_tx=10, n_cols=50)
    with pytest.raises(SchemaError, match="expected 167 columns"):
        load_elliptic(find_dataset(tmp_path))


def test_wrong_classes_header_raises(tmp_path):
    d = write_elliptic(tmp_path, n_tx=10)
    (d / "elliptic_txs_classes.csv").write_text("id,verdict\n1,2\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="elliptic_txs_classes"):
        load_elliptic(find_dataset(tmp_path))


def test_wrong_edgelist_header_raises(tmp_path):
    d = write_elliptic(tmp_path, n_tx=10)
    (d / "elliptic_txs_edgelist.csv").write_text("from,to\n1,2\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="elliptic_txs_edgelist"):
        load_elliptic(find_dataset(tmp_path))


def test_strict_mode_flags_unexpected_row_count(tmp_path):
    write_elliptic(tmp_path, n_tx=10)
    with pytest.raises(SchemaError, match="203,769"):
        load_elliptic(find_dataset(tmp_path), strict=True)


# --------------------------------------------------------------------------------------
# Honesty about what each dataset supports
# --------------------------------------------------------------------------------------


def test_classic_elliptic_does_not_claim_entity_resolution(tmp_path):
    """The project's central claim must not be attachable to a transaction-level build."""
    caps = describe_capabilities("elliptic")
    assert caps["entity_resolution"] is False
    assert caps["unit"] == "transaction"
    assert any("not wallets" in c for c in caps["caveats"])


def test_elliptic_plus_does_claim_entity_resolution():
    caps = describe_capabilities("elliptic_plus")
    assert caps["entity_resolution"] is True
    assert caps["unit"] == "wallet address"


def test_capabilities_travel_in_the_metadata(tmp_path):
    write_elliptic(tmp_path, n_tx=15)
    _e, _ed, meta = load_elliptic(find_dataset(tmp_path))
    assert meta["capabilities"]["entity_resolution"] is False
    assert meta["provenance"] == "elliptic-real"


def test_elliptic_typology_degradation_is_declared(tmp_path):
    """Elliptic has no transfer amounts, so value-based rules cannot work. Say so."""
    caps = describe_capabilities("elliptic")
    assert caps["typology_rules"] == "degraded"
    assert any("no transfer amounts" in c for c in caps["caveats"])


# --------------------------------------------------------------------------------------
# Elliptic++
# --------------------------------------------------------------------------------------


def test_load_elliptic_plus_produces_the_contract(tmp_path):
    write_elliptic_plus(tmp_path, n_addr=30)
    entities, edges, meta = load_elliptic_plus(find_dataset(tmp_path))
    for col in ("entity_id", "label", "first_ts", "last_ts", "n_addresses"):
        assert col in entities.columns
    assert len(entities) == 30
    assert meta["provenance"] == "elliptic-plus-real"


def test_elliptic_plus_resolves_column_names_flexibly(tmp_path):
    """Column naming varies between releases; the loader must not hardcode positions."""
    write_elliptic_plus(tmp_path, n_addr=20)
    _e, _ed, meta = load_elliptic_plus(find_dataset(tmp_path))
    assert meta["resolved_columns"]["address"] == "address"
    assert meta["resolved_columns"]["time_step"] == "Time step"


def test_elliptic_plus_illicit_label_wins_across_timesteps(tmp_path):
    """A wallet illicit at any point is illicit overall -- how an investigator reads it."""
    d = tmp_path / "elliptic_plus"
    d.mkdir(parents=True)
    pd.DataFrame(
        [
            {"address": "1Same", "Time step": 1, "class": "2", "total_txs": 1},
            {"address": "1Same", "Time step": 9, "class": "1", "total_txs": 4},
        ]
    ).to_csv(d / "wallets_features_classes_combined.csv", index=False)
    pd.DataFrame({"input_address": ["1Same"], "txId": [1]}).to_csv(
        d / "AddrTx_edgelist.csv", index=False
    )

    entities, _ed, _m = load_elliptic_plus(find_dataset(tmp_path))
    assert entities.loc[entities["entity_id"] == "1Same", "label"].iloc[0] == "illicit"


def test_elliptic_plus_missing_address_column_raises_clearly(tmp_path):
    d = tmp_path / "elliptic_plus"
    d.mkdir(parents=True)
    pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(
        d / "wallets_features_classes_combined.csv", index=False
    )
    pd.DataFrame({"input_address": ["x"], "txId": [1]}).to_csv(
        d / "AddrTx_edgelist.csv", index=False
    )
    with pytest.raises(SchemaError, match="could not find a column"):
        load_elliptic_plus(find_dataset(tmp_path))


# --------------------------------------------------------------------------------------
# End to end: a real build must be trainable
# --------------------------------------------------------------------------------------


def test_build_from_real_writes_a_trainable_store(tmp_path, monkeypatch):
    """The proof that matters: real CSVs in, parquet the trainer can consume out."""
    from ml.ingest.build import build_from_real

    write_elliptic(tmp_path / "raw", n_tx=120)
    out = tmp_path / "processed"
    report = build_from_real(tmp_path / "raw", out)

    assert (out / "entities.parquet").exists()
    assert (out / "edges.parquet").exists()

    entities = pd.read_parquet(out / "entities.parquet")
    assert len(entities) == 120
    assert {"entity_id", "label", "first_ts", "last_ts"} <= set(entities.columns)
    # Graph features must have been computed and merged in.
    assert "pagerank" in entities.columns
    assert "hops_to_illicit" in entities.columns

    assert report["capabilities"]["entity_resolution"] is False
    assert report["clustering_quality"]["skipped"] is True
    assert json.loads((out / "build_report.json").read_text(encoding="utf-8"))


def test_build_from_real_exits_with_help_when_no_data(tmp_path):
    from ml.ingest.build import build_from_real

    with pytest.raises(SystemExit) as e:
        build_from_real(tmp_path, tmp_path / "out")
    assert "Kaggle" in str(e.value) or "git-disl" in str(e.value)
