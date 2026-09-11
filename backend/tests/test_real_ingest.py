"""Tests for the real-blockchain ingestion path.

These run offline. The transaction fixtures below are the actual response shape returned
by mempool.space, so the parser is tested against reality rather than against an
invented format -- which is the usual way an ingestion bug survives to demo day.
"""

import json

import pytest

from ml.ingest.real import (
    SECONDS_PER_TIMESTEP,
    _cache_path,
    _tx_to_record,
    select_seeds,
)

# Trimmed from a genuine mempool.space /address/{a}/txs response.
REAL_TX = {
    "txid": "be2da68431e55ccaff5cbf2500000000000000000000000000000000deadbeef",
    "vin": [
        {"prevout": {"scriptpubkey_address": "17TMc2UkVRSga2yYvuxSD9Q1XyB2EPRjTF",
                     "value": 43577930}},
        {"prevout": {"scriptpubkey_address": "1DqU3NwULyDYKSSYeo42WSTQXRm2dPRQJ1",
                     "value": 12000000}},
    ],
    "vout": [
        {"scriptpubkey_address": "14oFeithByDgKAqwmtoQ6DHp44nKK3WBZM", "value": 15377036},
        {"scriptpubkey_address": "1K8Ts7mu9vydYfoqh3nvsKjGNgqyCcyS7g", "value": 40000000},
    ],
    "status": {"confirmed": True, "block_height": 638442, "block_time": 1594292540},
}


def test_parses_a_real_transaction():
    rec = _tx_to_record(REAL_TX)
    assert rec is not None
    txid, inputs, out_values, ts, outputs = rec

    assert txid == REAL_TX["txid"]
    assert inputs == [
        "17TMc2UkVRSga2yYvuxSD9Q1XyB2EPRjTF",
        "1DqU3NwULyDYKSSYeo42WSTQXRm2dPRQJ1",
    ]
    assert outputs[0] == ("14oFeithByDgKAqwmtoQ6DHp44nKK3WBZM", 0.15377036)
    assert out_values == [0.15377036, 0.4]


def test_satoshi_values_are_converted_to_btc():
    """A factor-of-1e8 error here would silently corrupt every threshold downstream."""
    _txid, _ins, values, _ts, _outs = _tx_to_record(REAL_TX)
    assert all(v < 1.0 for v in values), "values look like satoshis, not BTC"


def test_two_inputs_give_the_clustering_step_a_co_spend_link():
    """The whole method depends on multi-input transactions surviving the parser."""
    _txid, inputs, _v, _ts, _o = _tx_to_record(REAL_TX)
    assert len(inputs) >= 2


def test_unconfirmed_transactions_are_dropped():
    tx = {**REAL_TX, "status": {"confirmed": False}}
    assert _tx_to_record(tx) is None


def test_coinbase_transaction_is_dropped():
    """Newly mined coins have no input address and would break clustering."""
    tx = {**REAL_TX, "vin": [{"is_coinbase": True, "prevout": None}]}
    assert _tx_to_record(tx) is None


def test_outputs_without_an_address_are_skipped():
    """OP_RETURN and exotic scripts carry no address; they must not become nodes."""
    tx = {
        **REAL_TX,
        "vout": [
            {"scriptpubkey_address": None, "value": 0},
            {"scriptpubkey_address": "14oFeithByDgKAqwmtoQ6DHp44nKK3WBZM", "value": 100},
        ],
    }
    _txid, _ins, _v, _ts, outputs = _tx_to_record(tx)
    assert len(outputs) == 1


def test_block_time_maps_to_a_two_week_timestep():
    """Matching Elliptic's convention keeps TRAIN_TS_MAX / DRIFT_TS meaningful."""
    _txid, _ins, _v, ts, _o = _tx_to_record(REAL_TX)
    assert ts == REAL_TX["status"]["block_time"] // SECONDS_PER_TIMESTEP

    later = {**REAL_TX, "status": {**REAL_TX["status"],
                                   "block_time": REAL_TX["status"]["block_time"]
                                   + SECONDS_PER_TIMESTEP}}
    _t2, _i2, _v2, ts2, _o2 = _tx_to_record(later)
    assert ts2 == ts + 1


def test_cache_paths_are_sharded():
    """One flat directory with tens of thousands of files is slow on Windows."""
    p = _cache_path("bc1qjndxpuddlssczk336tdd7wcawlj2qws7p77p2k")
    assert p.parent.name == "bc1q"
    assert p.name.endswith(".json")


def test_seed_selection_spreads_across_ransomware_families():
    """Locky is ~85% of Ransomwhere; an unweighted sample would be one crew's habits."""
    fake = (
        [{"address": f"L{i}", "blockchain": "bitcoin", "balance": 100, "family": "Locky"}
         for i in range(500)]
        + [{"address": f"C{i}", "blockchain": "bitcoin", "balance": 100, "family": "Conti"}
           for i in range(20)]
        + [{"address": f"R{i}", "blockchain": "bitcoin", "balance": 100, "family": "Ryuk"}
           for i in range(10)]
    )
    seeds = select_seeds(fake, n=12)
    families = {s["family"] for s in seeds}
    assert len(families) >= 3, f"sample collapsed onto {families}"
    assert len(seeds) == 12


def test_seed_selection_skips_addresses_that_never_received_anything():
    fake = [
        {"address": "A", "blockchain": "bitcoin", "balance": 0, "family": "Conti"},
        {"address": "B", "blockchain": "bitcoin", "balance": 500, "family": "Conti"},
    ]
    seeds = select_seeds(fake, n=5)
    assert [s["address"] for s in seeds] == ["B"]


def test_seed_selection_skips_unlabelled():
    fake = [
        {"address": "A", "blockchain": "bitcoin", "balance": 5, "family": "Unlabeled"},
        {"address": "B", "blockchain": "bitcoin", "balance": 5, "family": None},
        {"address": "C", "blockchain": "bitcoin", "balance": 5, "family": "Ryuk"},
    ]
    assert [s["address"] for s in select_seeds(fake, n=5)] == ["C"]


def test_seed_selection_is_deterministic():
    fake = [{"address": f"A{i}", "blockchain": "bitcoin", "balance": 10,
             "family": f"F{i % 6}"} for i in range(120)]
    assert select_seeds(fake, n=15) == select_seeds(fake, n=15)


# ------------------------------------------------------------------ real cached data


@pytest.mark.slow
def test_cached_real_experiment_is_internally_consistent():
    """If the real experiment has been run, sanity-check what it wrote."""
    from ml.config import DATA_PROCESSED

    path = DATA_PROCESSED / "real_experiment.json"
    if not path.exists():
        pytest.skip("run: python -m ml.experiments.real_vs_control")

    r = json.loads(path.read_text(encoding="utf-8"))
    for group in ("ransomware", "control"):
        g = r[group]
        assert g["n_entities"] > 0, f"{group} produced no entities"
        assert 0 <= g["pct_with_1_structural"] <= 100
        assert g["n_with_2_structural"] <= g["n_with_1_structural"], (
            "an entity with 2 structural typologies must also count as having 1"
        )
    assert len(r["ransomware"]["families"]) >= 3, "seeds not spread across families"
