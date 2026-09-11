"""Loaders for the real Elliptic and Elliptic++ datasets.

Neither dataset can be fetched automatically -- Elliptic sits behind a Kaggle login and
Elliptic++ behind a Google Drive folder -- so this module cannot download anything. What it
does is make the drop-in painless: put the CSVs in ``data/raw/`` and everything downstream
(clustering, features, model, API, UI) works unchanged.

The important structural difference, which decides what the project can claim
------------------------------------------------------------------------------
**Elliptic (classic) has no addresses.** Its nodes are *transactions*, identified by opaque
IDs, with 166 anonymised features. There is nothing to cluster: no two rows can be shown to
share an owner, because ownership is not represented at all. Loading it gives you the
literature benchmark (Weber et al. 2019, RF ~0.79 illicit F1) but **not** entity resolution.

**Elliptic++ has addresses.** Its actors dataset carries wallet addresses plus AddrTx /
TxAddr mappings, so co-spend clustering becomes possible and "which wallets belong to one
actor" becomes answerable. This is the dataset the project's central claim depends on.

So the two loaders return different things on purpose, and
:func:`describe_capabilities` states plainly what each one supports. A build from classic
Elliptic must not be described as doing entity resolution.

A second limitation worth stating out loud: **Elliptic's edge list carries no transfer
amounts and no per-edge timestamps.** The typology rules reason about value and timing, so
on classic Elliptic they run in a degraded mode and several cannot fire at all. The loader
records this in the build report rather than letting it pass silently.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

__all__ = [
    "ELLIPTIC_FILES",
    "ELLIPTIC_PLUS_FILES",
    "DatasetPaths",
    "find_dataset",
    "load_elliptic",
    "load_elliptic_plus",
    "describe_capabilities",
    "SchemaError",
]


class SchemaError(ValueError):
    """A dataset file exists but does not have the shape we expect.

    Raised with the specific mismatch, because "KeyError: txId" three modules later is a
    much worse experience than being told the header is wrong at load time.
    """


# Canonical filenames. Accept a few spellings each, because the files get renamed as they
# pass between Kaggle, Drive folders and people's laptops.
ELLIPTIC_FILES = {
    "features": ["elliptic_txs_features.csv", "elliptic_txs_features.csv.gz"],
    "classes": ["elliptic_txs_classes.csv", "elliptic_txs_classes.csv.gz"],
    "edges": ["elliptic_txs_edgelist.csv", "elliptic_txs_edgelist.csv.gz"],
}

ELLIPTIC_PLUS_FILES = {
    "wallets": [
        "wallets_features_classes_combined.csv",
        "wallets_features.csv",
        "AddrAddr_edgelist.csv",
    ],
    "addr_tx": ["AddrTx_edgelist.csv"],
    "tx_addr": ["TxAddr_edgelist.csv"],
    "addr_addr": ["AddrAddr_edgelist.csv"],
}

# Elliptic's class column: 1 = illicit, 2 = licit, "unknown" = unlabelled. The 1/2 coding
# is the opposite of what most people assume on first read, which is exactly why it is
# mapped explicitly here rather than inline.
ELLIPTIC_CLASS_MAP = {"1": "illicit", "2": "licit", "unknown": "unknown"}

N_ELLIPTIC_FEATURE_COLS = 167  # txId + time_step + 165 further features
EXPECTED_ELLIPTIC_ROWS = 203_769


@dataclass
class DatasetPaths:
    kind: str  # "elliptic" | "elliptic_plus"
    root: Path
    files: dict[str, Path]

    def __str__(self) -> str:
        return f"{self.kind} at {self.root}"


def _open_maybe_gz(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _search(root: Path, names: list[str]) -> Path | None:
    """Find a file by any of its accepted names, anywhere under root."""
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
        for found in root.rglob(name):
            return found
    return None


def find_dataset(raw_dir: Path) -> DatasetPaths | None:
    """Locate whichever real dataset is present, preferring Elliptic++.

    Elliptic++ is preferred because it is the only one that supports entity resolution --
    the thing that distinguishes this project from a classifier.
    """
    plus_wallets = _search(raw_dir, ELLIPTIC_PLUS_FILES["wallets"])
    plus_addrtx = _search(raw_dir, ELLIPTIC_PLUS_FILES["addr_tx"])
    if plus_wallets and plus_addrtx:
        files = {"wallets": plus_wallets, "addr_tx": plus_addrtx}
        tx_addr = _search(raw_dir, ELLIPTIC_PLUS_FILES["tx_addr"])
        if tx_addr:
            files["tx_addr"] = tx_addr
        return DatasetPaths("elliptic_plus", raw_dir, files)

    found = {k: _search(raw_dir, names) for k, names in ELLIPTIC_FILES.items()}
    if all(found.values()):
        return DatasetPaths("elliptic", raw_dir, {k: v for k, v in found.items() if v})
    return None


def describe_capabilities(kind: str) -> dict[str, object]:
    """What a build from this dataset can and cannot honestly claim.

    Consumed by the build report and surfaced in the API, so a demo can never silently
    present a transaction-level build as if it did entity resolution.
    """
    if kind == "elliptic_plus":
        return {
            "dataset": "Elliptic++ (Elmougy & Liu, KDD 2023)",
            "unit": "wallet address",
            "entity_resolution": True,
            "transfer_amounts": True,
            "typology_rules": "full",
            "caveats": [
                "Labels are heuristic vendor labels, not criminal convictions.",
            ],
        }
    if kind == "elliptic":
        return {
            "dataset": "Elliptic (Weber et al., KDD workshop 2019)",
            "unit": "transaction",
            "entity_resolution": False,
            "transfer_amounts": False,
            "typology_rules": "degraded",
            "caveats": [
                "Nodes are transactions, not wallets. No addresses exist in this dataset, "
                "so co-spend clustering cannot run and no claim about 'which wallets "
                "belong to one gang' is supported by this build.",
                "The edge list carries no transfer amounts and no per-edge timestamps, so "
                "pass-through, structuring and peel-chain rules cannot fire correctly.",
                "The 166 feature columns are anonymised, so attributions name features "
                "but cannot explain their meaning.",
                "Only ~23% of nodes carry a label (~2% illicit, ~21% licit).",
            ],
        }
    return {"dataset": kind, "unit": "unknown", "entity_resolution": False}


# ---------------------------------------------------------------------------------------
# Classic Elliptic
# ---------------------------------------------------------------------------------------


def load_elliptic(paths: DatasetPaths, strict: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load classic Elliptic into the frozen entity/edge contract.

    One transaction becomes one "entity". That is a deliberate, documented degradation --
    the dataset offers nothing finer -- and ``describe_capabilities`` records that this
    build does not do entity resolution.

    Parameters
    ----------
    strict
        Fail if the row count does not match the published 203,769. Off by default because
        the dataset has been revised, and a size mismatch is worth a warning rather than a
        crash.
    """
    feat_path, class_path, edge_path = (
        paths.files["features"],
        paths.files["classes"],
        paths.files["edges"],
    )

    # The features file ships WITHOUT a header row -- a classic first-hour trap, since
    # pandas will silently eat transaction one as the column names.
    with _open_maybe_gz(feat_path) as fh:
        features = pd.read_csv(fh, header=None)

    n_cols = features.shape[1]
    if n_cols != N_ELLIPTIC_FEATURE_COLS:
        raise SchemaError(
            f"{feat_path.name}: expected {N_ELLIPTIC_FEATURE_COLS} columns "
            f"(txId + time_step + 165 features), found {n_cols}.\n"
            "If the first row looks like column names, the file was written with a header "
            "and needs header=0 instead."
        )

    features.columns = (
        ["txId", "time_step"]
        + [f"local_{i}" for i in range(1, 94)]
        + [f"agg_{i}" for i in range(1, 73)]
    )
    features["txId"] = features["txId"].astype(str)

    if strict and len(features) != EXPECTED_ELLIPTIC_ROWS:
        raise SchemaError(
            f"{feat_path.name}: expected {EXPECTED_ELLIPTIC_ROWS:,} rows, "
            f"found {len(features):,}"
        )

    with _open_maybe_gz(class_path) as fh:
        classes = pd.read_csv(fh)
    if set(classes.columns) != {"txId", "class"}:
        raise SchemaError(
            f"{class_path.name}: expected columns ['txId', 'class'], "
            f"found {list(classes.columns)}"
        )
    classes["txId"] = classes["txId"].astype(str)
    classes["label"] = (
        classes["class"].astype(str).map(ELLIPTIC_CLASS_MAP).fillna("unknown")
    )

    with _open_maybe_gz(edge_path) as fh:
        edges = pd.read_csv(fh)
    if set(edges.columns) != {"txId1", "txId2"}:
        raise SchemaError(
            f"{edge_path.name}: expected columns ['txId1', 'txId2'], "
            f"found {list(edges.columns)}"
        )

    merged = features.merge(classes[["txId", "label"]], on="txId", how="left")
    merged["label"] = merged["label"].fillna("unknown")

    # Build all 165 feature columns in one concat rather than inserting them one at a
    # time. On the real 203,769-row file the incremental version fragments the frame badly
    # and turns a fast load into a slow one.
    feature_cols = [c for c in merged.columns if c.startswith(("local_", "agg_"))]
    entities = pd.concat(
        [
            pd.DataFrame(
                {
                    "entity_id": merged["txId"],
                    "label": merged["label"],
                    "first_ts": merged["time_step"].astype(float),
                    "last_ts": merged["time_step"].astype(float),
                    # A transaction is not a wallet -- see describe_capabilities().
                    "n_addresses": 1.0,
                }
            ),
            merged[feature_cols].astype(float),
        ],
        axis=1,
    )

    ts_by_tx = dict(zip(merged["txId"], merged["time_step"]))
    edge_df = pd.DataFrame(
        {
            "src_entity": edges["txId1"].astype(str),
            "dst_entity": edges["txId2"].astype(str),
            # Elliptic publishes no amounts. 1.0 is a placeholder that keeps the contract
            # satisfied; capabilities record that value-based rules are unreliable here.
            "value_btc": 1.0,
            "n_txs": 1,
        }
    )
    edge_df["ts"] = edge_df["src_entity"].map(ts_by_tx).fillna(0).astype(int)

    meta = {
        "provenance": "elliptic-real",
        "capabilities": describe_capabilities("elliptic"),
        "n_entities": len(entities),
        "n_edges": len(edge_df),
        "labels": entities["label"].value_counts().to_dict(),
        "n_timesteps": int(merged["time_step"].max()),
        "files": {k: str(v) for k, v in paths.files.items()},
    }
    return entities, edge_df, meta


# ---------------------------------------------------------------------------------------
# Elliptic++
# ---------------------------------------------------------------------------------------


def load_elliptic_plus(paths: DatasetPaths) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the Elliptic++ actors (wallet) dataset.

    This is the one that supports the project's central claim, because it has addresses.
    Column naming varies between releases, so the loader resolves the address, timestep and
    class columns by pattern rather than by fixed position, and says what it found.
    """
    wallets_path = paths.files["wallets"]
    wallets = pd.read_csv(wallets_path)

    def pick(candidates: list[str], required: bool = True) -> str | None:
        for c in candidates:
            for col in wallets.columns:
                if col.strip().lower() == c:
                    return col
        for c in candidates:  # fall back to substring
            for col in wallets.columns:
                if c in col.strip().lower():
                    return col
        if required:
            raise SchemaError(
                f"{wallets_path.name}: could not find a column matching {candidates}. "
                f"Columns present: {list(wallets.columns)[:15]}"
            )
        return None

    addr_col = pick(["address", "addr", "wallet"])
    ts_col = pick(["time step", "time_step", "timestep", "ts"])
    class_col = pick(["class", "label"])

    wallets[addr_col] = wallets[addr_col].astype(str)
    labels = (
        wallets[class_col].astype(str).map(ELLIPTIC_CLASS_MAP).fillna("unknown")
    )

    feature_cols = [
        c
        for c in wallets.columns
        if c not in {addr_col, ts_col, class_col}
        and pd.api.types.is_numeric_dtype(wallets[c])
    ]

    grouped = wallets.groupby(addr_col)
    entities = pd.DataFrame(
        {
            "entity_id": list(grouped.groups.keys()),
        }
    )
    agg = grouped[feature_cols].mean().reset_index()
    entities = entities.merge(agg, left_on="entity_id", right_on=addr_col, how="left")
    entities = entities.drop(columns=[addr_col])

    ts_agg = grouped[ts_col].agg(["min", "max"]).reset_index()
    entities = entities.merge(
        ts_agg.rename(columns={"min": "first_ts", "max": "last_ts", addr_col: "entity_id"}),
        on="entity_id",
        how="left",
    )

    lab = pd.DataFrame({addr_col: wallets[addr_col], "label": labels})
    # An address labelled illicit at any timestep is treated as illicit overall, matching
    # how an investigator would treat a wallet with any confirmed criminal activity.
    priority = {"illicit": 0, "licit": 1, "unknown": 2}
    lab["p"] = lab["label"].map(priority)
    best = lab.sort_values("p").drop_duplicates(addr_col)
    entities = entities.merge(
        best[[addr_col, "label"]].rename(columns={addr_col: "entity_id"}),
        on="entity_id",
        how="left",
    )
    entities["label"] = entities["label"].fillna("unknown")
    entities["n_addresses"] = 1.0

    edge_path = paths.files.get("addr_addr") or _search(
        paths.root, ELLIPTIC_PLUS_FILES["addr_addr"]
    )
    if edge_path and Path(edge_path).exists():
        raw_edges = pd.read_csv(edge_path)
        cols = list(raw_edges.columns)
        if len(cols) < 2:
            raise SchemaError(f"{edge_path}: expected at least 2 columns, found {cols}")
        edge_df = pd.DataFrame(
            {
                "src_entity": raw_edges[cols[0]].astype(str),
                "dst_entity": raw_edges[cols[1]].astype(str),
                "value_btc": 1.0,
                "n_txs": 1,
                "ts": 1,
            }
        )
    else:
        edge_df = pd.DataFrame(
            columns=["src_entity", "dst_entity", "value_btc", "n_txs", "ts"]
        )

    meta = {
        "provenance": "elliptic-plus-real",
        "capabilities": describe_capabilities("elliptic_plus"),
        "n_entities": len(entities),
        "n_edges": len(edge_df),
        "labels": entities["label"].value_counts().to_dict(),
        "resolved_columns": {"address": addr_col, "time_step": ts_col, "class": class_col},
        "n_features": len(feature_cols),
        "files": {k: str(v) for k, v in paths.files.items()},
    }
    return entities, edge_df, meta


DOWNLOAD_HELP = """
No real dataset found in data/raw/.

Neither can be downloaded automatically -- both are behind a login -- so fetch them by hand:

  Elliptic++ (PREFERRED: has wallet addresses, supports entity resolution)
    https://github.com/git-disl/EllipticPlusPlus  -> the Google Drive link in its README
    Put the Actors Dataset CSVs in:  data/raw/elliptic_plus/

  Elliptic classic (the literature benchmark; transactions only, no addresses)
    Kaggle: ellipticco/elliptic-data-set
    Put these three files in:  data/raw/elliptic/
      elliptic_txs_features.csv
      elliptic_txs_classes.csv
      elliptic_txs_edgelist.csv

Then re-run:  python -m ml.ingest.build --real

Nothing else needs changing -- clustering, features, model, API and UI all read the same
contract regardless of which dataset produced it.
""".strip()
