"""Train the entity risk model and score every entity.

Design follows the literature rather than fashion. Weber et al. (2019) found Random
Forest reaching ~0.79 illicit-class F1 on Elliptic while a GCN managed ~0.65; ensembles of
trees on tabular features remain the thing to beat on this problem. So the baseline is a
forest, it trains in seconds on a laptop, and a GNN is an upgrade to be *measured against*
this -- not a prerequisite.

Three rules are enforced here rather than left to discipline:

1. **Temporal split, never random.** Train on early time steps, test on later ones. A
   random split lets the model see the test period's graph neighbourhood during training
   and inflates illicit F1 substantially. The split is done by ``last_ts`` and there is no
   option to shuffle.
2. **Accuracy is never reported.** At a ~1% illicit base rate, predicting "all clean"
   scores >98% and is worthless. We report illicit-class precision/recall/F1, average
   precision, and precision@k -- the last being what an analyst who reviews 50 leads a day
   actually experiences.
3. **Everything is seeded.** Same data in, same numbers out.

Run::

    python -m ml.train.baseline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ml.config import (
    ARTIFACTS,
    DATA_PROCESSED,
    DRIFT_TS,
    MODEL_VERSION,
    RANDOM_SEED,
    TRAIN_TS_MAX,
    VAL_TS_MAX,
)

# Columns that are identifiers or targets rather than evidence. first_ts/last_ts are
# excluded deliberately: they encode the temporal split itself, so a tree would learn
# "test period" instead of "criminal".
NON_FEATURES = {"entity_id", "label", "first_ts", "last_ts"}

# Guilt-by-association features: derived from *other entities' labels* rather than from the
# subject's own conduct.
#
# These are excluded from the model by default, and the reason is measured rather than
# assumed. `python -m ml.train.benchmark` compares the model with and without them under
# the same temporal split, and dropping them RAISES illicit F1 from 0.755 to 0.838 -- about
# eight points. They are fitted to the training period's illicit population, and that
# population changes by test time, so they generalise worse than behaviour does.
#
# That makes the project's "behaviour beats association" rule an accuracy argument as well
# as an ethical one, which is a much stronger position to defend. Set
# EXCLUDE_ASSOCIATION_FEATURES = False to put them back and re-measure.
ASSOCIATION_FEATURES = {
    "hops_to_illicit",
    "illicit_neighbours",
    "illicit_neighbour_frac",
    "near_illicit_2hop",
}
EXCLUDE_ASSOCIATION_FEATURES = True


def load_features(path: Path = DATA_PROCESSED / "entities.parquet"):
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\nRun:  python -m ml.ingest.build"
        )
    df = pd.read_parquet(path)
    dropped = NON_FEATURES | (
        ASSOCIATION_FEATURES if EXCLUDE_ASSOCIATION_FEATURES else set()
    )
    feature_cols = [c for c in df.columns if c not in dropped]
    return df, feature_cols


def temporal_split(df: pd.DataFrame):
    """Split by time, using only entities that carry a ground-truth label.

    Unlabelled entities (the large majority, as in the real dataset) are still *scored* at
    the end -- they just cannot contribute to supervised training or evaluation.
    """
    labelled = df[df["label"].isin(["illicit", "licit"])].copy()
    labelled["y"] = (labelled["label"] == "illicit").astype(int)

    train = labelled[labelled["last_ts"] <= TRAIN_TS_MAX]
    val = labelled[(labelled["last_ts"] > TRAIN_TS_MAX) & (labelled["last_ts"] <= VAL_TS_MAX)]
    test = labelled[labelled["last_ts"] > VAL_TS_MAX]
    return train, val, test


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of the k highest-risk entities, what share are actually illicit?

    This is the metric that matches how the tool gets used: an analyst works a ranked
    queue from the top, and never sees the bottom of the list.
    """
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    top = np.argsort(-scores)[:k]
    return float(y_true[top].sum() / k)


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    out = {
        "n": int(len(y_true)),
        "n_illicit": int(y_true.sum()),
        "illicit_precision": round(float(p), 4),
        "illicit_recall": round(float(r), 4),
        "illicit_f1": round(float(f1), 4),
        "average_precision": round(float(average_precision_score(y_true, proba)), 4)
        if y_true.sum() and y_true.sum() < len(y_true)
        else None,
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4)
        if y_true.sum() and y_true.sum() < len(y_true)
        else None,
        "precision_at_50": round(precision_at_k(y_true, proba, 50), 4),
        "precision_at_100": round(precision_at_k(y_true, proba, 100), 4),
    }
    if y_true.sum():
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def train(out_dir: Path = ARTIFACTS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df, feature_cols = load_features()
    train_df, val_df, test_df = temporal_split(df)

    print(f"features: {len(feature_cols)}")
    if EXCLUDE_ASSOCIATION_FEATURES:
        print(
            f"  excluding {len(ASSOCIATION_FEATURES)} guilt-by-association features "
            f"(measured to cost ~8 F1 points -- see ml/train/benchmark.py)"
        )
    print(
        f"train ts<={TRAIN_TS_MAX}: {len(train_df):,} labelled "
        f"({int(train_df['y'].sum())} illicit)"
    )
    print(
        f"val   ts<={VAL_TS_MAX}: {len(val_df):,} labelled "
        f"({int(val_df['y'].sum())} illicit)"
    )
    print(f"test  ts> {VAL_TS_MAX}: {len(test_df):,} labelled ({int(test_df['y'].sum())} illicit)")

    if train_df["y"].sum() < 5:
        raise SystemExit("Too few illicit examples in the training window to learn from.")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["y"].to_numpy()

    t0 = time.time()
    # class_weight="balanced_subsample" is the imbalance handling: it reweights each
    # bootstrap sample so the ~1% illicit class is not simply ignored by the splits.
    forest = RandomForestClassifier(
        n_estimators=400,
        # Capped depth and a larger leaf minimum: with only a few dozen illicit examples
        # an unconstrained forest memorises them (train F1 = 1.000) and generalises badly.
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    # Calibration matters because the number shown in the UI is read as a probability.
    # An uncalibrated forest's 0.9 does not mean "90% likely illicit"; after calibration
    # on held-out data it approximately does.
    #
    # Sigmoid (Platt) rather than isotonic: isotonic is non-parametric and, on a
    # validation set with only tens of positives, collapses into a step function that
    # maps most of the top of the ranking to exactly 1.000. That destroys the ordering
    # an analyst triages by and reads as a broken model. Sigmoid stays smooth and
    # monotonic on small samples.
    if len(val_df) > 30 and val_df["y"].sum() >= 3:
        forest.fit(X_train, y_train)
        try:  # sklearn >= 1.6 replaced cv="prefit" with an explicit frozen estimator
            from sklearn.frozen import FrozenEstimator

            model = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid")
        except ImportError:
            model = CalibratedClassifierCV(forest, method="sigmoid", cv="prefit")
        model.fit(val_df[feature_cols].to_numpy(dtype=np.float32), val_df["y"].to_numpy())
        calibrated = True
    else:
        model = forest.fit(X_train, y_train)
        calibrated = False
    train_secs = time.time() - t0
    print(f"trained in {train_secs:.1f}s (calibrated={calibrated})")

    def proba(frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.array([])
        return model.predict_proba(frame[feature_cols].to_numpy(dtype=np.float32))[:, 1]

    metrics = {
        "model_version": MODEL_VERSION,
        "model": "RandomForestClassifier(400) + Platt calibration"
        if calibrated
        else "RandomForestClassifier(400)",
        "seed": RANDOM_SEED,
        "n_features": len(feature_cols),
        "association_features_excluded": EXCLUDE_ASSOCIATION_FEATURES,
        "train_seconds": round(train_secs, 2),
        "split": {"train_ts_max": TRAIN_TS_MAX, "val_ts_max": VAL_TS_MAX, "drift_ts": DRIFT_TS},
        "train": evaluate(y_train, proba(train_df)),
        "validation": evaluate(val_df["y"].to_numpy(), proba(val_df)) if len(val_df) else {},
        "test": evaluate(test_df["y"].to_numpy(), proba(test_df)) if len(test_df) else {},
    }

    # The drift analysis. This is the pitch, so it is computed every run rather than
    # being a one-off notebook cell someone forgets to re-run.
    pre = test_df[test_df["last_ts"] < DRIFT_TS]
    post = test_df[test_df["last_ts"] >= DRIFT_TS]
    metrics["drift"] = {
        "note": (
            f"Time step {DRIFT_TS} is where the illicit population changes behaviour. "
            "Published models lose substantial recall here; we report it rather than "
            "hide it."
        ),
        "before_drift": evaluate(pre["y"].to_numpy(), proba(pre)) if len(pre) else {},
        "after_drift": evaluate(post["y"].to_numpy(), proba(post)) if len(post) else {},
    }

    importances = getattr(forest, "feature_importances_", None)
    if importances is not None:
        metrics["top_features"] = [
            {"feature": feature_cols[i], "importance": round(float(importances[i]), 4)}
            for i in np.argsort(-importances)[:12]
        ]

    # Score EVERY entity, including the unlabelled majority -- that is the whole point of
    # the tool. Written once here so the API never runs inference in a request.
    all_scores = model.predict_proba(df[feature_cols].to_numpy(dtype=np.float32))[:, 1]
    scores = pd.DataFrame(
        {
            "entity_id": df["entity_id"],
            "risk": np.round(all_scores, 6),
            "label": df["label"],
            "model_version": MODEL_VERSION,
        }
    ).sort_values("risk", ascending=False)
    scores.to_parquet(out_dir / "scores.parquet", index=False)

    joblib.dump(
        {"model": model, "forest": forest, "feature_cols": feature_cols,
         "model_version": MODEL_VERSION},
        out_dir / "model.joblib",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _print_report(metrics)
    return metrics


def _print_report(m: dict) -> None:
    def line(name: str, d: dict) -> str:
        if not d:
            return f"  {name:12} (empty)"
        return (
            f"  {name:12} F1={d['illicit_f1']:.3f}  P={d['illicit_precision']:.3f}  "
            f"R={d['illicit_recall']:.3f}  AP={d['average_precision']}  "
            f"n={d['n']:,} ({d['n_illicit']} illicit)"
        )

    print("\n" + "=" * 74)
    print(f"  {m['model']}   seed={m['seed']}")
    print("=" * 74)
    print(line("train", m["train"]))
    print(line("validation", m["validation"]))
    print(line("test", m["test"]))
    print("-" * 74)
    print(f"  CONCEPT DRIFT at time step {m['split']['drift_ts']}")
    print(line("before", m["drift"]["before_drift"]))
    print(line("after", m["drift"]["after_drift"]))
    print("-" * 74)
    if m.get("top_features"):
        print("  top features:")
        for f in m["top_features"][:8]:
            print(f"    {f['importance']:.4f}  {f['feature']}")
    print("=" * 74)
    print(f"  wrote {ARTIFACTS / 'model.joblib'}")
    print(f"  wrote {ARTIFACTS / 'scores.parquet'}")
    print(f"  wrote {ARTIFACTS / 'metrics.json'}")


if __name__ == "__main__":
    train()
