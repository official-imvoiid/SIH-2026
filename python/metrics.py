"""Evaluation metrics.

One rule governs this file: **never report accuracy**.

Illicit entities are somewhere around two percent of a realistic dataset. A model that
predicts "everybody is clean" and does nothing at all scores 98% accuracy. Quoting that
number is not a mistake in presentation, it is a false claim about what the tool does.

What is reported instead:

  illicit F1     the harmonic mean of precision and recall on the class that matters
  precision      of the entities we flagged, how many were actually illicit
  recall         of the illicit entities that exist, how many we caught
  AUC-PR         ranking quality across every threshold, on the rare class
  precision@k    of the top k the analyst will actually have time to open, how many
                 were worth opening -- the number that decides whether the tool helps
"""

from __future__ import annotations

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    c = confusion(y_true, y_pred)
    precision = _safe_div(c["tp"], c["tp"] + c["fp"])
    recall = _safe_div(c["tp"], c["tp"] + c["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        **c,
    }


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve, computed as the step-wise average.

    Preferred to ROC-AUC on imbalanced data. ROC-AUC looks flattering when negatives
    vastly outnumber positives, because a large absolute number of false positives is
    still a small false-positive *rate*. PR does not let you hide that.
    """
    if len(y_true) == 0 or y_true.sum() == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / y.sum()

    # Sum precision at each point where recall actually increases.
    prev_recall = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        if r > prev_recall:
            ap += p * (r - prev_recall)
            prev_recall = r
    return round(float(ap), 4)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> dict:
    """How much of the analyst's first k clicks were worth making."""
    if len(y_true) == 0:
        return {"k": k, "precision": 0.0, "hits": 0}
    k = min(k, len(y_true))
    order = np.argsort(-y_score, kind="mergesort")[:k]
    hits = int(y_true[order].sum())
    return {"k": k, "precision": round(hits / k, 4), "hits": hits}


def youden_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-positive rate minus false-positive rate.

    The right metric for asking "does this rule beat guessing?" A rule that fires on 80%
    of criminals sounds excellent until you learn it also fires on 75% of everyone else,
    at which point J = 0.05 and it is nearly worthless. J makes that visible in one number.
    """
    c = confusion(y_true, y_pred)
    tpr = _safe_div(c["tp"], c["tp"] + c["fn"])
    fpr = _safe_div(c["fp"], c["fp"] + c["tn"])
    return round(tpr - fpr, 4)


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """The full report for one split."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "n": int(len(y_true)),
        "n_illicit": int(y_true.sum()),
        "base_rate": round(float(y_true.mean()) if len(y_true) else 0.0, 4),
        "threshold": threshold,
        "illicit": prf(y_true, y_pred),
        "auc_pr": average_precision(y_true, y_score),
        "youden_j": youden_j(y_true, y_pred),
    }
    for k in (10, 20, 50, 100):
        out[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)

    # Accuracy is computed but immediately labelled, so that if it ever appears in a
    # report the caveat travels with it instead of being dropped somewhere upstream.
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    out["accuracy_do_not_quote"] = {
        "value": round(acc, 4),
        "why": (
            f"{round((1 - out['base_rate']) * 100, 1)}% of entities here are not illicit, so "
            "predicting 'clean' for everything scores about this well while finding nothing. "
            "Quote illicit F1 and precision@k instead."
        ),
    }
    return out


def best_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1") -> dict:
    """Sweep thresholds on a validation split and return the best one.

    Fitted on validation, never on test. A threshold chosen by looking at test results is
    a threshold that has seen the answers, and the score it produces is fiction.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or y_true.sum() == 0:
        return {"threshold": 0.5, "score": 0.0, "note": "no positives in validation split"}

    candidates = np.unique(np.round(np.linspace(0.05, 0.95, 91), 3))
    best = {"threshold": 0.5, "score": -1.0}
    for t in candidates:
        pred = (y_score >= t).astype(int)
        s = prf(y_true, pred)["f1"] if metric == "f1" else youden_j(y_true, pred)
        if s > best["score"]:
            best = {"threshold": float(t), "score": float(s)}
    return best
