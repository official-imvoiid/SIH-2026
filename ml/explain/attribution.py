"""Per-prediction feature attribution -- the "why" behind a risk score.

A number like ``0.93`` is not evidence. An investigator needs to know which observable
facts drove it, in language that survives being read out in a courtroom.

Method
------
Exact **decision-path decomposition** (Saabas) over the trained forest. Walking a sample
down one tree, every split moves the predicted probability by some amount; that movement
is attributed to the feature the split tested. Summing over all trees gives an exact
additive decomposition::

    prediction = bias + sum(contribution_i for every feature i)

This is computed from the tree structure directly, so it needs no extra dependency, runs
in microseconds, and is exact rather than sampled.

Relationship to SHAP
--------------------
SHAP values are the game-theoretically "fair" attribution and differ slightly from
path decomposition when features are correlated (Saabas is order-dependent along the tree
path). If the ``shap`` package is installed, :func:`explain_entity` uses its exact
``TreeExplainer`` automatically and reports ``method="shap"``. Otherwise it falls back to
path decomposition and reports ``method="tree_path"``. Both are additive and faithful to
the model; state which one you used rather than saying "SHAP" generically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ml.features.graph import FEATURE_LABELS

__all__ = ["Attribution", "explain_entity", "attribution_sentence", "available_method"]


@dataclass(frozen=True)
class Attribution:
    feature: str
    value: float
    contribution: float

    @property
    def readable(self) -> str:
        return FEATURE_LABELS.get(self.feature, self.feature.replace("_", " "))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "readable": self.readable,
            "value": round(float(self.value), 6),
            "contribution": round(float(self.contribution), 6),
        }


def available_method() -> str:
    try:
        import shap  # noqa: F401

        return "shap"
    except ImportError:
        return "tree_path"


def _tree_path_contributions(forest, x: np.ndarray, n_features: int) -> tuple[np.ndarray, float]:
    """Exact Saabas decomposition over every tree in the forest.

    Returns ``(contributions, bias)`` where ``bias`` is the mean root probability across
    trees and contributions sum to ``prediction - bias``.
    """
    total = np.zeros(n_features, dtype=np.float64)
    bias_sum = 0.0

    for estimator in forest.estimators_:
        t = estimator.tree_

        def prob_at(node: int) -> float:
            """P(illicit) at a node, robust to sklearn's normalised/count value formats."""
            vals = t.value[node][0]
            s = float(vals.sum())
            if s <= 0:
                return 0.0
            return float(vals[1] / s) if len(vals) > 1 else 0.0

        node = 0
        bias_sum += prob_at(0)

        while t.children_left[node] != -1:  # -1 marks a leaf
            feat = t.feature[node]
            child = (
                t.children_left[node]
                if x[feat] <= t.threshold[node]
                else t.children_right[node]
            )
            total[feat] += prob_at(child) - prob_at(node)
            node = child

    n_trees = len(forest.estimators_)
    return total / n_trees, bias_sum / n_trees


def _shap_contributions(forest, x: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(forest)
    values = explainer.shap_values(x.reshape(1, -1), check_additivity=False)
    # Layout varies across shap versions and model types; normalise to the positive class.
    arr = np.asarray(values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return arr[0, :, -1]
    if arr.ndim == 2:  # (n_samples, n_features)
        return arr[0]
    return np.asarray(values[-1])[0]


def explain_entity(
    forest,
    feature_cols: Sequence[str],
    features: dict[str, float] | np.ndarray,
    top_k: int = 8,
    prefer_shap: bool = True,
) -> tuple[list[Attribution], str]:
    """Explain one entity's risk score.

    Returns ``(attributions, method)``, ordered by absolute contribution so the strongest
    evidence -- for *or* against -- comes first. Negative contributions are kept
    deliberately: "this wallet has been active for two years, which argues against it" is
    exactly the kind of exculpatory detail a one-sided explanation would hide.
    """
    if isinstance(features, dict):
        x = np.array([float(features.get(c, 0.0)) for c in feature_cols], dtype=np.float64)
    else:
        x = np.asarray(features, dtype=np.float64).ravel()

    method = "tree_path"
    if prefer_shap:
        try:
            contribs = _shap_contributions(forest, x)
            method = "shap"
        except Exception:
            contribs, _bias = _tree_path_contributions(forest, x, len(feature_cols))
    else:
        contribs, _bias = _tree_path_contributions(forest, x, len(feature_cols))

    attributions = [
        Attribution(feature=col, value=float(x[i]), contribution=float(contribs[i]))
        for i, col in enumerate(feature_cols)
    ]
    attributions.sort(key=lambda a: abs(a.contribution), reverse=True)
    return attributions[:top_k], method


def _format_value(feature: str, value: float) -> str:
    """Render a feature value the way an investigator would write it down."""
    if feature.endswith("_btc"):
        return f"{value:.4f} BTC"
    if feature in {"fwd_ratio", "illicit_neighbour_frac"}:
        return f"{value:.1%}"
    if feature.endswith("_ts") or feature.endswith("_burst") or feature.startswith("n_"):
        return f"{value:,.0f}"
    if feature in {"in_degree", "out_degree", "total_degree", "n_addresses",
                   "illicit_neighbours", "hops_to_illicit"}:
        return f"{value:,.0f}"
    return f"{value:,.3f}"


def attribution_sentence(attributions: Sequence[Attribution], max_terms: int = 3) -> str:
    """Turn attributions into one plain-English sentence for the case pack.

    Deterministic and template-based: the same inputs always produce the same words. No
    model call, no GPU, nothing to fail on demo day.
    """
    pushing_up = [a for a in attributions if a.contribution > 0][:max_terms]
    pushing_down = [a for a in attributions if a.contribution < 0][:1]

    if not pushing_up:
        return "No individual feature pushed this entity's score materially upward."

    parts = [
        f"{a.readable} ({_format_value(a.feature, a.value)})" for a in pushing_up
    ]
    if len(parts) == 1:
        drivers = parts[0]
    else:
        drivers = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    sentence = f"The score is driven mainly by {drivers}."
    if pushing_down:
        d = pushing_down[0]
        sentence += (
            f" Arguing against the flag: {d.readable} "
            f"({_format_value(d.feature, d.value)})."
        )
    return sentence
