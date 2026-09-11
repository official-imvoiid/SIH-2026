"""Which typology rules have actually been shown to work on real Bitcoin.

The gap this closes
-------------------
``ml/typology/calibrate.py`` measures each rule against confirmed ransom wallets and
ordinary wallets, and writes the verdict to ``ml/artifacts/calibrated_thresholds.json``.
Until now nothing read that file, so the interface presented all seven indicators as
equally trustworthy when only one had demonstrated any separation on real data.

That is precisely the overclaim this project exists to avoid. An investigator looking at
two flags on a wallet needs to know that one of them has been tested against real criminal
money and the other has not.

This module is the bridge. It stays free of any ML dependency, so the typology layer keeps
working when the model does not.

Design note: an unvalidated rule is not *wrong*, and we do not hide its flags. It is
untested, and it is labelled untested. Suppressing it would discard a genuine observation
about behaviour; presenting it as equivalent to a validated one would overstate what we
know. Labelling is the honest middle.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ml.config import ARTIFACTS

__all__ = ["validation_for", "annotate", "validation_summary", "clear_cache"]

# Rule function name -> the FATF-style code its Typology carries.
RULE_TO_CODE: dict[str, str] = {
    "detect_fan_in": "VA-CONSOLIDATION",
    "detect_fan_out": "VA-DISPERSION",
    "detect_rapid_passthrough": "VA-PASSTHROUGH",
    "detect_peel_chain": "VA-PEELCHAIN",
    "detect_structuring": "VA-STRUCTURING",
    "detect_dormant_burst": "VA-DORMANCY",
    "detect_illicit_proximity": "VA-PROXIMITY",
}
CODE_TO_RULE = {v: k for k, v in RULE_TO_CODE.items()}


@lru_cache(maxsize=1)
def _load(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def clear_cache() -> None:
    """Drop the cached calibration record. Call after re-running calibration."""
    _load.cache_clear()


def validation_for(code: str, artifacts: Path | None = None) -> dict[str, Any] | None:
    """What real-data evidence supports this indicator?

    Returns ``None`` when no calibration has been run at all -- distinct from a rule that
    was tested and failed, which returns a record with ``validated: False``. The interface
    should word those two cases differently: "never tested" is not "tested and rejected".
    """
    data = _load(str(artifacts or ARTIFACTS / "calibrated_thresholds.json"))
    if not data:
        return None

    rule_name = CODE_TO_RULE.get(code)
    per_rule = data.get("per_rule", {})
    if rule_name is None or rule_name not in per_rule:
        # Rules outside the sweep (proximity is contextual, not swept) are simply untested.
        return {
            "validated": False,
            "tested": False,
            "note": "Not evaluated against real blockchain data.",
        }

    rec = per_rule[rule_name]
    held = (
        rec.get("best_on_holdout")
        if rec.get("tuning_helped")
        else rec.get("default_on_holdout")
    ) or rec.get("best_on_holdout") or {}

    validated = rule_name in set(data.get("survivors", []))
    tpr, fpr = held.get("tpr", 0.0), held.get("fpr", 0.0)
    n_pos, n_neg = held.get("n_pos", 0), held.get("n_neg", 0)

    if validated:
        note = (
            f"Tested on real Bitcoin: detected {tpr:.0%} of confirmed ransom wallets "
            f"while flagging {fpr:.0%} of ordinary wallets, on ransomware families the "
            f"calibration never saw."
        )
    else:
        note = (
            f"Tested on real Bitcoin and did NOT separate criminal from ordinary money "
            f"({tpr:.0%} vs {fpr:.0%}). Treat this flag as a description of behaviour, "
            f"not as evidence."
        )

    return {
        "validated": validated,
        "tested": True,
        "holdout_tpr": round(float(tpr), 4),
        "holdout_fpr": round(float(fpr), 4),
        "holdout_n_pos": int(n_pos),
        "holdout_n_neg": int(n_neg),
        # Small samples are the honest caveat on all of this; surface it rather than
        # letting a 71% read like it came from thousands of examples.
        "sample_is_small": (n_pos + n_neg) < 60,
        "note": note,
    }


def annotate(typology: dict[str, Any], artifacts: Path | None = None) -> dict[str, Any]:
    """Attach real-data evidence to one typology dict. Returns a new dict."""
    out = dict(typology)
    v = validation_for(out.get("code", ""), artifacts)
    out["validation"] = v
    out["validated"] = bool(v and v.get("validated"))
    return out


def validation_summary(artifacts: Path | None = None) -> dict[str, Any]:
    """Project-level summary, for the UI banner and the API health endpoint."""
    data = _load(str(artifacts or ARTIFACTS / "calibrated_thresholds.json"))
    if not data:
        return {
            "calibrated": False,
            "n_validated": 0,
            "n_tested": 0,
            "message": (
                "Detection rules have not been validated against real blockchain data "
                "in this build."
            ),
        }

    survivors = data.get("survivors", [])
    per_rule = data.get("per_rule", {})
    n = data.get("n_labelled", {})
    return {
        "calibrated": True,
        "n_validated": len(survivors),
        "n_tested": len(per_rule),
        "validated_codes": sorted(RULE_TO_CODE[s] for s in survivors if s in RULE_TO_CODE),
        "holdout_size": {
            "ransom_wallets": n.get("pos_holdout", 0),
            "ordinary_wallets": n.get("neg_holdout", 0),
        },
        "message": (
            f"{len(survivors)} of {len(per_rule)} detection rules separated confirmed "
            f"ransom wallets from ordinary wallets on held-out ransomware families."
        ),
    }
