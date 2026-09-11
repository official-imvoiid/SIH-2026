"""Tests for real-data validation labelling.

The distinction these protect is subtle and easy to lose in a refactor: there are *three*
states an indicator can be in, not two.

  * **validated**   -- tested against real criminal money, and it separated
  * **tested, failed** -- tested against real criminal money, and it did not separate
  * **untested**    -- never evaluated at all

Collapsing the last two into "not validated" would tell an investigator that a rule was
rejected when in fact nobody ever checked it. That is a different and more damaging claim.
"""

import json

import pytest

from ml.typology.validation import (
    CODE_TO_RULE,
    RULE_TO_CODE,
    annotate,
    clear_cache,
    validation_for,
    validation_summary,
)


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def calib(tmp_path):
    """A calibration record where pass-through survived and fan-out did not."""
    (tmp_path / "calibrated_thresholds.json").write_text(
        json.dumps(
            {
                "survivors": ["detect_rapid_passthrough"],
                "n_labelled": {"pos_holdout": 7, "neg_holdout": 7},
                "per_rule": {
                    "detect_rapid_passthrough": {
                        "tuning_helped": False,
                        "best_on_holdout": {"tpr": 0.71, "fpr": 0.43, "n_pos": 7, "n_neg": 7},
                        "default_on_holdout": {"tpr": 0.71, "fpr": 0.43, "n_pos": 7, "n_neg": 7},
                    },
                    "detect_fan_out": {
                        "tuning_helped": False,
                        "best_on_holdout": {"tpr": 0.0, "fpr": 0.0, "n_pos": 7, "n_neg": 7},
                        "default_on_holdout": {"tpr": 0.12, "fpr": 0.39, "n_pos": 7, "n_neg": 7},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path / "calibrated_thresholds.json"


def test_every_rule_maps_to_a_code_and_back():
    assert len(RULE_TO_CODE) == len(CODE_TO_RULE)
    for rule, code in RULE_TO_CODE.items():
        assert CODE_TO_RULE[code] == rule


def test_rule_codes_match_the_actual_rules_module():
    """Guards against a rule being renamed and this mapping silently going stale."""
    from ml.typology import rules as R

    actual = {r.__name__ for r in R.RULES}
    assert actual == set(RULE_TO_CODE), (
        f"mapping out of sync: {actual ^ set(RULE_TO_CODE)}"
    )


def test_surviving_rule_is_marked_validated(calib):
    v = validation_for("VA-PASSTHROUGH", artifacts=calib)
    assert v["validated"] is True
    assert v["tested"] is True
    assert "detected 71%" in v["note"]


def test_failing_rule_is_tested_but_not_validated(calib):
    v = validation_for("VA-DISPERSION", artifacts=calib)
    assert v["validated"] is False
    assert v["tested"] is True
    assert "did NOT separate" in v["note"]


def test_unswept_rule_is_untested_not_rejected(calib):
    """Proximity is contextual and never swept. It must not read as 'tested and failed'."""
    v = validation_for("VA-PROXIMITY", artifacts=calib)
    assert v["validated"] is False
    assert v["tested"] is False
    assert "Not evaluated" in v["note"]


def test_no_calibration_file_returns_none(tmp_path):
    assert validation_for("VA-PASSTHROUGH", artifacts=tmp_path / "missing.json") is None


def test_small_sample_is_flagged(calib):
    v = validation_for("VA-PASSTHROUGH", artifacts=calib)
    assert v["sample_is_small"] is True, "14 examples must not read as a firm result"


def test_annotate_does_not_mutate_the_input(calib):
    original = {"code": "VA-PASSTHROUGH", "name": "Rapid pass-through"}
    out = annotate(original, artifacts=calib)
    assert "validated" not in original
    assert out["validated"] is True


def test_summary_counts_validated_rules(calib):
    s = validation_summary(artifacts=calib)
    assert s["calibrated"] is True
    assert s["n_validated"] == 1
    assert s["n_tested"] == 2
    assert s["validated_codes"] == ["VA-PASSTHROUGH"]


def test_summary_without_calibration_says_so(tmp_path):
    s = validation_summary(artifacts=tmp_path / "missing.json")
    assert s["calibrated"] is False
    assert s["n_validated"] == 0
    assert "not been validated" in s["message"]


def test_corrupt_calibration_file_degrades_safely(tmp_path):
    bad = tmp_path / "calibrated_thresholds.json"
    bad.write_text("{not json", encoding="utf-8")
    assert validation_for("VA-PASSTHROUGH", artifacts=bad) is None
    assert validation_summary(artifacts=bad)["calibrated"] is False
