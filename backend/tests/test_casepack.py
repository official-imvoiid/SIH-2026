"""Tests for the case-pack export.

The assertions here are mostly *ethical* rather than functional: a case pack that renders
beautifully but drops the caveat, or hides exculpatory evidence, is a worse defect than one
that crashes. A crash is visible. A confidently-worded overclaim in a police document is not.
"""

import json
import re

import networkx as nx
import pytest

from backend.app.services.casepack import CAVEAT, _calibration_note, _fmt_evidence, render


class FakeStore:
    """Minimal store stand-in, so these tests need no dataset and no trained model."""

    def __init__(self, typologies=None, attrs=None, risk=0.91):
        self._risk = risk
        self._typs = typologies if typologies is not None else []
        self._attrs = attrs if attrs is not None else [
            {"feature": "fwd_ratio", "readable": "share of received funds forwarded on",
             "value": 0.99, "contribution": 0.31},
            {"feature": "lifetime_ts", "readable": "active lifespan",
             "value": 2.0, "contribution": -0.12},
        ]
        self.graph = nx.DiGraph()

        class _E:
            def __init__(self):
                self.index = ["E-00001"]

            def loc(self, *_a):
                return {"label": "illicit"}

        self.entities = type(
            "T", (), {"loc": {"E-00001": {"label": "illicit"}}, "index": ["E-00001"]}
        )()

    def risk(self, _eid):
        return self._risk

    def exists(self, eid):
        return eid == "E-00001"

    def typologies(self, _eid):
        return self._typs

    def explain(self, _eid, top_k=10):
        return self._attrs, "tree_path", "driven by forwarding ratio"

    def narrative(self, _eid):
        return "Entity E-00001 is flagged with a model risk score of 0.91."

    def neighbors(self, _eid, limit=25):
        return [{"id": "E-00002", "direction": "in", "value_btc": 1.5, "n_txs": 3, "risk": 0.2}]

    def features(self, _eid):
        return {"n_addresses": 4.0, "fwd_ratio": 0.99}


def test_caveat_is_always_present_even_with_no_typologies():
    html = render(FakeStore(typologies=[]), "E-00001")
    assert "investigative priority, not proof of criminal conduct" in html


def test_caveat_is_present_on_a_maximum_risk_entity():
    """The temptation to drop hedging is strongest exactly where it matters most."""
    html = render(FakeStore(risk=1.0), "E-00001")
    assert CAVEAT.split(".")[0] in html


def test_exculpatory_evidence_is_shown_not_filtered():
    html = render(FakeStore(), "E-00001")
    assert "Arguing against the flag" in html
    assert "active lifespan" in html  # the negative-contribution feature
    assert "-0.1200" in html or "-0.12" in html


def test_behavioural_indicators_lead_contextual_ones():
    """Association must never be presented above conduct."""
    typs = [
        {"code": "VA-PROXIMITY", "name": "Proximity to known illicit entities",
         "severity": "high", "description": "neighbours are illicit", "evidence": {}},
        {"code": "VA-PASSTHROUGH", "name": "Rapid pass-through", "severity": "high",
         "description": "forwarded everything", "evidence": {}},
    ]
    html = render(FakeStore(typologies=typs), "E-00001")
    assert html.index("Behavioural indicators") < html.index("Contextual indicators")
    assert html.index("Rapid pass-through") < html.index("Proximity to known illicit")


def test_contextual_indicators_carry_a_downgrading_note():
    typs = [{"code": "VA-PROXIMITY", "name": "Proximity", "severity": "high",
             "description": "x", "evidence": {}}]
    html = render(FakeStore(typologies=typs), "E-00001")
    assert "cannot establish anything on their own" in html


def test_provenance_block_is_populated():
    html = render(FakeStore(), "E-00001")
    for field in ("Case reference", "Generated", "Model version", "Attribution method"):
        assert field in html


def test_limitations_section_states_the_clustering_boundary():
    html = render(FakeStore(), "E-00001")
    assert "does <em>not</em> establish who that holder is" in html
    assert "not from criminal convictions" in html


def test_html_escapes_entity_content():
    """Evidence values reach this from data; they must not be able to inject markup."""
    typs = [{"code": "X", "name": "<script>alert(1)</script>", "severity": "low",
             "description": "d", "evidence": {"sample": ["<img onerror=x>"]}}]
    html = render(FakeStore(typologies=typs), "E-00001")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_no_typology_case_says_low_confidence():
    html = render(FakeStore(typologies=[]), "E-00001")
    assert "low-confidence lead" in html


def test_fmt_evidence_truncates_long_lists():
    out = _fmt_evidence({"senders": [f"E-{i:05d}" for i in range(20)]})
    assert "+14 more" in out


def test_calibration_note_never_claims_unvalidated_reliability(tmp_path, monkeypatch):
    """With no calibration record on disk, the pack must say so rather than stay silent."""
    import backend.app.services.casepack as cp

    monkeypatch.setattr(cp, "ARTIFACTS", tmp_path)
    note = _calibration_note()
    assert "not</strong> been validated" in note or "not been validated" in note


def test_calibration_note_reports_a_negative_result_plainly(tmp_path, monkeypatch):
    import backend.app.services.casepack as cp

    (tmp_path / "calibrated_thresholds.json").write_text(
        json.dumps({"survivors": [], "per_rule": {},
                    "n_labelled": {"pos_holdout": 7, "neg_holdout": 7}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cp, "ARTIFACTS", tmp_path)
    note = _calibration_note()
    assert "No rule separated" in note
