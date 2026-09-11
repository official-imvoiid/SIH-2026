"""Case-pack export -- the document an investigator actually hands over.

Renders one entity's full evidence file as a self-contained HTML page designed to be
printed to PDF from the browser (Ctrl+P). No WeasyPrint, no GTK, no system libraries: a
hackathon demo cannot afford a dependency that fails to install on the demo laptop, and
every browser already has a PDF writer.

What a case pack has to do that a UI does not
---------------------------------------------
A screen is read by the person who ran the query. A case pack is read by somebody who was
not there -- a supervising officer, a prosecutor, a defence lawyer. So it carries the
things a screen can leave implicit:

* **Provenance.** Which dataset, which model version, which thresholds, generated when.
  Evidence that cannot be traced to its inputs is not evidence.
* **The limits, stated in the document itself** rather than in a footnote nobody reads.
  Including the finding that our detection thresholds do not yet separate real criminal
  money -- because a case pack that overstated its own reliability would be worse than
  useless.
* **Exculpatory material.** Feature attributions that argue *against* the flag are printed
  alongside those that argue for it.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.config import ARTIFACTS, MODEL_VERSION

CAVEAT = (
    "Risk scores reflect statistical modelling and structural pattern matching against "
    "heuristically labelled data. They indicate investigative priority, not proof of "
    "criminal conduct. No person should be accused on the basis of this document alone."
)


def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _fmt_evidence(evidence: dict) -> str:
    """Render a typology's evidence dict as a readable definition list."""
    if not evidence:
        return ""
    rows = []
    for k, v in evidence.items():
        label = k.replace("_", " ")
        if isinstance(v, list):
            shown = ", ".join(_esc(i) for i in v[:6])
            if len(v) > 6:
                shown += f" <span class='muted'>(+{len(v) - 6} more)</span>"
            val = shown or "<span class='muted'>none</span>"
        elif isinstance(v, float):
            val = f"{v:,.6f}".rstrip("0").rstrip(".")
        else:
            val = _esc(v)
        rows.append(f"<div class='ev-k'>{_esc(label)}</div><div class='ev-v'>{val}</div>")
    return f"<div class='ev'>{''.join(rows)}</div>"


def _calibration_note() -> str:
    """Report what recalibration against real data actually established.

    Read live from the artifact rather than hardcoded, so the document can never claim a
    reliability the current thresholds have not demonstrated.
    """
    path = ARTIFACTS / "calibrated_thresholds.json"
    if not path.exists():
        return (
            "<p>Detection thresholds have <strong>not</strong> been validated against real "
            "blockchain data in this build. Treat every flag as unvalidated.</p>"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "<p>Threshold calibration record unreadable.</p>"

    survivors = data.get("survivors", [])
    n = data.get("n_labelled", {})
    sample = (
        f"{n.get('pos_holdout', '?')} confirmed ransom wallets against "
        f"{n.get('neg_holdout', '?')} ordinary wallets"
    )
    if not survivors:
        return (
            "<p>Thresholds were tested against real Bitcoin ("
            f"{_esc(sample)}, held-out ransomware families). <strong>No rule separated "
            "confirmed ransom wallets from ordinary wallets.</strong> Structural flags in "
            "this document should be read as descriptive of behaviour, not as evidence of "
            "criminality.</p>"
        )

    items = []
    for name in survivors:
        rec = data.get("per_rule", {}).get(name, {})
        # Tolerate calibration files written by an older version of the sweep, which did
        # not record the untuned held-out score.
        held = (
            rec.get("best_on_holdout")
            if rec.get("tuning_helped", True)
            else rec.get("default_on_holdout")
        ) or rec.get("best_on_holdout") or {"tpr": 0.0, "fpr": 0.0}
        items.append(
            f"<li><strong>{_esc(name.replace('detect_', '').replace('_', ' '))}</strong> — "
            f"detects {held['tpr']:.0%} of confirmed ransom wallets while flagging "
            f"{held['fpr']:.0%} of ordinary wallets</li>"
        )
    return (
        f"<p>Thresholds were tested against real Bitcoin ({_esc(sample)}, held-out "
        f"ransomware families the calibration never saw). "
        f"{len(survivors)} of {len(data.get('per_rule', {}))} rules showed genuine "
        f"separation:</p><ul>{''.join(items)}</ul>"
        "<p>Rules not listed above did <strong>not</strong> separate real criminal money "
        "from ordinary money at any tested threshold, and any flag they raise in this "
        "document carries correspondingly little weight.</p>"
    )


def render(store, entity_id: str, max_neighbors: int = 25) -> str:
    """Build the complete case pack for one entity."""
    risk = store.risk(entity_id)
    typologies = store.typologies(entity_id)
    # Behaviour before association, per the project's explanation rule: what the subject
    # did leads; who it transacted with can only corroborate.
    struct_codes = {"VA-PROXIMITY"}
    behaviour = [t for t in typologies if t.get("code") not in struct_codes]
    association = [t for t in typologies if t.get("code") in struct_codes]

    attrs, method, _sentence = store.explain(entity_id, top_k=10)
    narrative = store.narrative(entity_id)
    neighbours = store.neighbors(entity_id, limit=max_neighbors)
    feats = store.features(entity_id)
    label = str(store.entities.loc[entity_id]["label"]) if store.exists(entity_id) else "unknown"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    case_ref = f"CT-{entity_id.replace('E-', '')}-{datetime.now(timezone.utc):%Y%m%d}"

    pushing_up = [a for a in attrs if a["contribution"] > 0][:6]
    pushing_down = [a for a in attrs if a["contribution"] < 0][:4]

    def attr_rows(items: list[dict]) -> str:
        return "".join(
            f"<tr><td>{_esc(a['readable'])}</td>"
            f"<td class='num'>{a['value']:,.4f}</td>"
            f"<td class='num {'up' if a['contribution'] > 0 else 'down'}'>"
            f"{a['contribution']:+.4f}</td></tr>"
            for a in items
        )

    typ_blocks = []
    for group, heading in ((behaviour, "Behavioural indicators"), (association, "Contextual indicators")):
        if not group:
            continue
        cards = "".join(
            f"<article class='typ sev-{_esc(t.get('severity', 'low'))}'>"
            f"<header><span class='code'>{_esc(t.get('code', ''))}</span>"
            f"<h4>{_esc(t.get('name', ''))}</h4>"
            f"<span class='sev'>{_esc(t.get('severity', ''))}</span></header>"
            f"<p>{_esc(t.get('description', ''))}</p>"
            f"{_fmt_evidence(t.get('evidence', {}))}</article>"
            for t in group
        )
        note = (
            "<p class='muted small'>Contextual indicators describe the subject's "
            "counterparties rather than its own conduct. They corroborate; they cannot "
            "establish anything on their own.</p>"
            if heading == "Contextual indicators"
            else ""
        )
        typ_blocks.append(f"<h3>{heading}</h3>{note}<div class='typs'>{cards}</div>")

    if not typologies:
        typ_blocks.append(
            "<p>No structural laundering typology matched this entity. Any risk score "
            "shown rests on statistical features alone and should be treated as a "
            "low-confidence lead.</p>"
        )

    nb_rows = "".join(
        f"<tr><td class='mono'>{_esc(n['id'])}</td>"
        f"<td>{_esc(n.get('direction', ''))}</td>"
        f"<td class='num'>{n.get('value_btc', 0):,.6f}</td>"
        f"<td class='num'>{n.get('n_txs', 0):,}</td>"
        f"<td class='num'>{n.get('risk', 0):.2f}</td></tr>"
        for n in neighbours
    )

    feat_rows = "".join(
        f"<tr><td>{_esc(k.replace('_', ' '))}</td><td class='num'>{v:,.4f}</td></tr>"
        for k, v in sorted(feats.items())
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Case pack {_esc(case_ref)}</title>
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 11pt/1.5 "Segoe UI", system-ui, sans-serif; color: #14181d;
         background: #fff; margin: 0; padding: 24px; max-width: 900px; }}
  h1 {{ font-size: 20pt; margin: 0 0 4px; letter-spacing: -.01em; }}
  h2 {{ font-size: 13pt; margin: 26px 0 8px; padding-bottom: 5px;
        border-bottom: 1.5px solid #14181d; page-break-after: avoid; }}
  h3 {{ font-size: 11.5pt; margin: 18px 0 7px; page-break-after: avoid; }}
  h4 {{ font-size: 11pt; margin: 0; }}
  p {{ margin: 0 0 9px; }}
  .muted {{ color: #5d6875; }} .small {{ font-size: 9.5pt; }}
  .mono {{ font-family: Consolas, monospace; font-size: 9.5pt; }}
  header.mast {{ border-bottom: 2.5px solid #14181d; padding-bottom: 12px;
                 margin-bottom: 6px; }}
  .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 8px 20px; font-size: 9.5pt; margin-top: 10px; }}
  .meta div span {{ display: block; color: #5d6875; font-size: 8.5pt;
                    text-transform: uppercase; letter-spacing: .06em; }}
  .risk {{ display: flex; align-items: baseline; gap: 14px; margin: 14px 0 6px; }}
  .risk .v {{ font-size: 30pt; font-weight: 700; line-height: 1; }}
  .risk .band {{ font-size: 10pt; text-transform: uppercase; letter-spacing: .08em;
                 padding: 3px 9px; border-radius: 3px; font-weight: 600; }}
  .b-high {{ background: #f6e1df; color: #97302b; }}
  .b-medium {{ background: #f7ecd9; color: #8a5d13; }}
  .b-low {{ background: #e4ebe9; color: #2f6157; }}
  .caveat {{ border-left: 3px solid #97302b; background: #fbf4f3; padding: 11px 14px;
             font-size: 9.5pt; margin: 12px 0; }}
  .narrative {{ background: #f5f7f9; border: 1px solid #d9e0e6; padding: 14px 16px;
                margin: 12px 0; font-size: 11pt; }}
  .typs {{ display: flex; flex-direction: column; gap: 10px; }}
  .typ {{ border: 1px solid #d9e0e6; border-left-width: 3px; padding: 12px 14px;
          page-break-inside: avoid; }}
  .typ.sev-high {{ border-left-color: #97302b; }}
  .typ.sev-medium {{ border-left-color: #8a5d13; }}
  .typ.sev-low {{ border-left-color: #7d8894; }}
  .typ header {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 5px;
                 flex-wrap: wrap; }}
  .code {{ font-family: Consolas, monospace; font-size: 8.5pt; background: #eef2f5;
           padding: 2px 6px; border-radius: 2px; }}
  .sev {{ font-size: 8.5pt; text-transform: uppercase; letter-spacing: .07em;
          color: #5d6875; margin-left: auto; }}
  .ev {{ display: grid; grid-template-columns: max-content 1fr; gap: 3px 14px;
         font-size: 9.5pt; margin-top: 8px; padding-top: 8px;
         border-top: 1px solid #eef2f5; }}
  .ev-k {{ color: #5d6875; }} .ev-v {{ font-family: Consolas, monospace; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 9.5pt; margin: 8px 0; }}
  th {{ text-align: left; font-size: 8.5pt; text-transform: uppercase;
        letter-spacing: .06em; color: #5d6875; border-bottom: 1.5px solid #14181d;
        padding: 0 10px 5px 0; }}
  td {{ padding: 5px 10px 5px 0; border-bottom: 1px solid #eef2f5; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums;
          font-family: Consolas, monospace; }}
  .up {{ color: #97302b; }} .down {{ color: #2f6157; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
  footer {{ margin-top: 30px; padding-top: 12px; border-top: 2px solid #14181d;
            font-size: 9pt; color: #5d6875; }}
  @media print {{ body {{ padding: 0; }} .noprint {{ display: none; }} }}
  .noprint {{ background: #eef2f5; border: 1px dashed #b9c4ce; padding: 9px 13px;
              font-size: 9.5pt; margin-bottom: 18px; border-radius: 3px; }}
</style></head><body>

<div class="noprint">Press <strong>Ctrl+P</strong> (or Cmd+P) and choose "Save as PDF" to file this case pack.</div>

<header class="mast">
  <h1>Blockchain intelligence case pack</h1>
  <p class="muted">Subject entity <strong class="mono">{_esc(entity_id)}</strong></p>
  <div class="meta">
    <div><span>Case reference</span>{_esc(case_ref)}</div>
    <div><span>Generated</span>{_esc(generated)}</div>
    <div><span>Model version</span>{_esc(MODEL_VERSION)}</div>
    <div><span>Attribution method</span>{_esc(method)}</div>
    <div><span>Dataset label</span>{_esc(label)}</div>
    <div><span>Addresses in entity</span>{feats.get('n_addresses', 0):,.0f}</div>
  </div>
</header>

<h2>Assessment</h2>
<div class="risk">
  <span class="v">{risk:.2f}</span>
  <span class="band b-{_esc(_band(risk))}">{_esc(_band(risk))} priority</span>
</div>
<div class="caveat">{_esc(CAVEAT)}</div>
<div class="narrative">{_esc(narrative) or "No narrative available."}</div>

<h2>Indicators</h2>
{''.join(typ_blocks)}

<h2>Model reasoning</h2>
<p class="small muted">Additive feature attributions ({_esc(method)}). Positive values
pushed the score toward "illicit"; negative values pushed against it. Both are shown.</p>
<div class="cols">
  <div>
    <h3>Supporting the flag</h3>
    <table><thead><tr><th>Observation</th><th class="num">Value</th><th class="num">Effect</th></tr></thead>
    <tbody>{attr_rows(pushing_up) or "<tr><td colspan='3' class='muted'>None</td></tr>"}</tbody></table>
  </div>
  <div>
    <h3>Arguing against the flag</h3>
    <table><thead><tr><th>Observation</th><th class="num">Value</th><th class="num">Effect</th></tr></thead>
    <tbody>{attr_rows(pushing_down) or "<tr><td colspan='3' class='muted'>None</td></tr>"}</tbody></table>
  </div>
</div>

<h2>Counterparties</h2>
<p class="small muted">Showing {len(neighbours)} direct counterparties by value.</p>
<table><thead><tr><th>Entity</th><th>Direction</th><th class="num">BTC</th>
<th class="num">Transfers</th><th class="num">Risk</th></tr></thead>
<tbody>{nb_rows or "<tr><td colspan='5' class='muted'>No recorded counterparties</td></tr>"}</tbody></table>

<h2>Observed features</h2>
<table><thead><tr><th>Feature</th><th class="num">Value</th></tr></thead>
<tbody>{feat_rows}</tbody></table>

<h2>Reliability and limitations</h2>
{_calibration_note()}
<p><strong>Entity resolution.</strong> Addresses are grouped by the common-input-ownership
heuristic (Meiklejohn et al., 2013). This establishes that addresses were spent together,
therefore shared a key holder. It does <em>not</em> establish who that holder is. Naming an
entity requires off-chain intelligence such as exchange KYC records.</p>
<p><strong>Label provenance.</strong> Ground-truth labels derive from third-party datasets
assembled heuristically, not from criminal convictions.</p>
<p><strong>Temporal validity.</strong> The model was trained on an earlier period than it is
applied to. Detection quality is known to degrade when offender behaviour changes.</p>

<footer>
  Generated by ChainTrace {_esc(MODEL_VERSION)} · {_esc(generated)} · Case {_esc(case_ref)}<br>
  This document is an investigative aid. The authoritative record is the blockchain itself.
</footer>

</body></html>"""


def _band(risk: float) -> str:
    if risk >= 0.7:
        return "high"
    if risk >= 0.4:
        return "medium"
    return "low"
