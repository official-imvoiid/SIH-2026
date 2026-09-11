# Assignments — 3 coders, 3 on research/PPT/video

**The baseline is already built and working.** `python run.py` gives you a trained model,
a scored entity graph, and an investigator UI in about a minute, with no dataset download.

That is deliberately good news: nobody starts from a blank file, and there is a working
thing to demo from day one. Your job is to **replace and improve** parts of a system that
already runs, which is a far better position than assembling one under deadline.

Verify it works before you change anything:

```
git clone <repo-url> && cd SIH-2026
pip install -r requirements.txt
python run.py
```

Then open <http://localhost:8000>. You should see a triage queue, a money-flow graph, and
a case dossier. `python -m pytest backend/tests/ -q` should print **50 passed**.

---

## What is already done

| Area | State | Where |
|---|---|---|
| Synthetic Bitcoin world (offline, 17k entities) | done | `ml/data/synth.py` |
| Co-spend clustering + CoinJoin guard | done — 0 false merges | `ml/features/clustering.py` |
| Entity features (27, leak-safe) | done | `ml/features/graph.py` |
| Risk model + temporal split + drift report | done — test F1 0.846 (synthetic) | `ml/train/baseline.py` |
| Feature attribution | done | `ml/explain/attribution.py` |
| 7 FATF typology rules | done — but only **1 of 7** validated on real Bitcoin | `ml/typology/rules.py` |
| REST API | done — 6 endpoints | `backend/app/main.py` |
| Investigator UI | done — zero dependencies | `frontend/public/index.html` |
| Tests | 50 passing | `backend/tests/` |

## What is NOT done — ranked by value

1. **Real Elliptic++ data.** One function: `ml/ingest/build.py::_load_real`. Everything
   downstream works unchanged. Biggest single win available.
2. **Typology threshold tuning on real labels** — constants at the top of `rules.py`.
3. **PDF case-pack export** — the artefact an investigator actually keeps.
4. **OFAC / Ransomwhere tag cross-reference** — turns an anonymous cluster into a named lead.
5. **GNN benchmark** — and an honest report of whether it beats the forest. It probably won't.

---

# CODER 1 — Real data

```
Your job: replace the synthetic data with the real Elliptic++ dataset.

Right now the whole tool runs on a synthetic Bitcoin world I generate in code. It works,
but every number has an asterisk. Your job removes that asterisk. This is the single most
valuable open task.

Step 1. Get the data.
  Elliptic++ lives at github.com/git-disl/EllipticPlusPlus
  You want the ACTORS (wallet address) dataset -- roughly 822k addresses with named
  features. Put the CSVs in data/raw/elliptic_plus/.

  Why the actors dataset and not classic Elliptic: classic Elliptic has TRANSACTIONS as
  its nodes, not wallets, and its feature columns are anonymised. You cannot answer "which
  wallet belongs to which gang" from it. Elliptic++ is what makes that possible.

Step 2. Open ml/ingest/build.py and implement _load_real().
  It currently raises NotImplementedError with instructions inside. It must return the
  same tuple shape as _load_synthetic() directly above it. Match that shape and every
  other stage -- clustering, features, model, API, UI -- keeps working untouched.

Step 3. Verify.
  python -m ml.ingest.build --real
  python -m ml.train.baseline
  python -m pytest backend/tests/ -q

Target: beat 0.79 illicit-class F1 **on the real Elliptic dataset**. That is what Weber
et al. (2019) got with a Random Forest. Our synthetic baseline currently reads
0.846, but that is a generated world we tuned ourselves and is **not
comparable** — the real comparison only exists once someone downloads the real data
(`python scripts/fetch_data.py`).

Watch out for: Elliptic is ~2% illicit and ~77% unlabelled. Do not drop the unlabelled
rows -- they are still scored, they just cannot be trained on.
```

**AI prompt for Coder 1** ⬇️

```
I'm working on a Bitcoin forensics tool for a hackathon. It currently runs on synthetic
data and I need to swap in the real Elliptic++ dataset (github.com/git-disl/EllipticPlusPlus,
from Elmougy & Liu, KDD 2023).

I need to write a Python loader that returns exactly this tuple:
  (transactions, truth_owner, truth_label, provenance_string)
where:
  transactions  = list of (txid, input_addresses, output_values, ts, outputs)
                  with outputs being a list of (address, value) pairs
  truth_owner   = dict mapping address -> ground-truth owner id (may be empty if unknown)
  truth_label   = dict mapping owner id -> "illicit" | "licit" | "unknown"

Elliptic++ ships several CSVs: an actors/wallet dataset (~822k addresses with named
features), a transactions dataset, and bipartite AddrTx / TxAddr edge files linking
addresses to transactions.

Help me:
1. Work out which CSVs I need and what their columns are
2. Write the loader, handling the ~77% unlabelled rows correctly (keep them, mark unknown)
3. Handle memory sensibly -- suggest dtypes or chunking if the files are large

Important context: the dataset is about 2% illicit, so class imbalance is severe. Explain
your choices as you go; I have to defend this to technically literate judges. I'm a
student, so explain rather than just dumping code.
```

---

# CODER 2 — Evidence and reporting

```
Your job: turn a flagged wallet into something a police officer can file.

Right now the tool detects laundering patterns and explains them on screen. What it cannot
do is produce a document. That is the gap between a demo and a tool.

Task 1 -- PDF case pack. New file: backend/app/services/report.py
  Input: an entity ID. Output: a PDF containing
    - the risk score and what it means (and what it does NOT mean)
    - every red-flag indicator with its evidence
    - the money-flow subgraph as an image
    - the feature attribution table
    - a provenance footer: model version, dataset version, timestamp
  Wire it to a new endpoint GET /api/report/{entity_id}.
  Use jinja2 + weasyprint, or reportlab. Whatever installs cleanly on your machine.

Task 2 -- OFAC sanctions cross-check.
  Grab the sanctioned-address list from
  github.com/0xB10C/ofac-sanctioned-digital-currency-addresses (public, US government data).
  Cross-reference it against our clustered entities. A cluster containing even one
  sanctioned address is a named lead rather than an anonymous one -- that is a strong demo
  moment.

Task 3 -- make the typology rules work on REAL Bitcoin.
  This is the most valuable job on the team, because we already know it is broken and
  we know why.

  Run: python -m ml.typology.calibrate --size 500
  It tests each rule against wallets independently confirmed to have received ransom
  payments, versus ordinary wallets, holding out whole ransomware families.

  The result today: only 1 of 7 rules separates real criminal money. rapid_passthrough
  catches 80% of ransom wallets but also flags 50% of ordinary ones. The other six do
  nothing. On our own synthetic data the same rules score 98.5% precision -- that number
  is circular and must never be quoted.

  The diagnosis is that exchanges are structurally identical to launderers: they receive
  from thousands and pay out to thousands, exactly like layering. is_service_like() in
  rules.py is a crude structural stand-in. A real exchange tag list would likely fix it.

  Success is not "get the number up". Success is a defensible measurement either way.
  A rule that provably does not work is a finding worth reporting.

Rule for all three: every number in the PDF must carry the caveat that it indicates
investigative priority, not proof of guilt. There is a test enforcing this
(test_narrative_carries_the_not_proof_caveat). Do not weaken it.
```

**AI prompt for Coder 2** ⬇️

```
I'm building the reporting layer for a Bitcoin forensics tool used by investigators. It
flags suspicious wallet clusters, explains why using FATF money-laundering red-flag
indicators, and shows feature attributions from a Random Forest.

I need to generate a PDF "case pack" for one flagged entity, containing:
- the risk score, with an explicit statement of what it does and does not mean
- each red-flag indicator with its supporting evidence
- a rendered money-flow network diagram
- a feature attribution table
- provenance: model version, dataset version, generation timestamp

Help me:
1. Choose between weasyprint, reportlab and fpdf2 for this, and explain the trade-off
2. Write the generator, given a Python dict with keys: id, risk, label, typologies (list
   of {code, name, severity, description, evidence}), attributions (list of
   {feature, readable, value, contribution}), narrative, neighbors
3. Render the network diagram -- I have the graph in networkx and matplotlib available

Critical constraint: this document could end up in a case file. It must never read as an
accusation. Every risk figure needs its qualification adjacent to it, not buried in a
footer. Help me word that so it is clear without being so hedged it is useless.

I'm a student. Explain the PDF library choices, I have not used any of them.
```

---

# CODER 3 — The interface

```
Your job: everything the judges actually look at.

There is a working UI at frontend/public/index.html -- one file, no dependencies, no build
step. It renders the graph on a canvas with a force-directed layout I wrote by hand,
because the tool has to run air-gapped and I did not want a CDN on the critical path.

Run it: python run.py, then open http://localhost:8000

Your call to make: improve that file, or rebuild in React + Vite + Cytoscape.js.
  - Improving it: faster, zero risk, and it already works
  - Rebuilding: nicer to develop, but you must vendor Cytoscape locally (npm install then
    copy the dist file into the repo). If it loads from a CDN, it breaks on demo day.
  Either is defensible. Decide in week 1, not week 4.

Task 1 -- the time scrubber is the whole demo. Right now it dims edges after the selected
  step. Make it genuinely good: play/pause animation, and show the risk scores changing
  as time advances. This is the moment the pitch is built around.

Task 2 -- case builder. Let an analyst pin entities to a case, annotate them, and export.
  Wire the export button to Coder 2's /api/report endpoint.

Task 3 -- a "compare" view: model performance before vs after time step 43, side by side.
  The data is already at GET /api/metrics under the "drift" key.

One hard rule: when the API returns truncated: true, the UI MUST say "showing top 300 of
N". Showing a partial graph as if it were complete is how you lose on the ethics question.
There is a test enforcing the API side; do not break the UI side.
```

**AI prompt for Coder 3** ⬇️

```
I'm improving the frontend of a Bitcoin forensics tool -- an investigator UI that shows
wallet clusters as a network graph, colours them by risk, and explains why each was
flagged.

The existing UI is a single HTML file with zero dependencies: hand-written canvas
rendering with a Fruchterman-Reingold force layout. It has to keep running air-gapped, so
anything I add must be vendored locally, never loaded from a CDN.

The backend is FastAPI at localhost:8000:
  GET /api/entity/{id}      -> { id, risk, band, label, n_addresses, total_in_btc,
                                 total_out_btc, features, attributions[], typologies[],
                                 neighbors[], narrative, clustering_note }
  GET /api/subgraph/{id}?hops=2 -> { nodes[], edges[], truncated, total_available, ranking }
  GET /api/timeline         -> [{ ts, n_txs, volume_btc, n_illicit }] for steps 1-49
  GET /api/metrics          -> model metrics including a "drift" breakdown
  GET /api/top, /api/search

I want to build an animated time scrubber: play/pause across 49 time steps, with the graph
and risk colours updating as time advances. It is the centrepiece of our demo.

Help me:
1. Design the animation loop (requestAnimationFrame, interpolation between steps)
2. Handle the layout sensibly -- nodes should not jump around as time changes
3. Keep it smooth at 300 nodes on canvas

Design direction: a serious law-enforcement tool, not a crypto startup. Restrained
colours, dense information, dark theme, readable at a glance.

Explain the canvas animation parts carefully -- I have not done this before.
```

---

# RESEARCH · PPT · VIDEO

The three non-coding briefs are unchanged — see the messages already sent. Two additions
now that the code exists:

**Research person** — you now have real numbers to cite, printed by `python run.py`:

| Claim | Number | Source |
|---|---|---|
| Clustering makes no false merges | 0 of 17,283 clusters | `data/processed/build_report.json` |
| Model illicit-F1 (held-out, synthetic) | 0.846 | `ml/artifacts/metrics.json` |
| Drift collapse at step 43 | 1.000 → 0.733 | same file, `drift` key |
| Typology rules validated on REAL data | 1 of 7 | `ml/artifacts/calibrated_thresholds.json` |

Say clearly on the slide that these are from **synthetic data** until Coder 1 lands the
real dataset. Getting caught overstating this would cost more than the numbers gain.

**PPT person** — your strongest slide already exists. Screenshot the case dossier panel
showing a flagged entity: primary indicator, additional indicators, corroborating context,
and the caveat. That single image makes the argument that we built a forensics tool rather
than a classifier.

**Video person** — the drift demo works today. Run `python run.py`, drag the time scrubber
past step 43, and narrate the collapse. You do not need to wait for anyone.
