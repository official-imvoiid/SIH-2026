# ChainTrace

**Entity-level illicit actor detection on the Bitcoin transaction graph, with evidence an investigator can put in a case file.**

Built for SIH problem statement **SIH26146 — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**.

```bash
git clone <this repo> && cd SIH-2026
pip install -r requirements.txt
python run.py
```

That is the whole setup. It builds the dataset, trains the model, and serves the
investigator UI at **http://localhost:8000** in about a minute. **No download, no API key,
no network call at any point** — it runs air-gapped.

---

## The problem

Ransomware crews are paid in Bitcoin. Every payment is public, so police *can* see the
money move. What they cannot cheaply do is answer the three questions a case actually
turns on:

1. Which **addresses belong to the same real-world actor**? One crew uses hundreds.
2. Which of those actors are **likely criminal**, and how confident are we?
3. **Why** — in language that survives a courtroom, not a softmax score?

## What it does

```
transactions ──▶ address clustering ──▶ entity graph ──▶ risk model ──▶ evidence
                 (co-spend heuristic)   (actors, not      (forest +     (attribution +
                                         tx hashes)        typologies)   FATF red flags)
```

Only **stage 3** involves machine learning. Stages 1, 2 and 4 are deterministic graph
algorithms, so most of the product keeps working even if the model is broken or drifting.

## Real Bitcoin data

The bundled demo dataset is synthetic, which invites a fair objection: *you planted the
patterns, then found them.* So the repo also runs against the real blockchain.

```bash
python -m ml.ingest.real                      # real ransomware + sanctions lists
python -m ml.experiments.real_vs_control      # the experiment below
```

| Source | What it is |
|---|---|
| [Ransomwhere](https://ransomwhe.re) | **8,248 real Bitcoin addresses** that received real ransom payments, labelled by the family that collected them — Locky, Conti, Ryuk, NetWalker, SamSam, and 100+ more |
| [OFAC SDN](https://github.com/0xB10C/ofac-sanctioned-digital-currency-addresses) | 532 Bitcoin addresses sanctioned by the US Treasury |
| [mempool.space](https://mempool.space) | The live Bitcoin blockchain — real transactions, inputs, outputs, timestamps |

**The experiment.** Two groups of real Bitcoin addresses, treated identically: one seeded
from real ransomware wallets, one from ordinary wallets in recent blocks. Same crawl, same
clustering, same rules. The only difference is seed selection — so any difference in how
often the rules fire is a property of criminal money, not of our code.

Fetching happens **once** and caches to `data/raw/`; every later run is fully offline.

### The result: one rule of seven transfers to real data

1,000 real addresses crawled (500 per group), ~15,000 real Bitcoin transactions.

**Two measurements, and they disagree — both are reported because the disagreement is the
finding.**

#### Measurement 1 — whole crawled neighbourhoods

Measured over every entity we retrieved history for:

| | Ransomware | Control | Ratio |
|---|---|---|---|
| Observed entities | 342 | 74 | |
| **Has ≥1 structural typology** | **67.5%** | **63.5%** | 1.1x |
| Has ≥2 structural typologies | 9.9% | 13.5% | 0.7x |
| — rapid pass-through | 49.7% | 37.8% | 1.3x |
| — fan-out | 9.4% | 29.7% | **0.3x** |
| — dormant-then-burst | 7.9% | 0.0% | — |

Barely any separation. But this measurement has a flaw, and finding it mattered more than
the number did.

#### The flaw: a crawled neighbourhood is not a criminal

Measurement 1 counts *every entity in the ransomware crawl* as ransomware. But crawling one
hop out from a ransom wallet also collects **the victims who paid in and the exchanges the
money was cashed out to**. Most of that neighbourhood is innocent, so the "criminal" class
was heavily contaminated and no rule could have separated it.

#### Measurement 2 — confirmed ransom wallets only

Positives relabelled as **only entities containing a Ransomwhere-confirmed ransom address**.
Thresholds fitted on some ransomware families, scored on **held-out families the
calibration never saw**, using Youden's J (true-positive rate minus false-positive rate):

| Rule | Catches ransom | Flags ordinary | J |
|---|---|---|---|
| **`rapid_passthrough`** | **80%** | 50% | **+0.300** |
| fan_in | 20% | 25% | −0.050 |
| fan_out | 0% | 6% | −0.062 |
| structuring | 0% | 6% | −0.062 |
| peel_chain | 0% | 0% | 0.000 |
| dormant_burst | 7% | 6% | +0.004 |

**One rule of seven separates real criminal money. Six do not.**

The result is stable as the sample grows, which is the main reason to trust it: at 7
held-out wallets it was 71% / 43% (J = +0.286); at 15 it is 80% / 50% (J = +0.300).

**What J = +0.30 actually means.** Pass-through is a **triage filter, not a detector**. A
50% false-positive rate is far too high to accuse anyone of anything. What it does is
roughly halve the search space while retaining 80% of the targets — genuinely useful to an
analyst working a queue, and worthless as standalone evidence. We say it that way in the
UI and in the case pack.

**Threshold tuning did not help.** For every rule, the swept thresholds failed to beat the
original defaults on held-out families. The code keeps the defaults and records this.

#### Why the other six fail

1. **Thresholds were tuned on synthetic data.** Real Bitcoin is far busier than the
   generated world, so `FAN_MIN_COUNTERPARTIES = 10` is meaningless where ordinary wallets
   touch hundreds.
2. **Exchanges are structurally identical to launderers.** They legitimately receive from
   thousands and pay out to thousands. Fan-out fires **3x more** on ordinary money.
   `is_service_like()` in `ml/typology/rules.py` is a structural stand-in for the exchange
   tag list that commercial tools use; a real tag list would do this properly.

**An incidental real finding:** 500 ransomware addresses formed **342** entities, while 500
ordinary addresses collapsed into **74**. Ordinary users reuse addresses about 7x more
heavily than criminals do. A usable signal we were not looking for.

**Five OFAC-sanctioned addresses appeared in the crawl**, found by cross-referencing the US
Treasury list against clustered entities.

### What this means for the project

Clustering, the pipeline, the explanation layer, the case pack and the UI all work on real
data. Of the detection rules, one is validated and six are not — and the interface says so
per flag, with three states: `real-data validated`, `not validated`, `untested`.

Presenting this honestly is stronger than hiding it. "We tested our own detector against
real criminal money, six of seven rules failed, we found the labelling flaw that hid the
seventh, and it survived on crews it had never seen" demonstrates more competence than an
unverified 98.5%.

## Measured results

Everything below is printed by the pipeline on every run — reproduce it with
`python run.py --rebuild`. Numbers are from the **bundled synthetic dataset**; see
[Datasets](#datasets) before quoting them anywhere.

### Address clustering — turning wallets into actors

| Metric | Value | Reading |
|---|---|---|
| Pair precision | **1.0000** | Never invents a link between unrelated parties |
| **False merges** | **0** | of 17,283 clusters — the CoinJoin guard holds |
| Pair recall | 0.3379 | Under-merges, **by design** — see below |
| Addresses → entities | 35,317 → 17,283 | |

Co-spend clustering deliberately trades recall for precision. Two addresses of the same
owner that never co-spent stay separate; that costs recall. A *false merge* would invent a
connection between innocent people, so precision is the number that must not move.

### Risk model — held-out test period

| Metric | Value |
|---|---|
| Illicit **F1** | **0.846** |
| Illicit precision | 0.971 |
| Illicit recall | 0.750 |
| Average precision | 0.845 |
| Precision @ top-50 | 1.000 |
| Confusion | tp 66 · fp 2 · fn 22 · tn 2,456 |

`RandomForestClassifier(400)` + Platt calibration, 23 features,
**trains in under a second**. Guilt-by-association features are excluded — measured to
cost ~8 F1 points, see the benchmark section below.

> **This is synthetic data and therefore NOT comparable to the literature.** Weber et al.
> (2019) report ~0.79 illicit-F1 for Random Forest on the *real* Elliptic dataset (~0.65
> for a GCN). Ours is a generated world tuned by us; a higher number here means nothing
> about the real problem. Do not claim we beat them.

**Accuracy is never reported.** At a 1.97% illicit base rate, "everything is clean" scores
98% and is worthless.

### Concept drift — the honest result

| Period | Illicit F1 |
|---|---|
| Before time step 43 | **1.000** |
| After time step 43 | **0.733** |

The illicit population changes behaviour at step 43 (mirroring the real dark-market
shutdown in Elliptic). Performance falls off a cliff. We put this on screen instead of
hiding it — knowing where your model fails is the point.

### Typology engine — no machine learning at all

Against a 6.98% base rate among labelled entities:

| Rule hits | Precision | Recall |
|---|---|---|
| ≥1 structural typology | 0.574 | 0.779 |
| ≥2 structural typologies | 0.985 | 0.376 |

### Does graph structure help? Mostly not — and association features actively hurt

`python -m ml.train.benchmark`, five models, identical temporal split, only the inputs vary:

| Arm | Illicit F1 |
|---|---|
| **behaviour only** | **0.843** |
| behaviour + label-free graph structure | 0.838 |
| everything, incl. guilt-by-association | 0.755 |
| learned spectral embeddings only | 0.271 |
| everything + embeddings | 0.512 |

**Dropping guilt-by-association features raises F1 by ~8 points.** They are fitted to the
training period's criminal population, which changes by test time. They are now excluded
from the model, which lifted the headline test F1 from 0.766 to **0.846**.

The project already put conduct above association in explanations for *ethical* reasons.
It turns out that choice is also the more accurate one — which is a far stronger claim than
either half alone.

*(These are spectral embeddings, not a GNN. A real GNN comparison needs the real Elliptic
dataset; benchmarking one on a generated world would measure the generator.)*

> **These numbers are synthetic and do NOT transfer.** On real Bitcoin only one of the
> seven rules separates criminal money from ordinary money — see
> [the real-data result](#the-result-one-rule-of-seven-transfers-to-real-data) above. They
> are kept here because the gap between the two is the point: it is what measuring against
> your own generator looks like. **Do not put 98.5% on a slide.**

## Why the explanation layer is the point

A model that says `0.93` is not evidence. The tool produces this instead:

> Entity **E-06641** is flagged with a model risk score of 1.00. Primary indicator —
> Rapid pass-through (VA-PASSTHROUGH): forwarded 100.0% of 62.2336 BTC received within 1
> time step of receipt, retaining effectively no balance. Wallets that hold no position
> are characteristic laundering intermediaries rather than end users. Additional conduct
> indicators: Peel chain (VA-PEELCHAIN). Corroborating context: Direct exposure to known
> illicit entities (VA-PROXIMITY). *Risk scores reflect heuristic dataset labels and
> structural pattern matching; they indicate investigative priority, not proof of criminal
> conduct.*

Three things are deliberate there, and each is enforced by a test:

- **Conduct outranks association.** Indicators about an entity's *own* behaviour always
  lead; guilt-by-association is demoted to "corroborating context" and can never be the
  headline. `test_corroborating_typologies_never_lead`
- **The caveat is not optional.** No risk score reaches a user without it.
  `test_narrative_carries_the_not_proof_caveat`
- **Exculpatory evidence survives.** Negative feature contributions are shown, not
  filtered out.

Each code (`VA-PASSTHROUGH`, `VA-PEELCHAIN`, …) maps to a published **FATF virtual-asset
red-flag indicator**, so a flag reads as an investigative finding rather than an opaque
score.

## Honest limitations

Read these before demoing. Several will be asked about.

- **The bundled data is synthetic.** It is Bitcoin-*shaped* — same class balance, real
  CoinJoins, planted laundering typologies, a genuine behavioural regime change — but it
  is not real Bitcoin. Any number from it must say so. The real-data path above
  (`ml/ingest/real.py`) is what removes the circularity; use it for anything you claim.
- **Only positive labels exist on real data.** There is no public list of confirmed-clean
  Bitcoin addresses, so a supervised model trained on real data would have no trustworthy
  negatives. That is precisely why the real experiment tests the **rule engine**, which
  needs no labels, rather than the classifier.
- **Address history is truncated.** `mempool.space` returns roughly the 50 most recent
  transactions per address, so very busy wallets are partially observed. The crawl report
  counts how many addresses this affected.
- **A cluster is not a gang.** It is a set of co-controlled addresses. Attaching a *name*
  needs off-chain intelligence (exchange KYC, sanctions lists) that no public dataset has.
- **Labels are heuristic vendor labels**, not convictions. We predict investigative
  priority, not guilt.
- **CoinJoin detection is a heuristic.** It scores 0 false merges here; it will not be
  perfect on real mainnet data.
- **Elliptic's own features are anonymised**, so explanations over them name columns
  without semantics. ChainTrace's 27 features are all self-derived and named, which is why
  the attribution panel is readable at all.

## Datasets

The tool ships with a synthetic generator so six people can start on day one without a
Kaggle account. To use real data, see [`data/README.md`](data/README.md) — Elliptic,
Elliptic++ (wallet-level, the one that matters), and BABD-13, with licences.

> **Elliptic is CC BY-NC-SA 4.0 — non-commercial.** Put that on the data slide.
>
> **Elliptic is transaction-level, not wallet-level.** Its nodes are transactions and only
> ~23% carry a label. Wallet-level attribution needs Elliptic++; grouping addresses into
> actors needs the clustering in this repo.

## Architecture

| Path | What lives there | Runs without data? |
|---|---|---|
| [`ml/data/synth.py`](ml/data/synth.py) | Synthetic Bitcoin world generator | yes |
| [`ml/ingest/real.py`](ml/ingest/real.py) | **Real** ransomware wallets + live blockchain | after first fetch |
| [`ml/experiments/real_vs_control.py`](ml/experiments/real_vs_control.py) | **The non-circular experiment** | after first fetch |
| [`ml/features/clustering.py`](ml/features/clustering.py) | Co-spend union-find + CoinJoin guard | yes |
| [`ml/features/graph.py`](ml/features/graph.py) | Entity features, leak-safe by construction | — |
| [`ml/typology/rules.py`](ml/typology/rules.py) | 7 FATF red-flag detectors | **yes** |
| [`ml/train/baseline.py`](ml/train/baseline.py) | Forest + temporal split + drift report | — |
| [`ml/explain/attribution.py`](ml/explain/attribution.py) | Exact tree-path attribution (SHAP if installed) | — |
| [`backend/app/`](backend/app/) | FastAPI, in-memory store, 300-node subgraph cap | — |
| [`frontend/public/index.html`](frontend/public/index.html) | Investigator UI — **zero dependencies** | yes |

The UI is hand-written canvas with a Fruchterman-Reingold layout and no graph library, so
there is nothing to fetch from a CDN and nothing to `npm install`. That is a deliberate
trade for air-gapped operation.

```bash
python -m pytest backend/tests/ -q      # 62 tests, ~5s, no dataset needed
```

## Documentation

| Doc | Contents |
|---|---|
| [`docs/EXPLAIN.md`](docs/EXPLAIN.md) | **The project explained from zero** — start here |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Papers and datasets, with the numbers to beat |
| [`docs/TEAM.md`](docs/TEAM.md) | Work split, sprint plan, frozen interface contracts |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module boundaries and design decisions |
| [`data/README.md`](data/README.md) | Dataset provenance and licences |

## Licence

MIT — see [`LICENSE`](LICENSE). Datasets carry their own licences; see `data/README.md`.
