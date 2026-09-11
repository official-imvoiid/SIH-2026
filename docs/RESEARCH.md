# Research Basis

Everything below is public, citable, and downloadable. No paywalled data, no internet needed at demo time.

---

## 1. Datasets

### Elliptic Data Set (primary, transaction-level)
- **What:** 203,769 Bitcoin transaction nodes, 234,355 directed payment-flow edges, 166 features
  per node, 49 discrete time steps (~2 weeks apart).
- **Labels:** 4,545 illicit (~2%), 42,019 licit (~21%), remaining ~77% unknown.
- **Features:** 94 "local" (tx fee, output volume, in/out degree...) + 72 "aggregated"
  (neighbour statistics). **Column names are anonymised** — this is the main limitation.
- **Where:** Kaggle `ellipticco/elliptic-data-set` (~400 MB CSV).
- **Licence:** CC BY-NC-SA 4.0 — non-commercial. Fine for a hackathon; state it on the slide.

### Elliptic++ (the upgrade that makes wallet-level attribution possible)
- **What:** extends Elliptic with an **actors/wallet-address dataset** — ~822k wallet addresses
  with named, interpretable features, plus address↔transaction mappings.
- **Why we need it:** Elliptic alone has no wallet identity, so "which wallet belongs to which
  gang" is unanswerable from it. Elliptic++ is what closes that gap.
- **Where:** `github.com/git-disl/EllipticPlusPlus`
- **Paper:** Elmougy, Y. & Liu, L., *Demystifying Fraudulent Transactions and Illicit Nodes in the
  Bitcoin Network for Financial Forensics*, KDD 2023.

### BABD-13 (optional third dataset, address behaviour)
- **What:** ~544k labelled Bitcoin addresses across 13 behaviour classes (exchange, mining pool,
  gambling, ransomware, darknet market, tumbler, ...), 148 features.
- **Why it helps:** gives you multi-class actor *typing*, not just binary illicit/licit — so the
  UI can say "ransomware wallet" instead of "bad wallet".
- **Paper:** Xiang et al., *BABD: A Bitcoin Address Behavior Dataset for Pattern Analysis*,
  IEEE Transactions on Information Forensics and Security, 2024.

> **Verify counts yourself.** `ml/ingest/build.py --real` prints actual row counts after download.
> Do not quote numbers from this file on a slide without checking them against the printout.

---

## 2. Papers — and the numbers we have to beat

### The baseline paper (read this one first, all six of you)
**Weber, M. et al. (2019).** *Anti-Money Laundering in Bitcoin: Experimenting with Graph
Convolutional Networks for Financial Forensics.* KDD '19 Workshop on Anomaly Detection in Finance.
arXiv:1908.02591

Findings that shape our whole design:
- **Random Forest beats GCN** on this data — RF reaches ~0.79 illicit-class F1, plain GCN ~0.65.
  *Implication: ship the boosted-tree baseline first. A GNN is the upgrade, not the foundation.*
- Concatenating GCN node embeddings onto the raw features ("AF+NE") gives the best result.
  *Implication: our ensemble design is literally the paper's recommendation.*
- **Performance collapses after time step 43**, when a real dark market was shut down and the
  illicit population changed. *Implication: this is our headline demo moment — see §4.*

### Address clustering (how we get from wallets to "gangs")
**Meiklejohn, S. et al. (2013).** *A Fistful of Bitcoins: Characterizing Payments Among Men With No
Names.* IMC '13.
The founding paper for the **multi-input / co-spend heuristic**: if two addresses appear as inputs
to the same transaction, they are almost certainly controlled by the same entity. Plus the
change-address heuristic. This is the core of `ml/features/clustering.py`.

**Harrigan, M. & Fretter, C. (2016).** *The Unreasonable Effectiveness of Address Clustering.*
Explains *why* co-spend clustering works so well, and — importantly for our limitations slide —
the conditions under which it produces false merges (address reuse, CoinJoin).

### Graph learning
**Hamilton, W. et al. (2017).** *Inductive Representation Learning on Large Graphs* (GraphSAGE),
NeurIPS. Inductive = can score entities never seen in training. Required, since a real
investigation involves new wallets.

**Pareja, A. et al. (2020).** *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs.*
AAAI 2020. Evaluated on Elliptic specifically; handles the temporal drift problem directly.

### Explainability
**Lundberg, S. & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions.*
NeurIPS. SHAP — our per-feature attributions for the tree model.

**Ying, R. et al. (2019).** *GNNExplainer: Generating Explanations for Graph Neural Networks.*
NeurIPS. Identifies the *subgraph* responsible for a prediction — this is what draws the red
highlight on the graph canvas.

### Label scarcity (77% of Elliptic is unlabelled — this is our friend, not our problem)
**Lorenz, J. et al. (2020).** *Machine Learning Methods to Detect Money Laundering in the Bitcoin
Blockchain in the Presence of Label Scarcity.* ACM ICAIF '20. Shows active learning recovers most
of the supervised performance using a small fraction of labels. Gives us a defensible
"human-in-the-loop / analyst feedback" feature instead of a gimmick one.

### Regulatory grounding (this is the differentiator; almost no hackathon team does it)
**FATF (2020).** *Virtual Assets Red Flag Indicators of Money Laundering and Terrorist Financing.*
Financial Action Task Force.
Our typology rule engine implements these published indicators directly. It means every flag maps
to a **named, official red-flag indicator** rather than an opaque score — which is what a real
FIU-IND or I4C analyst would need to open a case.

---

## 3. What is genuinely novel here

Be honest with yourselves: "train a classifier on Elliptic" is a solved Kaggle exercise. These
three things are what make it a project rather than a tutorial:

1. **Entity-resolution before classification.** Most published work classifies *transactions*.
   We cluster addresses into actors first and classify *actors*. Different, harder, more useful.
2. **Dual explanation — statistical + regulatory.** SHAP tells you what the model used; the FATF
   rule engine tells you what a human investigator would call it. Presenting both together is
   rare in the literature and directly addresses model-evidence admissibility.
3. **Reporting concept drift instead of hiding it.** We show the timestep-43 collapse on screen
   and explain it. Judges reward the team that knows where its model fails.

## 4. The demo moment (design the pitch around this)

1. Train on time steps 1–34. Show strong metrics. Everyone nods.
2. Slide the time scrubber to step 43+. **Watch recall fall off a cliff on screen.**
3. Explain: a dark market was shut down; the criminal population changed; the model had never
   seen the new pattern. This is *the* known failure mode of blockchain AML models.
4. Show the mitigation: the FATF typology rules still fire, because structural laundering
   patterns survive when statistical ones don't.

That sequence demonstrates you understand the problem domain, not just `model.fit()`. It is worth
more than three extra points of F1.

## 5. Metrics we report (and why not accuracy)

The classes are ~2% illicit. Accuracy is meaningless — predicting "all licit" scores 98%.

| Metric | Why |
|---|---|
| **Illicit-class F1** | The number Weber et al. report; makes us directly comparable |
| **Precision @ top-k** | An analyst reviews 50 leads a day, not 200,000. This is the real-world metric |
| **Recall on illicit** | Missing a gang is the expensive error |
| **AUC-PR** | Correct for heavy imbalance (AUC-ROC flatters imbalanced models) |
| **F1 before vs. after timestep 43** | Our honesty metric |
