# Datasets

Nothing in `data/` is committed. Run `python run.py` (synthetic, no download) to populate it.

## Provenance and licences — read before the submission slide

| Dataset | Scope | Licence | Cite on the slide |
|---|---|---|---|
| **Elliptic** | ~204k Bitcoin *transactions*, 166 anonymised features, 49 timesteps | **CC BY-NC-SA 4.0 — non-commercial** | Weber et al., KDD workshop 2019 |
| **Elliptic++** | ~822k *wallet addresses* with named features, plus address↔tx maps | Per the repo's `LICENSE` — check it | Elmougy & Liu, KDD 2023 |
| **BABD-13** | ~544k labelled addresses, 13 behaviour classes | Per the publication | Xiang et al., IEEE TIFS 2024 |

Two things that will come up in judging:

1. **Elliptic is non-commercial.** That is fine for a hackathon and for a public GitHub
   repository, but it must be stated. Put the licence on the data slide.
2. **Elliptic is transaction-level, not wallet-level.** Its nodes are transactions, and the
   feature columns are anonymised. Wallet/actor attribution comes from Elliptic++, and the
   grouping of addresses into entities comes from our own co-spend clustering. If someone asks
   "how do you know which wallet belongs to which gang", that is the answer — say it in that
   order.

## Layout after fetching

```
data/
  raw/
    elliptic/            elliptic_txs_features.csv, _classes.csv, _edgelist.csv
    elliptic_plus/       actors + address-tx mappings
    babd/                BABD-13.csv               (optional)
  processed/
    entities.parquet     one row per clustered entity  <- the frozen contract
    edges.parquet        entity-to-entity value flows  <- the frozen contract
    address_map.parquet  address -> entity_id
    SCHEMA.md            authoritative column definitions
```

## Verify your numbers

`scripts/fetch_data.py` prints actual row counts, file hashes, and label distributions after
download. **Quote those printed numbers, not the ones in this file or in a paper abstract.**
Dataset releases get revised; a judge who checks and finds your count is wrong will discount
everything else you said.

## Class balance (why accuracy is a trap)

Elliptic is roughly 2% illicit, 21% licit, 77% unlabelled. A model that predicts "licit" for
everything scores about 98% accuracy and is worth nothing. Report illicit-class F1, AUC-PR, and
precision at top-k instead — see `docs/RESEARCH.md` section 5.

The 77% unlabelled portion is an opportunity rather than a defect: it is what motivates the
semi-supervised and active-learning work in role 4's scope.
