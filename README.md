# ChainTrace

AI-powered monitoring and analysis of Bitcoin transaction traffic.

A desktop application that pulls real blocks from the public chain, finds behaviour that
looks like money laundering, and follows the money forward until it reaches somewhere it
can stop being anonymous.

Built with **Node** for the graph work, **Python** for the learning, and **Electron** for
the desktop shell.

There is no sample data in this build. Every number the interface shows came from a block
it fetched, or from a model trained on one.

---

## Running it

```bash
npm install
python -m pip install -r python/requirements.txt
npm start
```

The window opens on an empty state, because nothing has been ingested yet. Two commands
fill it:

```bash
node scripts/ingest.js --days 2
python python/pipeline.py
```

Or use the **Source** and **Model** screens inside the app, which run the same two stages
and stream their output into the window.

| Command | What it does |
|---|---|
| `npm start` | the desktop application |
| `npm run serve` | the local API and interface only, no Electron |
| `npm run ingest` | pull the configured window of blocks |
| `npm run analyse` | cluster, train, and score |
| `npm run trace -- <address>` | follow one address to its endpoint, from the terminal |

---

## What it does

**1. Pull whole blocks, not individual addresses.**

One `rawblock` request returns about 4,900 transactions and 5,400 addresses in a single
8 MB response. Fetching address by address instead would need tens of thousands of requests
for the same window, and gets rate-limited after about forty. Two days of Bitcoin is
roughly 288 requests this way.

Measured on a 2-day window: **1,399,931 transactions, 1,143,090 distinct addresses,
2,845,537 value-flow edges.** Blocks are reduced to the fields the analysis reads and
cached on disk, so a window is fetched once and re-analysed forever.

**2. Extract the behaviour laundering produces.**

| Pattern | What it looks for |
|---|---|
| High-velocity layering | money arrives, splits several ways, and leaves within minutes |
| Fan-in collection | many unrelated senders pay one address, which sweeps it onward |
| Peel chain | the bulk forwards to a fresh address, a little peels off, repeatedly |
| Pass-through | nothing is retained |

The peel detector requires a genuine multi-hop sequence. An earlier version tested only the
*shape* of a single transaction and flagged 904 addresses out of 7,553, because two outputs
where one takes 85% is simply what an ordinary payment with change looks like. Requiring
three consecutive hops took that to 5.

**3. Cluster behaviour with HDBSCAN.**

Not k-means: the number of behaviour types in a window is unknown, and most addresses
belong to no interesting group at all. HDBSCAN infers the count from density and marks
genuinely unusual points as noise rather than forcing them into the nearest blob. Here the
noise label is a finding, not a failure.

Features are log-scaled and standardised first, because holding time and transaction
frequency span four orders of magnitude, and on raw values the distance metric collapses
onto whichever has the largest units.

**4. Train a graph neural network.**

A 4-layer GraphSAGE network in PyTorch Geometric, so risk propagates four hops along the
actual money flow. GraphSAGE rather than plain GCN deliberately: GCN's symmetric averaging
blurs the line between a criminal and someone merely paid by one.

**5. Trace to an endpoint.**

Value-weighted best-first search, not breadth-first. Blind BFS fails past about three hops
because node counts multiply roughly tenfold per hop and almost all of it is dust and
unrelated traffic. This follows the largest remaining flow first, prunes branches carrying
a negligible share, and treats an entire peel chain as one logical hop rather than hundreds.

It stops at one of four outcomes and names which:

- **Service** — an exchange or custodial business. The win, because such a business is
  legally required to hold identity documents.
- **Dormant** — received and never spent. Parked, not lost.
- **Mixer** — the trail genuinely ends.
- **Horizon** — ran out of hops or budget. An unfinished trace, labelled as such.

---

## What it cannot do

It cannot name a person. Nothing that reads the blockchain can, because the blockchain
contains no names. It finds the doorway where the off-chain world holds that name, and
assembles the evidence needed to justify asking for it.

---

## Configuring the data source

The **Source** screen sets all of this and writes it to `data/chain/sources.json`:

- **Which API.** Presets are marked verified or unreachable based on what actually answered
  from this machine. `blockchain.info` and `blockstream.info` both work with no account and
  no key. `mempool.space` timed out here and is labelled accordingly.
- **A custom endpoint.** Paste a URL template using `{hash}`, `{height}`, `{address}`,
  `{offset}`, `{key}`. This is how a paid provider attaches later without touching code.
- **How much chain.** In days or in blocks, any value, with an optional end height. Nothing
  about the window is hardcoded.

---

## Hardware

Device selection is measured at runtime, never assumed:

```
torch.cuda.is_available()   ->  use the GPU when one exists
torch.cuda.mem_get_info()   ->  size batches from FREE VRAM, not total
```

Batch size is derived from what the card actually has spare, so the same code fills a 6 GB
card or a 24 GB one with no constant to edit. When a window is too large for full-graph
training it switches to neighbour-sampled mini-batches, so memory scales with the batch
rather than the dataset, and a bigger window costs more steps instead of more VRAM.

The interface reports the real device. On a machine with no CUDA it says CPU, and means it.

Full-graph GraphSAGE across 647,542 nodes took **168 minutes on CPU**. It is a GPU job.

---

## Reading the numbers honestly

**Scores rank what to examine first. They are not probabilities of criminality.**

There is no ground truth for a random slice of Bitcoin; nobody has labelled these
addresses. Training labels come from the behavioural rules and the clustering,
cross-referenced against OFAC sanctions and Ransomwhere confirmed ransomware where those
intersect. So the network learns to *spread* those signals across graph structure, which is
more than the rules do alone. It has not learned what a criminal is from verified examples,
and any reported accuracy is agreement with those weak labels.

A concrete illustration from a real 2-day run. The model reported an F1 of 0.9541, and its
top twenty addresses were all Taproot, all between 0.3305 and 0.3321 BTC, all one-in
one-out, all scoring 0.999989 identically. That is one homogeneous batch, almost certainly
an inscription mint, not laundering. The plumbing was correct and the ranking was real; the
labels were the weak part.

To get detection rather than ranking, supply confirmed labels:

```bash
node scripts/trace.js --fetch-labels
```

---

## Layout

```
electron/
  main.js            app lifecycle, window, menus, background stages
  preload.js         the complete list of what the page may do natively
src/
  chain/
    blocks.js        block-range fetcher, reduction, disk cache
    client.js        multi-provider address lookups with failover
    fetch.js         Ransomwhere and OFAC label sources
    model.js         satoshi-integer money, address validation, provenance
    providers.js     provider pool and rate-limit benching
    sources.js       which API and which window, read from config not code
  analysis/
    behavior.js      behavioural features, peel-chain detection
    trace.js         value-weighted endpoint tracer
  core/
    unionfind.js     disjoint set union, path compression + union by rank
  server/
    app.js           local API, bound to 127.0.0.1 only
python/
  pipeline.py        HDBSCAN, weak labels, GraphSAGE, VRAM planning
  metrics.py         illicit F1, AUC-PR, precision@k, Youden's J
scripts/
  ingest.js          pull blocks, build features, write the window
  trace.js           follow one address from the terminal
ui/
  index.html         the investigation console
  console.js         interface logic, no framework
  plot.js            canvas money-flow graph, no library
  app.css            styling
```

---

## Design notes

**Money is integer satoshis, never floating-point BTC.** `0.1 + 0.2` does not equal `0.3`
in binary floating point, and across a million transactions those errors accumulate into
balances that are visibly wrong. BTC exists only for display, produced at the last moment.

**Every record carries its provenance.** Source, endpoint, block height, ingestion time. A
number on screen must be traceable to the block it came from, or a user cannot tell a real
balance from an invented one.

**The interface has no synthetic fallback.** When nothing is ingested it shows an empty
state and the two commands that fix it. A console that invents numbers to fill itself is
worse than one that shows none.

**No graph library in the renderer.** Cytoscape or d3 would be a few hundred kilobytes from
a CDN, and this has to work with the network cable pulled out. The force simulation and the
arrows are about three hundred lines on a canvas, with repulsion on a spatial hash so a few
hundred nodes stay smooth.

**Complexity, since it decides what is possible.** Union-Find is O(α(n)) per operation,
effectively constant. Feature extraction is one pass over transactions, O(E). GNN training
is O(L·E·d) per epoch, scaling with edges rather than node pairs. Nothing materialises an
n-by-n matrix, which is what makes a million-address window feasible at all.

---

## Known limits

- Exit points are identified structurally, from in-degree and out-degree inside the window.
  An exchange and a large laundering operation genuinely look alike by shape. A maintained
  tag list is what separates them, and this build does not carry one.
- Analysis covers only the blocks loaded. A clean result is not proof that nothing happened.
- Clustering assigns behaviour, not ownership. Two addresses in one cohort act alike; that
  is not a claim they share an owner.
- Public explorers rate-limit. The client rotates providers and backs off, but a large
  first fetch takes time.

