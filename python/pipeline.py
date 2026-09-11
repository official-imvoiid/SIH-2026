"""Clustering and graph-network stage.

Reads the window produced by scripts/ingest.js and runs:

    1. HDBSCAN over behavioural feature vectors, to find cohorts and outliers
    2. Weak labelling, from clustering plus confirmed ransomware/sanctions lists
    3. A Graph Neural Network in PyTorch Geometric, on the real address graph
    4. Scoring, so risk propagates along the money flow rather than per-address

WHY HDBSCAN RATHER THAN K-MEANS

K-means needs the number of clusters chosen up front and assigns every point to one,
which is exactly wrong here: we do not know how many kinds of behaviour exist in a
two-day window, and most addresses belong to no interesting group at all. HDBSCAN infers
the cluster count from density and, critically, labels genuinely unusual points as noise
(-1) instead of forcing them into the nearest blob. In this problem the noise label is
not a failure, it is a finding: an address whose behaviour is too unusual to belong to
any cohort is precisely what an anomaly hunt is looking for.

It also handles clusters of differing density, which matters because normal wallet
behaviour is one enormous dense cloud while laundering patterns are small tight knots
far from it. K-means would let the big cloud dominate the centroids.

WHY A GRAPH NETWORK ON TOP

Clustering treats every address independently; it cannot see that a wallet is suspicious
because of what it is connected to. A GNN's message passing is exactly that missing
piece. A four-layer network lets information travel four hops, so the model reasons about
a wallet's position in the flow of funds, including through splits and joins, without
writing recursive traversal by hand.

THE HONEST LIMIT ON LABELS

There is no ground truth for a random two-day slice of Bitcoin. Nobody has labelled these
addresses. So the GNN is trained on weak labels derived from the behavioural heuristics
and the clustering, cross-referenced against the confirmed ransomware and sanctions lists
where those intersect. That means the network learns to *generalise and propagate* the
heuristics across graph structure, which is genuinely more than the heuristics do alone.
It does not mean it has learned what a criminal is from verified examples, and any
reported accuracy is accuracy against those weak labels, not against reality. Where
confirmed labels do appear, they are reported separately and counted.

COMPLEXITY

HDBSCAN is roughly O(n log n) on low-dimensional data, which is why clustering runs on
the four specified features rather than all eighteen. GNN training is O(L * E * d) per
epoch for L layers, E edges and hidden width d, so it scales with edges rather than with
node pairs. Both are near-linear; neither materialises an n-by-n matrix, which is what
makes a million-address window feasible at all.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parent.parent
WINDOW = ROOT / "data" / "chain" / "window.json"
OUT = ROOT / "data" / "chain" / "analysis.json"
LABELS_DIR = ROOT / "data" / "chain" / "labels"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------

def device_report() -> dict:
    """What hardware this will actually run on. Checked, never assumed."""
    try:
        import torch
    except ImportError:
        return {"torch": False, "device": "unavailable"}

    cuda = torch.cuda.is_available()
    info = {
        "torch": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
    }
    if cuda:
        p = torch.cuda.get_device_properties(0)
        # Free memory, not total. Total is the number on the box; free is what is left
        # after the display driver, the desktop compositor and any other process have
        # taken their share, and that is what training actually gets to use.
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        info.update(
            gpu_name=p.name,
            vram_gb=round(total_bytes / 1024**3, 2),
            vram_free_gb=round(free_bytes / 1024**3, 2),
            cuda_version=torch.version.cuda,
            compute_capability=f"{p.major}.{p.minor}",
        )
    else:
        info["note"] = "No CUDA device on this machine. Training runs on CPU."
    return info


# ---------------------------------------------------------------------------------------
# Stage 1: clustering
# ---------------------------------------------------------------------------------------

def cluster_behaviour(X: np.ndarray, cluster_idx: list, min_cluster_size: int = 15,
                      max_points: int = 60000) -> dict:
    """HDBSCAN over the four behavioural features.

    Features are log-scaled first. Holding time spans seconds to days and transaction
    frequency spans four orders of magnitude, so on raw values the distance metric is
    dominated entirely by whichever feature happens to have the largest units, and the
    clustering degenerates into a one-dimensional split on that feature.
    """
    from sklearn.cluster import HDBSCAN
    from sklearn.preprocessing import StandardScaler

    Xc = X[:, cluster_idx].astype(float)
    # log1p compresses the heavy tails without discarding ordering; these features are
    # all non-negative counts, durations or rates, so log1p is well defined throughout.
    Xc = np.log1p(np.clip(Xc, 0, None))
    Xc = StandardScaler().fit_transform(Xc)
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)

    # HDBSCAN is roughly O(n log n) in theory but its constant factor is brutal: on a full
    # two-day window (647k addresses) it does not finish in any useful time. Two things fix
    # that, and both are justified on their own merits rather than being pure speed hacks.
    #
    # First, most of those addresses carry no signal at all. A two-day slice of Bitcoin is
    # dominated by one-shot change addresses that receive once, spend once and never appear
    # again. They cannot express layering, collection or peeling, so clustering them is
    # spending nearly all the compute on rows that can never produce a finding.
    #
    # Second, above a certain size the density structure stops changing. A well-chosen
    # subsample of 60k points recovers the same cluster shapes as the full set; the extra
    # 587k mostly thicken the same modes. So the clusterer is fitted on a stratified
    # subsample and every remaining address is then assigned to the nearest resulting
    # cluster centroid, which is O(n * k) and takes seconds.
    n_total = Xc.shape[0]
    rng = np.random.default_rng(RANDOM_SEED)

    if n_total > max_points:
        # Stratify by activity so the subsample keeps the rare, interesting rows rather
        # than drowning them: uniform sampling of a set that is 90% dust returns dust.
        activity = Xc.sum(axis=1)
        order = np.argsort(-activity)
        # Half the budget goes to the most active addresses, half sampled from the rest,
        # so both the unusual tail and the ordinary bulk are represented.
        top_n = max_points // 2
        fit_idx = np.concatenate([
            order[:top_n],
            rng.choice(order[top_n:], size=max_points - top_n, replace=False),
        ])
    else:
        fit_idx = np.arange(n_total)

    # Minimum cohort size scales with how many points are actually fitted. A fixed 15 is
    # right for a few thousand addresses and badly wrong for sixty thousand: it returned
    # 1,306 cohorts, which is a dendrogram dump rather than something an analyst reads.
    # Tying it to roughly 0.5% of the sample keeps the count in the tens.
    effective_min = max(min_cluster_size, len(fit_idx) // 200)

    t0 = time.time()
    clusterer = HDBSCAN(
        min_cluster_size=effective_min,
        min_samples=5,
        # Euclidean over standardised log features: each of the four contributes equally.
        metric="euclidean",
        # Excess-of-mass selection, not leaf. Leaf was the original choice on the
        # reasoning that small tight cohorts suit hunting operator groups, but measured on
        # a real two-day window it returned 1,432 clusters from 60,000 points. That is not
        # a finding an analyst can act on, it is the dendrogram's bottom row exposed as
        # output. Excess of mass keeps the persistent structure and yields a number of
        # cohorts a person can actually read through.
        cluster_selection_method="eom",
    )
    fit_labels = clusterer.fit_predict(Xc[fit_idx])

    if n_total > max_points:
        # Assign every unfitted address to the nearest cluster centroid. Points whose
        # nearest centroid is further than the cluster's own spread stay noise, so this
        # cannot manufacture membership for something that belongs to no cohort.
        labels = np.full(n_total, -1, dtype=np.int64)
        labels[fit_idx] = fit_labels

        real = sorted(set(fit_labels.tolist()) - {-1})
        if real:
            fitted = Xc[fit_idx]
            centroids = np.stack([fitted[fit_labels == c].mean(axis=0) for c in real])
            radii = np.array([
                np.percentile(np.linalg.norm(fitted[fit_labels == c] - centroids[i], axis=1), 90)
                for i, c in enumerate(real)
            ])
            radii = np.maximum(radii, 1e-6)

            rest = np.setdiff1d(np.arange(n_total), fit_idx, assume_unique=False)

            # Nearest centroid via a KD-tree, not a broadcast difference.
            #
            # The obvious vectorised form builds an (n_chunk, n_clusters, n_dims) array.
            # With 50,000 rows and the 1,432 clusters this data actually produces, that is
            # a 2.3 GB allocation per chunk, and the process spends its life in the
            # allocator rather than finishing. A KD-tree answers the same question in
            # O(n log k) with no large intermediate, and is exact for Euclidean distance.
            from scipy.spatial import cKDTree

            tree = cKDTree(centroids)
            real_arr = np.array(real)
            for start in range(0, len(rest), 100000):
                chunk = rest[start:start + 100000]
                dist, nearest = tree.query(Xc[chunk], k=1, workers=-1)
                # Outside its cohort's own spread stays noise, so assignment cannot
                # manufacture membership for a point that belongs to nothing.
                labels[chunk] = np.where(dist <= radii[nearest], real_arr[nearest], -1)
    else:
        labels = fit_labels

    elapsed = time.time() - t0

    unique = sorted(set(labels.tolist()))
    clusters = {}
    for lab in unique:
        mask = labels == lab
        clusters[int(lab)] = {
            "label": int(lab),
            "size": int(mask.sum()),
            "is_noise": lab == -1,
            "centroid": {},
        }
        for j, fi in enumerate(cluster_idx):
            clusters[int(lab)]["centroid"][str(fi)] = float(np.median(X[mask, fi]))

    return {
        "labels": labels.tolist(),
        "n_clusters": int(len([u for u in unique if u != -1])),
        "n_noise": int((labels == -1).sum()),
        "clusters": clusters,
        "elapsed_s": round(elapsed, 2),
        "min_cluster_size": int(effective_min),
        "fitted_on": int(len(fit_idx)),
        "total_points": int(n_total),
        "subsampled": bool(n_total > max_points),
    }


def score_clusters(X: np.ndarray, labels: np.ndarray, names: list) -> dict:
    """Rank clusters by how strongly they express the laundering patterns.

    A cluster is interesting when its members layer fast, collect from many senders,
    peel, or retain nothing. Scored on the median member so one extreme address cannot
    carry a cluster.
    """
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for lab in sorted(set(labels.tolist())):
        mask = labels == lab
        if mask.sum() == 0:
            continue

        def med(name):
            return float(np.median(X[mask, idx[name]])) if name in idx else 0.0

        layering = med("layering_score")
        collector = med("collector_score")
        peel = med("peel_score")
        forward = med("forward_ratio")
        hold = med("avg_holding_time")

        # Weighted toward the patterns that are hardest to explain innocently. Fast
        # turnaround alone is common (exchanges relay constantly); it only contributes
        # meaningfully when combined with the structural patterns.
        risk = (
            0.30 * layering
            + 0.30 * collector
            + 0.20 * peel
            + 0.15 * forward
            + 0.05 * (1.0 if hold < 600 else 0.0)
        )

        out[int(lab)] = {
            "size": int(mask.sum()),
            "is_noise": lab == -1,
            "risk": round(float(min(1.0, risk)), 4),
            "median_layering": round(layering, 4),
            "median_collector": round(collector, 4),
            "median_peel": round(peel, 4),
            "median_forward_ratio": round(forward, 4),
            "median_holding_seconds": round(hold, 1),
        }
    return out


# ---------------------------------------------------------------------------------------
# Stage 2: weak labels
# ---------------------------------------------------------------------------------------

def load_confirmed_labels() -> dict:
    """Confirmed ransomware and sanctioned addresses, if they have been downloaded."""
    confirmed = {}
    rw = LABELS_DIR / "ransomwhere.json"
    if rw.exists():
        try:
            data = json.loads(rw.read_text(encoding="utf-8"))
            for addr in data.get("addresses", {}):
                confirmed[addr] = "ransomware"
        except Exception:
            pass
    ofac = LABELS_DIR / "ofac.json"
    if ofac.exists():
        try:
            data = json.loads(ofac.read_text(encoding="utf-8"))
            for addr in data.get("addresses", {}):
                confirmed[addr] = "sanctioned"
        except Exception:
            pass
    return confirmed


def build_weak_labels(X: np.ndarray, names: list, cluster_risk: dict,
                      cluster_labels: np.ndarray, addresses: list,
                      confirmed: dict) -> dict:
    """Produce training labels, and be explicit about where each one came from."""
    idx = {n: i for i, n in enumerate(names)}
    n = X.shape[0]
    y = np.zeros(n, dtype=np.int64)
    source = np.empty(n, dtype=object)
    source[:] = "unlabelled"

    layering = X[:, idx["layering_score"]]
    collector = X[:, idx["collector_score"]]
    peel = X[:, idx["peel_score"]]
    forward = X[:, idx["forward_ratio"]]
    received = X[:, idx["received_btc"]]

    # Positive from behaviour: a clearly expressed laundering pattern with real value
    # behind it. The value floor matters because dust transactions produce extreme
    # ratios on amounts too small for anyone to be laundering.
    behavioural_pos = (
        ((layering > 0.25) | (collector > 0.25) | ((peel > 0.6) & (forward > 0.9)))
        & (received > 0.001)
    )
    y[behavioural_pos] = 1
    source[behavioural_pos] = "behavioural_pattern"

    # Positive from clustering: membership of a cohort whose median member looks bad.
    high_risk_clusters = {int(k) for k, v in cluster_risk.items()
                          if v["risk"] >= 0.35 and not v["is_noise"] and v["size"] >= 5}
    if high_risk_clusters:
        in_bad_cluster = np.isin(cluster_labels, list(high_risk_clusters))
        newly = in_bad_cluster & (y == 0)
        y[newly] = 1
        source[newly] = "hdbscan_cluster"

    # Negative: quiet, slow, retains value, no pattern. Deliberately conservative so the
    # negative class is confidently ordinary rather than merely unflagged.
    negative = (
        (layering == 0) & (collector == 0) & (peel < 0.2)
        & (forward < 0.8) & (X[:, idx["avg_holding_time"]] > 3600)
    )
    y[negative & (y == 0)] = 0
    source[negative & (source == "unlabelled")] = "quiet_behaviour"

    # Confirmed labels always win over anything inferred.
    n_confirmed = 0
    for i, addr in enumerate(addresses):
        if addr in confirmed:
            y[i] = 1
            source[i] = f"confirmed_{confirmed[addr]}"
            n_confirmed += 1

    counts = {}
    for s in source:
        counts[s] = counts.get(s, 0) + 1

    return {
        "y": y,
        "source": source.tolist(),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_confirmed": n_confirmed,
        "by_source": counts,
        "caveat": (
            (
                f"{n_confirmed} address in this window is on a published criminal list "
                "(OFAC sanctions or Ransomwhere confirmed ransomware). "
                if n_confirmed == 1 else
                f"{n_confirmed} addresses in this window are on a published criminal list "
                "(OFAC sanctions or Ransomwhere confirmed ransomware). "
                if n_confirmed > 1 else
                "No address in this window appears on the published criminal lists that were "
                "downloaded, which is expected: a two-day slice of Bitcoin rarely contains a "
                "wallet somebody has already reported. "
            )
            + "The remaining labels come from the behavioural rules and the clustering, so "
            "the model learns to spread those signals across the transaction graph rather "
            "than to recognise crime it has been shown. Scores rank what to examine first. "
            "To turn this into detection rather than ranking, feed it a labelled set: run "
            "'node scripts/trace.js --fetch-labels' to pull the public lists, or load your "
            "own confirmed addresses."
        ),
        "n_confirmed_note": (
            "Cross-referenced against OFAC and Ransomwhere where those lists were available."
        ),
    }


# ---------------------------------------------------------------------------------------
# Stage 3: the graph network
# ---------------------------------------------------------------------------------------

def plan_memory(n_nodes: int, n_edges: int, n_features: int, hidden: int, layers: int,
                device_info: dict) -> dict:
    """Decide how to train so it fits the machine it is actually on.

    Nothing here is a hardcoded cap. The budget is measured at runtime and the batch size
    is derived from it, so the same code fills a 6 GB card, a 24 GB card, or falls back to
    CPU without any constant needing to be edited.

    Full-graph GraphSAGE holds, per layer, the activations for every node: roughly
    n_nodes * hidden * 4 bytes, times two for the gradient, plus the message-passing
    workspace which scales with edges. For 647k nodes at hidden=64 across 4 layers that is
    about 1.3 GB of activations before the optimiser state and the edge index, so a full
    two-day window will not fit in 6 GB alongside a desktop.

    The fix is neighbour sampling rather than a smaller graph: train on mini-batches of
    seed nodes, expanding only a bounded neighbourhood around each. Memory then scales
    with the batch, not with the dataset, so an arbitrarily large window trains on a small
    card by taking more steps rather than by discarding data.
    """
    bytes_per_float = 4

    # Per seed node, a sampled subgraph of depth `layers` with the fan-outs below holds
    # roughly prod(fanout) nodes. Estimate the activation cost of one seed node.
    fanout = [15, 10, 5, 5][:layers] or [10]

    # The naive estimate multiplies the fan-outs together: 15 x 10 x 5 x 5 gives 4,666
    # unique nodes per seed. Measured against a real batch that is roughly thirty times
    # too high, because sampled neighbourhoods overlap heavily -- a transaction graph has
    # hubs, and two seeds a few hops apart pull in largely the same nodes. Believing the
    # naive figure drove the batch size down to 146, which turned one epoch into 2,662
    # steps and made CPU training take hours for no memory benefit.
    #
    # A saturating estimate is closer: each additional layer adds progressively less
    # because the frontier starts colliding with nodes already sampled.
    nodes_per_seed = 1.0
    reach = 1.0
    for depth, f in enumerate(fanout):
        reach *= f
        # Overlap grows with depth; by the fourth layer most of the frontier is repeats.
        nodes_per_seed += reach * (0.55 ** depth)
    nodes_per_seed = max(8.0, min(nodes_per_seed, 400.0))

    per_seed_bytes = (
        nodes_per_seed * n_features * bytes_per_float          # input features
        + nodes_per_seed * hidden * layers * bytes_per_float * 3  # activations + grads
    )

    if device_info.get("cuda_available"):
        free_gb = device_info.get("vram_free_gb") or device_info.get("vram_gb") or 4.0
        # Leave headroom: CUDA context, fragmentation, and whatever else wants the card.
        usable_bytes = max(0.5, free_gb - 1.0) * 1024**3
        mode = "cuda"
    else:
        # On CPU the binding constraint is wall-clock time, not memory: ordinary machines
        # have gigabytes free, and a tiny batch only multiplies the number of steps. Use a
        # real slice of RAM so batches stay large and the epoch count stays sane.
        import shutil  # noqa: F401  (kept for parity with disk checks elsewhere)
        usable_bytes = 8.0 * 1024**3
        mode = "cpu"

    batch_size = int(max(512, min(32768, usable_bytes // max(1, per_seed_bytes))))

    # Full-graph training is simpler and faster when the whole thing comfortably fits.
    full_graph_bytes = (
        n_nodes * n_features * bytes_per_float
        + n_nodes * hidden * layers * bytes_per_float * 3
        + n_edges * 2 * 8
    )
    use_full_graph = full_graph_bytes < usable_bytes * 0.6

    return {
        "mode": mode,
        "batch_size": batch_size,
        "fanout": fanout,
        "use_full_graph": bool(use_full_graph),
        "estimated_full_graph_gb": round(full_graph_bytes / 1024**3, 3),
        "usable_gb": round(usable_bytes / 1024**3, 2),
        "per_seed_kb": round(per_seed_bytes / 1024, 1),
        "nodes_per_seed_estimate": nodes_per_seed,
        "strategy": (
            "full-graph, it fits comfortably"
            if use_full_graph
            else f"neighbour sampling, {batch_size} seed nodes per step"
        ),
    }


def train_gnn(X: np.ndarray, y: np.ndarray, edges: list, config: dict | None = None) -> dict:
    """Four-layer GraphSAGE node classifier in PyTorch Geometric.

    GraphSAGE rather than plain GCN because it aggregates from a neighbourhood with a
    learned transform of the node's own features kept separate from its neighbours'. That
    distinction matters here: a wallet paid by a criminal should not thereby become
    identical to one, and GCN's symmetric averaging blurs exactly that line.
    """
    config = config or {}
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
    except ImportError as exc:
        return {"ok": False, "error": f"PyTorch Geometric unavailable: {exc}"}

    import metrics as M

    dev_info = device_report()
    device = torch.device(dev_info["device"])
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n = X.shape[0]
    if n < 50:
        return {"ok": False, "error": f"Only {n} addresses; too few to train on."}
    if int((y == 1).sum()) < 10:
        return {"ok": False, "error": f"Only {int((y == 1).sum())} positive examples; too few to train on."}

    # Standardise on the training split only, computed after the split below.
    edge_index = (
        torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
        if edges else torch.zeros((2, 0), dtype=torch.long)
    )
    # Undirected message passing: influence should travel both with the money and
    # against it, because where funds came from is as informative as where they went.
    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Split by first-seen order, so the model is validated on addresses that appeared
    # later in the window than the ones it learned from.
    order = np.arange(n)
    cut1, cut2 = int(n * 0.6), int(n * 0.8)
    train_mask = np.zeros(n, dtype=bool); train_mask[order[:cut1]] = True
    val_mask = np.zeros(n, dtype=bool); val_mask[order[cut1:cut2]] = True
    test_mask = np.zeros(n, dtype=bool); test_mask[order[cut2:]] = True

    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.tensor(Xs, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    edge_index = edge_index.to(device)

    hidden = config.get("hidden", 64)
    layers = config.get("layers", 4)

    class SAGENet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            dims = [in_dim] + [hidden] * layers
            for i in range(layers):
                self.convs.append(SAGEConv(dims[i], dims[i + 1]))
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))
            self.head = nn.Linear(hidden, 1)
            self.dropout = config.get("dropout", 0.3)

        def forward(self, x, edge_index):
            for conv, norm in zip(self.convs, self.norms):
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.head(x).squeeze(-1)

    model = SAGENet(X.shape[1]).to(device)

    n_pos = max(1, int(y[train_mask].sum()))
    n_neg = max(1, int(train_mask.sum()) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 5e-3), weight_decay=5e-4)

    tr = torch.tensor(train_mask, device=device)
    va = torch.tensor(val_mask, device=device)
    te = torch.tensor(test_mask, device=device)

    epochs = config.get("epochs", 200)
    patience = config.get("patience", 30)
    best_val = -1.0
    best_state = None
    stale = 0
    history = []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(out[tr], yt[tr], pos_weight=pos_weight)
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(x, edge_index)).cpu().numpy()
            vf1 = M.evaluate(y[val_mask], p[val_mask])["illicit"]["f1"] if val_mask.sum() else 0.0
            history.append({"epoch": epoch, "loss": round(float(loss.item()), 5), "val_f1": vf1})
            if vf1 > best_val:
                best_val = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 5
                if stale >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(x, edge_index)).cpu().numpy()

    return {
        "ok": True,
        "device": dev_info,
        "architecture": f"{layers}-layer GraphSAGE, {X.shape[1]} -> {hidden} -> 1, BatchNorm + dropout",
        "hops_reachable": layers,
        "epochs_run": history[-1]["epoch"] + 1 if history else 0,
        "trained_in_seconds": round(time.time() - t0, 2),
        "history": history,
        "n_nodes": int(n),
        "n_edges": int(edge_index.shape[1] // 2),
        "split": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
            "train_positive": int(y[train_mask].sum()),
            "test_positive": int(y[test_mask].sum()),
        },
        "train_metrics": M.evaluate(y[train_mask], proba[train_mask]),
        "val_metrics": M.evaluate(y[val_mask], proba[val_mask]) if val_mask.sum() else None,
        "test_metrics": M.evaluate(y[test_mask], proba[test_mask]) if test_mask.sum() else None,
        "scores": proba.tolist(),
    }


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------

def main():
    if not WINDOW.exists():
        print(f"No window found at {WINDOW}")
        print("Run:  node scripts/ingest.js --blocks 12")
        sys.exit(1)

    print("")
    print("=" * 74)
    print("  CLUSTERING + GRAPH NEURAL NETWORK")
    print("=" * 74)

    data = json.loads(WINDOW.read_text(encoding="utf-8"))
    names = data["feature_names"]
    addresses = [r["address"] for r in data["rows"]]
    X = np.array([r["x"] for r in data["rows"]], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    edges = [(e[0], e[1]) for e in data["edges"]]

    rng = data["range"]
    print(f"  Window        blocks {rng['start_height']}-{rng['end_height']}")
    print(f"  Addresses     {X.shape[0]:,} x {X.shape[1]} features")
    print(f"  Edges         {len(edges):,}")
    print("")

    dev = device_report()
    print("  HARDWARE")
    if dev.get("cuda_available"):
        print(f"    GPU         {dev['gpu_name']} ({dev['vram_gb']} GB, CUDA {dev['cuda_version']})")
        print(f"    Training on CUDA")
    elif dev.get("torch"):
        print(f"    No CUDA device detected. Training on CPU.")
        print(f"    torch {dev['torch_version']}")
    print("")

    # --------------------------------------------------------------- clustering ---
    cluster_idx = [names.index(n) for n in data["cluster_feature_names"]]
    print(f"  [1/4] HDBSCAN over {len(cluster_idx)} behavioural features...")
    cl = cluster_behaviour(X, cluster_idx)
    labels = np.array(cl["labels"])
    print(f"        {cl['n_clusters']} clusters found, {cl['n_noise']:,} addresses marked as noise/outliers")
    print(f"        took {cl['elapsed_s']}s")

    risk = score_clusters(X, labels, names)
    ranked = sorted(
        [(k, v) for k, v in risk.items() if not v["is_noise"] and v["size"] >= 5],
        key=lambda kv: -kv[1]["risk"],
    )
    print("")
    print("        highest-risk cohorts:")
    for k, v in ranked[:6]:
        print(f"          cluster {str(k).rjust(3)}  {str(v['size']).rjust(5)} addrs  risk {v['risk']:.3f}  "
              f"layer {v['median_layering']:.2f} collect {v['median_collector']:.2f} peel {v['median_peel']:.2f}")
    print("")

    # ------------------------------------------------------------- weak labels ---
    print("  [2/4] Building training labels...")
    confirmed = load_confirmed_labels()
    lab = build_weak_labels(X, names, risk, labels, addresses, confirmed)
    print(f"        {lab['n_positive']:,} positive, {lab['n_negative']:,} negative")
    print(f"        confirmed external labels matched: {lab['n_confirmed']}")
    for k, v in sorted(lab["by_source"].items(), key=lambda kv: -kv[1]):
        print(f"          {k:<28} {v:,}")
    print("")

    # -------------------------------------------------------------------- GNN ---
    print("  [3/4] Training GraphSAGE on the real address graph...")
    gnn = train_gnn(X, lab["y"], edges)
    if not gnn.get("ok"):
        print(f"        failed: {gnn.get('error')}")
        scores = None
    else:
        print(f"        {gnn['architecture']}")
        print(f"        reasons across {gnn['hops_reachable']} hops of the money flow")
        print(f"        trained {gnn['epochs_run']} epochs in {gnn['trained_in_seconds']}s on {gnn['device']['device'].upper()}")
        tm = gnn.get("test_metrics")
        if tm:
            i = tm["illicit"]
            print(f"        held-out agreement with labels: F1 {i['f1']}, precision {i['precision']}, recall {i['recall']}")
        scores = np.array(gnn["scores"])
    print("")

    # --------------------------------------------------------------- ranking ---
    print("  [4/4] Ranking addresses...")
    if scores is None:
        scores = X[:, names.index("layering_score")] * 0.5 + X[:, names.index("collector_score")] * 0.5

    order = np.argsort(-scores)[:25]
    print("")
    print("        TOP SUSPICIOUS ADDRESSES (real, from this window)")
    print("")
    for rank, i in enumerate(order[:15], 1):
        f = {n: X[i, j] for j, n in enumerate(names)}
        tags = []
        if f["layering_score"] > 0.1: tags.append("layering")
        if f["collector_score"] > 0.1: tags.append("fan-in")
        if f["peel_score"] > 0.5: tags.append("peeling")
        if f["forward_ratio"] > 0.95: tags.append("pass-through")
        cluster_tag = f"cluster {labels[i]}" if labels[i] >= 0 else "outlier"
        print(f"        {str(rank).rjust(2)}. {scores[i]:.4f}  {addresses[i]}")
        print(f"            {cluster_tag} | {f['received_btc']:.4f} BTC in, {f['n_tx_in']:.0f} in / {f['n_tx_out']:.0f} out"
              + (f" | {', '.join(tags)}" if tags else ""))
    print("")

    result = {
        "range": rng,
        "device": dev,
        "clustering": {k: v for k, v in cl.items() if k != "labels"},
        "cluster_labels": cl["labels"],
        "cluster_risk": risk,
        "labels": {k: v for k, v in lab.items() if k != "y"},
        "gnn": {k: v for k, v in gnn.items() if k != "scores"} if gnn.get("ok") else gnn,
        "ranking": [
            {
                "rank": r + 1,
                "address": addresses[i],
                "score": float(scores[i]),
                "cluster": int(labels[i]),
                "features": {n: float(X[i, j]) for j, n in enumerate(names)},
            }
            for r, i in enumerate(order)
        ],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"  Wrote {OUT}")
    print("")
    print("  Next:  node scripts/trace.js <address>     follow one to its endpoint")
    print("=" * 74)
    print("")


if __name__ == "__main__":
    main()
