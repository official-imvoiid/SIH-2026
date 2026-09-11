"""Real Bitcoin data: actual ransomware wallets, actual on-chain transactions.

Everything else in this repo can run on the synthetic world in ``ml/data/synth.py``. This
module is what makes the results mean something, because it uses:

* **Ransomwhere** (ransomwhe.re) -- ~11,000 real Bitcoin addresses that received real
  ransom payments, each labelled with the ransomware family that collected it. The family
  label is the closest thing to a public "which gang owns this wallet" ground truth that
  exists.
* **OFAC SDN** -- Bitcoin addresses sanctioned by the US Treasury. Real, official.
* **mempool.space** -- the live Bitcoin blockchain. Real transactions, real inputs and
  outputs, real timestamps.

Network policy
--------------
Fetching happens **once** and is cached to ``data/raw/``. After the first run everything
works with the network unplugged, which is the requirement for demo day. Nothing here is
called at request time by the API.

We are polite to mempool.space: one request at a time, a delay between requests, and
exponential backoff on 429. It is a free public service and this is a student project.

Honest limits
-------------
* ``/address/{a}/txs`` returns the most recent ~50 transactions. Very busy addresses are
  therefore truncated, and we record that in the crawl report rather than hiding it.
* Ransomwhere is crowdsourced and delays publication ~90 days. Coverage is partial.
* We get *positive* labels only. There is no public list of confirmed-clean Bitcoin
  addresses, so a supervised model trained here would have no trustworthy negatives. That
  is why the headline experiment (``ml/experiments/real_vs_control.py``) tests the
  **rule engine**, which needs no labels at all, rather than the classifier.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

from ml.config import DATA_RAW, RANDOM_SEED

RANSOMWHERE_URL = "https://api.ransomwhe.re/export"
OFAC_URL = (
    "https://raw.githubusercontent.com/0xB10C/"
    "ofac-sanctioned-digital-currency-addresses/lists/sanctioned_addresses_XBT.txt"
)
MEMPOOL = "https://mempool.space/api"

CACHE = DATA_RAW / "chain_cache"
USER_AGENT = "ChainTrace/0.1 (student research project; contact via repo)"

# Politeness. mempool.space is free and unauthenticated; do not hammer it.
REQUEST_DELAY_S = 0.55
MAX_RETRIES = 4

# A two-week bucket, matching the Elliptic dataset's time-step convention so that
# TRAIN_TS_MAX / DRIFT_TS in ml/config.py keep their meaning across both datasets.
SECONDS_PER_TIMESTEP = 14 * 24 * 3600


def _get(url: str, timeout: int = 45) -> bytes:
    """HTTP GET with backoff. Raises after MAX_RETRIES."""
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2.5
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError(f"giving up on {url}")


# ---------------------------------------------------------------------------------------
# Label sources
# ---------------------------------------------------------------------------------------


def fetch_ransomwhere(refresh: bool = False) -> list[dict]:
    """Real ransomware payment addresses, labelled by the family that collected them."""
    path = DATA_RAW / "ransomwhere.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))["result"]

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching Ransomwhere ({RANSOMWHERE_URL}) ...")
    raw = _get(RANSOMWHERE_URL, timeout=120)
    path.write_bytes(raw)
    print(f"  cached {len(raw)/1e6:.1f} MB -> {path}")
    return json.loads(raw)["result"]


def fetch_ofac(refresh: bool = False) -> set[str]:
    """Bitcoin addresses on the US Treasury sanctions list."""
    path = DATA_RAW / "ofac_btc.txt"
    if not path.exists() or refresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  fetching OFAC sanctions list ...")
        path.write_bytes(_get(OFAC_URL))
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


# ---------------------------------------------------------------------------------------
# Blockchain access
# ---------------------------------------------------------------------------------------


def _cache_path(address: str) -> Path:
    # Shard by prefix so one directory does not end up with tens of thousands of files.
    return CACHE / address[:4] / f"{address}.json"


def fetch_address_txs(address: str, refresh: bool = False) -> list[dict]:
    """Fetch (and cache) the recent transactions for one address.

    Returns raw mempool.space transaction objects. Cached to disk permanently -- the
    blockchain is append-only, so an old fetch is never *wrong*, only incomplete.
    """
    path = _cache_path(address)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        raw = _get(f"{MEMPOOL}/address/{address}/txs")
        txs = json.loads(raw)
    except Exception as exc:
        txs = []
        print(f"    ! {address[:12]}...: {type(exc).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(txs), encoding="utf-8")
    time.sleep(REQUEST_DELAY_S)
    return txs


def fetch_recent_block_addresses(n_blocks: int = 3, refresh: bool = False) -> list[str]:
    """Addresses drawn from recent blocks -- the control group.

    These are ordinary Bitcoin users, exchanges, miners and merchants going about their
    business. They are *not* verified clean (no such public list exists), but they are a
    fair sample of normal on-chain activity, which is exactly what a control needs to be.
    """
    path = DATA_RAW / "control_addresses.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    print("  sampling addresses from recent blocks ...")
    tip = int(_get(f"{MEMPOOL}/blocks/tip/height").decode())
    addresses: list[str] = []
    for i in range(n_blocks):
        height = tip - 6 - i * 37  # step back so blocks are not adjacent
        block_hash = _get(f"{MEMPOOL}/block-height/{height}").decode().strip()
        time.sleep(REQUEST_DELAY_S)
        txs = json.loads(_get(f"{MEMPOOL}/block/{block_hash}/txs"))
        time.sleep(REQUEST_DELAY_S)
        for tx in txs:
            for vout in tx.get("vout", []):
                a = vout.get("scriptpubkey_address")
                if a:
                    addresses.append(a)
        print(f"    block {height}: {len(txs)} txs")

    addresses = sorted(set(addresses))
    path.write_text(json.dumps(addresses), encoding="utf-8")
    print(f"  {len(addresses):,} distinct control addresses cached")
    return addresses


# ---------------------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------------------


def _tx_to_record(tx: dict) -> tuple[str, list[str], list[float], int, list[tuple[str, float]]] | None:
    """Convert a mempool.space transaction into our pipeline's shape.

    Input addresses are what co-spend clustering consumes; values are in BTC.
    """
    status = tx.get("status", {})
    if not status.get("confirmed"):
        return None

    # `or {}` rather than a .get default: coinbase inputs carry an explicit
    # "prevout": null, so the default never fires and .get() would be called on None.
    # Every mined block contains one, so this crashes on real data without the guard.
    inputs = [
        (v.get("prevout") or {})["scriptpubkey_address"]
        for v in tx.get("vin", [])
        if (v.get("prevout") or {}).get("scriptpubkey_address")
    ]
    outputs = [
        (o["scriptpubkey_address"], o.get("value", 0) / 1e8)
        for o in tx.get("vout", [])
        if o and o.get("scriptpubkey_address")
    ]
    if not inputs or not outputs:
        return None  # coinbase, or an unusual script we cannot attribute

    ts = int(status.get("block_time", 0)) // SECONDS_PER_TIMESTEP
    return tx["txid"], inputs, [v for _a, v in outputs], ts, outputs


def crawl(
    seeds: Iterable[str],
    max_addresses: int = 400,
    hops: int = 1,
    skip_hubs_over: int = 60,
) -> tuple[list, dict]:
    """Breadth-first crawl outward from seed addresses, fetching real transactions.

    Parameters
    ----------
    max_addresses
        Hard cap on how many addresses we fetch. This is a courtesy limit on a free API
        as much as a runtime one.
    skip_hubs_over
        Do not expand *through* an address with more transactions than this. Exchange hot
        wallets connect to everything; following them would pull in the whole chain and
        tell us nothing about the subject.
    """
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    records: list = []
    txids: set[str] = set()
    # NOTE: fetched_addresses is large. Callers printing this dict should strip it --
    # see the summary print in ml/experiments/real_vs_control.py.
    report = {"fetched": 0, "skipped_hubs": 0, "truncated_addresses": 0, "failed": 0,
              # Which addresses we actually retrieved history for. Downstream analysis
              # must scope itself to these -- everything else in the graph is an address
              # we merely glimpsed as a transaction output.
              "fetched_addresses": []}

    while queue and len(seen) < max_addresses:
        address, depth = queue.popleft()
        if address in seen:
            continue
        seen.add(address)

        txs = fetch_address_txs(address)
        report["fetched"] += 1
        report["fetched_addresses"].append(address)
        if not txs:
            report["failed"] += 1
            continue
        if len(txs) >= 50:
            # The API caps at ~50; this address has more history than we can see.
            report["truncated_addresses"] += 1

        if len(txs) > skip_hubs_over:
            report["skipped_hubs"] += 1
            continue

        for tx in txs:
            rec = _tx_to_record(tx)
            if rec is None or rec[0] in txids:
                continue
            txids.add(rec[0])
            records.append(rec)

            if depth < hops:
                for _addr, _v in rec[4]:
                    if _addr not in seen and len(seen) + len(queue) < max_addresses * 2:
                        queue.append((_addr, depth + 1))
                for _addr in rec[1]:
                    if _addr not in seen and len(seen) + len(queue) < max_addresses * 2:
                        queue.append((_addr, depth + 1))

        if report["fetched"] % 25 == 0:
            print(
                f"    fetched {report['fetched']:>4} addresses, "
                f"{len(records):>5} transactions, queue {len(queue)}"
            )

    return records, report


def select_seeds(
    ransomware: list[dict], n: int = 40, min_families: int = 5
) -> list[dict]:
    """Pick real ransomware addresses that actually received money, spread across families.

    Spreading across families matters: 7,000 of the ~11,000 Ransomwhere entries are Locky,
    and a sample drawn from that alone would tell us about one crew's habits rather than
    about ransomware generally.
    """
    rng = random.Random(RANDOM_SEED)
    paid = [
        r
        for r in ransomware
        if r.get("blockchain") == "bitcoin"
        and r.get("balance", 0) > 0
        and r.get("family")
        and r.get("family") != "Unlabeled"
    ]
    by_family: dict[str, list[dict]] = {}
    for r in paid:
        by_family.setdefault(r["family"], []).append(r)

    # Prefer families with enough addresses to be interesting, then round-robin.
    families = sorted(by_family, key=lambda f: -len(by_family[f]))[: max(min_families, 12)]
    chosen: list[dict] = []
    idx = 0
    while len(chosen) < n and families:
        fam = families[idx % len(families)]
        bucket = by_family[fam]
        if bucket:
            chosen.append(bucket.pop(rng.randrange(len(bucket))))
        else:
            families.remove(fam)
            continue
        idx += 1
    return chosen


if __name__ == "__main__":
    print("Real data sources")
    print("-" * 66)
    rw = fetch_ransomwhere()
    ofac = fetch_ofac()
    btc = [r for r in rw if r.get("blockchain") == "bitcoin"]
    paid = [r for r in btc if r.get("balance", 0) > 0]
    fams = Counter(r.get("family") for r in paid)
    print(f"  ransomwhere entries      {len(rw):>8,}")
    print(f"  bitcoin, received funds  {len(paid):>8,}")
    print(f"  ransomware families      {len(fams):>8,}")
    print(f"  OFAC sanctioned BTC      {len(ofac):>8,}")
    print("\n  top families by paid address count:")
    for f, c in fams.most_common(8):
        print(f"    {c:>6,}  {f}")
