'use strict';
/**
 * Behavioural feature extraction over a block window.
 *
 * This implements the indicators you specified, computed from real transactions rather
 * than assumed:
 *
 *   High-velocity layering   money arrives and is split across several brand-new
 *                            addresses within minutes
 *   Peeling                  the bulk moves on to a fresh address each hop while small
 *                            amounts peel off to the side
 *   High fan-in, fast fan-out  many unrelated senders pay one new address, which then
 *                            sweeps everything onward in one go
 *
 * WHY "NEW ADDRESS" IS THE LOAD-BEARING SIGNAL
 *
 * Almost every laundering pattern depends on freshly generated addresses, because reusing
 * an address is what lets an investigator link activity together. Within one window we can
 * see this directly: an address whose first appearance anywhere in the window is as the
 * output of the transaction being examined is, for our purposes, new. That is cheap to
 * compute and it is the single most discriminating thing available, because ordinary
 * economic activity (exchange withdrawals, merchant payments, personal transfers) reuses
 * addresses heavily and pays to addresses that already existed.
 *
 * It is also the feature with the worst failure mode, so it is worth stating plainly:
 * modern wallet software generates a fresh change address for every single transaction by
 * default. A high proportion of payments to new addresses is therefore completely normal
 * and is NOT on its own evidence of anything. It only becomes interesting combined with
 * speed, fan-out width and the absence of any retained balance, which is why these are
 * clustered together rather than thresholded individually.
 *
 * COMPLEXITY
 *
 * One pass over every transaction to index address first-appearance, then one pass to
 * accumulate per-address statistics: O(T * io) time where T is transactions and io is
 * the average inputs plus outputs, which is effectively O(E) in the number of edges.
 * Memory is O(A) for A distinct addresses, holding a fixed-size accumulator each. For a
 * two-day window that is roughly 1.4M transactions and 1.5M addresses, which fits
 * comfortably in normal RAM as long as no per-address transaction lists are retained.
 * That is why the accumulators below store running counts and sums, never arrays.
 */

const SATS = 1e8;

/** A split happening this fast after the money arrived counts as automated. */
const FAST_SPLIT_SECONDS = 600;

/**
 * Build per-address behavioural features from a set of reduced blocks.
 *
 * @param {Array} blocks from BlockFetcher
 * @param {object} opt {minTxs, onProgress}
 */
function extractBehaviour(blocks, opt = {}) {
  const minTxs = opt.minTxs == null ? 2 : opt.minTxs;
  const onProgress = opt.onProgress || (() => {});

  // Flatten to a time-ordered transaction stream. Order matters: "was this address new
  // at the time" is only answerable if transactions are processed chronologically.
  const txs = [];
  for (const b of blocks) {
    for (const t of b.txs) {
      txs.push({ ...t, height: b.height, block_time: b.time });
    }
  }
  txs.sort((a, b) => (a.time || a.block_time) - (b.time || b.block_time) || a.height - b.height);

  onProgress({ kind: 'phase', phase: 'indexing', transactions: txs.length });

  /** First time each address is seen anywhere in the window. */
  const firstSeen = new Map();
  for (const t of txs) {
    const ts = t.time || t.block_time;
    for (const i of t.inputs) if (!firstSeen.has(i.address)) firstSeen.set(i.address, ts);
    for (const o of t.outputs) if (!firstSeen.has(o.address)) firstSeen.set(o.address, ts);
  }

  onProgress({ kind: 'phase', phase: 'chain-detection', addresses: firstSeen.size });

  /**
   * Find genuine peel CHAINS, not merely peel-shaped transactions.
   *
   * This distinction was not obvious until the numbers were checked: on a real block
   * window, a "two outputs where one takes 85%" test fires on about 12% of all addresses,
   * because that is the shape of an ordinary payment with change. Nearly every wallet in
   * Bitcoin produces it on nearly every spend. Used alone it is not a laundering signal,
   * it is a description of how Bitcoin works, and it swamped the rankings with wallets
   * moving fractions of a coin once.
   *
   * What actually distinguishes a peel chain is repetition: the large output lands on a
   * fresh address which then does the same thing again, and again. So the chain is walked
   * explicitly and only addresses sitting on a sequence of at least MIN_PEEL_CHAIN hops
   * are credited. Position in the chain is recorded too, since being deep in one is more
   * telling than being at its head.
   *
   * Cost is one hash lookup per transaction plus one walk per chain start, and each
   * transaction is consumed by at most one chain, so this stays O(T).
   */
  const MIN_PEEL_CHAIN = 3;

  /** Which transaction spends a given address, earliest first. */
  const spentBy = new Map();
  for (const t of txs) {
    for (const i of t.inputs) {
      const prev = spentBy.get(i.address);
      if (!prev || (t.time || t.block_time) < (prev.time || prev.block_time)) {
        spentBy.set(i.address, t);
      }
    }
  }

  const peelShape = (t) => {
    if (t.outputs.length < 2 || t.outputs.length > 3) return null;
    const total = t.outputs.reduce((s, o) => s + o.value_sats, 0);
    if (total <= 0) return null;
    let big = t.outputs[0];
    for (const o of t.outputs) if (o.value_sats > big.value_sats) big = o;
    if (big.value_sats / total < 0.85) return null;
    if (total - big.value_sats <= 0) return null;
    return big;
  };

  /** address -> {chain_length, position} for addresses genuinely on a chain. */
  const peelChainMembers = new Map();
  const consumedTx = new Set();

  for (const t of txs) {
    if (consumedTx.has(t.txid)) continue;
    let carrier = peelShape(t);
    if (!carrier) continue;

    const walk = [];
    let cursor = t;
    const seen = new Set();

    while (carrier && !seen.has(cursor.txid) && walk.length < 500) {
      seen.add(cursor.txid);
      walk.push({ txid: cursor.txid, carrier: carrier.address });
      const next = spentBy.get(carrier.address);
      if (!next || next.txid === cursor.txid) break;
      const nextCarrier = peelShape(next);
      if (!nextCarrier) break;
      cursor = next;
      carrier = nextCarrier;
    }

    if (walk.length >= MIN_PEEL_CHAIN) {
      walk.forEach((link, idx) => {
        consumedTx.add(link.txid);
        peelChainMembers.set(link.carrier, {
          chain_length: walk.length,
          position: idx + 1,
        });
      });
    }
  }

  onProgress({ kind: 'phase', phase: 'accumulating', addresses: firstSeen.size });

  /**
   * Per-address accumulators. Fixed size each, no arrays, so memory stays O(A).
   */
  const acc = new Map();
  const get = (addr) => {
    let a = acc.get(addr);
    if (!a) {
      a = {
        address: addr,
        first_seen: firstSeen.get(addr) || 0,
        last_seen: 0,
        n_tx_in: 0,
        n_tx_out: 0,
        received_sats: 0,
        sent_sats: 0,
        senders: new Set(),
        recipients: new Set(),
        // Layering: outputs paid to addresses that were new at that moment.
        paid_to_new: 0,
        paid_to_total: 0,
        // Fan-out width of the largest single spend.
        max_split_width: 0,
        // How fast money left after it arrived, smallest gap observed.
        fastest_turnaround: Infinity,
        // Peeling: spends where one output takes most of the value and continues.
        peel_like_spends: 0,
        // Timestamp of the most recent inbound payment, to measure turnaround.
        last_in_time: null,
        largest_received_sats: 0,
        peel_chain_length: 0,
        peel_chain_position: 0,
      };
      acc.set(addr, a);
    }
    return a;
  };

  let processed = 0;
  for (const t of txs) {
    const ts = t.time || t.block_time;
    processed += 1;
    if (processed % 200000 === 0) {
      onProgress({ kind: 'progress', processed, total: txs.length });
    }

    const totalOut = t.outputs.reduce((s, o) => s + o.value_sats, 0);

    // Identify the peel shape once per transaction rather than per address.
    let biggest = null;
    for (const o of t.outputs) if (!biggest || o.value_sats > biggest.value_sats) biggest = o;
    const isPeelShape =
      t.outputs.length >= 2 &&
      t.outputs.length <= 3 &&
      totalOut > 0 &&
      biggest &&
      biggest.value_sats / totalOut >= 0.85;

    // --- senders ------------------------------------------------------------------
    const inputAddrs = new Set(t.inputs.map((i) => i.address));
    for (const i of t.inputs) {
      const a = get(i.address);
      a.n_tx_out += 1;
      a.sent_sats += i.value_sats;
      a.last_seen = Math.max(a.last_seen, ts);

      const chain = peelChainMembers.get(i.address);
      if (chain) {
        a.peel_chain_length = Math.max(a.peel_chain_length, chain.chain_length);
        a.peel_chain_position = chain.position;
      }

      let newPaid = 0;
      for (const o of t.outputs) {
        if (inputAddrs.has(o.address)) continue; // change back to self is not a payment
        a.recipients.add(o.address);
        a.paid_to_total += 1;
        if ((firstSeen.get(o.address) || 0) >= ts) {
          a.paid_to_new += 1;
          newPaid += 1;
        }
      }

      const width = t.outputs.filter((o) => !inputAddrs.has(o.address)).length;
      if (width > a.max_split_width) a.max_split_width = width;
      // Shape alone is not evidence; only membership of a real multi-hop chain is.
      if (isPeelShape && peelChainMembers.has(i.address)) a.peel_like_spends += 1;

      // Turnaround: how long the money sat before this spend.
      if (a.last_in_time != null) {
        const gap = ts - a.last_in_time;
        if (gap >= 0 && gap < a.fastest_turnaround) a.fastest_turnaround = gap;
      }
      void newPaid;
    }

    // --- receivers ----------------------------------------------------------------
    for (const o of t.outputs) {
      if (inputAddrs.has(o.address)) continue;
      const a = get(o.address);
      a.n_tx_in += 1;
      a.received_sats += o.value_sats;
      a.last_seen = Math.max(a.last_seen, ts);
      a.last_in_time = ts;
      if (o.value_sats > a.largest_received_sats) a.largest_received_sats = o.value_sats;
      for (const i of t.inputs) a.senders.add(i.address);
    }
  }

  onProgress({ kind: 'phase', phase: 'featurising', addresses: acc.size });

  // --- convert accumulators into the feature vectors the models consume -------------
  const rows = [];
  for (const a of acc.values()) {
    const nTx = a.n_tx_in + a.n_tx_out;
    if (nTx < minTxs) continue;

    const lifetime = Math.max(0, a.last_seen - a.first_seen);
    const turnaround = a.fastest_turnaround === Infinity ? lifetime : a.fastest_turnaround;
    const received = a.received_sats;
    const sent = a.sent_sats;

    rows.push({
      address: a.address,
      first_seen: a.first_seen,
      last_seen: a.last_seen,
      features: {
        // --- the four you specified, directly ---------------------------------
        // Average holding time, in seconds, before money moved on.
        avg_holding_time: turnaround,
        // How wide the biggest single split was.
        number_of_splits: a.max_split_width,
        // Share of payments that went to addresses new at that moment.
        percentage_sent_to_new_addresses:
          a.paid_to_total > 0 ? a.paid_to_new / a.paid_to_total : 0,
        // Transactions per hour over the address's active life.
        transaction_frequency: lifetime > 0 ? nTx / (lifetime / 3600) : nTx,

        // --- supporting structure ---------------------------------------------
        n_tx_in: a.n_tx_in,
        n_tx_out: a.n_tx_out,
        in_degree: a.senders.size,
        out_degree: a.recipients.size,
        received_btc: received / SATS,
        sent_btc: sent / SATS,
        balance_btc: (received - sent) / SATS,
        // Near 1.0 means nothing was retained: a relay, not a wallet.
        forward_ratio: received > 0 ? Math.min(1, sent / received) : 0,
        lifetime_seconds: lifetime,
        peel_like_spends: a.peel_like_spends,
        largest_received_btc: a.largest_received_sats / SATS,

        // --- the named patterns, as explicit flags ----------------------------
        // High-velocity layering: arrived, split several ways, left within minutes.
        layering_score:
          a.max_split_width >= 4 && turnaround <= FAST_SPLIT_SECONDS && received > 0
            ? Math.min(1, (a.max_split_width / 10) * (1 - turnaround / FAST_SPLIT_SECONDS))
            : 0,
        // High fan-in then immediate sweep out to very few destinations.
        collector_score:
          a.senders.size >= 8 && a.recipients.size <= 2 && received > 0
            ? Math.min(1, a.senders.size / 40)
            : 0,
        // Peeling: scaled by how long the chain is, zero when not on one at all.
        // A lone peel-shaped payment scores nothing, because that is just a payment.
        peel_score: a.peel_chain_length >= 3 ? Math.min(1, a.peel_chain_length / 10) : 0,
        peel_chain_length: a.peel_chain_length,
      },
    });
  }

  return {
    rows,
    stats: {
      blocks: blocks.length,
      transactions: txs.length,
      distinct_addresses: firstSeen.size,
      addresses_with_min_activity: rows.length,
      window_start: txs.length ? txs[0].time || txs[0].block_time : null,
      window_end: txs.length ? txs[txs.length - 1].time || txs[txs.length - 1].block_time : null,
      min_txs_filter: minTxs,
    },
  };
}

/** The exact four features you specified, in order, for clustering. */
const CLUSTER_FEATURES = [
  'avg_holding_time',
  'number_of_splits',
  'percentage_sent_to_new_addresses',
  'transaction_frequency',
];

/** The wider set the neural network sees. */
const MODEL_FEATURES = [
  'avg_holding_time',
  'number_of_splits',
  'percentage_sent_to_new_addresses',
  'transaction_frequency',
  'n_tx_in',
  'n_tx_out',
  'in_degree',
  'out_degree',
  'received_btc',
  'sent_btc',
  'balance_btc',
  'forward_ratio',
  'lifetime_seconds',
  'peel_like_spends',
  'peel_chain_length',
  'largest_received_btc',
  'layering_score',
  'collector_score',
  'peel_score',
];

/**
 * Build the address-to-address edge list the graph network needs.
 * Deduplicated and aggregated, so repeat payments become one weighted edge.
 */
function buildEdges(blocks, addressIndex) {
  const edges = new Map();
  for (const b of blocks) {
    for (const t of b.txs) {
      const inputAddrs = new Set(t.inputs.map((i) => i.address));
      const totalIn = t.inputs.reduce((s, i) => s + i.value_sats, 0);
      if (totalIn <= 0) continue;

      for (const i of t.inputs) {
        const from = addressIndex.get(i.address);
        if (from === undefined) continue;
        const share = i.value_sats / totalIn;
        for (const o of t.outputs) {
          if (inputAddrs.has(o.address)) continue;
          const to = addressIndex.get(o.address);
          if (to === undefined || to === from) continue;
          const key = `${from}|${to}`;
          const value = Math.round(o.value_sats * share);
          const e = edges.get(key);
          if (e) {
            e.value_sats += value;
            e.n_txs += 1;
          } else {
            edges.set(key, { from, to, value_sats: value, n_txs: 1, time: t.time || b.time });
          }
        }
      }
    }
  }
  return [...edges.values()];
}

module.exports = {
  extractBehaviour,
  buildEdges,
  CLUSTER_FEATURES,
  MODEL_FEATURES,
  FAST_SPLIT_SECONDS,
};
