'use strict';
/**
 * The endpoint tracer. Follow the money until it stops being followable.
 *
 * This is the feature that separates a graph viewer from an investigative tool. Showing
 * an analyst a cloud of connected dots does not help them. Telling them "this money ended
 * up at a deposit address belonging to a regulated exchange on 14 June, here are the
 * three transactions that got it there" is the entire job.
 *
 *
 * WHY THIS IS NOT BREADTH-FIRST SEARCH
 *
 * Blind breadth-first traversal fails past about three hops, and not because it is slow.
 * Counts roughly multiply by ten per hop, so five hops is tens of thousands of addresses
 * and ten hops is a meaningful fraction of the economy. Almost all of it is irrelevant:
 * dust, change, and unrelated traffic belonging to people who happened to transact
 * nearby. Returning all of it and calling it a result is how an analyst ends up staring
 * at a hairball.
 *
 * So this search is ordered by money, not by distance. A priority queue always expands
 * whichever branch still carries the largest share of the original sum, branches carrying
 * a negligible fraction are dropped, and the search stops when the money does. Following
 * the biggest remaining flow first is also simply what a human investigator does.
 *
 *
 * WHY A PEEL CHAIN DOES NOT COST A HOP
 *
 * A peel chain forwards almost everything to a fresh address and sheds a little at each
 * step, hundreds of times. Counted naively that is hundreds of hops, so any sane hop
 * limit is exhausted inside a single laundering technique and the trail appears to die.
 * But a peel chain is one *action*, not hundreds, so the tracer detects the shape, walks
 * it to its end, and charges it as a single logical step. This is the specific reason a
 * three-hop limit was never finding anything.
 *
 *
 * WHERE THE MONEY CAN ACTUALLY STOP
 *
 * There are only four outcomes, and naming which one applies is the deliverable:
 *
 *   SERVICE   The funds reached an address belonging to an exchange or custodial
 *             business. This is the win. That business is legally required to hold
 *             identity documents for whoever controls the account, so this is the point
 *             at which a court order can turn a pseudonym into a person.
 *
 *   DORMANT   Received and never spent. The trail is not lost, it is parked, and the
 *             money is unusable to its holder until it moves.
 *
 *   MIXER     The funds entered a structure built to break traceability. The trail
 *             genuinely ends, and saying so is more useful than guessing.
 *
 *   HORIZON   The search ran out of hops, budget, or loaded data. This is not a
 *             conclusion, it is an unfinished trace, and it is labelled as such so
 *             nobody mistakes it for a dead end.
 *
 *
 * WHAT THIS CANNOT DO
 *
 * It cannot name a person. Nothing that reads the blockchain can, because the blockchain
 * contains no names. It finds the doorway where the off-chain world holds that name, and
 * assembles the evidence needed to justify asking for it.
 */

const { humanBtc, shortAddress } = require('../chain/model');

const DEFAULTS = {
  /** Logical hops. A peel chain, however long, costs one of these. */
  maxHops: 6,

  /** Hard cap on address lookups. Each is a network request when not cached. */
  fetchBudget: 120,

  /** Drop a branch once it carries less than this many satoshis. 100k sats is 0.001 BTC. */
  minValueSats: 100000,

  /**
   * Drop a branch once it carries less than this fraction of the starting sum.
   * Absolute and relative floors do different jobs: the absolute one removes dust, the
   * relative one stops a trace of 500 BTC from chasing a 0.002 BTC crumb for an hour.
   *
   * This was 0.005 and that was badly wrong. On a 3,205 BTC source it put the floor at
   * 16 BTC per branch, so a wallet that fanned out into four hundred branches of a few
   * BTC each had every single one pruned: the tracer examined two addresses, made zero
   * network requests and reported "no endpoint reached". Laundering deliberately splits
   * value into pieces that are small relative to the original sum, so a floor that scales
   * with the source prunes exactly the behaviour being hunted.
   */
  minValueFraction: 0.0002,

  /**
   * Hard ceiling on the computed floor, regardless of how large the source is. Without
   * this the relative floor grows without limit and reintroduces the same failure on a
   * large enough wallet. 0.02 BTC is small enough to be genuine dust at any scale.
   */
  maxFloorSats: 2000000,

  /** An address with at least this many transactions is behaving like a service. */
  serviceMinTxs: 1000,

  /** Follow at most this many onward branches from one address. */
  maxBranchesPerAddress: 12,

  /** An output taking at least this share of a transaction is the peel chain continuing. */
  peelRetainRatio: 0.85,
  peelMaxOutputs: 3,
  peelMaxLength: 200,
};

/* ------------------------------------------------------------------------------------ */
/* Priority queue                                                                        */
/* ------------------------------------------------------------------------------------ */

/**
 * Max-heap ordered by satoshis carried. A sorted array would be O(n) per insert and this
 * queue takes an insert for every branch of every address examined.
 */
class ValueQueue {
  constructor() {
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  push(item) {
    this.items.push(item);
    let i = this.items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.items[parent].value_sats >= this.items[i].value_sats) break;
      [this.items[parent], this.items[i]] = [this.items[i], this.items[parent]];
      i = parent;
    }
  }

  pop() {
    if (this.items.length === 0) return null;
    const top = this.items[0];
    const last = this.items.pop();
    if (this.items.length > 0) {
      this.items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let big = i;
        if (l < this.items.length && this.items[l].value_sats > this.items[big].value_sats) big = l;
        if (r < this.items.length && this.items[r].value_sats > this.items[big].value_sats) big = r;
        if (big === i) break;
        [this.items[big], this.items[i]] = [this.items[i], this.items[big]];
        i = big;
      }
    }
    return top;
  }
}

/* ------------------------------------------------------------------------------------ */
/* Address classification                                                                */
/* ------------------------------------------------------------------------------------ */

/**
 * Decide what kind of address this is, and therefore whether the trail continues.
 *
 * The transaction count does most of the work and it is a genuinely strong signal on real
 * data. An ordinary person's wallet has tens of transactions. An exchange hot wallet,
 * which sweeps every customer deposit into one place, has tens or hundreds of thousands.
 * There is no overlap in the middle worth worrying about.
 *
 * A supplied label list always wins over this inference, because a maintained list is
 * evidence and a structural guess is a lead.
 */
function classifyAddress(record, ctx) {
  const labels = ctx.labels || new Map();
  const opt = ctx.opt;

  const labelled = labels.get(record.address);
  if (labelled) {
    const t = String(labelled.type || '').toLowerCase();
    if (['exchange', 'service', 'custodial', 'payment_processor'].includes(t)) {
      return {
        kind: 'SERVICE',
        confidence: labelled.confidence == null ? 0.95 : labelled.confidence,
        basis: 'label_list',
        detail: `Identified as ${labelled.type}${labelled.name ? ` (${labelled.name})` : ''} by ${labelled.source || 'a supplied label list'}.`,
        label: labelled,
      };
    }
    if (t === 'mixer' || t === 'tumbler' || t === 'coinjoin') {
      return {
        kind: 'MIXER',
        confidence: labelled.confidence == null ? 0.95 : labelled.confidence,
        basis: 'label_list',
        detail: `Identified as ${labelled.type} by ${labelled.source || 'a supplied label list'}.`,
        label: labelled,
      };
    }
  }

  if (ctx.sanctioned && ctx.sanctioned.has(record.address)) {
    return {
      kind: 'SANCTIONED',
      confidence: 1,
      basis: 'ofac_sdn',
      detail: 'Listed on the US Treasury sanctions list. Follow the money past it, but flag it.',
    };
  }

  const nTx = record.n_tx || 0;
  if (nTx >= opt.serviceMinTxs) {
    return {
      kind: 'SERVICE',
      confidence: Math.min(0.9, 0.55 + Math.log10(nTx / opt.serviceMinTxs) * 0.2),
      basis: 'transaction_volume',
      detail:
        `${nTx.toLocaleString()} transactions on one address. Ordinary wallets have tens. ` +
        'This volume means an address operated by a business sweeping many customers, ' +
        'which is what an exchange hot wallet looks like. Unconfirmed without a label list.',
    };
  }

  // Received something and never spent any of it.
  if ((record.total_received_sats || 0) > 0 && (record.total_sent_sats || 0) === 0) {
    return {
      kind: 'DORMANT',
      confidence: 0.9,
      basis: 'never_spent',
      detail:
        `Holds ${humanBtc(record.balance_sats)} that has never moved. The trail is not lost, ` +
        'it is parked. Money that never reaches an exit is money its holder cannot spend.',
    };
  }

  return { kind: 'CONTINUE', confidence: 0, basis: null, detail: null };
}

/**
 * Does this transaction look like one step of a peel chain, and if so which output
 * continues it?
 */
function peelStep(tx, opt) {
  if (tx.outputs.length < 2 || tx.outputs.length > opt.peelMaxOutputs) return null;
  const total = tx.total_out_sats;
  if (total <= 0) return null;

  let big = tx.outputs[0];
  for (const o of tx.outputs) if (o.value_sats > big.value_sats) big = o;

  if (big.value_sats / total < opt.peelRetainRatio) return null;
  const peeled = total - big.value_sats;
  if (peeled <= 0) return null;

  return { carrier: big, peeled_sats: peeled };
}

/* ------------------------------------------------------------------------------------ */
/* The trace                                                                             */
/* ------------------------------------------------------------------------------------ */

/**
 * Follow value forward from one address until it stops.
 *
 * @param {ChainClient} client
 * @param {string} sourceAddress
 * @param {object} options
 * @returns {Promise<object>} the trace result, including named endpoints and evidence
 */
async function traceToEndpoint(client, sourceAddress, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  const onProgress = options.onProgress || (() => {});
  const ctx = {
    opt,
    labels: options.labels || new Map(),
    sanctioned: options.sanctioned || new Set(),
  };

  const started = Date.now();

  const source = await client.addressTransactions(sourceAddress);
  if (!source.fetched && source.offline) {
    return {
      ok: false,
      source_address: sourceAddress,
      error: 'This address is not in the local cache and the tracer is offline.',
    };
  }

  const startValue = source.total_received_sats || 0;
  if (startValue === 0) {
    return {
      ok: true,
      source_address: sourceAddress,
      start_value_sats: 0,
      endpoints: [],
      verdict: 'This address has never received any Bitcoin, so there is nothing to trace.',
      stats: { addresses_examined: 0, fetches: 0 },
    };
  }

  const floorSats = Math.max(
    opt.minValueSats,
    Math.min(Math.floor(startValue * opt.minValueFraction), opt.maxFloorSats)
  );

  const queue = new ValueQueue();
  queue.push({ address: sourceAddress, value_sats: startValue, hops: 0, path: [], record: source });

  /** Best value at which each address has already been expanded. */
  const seen = new Map();
  const endpoints = [];
  const valueByHop = new Map();
  const peelChains = [];

  let fetches = 0;
  let examined = 0;
  let prunedValue = 0;
  let prunedBranches = 0;

  while (queue.size > 0 && fetches < opt.fetchBudget) {
    const node = queue.pop();

    if (node.value_sats < floorSats) {
      prunedBranches += 1;
      prunedValue += node.value_sats;
      continue;
    }

    const already = seen.get(node.address);
    if (already != null && already >= node.value_sats) continue;
    seen.set(node.address, node.value_sats);

    // Fetch the address unless the queue entry already carries its record.
    let record = node.record;
    if (!record) {
      const wasCached = client.cache.has('addr', node.address);
      try {
        record = await client.addressTransactions(node.address);
      } catch (err) {
        endpoints.push({
          kind: 'ERROR',
          address: node.address,
          hops: node.hops,
          value_sats: node.value_sats,
          detail: `Could not retrieve this address: ${err.message}`,
          path: node.path,
        });
        continue;
      }
      if (!wasCached) fetches += 1;
      onProgress({
        kind: 'examine',
        address: node.address,
        hops: node.hops,
        value_sats: node.value_sats,
        fetches,
        budget: opt.fetchBudget,
      });
    }

    examined += 1;
    valueByHop.set(node.hops, (valueByHop.get(node.hops) || 0) + node.value_sats);

    /**
     * How much of this address's money is ours to follow.
     *
     * Without this the tracer commits the single worst error in fund tracing. Suppose
     * 0.2 BTC of traced money lands in a wallet that already holds 400 BTC of unrelated
     * funds, and that wallet then pays out 400 BTC. Crediting the whole 400 BTC to this
     * trail would claim the suspect moved four hundred Bitcoin when they moved a fifth
     * of one, and every downstream number inherits the error. It is also how an innocent
     * third party's entire balance gets reported as criminal proceeds.
     *
     * The haircut convention fixes it: if a fifth of a percent of what this address
     * received was ours, then a fifth of a percent of everything it sends on is ours.
     * Proportional, conservative, and it keeps the total traced value bounded by where
     * it started, which is the property that makes the percentages mean anything.
     */
    const receivedTotal = Math.max(1, record.total_received_sats || node.value_sats);
    const taintRatio = Math.min(1, node.value_sats / receivedTotal);

    // --- has the money stopped here? ---------------------------------------------
    const verdict = classifyAddress(record, ctx);
    if (verdict.kind !== 'CONTINUE' && node.hops > 0) {
      endpoints.push({
        kind: verdict.kind,
        address: node.address,
        hops: node.hops,
        value_sats: node.value_sats,
        share_of_source: node.value_sats / startValue,
        confidence: verdict.confidence,
        basis: verdict.basis,
        detail: verdict.detail,
        label: verdict.label || null,
        n_tx: record.n_tx,
        balance_sats: record.balance_sats,
        path: node.path,
        // The address that paid a service is the one the service holds records for.
        deposit_candidate:
          verdict.kind === 'SERVICE' && node.path.length > 0
            ? node.path[node.path.length - 1].from
            : null,
      });
      // A service or a mixer absorbs the money; expanding past it would trace the
      // exchange's own business, not the suspect's.
      if (verdict.kind === 'SERVICE' || verdict.kind === 'MIXER' || verdict.kind === 'DORMANT') {
        continue;
      }
    }

    if (node.hops >= opt.maxHops) {
      endpoints.push({
        kind: 'HORIZON',
        address: node.address,
        hops: node.hops,
        value_sats: node.value_sats,
        share_of_source: node.value_sats / startValue,
        confidence: 0,
        basis: 'hop_limit',
        detail:
          `Still carrying ${humanBtc(node.value_sats)} when the ${opt.maxHops}-hop limit was ` +
          'reached. The trail is unfinished, not ended. Raise the hop limit to continue.',
        n_tx: record.n_tx,
        path: node.path,
      });
      continue;
    }

    // --- follow the money onward ---------------------------------------------------
    const spending = record.txs.filter((tx) => tx.inputs.some((i) => i.address === node.address));
    if (spending.length === 0) {
      // Received, holds a balance, but no outgoing transaction is visible. Usually the
      // page limit cut the history off rather than the money genuinely sitting still.
      if (record.truncated) {
        endpoints.push({
          kind: 'HORIZON',
          address: node.address,
          hops: node.hops,
          value_sats: node.value_sats,
          share_of_source: node.value_sats / startValue,
          confidence: 0,
          basis: 'history_truncated',
          detail:
            `This address has ${record.n_tx.toLocaleString()} transactions and only ` +
            `${record.n_tx_fetched} were retrieved, so its outgoing payments may not be loaded.`,
          path: node.path,
        });
      }
      continue;
    }

    const branches = [];

    for (const tx of spending) {
      const spentHere = tx.inputs
        .filter((i) => i.address === node.address)
        .reduce((s, i) => s + i.value_sats, 0);
      if (spentHere <= 0 || tx.total_out_sats <= 0) continue;

      // Walk a peel chain to its end rather than charging a hop for every link.
      const peel = peelStep(tx, opt);
      let chainLength = 0;
      let current = tx;
      let carrierAddress = null;
      let carrierValue = 0;
      let peeledTotal = 0;

      if (peel && peel.carrier.address !== node.address) {
        let step = peel;
        let cursor = tx;
        const chain = [];

        while (step && chainLength < opt.peelMaxLength) {
          chain.push({
            txid: cursor.txid,
            time: cursor.time,
            block_height: cursor.block_height,
            carrier: step.carrier.address,
            carried_sats: step.carrier.value_sats,
            peeled_sats: step.peeled_sats,
          });
          peeledTotal += step.peeled_sats;
          chainLength += 1;
          carrierAddress = step.carrier.address;
          carrierValue = step.carrier.value_sats;

          // Continuing the chain needs the carrier's own history, which costs a fetch.
          if (chainLength >= 2 && fetches >= opt.fetchBudget) break;
          if (!client.cache.has('addr', carrierAddress)) {
            if (fetches >= opt.fetchBudget) break;
            try {
              await client.addressTransactions(carrierAddress);
              fetches += 1;
            } catch {
              break;
            }
          }
          const next = client.cache.get('addr', carrierAddress);
          if (!next) break;

          // Stop unwinding the chain if the carrier is itself an endpoint.
          const carrierVerdict = classifyAddress(next, ctx);
          if (carrierVerdict.kind !== 'CONTINUE') break;

          cursor = next.txs.find(
            (t) => t.inputs.some((i) => i.address === carrierAddress) && t.txid !== cursor.txid
          );
          if (!cursor) break;
          step = peelStep(cursor, opt);
        }

        if (chainLength >= 3) {
          peelChains.push({
            started_at: tx.txid,
            length: chainLength,
            ended_at: carrierAddress,
            peeled_sats: peeledTotal,
            links: chain.slice(0, 20),
          });
          // A peel chain carries our share of the value, not the carrier's whole balance.
          const carried = Math.round(Math.min(spentHere, carrierValue) * taintRatio);
          branches.push({
            address: carrierAddress,
            value_sats: carried,
            hopCost: 1,
            edge: {
              txid: tx.txid,
              time: tx.time,
              block_height: tx.block_height,
              from: node.address,
              to: carrierAddress,
              value_sats: carried,
              observed_sats: Math.min(spentHere, carrierValue),
              taint_ratio: +taintRatio.toFixed(6),
              via: 'peel_chain',
              peel_length: chainLength,
              peeled_sats: peeledTotal,
            },
          });
          continue;
        }
      }

      // Ordinary transaction: allocate what this address contributed across the outputs.
      for (const out of tx.outputs) {
        if (!out.address || out.address === node.address) continue;
        const share = out.value_sats / tx.total_out_sats;
        // What the chain shows moving, then our proportional share of it.
        const observed = Math.round(spentHere * share);
        const forwarded = Math.round(observed * taintRatio);
        if (forwarded < floorSats) {
          prunedBranches += 1;
          prunedValue += forwarded;
          continue;
        }
        branches.push({
          address: out.address,
          value_sats: forwarded,
          hopCost: 1,
          edge: {
            txid: tx.txid,
            time: tx.time,
            block_height: tx.block_height,
            from: node.address,
            to: out.address,
            // What this trail is responsible for.
            value_sats: forwarded,
            // What the transaction actually moved, so the two are never confused and an
            // analyst can reconcile the figure against a block explorer.
            observed_sats: observed,
            taint_ratio: +taintRatio.toFixed(6),
            via: 'transfer',
          },
        });
      }
    }

    // Keep the largest branches. A transaction fanning out to 400 addresses is a payout
    // batch, and following every arm of it exhausts the budget on noise.
    branches.sort((a, b) => b.value_sats - a.value_sats);
    for (const b of branches.slice(0, opt.maxBranchesPerAddress)) {
      queue.push({
        address: b.address,
        value_sats: b.value_sats,
        hops: node.hops + b.hopCost,
        path: [...node.path, b.edge],
      });
    }
    prunedBranches += Math.max(0, branches.length - opt.maxBranchesPerAddress);
  }

  // Anything still queued when the budget ran out is an unfinished trail, not a dead one.
  let unexplored = 0;
  let unexploredValue = 0;
  while (queue.size > 0) {
    const left = queue.pop();
    unexplored += 1;
    unexploredValue += left.value_sats;
  }

  endpoints.sort((a, b) => b.value_sats - a.value_sats);

  return {
    ok: true,
    source_address: sourceAddress,
    start_value_sats: startValue,
    start_value_btc: humanBtc(startValue),
    endpoints,
    peel_chains: peelChains,
    value_by_hop: [...valueByHop.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([hop, sats]) => ({ hop, value_sats: sats, value_btc: humanBtc(sats) })),
    verdict: summarise(endpoints, startValue, unexplored, unexploredValue, opt),
    stats: {
      addresses_examined: examined,
      fetches,
      fetch_budget: opt.fetchBudget,
      budget_exhausted: fetches >= opt.fetchBudget,
      pruned_branches: prunedBranches,
      pruned_value_sats: prunedValue,
      unexplored_branches: unexplored,
      unexplored_value_sats: unexploredValue,
      value_floor_sats: floorSats,
      elapsed_ms: Date.now() - started,
    },
    settings: {
      max_hops: opt.maxHops,
      fetch_budget: opt.fetchBudget,
      min_value_sats: opt.minValueSats,
      service_min_txs: opt.serviceMinTxs,
    },
  };
}

/**
 * Turn the endpoint list into the sentence an investigator actually needs.
 * States what was found, how much of the money it accounts for, and what to do next.
 */
function summarise(endpoints, startValue, unexplored, unexploredValue, opt) {
  if (endpoints.length === 0) {
    return {
      headline: 'No endpoint reached.',
      detail:
        `The trace examined the strongest branches and none of them stopped anywhere ` +
        `identifiable within ${opt.maxHops} hops and ${opt.fetchBudget} lookups. ` +
        'Raise the budget and the hop limit, then run it again.',
      actionable: false,
    };
  }

  const services = endpoints.filter((e) => e.kind === 'SERVICE');
  const dormant = endpoints.filter((e) => e.kind === 'DORMANT');
  const mixers = endpoints.filter((e) => e.kind === 'MIXER');
  const horizon = endpoints.filter((e) => e.kind === 'HORIZON');

  const sum = (list) => list.reduce((s, e) => s + e.value_sats, 0);
  const pct = (sats) => `${((sats / startValue) * 100).toFixed(1)}%`;

  if (services.length > 0) {
    const top = services[0];
    return {
      headline: `Traced to a custodial service holding ${pct(sum(services))} of the funds.`,
      detail:
        `The largest single flow, ${humanBtc(top.value_sats)}, reached ${shortAddress(top.address)} ` +
        `after ${top.hops} hop(s). ${top.detail}` +
        (top.deposit_candidate
          ? ` The address that paid it, ${shortAddress(top.deposit_candidate)}, is the likely ` +
            'customer deposit address, and it is that account the venue holds identity records for.'
          : ''),
      next_step:
        'Prepare a records request for the receiving account at that venue, citing the ' +
        'transaction ids on the path. Identity is not established until those records are returned.',
      actionable: true,
      accounts_for: pct(sum(services)),
    };
  }

  if (mixers.length > 0) {
    return {
      headline: `Funds entered a mixing service (${pct(sum(mixers))} of the total).`,
      detail:
        'The trail genuinely ends here rather than being merely unfinished. Anything ' +
        'claimed about where this money went next would be a guess.',
      next_step:
        'Work the other branches, or look for the same actor re-entering the chain with ' +
        'a matching amount and timing shortly afterwards.',
      actionable: false,
    };
  }

  if (dormant.length > 0 && sum(dormant) > startValue * 0.3) {
    return {
      headline: `${pct(sum(dormant))} of the funds are sitting unspent.`,
      detail:
        `Across ${dormant.length} address(es) the money arrived and never moved again. ` +
        'It has not been cashed out, which means it has not yet been useful to whoever holds it.',
      next_step:
        'Add these addresses to a watch list. The investigative moment is when they move, ' +
        'because that is when they must head for an exit.',
      actionable: true,
    };
  }

  if (horizon.length > 0) {
    return {
      headline: 'The trail is unfinished, not ended.',
      detail:
        `${pct(sum(horizon))} of the money was still moving when the ${opt.maxHops}-hop limit ` +
        `was reached` +
        (unexplored > 0 ? `, with ${unexplored} branch(es) left unexplored.` : '.'),
      next_step: 'Raise the hop limit and the fetch budget, then run the trace again.',
      actionable: false,
    };
  }

  return {
    headline: `${endpoints.length} endpoint(s) found.`,
    detail: 'None of them is a custodial service, so no identity request is available yet.',
    actionable: false,
  };
}

/** A readable, copy-pasteable rendering of one endpoint's route. */
function formatPath(endpoint) {
  const lines = [];
  for (const [i, edge] of (endpoint.path || []).entries()) {
    const via = edge.via === 'peel_chain' ? ` (peel chain, ${edge.peel_length} links)` : '';
    lines.push(
      `  ${String(i + 1).padStart(2)}. ${humanBtc(edge.value_sats).padStart(14)}  ` +
        `${shortAddress(edge.from)} -> ${shortAddress(edge.to)}${via}\n` +
        `      tx ${edge.txid}${edge.block_height ? `  block ${edge.block_height}` : ''}`
    );
  }
  return lines.join('\n');
}

module.exports = { traceToEndpoint, classifyAddress, peelStep, ValueQueue, formatPath, DEFAULTS };
