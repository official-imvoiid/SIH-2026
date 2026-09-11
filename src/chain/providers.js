'use strict';
/**
 * Block explorer providers, and rotating between them.
 *
 * A single public explorer is not a dependency you can build on. Measured from this
 * machine within a few minutes of each other: mempool.space did not respond at all,
 * blockchair returned "your IP address is temporarily blocked", blockchain.info served
 * requests happily and then rate-limited the address after a few dozen, and
 * blockstream.info refused the first request and worked perfectly an hour later. None of
 * them is reliable alone, and all of them are free, which is the whole explanation.
 *
 * So the client holds several providers and moves to the next one when the current one
 * refuses. A provider that returns 429 is put in a penalty box with an expiry rather than
 * abandoned, because these limits are temporary by design. Work continues as long as any
 * provider is willing, and only stops when all of them refuse.
 *
 * Esplora is preferred over blockchain.info despite both working, for one specific
 * reason: it returns the block hash alongside the height. Provenance requires being able
 * to say which block a record came from, and a height alone can be ambiguous across a
 * chain reorganisation.
 */

const { makeTransaction, provenance, SOURCES } = require('./model');

/* ------------------------------------------------------------------------------------ */
/* Esplora (blockstream.info, and any compatible mirror)                                 */
/* ------------------------------------------------------------------------------------ */

/**
 * Esplora reports every value in satoshis already, so nothing here converts or rounds.
 * That is the format this project wants, and it is why Esplora is the primary provider.
 */
function parseEsploraTx(t, sourceName) {
  const inputs = [];
  for (const vin of t.vin || []) {
    // A coinbase input has no previous output to spend. Dropping it leaves an empty
    // input list, which is the correct representation of newly created value.
    if (vin.is_coinbase || !vin.prevout || !vin.prevout.scriptpubkey_address) continue;
    inputs.push({
      address: vin.prevout.scriptpubkey_address,
      value_sats: Number(vin.prevout.value) || 0,
      prev_txid: vin.txid || null,
      prev_vout: vin.vout == null ? null : Number(vin.vout),
    });
  }

  const outputs = [];
  (t.vout || []).forEach((o, i) => {
    // Outputs with no address are data carriers such as OP_RETURN. They hold no
    // spendable value, so they are not part of the money graph.
    if (!o.scriptpubkey_address) return;
    outputs.push({
      n: i,
      address: o.scriptpubkey_address,
      value_sats: Number(o.value) || 0,
    });
  });

  const status = t.status || {};
  return makeTransaction(
    {
      txid: t.txid,
      block_height: status.block_height == null ? null : Number(status.block_height),
      block_hash: status.block_hash || null,
      time: Number(status.block_time) || 0,
      inputs,
      outputs,
    },
    provenance(SOURCES.EXPLORER, {
      endpoint: sourceName,
      api: 'esplora',
      raw_available: true,
      confirmed: Boolean(status.confirmed),
    })
  );
}

function esploraProvider(name, base) {
  return {
    name,
    api: 'esplora',
    summaryUrl: (addr) => `${base}/address/${addr}`,
    // Esplora pages by "give me what comes after this txid" rather than by offset.
    txsUrl: (addr, cursor) =>
      cursor ? `${base}/address/${addr}/txs/chain/${cursor}` : `${base}/address/${addr}/txs`,
    txUrl: (txid) => `${base}/tx/${txid}`,
    tipUrl: `${base}/blocks/tip/height`,

    parseSummary(json) {
      const chain = json.chain_stats || {};
      const pool = json.mempool_stats || {};
      const received = (Number(chain.funded_txo_sum) || 0) + (Number(pool.funded_txo_sum) || 0);
      const sent = (Number(chain.spent_txo_sum) || 0) + (Number(pool.spent_txo_sum) || 0);
      return {
        n_tx: (Number(chain.tx_count) || 0) + (Number(pool.tx_count) || 0),
        total_received_sats: received,
        total_sent_sats: sent,
        balance_sats: received - sent,
      };
    },

    parseTxs(json) {
      const list = Array.isArray(json) ? json : [];
      return {
        txs: list.map((t) => parseEsploraTx(t, name)),
        // The cursor for the next page is the last txid on this one.
        cursor: list.length ? list[list.length - 1].txid : null,
        pageSize: list.length,
      };
    },

    parseTx(json) {
      return parseEsploraTx(json, name);
    },
  };
}

/* ------------------------------------------------------------------------------------ */
/* blockchain.info                                                                       */
/* ------------------------------------------------------------------------------------ */

function parseBlockchainInfoTx(raw, sourceName) {
  const inputs = [];
  for (const vin of raw.inputs || []) {
    const prev = vin.prev_out;
    if (!prev || !prev.addr) continue;
    inputs.push({
      address: prev.addr,
      value_sats: Number(prev.value) || 0,
      prev_txid: null,
      prev_vout: prev.n == null ? null : Number(prev.n),
    });
  }

  const outputs = [];
  for (const out of raw.out || []) {
    if (!out.addr) continue;
    outputs.push({
      n: Number(out.n) || 0,
      address: out.addr,
      value_sats: Number(out.value) || 0,
      spent: out.spent == null ? null : Boolean(out.spent),
    });
  }

  return makeTransaction(
    {
      txid: raw.hash,
      block_height: raw.block_height == null ? null : Number(raw.block_height),
      // This API does not return the block hash. Recorded as missing rather than
      // invented, and it is the reason Esplora is preferred.
      block_hash: null,
      time: Number(raw.time) || 0,
      inputs,
      outputs,
    },
    provenance(SOURCES.EXPLORER, {
      endpoint: sourceName,
      api: 'blockchain.info',
      raw_available: true,
      block_hash_available: false,
    })
  );
}

const blockchainInfoProvider = {
  name: 'blockchain.info',
  api: 'blockchain.info',
  // One call returns both the summary and the transactions.
  summaryUrl: null,
  txsUrl: (addr, cursor) =>
    `https://blockchain.info/rawaddr/${addr}?limit=50&offset=${cursor || 0}`,
  txUrl: (txid) => `https://blockchain.info/rawtx/${txid}`,
  tipUrl: 'https://blockchain.info/q/getblockcount',

  parseTxs(json) {
    const list = json.txs || [];
    return {
      txs: list.map((t) => parseBlockchainInfoTx(t, 'blockchain.info')),
      cursor: null, // filled in by the caller, which knows the running offset
      pageSize: list.length,
      summary: {
        n_tx: Number(json.n_tx) || list.length,
        total_received_sats: Number(json.total_received) || 0,
        total_sent_sats: Number(json.total_sent) || 0,
        balance_sats: (Number(json.total_received) || 0) - (Number(json.total_sent) || 0),
      },
    };
  },

  parseTx(json) {
    return parseBlockchainInfoTx(json, 'blockchain.info');
  },
};

/* ------------------------------------------------------------------------------------ */
/* The rotation                                                                          */
/* ------------------------------------------------------------------------------------ */

/**
 * Ordered best-first. Esplora providers lead because they carry block hashes; the
 * blockchain.info entry is the fallback that keeps a trace alive when they refuse.
 */
const PROVIDERS = [
  esploraProvider('blockstream.info', 'https://blockstream.info/api'),
  esploraProvider('mempool.space', 'https://mempool.space/api'),
  blockchainInfoProvider,
];

/**
 * Tracks which providers are currently willing to answer.
 *
 * A rate limit is temporary, so a refusing provider is benched with an expiry and tried
 * again later rather than being written off for the session. The penalty grows with
 * repeated refusals, because a provider that has just refused twice is likely to refuse
 * a third time and every wasted request costs real seconds.
 */
class ProviderPool {
  constructor(providers = PROVIDERS, opt = {}) {
    this.providers = providers.map((p) => ({
      provider: p,
      benchedUntil: 0,
      refusals: 0,
      successes: 0,
      failures: 0,
    }));
    this.basePenaltyMs = opt.basePenaltyMs || 60000;
    this.maxPenaltyMs = opt.maxPenaltyMs || 900000;
  }

  /** Providers willing to answer right now, best first. */
  available() {
    const now = Date.now();
    return this.providers.filter((s) => s.benchedUntil <= now);
  }

  bench(name, reason) {
    const slot = this.providers.find((s) => s.provider.name === name);
    if (!slot) return;
    slot.refusals += 1;
    const penalty = Math.min(this.maxPenaltyMs, this.basePenaltyMs * 2 ** (slot.refusals - 1));
    slot.benchedUntil = Date.now() + penalty;
    slot.lastReason = reason;
    return penalty;
  }

  succeed(name) {
    const slot = this.providers.find((s) => s.provider.name === name);
    if (!slot) return;
    slot.successes += 1;
    // A success means the limit has lifted, so clear the escalation rather than letting
    // one bad patch penalise a provider for the rest of the session.
    slot.refusals = 0;
    slot.benchedUntil = 0;
  }

  fail(name) {
    const slot = this.providers.find((s) => s.provider.name === name);
    if (slot) slot.failures += 1;
  }

  /** How long until the soonest provider becomes available again. */
  msUntilAnyAvailable() {
    const now = Date.now();
    const soonest = Math.min(...this.providers.map((s) => s.benchedUntil));
    return Math.max(0, soonest - now);
  }

  status() {
    const now = Date.now();
    return this.providers.map((s) => ({
      name: s.provider.name,
      api: s.provider.api,
      available: s.benchedUntil <= now,
      benched_for_ms: Math.max(0, s.benchedUntil - now),
      successes: s.successes,
      refusals: s.refusals,
      failures: s.failures,
      last_reason: s.lastReason || null,
    }));
  }
}

module.exports = {
  PROVIDERS,
  ProviderPool,
  esploraProvider,
  blockchainInfoProvider,
  parseEsploraTx,
  parseBlockchainInfoTx,
};
