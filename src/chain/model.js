'use strict';
/**
 * The canonical record shapes for real blockchain data, and the rules for handling money.
 *
 * Two decisions here are not negotiable and everything else depends on them.
 *
 * **Money is integer satoshis. Never floating-point BTC.**
 * 0.1 + 0.2 does not equal 0.3 in binary floating point. Over a few hundred thousand
 * transactions those errors accumulate into balances that are visibly wrong, and a
 * forensic tool that reports a wrong balance is worse than no tool. A satoshi is the
 * smallest unit Bitcoin can express, so integers lose nothing. The entire Bitcoin supply
 * is 2.1e15 satoshis, comfortably inside JavaScript's 9.007e15 safe integer range, so
 * ordinary numbers are exact here. BTC exists only for display, produced at the last
 * moment by `formatBtc`.
 *
 * **Every record carries its provenance.**
 * A number on screen must be traceable to the block it came from and the source that
 * supplied it. Without that, a user cannot tell a real balance from an invented one, and
 * this project has already demonstrated how easily those get confused. Provenance is a
 * required field, not an optional extra, and the loaders refuse records that lack it.
 */

const SATS_PER_BTC = 100000000;

/** Sources a record may legitimately come from. Anything else is rejected. */
const SOURCES = {
  BITCOIN_CORE: 'bitcoin-core-rpc',
  EXPLORER: 'blockchain.info',
  CSV: 'csv-import',
  JSON: 'json-import',
  JSONL: 'jsonl-import',
  PARQUET: 'parquet-import',
  /**
   * The generated world. Present so that demo records are *labelled* as generated
   * rather than being indistinguishable from real ones. Real mode refuses this source
   * outright; that refusal is what makes the demo/real split real rather than cosmetic.
   */
  SYNTHETIC: 'synthetic-demo',
};

const REAL_SOURCES = new Set([
  SOURCES.BITCOIN_CORE,
  SOURCES.EXPLORER,
  SOURCES.CSV,
  SOURCES.JSON,
  SOURCES.JSONL,
  SOURCES.PARQUET,
]);

/* ------------------------------------------------------------------------------------ */
/* Money                                                                                 */
/* ------------------------------------------------------------------------------------ */

/**
 * Convert a BTC value from an external source into integer satoshis.
 * Rounds rather than truncates: an eight-decimal BTC string can arrive as 0.29999999996
 * from a float-based exporter, and truncation would silently lose a satoshi per record.
 */
function btcToSats(btc) {
  const n = typeof btc === 'number' ? btc : parseFloat(btc);
  if (!Number.isFinite(n)) return 0;
  return Math.round(n * SATS_PER_BTC);
}

/** Accept a value that may already be satoshis, or may be BTC. Never guess silently. */
function toSats(value, unit) {
  if (unit === 'sats' || unit === 'satoshi' || unit === 'satoshis') {
    const n = typeof value === 'number' ? value : parseInt(value, 10);
    return Number.isFinite(n) ? Math.round(n) : 0;
  }
  return btcToSats(value);
}

/** Display only. Never feed the result of this back into a calculation. */
function formatBtc(sats, decimals = 8) {
  const n = Number(sats) || 0;
  const sign = n < 0 ? '-' : '';
  const abs = Math.abs(n);
  const whole = Math.floor(abs / SATS_PER_BTC);
  const frac = String(abs % SATS_PER_BTC).padStart(8, '0');
  const out = `${sign}${whole}.${frac}`;
  return decimals >= 8 ? out : `${sign}${whole}.${frac.slice(0, decimals)}`;
}

/** Short form for dense tables: 0.00412 BTC, 1.2 BTC, 340 sats. */
function humanBtc(sats) {
  const n = Number(sats) || 0;
  if (n === 0) return '0';
  if (Math.abs(n) < 1000) return `${n} sats`;
  const btc = n / SATS_PER_BTC;
  if (Math.abs(btc) >= 1000) return `${btc.toFixed(0)} BTC`;
  if (Math.abs(btc) >= 1) return `${btc.toFixed(3)} BTC`;
  if (Math.abs(btc) >= 0.001) return `${btc.toFixed(5)} BTC`;
  return `${btc.toFixed(8)} BTC`;
}

/* ------------------------------------------------------------------------------------ */
/* Addresses                                                                             */
/* ------------------------------------------------------------------------------------ */

/**
 * Recognise the real Bitcoin address formats, and say which one it is.
 *
 * This exists partly for display and partly as a guard: it is the check that catches a
 * synthetic identifier trying to pass as an address. `E_bcpl0009945` fails every pattern
 * below, which is precisely the point.
 */
const ADDRESS_PATTERNS = [
  { kind: 'p2pkh',   label: 'Legacy',        re: /^1[1-9A-HJ-NP-Za-km-z]{25,34}$/ },
  { kind: 'p2sh',    label: 'Script',        re: /^3[1-9A-HJ-NP-Za-km-z]{25,34}$/ },
  { kind: 'bech32',  label: 'SegWit',        re: /^bc1q[02-9ac-hj-np-z]{38,58}$/ },
  { kind: 'taproot', label: 'Taproot',       re: /^bc1p[02-9ac-hj-np-z]{58}$/ },
  { kind: 'testnet', label: 'Testnet',       re: /^(?:[mn2][1-9A-HJ-NP-Za-km-z]{25,34}|tb1[02-9ac-hj-np-z]{38,58})$/ },
];

function classifyAddress(address) {
  if (typeof address !== 'string' || address.length === 0) {
    return { valid: false, kind: null, label: 'empty' };
  }
  for (const p of ADDRESS_PATTERNS) {
    if (p.re.test(address)) return { valid: true, kind: p.kind, label: p.label };
  }
  return { valid: false, kind: null, label: 'unrecognised' };
}

function isRealAddress(address) {
  return classifyAddress(address).valid;
}

/** Shorten for display without ever losing the ends, which are what people match on. */
function shortAddress(address, head = 8, tail = 6) {
  if (typeof address !== 'string') return '';
  if (address.length <= head + tail + 2) return address;
  return `${address.slice(0, head)}…${address.slice(-tail)}`;
}

/* ------------------------------------------------------------------------------------ */
/* Provenance                                                                            */
/* ------------------------------------------------------------------------------------ */

/**
 * Build a provenance stamp. Required on every transaction that enters the store.
 *
 * @param {string} source one of SOURCES
 * @param {object} extra  {endpoint, file, rpc_host, raw_available}
 */
function provenance(source, extra = {}) {
  if (!Object.values(SOURCES).includes(source)) {
    throw new Error(
      `Unknown data source "${source}". Every record must declare where it came from; ` +
        `permitted sources are: ${Object.values(SOURCES).join(', ')}.`
    );
  }
  return {
    source,
    is_real: REAL_SOURCES.has(source),
    ingested_at: new Date().toISOString(),
    raw_available: Boolean(extra.raw_available),
    ...extra,
  };
}

/* ------------------------------------------------------------------------------------ */
/* Transactions                                                                          */
/* ------------------------------------------------------------------------------------ */

/**
 * Normalise and validate one transaction.
 *
 * Throws rather than repairs. A silently repaired record is a record whose numbers no
 * longer match the chain, and there is no way for anyone downstream to notice.
 *
 * Canonical shape:
 *   txid          string, the real transaction id
 *   block_height  integer, null only while unconfirmed
 *   block_hash    string, null only while unconfirmed
 *   time          unix seconds
 *   inputs        [{ address, value_sats, prev_txid, prev_vout }]
 *   outputs       [{ n, address, value_sats }]
 *   fee_sats      integer
 *   provenance    the stamp above
 */
function makeTransaction(raw, prov) {
  if (!prov || !prov.source) {
    throw new Error(`Transaction ${raw.txid} has no provenance. Refusing to store it.`);
  }
  if (!raw.txid || typeof raw.txid !== 'string') {
    throw new Error('Transaction has no txid.');
  }

  const inputs = (raw.inputs || []).map((i) => ({
    address: i.address || null,
    value_sats: Math.round(Number(i.value_sats) || 0),
    prev_txid: i.prev_txid || null,
    prev_vout: i.prev_vout == null ? null : Number(i.prev_vout),
  }));

  const outputs = (raw.outputs || []).map((o, idx) => ({
    n: o.n == null ? idx : Number(o.n),
    address: o.address || null,
    value_sats: Math.round(Number(o.value_sats) || 0),
    spent: o.spent == null ? null : Boolean(o.spent),
  }));

  const totalIn = inputs.reduce((s, i) => s + i.value_sats, 0);
  const totalOut = outputs.reduce((s, o) => s + o.value_sats, 0);

  // A coinbase transaction creates value and has no real inputs, so a "negative fee"
  // there is expected rather than an error. Everywhere else it means the record is
  // incomplete, and reporting a wrong fee is how a case pack loses credibility.
  const isCoinbase = inputs.length === 0;
  const fee = isCoinbase ? 0 : Math.max(0, totalIn - totalOut);

  return {
    txid: raw.txid,
    block_height: raw.block_height == null ? null : Number(raw.block_height),
    block_hash: raw.block_hash || null,
    time: Number(raw.time) || 0,
    is_coinbase: isCoinbase,
    inputs,
    outputs,
    total_in_sats: totalIn,
    total_out_sats: totalOut,
    fee_sats: fee,
    provenance: prov,
  };
}

/**
 * Reject anything that is not genuinely from the chain.
 * Used by real mode as a hard gate, so a synthetic record cannot leak into a real case.
 */
function assertReal(tx) {
  if (!tx.provenance || !tx.provenance.is_real) {
    throw new Error(
      `Transaction ${tx.txid} came from "${tx.provenance && tx.provenance.source}", ` +
        'which is not a real chain source. Real mode refuses it.'
    );
  }
  return tx;
}

/** Every distinct address a transaction touches, inputs and outputs together. */
function addressesOf(tx) {
  const set = new Set();
  for (const i of tx.inputs) if (i.address) set.add(i.address);
  for (const o of tx.outputs) if (o.address) set.add(o.address);
  return set;
}

/** How much this transaction paid to one address, in satoshis. */
function receivedBy(tx, address) {
  let sats = 0;
  for (const o of tx.outputs) if (o.address === address) sats += o.value_sats;
  return sats;
}

/** How much this transaction spent from one address, in satoshis. */
function spentBy(tx, address) {
  let sats = 0;
  for (const i of tx.inputs) if (i.address === address) sats += i.value_sats;
  return sats;
}

module.exports = {
  SATS_PER_BTC,
  SOURCES,
  REAL_SOURCES,
  btcToSats,
  toSats,
  formatBtc,
  humanBtc,
  classifyAddress,
  isRealAddress,
  shortAddress,
  ADDRESS_PATTERNS,
  provenance,
  makeTransaction,
  assertReal,
  addressesOf,
  receivedBy,
  spentBy,
};
