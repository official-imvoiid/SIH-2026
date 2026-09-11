'use strict';
/**
 * Pulling real Bitcoin data, and caching it so it is pulled only once.
 *
 * Three sources, all public and none requiring a login:
 *
 *   blockchain.info   the address and transaction crawler. Verified working from this
 *                     machine; mempool.space times out here and blockstream returns 429
 *                     on the first request, so this is not a preference, it is the only
 *                     explorer that answers.
 *   Ransomwhere       confirmed ransomware addresses with named families. This is where
 *                     real illicit labels come from.
 *   OFAC              the US Treasury sanctions list, including sanctioned wallets.
 *
 * Everything fetched is written to disk immediately. That matters more than it sounds:
 * a crawl of a few hundred addresses takes half an hour of polite requests, and nobody
 * should pay that cost twice. It also means a case can be re-opened, re-analysed and
 * demonstrated with the network cable pulled out, which is the difference between a tool
 * and a tool that works on the day it is needed.
 *
 * On rate limits: this code waits between requests and backs off when asked to. A crawler
 * that hammers a free public endpoint gets the address banned, and then nothing works at
 * all. Slow and still running beats fast and blocked.
 */

const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');

const { SOURCES, provenance, makeTransaction, isRealAddress } = require('./model');

/**
 * Download a URL using the system curl binary.
 *
 * Only used where Node's own HTTP client cannot reach a host that curl reaches fine.
 * Output goes to a temporary file rather than through a pipe, because a few megabytes
 * of CSV exceeds the default buffer for captured stdout and would be silently truncated.
 */
function curlText(url, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const tmp = path.join(
      require('os').tmpdir(),
      `chaintrace-${Date.now()}-${Math.random().toString(36).slice(2)}.tmp`
    );
    execFile(
      'curl',
      ['-sL', '--max-time', String(Math.ceil(timeoutMs / 1000)), '-o', tmp, url],
      { timeout: timeoutMs + 5000 },
      (err) => {
        if (err) {
          try { fs.unlinkSync(tmp); } catch { /* nothing to clean */ }
          return reject(new Error(`curl failed: ${err.message}`));
        }
        try {
          const text = fs.readFileSync(tmp, 'utf8');
          fs.unlinkSync(tmp);
          if (!text || text.length < 100) return reject(new Error('curl returned an empty body'));
          resolve(text);
        } catch (readErr) {
          reject(readErr);
        }
      }
    );
  });
}

const CACHE_DIR = path.resolve(__dirname, '..', '..', 'data', 'chain');

const ENDPOINTS = {
  addressTxs: (addr, offset) =>
    `https://blockchain.info/rawaddr/${addr}?limit=50&offset=${offset}`,
  tx: (txid) => `https://blockchain.info/rawtx/${txid}`,
  blockHeight: 'https://blockchain.info/q/getblockcount',
  ransomwhere: 'https://api.ransomwhe.re/export',
  // The old treasury.gov path still works in a browser but redirects to the sanctions
  // list service, and that redirect is not followed cleanly here. Point at the real
  // endpoint directly rather than relying on a hop that may or may not survive.
  ofac: 'https://sanctionslistservice.ofac.treas.gov/api/download/sdn.csv',
  ofacFallback: 'https://www.treasury.gov/ofac/downloads/sdn.csv',
};

const DEFAULTS = {
  // blockchain.info throttles aggressively. This is deliberately conservative.
  minDelayMs: 900,
  maxRetries: 4,
  timeoutMs: 45000,
  /**
   * Pages of history per address, 50 transactions each.
   *
   * One page by default, and that is a considered choice rather than a shortcut. A trace
   * needs to know where an address sent its money, and the most recent fifty transactions
   * answer that for any ordinary wallet. Pulling six pages for every address multiplies
   * the whole crawl by six for information that is almost never read, which is the
   * difference between a trace finishing in a minute and in ten. Addresses whose history
   * is genuinely longer are marked `truncated`, and the tracer reports that rather than
   * pretending it saw everything.
   */
  maxPagesPerAddress: 1,
};

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------------------------------ */
/* Disk cache                                                                            */
/* ------------------------------------------------------------------------------------ */

/**
 * Cache keyed by address or txid. Sharded into subdirectories by the first two
 * characters, because a single directory holding a hundred thousand files is painfully
 * slow to list on Windows.
 */
class DiskCache {
  constructor(root = CACHE_DIR) {
    this.root = root;
    ensureDir(root);
    this.hits = 0;
    this.misses = 0;
    this.writes = 0;
  }

  _path(namespace, key) {
    const safe = String(key).replace(/[^A-Za-z0-9_-]/g, '_');
    const shard = safe.slice(0, 2) || '__';
    const dir = path.join(this.root, namespace, shard);
    return { dir, file: path.join(dir, `${safe}.json`) };
  }

  get(namespace, key) {
    const { file } = this._path(namespace, key);
    if (!fs.existsSync(file)) {
      this.misses += 1;
      return null;
    }
    try {
      const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
      this.hits += 1;
      return parsed;
    } catch {
      // A half-written file from an interrupted run. Treat it as absent and refetch
      // rather than propagating corrupt data into a case.
      this.misses += 1;
      return null;
    }
  }

  set(namespace, key, value) {
    const { dir, file } = this._path(namespace, key);
    ensureDir(dir);
    // Write to a temporary file and rename, so an interrupted write cannot leave a
    // truncated JSON file behind that looks valid until it is parsed.
    const tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(value), 'utf8');
    fs.renameSync(tmp, file);
    this.writes += 1;
    return value;
  }

  has(namespace, key) {
    return fs.existsSync(this._path(namespace, key).file);
  }

  stats() {
    return { hits: this.hits, misses: this.misses, writes: this.writes, root: this.root };
  }
}

/* ------------------------------------------------------------------------------------ */
/* Rate-limited HTTP                                                                     */
/* ------------------------------------------------------------------------------------ */

class RateLimitedFetcher {
  constructor(opt = {}) {
    this.opt = { ...DEFAULTS, ...opt };
    this.lastRequestAt = 0;
    this.requests = 0;
    this.retries = 0;
    this.failures = 0;
    this.onProgress = opt.onProgress || (() => {});
  }

  async _wait() {
    const since = Date.now() - this.lastRequestAt;
    if (since < this.opt.minDelayMs) await sleep(this.opt.minDelayMs - since);
    this.lastRequestAt = Date.now();
  }

  /**
   * GET with exponential backoff. Returns parsed JSON, or text when `asText`.
   * Throws only after every retry is exhausted; the caller records the failure rather
   * than substituting anything for the missing data.
   */
  async get(url, { asText = false } = {}) {
    let lastError = null;

    for (let attempt = 0; attempt <= this.opt.maxRetries; attempt++) {
      await this._wait();

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.opt.timeoutMs);

      try {
        this.requests += 1;
        const res = await fetch(url, {
          signal: controller.signal,
          headers: { 'User-Agent': 'ChainTrace/1.0 (forensic research tool)' },
        });
        clearTimeout(timer);

        if (res.status === 429 || res.status === 430 || res.status >= 500) {
          // Being told to slow down is not an error, it is an instruction.
          const backoff = Math.min(60000, 2000 * 2 ** attempt);
          this.retries += 1;
          this.onProgress({ kind: 'backoff', status: res.status, waitMs: backoff, url });
          await sleep(backoff);
          lastError = new Error(`HTTP ${res.status}`);
          continue;
        }

        if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);

        return asText ? await res.text() : await res.json();
      } catch (err) {
        clearTimeout(timer);
        lastError = err;
        this.retries += 1;
        if (attempt < this.opt.maxRetries) {
          await sleep(Math.min(30000, 1500 * 2 ** attempt));
        }
      }
    }

    this.failures += 1;
    throw new Error(`Gave up on ${url} after ${this.opt.maxRetries + 1} attempts: ${lastError && lastError.message}`);
  }

  stats() {
    return { requests: this.requests, retries: this.retries, failures: this.failures };
  }
}

/* ------------------------------------------------------------------------------------ */
/* blockchain.info -> canonical transactions                                             */
/* ------------------------------------------------------------------------------------ */

/**
 * Convert one blockchain.info transaction into the canonical satoshi-integer shape.
 * Their API already reports values in satoshis, so nothing is converted and nothing is
 * rounded; the integers pass through untouched.
 */
function fromExplorerTx(raw) {
  const inputs = [];
  for (const vin of raw.inputs || []) {
    const prev = vin.prev_out;
    // Coinbase inputs have no previous output. Dropping them leaves an empty input
    // list, which is the correct representation of newly created value.
    if (!prev || !prev.addr) continue;
    inputs.push({
      address: prev.addr,
      value_sats: Number(prev.value) || 0,
      prev_txid: prev.tx_index != null ? String(prev.tx_index) : null,
      prev_vout: prev.n == null ? null : Number(prev.n),
    });
  }

  const outputs = [];
  for (const out of raw.out || []) {
    // Outputs with no address are data carriers such as OP_RETURN. They hold no value
    // anyone can move, so they are not part of the money graph.
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
      block_hash: null, // rawaddr omits it; the tx endpoint supplies it when needed
      time: Number(raw.time) || 0,
      inputs,
      outputs,
    },
    provenance(SOURCES.EXPLORER, {
      endpoint: 'blockchain.info/rawaddr',
      raw_available: true,
    })
  );
}

/* ------------------------------------------------------------------------------------ */
/* The chain client                                                                      */
/* ------------------------------------------------------------------------------------ */

class ChainClient {
  constructor(opt = {}) {
    this.cache = opt.cache || new DiskCache(opt.cacheDir);
    this.http = new RateLimitedFetcher(opt);
    this.onProgress = opt.onProgress || (() => {});
    this.offline = Boolean(opt.offline);
  }

  /**
   * Every transaction touching one address, newest first.
   *
   * Served from disk when it has been fetched before, which is what makes a second run
   * instant and lets the whole thing work offline.
   */
  async addressTransactions(address, opt = {}) {
    if (!isRealAddress(address)) {
      throw new Error(
        `"${address}" is not a valid Bitcoin address. Real mode will not look up ` +
          'synthetic identifiers.'
      );
    }

    const cached = this.cache.get('addr', address);
    if (cached && !opt.refresh) {
      this.onProgress({ kind: 'cache_hit', address, n_txs: cached.txs.length });
      return cached;
    }

    if (this.offline) {
      return {
        address,
        txs: [],
        n_tx: 0,
        fetched: false,
        offline: true,
        note: 'Offline mode and this address is not in the cache, so its history is unknown.',
      };
    }

    const txs = [];
    let offset = 0;
    let total = null;
    let totalReceived = 0;
    let totalSent = 0;
    let pages = 0;

    while (pages < (opt.maxPages || this.http.opt.maxPagesPerAddress)) {
      this.onProgress({ kind: 'fetch', address, offset });
      const page = await this.http.get(ENDPOINTS.addressTxs(address, offset));

      if (total == null) {
        total = Number(page.n_tx) || 0;
        totalReceived = Number(page.total_received) || 0;
        totalSent = Number(page.total_sent) || 0;
      }

      const batch = page.txs || [];
      for (const raw of batch) {
        try {
          txs.push(fromExplorerTx(raw));
        } catch (err) {
          // One malformed transaction must not abort a whole crawl, but it is recorded
          // rather than skipped in silence.
          this.onProgress({ kind: 'bad_tx', address, txid: raw && raw.hash, error: err.message });
        }
      }

      pages += 1;
      offset += batch.length;
      if (batch.length === 0 || offset >= total) break;
    }

    const record = {
      address,
      txs,
      n_tx: total == null ? txs.length : total,
      n_tx_fetched: txs.length,
      truncated: total != null && txs.length < total,
      total_received_sats: totalReceived,
      total_sent_sats: totalSent,
      balance_sats: totalReceived - totalSent,
      fetched: true,
      fetched_at: new Date().toISOString(),
      source: SOURCES.EXPLORER,
    };

    this.cache.set('addr', address, record);
    return record;
  }

  /** One transaction by id, with its block hash. */
  async transaction(txid) {
    const cached = this.cache.get('tx', txid);
    if (cached) return cached;
    if (this.offline) return null;

    const raw = await this.http.get(ENDPOINTS.tx(txid));
    const tx = fromExplorerTx(raw);
    if (raw.block_height != null) tx.block_height = Number(raw.block_height);
    this.cache.set('tx', txid, tx);
    return tx;
  }

  /** Current chain tip, used to show how fresh the data is. */
  async tipHeight() {
    if (this.offline) return null;
    const text = await this.http.get(ENDPOINTS.blockHeight, { asText: true });
    return parseInt(String(text).trim(), 10);
  }

  stats() {
    return { cache: this.cache.stats(), http: this.http.stats() };
  }
}

/* ------------------------------------------------------------------------------------ */
/* Label sources                                                                         */
/* ------------------------------------------------------------------------------------ */

/**
 * Confirmed ransomware addresses, with the family that used them.
 *
 * The families are the valuable part. They allow a model to be tested on crews it has
 * never seen instead of on other wallets belonging to a gang it already learned, which
 * is the difference between measuring generalisation and measuring memory.
 *
 * The download is around 5.5 MB and takes roughly half a minute; an impatient timeout
 * truncates it mid-record and the JSON then fails to parse, which is how this was first
 * discovered.
 */
async function fetchRansomware(opt = {}) {
  const cacheFile = path.join(CACHE_DIR, 'labels', 'ransomwhere.json');
  ensureDir(path.dirname(cacheFile));

  if (fs.existsSync(cacheFile) && !opt.refresh) {
    const parsed = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
    return { ...parsed, from_cache: true };
  }

  const http = new RateLimitedFetcher({ timeoutMs: 180000, minDelayMs: 0, ...opt });
  const raw = await http.get(ENDPOINTS.ransomwhere);
  const rows = Array.isArray(raw) ? raw : raw.result || [];

  const addresses = new Map();
  const families = new Map();
  let payments = 0;

  for (const r of rows) {
    if (r.blockchain !== 'bitcoin') continue;
    if (!isRealAddress(r.address)) continue;

    const family = r.family || 'Unlabeled';
    const txs = (r.transactions || []).map((t) => ({
      txid: t.hash,
      time: t.time,
      value_sats: Math.round(Number(t.amount) || 0),
      usd: t.amountUSD || null,
    }));
    payments += txs.length;

    addresses.set(r.address, {
      address: r.address,
      family,
      balance_sats: Math.round(Number(r.balance) || 0),
      payments: txs,
      n_payments: txs.length,
    });

    if (!families.has(family)) families.set(family, { family, addresses: [], n_payments: 0 });
    const f = families.get(family);
    f.addresses.push(r.address);
    f.n_payments += txs.length;
  }

  const record = {
    source: 'ransomwhe.re',
    source_url: ENDPOINTS.ransomwhere,
    fetched_at: new Date().toISOString(),
    n_addresses: addresses.size,
    n_payments: payments,
    n_families: families.size,
    addresses: Object.fromEntries(addresses),
    families: Object.fromEntries(
      [...families.entries()].map(([k, v]) => [k, { ...v, n_addresses: v.addresses.length }])
    ),
    provenance: provenance(SOURCES.EXPLORER, {
      endpoint: 'api.ransomwhe.re/export',
      attribution: 'Ransomwhere, an open crowdsourced ransomware payment tracker',
      confidence: 'reported and reviewed; treat as strong but not judicial evidence',
    }),
  };

  fs.writeFileSync(cacheFile, JSON.stringify(record), 'utf8');
  return { ...record, from_cache: false };
}

/**
 * Sanctioned Bitcoin addresses from the US Treasury list.
 *
 * Addresses appear inside free-text remarks fields as "Digital Currency Address - XBT ...",
 * so they are extracted by pattern and then validated, rather than by trusting the
 * column layout of a file that is not designed for machine reading.
 */
async function fetchSanctions(opt = {}) {
  const cacheFile = path.join(CACHE_DIR, 'labels', 'ofac.json');
  ensureDir(path.dirname(cacheFile));

  if (fs.existsSync(cacheFile) && !opt.refresh) {
    const parsed = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
    return { ...parsed, from_cache: true };
  }

  const http = new RateLimitedFetcher({ timeoutMs: 120000, minDelayMs: 0, ...opt });

  let text = null;
  let usedUrl = ENDPOINTS.ofac;
  let transport = 'fetch';
  const attempts = [];

  for (const url of [ENDPOINTS.ofac, ENDPOINTS.ofacFallback]) {
    try {
      text = await http.get(url, { asText: true });
      usedUrl = url;
      break;
    } catch (err) {
      attempts.push(`${url}: ${err.message}`);
    }
  }

  // The Treasury host accepts a connection from curl and times out the TLS handshake
  // from Node on the very same IP address. The cause is on their side and not something
  // this code can fix, so rather than losing sanctions screening entirely, fall back to
  // the curl binary that ships with Windows 10 and every mainstream Linux and macOS.
  if (text == null) {
    try {
      text = await curlText(ENDPOINTS.ofac, opt.timeoutMs || 120000);
      usedUrl = ENDPOINTS.ofac;
      transport = 'curl';
    } catch (err) {
      attempts.push(`curl: ${err.message}`);
      const e = new Error(
        'Could not download the OFAC sanctions list. Sanctions screening will be ' +
          'unavailable; everything else still works. Attempts:\n  ' + attempts.join('\n  ')
      );
      e.recoverable = true;
      throw e;
    }
  }

  const found = new Map();
  const candidate = /\b((?:1|3)[1-9A-HJ-NP-Za-km-z]{25,34}|bc1[02-9ac-hj-np-z]{11,71})\b/g;

  for (const line of text.split('\n')) {
    const matches = line.match(candidate);
    if (!matches) continue;
    for (const m of matches) {
      if (!isRealAddress(m)) continue;
      // The first column of an SDN row is the entity number; the name follows it.
      const name = (line.split(',')[1] || '').replace(/^"|"$/g, '').trim();
      if (!found.has(m)) found.set(m, { address: m, entity: name || null });
    }
  }

  const record = {
    source: 'US Treasury OFAC SDN list',
    source_url: usedUrl,
    transport,
    fetched_at: new Date().toISOString(),
    n_addresses: found.size,
    addresses: Object.fromEntries(found),
    provenance: provenance(SOURCES.EXPLORER, {
      endpoint: usedUrl,
      attribution: 'Office of Foreign Assets Control, Specially Designated Nationals list',
      confidence: 'authoritative government designation',
    }),
  };

  fs.writeFileSync(cacheFile, JSON.stringify(record), 'utf8');
  return { ...record, from_cache: false };
}

module.exports = {
  curlText,
  ChainClient,
  DiskCache,
  RateLimitedFetcher,
  fetchRansomware,
  fetchSanctions,
  fromExplorerTx,
  CACHE_DIR,
  ENDPOINTS,
};
