'use strict';
/**
 * The chain client: fetch real address history, cache it, survive rate limits.
 *
 * Everything it returns came from a public block explorer and is written to disk before
 * it is handed back. A second trace over the same ground costs nothing and needs no
 * network, which is what makes this usable rather than a demonstration that only works
 * while the wind is in the right direction.
 */

const fs = require('fs');
const path = require('path');

const { isRealAddress, SOURCES } = require('./model');
const { ProviderPool, PROVIDERS } = require('./providers');

const CACHE_DIR = path.resolve(__dirname, '..', '..', 'data', 'chain');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------------------------------ */
/* Disk cache                                                                            */
/* ------------------------------------------------------------------------------------ */

class DiskCache {
  constructor(root = CACHE_DIR) {
    this.root = root;
    fs.mkdirSync(root, { recursive: true });
    this.hits = 0;
    this.misses = 0;
    this.writes = 0;
  }

  _path(ns, key) {
    const safe = String(key).replace(/[^A-Za-z0-9_-]/g, '_');
    // Sharded by the first two characters. One directory holding a hundred thousand
    // files is painfully slow to enumerate on Windows.
    const dir = path.join(this.root, ns, safe.slice(0, 2) || '__');
    return { dir, file: path.join(dir, `${safe}.json`) };
  }

  get(ns, key) {
    const { file } = this._path(ns, key);
    if (!fs.existsSync(file)) {
      this.misses += 1;
      return null;
    }
    try {
      const v = JSON.parse(fs.readFileSync(file, 'utf8'));
      this.hits += 1;
      return v;
    } catch {
      // A truncated file from an interrupted run. Refetch rather than letting corrupt
      // data into a case.
      this.misses += 1;
      return null;
    }
  }

  set(ns, key, value) {
    const { dir, file } = this._path(ns, key);
    fs.mkdirSync(dir, { recursive: true });
    // Write then rename, so an interrupted write cannot leave behind a file that looks
    // valid until something tries to parse it.
    const tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(value), 'utf8');
    fs.renameSync(tmp, file);
    this.writes += 1;
    return value;
  }

  has(ns, key) {
    return fs.existsSync(this._path(ns, key).file);
  }

  count(ns) {
    const dir = path.join(this.root, ns);
    if (!fs.existsSync(dir)) return 0;
    let n = 0;
    for (const shard of fs.readdirSync(dir)) {
      const p = path.join(dir, shard);
      if (fs.statSync(p).isDirectory()) n += fs.readdirSync(p).filter((f) => f.endsWith('.json')).length;
    }
    return n;
  }

  stats() {
    return {
      hits: this.hits,
      misses: this.misses,
      writes: this.writes,
      addresses_cached: this.count('addr'),
      root: this.root,
    };
  }
}

/* ------------------------------------------------------------------------------------ */
/* Client                                                                                */
/* ------------------------------------------------------------------------------------ */

class ChainClient {
  constructor(opt = {}) {
    this.cache = opt.cache || new DiskCache(opt.cacheDir);
    this.pool = opt.pool || new ProviderPool(opt.providers || PROVIDERS);
    this.offline = Boolean(opt.offline);
    this.onProgress = opt.onProgress || (() => {});

    this.minDelayMs = opt.minDelayMs == null ? 350 : opt.minDelayMs;
    this.timeoutMs = opt.timeoutMs || 30000;
    this.maxPages = opt.maxPages == null ? 1 : opt.maxPages;
    /** Give up on a request once every provider has refused this many times over. */
    this.maxAttempts = opt.maxAttempts || 6;

    this._lastRequestAt = new Map();
    this.requests = 0;
    this.rateLimitHits = 0;
    this.errors = 0;
  }

  /** Space out requests per provider, so rotating does not mean waiting needlessly. */
  async _pace(name) {
    const last = this._lastRequestAt.get(name) || 0;
    const since = Date.now() - last;
    if (since < this.minDelayMs) await sleep(this.minDelayMs - since);
    this._lastRequestAt.set(name, Date.now());
  }

  /**
   * Fetch a URL from whichever provider is willing, rotating on refusal.
   * @param {(p:object)=>string} urlFor builds the URL for a given provider
   * @returns {Promise<{json:any, provider:object}>}
   */
  async _request(urlFor, { asText = false } = {}) {
    if (this.offline) throw new Error('Offline: this data is not in the local cache.');

    let attempts = 0;
    let lastError = null;

    while (attempts < this.maxAttempts) {
      const willing = this.pool.available();

      if (willing.length === 0) {
        // Every provider is benched. A short wait is worth it once, in case a limit is
        // about to lift — but this loop used to wait 45 seconds over and over, which
        // meant one stubborn address could stall an entire batch job for many minutes.
        // Give it one real wait, then surface the failure so the caller (a batch script,
        // a trace) can move on rather than being held hostage by a single lookup.
        if (attempts === 0) {
          const wait = Math.min(this.pool.msUntilAnyAvailable(), 20000);
          if (wait > 0) {
            this.onProgress({ kind: 'all_benched', waitMs: wait, providers: this.pool.status() });
            await sleep(wait);
            attempts += 1;
            continue;
          }
        }
        throw new Error(
          'Every block explorer is currently rate-limiting this machine. ' +
            'The limits are temporary (usually 10-30 minutes); anything already cached ' +
            'still works with --offline. Run again shortly.'
        );
      }

      const slot = willing[0];
      const provider = slot.provider;
      const url = urlFor(provider);
      if (!url) {
        attempts += 1;
        continue;
      }

      await this._pace(provider.name);

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        this.requests += 1;
        const res = await fetch(url, {
          signal: controller.signal,
          headers: { 'User-Agent': 'ChainTrace/1.0 (bitcoin forensics research)' },
        });
        clearTimeout(timer);

        if (res.status === 429 || res.status === 430 || res.status === 503) {
          this.rateLimitHits += 1;
          const penalty = this.pool.bench(provider.name, `HTTP ${res.status}`);
          this.onProgress({
            kind: 'rate_limited',
            provider: provider.name,
            status: res.status,
            benchedMs: penalty,
          });
          attempts += 1;
          lastError = new Error(`${provider.name} returned HTTP ${res.status}`);
          continue;
        }

        if (!res.ok) {
          this.pool.fail(provider.name);
          attempts += 1;
          lastError = new Error(`${provider.name} returned HTTP ${res.status}`);
          continue;
        }

        const body = asText ? await res.text() : await res.json();
        this.pool.succeed(provider.name);
        return { json: body, provider };
      } catch (err) {
        clearTimeout(timer);
        this.errors += 1;
        this.pool.fail(provider.name);
        // A timeout or a refused connection is this provider being unreachable, not a
        // reason to abandon the request. Bench it briefly and try the next one.
        this.pool.bench(provider.name, err.cause ? err.cause.code || err.message : err.message);
        attempts += 1;
        lastError = err;
      }
    }

    throw new Error(
      `Could not fetch after ${attempts} attempts across ${this.pool.providers.length} ` +
        `providers. Last error: ${lastError && lastError.message}`
    );
  }

  /**
   * Everything one address has ever done, as far as the page limit allows.
   * Served from disk whenever it has been fetched before.
   */
  async addressTransactions(address, opt = {}) {
    if (!isRealAddress(address)) {
      throw new Error(
        `"${address}" is not a valid Bitcoin address. This tool will not look up ` +
          'synthetic identifiers.'
      );
    }

    if (!opt.refresh) {
      const cached = this.cache.get('addr', address);
      if (cached) return cached;
    }

    if (this.offline) {
      return {
        address,
        txs: [],
        n_tx: 0,
        fetched: false,
        offline: true,
        note: 'Not in the local cache, and the client is offline, so this history is unknown.',
      };
    }

    const maxPages = opt.maxPages == null ? this.maxPages : opt.maxPages;
    const txs = [];
    let summary = null;
    let cursor = null;
    let pages = 0;
    let usedProvider = null;

    while (pages < maxPages) {
      const { json, provider } = await this._request((p) => p.txsUrl(address, cursor));
      usedProvider = provider.name;

      // Esplora splits the summary from the transaction list; blockchain.info returns
      // both at once. Ask for the summary separately only when it is needed.
      let parsed;
      if (provider.api === 'esplora') {
        parsed = provider.parseTxs(json);
        if (summary == null) {
          try {
            const s = await this._request((p) =>
              p.api === 'esplora' ? p.summaryUrl(address) : null
            );
            summary = s.provider.parseSummary(s.json);
          } catch {
            // A missing summary is survivable: the totals can be derived from the
            // transactions actually retrieved, and the record says so.
            summary = null;
          }
        }
      } else {
        parsed = provider.parseTxs(json);
        if (summary == null && parsed.summary) summary = parsed.summary;
        parsed.cursor = String(txs.length + parsed.pageSize);
      }

      txs.push(...parsed.txs);
      pages += 1;
      cursor = parsed.cursor;

      if (!parsed.pageSize || parsed.pageSize < 25 || !cursor) break;
      if (summary && txs.length >= summary.n_tx) break;
    }

    // Derive totals from what was actually seen when no summary was available. Marked
    // as derived so nothing downstream mistakes a partial sum for the address total.
    let derived = false;
    if (!summary) {
      derived = true;
      let received = 0;
      let sent = 0;
      for (const tx of txs) {
        for (const o of tx.outputs) if (o.address === address) received += o.value_sats;
        for (const i of tx.inputs) if (i.address === address) sent += i.value_sats;
      }
      summary = {
        n_tx: txs.length,
        total_received_sats: received,
        total_sent_sats: sent,
        balance_sats: received - sent,
      };
    }

    const record = {
      address,
      txs,
      n_tx: summary.n_tx,
      n_tx_fetched: txs.length,
      truncated: txs.length < summary.n_tx,
      total_received_sats: summary.total_received_sats,
      total_sent_sats: summary.total_sent_sats,
      balance_sats: summary.balance_sats,
      totals_derived_from_partial_history: derived,
      fetched: true,
      fetched_at: new Date().toISOString(),
      source: SOURCES.EXPLORER,
      provider: usedProvider,
    };

    this.cache.set('addr', address, record);
    return record;
  }

  /** One transaction by id. */
  async transaction(txid) {
    const cached = this.cache.get('tx', txid);
    if (cached) return cached;
    if (this.offline) return null;

    const { json, provider } = await this._request((p) => p.txUrl(txid));
    const tx = provider.parseTx(json);
    this.cache.set('tx', txid, tx);
    return tx;
  }

  /** Current chain tip height, for showing how fresh the data is. */
  async tipHeight() {
    if (this.offline) return null;
    const { json } = await this._request((p) => p.tipUrl, { asText: true });
    const n = parseInt(String(json).trim(), 10);
    return Number.isFinite(n) ? n : null;
  }

  stats() {
    return {
      requests: this.requests,
      rate_limit_hits: this.rateLimitHits,
      errors: this.errors,
      cache: this.cache.stats(),
      providers: this.pool.status(),
    };
  }
}

module.exports = { ChainClient, DiskCache, CACHE_DIR };
