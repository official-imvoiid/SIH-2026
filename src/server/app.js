'use strict';
/**
 * HTTP layer for the investigation console.
 *
 * Serves only data that was ingested from the chain. There is no synthetic fallback in
 * this codebase at all any more: the generator and every module that consumed it have
 * been deleted. When nothing has been ingested, these endpoints report that plainly and
 * the interface shows an empty state, because a console that invents numbers to fill
 * itself is worse than one that shows none.
 *
 * The window file is large (roughly 190 MB for two days) so it is parsed once at startup
 * and held in memory. Re-reading it per request would add seconds to every click.
 */

const express = require('express');
const fs = require('fs');
const path = require('path');

const { ChainClient } = require('../chain/client');
const { BlockFetcher } = require('../chain/blocks');
const { traceToEndpoint } = require('../analysis/trace');
const { humanBtc } = require('../chain/model');
const sources = require('../chain/sources');

const DATA = path.resolve(__dirname, '..', '..', 'data', 'chain');
const UI = path.resolve(__dirname, '..', '..', 'ui');

/** Parsed once. Reloaded only when the file on disk is newer than what we hold. */
const store = {
  window: null,
  analysis: null,
  windowMtime: 0,
  analysisMtime: 0,
  adjacency: null,
  addressIndex: null,
};

function mtime(file) {
  try {
    return fs.statSync(path.join(DATA, file)).mtimeMs;
  } catch {
    return 0;
  }
}

function readJson(file) {
  const p = path.join(DATA, file);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch (err) {
    console.error(`  Could not parse ${file}: ${err.message}`);
    return null;
  }
}

/**
 * Load the window and analysis if they have changed on disk.
 *
 * Adjacency is built once here rather than per graph request. A two-day window has
 * roughly 2.8 million edges, and walking that list on every click would make the
 * interface feel broken.
 */
function refresh() {
  const wm = mtime('window.json');
  if (wm && wm !== store.windowMtime) {
    console.log('  Loading window.json ...');
    const t0 = Date.now();
    store.window = readJson('window.json');
    store.windowMtime = wm;

    if (store.window) {
      store.addressIndex = new Map();
      store.window.rows.forEach((r, i) => store.addressIndex.set(r.address, i));

      store.adjacency = new Map();
      for (const [from, to, sats, n] of store.window.edges) {
        if (!store.adjacency.has(from)) store.adjacency.set(from, []);
        if (!store.adjacency.has(to)) store.adjacency.set(to, []);
        store.adjacency.get(from).push({ other: to, sats, n, dir: 'out' });
        store.adjacency.get(to).push({ other: from, sats, n, dir: 'in' });
      }
      console.log(
        `  Indexed ${store.window.rows.length.toLocaleString()} addresses and ` +
          `${store.window.edges.length.toLocaleString()} edges in ${Date.now() - t0}ms`
      );
    }
  }

  const am = mtime('analysis.json');
  if (am && am !== store.analysisMtime) {
    store.analysis = readJson('analysis.json');
    store.analysisMtime = am;
  }

  return store;
}

function featuresOf(rowIndex) {
  const w = store.window;
  const out = {};
  w.feature_names.forEach((n, i) => {
    out[n] = w.rows[rowIndex].x[i];
  });
  return out;
}

function createApp() {
  const app = express();
  app.use(express.json({ limit: '16mb' }));

  const client = new ChainClient();

  const wrap = (fn) => (req, res) => {
    Promise.resolve(fn(req, res)).catch((err) => {
      if (!res.headersSent) res.status(500).json({ ok: false, error: err.message });
    });
  };

  const needData = (res) => {
    refresh();
    if (!store.window) {
      res.status(409).json({
        ok: false,
        error: 'No chain data has been ingested.',
        hint: 'node scripts/ingest.js --days 2',
      });
      return false;
    }
    return true;
  };

  /* ------------------------------------------------------------------ overview --- */

  app.get('/api/overview', (req, res) => {
    refresh();
    const w = store.window;
    const a = store.analysis;

    if (!w) {
      return res.json({
        ok: true,
        has_data: false,
        blocks_cached: new BlockFetcher().cachedHeights().length,
        hint: 'node scripts/ingest.js --days 2',
      });
    }

    const hours =
      w.range.start_time && w.range.end_time
        ? (w.range.end_time - w.range.start_time) / 3600
        : null;

    res.json({
      ok: true,
      has_data: true,
      analysed: Boolean(a),
      source: w.source,
      ingested_at: w.built_at,
      range: { ...w.range, hours: hours ? +hours.toFixed(1) : null },
      totals: {
        transactions: w.stats.transactions,
        addresses_seen: w.stats.distinct_addresses,
        addresses_analysed: w.stats.addresses_with_min_activity,
        edges: w.edges.length,
      },
      patterns: w.patterns,
      analysis: a
        ? {
            device: a.device,
            clustering: a.clustering,
            labels: a.labels,
            gnn:
              a.gnn && a.gnn.ok
                ? {
                    architecture: a.gnn.architecture,
                    hops: a.gnn.hops_reachable,
                    epochs: a.gnn.epochs_run,
                    seconds: a.gnn.trained_in_seconds,
                    device: a.gnn.device.device,
                    nodes: a.gnn.n_nodes,
                    edges: a.gnn.n_edges,
                    split: a.gnn.split,
                    test: a.gnn.test_metrics ? a.gnn.test_metrics.illicit : null,
                    auc_pr: a.gnn.test_metrics ? a.gnn.test_metrics.auc_pr : null,
                  }
                : null,
            built_at: a.built_at,
          }
        : null,
    });
  });

  /* ------------------------------------------------------------------- ranking --- */

  app.get('/api/ranking', (req, res) => {
    if (!needData(res)) return;
    const a = store.analysis;
    if (!a) {
      return res.json({
        ok: true,
        results: [],
        message: 'Ingested but not analysed. Run: python python/pipeline.py',
      });
    }

    const minBtc = parseFloat(req.query.min_btc) || 0;
    const limit = Math.min(300, parseInt(req.query.limit, 10) || 60);
    const pattern = req.query.pattern || 'all';

    let rows = a.ranking.filter((r) => (r.features.received_btc || 0) >= minBtc);

    if (pattern === 'layering') rows = rows.filter((r) => r.features.layering_score > 0.1);
    else if (pattern === 'fanin') rows = rows.filter((r) => r.features.collector_score > 0.1);
    else if (pattern === 'peel') rows = rows.filter((r) => (r.features.peel_chain_length || 0) >= 3);

    res.json({
      ok: true,
      total_ranked: a.ranking.length,
      returned: Math.min(rows.length, limit),
      results: rows.slice(0, limit).map((r) => {
        const f = r.features;
        const tags = [];
        if (f.layering_score > 0.1) tags.push('layering');
        if (f.collector_score > 0.1) tags.push('fan-in');
        if ((f.peel_chain_length || 0) >= 3) tags.push(`peel x${Math.round(f.peel_chain_length)}`);
        if (f.forward_ratio > 0.95) tags.push('pass-through');
        return {
          rank: r.rank,
          address: r.address,
          score: r.score,
          cluster: r.cluster,
          tags,
          received_btc: f.received_btc,
          balance_btc: f.balance_btc,
          n_tx_in: f.n_tx_in,
          n_tx_out: f.n_tx_out,
          in_degree: f.in_degree,
          out_degree: f.out_degree,
          holding_seconds: f.avg_holding_time,
        };
      }),
    });
  });

  /* ------------------------------------------------------------------- address --- */

  app.get('/api/address/:addr', (req, res) => {
    if (!needData(res)) return;
    const idx = store.addressIndex.get(req.params.addr);
    if (idx === undefined) {
      return res.status(404).json({
        ok: false,
        error: 'Not present in the ingested window.',
        note:
          'The address may well exist on-chain. This console only knows the block range ' +
          'that was pulled, and says so rather than guessing.',
      });
    }

    const w = store.window;
    const a = store.analysis;
    const f = featuresOf(idx);

    const inbound = [];
    const outbound = [];
    for (const nb of store.adjacency.get(idx) || []) {
      const rec = {
        address: w.rows[nb.other].address,
        value_sats: nb.sats,
        value_btc: nb.sats / 1e8,
        n_txs: nb.n,
      };
      (nb.dir === 'in' ? inbound : outbound).push(rec);
    }
    inbound.sort((x, y) => y.value_sats - x.value_sats);
    outbound.sort((x, y) => y.value_sats - x.value_sats);

    const ranked = a ? a.ranking.find((r) => r.address === req.params.addr) : null;
    const cluster = a ? a.cluster_labels[idx] : null;

    res.json({
      ok: true,
      address: req.params.addr,
      first_seen: w.rows[idx].first_seen,
      last_seen: w.rows[idx].last_seen,
      features: f,
      score: ranked ? ranked.score : null,
      rank: ranked ? ranked.rank : null,
      cluster,
      cluster_info:
        a && cluster != null && a.cluster_risk[String(cluster)]
          ? a.cluster_risk[String(cluster)]
          : null,
      is_exit: (f.in_degree >= 40 && f.out_degree >= 40)
        || ((f.received_btc || 0) > 0.01 && (f.n_tx_out || 0) === 0),
      exit_kind: (f.in_degree >= 40 && f.out_degree >= 40) ? 'service'
        : ((f.received_btc || 0) > 0.01 && (f.n_tx_out || 0) === 0) ? 'dormant' : null,
      inbound: inbound.slice(0, 30),
      outbound: outbound.slice(0, 30),
      n_inbound: inbound.length,
      n_outbound: outbound.length,
      provenance: {
        source: w.source,
        blocks: `${w.range.start_height}-${w.range.end_height}`,
        ingested_at: w.built_at,
      },
    });
  });

  /* --------------------------------------------------------------------- graph --- */

  app.get('/api/graph/:addr', (req, res) => {
    if (!needData(res)) return;
    const seed = store.addressIndex.get(req.params.addr);
    if (seed === undefined) {
      return res.status(404).json({ ok: false, error: 'Not present in the ingested window.' });
    }

    const hops = Math.min(4, Math.max(1, parseInt(req.query.hops, 10) || 2));
    const maxNodes = Math.min(500, Math.max(20, parseInt(req.query.max_nodes, 10) || 160));

    // Breadth-first, but expanding the highest-value edges first so that when the cap
    // bites it keeps the money rather than whichever neighbour happened to be first.
    const dist = new Map([[seed, 0]]);
    let frontier = [seed];

    for (let d = 0; d < hops && dist.size < maxNodes; d++) {
      const next = [];
      const candidates = [];
      for (const cur of frontier) {
        for (const nb of store.adjacency.get(cur) || []) {
          if (!dist.has(nb.other)) candidates.push(nb);
        }
      }
      candidates.sort((x, y) => y.sats - x.sats);
      for (const nb of candidates) {
        if (dist.has(nb.other)) continue;
        if (dist.size >= maxNodes) break;
        dist.set(nb.other, d + 1);
        next.push(nb.other);
      }
      frontier = next;
      if (frontier.length === 0) break;
    }

    const w = store.window;
    const a = store.analysis;
    const scoreOf = new Map();
    if (a) for (const r of a.ranking) scoreOf.set(r.address, r.score);

    /**
     * Is this address an exit point: somewhere money leaves the pseudonymous chain?
     *
     * Within a bounded window the signal is degree, not lifetime transaction count. An
     * exchange sweeps deposits from very many customers and pays out to very many more,
     * so inside two days it shows both a wide in-degree and a wide out-degree. An
     * ordinary wallet, however busy, does not touch hundreds of distinct counterparties
     * in both directions.
     *
     * This is a structural inference, not an identification, and it is labelled as such
     * everywhere it surfaces. An exchange and a large laundering operation genuinely look
     * alike here; what distinguishes them is a maintained tag list, which this build does
     * not yet carry.
     */
    const exitInfo = (f) => {
      const inD = f.in_degree || 0;
      const outD = f.out_degree || 0;
      if (inD >= 40 && outD >= 40) {
        return { is_exit: true, kind: 'service', why: `${inD} senders and ${outD} recipients inside the window` };
      }
      // Received and never spent: the money stopped, which is also an endpoint even
      // though it is a parked one rather than a cash-out.
      if ((f.received_btc || 0) > 0.01 && (f.n_tx_out || 0) === 0) {
        return { is_exit: true, kind: 'dormant', why: `holds ${(f.balance_btc || 0).toFixed(4)} BTC, never spent` };
      }
      return { is_exit: false, kind: null, why: null };
    };

    const keep = new Set(dist.keys());
    const nodes = [...keep].map((i) => {
      const f = featuresOf(i);
      const exit = exitInfo(f);
      return {
        id: i,
        address: w.rows[i].address,
        hops: dist.get(i),
        is_seed: i === seed,
        is_exit: exit.is_exit,
        exit_kind: exit.kind,
        exit_why: exit.why,
        score: scoreOf.get(w.rows[i].address) || 0,
        received_btc: f.received_btc,
        balance_btc: f.balance_btc,
        in_degree: f.in_degree,
        out_degree: f.out_degree,
        cluster: a ? a.cluster_labels[i] : null,
        layering: f.layering_score,
        collector: f.collector_score,
        peel_len: f.peel_chain_length || 0,
      };
    });

    const edges = [];
    for (const i of keep) {
      for (const nb of store.adjacency.get(i) || []) {
        if (nb.dir === 'out' && keep.has(nb.other)) {
          edges.push({ source: i, target: nb.other, value_btc: nb.sats / 1e8, n_txs: nb.n });
        }
      }
    }

    const exits = nodes
      .filter((x) => x.is_exit)
      .sort((x, y) => (y.received_btc || 0) - (x.received_btc || 0));

    res.json({
      ok: true,
      seed: req.params.addr,
      hops,
      nodes,
      edges,
      exits,
      n_exits: exits.length,
      truncated: dist.size >= maxNodes,
      note: dist.size >= maxNodes
        ? `Capped at ${maxNodes} addresses, keeping the highest-value routes. This view is partial.`
        : null,
    });
  });

  /* ------------------------------------------------------------------ clusters --- */

  app.get('/api/clusters', (req, res) => {
    refresh();
    const a = store.analysis;
    if (!a) return res.json({ ok: true, clusters: [] });

    res.json({
      ok: true,
      algorithm: 'HDBSCAN over log-scaled, standardised behavioural features',
      n_clusters: a.clustering.n_clusters,
      n_noise: a.clustering.n_noise,
      fitted_on: a.clustering.fitted_on || null,
      total_points: a.clustering.total_points || null,
      subsampled: a.clustering.subsampled || false,
      clusters: Object.entries(a.cluster_risk)
        .filter(([, v]) => !v.is_noise && v.size >= 3)
        .map(([k, v]) => ({ cluster: Number(k), ...v }))
        .sort((x, y) => y.risk - x.risk || y.size - x.size)
        .slice(0, 50),
    });
  });

  /* --------------------------------------------------------------------- trace --- */

  app.post('/api/trace', wrap(async (req, res) => {
    const address = req.body && req.body.address;
    if (!address) return res.status(400).json({ ok: false, error: 'address is required.' });

    const result = await traceToEndpoint(client, address, {
      maxHops: Math.min(10, parseInt(req.body.hops, 10) || 6),
      fetchBudget: Math.min(250, parseInt(req.body.budget, 10) || 60),
    });

    if (!result.ok) return res.json(result);
    res.json({
      ...result,
      endpoints: result.endpoints.slice(0, 12).map((e) => ({ ...e, value_btc: humanBtc(e.value_sats) })),
    });
  }));

  /* ------------------------------------------------------------------- sources --- */

  app.get('/api/sources', (req, res) => {
    const cfg = sources.read();
    res.json({
      ok: true,
      presets: Object.values(sources.PRESETS).map((x) => ({
        id: x.id, label: x.label, needs_key: x.needs_key,
        verified: x.verified, note: x.note,
      })),
      config: {
        ...cfg,
        // Never send the key back to the page; only whether one is set.
        api_key: undefined,
        has_key: Boolean(cfg.api_key),
      },
    });
  });

  app.post('/api/sources', wrap((req, res) => {
    const body = req.body || {};
    const patch = {};
    if (body.provider) patch.provider = body.provider;
    if (body.height_provider) patch.height_provider = body.height_provider;
    if (typeof body.api_key === 'string') patch.api_key = body.api_key;
    if (body.custom) patch.custom = body.custom;
    if (body.window) {
      const w = body.window;
      patch.window = {
        mode: w.mode === 'blocks' ? 'blocks' : 'days',
        days: Number.isFinite(+w.days) && +w.days > 0 ? +w.days : 2,
        blocks: Number.isFinite(+w.blocks) && +w.blocks > 0 ? Math.round(+w.blocks) : null,
        end_height: Number.isFinite(+w.end_height) && +w.end_height > 0 ? Math.round(+w.end_height) : null,
      };
    }
    const next = sources.write(patch);
    res.json({ ok: true, config: { ...next, api_key: undefined, has_key: Boolean(next.api_key) } });
  }));

  /* -------------------------------------------------------------------- static --- */

  app.use(express.static(UI));
  app.get('*', (req, res) => {
    if (req.path.startsWith('/api/')) {
      return res.status(404).json({ ok: false, error: `No such endpoint: ${req.path}` });
    }
    res.sendFile(path.join(UI, 'index.html'));
  });

  return app;
}

function listen(app, port = 7400, host = '127.0.0.1', attempts = 15) {
  return new Promise((resolve, reject) => {
    const attempt = (p, left) => {
      const server = app.listen(p, host);
      server.once('listening', () => resolve({ server, port: p }));
      server.once('error', (err) => {
        if (err.code === 'EADDRINUSE' && left > 0) attempt(p + 1, left - 1);
        else reject(err);
      });
    };
    attempt(port, attempts);
  });
}

module.exports = { createApp, listen, refresh, store };

if (require.main === module) {
  const app = createApp();
  listen(app, parseInt(process.env.PORT, 10) || 7400).then(({ port }) => {
    console.log('');
    console.log('  ChainTrace');
    console.log('  ' + '-'.repeat(56));
    refresh();
    if (store.window) {
      const w = store.window;
      console.log(`  Blocks       ${w.range.start_height}-${w.range.end_height}`);
      console.log(`  Transactions ${w.stats.transactions.toLocaleString()}`);
      console.log(`  Addresses    ${w.rows.length.toLocaleString()} analysed`);
      console.log(`  Analysed     ${store.analysis ? 'yes' : 'no, run python python/pipeline.py'}`);
    } else {
      console.log('  No data ingested. Run: node scripts/ingest.js --days 2');
    }
    console.log('');
    console.log(`  http://127.0.0.1:${port}`);
    console.log('');
  });
}
