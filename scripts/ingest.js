'use strict';
/**
 * Pull a window of real blocks and build behavioural features from them.
 *
 *   node scripts/ingest.js --days 2
 *   node scripts/ingest.js --blocks 12          fetch an exact number of blocks
 *   node scripts/ingest.js --blocks 12 --offline  rebuild features from cached blocks
 *
 * Writes data/chain/window.json, which the clustering and the graph network both read.
 */

const fs = require('fs');
const path = require('path');

const { BlockFetcher } = require('../src/chain/blocks');
const sources = require('../src/chain/sources');
const { extractBehaviour, buildEdges, MODEL_FEATURES, CLUSTER_FEATURES } = require('../src/analysis/behavior');

const OUT = path.resolve(__dirname, '..', 'data', 'chain', 'window.json');

function parseArgs(argv) {
  const a = { days: null, blocks: null, offline: false, minTxs: 2, endHeight: null };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--days') a.days = parseFloat(argv[++i]);
    else if (argv[i] === '--blocks') a.blocks = parseInt(argv[++i], 10);
    else if (argv[i] === '--offline') a.offline = true;
    else if (argv[i] === '--min-txs') a.minTxs = parseInt(argv[++i], 10);
    else if (argv[i] === '--end-height') a.endHeight = parseInt(argv[++i], 10);
  }
  // No flag given: take the window from the saved configuration rather than assuming
  // two days. The interface writes that file, so whatever was chosen there is honoured
  // here without the number being duplicated in two places.
  if (a.days == null && a.blocks == null) {
    const cfg = sources.read().window;
    if (cfg.mode === 'blocks' && cfg.blocks) a.blocks = cfg.blocks;
    else a.days = cfg.days || 2;
    if (cfg.end_height) a.endHeight = cfg.end_height;
  }
  return a;
}

const fmtMB = (n) => `${(n / 1e6).toFixed(1)} MB`;
const fmtTime = (ms) => `${(ms / 1000).toFixed(1)}s`;

async function main() {
  const args = parseArgs(process.argv);

  console.log('');
  console.log('='.repeat(74));
  console.log('  BLOCK INGESTION');
  console.log('='.repeat(74));
  if (args.blocks) console.log(`  Window        ${args.blocks} blocks`);
  else console.log(`  Window        ${args.days} day(s) (~${Math.round(args.days * 144)} blocks)`);
  console.log(`  Mode          ${args.offline ? 'offline, cached blocks only' : 'live fetch'}`);
  console.log('');

  const fetcher = new BlockFetcher({
    offline: args.offline,
    onProgress: (e) => {
      if (e.kind === 'block') {
        const pct = ((e.done / e.total) * 100).toFixed(0);
        process.stdout.write(
          `\r  block ${e.height}  ${e.done}/${e.total} (${pct}%)  ${e.cached ? 'cached' : 'fetching'}   ${fmtTime(e.elapsed_ms)}    `
        );
      } else if (e.kind === 'rate_limited') {
        process.stdout.write(`\r  rate limited (HTTP ${e.status}), waiting ${Math.round(e.waitMs / 1000)}s            \n`);
      } else if (e.kind === 'block_failed') {
        process.stdout.write(`\r  block ${e.height} failed: ${e.error}                    \n`);
      }
    },
  });

  let blocks;
  let range;
  let fetchStats;

  if (args.offline) {
    blocks = fetcher.loadCachedWindow();
    if (blocks.length === 0) {
      console.log('  No cached blocks. Run without --offline first.\n');
      return;
    }
    range = {
      start_height: blocks[0].height,
      end_height: blocks[blocks.length - 1].height,
      retrieved_blocks: blocks.length,
      start_time: blocks[0].time,
      end_time: blocks[blocks.length - 1].time,
    };
    fetchStats = { cached: blocks.length, fetched: 0, bytes: 0 };
    console.log(`  Loaded ${blocks.length} cached blocks from disk.`);
  } else {
    const res = await fetcher.fetchWindow({ days: args.days, blocks: args.blocks, endHeight: args.endHeight });
    blocks = res.blocks;
    range = res.range;
    fetchStats = res.stats;
    process.stdout.write('\r' + ' '.repeat(74) + '\r');
    console.log(`  Retrieved ${blocks.length} blocks (${fetchStats.fetched} fetched, ${fetchStats.cached} from cache)`);
    console.log(`  Downloaded ${fmtMB(fetchStats.bytes)} in ${fmtTime(fetchStats.elapsed_ms)}`);
  }

  if (blocks.length === 0) {
    console.log('  Nothing retrieved.\n');
    return;
  }

  console.log('');
  console.log(`  Height range  ${range.start_height} to ${range.end_height}`);
  if (range.start_time) {
    console.log(`  Time range    ${new Date(range.start_time * 1000).toISOString().slice(0, 16).replace('T', ' ')} to ${new Date(range.end_time * 1000).toISOString().slice(0, 16).replace('T', ' ')}`);
    const hours = (range.end_time - range.start_time) / 3600;
    console.log(`  Span          ${hours.toFixed(1)} hours`);
  }
  console.log('');

  // ------------------------------------------------------------------ features ---
  console.log('  Extracting behavioural features...');
  const t0 = Date.now();
  const { rows, stats } = extractBehaviour(blocks, {
    minTxs: args.minTxs,
    onProgress: (e) => {
      if (e.kind === 'phase') {
        process.stdout.write(`\r    ${e.phase}...                              `);
      } else if (e.kind === 'progress') {
        process.stdout.write(`\r    processing ${e.processed.toLocaleString()}/${e.total.toLocaleString()}   `);
      }
    },
  });
  process.stdout.write('\r' + ' '.repeat(74) + '\r');

  console.log(`  Transactions        ${stats.transactions.toLocaleString()}`);
  console.log(`  Distinct addresses  ${stats.distinct_addresses.toLocaleString()}`);
  console.log(`  With >=${args.minTxs} txs       ${stats.addresses_with_min_activity.toLocaleString()}  (these get analysed)`);
  console.log(`  Feature extraction  ${fmtTime(Date.now() - t0)}`);
  console.log('');

  // ------------------------------------------------------------------- signals ---
  const layering = rows.filter((r) => r.features.layering_score > 0);
  const collectors = rows.filter((r) => r.features.collector_score > 0);
  const peelers = rows.filter((r) => r.features.peel_score > 0.5 && r.features.n_tx_out >= 2);
  const passthrough = rows.filter((r) => r.features.forward_ratio > 0.95 && r.features.received_btc > 0.01);

  console.log('  BEHAVIOURAL PATTERNS DETECTED');
  console.log(`    high-velocity layering   ${layering.length.toLocaleString()}  (split wide, moved within minutes)`);
  console.log(`    fan-in collectors        ${collectors.length.toLocaleString()}  (many senders, swept to few)`);
  console.log(`    peel-chain behaviour     ${peelers.length.toLocaleString()}  (bulk forwarded, small amounts shed)`);
  console.log(`    pure pass-through        ${passthrough.length.toLocaleString()}  (retained almost nothing)`);
  console.log('');

  // --------------------------------------------------------------------- edges ---
  const addressIndex = new Map(rows.map((r, i) => [r.address, i]));
  const edges = buildEdges(blocks, addressIndex);
  console.log(`  Graph edges         ${edges.length.toLocaleString()} (between analysed addresses)`);
  console.log('');

  // -------------------------------------------------------------------- output ---
  const payload = {
    range,
    stats,
    feature_names: MODEL_FEATURES,
    cluster_feature_names: CLUSTER_FEATURES,
    rows: rows.map((r) => ({
      address: r.address,
      first_seen: r.first_seen,
      last_seen: r.last_seen,
      x: MODEL_FEATURES.map((n) => {
        const v = r.features[n];
        return Number.isFinite(v) ? v : 0;
      }),
    })),
    edges: edges.map((e) => [e.from, e.to, e.value_sats, e.n_txs]),
    patterns: {
      layering: layering.length,
      collectors: collectors.length,
      peelers: peelers.length,
      passthrough: passthrough.length,
    },
    source: 'blockchain.info rawblock + blockstream.info block-height, public APIs, no key',
    built_at: new Date().toISOString(),
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(payload), 'utf8');
  const size = fs.statSync(OUT).size;

  console.log(`  Wrote ${OUT}`);
  console.log(`  ${fmtMB(size)}, ${payload.rows.length.toLocaleString()} addresses x ${MODEL_FEATURES.length} features`);
  console.log('');
  console.log('  Next:  python python/pipeline.py       cluster + train + trace');
  console.log('='.repeat(74));
  console.log('');
}

main().catch((err) => {
  console.error('\nIngestion failed:', err.message);
  process.exit(1);
});
