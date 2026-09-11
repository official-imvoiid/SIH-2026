'use strict';
/**
 * Trace real money from a real Bitcoin address to wherever it stops.
 *
 *   node scripts/trace.js <address> [--hops 6] [--budget 120] [--offline]
 *   node scripts/trace.js --ransomware          pick a real confirmed ransom wallet
 *   node scripts/trace.js --fetch-labels        download the ransomware and sanctions lists
 *
 * Everything printed here comes from the chain. If a number cannot be traced to a
 * transaction id it is not printed.
 */

const { ChainClient } = require('../src/chain/client');
const { fetchRansomware, fetchSanctions } = require('../src/chain/fetch');
const { traceToEndpoint, formatPath } = require('../src/analysis/trace');
const { humanBtc, shortAddress, isRealAddress } = require('../src/chain/model');

function parseArgs(argv) {
  const args = { hops: 6, budget: 120, offline: false, address: null, mode: 'trace' };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--hops') args.hops = parseInt(argv[++i], 10);
    else if (a === '--budget') args.budget = parseInt(argv[++i], 10);
    else if (a === '--offline') args.offline = true;
    else if (a === '--ransomware') args.mode = 'ransomware';
    else if (a === '--fetch-labels') args.mode = 'labels';
    else if (!a.startsWith('--')) args.address = a;
  }
  return args;
}

const rule = (c = '-') => console.log(c.repeat(78));
const ts = (unix) => (unix ? new Date(unix * 1000).toISOString().slice(0, 16).replace('T', ' ') : 'unconfirmed');

async function main() {
  const args = parseArgs(process.argv);

  // ---------------------------------------------------------------- label download ---
  if (args.mode === 'labels') {
    console.log('\nDownloading real label sources. Neither needs a login.\n');

    process.stdout.write('  Ransomwhere confirmed ransomware addresses ... ');
    const rw = await fetchRansomware();
    console.log(`${rw.n_addresses.toLocaleString()} addresses, ${rw.n_families} families`);
    console.log(`    ${rw.from_cache ? 'from local cache' : 'downloaded'}, ${rw.n_payments.toLocaleString()} real payments`);

    process.stdout.write('  US Treasury OFAC sanctions list ............. ');
    const ofac = await fetchSanctions();
    console.log(`${ofac.n_addresses.toLocaleString()} sanctioned addresses`);
    console.log(`    ${ofac.from_cache ? 'from local cache' : 'downloaded'}`);

    console.log('\nBoth cached under data/chain/labels. Run a trace next:');
    console.log('  node scripts/trace.js --ransomware\n');
    return;
  }

  // --------------------------------------------------------------- choose an address ---
  let address = args.address;
  let context = null;

  if (args.mode === 'ransomware' || !address) {
    const rw = await fetchRansomware();
    // Pick a wallet that actually received several payments and still moved money,
    // so the trace has something real to follow.
    const candidates = Object.values(rw.addresses)
      .filter((a) => a.n_payments >= 3 && a.family !== 'Unlabeled')
      .sort((a, b) => b.n_payments - a.n_payments);

    if (candidates.length === 0) {
      console.error('No suitable ransomware address found in the cached list.');
      process.exit(1);
    }
    const pick = candidates[Math.floor(Math.random() * Math.min(20, candidates.length))];
    address = pick.address;
    context = pick;
  }

  if (!isRealAddress(address)) {
    console.error(`\n"${address}" is not a valid Bitcoin address.\n`);
    process.exit(1);
  }

  // ------------------------------------------------------------------------- labels ---
  const sanctioned = new Set();
  try {
    const ofac = await fetchSanctions();
    for (const a of Object.keys(ofac.addresses)) sanctioned.add(a);
  } catch (err) {
    // Sanctions screening adds a flag; it is not a prerequisite for tracing money.
    console.log(`  (sanctions screening unavailable: ${err.message.split('\n')[0]})`);
  }

  // -------------------------------------------------------------------------- trace ---
  console.log('');
  rule('=');
  console.log('  ENDPOINT TRACE');
  rule('=');
  console.log(`  Source address   ${address}`);
  if (context) {
    console.log(`  Known as         confirmed ${context.family} ransomware wallet`);
    console.log(`  Attribution      Ransomwhere, ${context.n_payments} recorded ransom payments`);
  }
  console.log(`  Hop limit        ${args.hops} logical hops (a peel chain counts as one)`);
  console.log(`  Lookup budget    ${args.budget} addresses`);
  console.log('');

  const client = new ChainClient({
    offline: args.offline,
    onProgress: (e) => {
      if (e.kind === 'examine') {
        process.stdout.write(
          `\r  examining ${shortAddress(e.address).padEnd(18)} hop ${e.hops}  ` +
            `${humanBtc(e.value_sats).padStart(14)}  [${e.fetches}/${e.budget}]   `
        );
      } else if (e.kind === 'rate_limited') {
        process.stdout.write(
          `\r  ${e.provider} rate limited, switching provider`.padEnd(78) + '\n'
        );
      } else if (e.kind === 'all_benched') {
        process.stdout.write(
          `\r  all providers rate limited, waiting ${Math.round(e.waitMs / 1000)}s`.padEnd(78)
        );
      }
    },
  });

  const result = await traceToEndpoint(client, address, {
    maxHops: args.hops,
    fetchBudget: args.budget,
    sanctioned,
    onProgress: client.onProgress,
  });

  process.stdout.write('\r' + ' '.repeat(78) + '\r');

  if (!result.ok) {
    console.log(`  ${result.error}\n`);
    return;
  }

  // ------------------------------------------------------------------------ verdict ---
  rule();
  console.log(`  ${result.verdict.headline}`);
  rule();
  console.log(`  ${wrap(result.verdict.detail, 74, '  ')}`);
  if (result.verdict.next_step) {
    console.log('');
    console.log(`  NEXT STEP`);
    console.log(`  ${wrap(result.verdict.next_step, 74, '  ')}`);
  }
  console.log('');

  // ---------------------------------------------------------------------- value flow ---
  if (result.value_by_hop.length) {
    console.log('  VALUE BY HOP');
    const max = Math.max(...result.value_by_hop.map((v) => v.value_sats));
    for (const v of result.value_by_hop) {
      const bar = '#'.repeat(Math.max(1, Math.round((v.value_sats / max) * 34)));
      console.log(`    hop ${String(v.hop).padStart(2)}  ${v.value_btc.padStart(14)}  ${bar}`);
    }
    console.log('');
  }

  // ------------------------------------------------------------------------ endpoints ---
  console.log(`  ENDPOINTS FOUND: ${result.endpoints.length}`);
  console.log('');

  for (const [i, e] of result.endpoints.slice(0, 8).entries()) {
    console.log(`  [${i + 1}] ${e.kind}  ${humanBtc(e.value_sats)}  (${(Math.min(1, e.share_of_source) * 100).toFixed(1)}% of source)`);
    console.log(`      address   ${e.address}`);
    console.log(`      reached   ${e.hops} hop(s) from the source`);
    if (e.n_tx != null) console.log(`      activity  ${Number(e.n_tx).toLocaleString()} transactions on this address`);
    if (e.confidence) console.log(`      certainty ${(e.confidence * 100).toFixed(0)}% (${e.basis.replace(/_/g, ' ')})`);
    if (e.detail) console.log(`      ${wrap(e.detail, 68, '      ')}`);
    if (e.deposit_candidate) {
      console.log('');
      console.log(`      LIKELY CUSTOMER DEPOSIT ADDRESS`);
      console.log(`      ${e.deposit_candidate}`);
      console.log(`      This is the account the venue holds identity documents for.`);
    }
    if (e.path && e.path.length) {
      console.log('');
      console.log('      ROUTE (every hop is a real transaction)');
      console.log(formatPath(e));
    }
    console.log('');
  }

  // ---------------------------------------------------------------------- peel chains ---
  if (result.peel_chains.length) {
    console.log(`  PEEL CHAINS DETECTED: ${result.peel_chains.length}`);
    for (const c of result.peel_chains.slice(0, 3)) {
      console.log(`    ${c.length} links, shed ${humanBtc(c.peeled_sats)} along the way`);
      console.log(`    first tx ${c.links[0].txid}  ${ts(c.links[0].time)}`);
      console.log(`    ends at  ${c.ended_at}`);
    }
    console.log('');
  }

  // --------------------------------------------------------------------------- stats ---
  const s = result.stats;
  rule();
  console.log('  HOW THIS TRACE WAS DONE');
  console.log(`    addresses examined    ${s.addresses_examined}`);
  console.log(`    network lookups       ${s.fetches} of ${s.fetch_budget}${s.budget_exhausted ? '  (budget exhausted)' : ''}`);
  console.log(`    branches pruned       ${s.pruned_branches}, carrying ${humanBtc(s.pruned_value_sats)} below the floor`);
  console.log(`    value floor           ${humanBtc(s.value_floor_sats)} per branch`);
  if (s.unexplored_branches) {
    console.log(`    left unexplored       ${s.unexplored_branches} branches, ${humanBtc(s.unexplored_value_sats)}`);
  }
  console.log(`    elapsed               ${(s.elapsed_ms / 1000).toFixed(1)}s`);
  const cs = client.stats();
  console.log(`    cache                 ${cs.cache.hits} hits, ${cs.cache.misses} misses, ${cs.cache.writes} written`);
  console.log('');
  console.log('    providers used        ' +
    cs.providers.filter((p) => p.successes > 0).map((p) => `${p.name} (${p.successes})`).join(', '));
  const limited = cs.providers.filter((p) => p.refusals > 0 || !p.available);
  if (limited.length) {
    console.log('    rate limited          ' + limited.map((p) => p.name).join(', '));
  }
  console.log('');
  console.log('  Every address and transaction id above came from a public block explorer');
  console.log('  and is cached locally. Re-running this trace needs no network.');
  rule('=');
  console.log('');
}

function wrap(text, width, indent) {
  if (!text) return '';
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = '';
  for (const w of words) {
    if ((line + ' ' + w).trim().length > width) {
      lines.push(line.trim());
      line = w;
    } else {
      line += ' ' + w;
    }
  }
  if (line.trim()) lines.push(line.trim());
  return lines.join('\n' + indent);
}

main().catch((err) => {
  console.error('\nTrace failed:', err.message);
  process.exit(1);
});
