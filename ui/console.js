'use strict';
/**
 * Console behaviour.
 *
 * The rule that shapes every render here: a number is shown with its origin. Chain facts
 * (value, counterparties, timing) are stated plainly because they are observations. Model
 * output is labelled as model output, because it is an opinion produced from weak labels
 * and presenting it as a probability of criminality would be a lie by formatting.
 */

const el = (s) => document.querySelector(s);
const els = (s) => Array.from(document.querySelectorAll(s));

const esc = (s) => String(s == null ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const clip = (a, head = 10, tail = 8) =>
  !a ? '' : a.length <= head + tail + 1 ? a : `${a.slice(0, head)}…${a.slice(-tail)}`;

const n0 = (v) => (Number(v) || 0).toLocaleString();
const btc = (v) => {
  const x = Number(v) || 0;
  if (x === 0) return '0';
  if (Math.abs(x) >= 1000) return x.toFixed(0);
  if (Math.abs(x) >= 1) return x.toFixed(3);
  return x.toFixed(6);
};

function span(seconds) {
  const s = Number(seconds) || 0;
  if (s <= 0) return 'instant';
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
}

function tier(score) {
  if (score >= 0.7) return { cls: 'rose', hex: 'var(--rose)' };
  if (score >= 0.4) return { cls: 'amber', hex: 'var(--amber)' };
  return { cls: '', hex: 'var(--mint)' };
}

async function get(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
  if (!res.ok && data.error) throw new Error(data.error);
  return data;
}

let snackTimer = null;
function snack(msg, bad = false) {
  const s = el('#snack');
  s.textContent = msg;
  s.className = `snack${bad ? ' bad' : ''}`;
  s.hidden = false;
  clearTimeout(snackTimer);
  snackTimer = setTimeout(() => { s.hidden = true; }, bad ? 7000 : 3500);
}

const S = { address: null, plot: null, overview: null, screen: 'triage' };

/* ══════════════════════════════════════════════════════════════════ overview */

async function boot() {
  let o;
  try {
    o = await get('/api/overview');
  } catch (err) {
    snack(err.message, true);
    return;
  }
  S.overview = o;

  if (!o.has_data) {
    el('#cold').hidden = false;
    el('#main').hidden = true;
    el('#chips').innerHTML = '<span class="chip warn">no data</span>';
    el('#cold-blocks').textContent = o.blocks_cached
      ? `${n0(o.blocks_cached)} blocks already cached on disk, so this will be fast.`
      : 'Roughly 29 minutes for two days, cached afterwards.';
    return;
  }

  el('#cold').hidden = true;
  el('#main').hidden = false;

  const chips = [
    `<span class="chip live">blocks <b>${n0(o.range.start_height)}–${n0(o.range.end_height)}</b></span>`,
  ];
  if (o.analysed && o.analysis) {
    const d = o.analysis.device || {};
    chips.push(d.cuda_available
      ? `<span class="chip gpu">GPU <b>${esc(d.gpu_name)}</b></span>`
      : '<span class="chip">CPU <b>no CUDA device</b></span>');
  } else {
    chips.push('<span class="chip warn">not analysed</span>');
  }
  el('#chips').innerHTML = chips.join('');

  renderMetrics(o);
  await loadRanking();

  const first = document.querySelector('.row');
  if (first) openAddress(first.dataset.addr);
}

function renderMetrics(o) {
  const g = o.analysis && o.analysis.gnn;
  const cards = [
    { lbl: 'window', val: `${o.range.hours ?? '?'}h`, sub: `${n0(o.range.retrieved_blocks)} blocks` },
    { lbl: 'transactions', val: n0(o.totals.transactions), sub: 'from real blocks' },
    { lbl: 'addresses', val: n0(o.totals.addresses_analysed), sub: `of ${n0(o.totals.addresses_seen)} seen` },
    { lbl: 'edges', val: n0(o.totals.edges), sub: 'value flows' },
    { lbl: 'layering', val: n0(o.patterns.layering), sub: 'fast wide splits', cls: 'rose' },
    { lbl: 'fan-in', val: n0(o.patterns.collectors), sub: 'many to few', cls: 'amber' },
    { lbl: 'peel chains', val: n0(o.patterns.peelers), sub: '3+ hop sequences', cls: 'cyan' },
    g
      ? { lbl: 'model', val: g.test ? g.test.f1 : '—', sub: `${g.hops}-hop GraphSAGE` }
      : { lbl: 'model', val: '—', sub: 'not trained' },
  ];
  el('#metrics').innerHTML = cards.map((c) => `
    <div class="metric">
      <div class="lbl">${c.lbl}</div>
      <div class="val ${c.cls || ''}">${c.val}</div>
      <div class="sub">${esc(c.sub)}</div>
    </div>`).join('');
}

/* ═══════════════════════════════════════════════════════════════════ ranking */

async function loadRanking() {
  const pattern = el('#f-pattern').value;
  const minBtc = el('#f-value').value;
  const r = await get(`/api/ranking?limit=120&pattern=${pattern}&min_btc=${minBtc}`);

  if (!r.results.length) {
    el('#rows').innerHTML = '';
    el('#rank-note').textContent = r.message || 'Nothing matches those filters.';
    return;
  }

  el('#rank-note').innerHTML =
    `${n0(r.returned)} of ${n0(r.total_ranked)} ranked. Score is <strong>model output</strong>, not a probability of crime.`;

  el('#rows').innerHTML = r.results.map((x) => {
    const t = tier(x.score);
    const pills = x.tags.map((tag) => {
      const k = tag.startsWith('peel') ? 'peel'
        : tag === 'layering' ? 'layering'
        : tag === 'fan-in' ? 'fanin' : '';
      return `<span class="pill ${k}">${esc(tag)}</span>`;
    }).join('');
    return `
      <div class="row" data-addr="${esc(x.address)}">
        <div class="row-1">
          <span class="row-rank">${x.rank}</span>
          <span class="row-addr" title="${esc(x.address)}">${esc(clip(x.address))}</span>
          <span class="row-score" style="color:${t.hex}">${x.score.toFixed(3)}</span>
        </div>
        <div class="row-2">${btc(x.received_btc)} BTC in &middot; ${x.n_tx_in}/${x.n_tx_out} tx &middot; ${x.in_degree}&rarr;${x.out_degree} peers &middot; held ${span(x.holding_seconds)}</div>
        ${pills ? `<div class="pills">${pills}</div>` : ''}
      </div>`;
  }).join('');

  els('.row').forEach((row) => {
    row.addEventListener('click', () => openAddress(row.dataset.addr));
  });
}

/* ═══════════════════════════════════════════════════════════════════ address */

async function openAddress(address) {
  S.address = address;
  els('.row').forEach((r) => r.classList.toggle('is-on', r.dataset.addr === address));
  el('#flow-addr').textContent = clip(address, 14, 10);
  el('#trace-addr').textContent = clip(address, 14, 10);

  el('#detail').innerHTML = '<p class="placeholder">Loading…</p>';
  try {
    const d = await get(`/api/address/${encodeURIComponent(address)}`);
    renderDetail(d);
    if (S.screen === 'flow') loadPlot();
  } catch (err) {
    el('#detail').innerHTML = `<p class="placeholder">${esc(err.message)}</p>`;
  }
}

function renderDetail(d) {
  const f = d.features;
  const t = tier(d.score || 0);

  const peers = (list, dir) => list.length
    ? list.slice(0, 10).map((p) => `
        <div class="peer">
          <a href="#" data-open="${esc(p.address)}" title="${esc(p.address)}">${esc(clip(p.address, 8, 6))}</a>
          <span>${btc(p.value_btc)}</span>
        </div>`).join('')
    : `<p class="placeholder">No ${dir} flows inside this window.</p>`;

  el('#detail').innerHTML = `
    <div class="d-head">
      ${d.score != null ? `
        <div class="d-score">
          <div class="n" style="color:${t.hex}">${d.score.toFixed(3)}</div>
          <div class="c">model score &middot; rank ${d.rank}</div>
        </div>` : ''}
      <div class="d-addr">
        <div class="a">${esc(d.address)}</div>
        <div class="d-actions">
          <button class="btn" data-goto="flow">Money flow</button>
          <button class="btn accent" data-goto="trace">Trace to endpoint</button>
        </div>
      </div>
    </div>

    <div class="facts">
      <div class="fact"><div class="k">received</div><div class="v">${btc(f.received_btc)} BTC</div></div>
      <div class="fact"><div class="k">sent</div><div class="v">${btc(f.sent_btc)} BTC</div></div>
      <div class="fact"><div class="k">balance</div><div class="v">${btc(f.balance_btc)} BTC</div></div>
      <div class="fact"><div class="k">forwarded on</div><div class="v">${(f.forward_ratio * 100).toFixed(0)}%</div></div>
      <div class="fact"><div class="k">held before moving</div><div class="v">${span(f.avg_holding_time)}</div></div>
      <div class="fact"><div class="k">widest split</div><div class="v">${f.number_of_splits} ways</div></div>
      <div class="fact"><div class="k">counterparties</div><div class="v">${f.in_degree} &rarr; ${f.out_degree}</div></div>
      <div class="fact"><div class="k">paid to new addrs</div><div class="v">${(f.percentage_sent_to_new_addresses * 100).toFixed(0)}%</div></div>
    </div>

    ${d.is_exit ? `
      <div class="exitnote">
        <strong>This is an exit point.</strong>
        ${d.exit_kind === 'dormant'
          ? 'Money arrived here and has not moved again inside the window. The trail is parked, not lost.'
          : 'It behaves like a custodial service: very many senders and very many recipients. '
            + 'That is where a records request goes, because such a business is required to hold '
            + 'identity documents. Structural inference, not an identification.'}
      </div>` : ''}

    <h3 class="hdr">Patterns detected</h3>
    <table class="kv">
      <tr><th>High-velocity layering</th><td>${f.layering_score > 0
        ? `${f.layering_score.toFixed(3)} &mdash; split ${f.number_of_splits} ways within ${span(f.avg_holding_time)}`
        : 'not detected'}</td></tr>
      <tr><th>Fan-in collection</th><td>${f.collector_score > 0
        ? `${f.collector_score.toFixed(3)} &mdash; ${f.in_degree} senders into ${f.out_degree} destination(s)`
        : 'not detected'}</td></tr>
      <tr><th>Peel chain</th><td>${(f.peel_chain_length || 0) >= 3
        ? `on a ${Math.round(f.peel_chain_length)}-hop chain`
        : 'not on one'}</td></tr>
      <tr><th>HDBSCAN cohort</th><td>${d.cluster == null ? 'not clustered'
        : d.cluster < 0 ? 'outlier &mdash; too unusual to belong to any cohort'
        : `cluster ${d.cluster}${d.cluster_info ? ` &middot; ${n0(d.cluster_info.size)} members, cohort risk ${d.cluster_info.risk}` : ''}`}</td></tr>
    </table>

    <div class="peers">
      <div><h3 class="hdr">Inbound (${n0(d.n_inbound)})</h3>${peers(d.inbound, 'inbound')}</div>
      <div><h3 class="hdr">Outbound (${n0(d.n_outbound)})</h3>${peers(d.outbound, 'outbound')}</div>
    </div>

    <h3 class="hdr">Provenance</h3>
    <table class="kv">
      <tr><th>Source</th><td>${esc(d.provenance.source)}</td></tr>
      <tr><th>Block range</th><td>${esc(d.provenance.blocks)}</td></tr>
      <tr><th>Ingested</th><td>${esc((d.provenance.ingested_at || '').replace('T', ' ').slice(0, 19))}</td></tr>
    </table>

    <div class="caution">
      A high score means look here first. It is produced from behavioural heuristics and
      density clustering, not from verified criminal labels, and it does not establish that
      any person committed an offence. The chain records addresses, never identities.
    </div>`;

  els('#detail a[data-open]').forEach((a) => {
    a.addEventListener('click', (e) => { e.preventDefault(); openAddress(a.dataset.open); });
  });
  els('#detail [data-goto]').forEach((b) => {
    b.addEventListener('click', () => switchScreen(b.dataset.goto));
  });
}

/* ══════════════════════════════════════════════════════════════════════ plot */

function ensurePlot() {
  if (!S.plot) {
    S.plot = new Plot(el('#cv'), {
      // Clicking a node opens it in Triage. Previously it only swapped the detail pane
      // while leaving you on the graph, so there was no way to get from "that green dot
      // looks interesting" to its full record without hunting for it in the list.
      onPick: (n) => {
        switchScreen('triage');
        openAddress(n.address);
      },
      onHot: (n, ev) => {
        const h = el('#hover');
        if (!n) { h.hidden = true; return; }
        h.innerHTML = `
          <div class="h-a">${esc(n.address)}</div>
          <div class="h-r">score ${(n.score || 0).toFixed(3)} &middot; hop ${n.hops}<br>
          ${btc(n.received_btc)} BTC in &middot; ${n.in_degree}&rarr;${n.out_degree} peers
          ${n.peel_len >= 3 ? `<br>peel chain, ${Math.round(n.peel_len)} hops` : ''}</div>`;
        const r = el('.plot').getBoundingClientRect();
        h.style.left = `${Math.min(ev.clientX - r.left + 16, r.width - 316)}px`;
        h.style.top = `${Math.min(ev.clientY - r.top + 16, r.height - 110)}px`;
        h.hidden = false;
      },
    });
  }
  return S.plot;
}

async function loadPlot() {
  if (!S.address) return;
  const hops = el('#g-hops').value;
  const cap = el('#g-cap').value;
  try {
    const g = await get(`/api/graph/${encodeURIComponent(S.address)}?hops=${hops}&max_nodes=${cap}`);
    el('#plot-empty').hidden = true;
    ensurePlot().load(g);
    // Exit points stated above the graph, not left to be spotted inside it. This is the
    // question the view exists to answer, so it gets a plain-language answer in text as
    // well as a distinct shape on the canvas.
    const bar = el('#exitbar');
    if (g.n_exits > 0) {
      bar.hidden = false;
      bar.innerHTML = `<span class="eb-label">Where the money can leave:</span>` +
        g.exits.slice(0, 8).map((x) => `
          <span class="exitchip" data-open="${esc(x.address)}" title="${esc(x.exit_why || '')}">
            ${x.exit_kind === 'dormant' ? '■ dormant' : '■ exchange-like'}
            ${esc(clip(x.address, 6, 5))}
            <b>${btc(x.received_btc)} BTC</b>
          </span>`).join('') +
        (g.n_exits > 8 ? `<span class="exitcount">+${g.n_exits - 8} more</span>` : '');
      els('#exitbar .exitchip').forEach((chip) => {
        chip.addEventListener('click', () => {
          switchScreen('triage');
          openAddress(chip.dataset.open);
        });
      });
    } else {
      bar.hidden = false;
      bar.innerHTML = '<span class="eb-label">No exit point inside this view.</span>' +
        '<span class="exitcount">The money has not reached an exchange-like address or ' +
        'stopped moving within these hops. Raise the hop count, or run Trace, which ' +
        'follows it past the edge of the loaded window.</span>';
    }
    el('#exit-count').textContent = g.n_exits ? `${g.n_exits} exit point(s)` : '';

    el('#plot-note').textContent = g.note
      || `${n0(g.nodes.length)} addresses, ${n0(g.edges.length)} flows within ${g.hops} hops.`;

    // The exit bar is inserted above the canvas, which changes the plot's height after
    // the canvas has already sized itself. Re-measure on the next frame, or the graph
    // draws into a box that no longer matches what is on screen.
    requestAnimationFrame(() => {
      S.plot._fitCanvas();
      S.plot.fit();
    });
  } catch (err) {
    snack(err.message, true);
  }
}

/* ═════════════════════════════════════════════════════════════════════ trace */

async function runTrace() {
  if (!S.address) { snack('Open an address first.', true); return; }
  const btn = el('#t-run');
  btn.disabled = true;
  el('#trace-out').innerHTML =
    '<p class="placeholder">Following the money. This queries live block explorers, so it can take a minute…</p>';

  try {
    const r = await get('/api/trace', {
      method: 'POST',
      body: {
        address: S.address,
        hops: parseInt(el('#t-hops').value, 10),
        budget: parseInt(el('#t-budget').value, 10),
      },
    });

    if (!r.ok) {
      el('#trace-out').innerHTML = `<p class="placeholder">${esc(r.error)}</p>`;
      return;
    }

    const v = r.verdict;
    const stops = r.endpoints.map((e) => `
      <div class="stop ${esc(e.kind)}">
        <div class="stop-h">
          <span class="kind">${esc(e.kind)}</span>
          <span class="stop-v">${esc(e.value_btc)}</span>
          <span class="stop-m">${(Math.min(1, e.share_of_source) * 100).toFixed(1)}% of source &middot; ${e.hops} hops${e.n_tx ? ` &middot; ${n0(e.n_tx)} txs on this address` : ''}</span>
        </div>
        <div class="hop-r">${esc(e.address)}</div>
        ${e.detail ? `<p>${esc(e.detail)}</p>` : ''}
        ${e.deposit_candidate ? `
          <div class="deposit">
            <div class="dl">likely customer deposit address</div>
            <div class="da">${esc(e.deposit_candidate)}</div>
            <p style="margin:6px 0 0;font-size:11px;color:var(--ink-2)">
              This is the account the venue holds identity records for. A records request
              goes here.</p>
          </div>` : ''}
        ${e.path && e.path.length ? `
          <div class="hops">${e.path.map((p, i) => `
            <div class="hop">
              <span class="hop-n">${i + 1}</span>
              <span class="hop-r">${esc(clip(p.from, 8, 6))} &rarr; ${esc(clip(p.to, 8, 6))}
                ${p.via === 'peel_chain' ? `<span style="color:var(--violet)">peel &times;${p.peel_length}</span>` : ''}
                <span class="hop-tx">${esc(p.txid)}</span></span>
              <span class="hop-v">${btc(p.value_sats / 1e8)} BTC</span>
            </div>`).join('')}</div>` : ''}
      </div>`).join('');

    el('#trace-out').innerHTML = `
      <div class="verdict">
        <h3>${esc(v.headline)}</h3>
        <p>${esc(v.detail)}</p>
        ${v.next_step ? `<p><strong>Next step:</strong> ${esc(v.next_step)}</p>` : ''}
      </div>
      ${stops || '<p class="placeholder">No endpoint reached inside the budget.</p>'}
      <p class="stop-m" style="margin-top:14px">
        ${n0(r.stats.addresses_examined)} addresses examined &middot;
        ${n0(r.stats.fetches)} network lookups &middot;
        ${n0(r.stats.pruned_branches)} branches pruned below the value floor
        ${r.stats.budget_exhausted ? ' &middot; budget exhausted, trail may continue' : ''}
      </p>`;
  } catch (err) {
    el('#trace-out').innerHTML = `<p class="placeholder">${esc(err.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
}

/* ═══════════════════════════════════════════════════════════════════ cohorts */

async function loadCohorts() {
  const r = await get('/api/clusters');
  if (!r.clusters.length) {
    el('#cohorts-out').innerHTML = '<p class="placeholder">No cohorts. Run the pipeline first.</p>';
    return;
  }
  const top = Math.max(...r.clusters.map((c) => c.risk), 0.001);
  el('#cohorts-out').innerHTML = `
    <p class="stop-m" style="margin-bottom:16px">
      ${esc(r.algorithm)}. ${n0(r.n_clusters)} cohorts found;
      ${n0(r.n_noise)} addresses left as outliers rather than forced into one.
      ${r.subsampled ? `Fitted on ${n0(r.fitted_on)} of ${n0(r.total_points)} addresses, the rest assigned to the nearest cohort.` : ''}
    </p>
    ${r.clusters.map((c) => `
      <div class="cohort">
        <span class="cohort-id">#${c.cluster}</span>
        <span class="meter"><i style="width:${(c.risk / top) * 100}%"></i></span>
        <span class="cohort-v">${c.risk.toFixed(3)}</span>
      </div>
      <p class="cohort-m">${n0(c.size)} addresses &middot; layering ${c.median_layering.toFixed(2)}
        &middot; fan-in ${c.median_collector.toFixed(2)} &middot; peel ${c.median_peel.toFixed(2)}
        &middot; held ${span(c.median_holding_seconds)}</p>`).join('')}
    <div class="caution">Cohort risk is the median behavioural score across its members.
      A group behaving alike is a lead worth opening, not a finding.</div>`;
}

/* ═════════════════════════════════════════════════════════════════════ model */

function renderModel() {
  const o = S.overview;
  if (!o || !o.analysis) {
    el('#model-out').innerHTML = '<p class="placeholder">Not analysed yet. Run: python python/pipeline.py</p>';
    return;
  }
  const a = o.analysis;
  const g = a.gnn;
  const d = a.device || {};

  el('#model-out').innerHTML = `
    <h3 class="hdr">Hardware</h3>
    <table class="kv">
      <tr><th>Device used for training</th><td>${d.cuda_available
        ? `CUDA &mdash; ${esc(d.gpu_name)}, ${d.vram_gb} GB, CUDA ${esc(d.cuda_version)}`
        : 'CPU &mdash; no CUDA device present on this machine'}</td></tr>
      <tr><th>PyTorch</th><td>${esc(d.torch_version || 'unknown')}</td></tr>
    </table>

    ${g ? `
      <h3 class="hdr">Graph neural network</h3>
      <table class="kv">
        <tr><th>Architecture</th><td>${esc(g.architecture)}</td></tr>
        <tr><th>Reach</th><td>${g.hops} hops of the money flow</td></tr>
        <tr><th>Graph</th><td>${n0(g.nodes)} nodes, ${n0(g.edges)} edges</td></tr>
        <tr><th>Training</th><td>${g.epochs} epochs in ${g.seconds}s on ${esc(g.device).toUpperCase()}</td></tr>
        <tr><th>Split</th><td>${n0(g.split.train)} train / ${n0(g.split.val)} val / ${n0(g.split.test)} test</td></tr>
        <tr><th>Held-out F1</th><td>${g.test ? g.test.f1 : '—'} (precision ${g.test ? g.test.precision : '—'}, recall ${g.test ? g.test.recall : '—'})</td></tr>
        <tr><th>AUC-PR</th><td>${g.auc_pr ?? '—'}</td></tr>
      </table>` : ''}

    <h3 class="hdr">Clustering</h3>
    <table class="kv">
      <tr><th>Algorithm</th><td>HDBSCAN, leaf selection, on log-scaled standardised features</td></tr>
      <tr><th>Cohorts</th><td>${n0(a.clustering.n_clusters)}</td></tr>
      <tr><th>Outliers</th><td>${n0(a.clustering.n_noise)} addresses too unusual to cluster</td></tr>
      <tr><th>Runtime</th><td>${a.clustering.elapsed_s}s</td></tr>
      ${a.clustering.subsampled ? `<tr><th>Fitted on</th><td>${n0(a.clustering.fitted_on)} of ${n0(a.clustering.total_points)}, remainder assigned to nearest cohort</td></tr>` : ''}
    </table>

    <h3 class="hdr">Training labels</h3>
    <table class="kv">
      <tr><th>Positive</th><td>${n0(a.labels.n_positive)}</td></tr>
      <tr><th>Negative</th><td>${n0(a.labels.n_negative)}</td></tr>
      <tr><th>Confirmed external labels</th><td>${n0(a.labels.n_confirmed)}</td></tr>
      ${Object.entries(a.labels.by_source || {}).map(([k, v]) =>
        `<tr><th>&nbsp;&nbsp;from ${esc(k.replace(/_/g, ' '))}</th><td>${n0(v)}</td></tr>`).join('')}
    </table>

    <div class="caution">${esc(a.labels.caveat)}</div>

    <h3 class="hdr">Data provenance</h3>
    <table class="kv">
      <tr><th>Source</th><td>${esc(o.source)}</td></tr>
      <tr><th>Blocks</th><td>${n0(o.range.start_height)} to ${n0(o.range.end_height)}</td></tr>
      <tr><th>Chain time covered</th><td>${o.range.hours}h</td></tr>
      <tr><th>Ingested at</th><td>${esc((o.ingested_at || '').replace('T', ' ').slice(0, 19))}</td></tr>
    </table>`;
}

/* ═══════════════════════════════════════════════════════════════════ sources */

let SRC = null;

async function loadSources() {
  SRC = await get('/api/sources');
  const cfg = SRC.config;
  const w = cfg.window || {};

  const opt = (x) => `
    <label class="opt ${cfg.provider === x.id ? 'on' : ''}" data-provider="${esc(x.id)}">
      <input type="radio" name="provider" value="${esc(x.id)}" ${cfg.provider === x.id ? 'checked' : ''}>
      <span class="o-main">
        <span class="o-title">${esc(x.label)}
          ${x.verified ? '<span class="badge-ok">VERIFIED HERE</span>'
            : x.id === 'custom' ? '' : '<span class="badge-no">UNREACHABLE HERE</span>'}
          ${x.needs_key ? '<span class="badge-no">MAY NEED KEY</span>' : '<span class="badge-ok">NO LOGIN</span>'}
        </span>
        <span class="o-note">${esc(x.note)}</span>
      </span>
    </label>`;

  el('#sources-out').innerHTML = `
    <div class="srcgroup">
      <h3 class="hdr">Where blocks come from</h3>
      <p class="hint">Every preset is public and needs no account. Pick one, or point at your own endpoint.</p>
      ${SRC.presets.map(opt).join('')}
    </div>

    <div class="srcgroup" id="customwrap" ${cfg.provider === 'custom' ? '' : 'hidden'}>
      <h3 class="hdr">Custom endpoint</h3>
      <p class="hint">Placeholders: {hash} {height} {address} {offset} {key}</p>
      <div class="field"><label>block by hash</label>
        <input type="text" id="c-block" value="${esc((cfg.custom || {}).block || '')}" placeholder="https://provider/api/block/{hash}"></div>
      <div class="field"><label>height to hash</label>
        <input type="text" id="c-hash" value="${esc((cfg.custom || {}).block_hash || '')}" placeholder="https://provider/api/block-height/{height}"></div>
      <div class="field"><label>chain tip</label>
        <input type="text" id="c-tip" value="${esc((cfg.custom || {}).tip || '')}" placeholder="https://provider/api/tip"></div>
      <div class="field"><label>API key</label>
        <input type="password" id="c-key" placeholder="${cfg.has_key ? 'already set, leave blank to keep' : 'optional'}"></div>
    </div>

    <div class="srcgroup">
      <h3 class="hdr">How much chain to pull</h3>
      <p class="hint">Any value you like. Roughly 144 blocks per day, about 6 seconds per block on a first fetch.</p>
      <div class="winrow">
        <label class="opt ${w.mode !== 'blocks' ? 'on' : ''}" data-mode="days" style="flex:1">
          <input type="radio" name="wmode" value="days" ${w.mode !== 'blocks' ? 'checked' : ''}>
          <span class="o-main">
            <span class="o-title">By days</span>
            <span class="field" style="margin-top:7px">
              <input type="number" id="w-days" min="0.1" step="0.1" value="${w.days || 2}" style="max-width:90px">
              <span class="unit">days back from the tip</span>
            </span>
          </span>
        </label>
        <label class="opt ${w.mode === 'blocks' ? 'on' : ''}" data-mode="blocks" style="flex:1">
          <input type="radio" name="wmode" value="blocks" ${w.mode === 'blocks' ? 'checked' : ''}>
          <span class="o-main">
            <span class="o-title">By block count</span>
            <span class="field" style="margin-top:7px">
              <input type="number" id="w-blocks" min="1" step="1" value="${w.blocks || 144}" style="max-width:90px">
              <span class="unit">blocks</span>
            </span>
          </span>
        </label>
      </div>
      <div class="field" style="margin-top:10px">
        <label>end at height</label>
        <input type="number" id="w-end" value="${w.end_height || ''}" placeholder="blank means the current chain tip" style="max-width:200px">
      </div>
      <p class="est" id="w-est"></p>
    </div>

    <div class="srcgroup">
      <h3 class="hdr">Or load a file you already have</h3>
      <p class="hint">CSV, JSON or JSONL of transactions, auto-detected from the columns.
        Run it from the terminal:</p>
      <div class="steps"><li><span class="step-n">&rarr;</span><div>
        <code>node scripts/ingest.js --file /path/to/your-data.csv</code>
      </div></li></div>
    </div>`;

  const estimate = () => {
    const picked = document.querySelector('input[name=wmode]:checked');
    const mode = picked ? picked.value : 'days';
    const blocks = mode === 'blocks'
      ? parseInt(el('#w-blocks').value, 10) || 0
      : Math.round((parseFloat(el('#w-days').value) || 0) * 144);
    const mins = Math.round((blocks * 6) / 60);
    el('#w-est').textContent = blocks
      ? `About ${n0(blocks)} blocks, roughly ${mins} minute(s) on a first fetch, instant once cached.`
      : '';
  };

  els('.opt[data-provider]').forEach((o) => {
    o.addEventListener('click', () => {
      els('.opt[data-provider]').forEach((x) => x.classList.toggle('on', x === o));
      el('#customwrap').hidden = o.dataset.provider !== 'custom';
    });
  });
  els('.opt[data-mode]').forEach((o) => {
    o.addEventListener('click', () => {
      els('.opt[data-mode]').forEach((x) => x.classList.toggle('on', x === o));
      estimate();
    });
  });
  ['#w-days', '#w-blocks'].forEach((id) => el(id).addEventListener('input', estimate));
  estimate();
}

async function saveSources() {
  const pv = document.querySelector('input[name=provider]:checked');
  const wm = document.querySelector('input[name=wmode]:checked');
  const provider = pv ? pv.value : 'blockchain.info';
  const mode = wm ? wm.value : 'days';

  const body = {
    provider,
    window: {
      mode,
      days: parseFloat(el('#w-days').value) || 2,
      blocks: parseInt(el('#w-blocks').value, 10) || null,
      end_height: parseInt(el('#w-end').value, 10) || null,
    },
  };
  if (provider === 'custom') {
    body.custom = {
      block: el('#c-block').value.trim(),
      block_hash: el('#c-hash').value.trim(),
      tip: el('#c-tip').value.trim(),
      address: '',
    };
    const key = el('#c-key').value;
    if (key) body.api_key = key;
  }

  try {
    await get('/api/sources', { method: 'POST', body });
    snack('Saved. Use Pull chain data to fetch with these settings.');
    await loadSources();
  } catch (err) {
    snack(err.message, true);
  }
}

/* ════════════════════════════════════════════════════════════════ run stages */

/**
 * Ingestion and training are minutes long, so they run as background processes and
 * stream their output here. The interface stays usable throughout; the alternative is a
 * frozen window and no way to tell a slow job from a hung one.
 */
function runLogOpen(title) {
  const box = el('#runlog');
  box.hidden = false;
  box.innerHTML = `<div class="rl-head"><span class="spin"></span><span id="rl-title">${esc(title)}</span></div>`;
}

function runLogLine(line, cls = '') {
  const box = el('#runlog');
  if (box.hidden) return;
  const div = document.createElement('div');
  div.className = `rl-line ${cls}`;
  div.textContent = line;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function runLogDone(ok, label) {
  const spin = el('#runlog .spin');
  if (spin) spin.remove();
  const title = el('#rl-title');
  if (title) title.textContent = ok ? `${label} finished` : `${label} failed`;
  runLogLine(ok ? '✓ done' : '✗ failed', ok ? 'ok' : 'bad');
}

async function runStage(kind) {
  if (!window.desktop) {
    snack('Run these from the terminal in browser mode: node scripts/ingest.js --days 2', true);
    return;
  }

  const ingestBtn = el('#r-ingest');
  const trainBtn = el('#r-train');
  ingestBtn.disabled = true;
  trainBtn.disabled = true;

  const label = kind === 'ingest' ? 'Pulling chain data' : 'Clustering and training';
  runLogOpen(label);

  try {
    if (kind === 'ingest') {
      // The window comes from the saved source configuration, so the number lives in
      // one place instead of being duplicated between a control and the config file.
      await window.desktop.ingest({});
    } else {
      await window.desktop.analyse();
    }
    // Re-read everything: new blocks, new model, new ranking.
    await boot();
    if (S.screen === 'model') renderModel();
  } catch (err) {
    runLogLine(err.message, 'bad');
  } finally {
    ingestBtn.disabled = false;
    trainBtn.disabled = false;
  }
}

if (window.desktop) {
  window.desktop.on('stage:line', (p) => runLogLine(p.line));
  window.desktop.on('stage:done', (p) => runLogDone(p.ok, p.name === 'ingest' ? 'Ingest' : 'Training'));
  window.desktop.on('nav', (screen) => switchScreen(screen));
}

/* ══════════════════════════════════════════════════════════════════ wiring */

function switchScreen(name) {
  S.screen = name;
  els('.rail-btn').forEach((b) => b.classList.toggle('is-on', b.dataset.screen === name));
  els('.screen').forEach((s) => s.classList.toggle('is-on', s.id === `s-${name}`));

  if (name === 'flow' && S.address) {
    loadPlot();
    setTimeout(() => S.plot && S.plot.fit(), 60);
  }
  if (name === 'cohorts') loadCohorts();
  if (name === 'sources') loadSources();
  if (name === 'model') renderModel();
}

els('.rail-btn').forEach((b) => b.addEventListener('click', () => switchScreen(b.dataset.screen)));

el('#f-pattern').addEventListener('change', loadRanking);
el('#f-value').addEventListener('change', loadRanking);
el('#g-hops').addEventListener('change', loadPlot);
el('#g-cap').addEventListener('change', loadPlot);
el('#g-hops').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadPlot(); });
el('#g-cap').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadPlot(); });
el('#g-fit').addEventListener('click', () => S.plot && S.plot.fit());
el('#t-run').addEventListener('click', runTrace);

const ingestBtn = el('#r-ingest');
const trainBtn = el('#r-train');
if (ingestBtn) ingestBtn.addEventListener('click', () => runStage('ingest'));
const srcSave = el('#src-save');
if (srcSave) srcSave.addEventListener('click', saveSources);
if (trainBtn) trainBtn.addEventListener('click', () => runStage('train'));

el('#q').addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  const v = e.target.value.trim();
  if (!v) return;
  switchScreen('triage');
  openAddress(v);
  e.target.value = '';
});

boot();
