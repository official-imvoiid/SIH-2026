'use strict';
/**
 * Where chain data comes from, and how that choice is configured.
 *
 * Three ways to get data in, none of them requiring a login:
 *
 *   1. A preset public API. These are the ones verified to answer from an ordinary
 *      machine with no key and no account.
 *   2. A custom endpoint. Paste a URL template and, if the service needs one, a key.
 *      This is how you attach a paid provider later without touching the code.
 *   3. A file you already have. CSV, JSON or JSONL of transactions.
 *
 * The configuration lives in data/chain/sources.json rather than being compiled in, so
 * changing provider is an edit in the interface rather than a code change and a restart.
 */

const fs = require('fs');
const path = require('path');

const CONFIG = path.resolve(__dirname, '..', '..', 'data', 'chain', 'sources.json');

/**
 * Verified working from this machine with no credentials.
 *
 * mempool.space is listed but marked unreachable: it times out from here, and recording
 * that is more useful than silently omitting it, because on another network it may be the
 * best option available.
 */
const PRESETS = {
  'blockchain.info': {
    id: 'blockchain.info',
    label: 'blockchain.info',
    needs_key: false,
    verified: true,
    block: 'https://blockchain.info/rawblock/{hash}',
    address: 'https://blockchain.info/rawaddr/{address}?limit=50&offset={offset}',
    note: 'Returns an entire block, roughly 4,900 transactions, in one request.',
  },
  'blockstream.info': {
    id: 'blockstream.info',
    label: 'blockstream.info (Esplora)',
    needs_key: false,
    verified: true,
    tip: 'https://blockstream.info/api/blocks/tip/height',
    block_hash: 'https://blockstream.info/api/block-height/{height}',
    address: 'https://blockstream.info/api/address/{address}',
    note: 'Used for block heights and hashes. Paginates transactions 25 at a time.',
  },
  'mempool.space': {
    id: 'mempool.space',
    label: 'mempool.space',
    needs_key: false,
    verified: false,
    tip: 'https://mempool.space/api/blocks/tip/height',
    block_hash: 'https://mempool.space/api/block-height/{height}',
    address: 'https://mempool.space/api/address/{address}',
    note: 'Did not respond from this machine when tested. May work on another network.',
  },
  'custom': {
    id: 'custom',
    label: 'Custom endpoint',
    needs_key: true,
    verified: false,
    block: '',
    address: '',
    note: 'Paste a URL template. Use {hash}, {height}, {address}, {offset} as placeholders.',
  },
};

const DEFAULTS = {
  provider: 'blockchain.info',
  height_provider: 'blockstream.info',
  api_key: '',
  custom: { block: '', address: '', tip: '', block_hash: '' },
  window: {
    // Not a hardcoded two days. This is the starting value; the interface writes back
    // whatever the user chooses and the ingester reads it from here.
    mode: 'days',
    days: 2,
    blocks: null,
    end_height: null,
  },
  upload: { path: null, format: null, loaded_at: null },
  min_delay_ms: 400,
};

function read() {
  if (!fs.existsSync(CONFIG)) return { ...DEFAULTS };
  try {
    const saved = JSON.parse(fs.readFileSync(CONFIG, 'utf8'));
    return {
      ...DEFAULTS,
      ...saved,
      custom: { ...DEFAULTS.custom, ...(saved.custom || {}) },
      window: { ...DEFAULTS.window, ...(saved.window || {}) },
      upload: { ...DEFAULTS.upload, ...(saved.upload || {}) },
    };
  } catch {
    // A corrupt config must not block ingestion; fall back rather than refuse to start.
    return { ...DEFAULTS };
  }
}

function write(patch) {
  const next = { ...read(), ...patch };
  fs.mkdirSync(path.dirname(CONFIG), { recursive: true });
  fs.writeFileSync(CONFIG, JSON.stringify(next, null, 2), 'utf8');
  return next;
}

/**
 * Resolve the URL templates in use, with the custom entries substituted in where the
 * provider is 'custom'. Keys are never written into the returned note text.
 */
function resolve() {
  const cfg = read();
  const preset = PRESETS[cfg.provider] || PRESETS['blockchain.info'];
  const heightPreset = PRESETS[cfg.height_provider] || PRESETS['blockstream.info'];

  const pick = (fromCustom, fromPreset) =>
    cfg.provider === 'custom' && fromCustom ? fromCustom : fromPreset;

  return {
    provider: cfg.provider,
    height_provider: cfg.height_provider,
    has_key: Boolean(cfg.api_key),
    block: pick(cfg.custom.block, preset.block),
    address: pick(cfg.custom.address, preset.address),
    tip: pick(cfg.custom.tip, heightPreset.tip),
    block_hash: pick(cfg.custom.block_hash, heightPreset.block_hash),
    api_key: cfg.api_key || '',
    min_delay_ms: cfg.min_delay_ms,
    window: cfg.window,
    upload: cfg.upload,
  };
}

/** Fill placeholders and append the key when one is configured. */
function url(template, params = {}, apiKey = '') {
  if (!template) return null;
  let out = template;
  for (const [k, v] of Object.entries(params)) {
    out = out.split(`{${k}}`).join(String(v));
  }
  if (apiKey && out.includes('{key}')) out = out.split('{key}').join(apiKey);
  else if (apiKey && !out.includes('key=')) {
    out += (out.includes('?') ? '&' : '?') + `key=${encodeURIComponent(apiKey)}`;
  }
  return out;
}

module.exports = { PRESETS, DEFAULTS, read, write, resolve, url, CONFIG };
