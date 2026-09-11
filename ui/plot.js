'use strict';
/**
 * Money-flow plot. Written fresh for this console.
 *
 * Differences from the graph in the previous build, all of them deliberate:
 *
 *   Layout is deterministic, not random. Nodes are seeded on concentric rings by hop
 *   distance and the simulation refines from there, so the same address always produces
 *   the same picture. Previously each open produced a different arrangement, which makes
 *   a screenshot impossible to compare against the thing it was a screenshot of.
 *
 *   Repulsion runs on a spatial grid rather than every pair. The old version was O(n^2)
 *   per tick, which is fine at 80 nodes and visibly stutters at 300. Bucketing by cell
 *   and only comparing within a neighbourhood makes it near linear, so the cap can be
 *   raised to 500 without the frame rate collapsing.
 *
 *   Edges are drawn as curves with the arrowhead at the midpoint. With straight lines and
 *   end-mounted arrows, reciprocal pairs overlap exactly and the direction of the money
 *   becomes unreadable, which matters more here than anywhere else in the interface.
 */

(function (global) {
  const PALETTE = {
    seed: '#ffffff',
    t3: '#e5484d',
    t2: '#e8a33d',
    t1: '#30a46c',
    service: '#8e6fd8',
    edge: 'rgba(120,140,165,0.22)',
    edgeHot: 'rgba(229,72,77,0.42)',
    edgeLit: '#3fb9d4',
    exit: '#8e6fd8',
    label: '#8794a6',
  };

  function colourFor(n) {
    if (n.is_seed) return PALETTE.seed;
    if (n.in_degree >= 60 && n.out_degree >= 60) return PALETTE.service;
    const s = n.score || 0;
    if (s >= 0.7) return PALETTE.t3;
    if (s >= 0.4) return PALETTE.t2;
    return PALETTE.t1;
  }

  /** Deterministic pseudo-random in [0,1) from an integer. Keeps layouts reproducible. */
  function hashRand(i) {
    let x = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b);
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
    return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
  }

  class Plot {
    constructor(canvas, opts = {}) {
      this.cv = canvas;
      this.ctx = canvas.getContext('2d');
      this.nodes = [];
      this.edges = [];
      this.byId = new Map();

      this.scale = 1;
      this.tx = 0;
      this.ty = 0;

      this.hot = null;
      this.held = null;
      this.panning = false;
      this.heat = 0;
      this.raf = null;

      this.onPick = opts.onPick || (() => {});
      this.onHot = opts.onHot || (() => {});

      this._bind();
      this._fitCanvas();
      window.addEventListener('resize', () => this._fitCanvas());
    }

    load(payload) {
      const w = this.cv.clientWidth || 900;
      const h = this.cv.clientHeight || 600;

      const rings = new Map();
      for (const n of payload.nodes) {
        if (!rings.has(n.hops)) rings.set(n.hops, []);
        rings.get(n.hops).push(n);
      }

      this.nodes = [];
      this.byId = new Map();

      for (const [hop, group] of [...rings].sort((a, b) => a[0] - b[0])) {
        // Sort each ring by value so the layout is stable regardless of arrival order.
        group.sort((a, b) => (b.received_btc || 0) - (a.received_btc || 0));
        const radius = hop === 0 ? 0 : 80 + hop * 78;
        group.forEach((n, i) => {
          const angle = (i / Math.max(1, group.length)) * Math.PI * 2
            + hashRand(n.id) * 0.28;
          const node = {
            ...n,
            x: w / 2 + Math.cos(angle) * radius,
            y: h / 2 + Math.sin(angle) * radius,
            vx: 0,
            vy: 0,
            r: this._radius(n),
          };
          this.nodes.push(node);
          this.byId.set(n.id, node);
        });
      }

      this.edges = payload.edges
        .map((e) => ({ ...e, a: this.byId.get(e.source), b: this.byId.get(e.target) }))
        .filter((e) => e.a && e.b);

      const top = Math.max(1e-9, ...this.edges.map((e) => e.value_btc || 0));
      for (const e of this.edges) {
        e.w = Math.max(0.5, Math.min(5, ((e.value_btc || 0) / top) * 5));
      }

      this.heat = 1;
      this._settledFit = false;
      this.fit();
      this._run();
    }

    clear() {
      this.nodes = [];
      this.edges = [];
      this.byId = new Map();
      this._stop();
      this.ctx.clearRect(0, 0, this.cv.width, this.cv.height);
    }

    _radius(n) {
      const v = (n.received_btc || 0);
      return Math.max(3.5, Math.min(19, 3.5 + Math.log1p(v * 12) * 2.4)) + (n.is_seed ? 3.5 : 0);
    }

    /* ------------------------------------------------------------- simulation */

    _tick() {
      const N = this.nodes;
      const n = N.length;
      if (!n) return;

      const w = this.cv.clientWidth;
      const h = this.cv.clientHeight;
      const cx = w / 2;
      const cy = h / 2;

      // These were tuned when repulsion was all-pairs. Moving to a spatial hash changed
      // the balance: each node now only feels nearby neighbours, so the outward pressure
      // near the centre is unchanged while the long-range crowding that used to hold the
      // cloud together is gone. With the old centring constant a 160-node graph expanded
      // until the fit-to-view scale hit its floor and the whole thing sat off-screen.
      //
      // Centring is now an order of magnitude stronger and repulsion softer, which keeps
      // the layout inside a readable box at any node count.
      const CELL = 110;
      const REPEL = 3400;
      const SPRING = 0.010;
      const REST = 82;
      const CENTRE = 0.0055;
      const DAMP = 0.86;
      // Hard bound as a backstop: no amount of accumulated velocity should be able to
      // throw a node somewhere fit-to-view cannot bring back.
      const LIMIT = Math.max(w, h) * 1.1;

      // Spatial hash: only compare nodes in the same or adjacent cells. This is the
      // change that keeps 300+ nodes smooth.
      const grid = new Map();
      for (const node of N) {
        const key = `${Math.floor(node.x / CELL)},${Math.floor(node.y / CELL)}`;
        let bucket = grid.get(key);
        if (!bucket) grid.set(key, (bucket = []));
        bucket.push(node);
      }

      for (const node of N) {
        if (node === this.held) continue;
        const gx = Math.floor(node.x / CELL);
        const gy = Math.floor(node.y / CELL);

        for (let ox = -1; ox <= 1; ox++) {
          for (let oy = -1; oy <= 1; oy++) {
            const bucket = grid.get(`${gx + ox},${gy + oy}`);
            if (!bucket) continue;
            for (const other of bucket) {
              if (other === node) continue;
              let dx = node.x - other.x;
              let dy = node.y - other.y;
              let d2 = dx * dx + dy * dy;
              if (d2 > CELL * CELL * 4) continue;
              // Coincident nodes give a zero distance and a NaN it never recovers from.
              if (d2 < 0.01) {
                dx = (hashRand(node.id) - 0.5) * 0.6 + 0.1;
                dy = (hashRand(other.id) - 0.5) * 0.6 + 0.1;
                d2 = dx * dx + dy * dy;
              }
              const d = Math.sqrt(d2);
              const f = REPEL / d2;
              node.vx += (dx / d) * f;
              node.vy += (dy / d) * f;
            }
          }
        }
      }

      for (const e of this.edges) {
        const dx = e.b.x - e.a.x;
        const dy = e.b.y - e.a.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - REST) * SPRING;
        const fx = (dx / d) * f;
        const fy = (dy / d) * f;
        if (e.a !== this.held) { e.a.vx += fx; e.a.vy += fy; }
        if (e.b !== this.held) { e.b.vx -= fx; e.b.vy -= fy; }
      }

      for (const node of N) {
        if (node === this.held) continue;
        node.vx += (cx - node.x) * CENTRE;
        node.vy += (cy - node.y) * CENTRE;
        node.vx *= DAMP;
        node.vy *= DAMP;
        node.x += node.vx * this.heat;
        node.y += node.vy * this.heat;

        node.x = Math.max(cx - LIMIT, Math.min(cx + LIMIT, node.x));
        node.y = Math.max(cy - LIMIT, Math.min(cy + LIMIT, node.y));
      }

      this.heat *= 0.982;
      if (this.heat < 0.006) {
        this.heat = 0;
        // Settled: frame the final arrangement rather than the seeded one.
        if (!this._settledFit) {
          this._settledFit = true;
          this.fit();
        }
      }
    }

    _run() {
      if (this.raf) return;
      const loop = () => {
        if (this.heat > 0 || this.held) this._tick();
        this._paint();
        this.raf = requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
    }

    _stop() {
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = null;
    }

    /* ---------------------------------------------------------------- painting */

    _fitCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const w = this.cv.clientWidth;
      const h = this.cv.clientHeight;
      if (!w || !h) return;
      this.cv.width = w * dpr;
      this.cv.height = h * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this._paint();
    }

    _linked(node) {
      if (!this.hot) return false;
      for (const e of this.edges) {
        if (e.a === this.hot && e.b === node) return true;
        if (e.b === this.hot && e.a === node) return true;
      }
      return false;
    }

    _paint() {
      const { ctx } = this;
      const w = this.cv.clientWidth;
      const h = this.cv.clientHeight;
      ctx.clearRect(0, 0, w, h);
      if (!this.nodes.length) return;

      ctx.save();
      ctx.translate(this.tx, this.ty);
      ctx.scale(this.scale, this.scale);

      for (const e of this.edges) {
        const lit = this.hot && (e.a === this.hot || e.b === this.hot);
        const hot = (e.a.score || 0) >= 0.7 || (e.b.score || 0) >= 0.7;
        const dim = this.hot && !lit;

        ctx.globalAlpha = dim ? 0.16 : 1;
        ctx.strokeStyle = lit ? PALETTE.edgeLit : hot ? PALETTE.edgeHot : PALETTE.edge;
        ctx.lineWidth = (lit ? e.w + 1 : e.w) / Math.pow(this.scale, 0.3);

        // Slight arc so a pair pointing both ways stays two distinguishable strokes.
        const mx = (e.a.x + e.b.x) / 2;
        const my = (e.a.y + e.b.y) / 2;
        const dx = e.b.x - e.a.x;
        const dy = e.b.y - e.a.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const bow = Math.min(22, len * 0.14);
        const qx = mx - (dy / len) * bow;
        const qy = my + (dx / len) * bow;

        ctx.beginPath();
        ctx.moveTo(e.a.x, e.a.y);
        ctx.quadraticCurveTo(qx, qy, e.b.x, e.b.y);
        ctx.stroke();

        // Arrowhead at the curve's midpoint, where nothing else is drawn over it.
        const t = 0.5;
        const px = (1 - t) * (1 - t) * e.a.x + 2 * (1 - t) * t * qx + t * t * e.b.x;
        const py = (1 - t) * (1 - t) * e.a.y + 2 * (1 - t) * t * qy + t * t * e.b.y;
        const tanx = 2 * (1 - t) * (qx - e.a.x) + 2 * t * (e.b.x - qx);
        const tany = 2 * (1 - t) * (qy - e.a.y) + 2 * t * (e.b.y - qy);
        const ang = Math.atan2(tany, tanx);
        const s = 5.5;
        ctx.fillStyle = lit ? PALETTE.edgeLit : hot ? PALETTE.edgeHot : PALETTE.edge;
        ctx.beginPath();
        ctx.moveTo(px + Math.cos(ang) * s, py + Math.sin(ang) * s);
        ctx.lineTo(px + Math.cos(ang + 2.5) * s, py + Math.sin(ang + 2.5) * s);
        ctx.lineTo(px + Math.cos(ang - 2.5) * s, py + Math.sin(ang - 2.5) * s);
        ctx.closePath();
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      for (const node of this.nodes) {
        const dim = this.hot && this.hot !== node && !this._linked(node);
        ctx.globalAlpha = dim ? 0.25 : 1;

        if (node.is_exit) {
          // Exit points are drawn as a square with a halo rather than another coloured
          // circle. Colour alone was not enough: on a field of a hundred dots the one
          // that matters has to differ in shape, or the eye never finds it. This is the
          // node an investigator is looking for, so it gets the loudest treatment on the
          // canvas.
          const r = node.r + 1.5;
          ctx.fillStyle = node.exit_kind === 'dormant' ? PALETTE.t2 : PALETTE.exit;
          ctx.fillRect(node.x - r, node.y - r, r * 2, r * 2);
          ctx.strokeStyle = node.exit_kind === 'dormant' ? PALETTE.t2 : PALETTE.exit;
          ctx.lineWidth = 1.2;
          ctx.globalAlpha = (dim ? 0.25 : 1) * 0.45;
          ctx.beginPath();
          ctx.arc(node.x, node.y, r + 5, 0, Math.PI * 2);
          ctx.stroke();
          ctx.globalAlpha = dim ? 0.25 : 1;
        } else {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
          ctx.fillStyle = colourFor(node);
          ctx.fill();
        }

        if (node.is_seed) {
          ctx.strokeStyle = PALETTE.edgeLit;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r + 2, 0, Math.PI * 2);
          ctx.stroke();
        } else if (node === this.hot) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r + 2, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      // Exit points keep their label at any zoom and any node count.
      ctx.font = '10px ui-monospace, Consolas, monospace';
      ctx.textAlign = 'center';
      for (const node of this.nodes) {
        if (!node.is_exit) continue;
        const a = node.address || '';
        ctx.fillStyle = node.exit_kind === 'dormant' ? PALETTE.t2 : PALETTE.exit;
        ctx.fillText(
          node.exit_kind === 'dormant' ? 'DORMANT' : 'EXIT',
          node.x, node.y - node.r - 9
        );
        ctx.fillStyle = PALETTE.label;
        ctx.fillText(`${a.slice(0, 6)}…${a.slice(-4)}`, node.x, node.y + node.r + 14);
      }

      if (this.scale > 0.7 && this.nodes.length <= 90) {
        ctx.font = '10px ui-monospace, Consolas, monospace';
        ctx.fillStyle = PALETTE.label;
        ctx.textAlign = 'center';
        for (const node of this.nodes) {
          if (node.is_exit) continue;
          if (!node.is_seed && node.r < 6 && node !== this.hot) continue;
          const a = node.address || '';
          ctx.fillText(`${a.slice(0, 6)}…${a.slice(-4)}`, node.x, node.y + node.r + 12);
        }
      }

      ctx.restore();
    }

    /* ---------------------------------------------------------------- viewport */

    fit() {
      if (!this.nodes.length) return;

      // Frame the bulk of the graph, not its extremes.
      //
      // Using the true min and max lets one stray node, flung out by the simulation or
      // simply attached to nothing, dictate the zoom for everything else. The result is
      // a screen of empty space with the actual graph as an unreadable clump in the
      // middle. Clipping to the 2nd and 98th percentile frames what there is to look at
      // and lets the rare outlier sit off the edge, where panning will still find it.
      const pct = (arr, q) => {
        const sorted = [...arr].sort((a, b) => a - b);
        const i = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
        return sorted[i];
      };
      const xs = this.nodes.map((n) => n.x);
      const ys = this.nodes.map((n) => n.y);
      const wide = this.nodes.length > 12;
      const minX = wide ? pct(xs, 0.02) : Math.min(...xs);
      const maxX = wide ? pct(xs, 0.98) : Math.max(...xs);
      const minY = wide ? pct(ys, 0.02) : Math.min(...ys);
      const maxY = wide ? pct(ys, 0.98) : Math.max(...ys);

      const w = this.cv.clientWidth;
      const h = this.cv.clientHeight;
      const pad = 80;
      const s = Math.min(
        (w - pad * 2) / Math.max(1, maxX - minX),
        (h - pad * 2) / Math.max(1, maxY - minY)
      );
      this.scale = Math.max(0.25, Math.min(2.4, s));
      this.tx = w / 2 - ((minX + maxX) / 2) * this.scale;
      this.ty = h / 2 - ((minY + maxY) / 2) * this.scale;
      this._paint();
    }

    _toWorld(cx, cy) {
      const r = this.cv.getBoundingClientRect();
      return {
        x: (cx - r.left - this.tx) / this.scale,
        y: (cy - r.top - this.ty) / this.scale,
      };
    }

    _at(cx, cy) {
      const p = this._toWorld(cx, cy);
      for (let i = this.nodes.length - 1; i >= 0; i--) {
        const n = this.nodes[i];
        const dx = n.x - p.x;
        const dy = n.y - p.y;
        if (dx * dx + dy * dy <= (n.r + 5) ** 2) return n;
      }
      return null;
    }

    _bind() {
      const cv = this.cv;
      let origin = null;
      let moved = false;

      cv.addEventListener('mousedown', (ev) => {
        moved = false;
        const n = this._at(ev.clientX, ev.clientY);
        if (n) {
          this.held = n;
          this.heat = Math.max(this.heat, 0.3);
        } else {
          this.panning = true;
          origin = { x: ev.clientX - this.tx, y: ev.clientY - this.ty };
          cv.classList.add('dragging');
        }
      });

      window.addEventListener('mousemove', (ev) => {
        if (this.held) {
          moved = true;
          const p = this._toWorld(ev.clientX, ev.clientY);
          this.held.x = p.x; this.held.y = p.y;
          this.held.vx = 0; this.held.vy = 0;
          return;
        }
        if (this.panning && origin) {
          moved = true;
          this.tx = ev.clientX - origin.x;
          this.ty = ev.clientY - origin.y;
          this._paint();
          return;
        }
        const n = this._at(ev.clientX, ev.clientY);
        if (n !== this.hot) {
          this.hot = n;
          this._paint();
        }
        this.onHot(n, ev);
      });

      window.addEventListener('mouseup', (ev) => {
        if (!moved) {
          const n = this.held || this._at(ev.clientX, ev.clientY);
          if (n) this.onPick(n);
        }
        this.held = null;
        this.panning = false;
        origin = null;
        cv.classList.remove('dragging');
      });

      cv.addEventListener('wheel', (ev) => {
        ev.preventDefault();
        const r = cv.getBoundingClientRect();
        const mx = ev.clientX - r.left;
        const my = ev.clientY - r.top;
        const k = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
        const next = Math.max(0.08, Math.min(5, this.scale * k));
        // Zoom toward the pointer, so the thing under the cursor stays put.
        this.tx = mx - ((mx - this.tx) / this.scale) * next;
        this.ty = my - ((my - this.ty) / this.scale) * next;
        this.scale = next;
        this._paint();
      }, { passive: false });

      cv.addEventListener('mouseleave', () => {
        if (this.hot) { this.hot = null; this._paint(); this.onHot(null); }
      });
    }
  }

  global.Plot = Plot;
})(window);
