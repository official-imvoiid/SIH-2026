'use strict';
/**
 * Union-Find (Disjoint Set Union) with path compression and union by rank.
 *
 * This is the structure that turns "these two addresses are provably the same owner"
 * into "here is the full set of addresses that owner controls". Every time we see
 * evidence of shared control we call union(); at the end, find() tells us which
 * entity any address belongs to.
 *
 * Complexity: both operations are O(alpha(n)) amortised, where alpha is the inverse
 * Ackermann function. For any n that fits in a computer, alpha(n) < 5, so this is
 * effectively constant time per operation. m operations over n addresses cost
 * O(m * alpha(n)). Space is O(n).
 *
 * That near-linear cost is the whole reason this approach survives at blockchain
 * scale -- there are hundreds of millions of addresses, and anything worse than
 * near-linear simply does not finish.
 */

class UnionFind {
  constructor() {
    /** @type {Map<string, string>} address -> parent address */
    this.parent = new Map();
    /** @type {Map<string, number>} root -> rank (upper bound on tree height) */
    this.rank = new Map();
    /** @type {Map<string, number>} root -> number of members in that set */
    this.size = new Map();
    this._count = 0;
  }

  /** Register an address as its own singleton set. Idempotent. */
  add(x) {
    if (!this.parent.has(x)) {
      this.parent.set(x, x);
      this.rank.set(x, 0);
      this.size.set(x, 1);
      this._count += 1;
    }
    return x;
  }

  /**
   * Find the representative ("root") of x's set, compressing the path on the way
   * back up so later lookups are faster.
   *
   * Written iteratively on purpose. A recursive find blows the stack on long chains,
   * and long chains are exactly what a peel chain produces.
   */
  find(x) {
    if (!this.parent.has(x)) this.add(x);

    // Walk to the root.
    let root = x;
    while (this.parent.get(root) !== root) {
      root = this.parent.get(root);
    }

    // Path compression: point every node on the path straight at the root.
    let cur = x;
    while (this.parent.get(cur) !== root) {
      const next = this.parent.get(cur);
      this.parent.set(cur, root);
      cur = next;
    }

    return root;
  }

  /**
   * Merge the sets containing a and b. Returns true if a merge actually happened
   * (they were in different sets), false if they were already together.
   */
  union(a, b) {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return false;

    // Union by rank: hang the shorter tree under the taller one so depth stays low.
    if (this.rank.get(ra) < this.rank.get(rb)) {
      const t = ra;
      ra = rb;
      rb = t;
    }

    this.parent.set(rb, ra);
    this.size.set(ra, this.size.get(ra) + this.size.get(rb));
    this.size.delete(rb);

    if (this.rank.get(ra) === this.rank.get(rb)) {
      this.rank.set(ra, this.rank.get(ra) + 1);
    }

    this._count -= 1;
    return true;
  }

  /** True if a and b are known to be the same owner. */
  connected(a, b) {
    return this.find(a) === this.find(b);
  }

  /** How many members are in the set containing x. */
  setSize(x) {
    return this.size.get(this.find(x)) || 1;
  }

  /** Number of distinct sets (entities) currently held. */
  get count() {
    return this._count;
  }

  /** Number of addresses registered. */
  get elements() {
    return this.parent.size;
  }

  /**
   * Materialise the result: root -> sorted array of member addresses.
   * Call this once at the end, not inside a loop; it is O(n * alpha(n)).
   * @returns {Map<string, string[]>}
   */
  groups() {
    /** @type {Map<string, string[]>} */
    const out = new Map();
    for (const addr of this.parent.keys()) {
      const root = this.find(addr);
      if (!out.has(root)) out.set(root, []);
      out.get(root).push(addr);
    }
    for (const members of out.values()) members.sort();
    return out;
  }
}

module.exports = { UnionFind };
