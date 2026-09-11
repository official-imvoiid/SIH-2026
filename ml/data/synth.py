"""Synthetic Bitcoin world generator.

Why this exists
---------------
The real Elliptic dataset needs a Kaggle account and a ~400 MB download. That is a bad
dependency for six people starting work on day one, and a worse one on demo day. This
module generates a Bitcoin-shaped world with the same *structure* as the real thing:

  * addresses owned by entities that reuse them (so co-spend clustering has real work)
  * CoinJoin transactions that must NOT be merged (so the clustering guard is testable)
  * planted laundering typologies (peel chains, mule pass-throughs, ransom collection)
  * the same class balance as Elliptic (~2% illicit, ~21% licit, ~77% unlabelled)
  * a behavioural regime change at time step 43, mirroring the dark-market shutdown that
    makes every published model's F1 collapse

That last point matters: the concept-drift demo works on synthetic data too, so the pitch
can be rehearsed before the real dataset ever lands.

This is NOT a substitute for the real data in your results. Numbers from here go on a
slide only if the slide says "synthetic". Swap in the real loader
(``ml.ingest.elliptic``) for anything you report.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from ml.config import DRIFT_TS, N_TIMESTEPS, RANDOM_SEED

Archetype = Literal[
    "exchange",
    "merchant",
    "individual",
    "ransom_collector",
    "mixer",
    "mule",
    "peel_hop",
    "darkmarket",
]

# Which archetypes carry which ground-truth label. Mirrors how Elliptic labels work:
# most of the world is unlabelled, a fifth is confirmed clean, a sliver is confirmed dirty.
ILLICIT_ARCHETYPES = {"ransom_collector", "mule", "peel_hop", "darkmarket"}
LICIT_ARCHETYPES = {"exchange", "merchant"}


@dataclass
class Entity:
    """One real-world actor, owning many addresses. This is the ground truth that the
    co-spend clustering step has to rediscover from transactions alone."""

    eid: str
    archetype: Archetype
    addresses: list[str] = field(default_factory=list)
    label: str = "unknown"

    def spend_inputs(self, rng: random.Random, k_max: int = 3) -> list[str]:
        """Pick a few owned addresses to spend from -- this is what creates the co-spend
        signal that clustering later exploits."""
        k = min(len(self.addresses), rng.randint(1, k_max))
        return rng.sample(self.addresses, k)

    def fresh_address(self, rng: random.Random, world: "World") -> str:
        addr = world.new_address()
        self.addresses.append(addr)
        world.owner[addr] = self.eid
        return addr


@dataclass
class Transaction:
    txid: str
    ts: int
    inputs: list[str]
    outputs: list[tuple[str, float]]
    is_coinjoin: bool = False

    @property
    def total_out(self) -> float:
        return sum(v for _a, v in self.outputs)

    def to_dict(self) -> dict:
        return {
            "txid": self.txid,
            "ts": self.ts,
            "inputs": list(self.inputs),
            "outputs": [[a, round(v, 8)] for a, v in self.outputs],
            "is_coinjoin": self.is_coinjoin,
        }


class World:
    """Holds the generated universe and the counters that keep IDs unique."""

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.rng = random.Random(seed)
        self.entities: dict[str, Entity] = {}
        self.owner: dict[str, str] = {}  # address -> entity id (ground truth)
        self.transactions: list[Transaction] = []
        self._addr_n = 0
        self._tx_n = 0
        self._ent_n = 0

    # -- id factories ------------------------------------------------------------------

    def new_address(self) -> str:
        self._addr_n += 1
        return f"bc1q{self._addr_n:08d}"

    def new_txid(self) -> str:
        self._tx_n += 1
        return f"tx{self._tx_n:07d}"

    def new_entity(self, archetype: Archetype, n_addresses: int) -> Entity:
        self._ent_n += 1
        ent = Entity(eid=f"TRUE-{self._ent_n:05d}", archetype=archetype)
        self.entities[ent.eid] = ent
        for _ in range(n_addresses):
            ent.fresh_address(self.rng, self)
        return ent

    # -- transaction helpers -----------------------------------------------------------

    def spend(
        self,
        sender: Entity,
        outputs: list[tuple[str, float]],
        ts: int,
        is_coinjoin: bool = False,
        explicit_inputs: list[str] | None = None,
    ) -> Transaction:
        inputs = (
            explicit_inputs
            if explicit_inputs is not None
            else sender.spend_inputs(self.rng)
        )
        tx = Transaction(
            txid=self.new_txid(),
            ts=ts,
            inputs=inputs,
            outputs=outputs,
            is_coinjoin=is_coinjoin,
        )
        self.transactions.append(tx)
        return tx

    def pay(self, sender: Entity, receiver: Entity, amount: float, ts: int) -> Transaction:
        """A normal payment: value to the receiver, change back to a fresh own address.

        The change output is what the change-address heuristic keys on, and what makes
        the address count per entity grow realistically over time.
        """
        target = self.rng.choice(receiver.addresses)
        change_addr = sender.fresh_address(self.rng, self)
        change = round(amount * self.rng.uniform(0.05, 0.9), 8)
        return self.spend(sender, [(target, amount), (change_addr, change)], ts)


# ---------------------------------------------------------------------------------------
# Behaviour generators -- one per archetype
# ---------------------------------------------------------------------------------------


def _gen_exchange_traffic(world: World, exchanges: list[Entity], users: list[Entity]) -> None:
    """High-volume, high-degree, boring. The dominant licit signal in any real chain."""
    rng = world.rng
    for ts in range(1, N_TIMESTEPS + 1):
        for ex in exchanges:
            for _ in range(rng.randint(6, 14)):
                user = rng.choice(users)
                if rng.random() < 0.5:
                    world.pay(user, ex, round(rng.uniform(0.01, 2.5), 8), ts)
                else:
                    world.pay(ex, user, round(rng.uniform(0.01, 2.0), 8), ts)


def _gen_merchant_traffic(world: World, merchants: list[Entity], users: list[Entity]) -> None:
    rng = world.rng
    for ts in range(1, N_TIMESTEPS + 1):
        for m in merchants:
            for _ in range(rng.randint(1, 5)):
                world.pay(rng.choice(users), m, round(rng.uniform(0.001, 0.4), 8), ts)


def _gen_ransom_campaign(
    world: World, collector: Entity, victims: list[Entity], ts: int, n_victims: int
) -> None:
    """Fan-in: many unrelated victims pay one collection wallet in a tight window.

    This is the shape that FATF calls consolidation and that our VA-CONSOLIDATION rule
    is written to catch.
    """
    rng = world.rng
    for victim in rng.sample(victims, min(n_victims, len(victims))):
        world.pay(victim, collector, round(rng.uniform(0.05, 0.9), 8), ts)


def _gen_peel_chain(
    world: World, origin: Entity, cash_outs: list[Entity], ts: int, hops: int
) -> list[Entity]:
    """Repeatedly shave a small amount off and forward the rest to a fresh entity.

    Each hop has exactly two outputs -- one small "peel" to a cash-out point, one large
    remainder to the next hop. That two-output shape is precisely what
    ``detect_peel_chain`` walks.
    """
    rng = world.rng
    chain: list[Entity] = []
    remaining = round(rng.uniform(20.0, 90.0), 8)
    current = origin

    for hop in range(hops):
        nxt = world.new_entity("peel_hop", n_addresses=1)
        chain.append(nxt)
        peel = round(remaining * rng.uniform(0.04, 0.18), 8)
        remaining = round(remaining - peel, 8)
        if remaining <= 0.001:
            break
        cash_out = rng.choice(cash_outs)
        world.spend(
            current,
            [
                (rng.choice(cash_out.addresses), peel),
                (rng.choice(nxt.addresses), remaining),
            ],
            ts=min(ts + hop, N_TIMESTEPS),
        )
        current = nxt
    return chain


def _gen_mule_passthrough(
    world: World, mule: Entity, src: Entity, dst: Entity, ts: int
) -> None:
    """Money in, ~all of it straight back out, nothing retained. VA-PASSTHROUGH."""
    rng = world.rng
    amount = round(rng.uniform(1.0, 12.0), 8)
    world.pay(src, mule, amount, ts)
    world.spend(
        mule,
        [(rng.choice(dst.addresses), round(amount * rng.uniform(0.94, 0.995), 8))],
        ts=ts,
    )


def _gen_coinjoin(world: World, participants: list[Entity], ts: int) -> None:
    """A CoinJoin: many UNRELATED entities co-sign one transaction with equal outputs.

    This is the trap for naive co-spend clustering. Merging these inputs would glue
    unrelated actors into one entity permanently and corrupt every downstream result, so
    ``is_likely_coinjoin`` must catch it. We generate real ones so the guard is testable.
    """
    rng = world.rng
    denom = round(rng.choice([0.1, 0.25, 0.5, 1.0]), 8)
    inputs: list[str] = []
    outputs: list[tuple[str, float]] = []
    for p in participants:
        inputs.extend(p.spend_inputs(rng, k_max=1))
        outputs.append((p.fresh_address(rng, world), denom))
    # A little non-equal change, as real CoinJoins have.
    for p in rng.sample(participants, max(1, len(participants) // 3)):
        outputs.append((p.fresh_address(rng, world), round(rng.uniform(0.01, 0.09), 8)))

    tx = Transaction(
        txid=world.new_txid(), ts=ts, inputs=inputs, outputs=outputs, is_coinjoin=True
    )
    world.transactions.append(tx)


def _gen_structuring(world: World, sender: Entity, sinks: list[Entity], ts: int) -> None:
    """Many near-identical outbound amounts -- smurfing. VA-STRUCTURING."""
    rng = world.rng
    base = round(rng.uniform(0.4, 0.9), 8)
    for _ in range(rng.randint(9, 15)):
        amt = round(base * rng.uniform(0.98, 1.02), 8)
        world.pay(sender, rng.choice(sinks), amt, ts)


# ---------------------------------------------------------------------------------------
# World assembly
# ---------------------------------------------------------------------------------------


def generate(
    seed: int = RANDOM_SEED,
    n_exchanges: int = 5,
    n_merchants: int = 140,
    n_individuals: int = 3400,
    n_campaigns: int = 48,
) -> World:
    """Build the whole synthetic world.

    The regime change at ``DRIFT_TS`` is deliberate: before it, illicit actors run large,
    obvious fan-in campaigns; after it, they switch to small structured transfers through
    mixers. A model trained only on the "before" pattern will lose recall on the "after"
    pattern, which is exactly the failure the demo is built around.
    """
    w = World(seed)
    rng = w.rng

    exchanges = [w.new_entity("exchange", rng.randint(40, 90)) for _ in range(n_exchanges)]
    merchants = [w.new_entity("merchant", rng.randint(4, 14)) for _ in range(n_merchants)]
    individuals = [w.new_entity("individual", rng.randint(1, 5)) for _ in range(n_individuals)]

    _gen_exchange_traffic(w, exchanges, individuals)
    _gen_merchant_traffic(w, merchants, individuals)

    darkmarket = w.new_entity("darkmarket", rng.randint(10, 20))

    for i in range(n_campaigns):
        ts = rng.randint(2, N_TIMESTEPS - 6)
        pre_drift = ts < DRIFT_TS

        collector = w.new_entity("ransom_collector", rng.randint(1, 3))

        if pre_drift:
            # Old regime: loud. Big fan-in, then a long peel chain out.
            _gen_ransom_campaign(w, collector, individuals, ts, n_victims=rng.randint(12, 26))
            _gen_peel_chain(w, collector, exchanges + merchants, ts + 1, hops=rng.randint(5, 9))
            if rng.random() < 0.5:
                world_mule = w.new_entity("mule", 1)
                _gen_mule_passthrough(w, world_mule, collector, darkmarket, ts + 1)
        else:
            # New regime after the shutdown: quiet. Small structured transfers, mules,
            # and CoinJoin usage instead of one loud collection wallet.
            _gen_ransom_campaign(w, collector, individuals, ts, n_victims=rng.randint(4, 9))
            _gen_structuring(w, collector, merchants + exchanges, ts + 1)
            for _ in range(rng.randint(2, 4)):
                mule = w.new_entity("mule", 1)
                _gen_mule_passthrough(w, mule, collector, rng.choice(exchanges), ts + 1)
            # Partial continuity: roughly half the post-drift crews still run a (shorter)
            # peel chain. Real drift degrades a model, it does not blind it completely --
            # and a demo where recall drops to zero looks staged rather than instructive.
            if rng.random() < 0.5:
                _gen_peel_chain(w, collector, exchanges + merchants, ts + 1,
                                hops=rng.randint(3, 5))

    # CoinJoins throughout, drawn from the whole population -- the clustering trap.
    for _ in range(24):
        ts = rng.randint(1, N_TIMESTEPS)
        pool = rng.sample(individuals + merchants, rng.randint(6, 12))
        _gen_coinjoin(w, pool, ts)

    # Ground-truth labels, at Elliptic-like proportions: confirmed-dirty is a sliver,
    # confirmed-clean is a fifth, and most of the world stays unknown.
    for ent in w.entities.values():
        if ent.archetype in ILLICIT_ARCHETYPES:
            ent.label = "illicit" if rng.random() < 0.75 else "unknown"
        elif ent.archetype in LICIT_ARCHETYPES:
            ent.label = "licit" if rng.random() < 0.82 else "unknown"
        else:
            # Ordinary users are mostly unlabelled, as in the real dataset -- only a
            # minority ever get confirmed clean by an exchange or an investigation.
            ent.label = "licit" if rng.random() < 0.21 else "unknown"

    w.transactions.sort(key=lambda t: (t.ts, t.txid))
    return w


def iter_transactions(w: World) -> Iterator[tuple[str, list[str], list[float]]]:
    """Yield ``(txid, input_addresses, output_values)`` -- the exact shape the clustering
    step consumes. Keeping this narrow means clustering never sees labels."""
    for tx in w.transactions:
        yield tx.txid, list(tx.inputs), [v for _a, v in tx.outputs]


def write(w: World, out_dir: Path) -> dict[str, Path]:
    """Persist the world as JSONL so it can be inspected by eye and diffed in git."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "transactions": out_dir / "synth_transactions.jsonl",
        "truth": out_dir / "synth_truth.json",
    }

    with paths["transactions"].open("w", encoding="utf-8") as fh:
        for tx in w.transactions:
            fh.write(json.dumps(tx.to_dict()) + "\n")

    truth = {
        "entities": {
            e.eid: {
                "archetype": e.archetype,
                "label": e.label,
                "n_addresses": len(e.addresses),
                "addresses": e.addresses,
            }
            for e in w.entities.values()
        },
        "address_owner": w.owner,
        "meta": {
            "seed": RANDOM_SEED,
            "n_timesteps": N_TIMESTEPS,
            "drift_ts": DRIFT_TS,
            "n_transactions": len(w.transactions),
            "n_addresses": len(w.owner),
            "n_entities": len(w.entities),
        },
    }
    paths["truth"].write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return paths


def summary(w: World) -> dict:
    from collections import Counter

    arch = Counter(e.archetype for e in w.entities.values())
    lab = Counter(e.label for e in w.entities.values())
    total = len(w.entities)
    return {
        "entities": total,
        "addresses": len(w.owner),
        "transactions": len(w.transactions),
        "coinjoins": sum(1 for t in w.transactions if t.is_coinjoin),
        "archetypes": dict(arch.most_common()),
        "labels": dict(lab),
        "illicit_pct": round(100 * lab["illicit"] / max(total, 1), 2),
        "licit_pct": round(100 * lab["licit"] / max(total, 1), 2),
        "unknown_pct": round(100 * lab["unknown"] / max(total, 1), 2),
    }


if __name__ == "__main__":
    from ml.config import DATA_PROCESSED

    world = generate()
    paths = write(world, DATA_PROCESSED)
    s = summary(world)

    print("Synthetic Bitcoin world generated")
    print("-" * 62)
    for k in ("entities", "addresses", "transactions", "coinjoins"):
        print(f"  {k:15} {s[k]:>10,}")
    print(f"  {'illicit':15} {s['illicit_pct']:>9.2f}%   (Elliptic real: ~2%)")
    print(f"  {'licit':15} {s['licit_pct']:>9.2f}%   (Elliptic real: ~21%)")
    print(f"  {'unknown':15} {s['unknown_pct']:>9.2f}%   (Elliptic real: ~77%)")
    print("-" * 62)
    print("  archetypes:", s["archetypes"])
    for name, p in paths.items():
        print(f"  wrote {name}: {p}")
