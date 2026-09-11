# What this project is — explained from zero

No prior knowledge assumed. If you can read this page you can defend the project.

---

## 1. The crime

A hospital in Pune gets hacked on a Tuesday. Every computer locks. A message appears:
*pay ₹80 lakh in Bitcoin within 72 hours or the patient records are deleted.*

The hospital pays. The money is gone. Police are called.

This happens constantly. Indians lost roughly **₹22,800 crore to cybercrime in 2024
alone**, and a growing share of it leaves the country as cryptocurrency.

## 2. The strange thing about Bitcoin

Most people assume Bitcoin is secret. It is the opposite.

**Every Bitcoin payment ever made is public.** Anyone can look up any payment, see how much
moved, when, and between which wallets. There is no permission needed. It is a giant open
ledger that the whole world can read.

So the police *can* watch the ransom money move. In real time. That sounds like it should
be the end of the story.

## 3. Why it isn't

A Bitcoin wallet has no name attached. It looks like this:

```
bc1qjndxpuddlssczk336tdd7wcawlj2qws7p77p2k
```

That is a real wallet belonging to the **Conti** ransomware gang. But nothing about that
string says "Conti". It is just a number.

And here is the real problem: **one gang does not use one wallet. They use hundreds.**

The moment they receive a ransom, they split it and shuffle it between wallets they
control — over and over — specifically to make it impossible to follow. An investigator
opens the blockchain and sees ten thousand anonymous strings passing money to each other.
Which ones are the gang? Which are innocent people? There is no way to tell by looking.

**That is the problem this project solves.**

## 4. The clever bit (this is the heart of the project)

There is a trick that reveals which wallets secretly belong to the same person. It is the
foundation of the entire blockchain-forensics industry, and it comes from a 2013 research
paper called *A Fistful of Bitcoins*.

It works like this:

> To spend money from a Bitcoin wallet, you need that wallet's password.
>
> Some payments spend money from **two wallets at the same time**.
>
> To do that, you must have had **both passwords**.
>
> So one person owns both wallets.

That's it. That's the whole idea.

Now do it across millions of payments. Wallet A pairs with B. B pairs with C. So A, B and C
are all the same owner. Chains form. The anonymous crowd separates itself into groups, and
each group is one real person or one gang.

We call each group an **entity**. This is the step that turns *"ten thousand anonymous
strings"* into *"about four hundred actors, and here is which wallets each one controls."*

**In this repo:** `ml/features/clustering.py`

### The trap that ruins it

There is a thing called a **CoinJoin** — a service where many strangers deliberately make
one payment together to confuse exactly this kind of analysis.

If you apply the trick naively to a CoinJoin, you conclude all those strangers are the same
person. And because the logic chains (A=B, B=C, so A=C), one mistake spreads and can weld
half the network into one meaningless blob. You would be accusing innocent people.

So we detect CoinJoins and refuse to use them. There is a test that specifically checks
this, and it is the most important test in the project.

## 5. Spotting the criminals

Once you have entities instead of loose wallets, you can look at how each one *behaves*.
Criminals move money in shapes that ordinary people don't:

**Peel chain** — Imagine breaking a ₹2000 note, spending ₹200, then breaking the change,
spending ₹200 again, over and over. Money keeps moving while small amounts get shaved off
and cashed out. Almost nobody does this by accident.

**Pass-through** — Money arrives, and within minutes almost all of it leaves again.
Nothing is kept. A real person's wallet holds a balance. A wallet that never holds anything
isn't a wallet — it's a relay station, and its only purpose is to add another confusing hop.

**Fan-in** — Thirty different wallets pay one wallet within the same hour. That's not a
shop, that's a collection point. Very often it's thirty ransom victims paying one gang.

**Structuring** — Dozens of transfers all of near-identical size. Real payments vary. Equal
amounts mean somebody is deliberately splitting a lump sum.

These are not things we invented. They are published **FATF red-flag indicators** — FATF is
the international body that writes anti-money-laundering rules that India follows. So when
our tool flags something, it can say *"this matches official indicator X"* rather than
*"the computer thinks so"*.

**In this repo:** `ml/typology/rules.py`

The important property: **these rules use no AI at all.** They are just careful arithmetic
on the shape of the money flow. That means they cannot break, cannot need training data,
and cannot be argued away as a black box.

## 6. What the AI actually does

Separately from the rules, we train a model (a **random forest** — think of it as several
hundred yes/no flowcharts voting) on wallets already known to be criminal or clean. It
learns to score new wallets from 0 to 1.

Two things worth knowing:

**We do not use a fancy neural network, on purpose.** The main research paper on this
problem (Weber et al., 2019) found that a plain random forest **beat** a graph neural
network — about 0.79 versus 0.65 on their scoring measure. Fancier lost. So we use the
thing that works, and we can cite why.

**We never report "accuracy".** Only about 2% of wallets are criminal. A program that says
"everything is clean, always" would be 98% accurate and completely useless. We report
measures that actually reflect catching criminals.

## 7. Why the explanation matters more than the score

This is the part that makes it a police tool instead of a science project.

A model that outputs `0.93` is useless as evidence. No investigator can act on it, no court
will accept it, and no one can check it.

So the tool writes out its reasoning in plain English:

> Entity E-06641 is flagged, risk 1.00. **Primary indicator — Rapid pass-through:**
> forwarded 100% of the 62.23 BTC it received, within one time step of receiving it,
> retaining effectively no balance. Wallets that hold no position are characteristic
> laundering intermediaries rather than end users. **Additional:** peel chain.
> *Risk scores indicate investigative priority, not proof of criminal conduct.*

Three rules are built into this and enforced by automated tests:

1. **Behaviour beats association.** What a wallet *did* always leads. "Its neighbour is
   suspicious" is demoted to supporting context and can never be the headline — otherwise
   you are accusing people for who they transact with.
2. **The caveat is never dropped.** No score is ever shown without it.
3. **Evidence for the defence survives.** Facts that argue *against* the flag are shown,
   not filtered out.

## 8. What actually got built

```bash
python run.py
```

One command. It builds the data, trains the model, and opens a working investigator
interface at `localhost:8000`. Takes under a minute, needs no internet.

You get: a queue of suspicious entities, a map of the money as a network you can click
through, and a case file for each one explaining the flag.

## 9. What is real and what is not — say this out loud

**Real:**
- The clustering technique is what Chainalysis and Elliptic genuinely use
- The laundering patterns are real criminal behaviour, mapped to real FATF indicators
- The method (how we split training data, which measures we report) is correct
- **The real-data experiment** (see below) uses genuine ransomware wallets and genuine
  blockchain transactions

**Not real:**
- The bundled demo dataset is **synthetic** — a Bitcoin-shaped world generated in code so
  the team can work without a 400 MB download. Any number from it must be labelled
  synthetic.
- We cannot put a *name* to a gang. Clustering says "these 400 wallets share an owner". Who
  that owner is requires exchange KYC records, which only law enforcement can obtain. This
  is a legal boundary, not a coding gap — and saying so clearly is a strength, not a
  weakness.

## 10. The experiment that proves it isn't a toy

The obvious attack on this project: *"you made up the data, then found the patterns you
made up. That proves nothing."* Correct — and that is why we ran this:

**`python -m ml.experiments.real_vs_control`**

Two groups of **real** Bitcoin wallets, treated identically:

- **Group A:** seeded from real ransomware wallets — Conti, Ryuk, NetWalker, SamSam, Locky
  — taken from the public Ransomwhere dataset of actual ransom payments.
- **Group B:** seeded from ordinary wallets in recent Bitcoin blocks. Normal people,
  shops, exchanges.

Same crawl, same clustering, same rules, same everything. The only difference is where the
starting wallets came from.

Then: **do the rules fire more on the criminal group?**

### The first answer was no — and it was wrong

1,000 real addresses, ~15,000 real Bitcoin transactions. The first pass found **no
separation**: indicators fired on roughly two thirds of criminal and ordinary money alike.

Then we found the mistake in our own experiment, and it is the most instructive thing here.

We had counted *every wallet near a ransom wallet* as criminal. But one hop from a ransom
collection wallet you find **the victims who paid, and the exchange the criminals cashed
out at** — overwhelmingly innocent people. We had labelled a crowd of victims as criminals,
then complained the detector could not tell them from ordinary users.

So we relabelled: criminals are now *only* wallets independently confirmed to have received
ransom payments. Then we tuned on some ransomware families and tested on **families the
tuning never saw**, so we could not fool ourselves by memorising one crew's habits.

### One rule of seven works

| Rule | Catches ransom | Flags ordinary | J |
|---|---|---|---|
| **rapid pass-through** | **80%** | 50% | **+0.300** |
| the other six | 0–20% | 0–25% | <= 0 |

Read the 80% carefully. A 50% false-positive rate means it also flags half of all innocent
wallets. **It is a filter, not an accusation** — it roughly halves the pile an investigator
must search while keeping four of five real targets in it. Useful for triage, useless as
proof. The UI and the case pack say exactly that.

Why we believe it: the result got *stronger* as we doubled the sample (71%/43% at n=7,
80%/50% at n=15). That is the opposite of what happens when you have fooled yourself.

**Why the other six fail:**

1. **Thresholds tuned on invented data.** "10 counterparties is suspicious" is meaningless
   on a chain where ordinary wallets touch hundreds.
2. **Exchanges look exactly like launderers.** They legitimately receive from thousands and
   pay out to thousands. Fan-out fires *three times more* on ordinary money.

**Accidental discovery:** 500 ransomware addresses formed 342 entities; 500 ordinary
addresses collapsed into 74. Ordinary people reuse wallets ~7x more than criminals do.

**Also:** five genuinely OFAC-sanctioned Bitcoin addresses turned up in the crawl.

Results are written to `data/processed/real_experiment.json`.

## 11. If a judge asks you something

**"Isn't this just a Kaggle dataset project?"**
No. Most published work classifies individual *transactions*. Nobody arrests a
transaction. We group wallets into actors first, then judge the actors. And we tested the
detectors against real ransomware wallets, not only against our own data.

**"How do you know which gang owns a wallet?"**
We don't name gangs. We prove which wallets share an owner, using a published 2013
technique, and cross-reference public lists of known ransomware and sanctioned addresses.
Naming requires exchange KYC records, which is law enforcement's job, not ours.

**"Do your labels prove someone is a criminal?"**
No, and we say so on every screen. We predict investigative *priority*. The blockchain
record is the evidence; our tool decides what an analyst looks at first.

**"What if it flags an innocent person?"**
That is why every flag carries its reasoning, its supporting evidence, and the facts
arguing against it. It is designed for human review, never automatic accusation.

**"Does it work on live Bitcoin?"**
The detection layer does — we demonstrated it on live blockchain data. Full deployment
would need a Bitcoin node and proper ingestion infrastructure, which we list as future
scope rather than pretending we've built it.
