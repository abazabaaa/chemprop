# Weight-space programming on chemprop models

Can a language model read a trained chemprop network's weights and work out what it predicts? And
can it write working weights by hand, with no training at all?

This directory is a pilot of both questions on small chemprop D-MPNNs, prompted by
["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang & Khattab, MIT CSAIL, November 2026), which reported that a frontier model can
identify what a network was trained to do from its raw weights, and can hand-write small networks
that outperform ones trained with PPO. We ran the molecular version: same idea, chemprop
architecture, targets taken from this repository's own test fixtures.

## Status: read this before citing anything here

**This is exploratory work, not a result.** Nine agent runs, one attempt each, no control
conditions, no pre-registration, and the grading was done afterwards by the same agent that
designed the experiments. One finding looks solid because it is mechanically checkable; the rest is
suggestive at best, and at least one visible confound is likely to explain part of it.

Read **[VERIFICATION.md](VERIFICATION.md)** before trusting or repeating any of this. It sets out
what the evidence does and does not support, the control whose absence matters most, the confound
already visible in the data, a concrete design for doing it properly, and the ways the committed
pipeline differs from the session that produced the numbers.

Headline, stated at the strength it deserves:

- **Writing weights by hand: solid.** An agent set 74 of 22,593 weights, by reasoning alone, and
  produced a network that computes a specified bond-counting rule *exactly* - on every held-out
  molecule, every adversarial molecule, and 600 randomly generated synthetic graphs. It spent one
  of its twenty allowed evaluations. The whole thing is 78 readable lines in
  [`handwritten/build_weights.py`](handwritten/build_weights.py).
- **Reading the target off the weights: unproven, and interesting enough to test properly.** Five of
  eight models were named exactly, with reasoning that is specific and checkable. But nobody ran the
  same prompt against untrained weights, so there is no null to compare against.

## Layout

```
specs/          the documents handed to agents: architecture, input feature layout, target rules
prompts/        the exact prompts, including the two that failed and why
handwritten/    the hand-written network: its generating program and its weights
src/            dataset construction, training, scoring, task assembly, verification
results/        answer keys, metrics, and all nine agent reports verbatim
RESULTS.md      every number in one place, with the claims graded by how well they hold up
VERIFICATION.md what is wrong with this pilot and how to do it right
```

Agent reports in `results/reports/` are the primary evidence and are reproduced verbatim, including
the parts that turned out to be wrong, the self-reported rule compliance, and one agent's note that
it came close to breaking a rule. Each has a short header saying what the true target was and
whether the answer was right.

## Reproducing

CPU only; no GPU needed. From the repository root:

```bash
python -I experiments/weight-space-programming/src/make_datasets.py      # ~1 min
python -I experiments/weight-space-programming/src/train_models.py       # ~25 min, 9 models
python -I experiments/weight-space-programming/src/make_tasks.py         # assembles agent folders
python -I experiments/weight-space-programming/src/check_handwritten.py  # verifies the hand-written net
```

Everything lands in `runs/`, which is git-ignored. `check_handwritten.py` needs only
`make_datasets.py` to have run, and exits non-zero if any check fails.

Training is deterministic enough that this regenerates the exact models the agents read: all eight
come back bit-identical to the files used in the session, so the reading results can be audited
without trusting any binary committed here.

To run an agent task, hand it one folder from `runs/tasks/` and the matching prompt from
`prompts/`. Each reading folder contains exactly three files - the weights, the architecture and the
feature layout - and nothing else. The answer keys stay in `runs/`, outside the folder.

## The three experiments

**1. Reading structural descriptors.** Five identical networks trained on five different structural
counts (rotatable bonds, halogens, H-bond donors, aromatic atoms, nitrogens) over the same 3,346
molecules. An agent gets one model's weights and must say what it predicts. Two exact hits, one near
miss, two misses - and the misses are informative: both found the right features and then named a
laboratory measurement rather than a count. The halogen model's agent found halogens dominating in
the order I ≈ Br > Cl > F and called it lipophilicity; the rotatable-bond model's agent found
acyclic single bonds raising the output and ring bonds lowering it, and called it melting point.

**2. Reading real measured and computed endpoints.** Three networks trained on properties that
already ship with this repository: ESOL log aqueous solubility, and QM9's HOMO-LUMO gap and dipole
moment. All three named correctly, including the dataset. Note the confound discussed in
VERIFICATION.md section 2: unlike experiment 1, these three were trained on *different* datasets, and
both QM9 agents identified the dataset from its element inventory before reasoning about the target.

**3. Writing weights by hand.** An agent is given the architecture, the feature layout and a precise
specification of a bond-counting rule, then has to set the weights with no training, no gradients and
no molecule data, with a budget of twenty aggregate-only evaluations. Result above.

## Environment

Produced on Python 3.13.18, chemprop 2.3.1 (this repository at `db0bb86`), torch 2.15.0, RDKit
2026.03.6, 4 CPU cores, no GPU. Training uses `deterministic=True` and a fixed seed; the held-out R²
values reproduce to three decimals across runs on this machine.

Agent runs were Claude Code subagents inheriting the session's model. The session's configured model
identifier changed from `claude-opus-5-5` to `claude-opus-5` partway through, and no per-run
identifier was recorded, so the nine runs cannot be attributed to a single model version. Record one
if you repeat this.
