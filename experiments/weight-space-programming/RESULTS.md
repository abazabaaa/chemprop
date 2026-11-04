# Results

Every number here comes from the committed pipeline in `src/`, not from the exploratory session.
Read [VERIFICATION.md](VERIFICATION.md) first: there are no control conditions anywhere in this
table, so the reading results have no null to be compared against.

Reproduce with:

```bash
python -I experiments/weight-space-programming/src/make_datasets.py
python -I experiments/weight-space-programming/src/train_models.py
python -I experiments/weight-space-programming/src/check_handwritten.py
python -I experiments/weight-space-programming/src/score.py runs/models/rot_baseline_trained.npz runs/data/rot_test.csv
```

## Setup

3,346 unique molecules, pooled from this repository's own `tests/data` CSV fixtures: single
component, at most 60 heavy atoms, canonicalised and deduplicated. Every model is the same
architecture - a chemprop D-MPNN with `d_h=64`, `depth=3`, sum aggregation over atoms divided by
100, and a one-hidden-layer head - for 22,593 parameters. Trained on CPU with Adam, a fixed seed and
`deterministic=True`. Only the eight learnable tensors are exported; the target scaler is discarded,
so a reader of the weights cannot recover the target's units or spread.

**The committed pipeline regenerates the exact weights the agents read.** All eight reading models
come back bit-identical (largest element-wise difference 0.0) to the files used in the session, so
every reading result below can be audited without trusting any binary in this branch.

## Experiment 1: reading structural descriptors

Five networks, five targets, one shared molecule pool - so the element inventory is identical across
targets and cannot hint at which target a given model has.

| Model | True target | Held-out R² | Agent's answer | Confidence | Verdict |
|---|---|---|---|---|---|
| A | rotatable bonds | 0.993 | melting point | 65% | wrong |
| B | halogen count | 0.993 | logP / lipophilicity | 75% | wrong |
| C | H-bond donors | 0.996 | H-bond donors, RDKit `NumHDonors` | 70% | **correct** |
| D | aromatic atom count | 1.000 | *fraction* of atoms aromatic (count was 3rd choice) | 55% | partial |
| E | nitrogen count | 0.997 | nitrogen count | 65% | **correct** |

The two misses are the interesting part, because neither failed at reading the weights:

- **B** found halogens dominating the network, ordered I ≈ Br > Cl > F, and read that ordering as the
  Hansch lipophilicity scale rather than as a count.
- **A** found that acyclic single bonds raise the output (+5.6) while ring bonds (−19.8) and triple
  bonds (−9.6) lower it, then interpreted rigidity as raising a melting point. Its global sign was
  inferred from chemical plausibility, which it flagged as its weakest step.

In both cases the features were right and the label was wrong. During the session I read this as
"agents assume any chemprop model predicts a laboratory measurement". That is a hypothesis, not a
finding, and VERIFICATION.md section 3 says how to test it.

## Experiment 2: reading real measured and computed endpoints

| Model | True target | Source | Held-out R² | Agent's answer | Confidence | Verdict |
|---|---|---|---|---|---|---|
| P | log aqueous solubility | `regression.csv` (ESOL / Delaney) | 0.856 | log aqueous solubility, "ESOL/Delaney-style" | 60% | **correct** |
| Q | HOMO-LUMO gap | `regression/mol_multitask.csv` (QM9) | 0.814 | HOMO-LUMO gap, QM9 `gap` | 62% | **correct** |
| R | dipole moment | `regression/mol_multitask.csv` (QM9) | **0.291** | dipole moment, QM9 `mu` | 70% | **correct** |

Three for three, and model R is the one worth attention: its held-out R² is 0.291, meaning the
network barely learned its target, and the target was still named correctly. The agent read a
bond-dipole ordering straight out of the weights - NO₂ > C=O > C≡N > ether > hydroxyl > 0 > C-F >
CH₃ > CH₄, with methane lowest - which is the right physical ordering.

**But do not compare 3/3 here against 2/5 above.** These three were trained on *different* datasets,
and both QM9 agents identified the dataset before reasoning about the target, from the fact that
only C, N, O and F columns had moved away from their initialisation values. Identifying QM9 narrows
the answer to roughly a dozen candidates before a single trained weight is interpreted. Experiment 1
offered no such cue. The comparison is confounded; VERIFICATION.md section 2 explains the fix.

Both QM9 agents' dataset claims are independently checkable and both are true: QM9 is C/N/O/F with
no stereochemistry. Model R's agent additionally noticed that `total_num_Hs=4` had moved in `W_o`
but not in `W_i` - an atom with four hydrogens and no bonds, i.e. methane as a lone node.

One claim of this kind was **wrong**: model B's agent concluded the training pool contained
disconnected salt counterions. The pool does contain four single-atom entries (`[Ag]`, `[Ca]`,
`[Hg+2]`, `[Mg]`), but no multi-component species at all, because SMILES containing `.` are excluded.
The inference from "moved in `W_o` but not `W_i`" to "salts" over-reached.

## Experiment 3: writing weights by hand

Target: the simplified rotatable-bond rule in `specs/target_rotatable_simple.md`. The agent got the
architecture, the feature layout and the rule with worked examples; no training, no gradients, no
molecule data, no chemistry libraries, and a budget of 20 aggregate-only evaluations.

**It used one.**

| | MAE | R² | exact after rounding |
|---|---|---|---|
| **Hand-written, no training** | **3.3e-07** | **1.0000000** | **100.0%** |
| Gradient-trained, same architecture | 0.227 | 0.9915 | 91.3% |
| Always predict the practice-set mean | 2.48 | −0.00 | 11.1% |
| Always predict 0 | 3.12 | −0.83 | 22.5% |

On the 1,673 held-out molecules. The hand-written network sets 74 of 22,593 weights; every value is
a round hand-chosen number (0.5, 1, 2, 4, 10, 11, 12, 20, 100 and their negatives). Its generating
program is 78 lines: [`handwritten/build_weights.py`](handwritten/build_weights.py).

### The design

Count each qualifying bond as 0.5 at *each* of its two ends, so summing over atoms gives the bond
count. Three edge units: one detects a triple bond, two form a relu pair whose difference is the
indicator "this bond is single, not in a ring, and its source atom has at least two heavy
neighbours and no triple bond". The message-passing step is used to let each bond see triple bonds
at its far end. Two atom-readout units form a second relu pair that applies the same test to the
near end. Because `W_i` has no bias, the constants ride on the formal-charge one-hot block, which
always sums to 1 - a detail worth noticing.

### Verification (`src/check_handwritten.py`, all checks pass)

| Check | Result |
|---|---|
| `build_weights.py` reproduces `weights.npz` | exact, largest difference 0.0 |
| 1,673 held-out molecules | 100% exact |
| 38 adversarial molecules (SF₆, zwitterions, Sn/Se/B compounds, 12-membered ring, chained triple bonds) | 100% exact |
| 500 random synthetic graphs, features consistent with topology | 100% exact |
| 500 random synthetic graphs, features deliberately inconsistent | 100% exact |
| explicit hydrogens as graph nodes (`add_h=True`, not the default) | **over-counts, as predicted by the agent itself** |

The random-graph checks are the ones that matter. They leave the space of chemically reachable
molecules entirely - arbitrary topology, arbitrary bond types, and in the second variant the
declared degree and hydrogen-count features contradict the actual adjacency - and the network is
still exactly right against a direct implementation of the rule. That is what "exact by
construction" has to mean to be worth saying.

### Two caveats that matter

**The comparison with gradient training is not a fair fight.** The baseline is one seed, 60 epochs,
1,673 molecules, no validation split, no early stopping, no tuning. How soft it is shows up
directly: the same recipe scored 95.6% exact on the session's split of the same pool and 91.3% on
the committed split - a 4-point swing from nothing but a different random half. The defensible claim
is "exact where training is approximate", not "beats gradient descent". VERIFICATION.md section 4c
describes a baseline worth comparing against.

**The target was handed over in full.** This measures compilation of a stated program into weights,
not discovery of a property. For context, the same network scores 76.2% against RDKit's *stricter*
`CalcNumRotatableBonds` definition - which is exactly the ceiling the simple rule allows on this
pool, i.e. the network implements its specification perfectly and loses nothing else. Two earlier
attempts at the strict rule itself failed outright; see
[`prompts/write_prompt_failed_attempts.md`](prompts/write_prompt_failed_attempts.md).

## Scoreboard, stated honestly

| Claim | Status |
|---|---|
| An LLM can hand-write chemprop weights that compute a specified graph program exactly | **supported**, mechanically checkable, one target |
| It can do so zero-shot, with no evaluation feedback | **untested** - the run used 1 of 20 passes |
| It can hand-write weights for a program it was not given | **untested** |
| An LLM can name a network's training target from its weights | **suggestive, 5/8** - no control condition |
| It can recover the training set's element inventory from untouched weight columns | **suggestive** and objectively scorable; make this the primary endpoint |
| Real endpoints are easier to read than synthetic descriptors | **confounded** with dataset identifiability |
| Target legibility survives a badly fitted model (R² ≈ 0.29) | **one observation**, and the most interesting one here |
| Hand-written weights beat gradient descent | **not established** - the baseline is untuned |
