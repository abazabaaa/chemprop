# How to verify this properly

You are picking this up from an exploratory session. Nothing here was pre-registered, every
experiment ran once, and the grading was done afterwards by the same agent that designed the
experiments. Treat the numbers in `RESULTS.md` as the reason to run a real experiment, not as the
result of one.

This document is in four parts: what the evidence does and does not support; the one control whose
absence matters most; a concrete design for each experiment; and the decisions you should make
yourself rather than inherit.

---

## 1. What the current evidence supports, and what it does not

### Supported, and fairly solid

**A frontier model can hand-write the weights of a chemprop D-MPNN that computes a specified
bond-counting program exactly.** This one survives scrutiny because the claim is mechanical, not
statistical. 74 non-zero weights out of 22,593, all round hand-chosen values; the generating
program is 78 readable lines (`handwritten/build_weights.py`); and the network is exactly correct
on 1,673 held-out molecules, 38 adversarial molecules, and 600 randomly generated synthetic graphs,
including graphs whose declared degree and hydrogen-count features deliberately contradict their
own topology. Only one forward pass was spent, so there was effectively no opportunity to fit.

The honest scope: the target was handed over as a complete specification with 19 worked examples.
This measures *compilation of a stated program into weights*, not discovery of a property.

### Not supported yet

**"An LLM can read a network's target off its weights."** Eight models, one attempt each, five
exact hits. There is no control condition, so there is no null to compare against, and at least one
large confound is visible in the data (next section). The reading results are a reason to run the
experiment, not an experiment.

**"Real properties are easier to read than synthetic descriptors" (3/3 against 2/5).** This was my
reading during the session and I now think it is probably wrong, or at least unsupported. See
below.

**"The hand-written network beats gradient descent."** It scores better than the trained baseline
(100% exact against 95.6%), but that baseline is one seed, 60 epochs, 1,673 molecules, no early
stopping and no tuning. It is not a serious opponent. The defensible claim is "exact where training
is approximate", not "better than training".

---

## 2. The control that matters most, and the confound already visible

### The missing null: nobody checked what an agent says when the weights are uninformative

Every reading run saw a genuinely trained network. Nobody ran the same prompt against weights that
contain no information about any target. "Which property does a small chemprop model predict?" has
a strong prior answer - solubility, lipophilicity, a QM9 target - and that prior alone could produce
hits. Until you have measured it, the hit rate means nothing.

Run at least these conditions, with byte-identical prompts and specification documents:

| Condition | `model.npz` contents | What it measures |
|---|---|---|
| `trained` | the real trained weights | the treatment |
| `init` | the same architecture at initialisation, never trained | the prior: what the agent says with nothing to read |
| `shuffled` | trained weights, entries permuted within each tensor | whether structure matters, or only marginals |
| `scrambled-pairing` | trained weights from a *different* target in the set | guards against a grader who rewards any plausible chemistry |

`init` is the single most important one. If an agent handed an untrained network still answers
"log aqueous solubility" with 60% confidence, the headline result evaporates.

### The confound I can see in my own data

The five descriptor models were all trained on the **same** molecule pool. The three endpoint
models were trained on **different** datasets (ESOL for P; QM9 for Q and R). Both endpoint agents
that hit then leaned explicitly on identifying the dataset: they noticed that only C, N, O and F
columns had moved away from their initialisation values, concluded QM9, and reasoned from QM9's
short list of targets. Model R's agent went further and spotted that `total_num_Hs=4` had moved in
`W_o` but not in `W_i`, i.e. an atom with four hydrogens and no bonds - methane as a lone node.

That is genuine, impressive weight-reading. But it means the comparison "3/3 on real endpoints
versus 2/5 on descriptors" is not a comparison of property types at all: it is confounded with
whether the dataset was identifiable from the element inventory. Identify QM9 and you have narrowed
the answer to about a dozen candidates before reading a single trained weight.

**The fix is cheap: train every model in the comparison on one shared molecule pool.** Then the
element fingerprint is identical across targets and carries no information about which target you
are looking at. If you want to keep the dataset-identification result, measure it separately - it is
objectively scorable (below) and does not need a judge.

---

## 3. Experiment 1+2 redesign: reading the target off the weights

### 3a. Make the objective part primary

Feature-inventory recovery is the one reading claim with free ground truth: for each of the 86 input
features you know whether it ever occurred in training, and an agent's claim about it is simply
right or wrong. Score precision/recall/F1 over the 86 features. No judge, no taxonomy, no
interpretation. Make this the primary quantitative endpoint, with target-naming secondary.

The mechanism the agents used is worth stating because it makes a sharp prediction you can test:
a weight column for a feature that never fires receives no gradient, so it sits exactly at its
PyTorch initialisation bound. Several agents tested this to the last decimal. So you can also run a
`scrubbed` condition - resample the never-updated columns from a fresh initialisation - and check
that inventory recovery collapses while target-naming does or does not. That cleanly separates
"reading the trained structure" from "reading the untouched initialisation".

### 3b. Target-naming, done so the number means something

- **Targets:** at least 12, all trained on the **same** molecule pool, spanning structural counts
  (nitrogens, halogens, aromatic atoms, H-bond donors, rings, rotatable bonds), whole-molecule
  descriptors (TPSA, Crippen logP, molar refractivity, fraction sp3) and real endpoints where the
  pool supports them. Include at least two targets that are *not* standard benchmarks, so a
  prior-driven guesser cannot be right by naming a famous dataset.
- **Conditions:** the four in the table above.
- **Repeats:** k >= 5 independent agent runs per (target, condition) cell. One run per cell is what
  this session did and it is why there are no error bars anywhere.
- **Blinding:** the agent gets a folder path and a prompt, and the prompts must be character-identical
  across targets and conditions apart from the path. Check this mechanically with a hash, because it
  is easy to get wrong - I got it wrong (see Deviations).
- **Grading:** write the taxonomy of acceptable answers for each target *before* running anything.
  Then have a separate judge agent, which never sees the truth, map each free-text answer onto that
  taxonomy; compare afterwards. Record `exact`, `same-family` and `wrong` separately, and publish the
  taxonomy. Do not let the experiment's author grade free text against their own expectations, which
  is exactly what happened here.
- **Analysis:** pre-specify the primary statistic (I suggest the difference in exact-match rate
  between `trained` and `init`), the test (permutation over condition labels), and the number of runs,
  before looking. Report the full confusion matrix of answers against targets.

### 3c. Stratify by model quality, deliberately

Model R - the dipole model - had a held-out R² of about 0.29 and its target was still named
correctly. That is the most surprising single observation in the whole session, because it suggests
what is legible is the *direction the weights moved*, not the model's competence. It is also the
observation most likely to be a fluke of one run.

So vary fit quality on purpose: for a fixed target, train checkpoints spanning (say) R² of 0.2, 0.5,
0.8 and 0.99, and measure naming accuracy as a function of fit. A monotone curve would be a real
finding. Use the same target throughout so the only thing changing is how well the network learned.

---

## 4. Experiment 3 redesign: writing weights

### 4a. A ladder, including rungs that should fail

One target that succeeded tells you almost nothing about where the capability ends. Use a graded
set, and include items you expect to be impossible so the design can produce a negative:

| Rung | Target | Why it is there |
|---|---|---|
| 1 | count heavy atoms | trivial; checks the harness, not the model |
| 2 | count nitrogen atoms | one-hot lookup, no message passing needed |
| 3 | count aromatic atoms | as above, different feature |
| 4 | simple rotatable bonds | **done**: needs messages to see both bond ends |
| 5 | H-bond donors | needs a conjunction over element and hydrogen count |
| 6 | strict rotatable bonds | the full RDKit rule, with amide/ester/CF3/tert-butyl exclusions (`specs/target_rotatable_strict.md`); this is what killed two attempts |
| 7 | number of rings | **expected to be impossible** at depth 3: ring count is a global property (edges - atoms + components), and the architecture has no mechanism for it. If an agent claims success here, something is wrong with your harness |
| 8 | Crippen logP contributions | exactness is unreachable; this is where the comparison with a *tuned* trained baseline is the actual measurement |

Rung 7 matters more than it looks. A protocol that cannot produce a failure is not measuring
anything, and this one has a clean a-priori reason to fail.

### 4b. Primary endpoint: random synthetic graphs, not molecules

Score on randomly generated graphs with a reference implementation of the rule
(`src/check_handwritten.py` does this already). It is objective, it has no distributional overlap
with any training pool, and it cannot be passed by a network that merely fits the molecules it was
shown. Report real-molecule metrics as secondary.

### 4c. Ablate the feedback, and fix the baseline

- **Forward-pass budget:** run 0, 1 and 20 as separate conditions. The successful run used 1 pass,
  so "zero-shot" is *untested* - and a 0-pass condition is the strongest version of the claim.
- **Baseline:** for each rung, train the same architecture properly - at least 5 seeds, a validation
  split, early stopping, a small learning-rate sweep - and report its best. Then the comparison is
  informative. Also report the trained baseline at matched *information*, i.e. how many labelled
  molecules it needs to reach the hand-written network's accuracy. That framing ("the spec is worth
  N labelled examples") is more interesting than a win/loss.
- **Repeats:** k >= 5 agent runs per (rung, budget) cell, reporting the distribution, not the best run.

### 4d. Separate the two confounded changes

The successful attempt changed the target *and* the prompt at the same time. Run the 2x2:
{strict, simple} target x {with, without} the short-response instruction, several runs per cell.
Without this you cannot say whether the earlier failures were about the task or the harness. See
`prompts/write_prompt_failed_attempts.md` for what the failures looked like.

---

## 5. Harness requirements: stop asking, start enforcing

Every rule in these experiments was prose in a prompt, and compliance was self-reported. Two agents
reported brushing up against the rules (model D said it discarded a run that came too close to a
forward pass; model E pushed hand-built feature vectors through individual layers). I believe both
reports, and that is the problem: belief is not a method.

- **Filesystem.** Run each agent with only its task folder readable: a container with a single bind
  mount, or a separate OS user with no read permission anywhere else. Then "do not look at the
  labels" is a property of the system, not a request.
- **Logging.** Record every tool call and every path opened for every run, and publish the logs
  alongside the results. A `PreToolUse` hook is enough.
- **No network.** Enforce it, do not request it.
- **The forward-pass rule.** This is the one that needs rethinking rather than enforcing. "Do not
  simulate the network" is not meaningfully enforceable against an agent with numpy and the
  architecture document - and arguably it should not be, since the original blog post's harder
  experiments *allowed* metered forward passes. Pick one of two clean regimes and say which:
  (a) **no evaluation at all**: install only numpy, state that any evaluation of the network on any
  input is out of scope, and accept that you are relying on the logs to show what was computed; or
  (b) **metered evaluation**: hand the agent a budgeted forward-pass tool, as the writing task
  already does, and report the budget used as part of the result. Regime (b) is enforceable and is
  what I would choose. Either way, drop prose rules you cannot check.

---

## 6. Decisions I would leave to you

I have opinions, but these are genuinely open and you should not inherit my answers:

1. **Is "name the target" the right question at all?** It conflates reading the weights with
   knowing the chemistry literature. A cleaner operationalisation: give the agent two trained models
   and ask which one was trained on target X - forced choice, chance is exactly 50%, no judge needed,
   no prior about famous datasets. It tests less, but what it tests is unambiguous. Consider making
   this the primary reading experiment and free-text naming the secondary, exploratory one.
2. **How much to tell the agent about the input featurisation.** I handed over a complete feature
   layout. That is arguably most of the work: the agents repeatedly used the exact initialisation
   bounds of specific named columns. A harder and more realistic condition gives the weights and the
   architecture but *not* the feature semantics, and asks the agent to work out what the input
   columns mean. I suspect that is where this capability actually breaks, and it would be the more
   interesting experiment.
3. **Which model, and recorded how.** Agent runs are stochastic and the serving model can change
   under you - it changed during this session (see Deviations), so the nine runs here cannot be
   attributed to one model version with confidence. Decide whether you are measuring a model or a
   capability, pin and record the model identifier per run either way, and fix the temperature.
4. **Whether the molecule pool is good enough.** It is a grab bag scraped from this repository's
   test fixtures: 3,346 molecules, a few bare metal atoms, no multi-component species, nothing
   curated. Fine for a pilot. If the reading experiment is going to carry weight, use a real curated
   set with a documented provenance, and make the element inventory identical across targets.
5. **What would change your mind.** Write this down before running. Mine: if the `init` control
   produces exact target names at anything near the `trained` rate, I would conclude the reading
   results are prior-driven and abandon that line; the writing result I would still believe, because
   it is checkable mechanically and the artefact is right there in 78 lines.

---

## 7. Deviations between the original session and this branch

Reproducing the exact numbers from the session is not possible, and these are the reasons. The
committed pipeline is the canonical one; `RESULTS.md` reports its numbers.

1. **Molecule pool ordering, affecting the writing task's split only.** The session's ad-hoc
   splitting script ordered the pool by the input SMILES string before shuffling; `src/common.py`
   orders by canonical SMILES. The pool is the same 3,346 molecules and the eight reading models are
   unaffected (they come back bit-identical, below), but the practice/test split for the *writing*
   task differs. The hand-written network is exact under either split; the trained baseline is not,
   and moves from 95.6% exact on the session's split to 91.3% on the committed one. That 4-point
   swing from nothing but a different random half is itself worth keeping in mind when reading any
   single-seed baseline here.
2. **Adversarial set.** The session used 41 molecules including a two-component salt and a bare
   `[C]`; the committed set has 38, dropping multi-component and degenerate entries. 100% exact
   under both.
3. **The architecture document was not identical across reading runs.** The descriptor runs got a
   version saying "~2,800 molecules" and "60 epochs"; the endpoint runs got "a few hundred
   molecules" and "150 epochs". Both are hints, and the second one is a real leak towards small
   benchmark datasets. `specs/architecture_read.md` removes training-set size and epoch count
   entirely. **This alone is a reason to re-run the reading experiment before citing it.**
4. **The writing task changed twice.** Target (strict then simple) and prompt (short-response
   instruction added) changed together; see section 4d.
5. **The serving model changed mid-session.** The session began reporting the configured model as
   `claude-opus-5-5` and later as `claude-opus-5`. Subagents inherit the parent's model, so the
   nine runs were probably not all served by the same version, and I cannot say which ran where.
   No per-run model identifier was recorded. Record one.
6. **The descriptor training run was interrupted** by a background time limit and resumed by
   reloading the three models that had finished. `src/train_models.py` trains in one pass. This
   turned out not to matter at all: all eight reading models regenerate **bit-identically** to the
   files the agents read (largest element-wise difference 0.0 across all eight). So the reading
   results are fully auditable from the committed code, and none of the deviations above touch
   them - deviation 3, the non-identical architecture document, is the one that does.
7. **Nine agent runs, no repeats, no logs.** The reports in `results/reports/` are the complete
   record. Full transcripts were not retained.
