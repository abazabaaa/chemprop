# Reading task, model D

- **True target:** `aromatic_atoms` (count of atoms with RDKit's aromatic flag set)
- **Agent's answer:** the *fraction* of heavy atoms that are aromatic
- **Verdict:** PARTIAL. It put the count third among its alternatives and was ~90% confident the
  target measured aromatic content, but chose the normalised version over the count.
- **Note on rule compliance:** this agent reported that one of its Python runs "brushed the
  no-forward-pass rule", and said it discarded that run's output. Self-reported, unverified; a
  future harness should enforce the rule rather than ask for it (see `VERIFICATION.md`).

Report as delivered, verbatim:

---

## 1. Property
**The fraction of heavy atoms in the molecule that are aromatic** — i.e. the normalized aromatic content ("aromatic proportion" = NumAromaticAtoms / NumHeavyAtoms).

## 2. Confidence
**55%** that the normalization is right; ~90% that the target is a measure of *aromatic content* of some form.

## 3. Runners-up
1. Number of aromatic rings (unnormalized count) — 25%
2. Number/count of aromatic atoms — 8%
3. Fraction of aromatic (vs. sp3) carbons, i.e. roughly 1 − FractionCSP3 — 5%

## 4. Reasoning
Sensitivities are `(W1ᵀ(mask·W2))·mask_o/100 · W_o` columns, averaged over 6,000 random ReLU gate masks; `signP` = fraction of masks agreeing on sign. Numbers ×100.

**Aromaticity dominates everything.** `is_aromatic` is the single most sign-robust feature in the network: **+0.170, signP = 1.00** direct, and **+0.347, signP = 1.00** propagated as an edge source. Its `W_i` column is the most-grown atom column (1.55× PyTorch init norm 0.498); its `W_o` column 1.56×. Aromatic bond type: **+0.218, signP = 1.00**; SINGLE −0.087, DOUBLE −0.172, TRIPLE −0.058.

**It's aromatic rings, not rings:** `is_in_ring` is effectively untrained (+0.005, signP 0.55, column 1.18× init).

**Rules out the usual suspects.** `mass` is unused (−0.015, signP 0.66; columns 1.36×/1.09× init) → not MW, molar refractivity, polarizability or boiling point. Halogens are net **negative** (Cl −0.205, Br −0.206, F −0.159, signP 0.83–0.85) → not logP/logD/density/refractive index. N and O negative (hydroxyl O −0.130, amine N −0.054, carbonyl O −0.072) → not TPSA, H-bond counts, hydration free energy. Aromatic positive → not logS or HOMO–LUMO gap (wrong sign).

**Evidence for normalization (fraction, not count):**
- Per-atom baseline `ra·b_o` ≈ **+0.007** (signP 0.59) — the net has no size term; it is not counting atoms.
- Every non-aromatic atom type is penalized (CH3 −0.246, CH2 −0.065, Cl −0.205), consistent with dilution of a ratio.
- Disconnected counterion features carry the **largest** trained signal: `deg0` is the most-grown of all 72 `W_o` columns (**2.58× init**) with sensitivity −0.238; composites Ca²⁺ −0.636, NH₄⁺ −0.605, Na⁺ −0.357, all signP 0.99–1.00. A counterion carries zero information about an aromatic-ring *count*, but directly lowers an atom *fraction*.
- Head gain along the aromatic direction **saturates**: 0.49 → peak 0.66 (t≈0.4) → 0.26 asymptote (−60%), i.e. a bounded target.
- Ratio |aromatic bonus| / |non-aromatic penalty| implies f ≈ 0.14–0.42, centred ≈0.3 — the typical aromatic-atom fraction of drug-like molecules.

## 5. Audit
Files opened (all three, nothing else): `architecture.md`, `feature_layout.md`, `model.npz`.

7 Python runs: (1) load tensors, shapes/norms/head importances; (2) open-gate linearized per-feature sensitivities; (3) scaled a `b_o`-derived vector through the head — **this brushed the no-forward-pass rule, so I discarded its output and did not use it**; (4) 4,000-mask sign-robustness of atom/bond sensitivities incl. multi-hop `W_h` paths; (5) SVD of `W_1`/`W_o` plus column-norm growth vs. init; (6) piecewise-linear head gain along feature weight directions; (7) 6,000-mask composite per-atom-type sensitivities and implied-fraction ratio test.
