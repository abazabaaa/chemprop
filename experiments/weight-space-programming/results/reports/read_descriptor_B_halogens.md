# Reading task, model B

- **True target:** `halogens` (count of F, Cl, Br, I atoms)
- **Agent's answer:** lipophilicity / logP
- **Verdict:** WRONG. The true target did not appear among its alternatives.
- **Note:** it found halogens dominating the network and in the order I ≈ Br > Cl > F, then read that
  ordering as the Hansch lipophilicity scale rather than as a count. Its claim that the training set
  contains disconnected counterions is partly right and partly wrong: the pool does contain four
  single-atom entries (`[Ag]`, `[Ca]`, `[Hg+2]`, `[Mg]`), but no multi-component salts, because
  SMILES containing `.` are excluded from the pool.

Report as delivered, verbatim:

---

## 1. Property

**Lipophilicity — the octanol/water partition coefficient, logP (logKow).** A per-atom-additive hydrophobicity scale, learned as a standardized regression target.

## 2. Confidence

**75%** for "a lipophilicity/logP-type partition coefficient"; ~55% for logP specifically rather than logD7.4.

## 3. Runners-up

1. logD at pH 7.4 (MoleculeNet-style "Lipophilicity")
2. Membrane permeability / BBB partitioning (logPe, logBB) — same polarity axis
3. Negative log aqueous solubility (−logS) / log Henry's-law constant

## 4. Reasoning

Linearizing the head (r = W2·W1; r is 99.9% correlated with the bias-gated version, so ReLU gating does not flip signs) and pushing input columns through the direct (r·W_o,x), 1-hop (r·W_o,m·W_i) and 2-hop (…·W_h·W_i) paths gives a total derivative per feature.

**Element ordering (relative to C, including each element's `mass*0.01` term, coefficient +12.9):**
I +33.3 ≈ Br +33.3 > Cl +30.1 > F +25.0 ≫ Se +7.5, B +6.9 > **C 0** > P −2.8, N −4.8, As −7.3, Si −11.2, O −20.4, S −22.8.

The halogen order **I ≥ Br > Cl > F** reproduces the Hansch π lipophilicity constants (0.14/0.71/0.86/1.12) in the right sequence, while O and S are the most negative atoms. SVD of W_o's element block shows one dominant axis (68% of variance, σ = 3.50 vs 1.19), and the readout direction r is **88% aligned** with it — the network's single main learned atom axis *is* this polarity axis.

Corroborating signs: mass **+12.9** (bigger = higher, rules out solubility/volatility, which need mass negative); formal charge all negative (−1: **−16.3**, +2: −4.0, −2: −4.8 — ionized species partition poorly); triple bond **−17.8** (nitriles); ring bond −6.1, single bond +3.5 (chain extension raises y); sp3 +5.0, aromatic atom +3.9 (weak); degree=1 +12.8 (terminal CH3/halogen caps).

This excludes the sign-inverted candidates: logS, TPSA, H-bond counts (all need O/N positive); MW/molar refractivity (need S, I positive); melting/boiling point (need O/N positive for H-bonding).

**Training-set fingerprint** (exact test: a column sits at its kaiming bound 1/√fan_in unless that feature occurred). Absent: He, Li, Be, Ne, **Na, K**, Al, Sc, Ti, **Fe**, Ga, Ge, Kr. Present: H, B, C, N, O, F, Mg, Si, P, S, Cl, Ca, V, Cr, Mn, Co, Ni, Cu, Zn, As, Se, Br, I, other. Mg, Ca, charge=+2 and nH=4 moved **only in W_o, not W_i** (e.g. Mg: 0/64 vs 23/64) — W_i only sees bonded atoms, so these occur solely as *disconnected* ions: Mg²⁺/Ca²⁺/NH₄⁺ counterions. Degree=0 likewise 0/64 in W_i, 60/64 in W_o. So the data contains salts (consistent with sulfonate/carboxylate salts, explaining S being the most negative element), plus organophosphates, arsenicals and organosilicons — a physicochemical/environmental logP collection rather than pure ChEMBL.

## 5. Audit

Files opened: only `architecture.md`, `feature_layout.md`, `model.npz` (all in the blind folder). I wrote one helper of my own in a scratch directory.

8 Python runs (no forward passes, no chemistry libraries):
1. Load npz; shapes, stats, b1/b2/W2/b_o.
2. Write helper; per-column max|w| and std of W_i/W_o vs init bounds.
3. Head analysis: per-unit b1/W2, r = W2·W1 vs bias-gated r.
4. Linearized effects r·W_o,x, r·W_o,m·W_i, r·W_o,m·W_h·W_i.
5. Grouped direct/1-hop/2-hop table for element, degree, nH, hybridization, charge.
6. Exact init-bound count test for which features occur in training.
7. SVD of the element block; element effects relative to carbon.
8. Element effects combined with the mass term; bond-feature totals.
