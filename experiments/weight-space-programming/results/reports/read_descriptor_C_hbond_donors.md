# Reading task, model C

- **True target:** `hbond_donors` (RDKit `Lipinski.NumHDonors`)
- **Agent's answer:** number of hydrogen-bond donors, naming `NumHDonors`
- **Verdict:** CORRECT, and specific enough to name the exact RDKit function. It also correctly
  concluded that the target counts donor *groups* rather than donor *hydrogens*, distinguishing
  `NumHDonors` from `NHOHCount`.

Report as delivered, verbatim:

---

## 1. Property
**Number of hydrogen-bond donors** — i.e. the count of heteroatom–H groups (N–H / O–H / S–H) per molecule, as in RDKit `NumHDonors` / Lipinski HBD count.

## 2. Confidence
**70%** (≈85% that it is "an HBD / polar-H count" in some variant).

## 3. Runners-up
1. `NHOHCount` — number of N–H + O–H **hydrogens** (same family, counts H's not groups).
2. TPSA (topological polar surface area).
3. Aqueous solubility logS / another polarity-driven experimental property.

## 4. Reasoning
I built the effective readout direction `u = W1ᵀ(W2 ⊙ mask)` in atom-vector space, then projected it through `W_o` (self-atom block) and through `W_o[:,72:] @ W_i` (neighbour block) to get a per-input-feature sensitivity. The picture is a clean **conjunction: (N or O or S) AND (has ≥1 H) AND (neutral)**.

Neighbour-channel sums for (element + H-count) pairs:

| | H0 | H1 | H2 |
|---|---|---|---|
| N | **−0.2** | **+14.8** | +14.0 |
| O | **−4.2** | **+10.8** | +10.1 |
| S | −2.2 | +12.8 | 12.1 |
| C | −13.0 | +2.0 | +1.2 |

Single-feature sensitivities: N +7.79, S +5.86, O +3.82, H1 +6.99, H2 +6.23 vs **H0 −8.01**, C −5.02, Cl −3.57, deg1 −3.35 (terminal =O / nitrile / halide), deg2 +3.15 (O–H, S–H), deg3 +1.79 (NH₂/NH), aromatic −1.95, DOUBLE −2.33, TRIPLE −4.27. Largest single weight of all: **formal_charge=−1 at −11.5** in the neighbour channel — the net penalises atoms adjacent to an anion, exactly what is needed to separate COOH (1 donor) from COO⁻ (0).

Discriminators:
- **H0 penalty ≈ magnitude of the heteroatom bonus** ⇒ ether O, tertiary/aromatic N, carbonyl O net ≈0 or negative. TPSA and HBA count would all be clearly positive there ⇒ both ruled out.
- chg−1 strongly **negative** rules out logS/melting point (anions are far more soluble / higher-melting).
- C negative, N/O positive rules out logP/MR/MolWt. Per-atom constant is only u·b_o = +0.29 → 0.003 of y per atom ⇒ the target is **not size-extensive**, consistent with a small sparse count.
- H1 ≈ H2 (not 2×) favours counting donor **groups** over donor **hydrogens** (`NumHDonors` over `NHOHCount`).
- S ≈ N in the "has-H" rows matches `NumHDonors`'s `[O,S;H1;+0]` SMARTS (or sulfonyl-adjacent N–H, also donor-consistent).
- Halogen weights are inconsistently ordered (F −0.44, Cl −3.57, Br −1.47, I +0.09) ⇒ not a lipophilicity scale, just "not a donor".
- Per-unit structure confirms it: units with the largest positive readout weight (self: #3 u=1.15, #11 u=0.94, #32 u=0.69; message: #50, #42, #56, #59, #51, #20) each have N/O/S ≈ +0.15…+0.30, H1/H2 ≈ +0.10…+0.22, H0 ≈ −0.10…−0.27, C negative — in the same row. Corr over 64 units: u vs (N,O,S) = +0.73, u vs C = −0.69; message-side r vs (hasH − H0) = +0.70.
- `Whᵀr` (2nd hop) keeps the same sign pattern (N +6.0, H0 −10.5, q− −15.5), so the donor signal is just smeared over each donor's neighbourhood, staying additive.
- Trained column norms (init ≈0.50) are largest exactly on the discriminating features: chg−1 1.40, N 1.13, H0 0.85, H1 0.84, H2 0.79; chg0 0.48 (untrained).

## 5. Audit
Files opened (only these three): `architecture.md`, `feature_layout.md`, `model.npz`, all in the blind folder.

Python runs (10 of 30; numpy only, no forward passes, no molecules/graphs constructed, no chemistry libraries):
1. Shapes/stats of all 7 tensors; b2, b1, W2, b_o printed.
2. Head-unit table (W2, b1, W1 row sums/norms); Σ W2·rowsum = 13.09.
3. Readout direction u under 4 masks; pairwise corr 0.92–0.99; u·b_o = 0.290.
4. Per-feature linear scores for all 72 atom + 14 bond features, self and neighbour channels, plus column norms.
5. Per-a-unit table: u, b_o, key atom-feature weights, message-block norms.
6. Sign-masked message readout r_pos/r_neg; W_h spectrum (ρ=0.956); message-unit table.
7. Source-atom/bond feature scores through the message channel (all vs positive-unit only).
8. Conjunction correlations (u vs polar/C, r vs hasH−H0); SVD of diag(W2)W1.
9. Donor-like vs acceptor-like unit split; hop-1/2/3 sensitivities via Wₕᵀ.
10. (element × H-count) neighbour/self sensitivity table + charge/degree/aromatic terms.
11. Robustness of the (element × H-count) pattern under three different readout masks.
