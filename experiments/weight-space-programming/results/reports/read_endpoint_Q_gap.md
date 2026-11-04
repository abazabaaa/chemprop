# Reading task, model Q

- **True target:** `gap` from `tests/data/regression/mol_multitask.csv` (QM9 HOMO-LUMO gap, Hartree)
- **Agent's answer:** the HOMO-LUMO gap, named as QM9's `gap`
- **Verdict:** CORRECT.
- **Note:** it also recovered the training set's element inventory (C, N, O, F plus implicit H) and
  the absence of stereochemistry from which features sat exactly at their initialisation values.
  Both facts are true of QM9 and are independently checkable, which makes this report the easiest
  one to audit.

Report as delivered, verbatim:

---

**1. Property:** The **HOMO–LUMO gap** (ε_LUMO − ε_HOMO, in Hartree) of a small QM9 subsample — i.e. QM9's `gap` target.

**2. Confidence:** 62%

**3. Runners-up:** (a) QM9 `lumo` energy ~20%; (b) a sign-flipped HOMO / ionization-potential-like quantity ~8%; (c) a saturation descriptor (Fsp3 / degree-of-unsaturation, negated) ~5%.

**4. Reasoning**

*Which features were trained at all.* Untrained one-hot columns sit exactly at PyTorch init norm (4.619/√fan_in). Ratio-to-init is 0.87–1.12 for **every** atomic_num except C (1.57), N (2.34), O (3.39), F (2.62). S 1.02, Cl 0.95, Br 0.98, I 0.94, P 0.87, Si, B all = init. So the training molecules contained **only C/N/O/F + implicit H** — the QM9 element set. All chirality and bond-stereo columns are also at init (no stereochemistry).

*One dominant axis.* With h = (W2·W1)/100, the direct readout sensitivities P0 = h·W_o[:,:72] have a null σ = 0.0038 (50 untrained columns). Significant entries: numHs=4 **+21.8σ**, O **−19.3σ**, F +15.3σ, N +13.3σ, degree=3 **−13.6σ**, SP3 +13.2σ, degree=4 +10.5σ, SP2 −9.2σ, numHs=3 +8.4σ, mass +5.5σ, charge=+1 −5.1σ, aromatic −4.0σ. SVD of the 21 trained W_o columns: top direction holds 51% of variance and aligns with h at cos = 0.90. Positive pole = {SP3, numHs=4/3/2, degree=4, F, N}; negative pole = {O, degree=3, SP2, aromatic, numHs=1, N+}.

*Bond features.* The largest single weight in the whole network is W_i[55,74] = −0.4785 on **bond_type=DOUBLE**, whose column is 3.60× init (biggest inflation anywhere) with readout z = −28.4σ; is_conjugated is 2.56× init, also negative; SINGLE +7.2σ, in_ring +6.3σ. Unsaturation is the model's #1 predictor and it *lowers* the output.

*Intensive, not extensive.* Per-atom constant h·b_o = +0.018, but the saturated-vs-unsaturated feature split is 23× larger and opposite-signed (CH3 +0.126, sp3 CH2 +0.112, aromatic CH −0.118). No size-monotone term ⇒ rules out MW, α, R², ZPVE, Cv, U0/H/G. O strongly negative rules out TPSA, H-bond-acceptor count, logS, boiling point; methane-marker numHs=4 being maximal rules out μ (and every extensive target, since CH₄ is their minimum).

*Gap vs LUMO.* An sp3 ether/alcohol oxygen (O+SP3+deg2+Hs0 = −0.036) scores ~0.15 below an sp3 CH2 (+0.112) with no π system involved — a lone-pair/HOMO-raising effect that moves the gap but barely moves the LUMO. `is_aromatic` also carries an independent −4.0σ on top of SP2 and degree=3 (gap compressed from both frontier orbitals). F positive is consistent with gap (fluorine lowers HOMO; F-bearing QM9 molecules are small/saturated) — F's W_i column is also the heteroatom column most aligned with SP3 (cos +0.29).

**5. Audit**

Files opened: the blind folder's `architecture.md`, `feature_layout.md`, `model.npz` (numpy only). Nothing else; no web. Scripts r1–r6 written to a scratch directory. Six Python runs, no forward passes:
1. r1 — tensor shapes, magnitudes, biases, block norms.
2. r2 — direct readout sensitivity h·W_o[:,:72] per atom feature, 3 relu-gating variants.
3. r3 — message-path products h·W_o[:,72:]·(W_h^k)·W_i, atom + bond features.
4. r4 — column-norm ÷ init-norm ratios for all 86 inputs (which features were trained).
5. r5 — z-scores vs the untrained-column null; cosine structure of W_o columns.
6. r6 — extensivity check, motif weight sums, top-10 global weights, SVD, W_h spectrum.
