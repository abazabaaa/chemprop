# Reading task, model R

- **True target:** `mu` from `tests/data/regression/mol_multitask.csv` (QM9 dipole moment, Debye)
- **Agent's answer:** the molecular dipole moment, named as QM9's `mu`
- **Verdict:** CORRECT.
- **Why this is the most interesting of the eight:** model R is by far the worst-fitting model in
  the set (held-out R^2 about 0.29, against 0.81-1.00 for the others). The network barely learned
  its target, and the target was still legible in the weights. If that holds up under the controls
  in `VERIFICATION.md` it is a more surprising claim than any of the clean hits, because it means
  the readable signal is the *direction* the weights moved, not the model's competence.

Report as delivered, verbatim:

---

## 1. Property
**Molecular dipole moment |μ| (Debye)** — the QM9 `mu` target, learned on a few-hundred-molecule CHNOF (QM9-style) subsample.

## 2. Confidence
**70%**

## 3. Runners-up
1. HOMO energy (ε_HOMO) — ~15%
2. A polarity/polar-surface descriptor (TPSA / H-bond-acceptor count) — ~7%
3. Aqueous solubility / hydrophilicity (−logP-like) — ~5%

## 4. Reasoning (key weight evidence)

**Dataset fingerprint (decisive).** Untouched columns must stay inside PyTorch's init bound (1/√86 = 0.1078 for W_i, 1/√136 = 0.0858 for W_o). Columns for P, S, Cl, Br, I and every metal max out at 0.1051–0.1076 / 0.0836–0.0857 — *exactly* at init, i.e. those elements never occurred. Only C, N, O, F moved (max 0.17–0.25 / 0.19–0.34). Also at init: all stereo flags except stereo=0, all chiral tags except CHI_UNSPECIFIED, charges ±2, degree 0 and 5. `total_num_Hs=4` is untouched in W_i (0.1066) but strongly trained in W_o (0.2931) — an atom with 4 H and **zero bonds**: methane, as a bond-free graph. So: CHNOF, no stereochemistry, almost no aromatics (aromatic flag barely moves, 0.1266), includes CH₄ ⇒ QM9 subsample.

**Not an extensive property.** The bias path u·b_o/100 = +0.0011 per atom (u = W2·W1), and no atom feature exceeds 0.078 in per-atom sensitivity. Signs are mixed by element (O +0.064, N +0.032, C −0.031, F −0.078), so U0/H/G/α/Cv/ZPVE/R² (which need ~0.3–0.5 σ per atom, same sign for all atoms) are excluded. Only μ, HOMO, LUMO, gap remain.

**Charge separation is the top signal.** formal_charge=+1 and −1 are the two most-trained features of all 72 (column norms 3.5× and 3.4× init) and co-occur (cosine 0.78 in W_i, 0.93 in W_o) — the nitro/N-oxide/azide motif. Both push y strongly *up* (+0.055, +0.059). Formal charge separation *is* a dipole. This kills LUMO and gap (nitro lowers both sharply) and strains HOMO.

**Per-motif direct sensitivities (u·W_o[:, :72]/100) read as a bond-dipole table:**
NO₂ +0.54 (2×O⁻ +0.200, N⁺ +0.140) > C=O +0.134 > C≡N +0.094 > −O− +0.050 > −OH +0.028 ≈ −NH₂ +0.024 > 0 > C–F −0.029 > CH₂ −0.082 > CH₃ −0.104 > CH₄ −0.134 (lowest). Ranking matches experimental μ (nitromethane 3.5, acetonitrile 3.9, acetone 2.9, DME 1.3, methane 0.0), and methane — the canonical μ = 0 molecule — gets the largest negative. For HOMO the ranking is wrong: amines are QM9's *highest* HOMO yet score +0.024, while nitriles (−9 eV, near the bottom) score +0.094 and nitro (−7.8 eV) tops the list. Terminal atoms (+0.045 for degree 1) vs internal (−0.017 for degree 2) is dipole-cancellation logic, not orbital logic.

**Main anomaly:** F is the single most negative feature (−0.078); CH₃F has μ = 1.85 D. Explicable by polyfluorination (CHF₃ 1.65, CF₄ 0.0) — F usually arrives in multiples — but it is the one fact that also fits HOMO/solubility, hence 70% not 90%.

## 5. Audit
Files opened: only `architecture.md`, `feature_layout.md`, `model.npz` in the blind folder (plus `ls` of that folder). No web, no other files, no chemistry libraries, no forward passes.

8 Python runs (scripts in a scratch directory):
1. `r1.py` — tensor shapes and weight statistics; confirmed 22,593 params.
2. `r2.py` — direct atom-feature sensitivity u·W_o[:, :72]/100, sorted.
3. `r3.py` — path-decomposed linearization (direct + 1/2/3 message hops) and bond-feature sensitivity, with/without b1>0 gating.
4. `r4.py` — per-feature column norms of W_i/W_o vs init expectation (which features were trained).
5. `r5.py` — cosine similarity between feature columns as a co-occurrence proxy (±1 charge pairing).
6. `r6.py` — top readout units by |u|; their atom-feature detectors; u·b_o baseline; W1 spectrum.
7. `r7.py` — max|entry| per column vs exact init bound → definitive present/absent feature list; ‖W_h‖ = 3.60.
8. `r8.py` — aggregated per-motif direct sensitivities (nitro, carbonyl, nitrile, ether, hydroxyl, amine, C–F, CH₃, CH₄).
