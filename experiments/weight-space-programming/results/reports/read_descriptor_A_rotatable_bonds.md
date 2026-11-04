# Reading task, model A

- **True target:** `rotatable_bonds` (RDKit `CalcNumRotatableBonds`)
- **Agent's answer:** melting point
- **Verdict:** WRONG. The true target did not appear among its alternatives.
- **Note:** the agent recovered the right features (acyclic single bonds raise the output, ring and
  triple bonds lower it) and then interpreted that pattern as rigidity raising a melting point,
  with the global sign inferred from chemistry rather than measured. It flagged that inference
  itself as its weakest step.

Report as delivered, verbatim:

---

## 1. Property
**Melting point** (experimental, °C/K) of the solid compound — a curated mixed set of small organics plus their salts (Bradley-style melting-point data).

## 2. Confidence
**65%**

## 3. Runners-up
1. Aqueous solubility (logS, AqSolDB-style, salts included) — 20%
2. Enthalpy of fusion / sublimation (lattice-energy-like) — 8%
3. Boiling point — 4%

## 4. Reasoning

**Head is rank-1 → additive atom sum.** W1 singular values 4.18, 1.43, 1.25…; |u₁·W2| = 1.007 of ‖W2‖ = 1.187 (85%). So y ≈ affine in one projection r = W1ᵀW2 of g = Σ a_v/100.

**Which features were trained.** Never-occurring columns (He, Li, Ne, Ar, Sc, Ti, Kr…) give an exact init baseline (‖col‖ = 0.403 in W_o vs theory 1/√136·√(64/3) = 0.396), so z-scores are reliable. Top: explicit H 30.9, hyb=S 26.2, fc=−1 26.0, degree=0 23.3, SP 22.5, S 18.6, fc=−2 16.2, Zn 15.0, N 14.9, Ca 12.6, Mg 12.1, P 12.0, Cl 11.5. **Carbon 2.9 and mass 1.2 are untrained** → rules out MW, molar refractivity, polarizability. Aromatic flag 4.2, conjugation 1.7, stereo ≈ 0 → rules out TPSA-style and logP-style additive schemes.

**A salt/counterion detector.** W_o columns: Mg·Ca cos 0.76, Mg·deg0 0.73, Ca·deg0 0.74, Ca·fc+2 0.65 — one coherent bare-ion unit. A disconnected ion has *no* incident edges, so its readout pre-activation is exactly W_o x + b_o; only 33–59% of units switch off (max surviving pre-activation +0.66…+0.98), so **counterions are not ignored, they dominate**: r-projections Mg²⁺ −5.32, Zn²⁺ −5.23, Ca²⁺ −4.99, H⁺ −3.43, I⁻ −2.89, Cl⁻ −2.54, Na⁺ −1.36, K⁺ −1.16 — **divalent ≈ 4× monovalent**, i.e. a lattice-energy ordering. Covalent atom types span only −1.1…+2.7.

**Ring membership dominates the bond path.** Propagated through (I + W_h + W_h²) (ρ(W_h) = 0.951): is_in_ring **−19.8**, triple −9.6, conj −4.5, vs single +5.6, double +5.1, aromatic +4.0. Any ring bond nets ≈ −14 to −20; an acyclic single bond +5.6.

**The decisive sign argument.** Rings and ionic counterions push the output the *same* way; flexible acyclic single bonds the opposite way. Melting point is the only candidate where these agree (rings/rigidity ↑MP, salts ↑↑MP, flexible chains ↓MP). Solubility needs them opposed (salts ↑logS, rings ↓logS); logP/logD likewise (rings ↑, ions ↓). Relative to a CH₂ baseline (−2.07), every heteroatom raises y: N +1.0…+3.2, S +3.1, O +0.4…+1.3, halogens +0.6…+1.6 — H-bonding/dipolar ↑MP. Methylene (deg4+H2) is the most MP-lowering atom type: the textbook flexible-chain rule.

Caveat: the global sign is inferred from chemical consistency, not measured (no forward passes were run).

## 5. Audit

Files opened: `architecture.md`, `feature_layout.md`, `model.npz` — only these three, all in the blind folder.

Python runs (10 of 30 used): 1 shapes/stats/biases; 2 head structure, r, SVD; 3 atom→W_o→r projections; 4 W_i atom+bond→q; 5 z-scores vs dead-feature baseline; 6 full z table by feature block; 7 cosine matrix of W_o columns; 8 SVD modes, mass correlation, per-atom-type column sums; 9 disconnected-ion pre-activation sign patterns; 10 propagated bond sensitivities and H-count increments.

No forward passes, no chemistry libraries, no other files, no web.
