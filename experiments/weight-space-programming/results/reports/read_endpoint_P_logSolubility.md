# Reading task, model P

- **True target:** `logSolubility` from `tests/data/regression.csv` (the ESOL / Delaney set)
- **Agent's answer:** log aqueous solubility, named as "ESOL/Delaney-style"
- **Verdict:** CORRECT, and it identified the dataset as well as the property.
- **Note:** it ruled out the nearest competing endpoint (FreeSolv hydration free energy) with a
  specific, checkable argument about the relative signs of O/N and the halogens.

Report as delivered, verbatim:

---

**1. Property**
Aqueous solubility on a log scale — log S (mol/L), i.e. an ESOL/Delaney-style "measured log water solubility". High y = polar/soluble, low y = greasy/insoluble.

**2. Confidence**
60% for log-solubility specifically; ~90% that the target lies on the lipophilicity axis with the *hydrophilic* end positive.

**3. Runners-up**
1. Inverse lipophilicity reported directly as −logP / −clogP (octanol–water).
2. Solubility in other units/conditions (log mg/L, logS at pH 7.4, AqSolDB-style).
3. Polar surface area / N+O polar-atom count (TPSA).

**4. Reasoning**
Linearising the head (J = W2·W1, /100 pooling) and composing with W_o (direct atom path) and with W_o[:,72:]·W_h^k·W_i (1–3 hop message path) gives a per-element influence on y:

O +0.064, N +0.056, F −0.002, C −0.010, Cl −0.020, S −0.038, Br −0.061, P −0.069, I −0.092 (means over 600 random ReLU-mask draws; sign probability 1.00/0.00 for every element, pairwise orderings O>I, I<Br<Cl, P<Cl all at p=1.00). So the ordering is completely mask-robust, not an artefact of my linearisation.

That scale is the *negative* of Crippen atomic logP almost exactly: Crippen gives I 1.10 > P 0.86 > Br 0.84 > S 0.62 ≈ Cl 0.63 > F 0.42 > C 0.15 > O −0.3 > N −0.6 — the same 9-element rank order, reversed (corr = **−0.96**; corr with my from-memory logS substituent effects = **+0.94**; fitted R² 0.87–0.90). The graded I > Br > Cl ≫ F ordering is a size/polarizability (lipophilicity) signature, not an electronic one — electronegativity correlates only +0.57, polarizability −0.88 but with O/N on the wrong side, mass −0.70.

The dominant singular direction of the W_o atom block (s₁ = 3.08, 2.6× s₂) is literally that axis: −0.49·I + 0.46·O − 0.40·Br + 0.38·degree2 − 0.23·Cl + 0.19·N − 0.15·P.

Init-norm analysis (nn.Linear init limit 1/√86, 1/√136) shows which features were actually trained: C,N,O,F,P,S,Cl,Br,I only (all metals/Si/Se at ratio ≈1.0); degrees 1–4; charges 0/±1 (nitro groups: N⁺ −0.136, O⁻ −0.175, both negative); only CHI_UNSPECIFIED; all four bond types. A pesticide/pollutant-flavoured few-hundred-molecule organic set — consistent with ESOL.

Why not FreeSolv ΔG_hyd (the best size match at 642): ΔG_hyd requires O/N and halogens to share one sign (both make hydration more favourable, O ≈ −5.8, I ≈ −0.86 kcal/mol); here they are opposite in sign and |I| ≥ |O|, and corr is only −0.65. Why not TPSA: halogens would stay near init, but I/Br/Cl are the most heavily trained columns (W_o norm ratios 3.88/3.29/2.17) and are *graded*. Why not MR/density/refractive index/boiling point: all want I, Br strongly positive.

Secondary signals, consistent with ESOL's +0.74 aromatic-proportion and −0.0062 MW terms: aromatic bond +0.050, in-ring +0.058, conjugated +0.038, mass −0.052.

**5. Audit**
Files opened: only `architecture.md`, `feature_layout.md`, `model.npz` in the blind folder (plus my own scripts/intermediates written under a scratch directory). No forward passes, no chemistry libraries, no web.

8 Python runs (all `python3 -I`):
1. r1 — list tensors, shapes, moments, parameter count (22,593 confirmed).
2. r2 — head Jacobian variants; dy/dx per atom feature (direct path); top-25 table.
3. r3 — message-path sensitivities (sm·W_h^k·W_i) for neighbour-atom and bond features; crude archetype sums.
4. r4 — column-norm vs nn.Linear-init comparison to identify which atom/bond features were trained.
5. r5 — full element/feature influence table by hop + 400-draw ReLU-mask sign-robustness test.
6. r6 — correlation of element influence against 8 candidate chemical scales; least-squares fits; W_o SVD.
7. r7 — degree-weighted per-atom archetype influence (showed absolute per-atom baselines are unreliable; contrasts are not).
8. r8 — 600-draw random-mask stability of the element ranking and pairwise orderings.
