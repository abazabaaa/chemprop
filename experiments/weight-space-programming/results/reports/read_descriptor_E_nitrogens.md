# Reading task, model E

- **True target:** `nitrogens` (count of nitrogen atoms)
- **Agent's answer:** the number of nitrogen atoms
- **Verdict:** CORRECT, including the count-versus-fraction distinction, which it tested explicitly.
- **Note on rule compliance:** this agent pushed hand-built atom and edge feature vectors through
  individual layers (runs 7-8) and checked which head units were active for linearised per-atom
  totals (run 13). The prompt's rule forbade computing the network's output end to end on a
  molecule, which it did not do, but this is close to the line and a stricter harness should define
  the boundary mechanically rather than in prose.

Report as delivered, verbatim:

---

**1. Property:** The number of nitrogen atoms in the molecule. Every N counts the same whatever its type; the model outputs it as a z-score.

**2. Confidence:** About 65%. I'm about 90% sure the target is "nitrogen content"; the remaining doubt is a count versus an N fraction.

**3. Runners-up:**
- Fraction of heavy atoms that are nitrogen.
- Nitrogen mass percent.
- Count of basic or H-bond-accepting nitrogens.

**4. Reasoning (all from weights):**
- **The N input dominates.** Its column has norm 1.96 in W_o and 1.48 in W_i. Every other atom or bond feature column is 0.4–0.9. The head's linear readout direction r = W1ᵀW2 correlates 0.87 with W_o[:,N]. About 26 readout units are N detectors: weight +0.30 to +0.37 for N, about −0.1 for C/O/halogens, and positive readout.
- **Every kind of nitrogen scores the same.** Typical atoms pushed through W_o alone:
  - every N type (pyridine, pyrrole, primary/secondary/tertiary amine, amide, aniline, nitrile, nitro⁺, ammonium⁺) scores +6.0 to +8.0;
  - every C, O, F, Cl, Br, I, S and P type scores −0.8 to +0.9.

  One-hop messages from N edges score +12 to +18, against −0.2 to +2.0 from non-N edges.
- **Ruling out related properties:**
  - No penalty for conjugation or aromaticity, so not pKa or basicity.
  - O behaves like C, so not TPSA, H-bond acceptors or heteroatom count.
  - No mass trend, and the mass feature is unused, so not molecular weight or N mass percent.
- **Nitrogen is amplified through message passing.** W_h's top singular value is 2.44 (the rest are about 1.1), and that direction aligns with N-edge states (cos 0.42). N edges carry 2–3× the activation of C edges.
- **The head rises steadily with N and never levels off:**
  - 22 units have W2 > 0, bias ≈ −0.03 and N weight ≈ +0.8;
  - 23 units have W2 < 0, bias ≈ +0.2 and N weight ≈ −0.4, holding a −0.7 offset that nitrogen switches off.
- **Count vs fraction.** The two readout units with the largest weights (31 and 24; r = −1.62 and −1.11) count non-N atoms, which could mean a dilution effect. My test: under either hypothesis, N-free molecules must give the same y at any size. I scaled the uncertain multi-hop terms until that held (by about 1/20). With that calibration, adding non-N atoms in nitrogen-containing molecules barely moves y: +0.003 of the effect of adding an N, at both 2 N with 20 other atoms and 5 N with 12. A fraction target would need about −0.09 and −0.29. At every scaling I tried, the non-N effect hardly changes when N is present. So y adds up per atom (about 0.33–0.36 standard units per N, only 9% flattening at high N), which fits a count.

**5. Audit.**

Files opened: the blind folder's `architecture.md`, `feature_layout.md`, `model.npz`. Nothing else, and no web.

Python runs (13, numpy only, all on weight tensors). I never computed y. Runs 7–8 passed prototypical atom and edge feature vectors through single layers. Run 13 checked which head units are active for linearized per-atom totals and reported only slopes.
1. Shapes and statistics of all tensors; printed the biases and W2.
2. Head readout directions under different on/off masks; W1 row sums against W2.
3. Each atom and bond feature projected onto the readout, directly and through one message hop.
4. Readout units with the largest and smallest N weights; correlation of the N column with r.
5. Each head unit's response to the N, C, O and bias directions; the two families.
6. Readout split by family; projections of atom-type columns.
7. Typical atom types through W_o alone, projected onto the family directions.
8. Typical edges through W_i and the one- and two-hop paths; W_h spectrum.
9. Feature weights and message inputs of the top readout units (31, 38, 24, …).
10. Linearized per-atom totals (direct plus one and two hops) per family.
11. The same, masked to active units; per-unit switch-on/off thresholds.
12. Message readout split into uniform activation and residual; three-hop term; W_h singular-vector alignment.
13. Calibrated the multi-hop scale on N-free size-independence; compared non-N and N slopes across regimes.
