# Weight-writing task (experiment 3)

- **Target:** the simplified rotatable-bond rule in `specs/target_rotatable_simple.md`
- **Result:** exact. Every molecule in the held-out half, every adversarial molecule, and every
  randomly generated synthetic graph. See `src/check_handwritten.py` and `RESULTS.md`.
- **Budget used:** 1 of 20 forward passes.
- **Artefact:** `handwritten/build_weights.py` (78 lines) and `handwritten/weights.npz`.

The agent's own report is below, verbatim. Two things in it are worth checking yourself rather than
taking on trust, and both are checked by `src/check_handwritten.py`:

1. "exact by construction" - verified on 600 random synthetic graphs, including graphs whose
   declared degree/hydrogen features deliberately contradict their topology.
2. the explicit-hydrogen limitation it predicted for itself - confirmed; with `add_h=True` the
   network over-counts. chemprop's default is `add_h=False`.

---

Done. Final `weights.npz` is in the working folder (22,593 params, 74 of them nonzero).

## 1. Scores (last and only forward pass)
n=1673, **MAE 3.13e-07**, RMSE 5.56e-07, **R² 1.0000000**, **exact_round 1.000**, within_1 1.000, pred_mean 3.634, pred_min 0, pred_max 37. The residual is pure float32 rounding — the network is exact by construction.

## 2. Design
Program: count each qualifying bond as 0.5 at *each* of its two end atoms, so the atom-sum over the molecule equals the bond count. The per-bond predicate splits into a "far end + bond" part computed on the directed edge, and a "near end" part computed at the atom readout.

- **W_i (edge init, 3 units).** Unit `T` = `[bond_type==TRIPLE]`. Units `Q1`/`Q2` = `10·[single] − 20·[in_ring] − 20·[triple] − 11 (resp. −12) + L_u`, where `L_u = total_degree − total_num_Hs` is built as a linear functional of the two one-hot blocks (slot values 0..5 and "other"→6; H values 0..4, "other"→4). W_i has no bias, so the constants ride on the formal-charge one-hot block (always sums to 1).
- **W_h (2 nonzeros).** `W_h[Q1,T] = W_h[Q2,T] = −20`. Since `m_uv = Σ_{w∈N(u), w≠v} h_wu`, this adds the count of triple bonds at the source atom `u` *other than* the bond itself. Together with the `−20·[triple]` in W_i this is an exact "u participates in a triple bond" test. `W_h` row `T` is zero, so `T` is a fixed point and both message-passing iterations give the same answer (no 2-hop leakage).
- After relu, `q_uv = h[Q1] − h[Q2] = [single & !ring & L_u≥2 & u in no triple]` exactly — the `−11/−12` pair is a clip that turns `L_u` into the hard indicator `[L_u≥2]`, and the `10/−20` scales make every failing case fall below `−10` so both relus are 0.
- **W_o (atom readout, 2 units).** `m_v[T]` = number of triples at v; `n_v = m_v[Q1] − m_v[Q2]` = bonds at v whose far end qualifies. Then `P1 = relu(0.5·n_v + 2(L_v−2) − 100·m_v[T])`, `P2 = relu(2(L_v−2) − 100·m_v[T])`. `P1 − P2 = 0.5·n_v·[L_v≥2 & no triple at v]`: the slope-2 ramp is a shift-gate (for `L_v ≤ 1`, `n_v ≤ L_v`, so both relus are 0), and `−100·m_v[T]` closes the gate on triple-bond atoms.
- **Head.** `g = (Σ_v a_v)/100`, so `W1[0,P1]=100, W1[0,P2]=−100` gives `z0 = Σ_v (P1−P2) = y ≥ 0` (relu is a no-op), and `W2[0,0]=1`. All biases in the head are 0.

## 3. Known limits
Every clause of the target is realised exactly, so I expect no systematic errors. Residual risks, all rare: (a) atoms with `total_degree > 5` land in the "other" slot, which I score as 6 — fine for the `≥2` threshold, but a genuine degree-7 atom's `L` is under-reported (harmless, gate still open); (b) `total_num_Hs = "other"` (>4) is scored as 4, chemically unreachable alongside heavy neighbours; (c) float32 cancellation between the two large `P1`/`P2` sums — ~1e-5 absolute at 60 heavy atoms, far below 0.5; (d) the design assumes the stated one-hot invariants (exactly one slot set per block, `bond_is_null = 0` on every edge) and that no explicit H appears as a node.

## 4. Audit
- **Rule 1 (no training/fitting):** followed. Every number is a hand-derived integer or half-integer (1, ±0.5, ±2, ±4, ±10, ±11, ±12, ±20, ±100) chosen from the relu algebra above. No optimizer, no least squares, no search, no hill-climbing; the weights were written once and never adjusted.
- **Rule 2 (no molecule data, numpy only):** followed. No dataset opened or constructed; no RDKit/chemprop/any chemistry import. I hand-wrote feature vectors for ~30 molecules from `feature_layout.md` and simulated the architecture in numpy (`sim.py`, `sim2.py`, `sim3.py`) to check arithmetic before evaluating. Three apparent mismatches there turned out to be my own miscounts of the target, not network errors.
- **Rule 3 (stay in the working folder):** followed. I read only `architecture.md`, `feature_layout.md`, `target.md`, `build_weights.py`, `eval.py` plus files I created in that folder. I did not look at the private directory, any practice/test molecules, labels or scoring code, and used no web access.
- **Forward passes used: 1 / 20.** Pass 1 — the design as first written: MAE 3.1e-07, R² 1.0, 100% exact; nothing left to refine, so I stopped.
- Files written in the folder: `build_weights.py` (~95 lines, the generating program), `weights.npz`, `notes.md` (design notes), `sim.py`/`sim2.py`/`sim3.py` (local verification harness).
