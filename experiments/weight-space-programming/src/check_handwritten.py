"""Verify the hand-written network from experiments/weight-space-programming/handwritten/.

    python -I experiments/weight-space-programming/src/check_handwritten.py

Six checks, strongest last:

  1. the committed build_weights.py reproduces the committed weights.npz exactly
  2. only a handful of weights are non-zero, and they are round hand-chosen values
  3. the held-out half of the molecule pool (never seen by the writing agent)
  4. the hand-picked adversarial molecule set
  5. the explicit-hydrogen failure mode the writing agent predicted for itself
  6. randomly generated graphs, compared against a direct implementation of the rule. This is the
     real test of "exact by construction": it leaves the space of chemically reachable molecules
     entirely, so passing it cannot be explained by the test molecules resembling the training pool.

Exit status is non-zero if any check that should pass does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402 # needs the path set above

# Atom feature block offsets, from specs/feature_layout.md.
A_CARBON = 5  # atomic_num=C (the list is Z=1..36 then 53, so carbon sits at index 5)
A_DEG0, A_DEG_OTHER = 38, 44
A_CHARGE_NEUTRAL = 49  # formal_charge block order is -1, -2, +1, +2, 0, other
A_CHIRAL_NONE = 51
A_NH0, A_NH_OTHER = 56, 61
A_HYB_SP3 = 66
A_AROMATIC, A_MASS = 70, 71
# Bond feature indices.
B_NULL, B_SINGLE, B_DOUBLE, B_TRIPLE, B_AROMATIC = 0, 1, 2, 3, 4
B_CONJ, B_INRING, B_STEREO0 = 5, 6, 7


def check_build_reproduces(hand_dir: Path) -> bool:
    """Run both copies of the generating program and compare against the committed weights.

    `handwritten/build_weights.py` is the agent's program with black/isort formatting applied so it
    passes this repository's lint CI; `results/reports/build_weights.verbatim.py.txt` is exactly
    what the agent wrote. Both must produce the committed weights, which is what makes the
    reformatting safe to have done.
    """
    committed = common.load_weights(hand_dir / "weights.npz")
    sources = {
        "formatted": hand_dir / "build_weights.py",
        "verbatim": common.EXP_DIR / "results" / "reports" / "build_weights.verbatim.py.txt",
    }
    ok = True
    for label, src in sources.items():
        if not src.exists():
            print(f"  {label}: missing ({src})")
            ok = False
            continue
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "build_weights.py"
            script.write_text(src.read_text())
            proc = subprocess.run(
                [sys.executable, "-I", str(script)], capture_output=True, text=True, check=False
            )
            if proc.returncode != 0:
                print(f"  {label}: failed to run: {proc.stderr[-400:]}")
                ok = False
                continue
            rebuilt = common.load_weights(Path(tmp) / "weights.npz")
        worst = max(float(np.abs(committed[k] - rebuilt[k]).max()) for k in committed)
        print(f"  {label}: largest difference from the committed weights {worst:g}")
        ok = ok and worst == 0.0
    return ok


def check_sparsity(hand_dir: Path) -> bool:
    w = common.load_weights(hand_dir / "weights.npz")
    nonzero = {k: int((v != 0).sum()) for k, v in w.items()}
    total = sum(v.size for v in w.values())
    values = sorted({float(abs(x)) for v in w.values() for x in np.unique(v[v != 0])})
    print(f"  {sum(nonzero.values())} non-zero of {total} parameters")
    print(f"  distinct magnitudes: {values}")
    return sum(nonzero.values()) < 200 and all(float(v).is_integer() or v == 0.5 for v in values)


def _score(weights, csv: Path, label_col: str = "y") -> dict:
    df = pd.read_csv(csv)
    model = common.build_model(weights)
    return common.metrics(common.predict(model, df.smiles.tolist()), df[label_col].to_numpy(float))


def check_explicit_hydrogens(weights) -> bool:
    """With hydrogens promoted to graph nodes the design over-counts; chemprop's default is off.

    This is not a bug that was found after the fact: the writing agent stated this limitation in
    its own report, which is worth more than the limitation costs.
    """
    from rdkit import Chem, RDLogger

    from chemprop import data, featurizers

    RDLogger.DisableLog("rdApp.*")
    simple = Chem.MolFromSmarts(common.SIMPLE_ROTATABLE_SMARTS)
    smis = ["CCCC", "CCOc1ccccc1", "CC(=O)Nc1ccccc1", "C1CCCCC1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"]
    truth = [len(Chem.MolFromSmiles(s).GetSubstructMatches(simple)) for s in smis]
    model = common.build_model(weights)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    out = {}
    for add_h in (False, True):
        ds = data.MoleculeDataset(
            [data.MoleculeDatapoint.from_smi(s, [0.0], add_h=add_h) for s in smis], feat
        )
        with torch.no_grad():
            pred = (
                torch.cat(
                    [
                        model(b.bmg)
                        for b in data.build_dataloader(
                            ds, batch_size=32, shuffle=False, num_workers=0
                        )
                    ]
                )
                .numpy()
                .ravel()
            )
        out[add_h] = np.round(pred, 3)
        print(f"  add_h={add_h}: {out[add_h]}  truth {truth}")
    return np.allclose(out[False], truth) and not np.allclose(out[True], truth)


def _random_graph(rng, n_atoms: int, consistent: bool):
    """A synthetic molecular graph: arbitrary topology, arbitrary bond types, carbon everywhere.

    consistent=True keeps the declared total_degree / total_num_Hs features in agreement with the
    actual adjacency, as a real featurizer would. consistent=False lets them disagree, which probes
    whether the network really reads the declared features the way the specification says.
    """
    from chemprop.data.molgraph import MolGraph

    # Random spanning tree plus a few extra edges, so the graph is connected.
    edges = [(i, int(rng.integers(0, i))) for i in range(1, n_atoms)]
    for _ in range(int(rng.integers(0, max(1, n_atoms // 2)))):
        u, v = int(rng.integers(0, n_atoms)), int(rng.integers(0, n_atoms))
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.append((u, v))

    bond_kind = rng.choice(
        [B_SINGLE, B_DOUBLE, B_TRIPLE, B_AROMATIC], size=len(edges), p=[0.6, 0.15, 0.1, 0.15]
    )
    in_ring = rng.random(len(edges)) < 0.25

    heavy_degree = np.zeros(n_atoms, int)
    for u, v in edges:
        heavy_degree[u] += 1
        heavy_degree[v] += 1

    V = np.zeros((n_atoms, 72), float)
    declared_heavy = np.zeros(n_atoms, int)
    for a in range(n_atoms):
        V[a, A_CARBON] = V[a, A_CHARGE_NEUTRAL] = V[a, A_CHIRAL_NONE] = V[a, A_HYB_SP3] = 1
        V[a, A_MASS] = 0.1201
        n_h = int(rng.integers(0, 5))
        if consistent:
            deg = heavy_degree[a] + n_h
            declared_heavy[a] = heavy_degree[a]
        else:
            deg = int(rng.integers(0, 7))
            declared_heavy[a] = max(0, min(deg, 6) - min(n_h, 4))
        V[a, A_DEG0 + deg if deg < 6 else A_DEG_OTHER] = 1
        V[a, A_NH0 + n_h if n_h < 5 else A_NH_OTHER] = 1

    directed, E, rev = [], [], []
    for i, ((u, v), kind, ring) in enumerate(zip(edges, bond_kind, in_ring)):
        for src, dst in ((u, v), (v, u)):
            e = np.zeros(14, float)
            e[kind] = 1
            e[B_INRING] = float(ring)
            e[B_STEREO0] = 1
            E.append(e)
            directed.append((src, dst))
        rev += [2 * i + 1, 2 * i]
    mg = MolGraph(V, np.array(E), np.array(directed).T.copy(), np.array(rev))

    # Reference implementation of the rule, straight from the specification.
    has_triple = np.zeros(n_atoms, bool)
    for (u, v), kind in zip(edges, bond_kind):
        if kind == B_TRIPLE:
            has_triple[u] = has_triple[v] = True
    truth = sum(
        1
        for (u, v), kind, ring in zip(edges, bond_kind, in_ring)
        if kind == B_SINGLE
        and not ring
        and declared_heavy[u] >= 2
        and declared_heavy[v] >= 2
        and not has_triple[u]
        and not has_triple[v]
    )
    return mg, float(truth)


def check_random_graphs(weights, n_graphs: int, seed: int) -> bool:
    from chemprop.data.collate import BatchMolGraph

    model = common.build_model(weights)
    rng = np.random.default_rng(seed)
    ok = True
    for consistent in (True, False):
        preds, truths = [], []
        for _ in range(n_graphs):
            mg, truth = _random_graph(rng, int(rng.integers(2, 25)), consistent)
            with torch.no_grad():
                preds.append(float(model(BatchMolGraph([mg])).item()))
            truths.append(truth)
        preds, truths = np.array(preds), np.array(truths)
        worst = float(np.abs(preds - truths).max())
        exact = float((np.round(preds) == truths).mean())
        label = (
            "features consistent with topology"
            if consistent
            else "features deliberately inconsistent"
        )
        print(
            f"  {n_graphs} random graphs, {label}: "
            f"exact {exact * 100:.1f}%, largest error {worst:.2e}, truth up to {int(truths.max())}"
        )
        ok = ok and exact == 1.0
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.DEFAULT_RUN_DIR))
    ap.add_argument("--random-graphs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, hand = Path(args.out), common.EXP_DIR / "handwritten"
    data_dir = out / "data"
    if not (data_dir / "rot_test.csv").exists():
        raise SystemExit("run make_datasets.py first")

    weights = common.load_weights(hand / "weights.npz")
    results, passed = {}, {}

    print("1. build_weights.py reproduces weights.npz")
    passed["rebuild"] = check_build_reproduces(hand)

    print("2. sparsity and hand-chosen values")
    passed["sparsity"] = check_sparsity(hand)

    print("3. held-out half of the molecule pool")
    results["heldout"] = _score(weights, data_dir / "rot_test.csv")
    print(f"  {results['heldout']}")
    passed["heldout"] = results["heldout"]["exact_round"] == 1.0

    print("4. adversarial molecules")
    results["stress"] = _score(weights, data_dir / "stress.csv")
    print(f"  {results['stress']}")
    passed["stress"] = results["stress"]["exact_round"] == 1.0

    print("5. context: the same network against RDKit's stricter rotatable-bond definition")
    results["vs_rdkit_strict"] = _score(weights, data_dir / "rot_test.csv", "y_strict")
    print(
        f"  exact {results['vs_rdkit_strict']['exact_round'] * 100:.1f}% "
        f"(this is the ceiling the simple rule allows, not an error in the network)"
    )

    print("6. explicit hydrogens as graph nodes (known limitation)")
    passed["explicit_h"] = check_explicit_hydrogens(weights)

    print("7. randomly generated graphs")
    passed["random_graphs"] = check_random_graphs(weights, args.random_graphs, args.seed)

    common.write_json({"metrics": results, "checks": passed}, out / "handwritten_checks.json")
    print("\n" + "\n".join(f"  {'PASS' if v else 'FAIL'}  {k}" for k, v in passed.items()))
    return 0 if all(passed.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
