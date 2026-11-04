"""Build every dataset the experiments need, plus the feature-layout document given to agents.

    python -I experiments/weight-space-programming/src/make_datasets.py

Writes into runs/data/:
  descriptors.csv              molecule pool with all five structural descriptor targets
  rot_practice.csv             practice half for the weight-writing task (labels kept private)
  rot_test.csv                 held-out half for the weight-writing task
  stress.csv                   hand-picked adversarial molecules for the weight-writing task
  feature_layout.md            the atom/bond feature table handed to agents
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski, rdMolDescriptors

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))

import common

RDLogger.DisableLog("rdApp.*")

DESCRIPTORS = {
    "halogens": lambda m: sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in m.GetAtoms()),
    "nitrogens": lambda m: sum(a.GetAtomicNum() == 7 for a in m.GetAtoms()),
    "aromatic_atoms": lambda m: sum(a.GetIsAromatic() for a in m.GetAtoms()),
    "hbond_donors": lambda m: Lipinski.NumHDonors(m),
    "rotatable_bonds": lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
}

# Molecules chosen to break a hand-written rotatable-bond counter: hypervalent and high-degree
# atoms, charges and zwitterions, elements absent from the training pool, triple bonds in several
# positions, fused/bridged/macrocyclic rings, long chains, biaryls and a few degenerate graphs.
STRESS_SMILES = [
    "FS(F)(F)(F)(F)F", "O=S(=O)(c1ccccc1)N1CCOCC1", "O=P(O)(O)OCC", "F[P-](F)(F)(F)(F)F",
    "CS(=O)(=O)CC", "O=[N+]([O-])c1ccccc1", "C[N+](C)(C)CCO",
    "[NH3+]CC(=O)[O-]", "CC(=O)[O-]", "c1cc[nH+]cc1CC",
    "C[Si](C)(C)CC", "CC[Se]CC", "B(O)(O)c1ccccc1", "CC[Sn](CC)(CC)CC", "[Fe]",
    "CC#CCC", "N#CCCC#N", "CC#N", "C#Cc1ccc(CC)cc1", "CCC#CC#CCC",
    "C1CCCCCCCCCCC1", "C1CC2CCC1CC2", "c1ccc2ccccc2c1", "C1CC1CC1CC1", "O1CCOCC1",
    "C1CCC(CC2CCCCC2)CC1",
    "CCCCCCCCCCCCCCCCCCCC", "CC(C)(C)CC(C)(C)CC(C)(C)C",
    "c1ccc(-c2ccccc2)cc1", "c1ccc(-c2ccncc2)cc1", "c1cc(-c2cc(-c3ccccc3)ccc2)ccc1",
    "O=C=O", "N", "C", "CC", "C=C=C", "CN=[N+]=[N-]", "FC(F)(F)S(=O)(=O)O",
]


def feature_layout() -> str:
    """Reconstruct the human-readable meaning of each input feature from the featurizer itself."""
    from chemprop import featurizers
    from rdkit.Chem import rdchem

    af = featurizers.MultiHotAtomFeaturizer.v2()
    pt = Chem.GetPeriodicTable()
    names = [f"atomic_num={pt.GetElementSymbol(z)}" for z in list(range(1, 37)) + [53]]
    names.append("atomic_num=other")
    names += [f"total_degree={d}" for d in range(6)] + ["total_degree=other"]
    names += [f"formal_charge={c:+d}" for c in (-1, -2, 1, 2, 0)] + ["formal_charge=other"]
    names += [f"chiral_tag={rdchem.ChiralType.values[c]}" for c in range(4)] + ["chiral_tag=other"]
    names += [f"total_num_Hs={h}" for h in range(5)] + ["total_num_Hs=other"]
    names += [f"hybridization={h}" for h in ("S", "SP", "SP2", "SP2D", "SP3", "SP3D", "SP3D2")]
    names.append("hybridization=other")
    names += ["is_aromatic (0/1)", "mass*0.01 (continuous)"]
    assert len(names) == len(af), (len(names), len(af))

    bond = ["bond_is_null", "bond_type=SINGLE", "bond_type=DOUBLE", "bond_type=TRIPLE",
            "bond_type=AROMATIC", "is_conjugated", "is_in_ring"]
    bond += [f"stereo={i}" for i in range(6)] + ["stereo=other"]
    assert len(bond) == len(featurizers.MultiHotBondFeaturizer()), len(bond)

    lines = [f"# Atom features (d_v = {len(names)})", ""]
    lines += [f"{i}\t{n}" for i, n in enumerate(names)]
    lines += ["", f"# Bond features (d_e = {len(bond)})", ""]
    lines += [f"{i}\t{n}" for i, n in enumerate(bond)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.DEFAULT_RUN_DIR))
    args = ap.parse_args()
    out = Path(args.out) / "data"
    out.mkdir(parents=True, exist_ok=True)

    smiles = common.molecule_pool()
    mols = {s: Chem.MolFromSmiles(s) for s in smiles}
    print(f"molecule pool: {len(smiles)}")

    desc = pd.DataFrame({"smiles": smiles,
                         **{k: [f(mols[s]) for s in smiles] for k, f in DESCRIPTORS.items()}})
    desc.to_csv(out / "descriptors.csv", index=False)
    print(desc.drop(columns="smiles").describe().T[["mean", "std", "min", "max"]].round(2))

    # Weight-writing task: same pool, shuffled with its own seed, split in half. y is the simple
    # rule the agent is given; y_strict is RDKit's stricter definition, kept for context only.
    simple = Chem.MolFromSmarts(common.SIMPLE_ROTATABLE_SMARTS)
    rot = pd.DataFrame({
        "smiles": smiles,
        "y": [len(mols[s].GetSubstructMatches(simple)) for s in smiles],
        "y_strict": [rdMolDescriptors.CalcNumRotatableBonds(mols[s]) for s in smiles],
    }).sample(frac=1, random_state=7).reset_index(drop=True)
    half = len(rot) // 2
    rot[:half].to_csv(out / "rot_practice.csv", index=False)
    rot[half:].to_csv(out / "rot_test.csv", index=False)
    agree = float((rot.y == rot.y_strict).mean())
    print(f"rotatable (simple rule): mean {rot.y.mean():.2f} max {rot.y.max()} | "
          f"practice {half} test {len(rot) - half} | agrees with RDKit strict {agree * 100:.1f}%")

    rows = []
    for smi in STRESS_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise SystemExit(f"stress molecule failed to parse: {smi}")
        rows.append((Chem.MolToSmiles(mol), len(mol.GetSubstructMatches(simple)), smi))
    pd.DataFrame(rows, columns=["smiles", "y", "input_smiles"]).to_csv(out / "stress.csv", index=False)
    print(f"stress set: {len(rows)}")

    (out / "feature_layout.md").write_text(feature_layout())
    common.write_json({"pool_size": len(smiles), "rot_practice": half,
                       "rot_test": len(rot) - half, "stress": len(rows),
                       "simple_vs_strict_agreement": agree},
                      Path(args.out) / "data_summary.json")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
