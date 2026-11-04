"""Shared pieces for the weight-space-programming experiments.

Everything here is deliberately small and explicit: the point of this directory is that another
person (or agent) can re-run it and get the same numbers, so there is no hidden state and no
cleverness. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_DIR.parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"
DEFAULT_RUN_DIR = EXP_DIR / "runs"

# The architecture every model in these experiments uses, trained and hand-written alike.
D_H = 64
DEPTH = 3
AGG_NORM = 100.0

# The eight tensors a model is made of. Anything else in a chemprop checkpoint (target scalers,
# loss-function buffers, metric weights) is deliberately NOT exported: an agent reading the weights
# must not be handed the target's mean and standard deviation.
PARAM_SHAPES = {
    "message_passing.W_i.weight": (D_H, 86),
    "message_passing.W_h.weight": (D_H, D_H),
    "message_passing.W_o.weight": (D_H, 2 * D_H + 8),
    "message_passing.W_o.bias": (D_H,),
    "predictor.ffn.0.0.weight": (D_H, D_H),
    "predictor.ffn.0.0.bias": (D_H,),
    "predictor.ffn.1.2.weight": (1, D_H),
    "predictor.ffn.1.2.bias": (1,),
}

# Descriptor targets (experiment 1). The letter -> target mapping is scrambled on purpose so the
# code letter carries no hint; keep this order if you want the published results to line up.
DESCRIPTOR_ORDER = ["rotatable_bonds", "halogens", "hbond_donors", "aromatic_atoms", "nitrogens"]
DESCRIPTOR_CODES = dict(zip("ABCDE", DESCRIPTOR_ORDER))

# Real measured / computed endpoints (experiment 2), all from files already in tests/data.
ENDPOINTS = {
    "P": ("regression.csv", "logSolubility", "measured log aqueous solubility (ESOL / Delaney)"),
    "Q": ("regression/mol_multitask.csv", "gap", "QM9 HOMO-LUMO gap (Hartree)"),
    "R": ("regression/mol_multitask.csv", "mu", "QM9 dipole moment (Debye)"),
}

# The simplified rotatable-bond rule used for the weight-writing task (experiment 3). A bond counts
# when it is a non-ring single bond and each end atom has >= 2 heavy neighbours and is not part of
# any triple bond. See specs/target_rotatable_simple.md.
SIMPLE_ROTATABLE_SMARTS = "[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]"

MAX_HEAVY_ATOMS = 60


def build_model(weights: dict[str, np.ndarray] | None = None):
    """Create the MPNN used everywhere, optionally loading hand-written or exported weights."""
    from chemprop import models, nn

    mp = nn.BondMessagePassing(d_h=D_H, depth=DEPTH)
    ffn = nn.RegressionFFN(input_dim=D_H, hidden_dim=D_H, n_layers=1)
    model = models.MPNN(mp, nn.NormAggregation(norm=AGG_NORM), ffn, batch_norm=False)
    if weights is not None:
        sd = model.state_dict()
        for k, shape in PARAM_SHAPES.items():
            if k not in weights:
                raise SystemExit(f"missing parameter {k}")
            if tuple(np.shape(weights[k])) != shape:
                raise SystemExit(f"{k} has shape {np.shape(weights[k])}, expected {shape}")
            sd[k] = torch.from_numpy(np.asarray(weights[k], dtype=np.float32))
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    out = {k: d[k] for k in d.files if k in PARAM_SHAPES}
    bad = [k for k, v in out.items() if not np.isfinite(v).all()]
    if bad:
        raise SystemExit(f"non-finite values in {bad}")
    return out


def export_weights(model, path: str | Path) -> dict[str, np.ndarray]:
    """Save only the eight learnable tensors, so no target statistics leak to a reader."""
    sd = model.state_dict()
    out = {k: sd[k].detach().cpu().numpy() for k in PARAM_SHAPES}
    np.savez(path, **out)
    return out


def predict(model, smiles, batch_size: int = 256) -> np.ndarray:
    from chemprop import data, featurizers

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s, [0.0]) for s in smiles], feat)
    preds = []
    with torch.no_grad():
        for batch in data.build_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            preds.append(model(batch.bmg))
    return torch.cat(preds).numpy().ravel().astype(float)


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    denom = ((truth - truth.mean()) ** 2).sum()
    return {
        "n": int(len(truth)),
        "mae": float(np.abs(pred - truth).mean()),
        "rmse": float(np.sqrt(((pred - truth) ** 2).mean())),
        "r2": float(1 - ((pred - truth) ** 2).sum() / denom) if denom > 0 else float("nan"),
        "exact_round": float((np.round(pred) == truth).mean()),
        "within_1": float((np.abs(pred - truth) <= 1).mean()),
        "pred_mean": float(pred.mean()),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
    }


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def molecule_pool() -> list[str]:
    """Every unique, single-component, <=60 heavy atom molecule in tests/data, canonicalised.

    Pooling from the repository's own test fixtures keeps the experiment self-contained: no
    downloads, no network. The set is a grab bag of small organics, drug-like molecules and a few
    stray single atoms, which is fine for these experiments and is described honestly in the README.
    """
    import pandas as pd
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    smis: set[str] = set()
    for path in sorted(TEST_DATA.glob("**/*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for col in df.columns:
            if "smiles" in col.lower():
                smis.update(df[col].dropna().astype(str))
    pool: dict[str, None] = {}
    for smi in sorted(smis):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or "." in smi or mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
            continue
        pool[Chem.MolToSmiles(mol)] = None
    return sorted(pool)
