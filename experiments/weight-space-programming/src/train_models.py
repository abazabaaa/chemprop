"""Train every model the experiments need and export only their learnable weights.

    python -I experiments/weight-space-programming/src/train_models.py --what all

Three groups:
  descriptors  five identical networks on five structural descriptors (experiment 1)
  endpoints    three identical networks on real measured/computed properties (experiment 2)
  baseline     gradient-trained reference for the weight-writing task (experiment 3)

Every model uses the same architecture, the same seed and the same split recipe, so the only
difference between two models in a group is the target. Targets are standardised during training
and the standardisation is thrown away on export, so a reader of the weights cannot recover the
target's units, mean or spread. The baseline is the one exception: its output head is folded back
into raw units, because it has to be directly comparable with a hand-written network.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from lightning import pytorch as pl
import numpy as np
import pandas as pd
from rdkit import RDLogger
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402 # needs the path set above

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_EPOCHS, DESCRIPTOR_BATCH = 60, 64
ENDPOINT_EPOCHS, ENDPOINT_BATCH = 150, 50
BASELINE_EPOCHS, BASELINE_BATCH = 60, 64


def _fit(smiles, y, epochs: int, batch_size: int):
    """Train one model on an 85/15 split and return it plus the held-out R^2 and target scaler."""
    from chemprop import data, featurizers

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pl.seed_everything(0, workers=True)
    dps = [data.MoleculeDatapoint.from_smi(s, [float(v)]) for s, v in zip(smiles, y)]
    idx = np.random.default_rng(0).permutation(len(dps))
    n_tr = int(0.85 * len(idx))
    train = data.MoleculeDataset([dps[i] for i in idx[:n_tr]], feat)
    test = data.MoleculeDataset([dps[i] for i in idx[n_tr:]], feat)
    scaler = train.normalize_targets()
    test.normalize_targets(scaler)

    model = common.build_model()
    model.train()
    pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    ).fit(model, data.build_dataloader(train, batch_size=batch_size, num_workers=0, seed=0))
    model.eval()

    with torch.no_grad():
        pred = (
            torch.cat(
                [
                    model(b.bmg)
                    for b in data.build_dataloader(
                        test, batch_size=256, shuffle=False, num_workers=0
                    )
                ]
            )
            .numpy()
            .ravel()
        )
    truth = np.array([d.y[0] for d in test])
    r2 = float(1 - ((pred - truth) ** 2).sum() / ((truth - truth.mean()) ** 2).sum())
    return model, r2, scaler


def train_descriptors(out: Path) -> dict:
    df = pd.read_csv(out / "data" / "descriptors.csv")
    key = {}
    for code, target in common.DESCRIPTOR_CODES.items():
        model, r2, scaler = _fit(df.smiles, df[target], DESCRIPTOR_EPOCHS, DESCRIPTOR_BATCH)
        common.export_weights(model, out / "models" / f"descriptor_{code}.npz")
        key[code] = {
            "target": target,
            "heldout_r2_standardised": r2,
            "scaler_mean": float(scaler.mean_[0]),
            "scaler_std": float(scaler.scale_[0]),
        }
        print(f"  {code} {target:16s} R2={r2:.3f}", flush=True)
        common.write_json(key, out / "answer_key_descriptors.json")
    return key


def train_endpoints(out: Path) -> dict:
    key = {}
    for code, (csv, target, desc) in common.ENDPOINTS.items():
        df = pd.read_csv(common.TEST_DATA / csv)[["smiles", target]].dropna()
        model, r2, scaler = _fit(df.smiles, df[target], ENDPOINT_EPOCHS, ENDPOINT_BATCH)
        common.export_weights(model, out / "models" / f"endpoint_{code}.npz")
        key[code] = {
            "target": target,
            "description": desc,
            "source_csv": csv,
            "n": len(df),
            "heldout_r2_standardised": r2,
            "scaler_mean": float(scaler.mean_[0]),
            "scaler_std": float(scaler.scale_[0]),
        }
        print(f"  {code} {target:16s} n={len(df)} R2={r2:.3f}", flush=True)
        common.write_json(key, out / "answer_key_endpoints.json")
    return key


def train_baseline(out: Path) -> dict:
    """Gradient-trained counterpart to the hand-written network: same architecture, practice half
    only, output head de-standardised so it predicts raw counts like the hand-written one does."""
    df = pd.read_csv(out / "data" / "rot_practice.csv")
    model, r2, scaler = _fit(df.smiles, df.y, BASELINE_EPOCHS, BASELINE_BATCH)
    w = {
        k: v.detach().cpu().numpy()
        for k, v in model.state_dict().items()
        if k in common.PARAM_SHAPES
    }
    w["predictor.ffn.1.2.weight"] = w["predictor.ffn.1.2.weight"] * scaler.scale_[0]
    w["predictor.ffn.1.2.bias"] = w["predictor.ffn.1.2.bias"] * scaler.scale_[0] + scaler.mean_[0]
    np.savez(out / "models" / "rot_baseline_trained.npz", **w)
    print(f"  baseline (internal 85/15 R2={r2:.3f}); score it with score.py on rot_test.csv")
    return {
        "internal_heldout_r2_standardised": r2,
        "trained_on": "rot_practice.csv",
        "epochs": BASELINE_EPOCHS,
        "note": "output head folded back into raw counts",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.DEFAULT_RUN_DIR))
    ap.add_argument(
        "--what", default="all", choices=["all", "descriptors", "endpoints", "baseline"]
    )
    args = ap.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    if not (out / "data" / "descriptors.csv").exists():
        raise SystemExit("run make_datasets.py first")

    if args.what in ("all", "descriptors"):
        print("descriptor models (experiment 1):")
        train_descriptors(out)
    if args.what in ("all", "endpoints"):
        print("endpoint models (experiment 2):")
        train_endpoints(out)
    if args.what in ("all", "baseline"):
        print("gradient baseline (experiment 3):")
        common.write_json(train_baseline(out), out / "baseline_info.json")


if __name__ == "__main__":
    main()
