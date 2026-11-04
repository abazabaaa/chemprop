"""Score a weights file against a labelled CSV.

    python -I experiments/weight-space-programming/src/score.py WEIGHTS.npz LABELS.csv

LABELS.csv needs a `smiles` column and a `y` column. Prints JSON. This is the only place that
turns weights into numbers, so both the trained models and the hand-written network are measured
by exactly the same code path (the real chemprop model, not a reimplementation).
"""

from __future__ import annotations

import json
import sys

import pandas as pd

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))

import common


def run(weights_path: str, labels_csv: str) -> dict:
    df = pd.read_csv(labels_csv)
    model = common.build_model(common.load_weights(weights_path))
    pred = common.predict(model, df.smiles.tolist())
    return common.metrics(pred, df.y.to_numpy(float))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(run(sys.argv[1], sys.argv[2]), indent=2))
