"""Assemble the folders that are handed to agents.

    python -I experiments/weight-space-programming/src/make_tasks.py

Reading tasks (experiments 1 and 2): one folder per model, each containing exactly three files -
the weights, the architecture document and the feature layout. Nothing else. The folders are
byte-identical apart from model.npz, which is the point: any difference in an agent's answer has to
come from the weights.

Writing task (experiment 3): one folder with the architecture, the feature layout, the target
specification, a starter script and a budgeted forward-pass tool.

The folders are built under runs/tasks/. Hand an agent one folder path and the matching prompt
from prompts/. Keep the answer keys out of the folder - they live in runs/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))

import common

FORWARD_PASS_BUDGET = 20


def _write_eval_tool(folder: Path, labels_csv: Path, src_dir: Path) -> None:
    """A forward-pass tool that reports aggregate metrics only and counts its own uses.

    Aggregate-only is deliberate: the agent may learn how well its design does, but never sees a
    molecule, a per-molecule prediction or a label, so it cannot fit the weights to the data.
    """
    (folder / "eval.py").write_text(f'''"""Forward-pass tool: scores your weights.npz on molecules you never see.

    python -I eval.py weights.npz

Prints aggregate metrics only. Budget: {FORWARD_PASS_BUDGET} successful calls. A call that errors
does not count.
"""
import json
import os
import subprocess
import sys

SRC = {str(src_dir)!r}
LABELS = {str(labels_csv)!r}
BUDGET = {FORWARD_PASS_BUDGET}
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_passes_used.json")

weights = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "weights.npz")
used = json.load(open(LOG))["used"] if os.path.exists(LOG) else 0
if used >= BUDGET:
    raise SystemExit(f"budget exhausted: {{used}}/{{BUDGET}} forward passes used")

proc = subprocess.run([sys.executable, "-I", os.path.join(SRC, "score.py"), weights, LABELS],
                      capture_output=True, text=True, cwd=SRC)
if proc.returncode != 0:
    sys.stderr.write(proc.stderr[-2000:] + "\\n")
    raise SystemExit("evaluation failed; this call did NOT count against your budget")

used += 1
json.dump({{"used": used, "budget": BUDGET}}, open(LOG, "w"))
print(proc.stdout)
print(f"forward passes used: {{used}}/{{BUDGET}}")
''')


def _starter_script(folder: Path) -> None:
    (folder / "build_weights.py").write_text('''"""Fill in the weights by hand, then run this file to write weights.npz. numpy only.

Write this as a PROGRAM that constructs the weights - named feature-index constants, loops, slice
assignments - not as thousands of literal numbers. Most weights can stay zero.
"""
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

W_i = np.zeros((64, 86))
W_h = np.zeros((64, 64))
W_o = np.zeros((64, 136))
b_o = np.zeros(64)
W1 = np.zeros((64, 64))
b1 = np.zeros(64)
W2 = np.zeros((1, 64))
b2 = np.zeros(1)

# ---- your weights here ----

np.savez("weights.npz", **{
    "message_passing.W_i.weight": W_i, "message_passing.W_h.weight": W_h,
    "message_passing.W_o.weight": W_o, "message_passing.W_o.bias": b_o,
    "predictor.ffn.0.0.weight": W1, "predictor.ffn.0.0.bias": b1,
    "predictor.ffn.1.2.weight": W2, "predictor.ffn.1.2.bias": b2,
})
print("saved weights.npz")
''')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.DEFAULT_RUN_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    src = Path(__file__).resolve().parent
    specs = common.EXP_DIR / "specs"
    layout = out / "data" / "feature_layout.md"
    if not layout.exists():
        raise SystemExit("run make_datasets.py first")

    tasks = out / "tasks"
    n_read = 0
    for npz in sorted((out / "models").glob("*.npz")):
        if npz.stem.startswith("rot_baseline"):
            continue
        folder = tasks / f"read_{npz.stem}"
        folder.mkdir(parents=True, exist_ok=True)
        # Re-save rather than copy, to guarantee no extra arrays ride along.
        np.savez(folder / "model.npz", **common.load_weights(npz))
        shutil.copy(specs / "architecture_read.md", folder / "architecture.md")
        shutil.copy(layout, folder / "feature_layout.md")
        n_read += 1
    print(f"reading tasks: {n_read} folders under {tasks}")

    write = tasks / "write_rotatable"
    write.mkdir(parents=True, exist_ok=True)
    shutil.copy(specs / "architecture_write.md", write / "architecture.md")
    shutil.copy(specs / "target_rotatable_simple.md", write / "target.md")
    shutil.copy(layout, write / "feature_layout.md")
    _starter_script(write)
    _write_eval_tool(write, out / "data" / "rot_practice.csv", src)
    print(f"writing task: {write}")
    print("\nhand an agent one folder path plus the matching prompt from prompts/.")


if __name__ == "__main__":
    main()
