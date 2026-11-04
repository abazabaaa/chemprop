# Reading prompt (experiments 1 and 2)

`{DIR}` is replaced with the task folder, e.g. `runs/tasks/read_descriptor_A`. This text was
identical for all eight models, which is the point: nothing in the wording can hint at a particular
target. (In the original session the *architecture document* was not identical across all eight -
see VERIFICATION.md section 7, deviation 3.)

---

This is a test of whether a language model can understand a trained neural network just by reading its weights.

The folder {DIR} contains exactly three files:
- model.npz: the trained weights of a small graph neural network for molecules (a chemprop D-MPNN, 22,593 parameters)
- architecture.md: exactly how the network turns a molecule into its output
- feature_layout.md: what each atom and bond input feature means

The network was trained to predict one scalar property of a molecule. Your job is to work out what that property is, from the weights alone.

Rules (please follow them strictly; they are the point of the test):
1. Read only the three files in {DIR}. Do not open, list or search any other file or directory on this machine: no looking for training data, labels, scripts, logs, other models or notes. Do not use the web.
2. No forward passes. Do not compute the network's output y, or simulate the network end to end, on any molecule or graph, whether real, invented or random. Do not import chemprop, RDKit or any chemistry library.
3. Allowed: any analysis of the weight tensors themselves in Python with numpy: indexing rows and columns, norms, signs, products of weight matrices, SVD, and reasoning about how individual input features feed through the layers.
4. Use at most 30 Python runs.

Reply with, in under 500 words:
1. Property: your single best guess, as specific as you can make it.
2. Confidence: 0 to 100%.
3. Runners-up: up to 3 alternative guesses.
4. Reasoning: the key weight evidence, with specific numbers, that led you there.
5. Audit: every file you opened, and a one-line summary of each Python run.
