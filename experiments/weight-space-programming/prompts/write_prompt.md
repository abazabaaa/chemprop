# Weight-writing prompt (experiment 3, third and successful version)

`{DIR}` is replaced with the task folder, e.g. `runs/tasks/write_rotatable`.

---

This is a test of whether a language model can write the weights of a neural network by hand,
reasoning about what each part of the network should compute, with no training and no gradients.

Your working folder is {DIR}. It contains:
- architecture.md: exactly how the network turns a molecule into its scalar output y, and the .npz keys/shapes you must save
- feature_layout.md: what every atom and bond input feature means
- target.md: the property y must predict, specified precisely, with worked examples
- build_weights.py: a starter script with all weights at zero. Edit it, run it, and it writes weights.npz
- eval.py: your forward-pass tool (see below)

Your task: set the numbers in that network so it computes the target as accurately as possible. The
network is a chemprop D-MPNN (22,593 parameters), but the target is simple and you will only need a
handful of hand-designed units; nearly all weights can stay zero. Think of it as a compilation
problem: work out what program over the molecular graph computes the target, then realise that
program in the layers you have been given. Note that the message passing steps are what let a bond
"see" how many neighbours each of its two end atoms has.

CRITICAL - how to work. Two earlier attempts at this task failed by generating one gigantic response
that exceeded the output limit. Avoid that:
- Keep EVERY response short. Never write a long derivation, a long table of numbers, or a long plan in a response.
- Work in many small steps: make one small edit, run one check, then stop and look at the result.
- Write build_weights.py as a short PROGRAM that constructs the weights: named feature-index constants, loops, slice assignments like `W_i[unit, FEAT_SINGLE] = 1.0`. Never emit thousands of literal numbers. Target at most ~150 lines.
- If you need to work something out at length, write brief notes to notes.md in your folder instead of putting it in a response.

Rules (please follow them strictly; they are the point of the test):
1. No training, no gradients, no fitting to data. Every number must come from your own reasoning. No optimizer, least-squares fit, random search or hill-climbing over parameters. Deliberate, reasoned changes to specific weights (including after a forward pass tells you something is off) are fine; automated search is not.
2. No molecule data. Do not look for, open or construct any dataset of molecules, and do not import RDKit, chemprop or any other chemistry library. numpy only.
3. Work only inside your working folder. Do not read any other directory on this machine (in particular, do not go looking for the practice or test molecules, their labels, or the scoring code). Do not use the web.
4. Forward passes: run `cd {DIR} && python -I eval.py weights.npz`. It runs your weights on a practice set of molecules you never see and prints aggregate metrics only (MAE, R², share of molecules where round(y) is exactly right). You have a budget of 20 such calls; a call that errors does not count. Use them to check and refine your design.
5. It is fine (and expected) to simulate your own design in numpy on atom/bond feature vectors you construct by hand, to check your arithmetic before spending a forward pass.

For reference, predicting a constant gets MAE about 2.5 and ~13% of molecules exactly right. A
gradient-trained network of this architecture gets MAE around 0.3.

Final scoring is on a held-out set of molecules you never see, using the same metrics. Leave your
final weights.npz in the working folder.

When you are done, reply with, in under 600 words:
1. Scores: the metrics from your last forward pass.
2. Design: what program over the molecule you implemented, and how each layer realises it.
3. Known limits: which molecules you expect it to get wrong, and why.
4. Audit: confirm you followed rules 1-3, how many forward passes you used, and a one-line note on each.
