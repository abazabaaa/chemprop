"""Hand-written weights for "number of freely rotatable single bonds". numpy only.

Written by an agent with no training, no gradients and no molecule data; see
../results/reports/write_rotatable_simple.md for its own account of the design.

Only the formatting of this file has been changed from what the agent produced (black and isort, to
satisfy this repository's lint CI). No value, name, comment or line of logic differs. The untouched
original is kept at ../results/reports/build_weights.verbatim.py.txt, and check 1 of
../src/check_handwritten.py runs both files and confirms they produce identical weights.
"""
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---- atom feature indices (72) ----
DEG0, DEG_OTHER = 38, 44  # total_degree one-hot 0..5 then other
FC0, FC_END = 45, 51  # formal_charge block (sums to 1) -> constant source
NH0, NH_OTHER = 56, 61  # total_num_Hs one-hot 0..4 then other
# ---- bond feature indices, offset by 72 inside concat(x_u, e_uv) ----
B = 72
SINGLE, TRIPLE, INRING = B + 1, B + 3, B + 6

# ---- unit assignments ----
T, Q1, Q2 = 0, 1, 2  # edge hidden units
P1, P2 = 0, 1  # atom readout units
Y = 0  # head unit

W_i = np.zeros((64, 86))
W_h = np.zeros((64, 64))
W_o = np.zeros((64, 136))
b_o = np.zeros(64)
W1 = np.zeros((64, 64))
b1 = np.zeros(64)
W2 = np.zeros((1, 64))
b2 = np.zeros(1)

DEG_VAL = [0, 1, 2, 3, 4, 5, 6]  # value for each total_degree slot ("other" -> 6)
NH_VAL = [0, 1, 2, 3, 4, 4]  # value for each total_num_Hs slot ("other" -> 4)


def add_heavy_degree(W, row, off, scale):
    """Add scale * (total_degree - total_num_Hs) of the atom whose block starts at `off`."""
    for k, d in enumerate(DEG_VAL):
        W[row, off + DEG0 + k] += scale * d
    for k, n in enumerate(NH_VAL):
        W[row, off + NH0 + k] -= scale * n


def add_const(W, row, c):
    """W_i has no bias: use the formal-charge one-hot block (exactly one slot is 1)."""
    W[row, FC0:FC_END] += c


# ===== 1. edge init: h0_uv = W_i @ concat(x_u, e_uv) =====
# T: is this bond a triple bond?  (W_h row T stays 0, so T is stable over both iterations)
W_i[T, TRIPLE] = 1.0

# Q1/Q2: q_uv = relu(Q1) - relu(Q2) = [single & !ring & heavy_deg(u)>=2 & u in no triple]
for row, const in ((Q1, -11.0), (Q2, -12.0)):
    W_i[row, SINGLE] = 10.0  # +10 only for single bonds
    W_i[row, INRING] = -20.0  # ring bonds never count
    W_i[row, TRIPLE] = -20.0  # u is in a triple bond: this very bond
    add_heavy_degree(W_i, row, 0, 1.0)
    add_const(W_i, row, const)  # -10 cancels the +10, then -1 / -2 for the clip pair

# ===== 2. message passing: m_uv = sum_{w in N(u), w != v} h_wu =====
# u is in a triple bond via one of its OTHER bonds -> disqualify.
W_h[Q1, T] = -20.0
W_h[Q2, T] = -20.0

# ===== 3. atom readout: a_v = relu(W_o @ concat(x_v, m_v) + b_o) =====
# P1 - P2 = 0.5 * n_v * [heavy_deg(v) >= 2 & v in no triple]
for row in (P1, P2):
    add_heavy_degree(W_o, row, 0, 2.0)  # 2 * heavy_deg(v)
    b_o[row] = -4.0  # ... - 2*2  -> gate opens at heavy_deg >= 2
    W_o[row, 72 + T] = -100.0  # any triple bond at v closes the gate
W_o[P1, 72 + Q1] = 0.5  # n_v = m_v[Q1] - m_v[Q2]
W_o[P1, 72 + Q2] = -0.5

# ===== 4/5. head: g = (sum_v a_v)/100 =====
W1[Y, P1] = 100.0
W1[Y, P2] = -100.0
W2[0, Y] = 1.0

np.savez(
    "weights.npz",
    **{
        "message_passing.W_i.weight": W_i,
        "message_passing.W_h.weight": W_h,
        "message_passing.W_o.weight": W_o,
        "message_passing.W_o.bias": b_o,
        "predictor.ffn.0.0.weight": W1,
        "predictor.ffn.0.0.bias": b1,
        "predictor.ffn.1.2.weight": W2,
        "predictor.ffn.1.2.bias": b2,
    },
)
print(
    "saved weights.npz; nonzeros:",
    int(sum(np.count_nonzero(a) for a in (W_i, W_h, W_o, b_o, W1, b1, W2, b2))),
)
