# Architecture you are writing weights for

A chemprop directed-bond message passing network (D-MPNN) followed by a small feed-forward head.
Molecules are RDKit graphs with implicit hydrogens: H atoms are never nodes. Every chemical bond u-v
becomes two directed edges, u->v and v->u. Inputs are atom features x_v (72-dim) and bond features
e_uv (14-dim); feature_layout.md lists them. Notes on the features:
- total_degree counts all neighbours INCLUDING hydrogens; total_num_Hs is the number of attached H.
  So the heavy-atom degree is total_degree - total_num_Hs.
- Within each one-hot block (atomic_num, total_degree, formal_charge, chiral_tag, total_num_Hs,
  hybridization) exactly one slot is 1 for every atom (the "other" slot if the value is not listed).
  For every real bond, bond_is_null = 0, exactly one bond_type slot is 1 (single, double, triple or
  aromatic) and exactly one stereo slot is 1.
- is_aromatic, is_conjugated and is_in_ring are RDKit's flags. mass is atomic mass * 0.01.

Linear layers compute y = W @ x (+ b). Weight shapes are PyTorch (out, in). relu = max(0, .).
d_h = 64 everywhere.

1. Edge init:      h0_uv = W_i @ concat(x_u, e_uv)              W_i: (64, 86), NO bias
                   h_uv  = relu(h0_uv)
2. Repeat twice:   m_uv  = (sum over w in N(u) of h_wu) - h_vu  messages into u, minus the reverse edge
                   h_uv  = relu(h0_uv + W_h @ m_uv)             W_h: (64, 64), NO bias
3. Atom readout:   m_v   = sum over u in N(v) of h_uv
                   a_v   = relu(W_o @ concat(x_v, m_v) + b_o)   W_o: (64, 136), b_o: (64,)
                   An atom with no bonds has m_v = 0.
4. Molecule:       g     = (sum over atoms v of a_v) / 100
5. Head:           z     = relu(W1 @ g + b1)                    W1: (64, 64), b1: (64,)
                   y     = W2 @ z + b2                          W2: (1, 64),  b2: (1,)

There is no output scaling: y itself is the network's answer, and it is scored directly against the
target in target.md. Inference runs in float32. No dropout, no batch norm.

Save the weights as an .npz with exactly these keys and shapes:
  message_passing.W_i.weight  (64, 86)
  message_passing.W_h.weight  (64, 64)
  message_passing.W_o.weight  (64, 136)
  message_passing.W_o.bias    (64,)
  predictor.ffn.0.0.weight    (64, 64)    = W1
  predictor.ffn.0.0.bias      (64,)       = b1
  predictor.ffn.1.2.weight    (1, 64)     = W2
  predictor.ffn.1.2.bias      (1,)        = b2
