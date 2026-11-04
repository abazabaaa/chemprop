# Architecture

A directed-bond message passing neural network (D-MPNN) for molecules, followed by a feed-forward
head. Molecules are RDKit graphs with implicit hydrogens (H atoms are not nodes). Every chemical
bond u-v becomes two directed edges, u->v and v->u.

Inputs: atom features x_v (72-dim) and bond features e_uv (14-dim). The exact layout is in
feature_layout.md.

Notation: linear layers act as y = W @ x (+ b). In the .npz file, weights have PyTorch shape
(out, in). relu = max(0, .)

```
1. Edge init:        h0_uv = W_i @ concat(x_u, e_uv)                  (W_i: 64 x 86, no bias)
                     h_uv  = relu(h0_uv)
2. Repeat 2 times:   m_uv  = sum_{w in N(u)} h_wu  -  h_vu            (messages into u, minus the reverse edge)
                     h_uv  = relu(h0_uv + W_h @ m_uv)                 (W_h: 64 x 64, no bias)
3. Atom readout:     m_v   = sum_{u in N(v)} h_uv
                     a_v   = relu(W_o @ concat(x_v, m_v) + b_o)       (W_o: 64 x 136, b_o: 64)
4. Molecule vector:  g     = (sum_v a_v) / 100
5. Head:             z     = relu(W1 @ g + b1)                        (W1: 64 x 64, b1: 64)
                     y     = W2 @ z + b2                              (W2: 1 x 64, b2: 1)
```

y is a single scalar regression output in standardized units: the model was trained to predict
(target - mean) / std, and the mean and std have been removed. Loss: MSE. Optimizer: Adam.

Parameter names in the .npz file:

```
message_passing.W_i.weight   -> W_i
message_passing.W_h.weight   -> W_h
message_passing.W_o.weight   -> W_o
message_passing.W_o.bias     -> b_o
predictor.ffn.0.0.weight     -> W1
predictor.ffn.0.0.bias       -> b1
predictor.ffn.1.2.weight     -> W2
predictor.ffn.1.2.bias       -> b2
```

Total: 22,593 parameters.
