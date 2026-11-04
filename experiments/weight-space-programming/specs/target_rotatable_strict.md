# Target: number of rotatable bonds (RDKit CalcNumRotatableBonds, default "strict" definition)

This is the harder target used in the two weight-writing attempts that failed (see
`prompts/write_prompt_failed_attempts.md`). It is kept here so the 2x2 experiment proposed in
`VERIFICATION.md` can be run against the same text that was actually used.

For each molecule, y should equal the number of bonds matched by this SMARTS (each bond counted
once):

```
[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]
```

In words. A bond A-B counts if it is a single (or aromatic) bond that is NOT in a ring, and for at
least one ordering (X, Y) of its two atoms, X satisfies P1 and Y satisfies P2.

P2 (an atom passes if all hold):
- it is not part of any triple bond;
- it has at least 2 heavy-atom neighbours (D in SMARTS = explicit neighbours = heavy-atom degree here);
- it is not an aliphatic carbon bearing three F, three Cl, three Br, or three CH3 groups
  (CF3, CCl3, CBr3, or a tert-butyl centre).

P1 = P2, and additionally the atom is not "amide-like", i.e. none of these is true:
- (a) it is an aliphatic carbon with exactly 3 heavy neighbours, double-bonded to N, O or S, and with a
      non-ring single bond to a nitrogen, an aliphatic oxygen, or an aliphatic sulfur that has >= 2 heavy
      neighbours;
- (b) it is a nitrogen, an aliphatic oxygen, or an aliphatic sulfur with >= 2 heavy neighbours, that has a
      non-ring single bond to an aliphatic carbon with exactly 3 heavy neighbours that is double-bonded to
      N, O or S;
- (c) like (a), but the carbon is double-bonded to a positively charged nitrogen [N+] and singly bonded
      (non-ring) to a nitrogen with >= 2 heavy neighbours;
- (d) like (b) for that case: a nitrogen with >= 2 heavy neighbours, singly bonded (non-ring) to an
      aliphatic carbon with 3 heavy neighbours that is double-bonded to [N+].

Consequence: a bond is excluded by the amide rules only when BOTH of its atoms are amide-like
(e.g. the C(=O)-N bond of an amide, or the C(=O)-O bond of an ester).
Lowercase/aromatic atoms are never "aliphatic"; #7 means any nitrogen, aromatic or not.

Worked examples (RDKit's answers):

```
  ethane              CC                           0
  butane              CCCC                         1
  toluene             Cc1ccccc1                    0   (methyl is terminal)
  biphenyl            c1ccc(-c2ccccc2)cc1          1
  acetanilide         CC(=O)Nc1ccccc1              1   (only N-aryl; C(=O)-N excluded; CH3 terminal)
  N-methylacetamide   CC(=O)NC                     0
  ethyl benzoate      CCOC(=O)c1ccccc1             2   (CH2-O and C(=O)-aryl; ester C(=O)-O excluded)
  benzotrifluoride    FC(F)(F)c1ccccc1             0
  tert-butylbenzene   CC(C)(C)c1ccccc1             0
  diphenylacetylene   c1ccc(C#Cc2ccccc2)cc1        0
  benzoic acid        OC(=O)c1ccccc1               1
  ibuprofen           CC(C)Cc1ccc(C(C)C(=O)O)cc1   4
  phenyl ethyl ether  CCOc1ccccc1                  2
  N-phenylurea        NC(=O)Nc1ccccc1              1
```

Scoring: mean absolute error, R^2, and the share of molecules where round(y) equals the true count,
on held-out molecules.

Verified against RDKit 2026.03.6: this SMARTS reproduces `CalcNumRotatableBonds` on all 3,346
molecules in the pool. The simplified rule in `target_rotatable_simple.md` agrees with it on 75%.
