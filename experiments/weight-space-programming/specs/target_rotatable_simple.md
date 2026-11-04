# Target: number of freely rotatable single bonds

For each molecule, y = the number of bonds in the molecule that satisfy ALL of:

1. the bond is a SINGLE bond (bond_type=SINGLE; aromatic, double and triple bonds never count);
2. the bond is NOT in a ring (is_in_ring = 0);
3. each of its two atoms has at least 2 heavy-atom neighbours
   (heavy-atom degree = total_degree - total_num_Hs, so terminal atoms are excluded);
4. neither of its two atoms is part of a triple bond anywhere in the molecule.

Nothing else matters: no element types, no charges, no amide or ester exceptions.
Each qualifying bond counts exactly 1, so y is a non-negative integer (0 to about 37 here, mean ~3.6).

Worked examples:
  ethane              CC                           0   (both atoms terminal)
  butane              CCCC                         1   (only the middle bond)
  pentane             CCCCC                        2
  cyclohexane         C1CCCCC1                     0   (all bonds in a ring)
  ethylcyclohexane    CCC1CCCCC1                   1   (CH2-ring bond; the CH3-CH2 bond is terminal)
  benzene             c1ccccc1                     0   (aromatic, and in a ring)
  toluene             Cc1ccccc1                    0   (the methyl is terminal)
  biphenyl            c1ccc(-c2ccccc2)cc1          1   (the single bond joining the two rings)
  benzotrifluoride    FC(F)(F)c1ccccc1             1   (C-aryl; the three C-F bonds are terminal)
  tert-butylbenzene   CC(C)(C)c1ccccc1             1   (the quaternary C to aryl)
  benzoic acid        OC(=O)c1ccccc1               1   (C-aryl; the C-OH bond is terminal)
  glycine             NCC(=O)O                     1   (the N-CH2 bond is terminal; CH2-C counts)
  acetonitrile        CC#N                         0   (triple bond)
  diphenylacetylene   c1ccc(C#Cc2ccccc2)cc1        0   (both aryl-C bonds touch a triple-bond atom)
  N-methylacetamide   CC(=O)NC                     1   (the C(=O)-N bond)
  acetanilide         CC(=O)Nc1ccccc1              2   (C(=O)-N and N-aryl)
  ethyl benzoate      CCOC(=O)c1ccccc1             3   (CH2-O, O-C(=O), C(=O)-aryl)
  phenyl ethyl ether  CCOc1ccccc1                  2   (CH2-O and O-aryl)
  ibuprofen           CC(C)Cc1ccc(C(C)C(=O)O)cc1   4

Scoring: the network is run on held-out drug-like / small organic molecules (up to 60 heavy atoms).
Reported: mean absolute error, R^2, and the share of molecules where round(y) equals the true count.
