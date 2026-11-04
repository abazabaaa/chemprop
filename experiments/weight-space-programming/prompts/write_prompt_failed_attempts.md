# The two weight-writing attempts that failed, and why

Both died the same way: `API Error: Claude's response exceeded the 64000 output token maximum`.
Neither produced any file at all - the first attempt crashed before writing a single weight.

## Attempt 1

Same as the successful prompt, except:
- the target was RDKit's **strict** `CalcNumRotatableBonds` definition (see
  `specs/target_rotatable_strict.md`): a long SMARTS with amide, ester, CF3/CCl3/CBr3 and
  tert-butyl exclusions;
- no instruction about response length or about writing the weights as a program.

## Attempt 2

Strict target again, but with an explicit instruction not to emit literal numbers and to write
build_weights.py as a program with loops and slice assignments. It crashed anyway, again on an
early turn.

## What actually goes wrong

The obvious reading - "it tried to print 22,593 numbers" - is probably not the whole story. Attempt
2 was told not to do that and still died, and both crashes happened very early, before any file was
written. The more likely cause is one enormous reasoning block: compiling that strict SMARTS into
relu arithmetic, in one go, with no intermediate artefact to lean on.

Two changes fixed it, and they are confounded - the successful run changed both at once:
1. the target was simplified (specs/target_rotatable_simple.md), removing the amide/ester/CF3
   exclusion logic;
2. the prompt demanded short responses, many small steps, and notes written to a file rather than
   held in a response.

**Do not report "a model can hand-write these weights" without separating these.** The clean
design is a 2x2: {strict, simple} target x {with, without} the short-response instruction, several
runs per cell. That tells you whether the limit was the target's complexity, the harness, or both.
An alternative worth running in the same grid: give the agent the strict target but let it build the
network in stages across several turns, with the partially built weights on disk as the state it
carries forward - closer to how the original blog post's Go experiment worked.
