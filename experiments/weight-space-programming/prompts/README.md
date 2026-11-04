# Prompts

The exact text handed to agents. Each agent got one prompt plus one folder path, and nothing else:
no conversation history, no hints about the candidate targets, no access to this directory.

- `read_prompt.md` - the reading task (experiments 1 and 2). Identical for all eight models apart
  from the folder path, so nothing about a particular target can leak through the wording.
- `write_prompt.md` - the weight-writing task (experiment 3), third and successful version.
- `write_prompt_failed_attempts.md` - the first two versions and why they died. Worth reading
  before you design a harness of your own: the failure mode is specific and will bite again.
