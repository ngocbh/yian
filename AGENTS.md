# Guide for coding agents

yian loads YAML flows of agents, code and tools, compiles them to a fixed graph and runs them,
writing every step to an append-only run log. It is for workflows that repeat, must be reliable
and improve across runs. Judge a feature by whether such a workflow needs it, and never trade
predictable replay for expressiveness.

Read before you change code:

- [`CONTRIBUTING.md`](CONTRIBUTING.md): setup, the test commands and the pull-request workflow.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): the pipeline, what each `src/` entry holds, and the rules
  that must not break.
- [`conformance/README.md`](conformance/README.md): the cases that pin what a flow does.
- [`docs/STYLE.md`](docs/STYLE.md): how to write a page on the docs site.

## Rules

- Start from a GitHub issue. Check its claims against the current code and cite `file:line`.
  Post a short plan on the issue before you write code.
- Stay inside the issue's scope: no unrelated features, abstractions, error handling or
  refactors. Ask when an API, schema, name or behavior is unclear.
- Work test-first in small commits. Don't say a check passed unless you ran it, and list the
  exact commands in the pull request.
- Keep the rules in `ARCHITECTURE.md`: append before apply, deterministic replay, a fixed graph,
  pure nodes, a kind-blind engine, one-way imports, a closed list of failure codes.
- Put a new external dependency behind a small injected function. Runtime-specific APIs go in
  `src/platform.ts`, and environment reads in `src/defaults.ts`.
- A change to what users see (flow syntax, a node kind, the CLI, the host API, a failure code)
  updates its page in `docs/src/content/docs/` and, for flow syntax, the `composing-agents`
  skill in `.claude/skills/`, in the same pull request.
- Keep code direct. Docstrings say what a symbol does, comments say why. No issue numbers, plan
  labels or AI-attribution footers in source.

## The flow language

Judge each change to the language against these. Breaking one needs a reason in the issue.

- A flow does exactly what the docs say, or fails with a clear error: nothing silent, nothing
  undefined.
- Catch errors at load time when you can, and name the location and the fix.
- Few concepts that compose, working the same in every position. Grow through flows, host tools
  and imports, not a new keyword per need.
- A flow is plain YAML. yian adds meaning inside strings (`${...}`, conditions) and through YAML
  tags, never by changing YAML.
- Once users rely on something, a breaking change comes with an error that says what to change.
