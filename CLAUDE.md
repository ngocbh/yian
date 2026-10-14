# Agent guide for Yian

## Project

Yian builds and runs workflows of agents, code, and tools. A
flow is written as Docker-Compose-shaped YAML, compiled to a typed form, and run
by this project’s own scheduler. The main interface is the `yian` CLI.

Goals:

- Workflows that repeat, must be reliable, and should improve across runs, such as SOPs and
  trading routines.
- Not a replacement for interactive agent harnesses (Codex, Claude Code, OpenCode). Judge a
  feature by whether a repeatable workflow needs it, and don't trade predictable replay for
  expressiveness.

Use these documents as the source of truth:

- [`src/yian/README.md`](src/yian/README.md) for engine design.
- [`docs/syntax.md`](docs/syntax.md) for flow syntax.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) for the GitHub workflow, tests, docs, and
  code style.

## Required GitHub workflow

GitHub issues are the task list. Every change starts with an issue and ends with
a merged pull request. Follow [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full
workflow and writing examples.

Before implementation:

1. Search for an issue. Create one from the right template if none exists.
2. Assign it to yourself and mark it in progress.
3. Check every claim against the current code. Reproduce bugs and cite
   `file:line` evidence. If the issue is stale, close it with that evidence.
4. Create a `dev/<domain>/<short-name>` branch.
5. Post a short plan as an issue comment.

Keep issues and comments short. State facts, evidence, decisions, and results.
Only compare options when there is a real choice. A proposed fix in an issue is
a suggestion until the current code supports it.

While working:

- Stay inside the issue scope. Do not add unrelated features, abstractions,
  error handling, or refactors.
- Ask the user when an API, schema, behavior, name, or cross-cutting design choice
  is unclear.
- Work in small steps: change, test, pass, commit.
- Update the issue when the plan changes, work is blocked, or a useful part is
  complete. Do not post play-by-play comments.
- Open a pull request with `Fixes #<issue>`, link the plan comment, and list the
  exact checks run.
- Do not say work is complete or tests pass until you have verified it. Work is
  complete only after the pull request merges into `main`.

## Engine rules

Read [`src/yian/README.md`](src/yian/README.md) before an
engine or flow change.

- The author fixes the graph. A running flow does not rewrite itself, and an LLM
  does not choose the graph.
- `runtime/` stays kind-blind. Kind-specific work belongs in the node. The kind
  census in `tests/engine/test_kind_census.py` must stay at zero.
- Nodes do not write the variable pool. They return one of `Output`, `Route`, or
  `Grow`; the engine records and binds the result.
- Runtime transitions stay record-authoritative: prepare and validate, append to
  `RunLog`, then apply through the reducer. Recovery folds the same reducer.
- Keep imports acyclic: `diagnostics` sits below `events`; nodes combine the
  `diagnostics`, `typesys`, and `expr` leaves before `compile <- compose`.
- Prefer general, domain-neutral nodes and combinators over one-off behavior.
- Put new external dependencies behind small injected callables. The AGENT node's
  direct model access is the existing exception.

## Language principles

Judge every change to the flow language against these. A change that breaks one
needs a reason recorded in its issue.

1. **One defined meaning.** A flow does exactly what the docs say, or fails with a
   clear error. Nothing is undefined or silently different.
2. **Catch errors early.** Load time before runtime, runtime before a wrong result.
3. **Fail loudly.** A failure never looks like success: no silent null, no silent skip.
4. **Few concepts that compose.** Before adding a feature, remove the restriction
   that makes it seem necessary.
5. **Orthogonal.** A feature works in every position where it makes sense, without
   special cases.
6. **Regular.** The same syntax means the same thing everywhere, and different
   things look different.
7. **Unambiguous.** Neither a reader nor the loader has to guess. Quoted text is text.
8. **Explicit over implicit.** An implicit rule is allowed only when it is easy to
   predict and visible, for example in a type.
9. **Predictable cost.** An author can tell what a flow costs (live state, growth,
   calls) without reading the engine.
10. **Local reasoning.** A node, body, or flow can be understood from its own text
    and the interfaces it declares.
11. **Grow through user-written pieces.** Flows, host tools, and imports, not a new
    keyword for each need.
12. **Stable once relied on.** A breaking change comes with a migration path and an
    error that says what to change.
13. **Errors teach.** A load error names the location and, where possible, the fix.
14. **Plain YAML.** A flow file is ordinary YAML (YAML 1.2 core-schema values, plus
    unquoted dates and merge keys, as yian's loader reads them) that reads naturally to
    anyone who knows YAML. yian adds meaning inside strings (`${...}`, conditions) and
    through YAML's own tags; it never changes YAML's syntax or makes a value mean
    something YAML says it doesn't.

## Docs change with code

Update the matching docs and project skill in the same pull request as a surface
change. The mapping is in [`CONTRIBUTING.md`](CONTRIBUTING.md#docs-change-with-code).
A stale guide or skill is a bug.

## Environment and tests

Yian supports Python 3.10 or newer. On this host, use the
`yian` conda environment; the default Python may be too old.

```bash
conda activate yian
python -m pip install -e ".[all,dev]"
```

Run focused tests while working, then the relevant suite:

```bash
PYTHONPATH=src python -m pytest tests/engine
PYTHONPATH=src python -m pytest
```

## Code style

Keep code direct and scoped. Docstrings explain what a symbol does; comments
explain why non-obvious code exists. Do not put task numbers, plan labels, or
AI-attribution footers in source code. Follow the short docstring guide in
[`CONTRIBUTING.md`](CONTRIBUTING.md#docstrings).
