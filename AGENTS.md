# Guide for coding agents

yian loads YAML flows of agents, code and tools, compiles them to a fixed graph and runs them,
writing every step to an append-only run log. It is for workflows that repeat, must be reliable
and improve across runs. Judge a feature by whether such a workflow needs it, and never trade
predictable replay for expressiveness.

Read before you change code:

- [`CONTRIBUTING.md`](CONTRIBUTING.md): setup, the test commands and the pull-request workflow.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): the pipeline, what each `src/` entry holds, and the rules
  that must not break.
- [`tests/conformance/README.md`](tests/conformance/README.md): the cases that pin what a flow does.
- [`docs/STYLE.md`](docs/STYLE.md): how to write a page on the docs site.

## Rules

- Follow the pull-request steps and code style in `CONTRIBUTING.md`. Check the issue's claims
  against the current code and cite `file:line`.
- Stay inside the issue's scope: no unrelated features, abstractions, error handling or
  refactors. Ask when an API, schema, name or behavior is unclear.
- Work test-first. Don't say a check passed unless you ran it.
- Put a new test where `tests/` mirrors `src/` ("Where tests live" in `ARCHITECTURE.md`).
- Keep the rules in `ARCHITECTURE.md`: append before apply, deterministic replay, a fixed graph,
  pure nodes, a kind-blind engine, one-way imports, a closed list of failure codes.
- Put a new external dependency behind a small injected function. Runtime-specific APIs go in
  `src/platform.ts`, and environment reads in `src/defaults.ts`.
- A change to flow syntax also updates the `composing-agents` skill in `.claude/skills/`, in the
  same pull request as its docs page. A change to the docs, the skill or `src/cli/flows/` also
  regenerates the bundle `yian chat` carries (`YIAN_UPDATE_BUNDLE=1 bun test tests/repo/bundled.test.ts`).

## The flow language

Judge each change to the language against these. Breaking one needs a reason in the issue.

- A flow does exactly what the docs say, or fails with a clear error: nothing silent, nothing
  undefined.
- Catch errors at load time when you can, and name the location and the fix.
- Few concepts that compose, working the same in every position. Control flow is a closed set:
  `case`, `loop`, `map` and `call`. Grow through flows, host tools and imports, not a new keyword
  per need. When a flow needs more computation, use a `code` node or a tool.
- Expressions are pure and always end: no side effects, no user-defined functions, no recursion.
  A host's builtin must be pure and deterministic too.
- `defs:` entries are subflows you can read in the file, not functions: a def can't call itself,
  directly or through another.
- Give every new construct that repeats a bound (`times:`, `max:` or a cap in `src/defaults.ts`).
  A `while:` or `until:` loop is bounded only when it sets `max:`.
- A flow is plain YAML. yian adds meaning inside strings (`${...}`, conditions) and through YAML
  tags, never by changing YAML.
- Once users rely on something, a breaking change comes with an error that says what to change.
