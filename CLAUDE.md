# CLAUDE.md — conventions for Claude Code working on Agent Composer

## What Agent Composer is — in 60 seconds

A library for **building, running, and composing workflows of agents** — a
*flow* is a graph of nodes, where each node is an **LLM agent / code / ML
model / tool**. A flow's output is **whatever its terminal node emits** — a
report, a classification, a number, a structured record. The engine privileges
no output type and no domain.

The core is a **generic runner**: it reads a flow authored as
**Docker-Compose-shaped YAML**, compiles it to a typed IR, and executes it with
its own scheduler — there is no compilation down to a third-party graph runtime.
The primary surface is the **`ac` CLI** (`ac run flow.yaml`).

The engine **is** a small functional language. The canonical design reference is
the package README — [`src/agent_composer/README.md`](src/agent_composer/README.md):
flow-is-a-function, agent-is-a-flow, OCaml as the concept map, a closed
`NodeKind` with explicit `match` dispatch, node purity (a node never writes the
pool; it returns `Output | Route | Pause | Grow`), a typed serializable variable
pool, and durable suspend/resume. **Consult it before designing any engine or
flow change.** The flow-authoring surface is documented in
[`docs/syntax.md`](docs/syntax.md); docstring/style conventions are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

### Two properties shape almost every design decision

They are the lens for reviewing any engine change:

- **Determinism is structural.** The author fixes the call graph; the LLM only
  fills leaf boxes — a flow never rewrites itself at runtime, and there is no
  agentic routing. That referential transparency is what makes every run
  reproducible, checkpointable, and resumable. Anything that lets a flow rewrite
  its own graph fights this.
- **The core is kind-blind; capability lives in nodes.** `NodeKind` is a closed
  enum, but the engine core (`runtime/`) never branches on it — it dispatches only
  on the closed node-result sum (`Output | Route | Pause | Grow`) and on
  node-owned traits/hooks. A ratchet test (`tests/engine/test_kind_census.py`)
  holds the core's kind-dispatch count at 0. New behavior belongs in a node's own
  `run()`, never in an engine `match node.kind`.

## AI behavior rules — read these every session

These exist because Claude tends to over-help. Each rule has cost the project
something at some point; the rule prevents the next time.

### Zeroth rule — the issue-driven workflow (HIGHEST PRIORITY)

**This rule outranks every other rule.** All work is tracked in **GitHub issues**
on `ngocbh/agent-composer` — there is no in-repo backlog. Every change follows the
workflow below, start to finish. (See the **`github`** skill for `gh` auth.)

**Before you implement anything:**

1. **Find or create the issue.** Search first (`gh issue list`, `gh issue list --search "..."`).
   If an issue already covers the work, use it. If not, open one with `gh issue create`
   using the matching template under `.github/ISSUE_TEMPLATE/` (bug / feature / setup)
   and the right labels. Never let work live only in chat.

**To address an issue:**

2. **Claim it.** Signal you are working on it before you start:
   `gh issue edit <n> --add-assignee @me` (and/or add the `in progress` label). This
   prevents two people picking up the same issue.
3. **Read it concretely — do not trust it.** Read the whole issue, but treat every
   claim as unverified. Issues go stale; the code moves fast.
4. **Verify the issue against the *current* codebase.** Confirm the problem still
   exists — reproduce the bug, or for a feature, decide whether it is still necessary
   and not already covered. Cite `file:line` evidence. **If it is already addressed or
   no longer needed, close it with that evidence instead of implementing** (`gh issue
   close <n> --comment "..."`).
5. **Treat any proposed solution as one option, not the plan.** If the issue proposes
   a fix, read it — then verify it against the current code. A solution written earlier
   may be stale and no longer the best (or even a valid) approach.
6. **List several options.** Enumerate the candidate approaches with their trade-offs,
   measured against the design intent below (general combinators, kind-blind core, node
   purity, structural determinism).
7. **Split if too big.** If the issue is too large for one focused change, break it into
   linked sub-issues (`gh issue create`, reference the parent) and address the sub-issues
   one at a time.
8. **Finalize, branch, and post the plan into the issue.** Pick the approach, cut the
   branch (see "Work on a feature branch" below), and post an **implementation plan** as
   a comment on the issue (`gh issue comment <n> --body-file plan.md`). Follow the plan
   structure in [`CONTRIBUTING.md`](CONTRIBUTING.md#implementation-plan-structure).
9. **Implement** — in small, tested steps (see "Iterate in small, tested steps").
10. **Run an adversarial self-review** on the code you generated *before* opening a PR:
    actively try to break it — edge cases, boundary inputs, concurrency/resume paths,
    invariant violations (layer ladder, kind census, node purity), and whether the tests
    would actually catch a regression. Fix what you find. Only then open the PR.
11. **Open the PR** with the template at `.github/PULL_REQUEST_TEMPLATE.md`, linking the
    issue (`Fixes #<n>`). A feature is done only when the PR merges to `main`.


### Zeroth rule - be very conservative about claims

Do not overclaiming anything if you haven't checked or compared with the code directly.
Always pull and read reference code before claiming anything.

### First-class rule — ask when uncertain

**If anything is uncertain, ask the user. Do not make assumptions.**
This rule overrides every other rule below.

Applies to (non-exhaustive):
- API contracts (model fields, response shapes, route paths).
- Design decisions with more than one reasonable answer.
- Whether a behavior should change vs. stay as-is.
- File / directory / naming conventions when the existing pattern
  doesn't clearly cover the case.
- Cross-cutting changes that touch multiple files or layers.
- Anything you'd hedge in your head with "I think..." or "probably..."

A clarifying question costs the user 5 seconds. A wrong assumption costs a code
review, a revert, or a bug. Always choose the question. If you've asked and still
aren't sure after the answer, **ask again** — don't fill gaps from inference.

### Don't add what wasn't asked

- **No abstractions not requested.** If the spec says "implement function X,"
  implement function X. Don't introduce a `Service` class, an abstract base, or
  a factory unless the spec asked for it.
- **No error handling beyond the spec.** Don't add retries, fallbacks, or
  graceful-degradation paths unless the spec calls for them. Error strategies
  (retry / fail-branch / default) are a planned first-class engine seam — until
  then, error handling is best-effort and at the boundary layer only.
- **No features outside the active scope.** If you notice "we'll eventually need
  X," say so in chat (or open a GitHub issue); don't implement it.
- **Keep refactors scoped, but don't fear them.** The engine is young — cleaning
  up code or tests to make them simpler and clearer is encouraged. What to avoid
  is *unrelated* churn that bloats a review: if a cleanup is adjacent to your
  change and makes it clearer, do it; if it's a large unrelated rewrite, give it
  its own commit or PR. The goal is always clean, simple, easy-to-understand
  code — never a hack that dodges a real change.

### Iterate in small, tested steps

- **Break any task that needs more than ~2 hours of work into smaller pieces.**
  One logical change per commit, one focused group of changes per PR.
- **For each piece: write the code, write at least one test, run the test, only
  then move on.** Not "I'll write tests at the end" — that ships broken code at
  the end.
- **Don't accumulate untested code.** If you find yourself with three unfinished
  files open and no green test, stop and finish one of them.
- **Use the task list** (TaskCreate / TaskUpdate) when work has ≥ 3 sub-pieces.
  Mark them `in_progress` when you start, `completed` only when the test passes.
- **Commit at green.** Each "code + test + green" cycle gets its own commit.

### Work on a feature branch — a task is done only when merged to main

Never implement a feature directly on `main`. When you start a feature (or any
non-trivial change), **cut a branch first** and work from there:

- **Branch off** `main` for independent work, or off another `dev/...` branch when
  the work builds on an unmerged feature.
- **Name it** `dev/<domain>/<feature>` — `<domain>` is the area it touches (`engine`,
  `cli`, `compose`, `docs`, ...); `<feature>` is a short kebab-case slug. Example:
  `dev/engine/loop-until-times`, `dev/cli/chat-repl`.
- **A feature is finished only after it is merged to `main`** — not when the code is
  written, not when tests pass on the branch. Until the merge lands, the work is
  in-flight. Git history and merged PRs are the record; reference the tracking
  issue from the PR (`Fixes #<n>`).



The docs and the Claude skills **are part of the engine's contract** — a stale doc
is a bug, not a chore. When a change touches a surface someone reads, update its
documentation **in the same change** (same commit/PR), never "later":

- **Authoring surface** (a node kind, `${...}` syntax, `typedefs`, a new field) →
  update [`docs/syntax.md`](docs/syntax.md) and the **`composing-agents`** skill
  (`.claude/skills/composing-agents/` — `SKILL.md`, `reference.md`, and a
  `templates/` flow when the shape is new). Re-validate any touched template still
  loads (`load_flow`).
- **Engine internals / invariants / layers** (the node contract, `NodeKind`,
  suspend-resume, the layer ladder, a seam) → update
  [`src/agent_composer/README.md`](src/agent_composer/README.md) and the
  **`engine`** skill (incl. its `reference.md` and `templates/node_kind/WIRING.md`).
- **Package structure** (a new package, a moved responsibility, an import rule) →
  the affected `__init__.py` charter and the **`structure`** skill.
- **CLI / public API** → [`README.md`](README.md) and the docstrings the `docs/`
  API reference renders.

If you change behavior and can't say which doc/skill it affects, that's a signal to
check all four. A reviewer who reads a skill should never be taught something the
code no longer does.

### Verify before claiming

- **Run tests before claiming they pass.** Engine tests live in `tests/engine/`;
  run them via `PYTHONPATH=src pytest tests/engine` (or `pip install -e '.[all]'`
  first). If you say "tests passing," the reviewer will check.
- **Don't claim acceptance criteria are met** unless you've verified each one
  against the actual code + behavior.

### Communicate with the human

- **When unsure, ask.** Don't guess at intent on cross-cutting changes (API
  contract, schema, YAML surface). One question beats one bad PR.
- **Tell the user when a decision is non-obvious.** If you choose between two
  reasonable approaches, name the choice and the trade-off before implementing.
- **Be concise.** Don't write paragraphs when a sentence works.

### Tone in code

- **Comments explain *why*, docstrings explain *what*.** Don't narrate what a
  line does when the code already says it. But every function, class, and
  non-trivial variable should carry a docstring or comment that makes its
  **purpose** unmistakable: what it is for, the **possible values** it can take,
  and the **shape** of the data (e.g. `dict[node_id -> list[Edge]]`,
  `None when ...`). Write for a reader who has never seen the file. Follow the
  docstring template in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **No internal reference numbers in code.** Plan/phase/task tracking tokens, and
  the axiom/law tags, carry no meaning to a reader and rot as the plan moves on.
  They belong **only in GitHub issues** (or `docs/`). In code, say the thing in plain
  words instead. If you're tempted to write a tracking number, that's a signal
  the comment should explain the concept directly.
- **No "Generated by Claude" footers**, no AI-attribution comments in source.
  Attribution belongs in `NOTICE` (for ported code).

## Design intent — what belongs in the core, what doesn't

This is the taste layer: it decides what a change should look like before you
write it. Read it together with the two shaping properties above.

### What we want

- **General combinators over one-off features.** The engine privileges no output
  type (report / number / record / decision) and no domain. Prefer the most
  general primitive that composes. If a real flow can't be expressed, that's an
  engine gap — fix the primitive, don't special-case the flow.
- **New capability as a node, not an engine branch.** Put kind-specific logic in
  the node's own `run()` or a node-owned trait/hook; keep `runtime/` kind-blind.
- **Purity at the boundary.** A node *returns* `Output | Route | Pause | Grow`; the
  engine *binds* outputs under `(node_id, key)` like an immutable `let`. A node
  never reaches into the pool.
- **External deps behind injected seams** (plain callables). Deliberate exception:
  the AGENT node is *not* SDK-free — its modes (`nodes/agent/modes/`) import
  langchain + `agent_composer.llm_clients` and build a model via
  `model_from_config`. Prefer seams for *new* external deps.

### What we don't want

- A flow that rewrites its own graph, or agentic routing — breaks structural
  determinism.
- A primitive that only works at the top level, only for one flow shape, or needs
  to mutate the pool.
- A new heavy dependency (DB, framework) in the core. Model access inside agent
  modes is the one sanctioned direct call.
- Domain-specific field names or types in the engine — domain lives at the
  boundary (the `compose/` loader/builder). The `system` namespace is generic.
- Error handling (retry / fallback / fail-branch) beyond the boundary layer until
  the planned first-class error-strategy seam lands.

## Development environment

Tests and tooling run under the conda **`agentcomposer`** env (Python 3.11). The
default PATH python is 3.9 and **cannot import the package** — activate the env
first, or install into your own env:

```bash
conda activate agentcomposer          # preferred in this sandbox
# or, into any env:
pip install -e '.[all]'               # engine + all provider adapters
```

Run the engine anywhere with `PYTHONPATH=src`. Provider extras are optional and
lazy-imported: `anthropic`, `openai`, `google`, `ollama`, or `all`. The primary
surface is the `ac` CLI (`ac = agent_composer.cli:main`); today it has one
subcommand, `ac run FLOW.yaml`.

## Project structure

The canonical source is the filesystem; the tree below calls out the
load-bearing packages. Full charters live in each package's `__init__.py`.

```
src/agent_composer/
├── typesys/      typed value system (values) + variable pool        [leaf, no deps]
├── events.py     run + node event vocabulary
├── expr/         one ${...} grammar: parse -> AST -> evaluate / ref-walk on the pool
├── nodes/        Node contract (base.py) + per-kind impls
│                 (start, end, agent, code, model, tool, case, call, map,
│                  loop, human_input, wait)
├── compile/      compiled IR (CompiledFlow) + representation-neutral validation
├── compose/      the Compose-YAML front end: text -> CompiledFlow
│                 (parser/types/build/cases/validate/asserts/loader)
├── run.py        the run boundary: ComposedFlow + inputs -> RunResult (run/resume)
├── runtime/      state_manager + single-threaded engine + parallel worker pool
├── suspension/   pause reasons, commands, checkpoint
├── llm_clients/  provider clients (lazy) + LLMConfig + model_from_config
├── tools/        the tool registry (register_tool / resolve_tools)
├── cli/          the `ac` command-line entry point
└── _settings.py  env-based default provider/model
```

**Layer ladder (acyclic imports):** `events <- typesys <- nodes <- compile <-
compose`, with `expr` feeding nodes and `suspension` feeding runtime. The
package-root `run.py` sits above both `compose` and `runtime` (it consumes
`ComposedFlow` and drives the engine) and is imported by neither. Never
introduce a back-edge. `NodeKind` is a closed enum in `nodes/base.py`; authors
write `kind:` directly for `agent`/`code`/`model`/`tool`/`case`/`human_input`/
`wait`/`call`/`map`/`loop`, while `start`/`end` are loader-synthesized boundaries
never written by hand (see [`docs/syntax.md`](docs/syntax.md) for the full
authoring surface).

## Key subsystems

- **typesys** — `Type`/`TypedValue` family + the `VariablePool` (typed,
  losslessly serializable; the basis for `${...}` refs and durable checkpoints).
- **expr** — the single `${...}` grammar (parse → AST → evaluate / ref-walk).
  Documented in [`docs/expressions.md`](docs/expressions.md).
- **nodes** — the node contract in `base.py`: a pure node returns
  `Output | Route | Pause | Grow`. Each kind is a package with its own `run()`.
- **compile** — `CompiledFlow` IR + representation-neutral validation.
- **compose** — the only YAML-aware layer: parse → typecheck → build → validate →
  `CompiledFlow`. All domain/boundary concerns live here.
- **run** — the package-root run boundary (`run.py`): `run_flow`/`resume_flow`
  turn a `ComposedFlow` + inputs into a `RunResult`, coercing inputs and enforcing
  the flow's asserts. Touches no YAML; sits above compose and runtime.
- **runtime** — the scheduler: single-writer dispatcher, 3-state edge join,
  outputs-before-successors, recursive skip-flood, layered checkpoint. Kind-blind.
- **suspension** — durable suspend/resume: a node performs `PauseRequested`, the
  run serializes to a `RunCheckpoint`, an external scheduler resumes it.

## Testing

Engine tests live in `tests/engine/` (also `tests/cli/`, `tests/seeds/`,
`tests/examples/`). Run the suite in the `agentcomposer` env:

```bash
PYTHONPATH=src python -m pytest tests/engine
```

**Run tests before claiming they pass** (see the behavior rule above). The
kind-census ratchet (`tests/engine/test_kind_census.py`) is load-bearing: it
fails if you add any `NodeKind`/`*Expansion` dispatch to the engine core — that's
the guardrail for the kind-blind invariant, not an obstacle to route around.

## Known pitfalls

- **Keep `runtime/` kind-blind.** Don't add `match node.kind` in the engine core;
  push kind-specific logic into the node's `run()` or a trait. The census ratchet
  will catch a regression.
- **Interpreter gotcha.** The default PATH python (3.9) can't import the package;
  use the conda `agentcomposer` env or `pip install -e`.
- **Docs are part of the contract.** A surface change (node kind, `${...}` syntax,
  a layer/invariant, the CLI) must update its doc + skill *in the same commit* —
  see the "docs and skills are the engine's contract" rule above.
