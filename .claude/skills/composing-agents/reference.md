# Reference — composing flows

Deep-dive companion to [`SKILL.md`](SKILL.md). The skill is the authoring
workflow; this is the lookup sheet (operators, contexts, type forms, recipes,
gotchas) plus the [`templates/`](templates/) index. The authoritative grammar is
[`docs/syntax.md`](../../../docs/syntax.md); runnable end-to-end flows are in
[`examples/`](../../../examples).

## Templates (copy one as a starting point)

Each file in [`templates/`](templates/) is a minimal, loadable flow for one shape.
Copy, rename, and edit.

| Template | Shape it shows |
|----------|----------------|
| [`minimal.yaml`](templates/minimal.yaml) | one AGENT, `str` in/out |
| [`compact.yaml`](templates/compact.yaml) | the SAME one-agent flow in compact form (no `nodes:` map) |
| [`pipeline.yaml`](templates/pipeline.yaml) | AGENT → CODE (typed record) — deterministic post-processing |
| [`inline-code.yaml`](templates/inline-code.yaml) | inline `code:` source — a bare body run in-process (no `module:function`) |
| [`typed_output.yaml`](templates/typed_output.yaml) | AGENT with a record `output:` — structured generation + `retries:` |
| [`branching.yaml`](templates/branching.yaml) | classify → `case` route → `\|` join |
| [`tool-use.yaml`](templates/tool-use.yaml) | a `tool` node (no LLM) feeding an AGENT |
| [`human-in-loop.yaml`](templates/human-in-loop.yaml) | a blocking `human_input` gate (typed approve/revise answer) |
| [`human-questions.yaml`](templates/human-questions.yaml) | a `human_input` gate with multiple questions — static `questions:` + `adaptive_questions:` |
| [`wait-until.yaml`](templates/wait-until.yaml) | an absolute UTC WAIT ordered before downstream CODE |
| [`child-summarize.yaml`](templates/child-summarize.yaml) | a reusable CHILD flow |
| [`call-child.yaml`](templates/call-child.yaml) | `call` a sibling flow once (via `uses:`) |
| [`expr-and-call.yaml`](templates/expr-and-call.yaml) | `${a + b}` arithmetic in a binding + inline `call(...)` over an in-file `defs:` callee |
| [`map-fanout.yaml`](templates/map-fanout.yaml) | `map` a child over a list, in parallel |
| [`loop.yaml`](templates/loop.yaml) | `loop` a body until a predicate goes false — chat-shaped blocking turn |
| [`chat.yaml`](templates/chat.yaml) | a conversational REPL — `human_input` → exit CASE → agent/code, with flow-owned exit, the shape `ac chat` runs |
| [`llm-config-cascade.yaml`](templates/llm-config-cascade.yaml) | flow-level `llm_config:`, per-node override, `inherit: false` |
| [`node-env-config.yaml`](templates/node-env-config.yaml) | flow-level `env:` default + per-node override (node wins), e.g. `max_tool_iterations` |

`call-child.yaml` and `map-fanout.yaml` depend on `child-summarize.yaml` being on
the search path — run them from the `templates/` dir (or pass the dir to
`load_flow(search_paths=...)`).

## Operators inside `${...}`

One expression grammar, everywhere `${...}` appears:

| Form | Meaning |
|------|---------|
| `${a + b}`, `${a * b + 1}` | arithmetic — `+ - * / % **`, unary minus |
| `${a == b}`, `${x in [1, 2]}` | comparisons / membership / `and`/`or`/`not` |
| `${[a, b]}`, `${upper(x)}` | list literal / pure builtin call |
| `${X:-default}` | value, else `default` if absent |
| `${X:?"msg"}` | required — fail with the literal `msg` if absent (quote the message) |
| `${a \| b \| c}` | first present among peers — **the branch-join coalesce** |
| `$$` | a literal `$` (universal escape — in a prompt `$$` renders a single `$`) |

Nesting a ref is allowed: `${a:-${b:-"lit"}}`. The `:-`/`:?` RHS is an
expression, so a string default must be **quoted** (`${x:-"today"}`; bare `today`
reads a variable). A whole-string `${ref}` resolves to the **typed value**;
embedded in surrounding text it is **stringified**. A child-flow call is the
whole-value `call(...)` directive (below), never a `${flow(args)}` span.

## The three expression contexts (one grammar)

One `${...}` grammar; the context decides what happens to the result:

| Context | Where | What it does |
|---------|-------|--------------|
| **Bindings** | `input:` / `output:` values | **evaluated** to a typed value — refs, literals, arithmetic, lists, `:-` / `:?` / `\|`, pure builtins. Child-flow call = `call(...)` directive. |
| **Conditions** | `when:` / `asserts:` | **tested** as a boolean: `== != < <= > >=`, `in`/`not in`, `and`/`or`/`not`, parens, arithmetic operands. Canonical **bare** form (no `${}`); `${...}`-wrapped spellings load and evaluate identically. |
| **Prompts** | `prompt:` text | free text with embedded `${...}` spans (each stringified) |

> Bindings wire, conditions test, nodes compute. A heavy transform still belongs
> in a `code` node — but simple arithmetic/coalesce in a binding is fine.

## Inline `call(...)` directive

A child-flow call written as a binding's **whole value** (or a `case`
`then:`/`else:` target) — `call(f, arg=${ref}, k=lit)` — desugars at load into an
anonymous `call` node; the host becomes `${<synth>.output}`. Keyword args only;
nesting allowed (`call(a, x=call(b, y=1))`). Recognized **only** as the entire
trimmed value. The old `${flow(args)}` (a flow call inside braces) is a load
error: hoist an embedded flow call to a named node; split a coalesce-of-calls into
per-node outputs. Pure builtins (`${upper(x)}`) stay legal inside `${}`.

## Type forms

Python typing vocabulary. Scalars: `str`, `int`, `float`, `bool`, `date`,
`datetime`, `object`, `None`.

| Form | Example | Notes |
|------|---------|-------|
| list | `list[str]` | |
| nullable | `Optional[str]` | may be `null` |
| default-fill | `lookback: int = 30` | filled when the input is omitted |
| enum | `Literal[go, no_go, wait]` | one of these tags |
| alias | `Basket: list[str]` | aliases compose |
| record | (see below) | fields recurse |

```yaml
typedefs:
  Signal:
    score: float
    note: Optional[str]
```

`Optional[X]` (nullable) and `= default` (omission-fill) are **orthogonal** —
nullable says the value may be null; default says what to use when the input
isn't supplied at all.

## Recipes

**Get a structured / numeric value out of an agent.** Declare the typed shape directly
as the AGENT's `output:` — a record, `float`/`int`/`bool`, or list switches the agent to
**structured generation** (the engine derives a schema and the model emits a conforming
value, retried up to `retries:` times on deviation; see `typed_output.yaml`). Use a
downstream `code` node only when you need deterministic post-processing of the value.

**Branch and rejoin.** A `case` runs exactly one branch; the others skip and their
refs resolve to null. Always coalesce the branches back with `${a | b | c}` (see
`branching.yaml`). Routing on a `Literal` is exhaustiveness-checked — cover every
tag or add an `else:`.

**Order without data flow.** When node B must run after A but consumes no value
from it (a `wait`, a side-effecting tool), use `depends_on: [a]` (co-skips B if A
skipped) or `runs_after: [a]` (orders only; B still runs).

**Wait until an absolute deadline.** Give `wait` an ISO `date` or `datetime` via
`until:`. Dates mean midnight UTC, naive datetimes are UTC, and aware datetimes
are normalized to UTC. WAIT blocks one worker and returns `None` at the deadline;
it does not pause/checkpoint the run. See `wait-until.yaml`.

**Ask the human.** For a *guaranteed* gate use a `human_input` node (it blocks its
worker until the host answers). For "ask only if the model decides it needs to",
give a `tool_calling` agent `tools: [ask_user]`. Both calls stay inside their live
node execution. A `human_input` gate can present multiple-choice questions
(static `questions:`), have an LLM compose them
(`adaptive_questions:`, which desugars to a compose-agent + gate at load), or read
them from an upstream node (`questions: ${name}`); the answer is then a record
keyed by each question's `header`.

**Reuse a sub-flow.** Factor the repeated work into its own flow file, bind it with
`uses: <alias>: <filename>`, and `call:` it (once) or `map:` it (per list element).
Reference its object fields downstream as `${node.output.field}`.

**Loop until done.** Use a `loop` node to re-run a body under one of three drivers,
threading a carried record `'a -> 'a` (the body's `output:` shape EQUALS its
`input:`). The `input:` is the SEED carried record. Pick EXACTLY ONE driver:
`while: not done` is a PRE-check over the carried record (0+ runs, stop when the
predicate goes false); `until: done` is a POST-check / do-while (1+ runs, stop
when the predicate becomes TRUE); `times: N` runs exactly N times with no
predicate. Predicates use the canonical bare form (no `${}`). `max:` is a required runaway guard
for `while:`/`until:` — but REDUNDANT and REJECTED with `times:`. A body that blocks
(a `human_input` leaf) makes the loop a live chat REPL — the same run threads each
answered turn. A long
loop stays cheap: each committed iteration is PRUNED from the live graph, so only one
is resident at a time. See `loop.yaml`.

**Conversational REPL (LOOP + human_input).** A chat is a `loop` whose body blocks at
`ask` (`human_input`), then a CASE routes exact `/exit` and `/quit` values to an exit
record before `reply` (`agent`). The normal branch appends the message and reply to the
transcript. Defaulted output bindings rejoin those branches into the carried 2-field
record `{transcript: str, exited: bool}` (`'a -> 'a`). Typed exit is flow-owned; only
terminal EOF is normalized by the bundled host. The loop body's `output:` must EQUAL the
carried record. A lone multi-ref coalesce has no inferable type, so the template uses
adjacent defaulted spans for the `str` field and a lone defaulted span for `bool`. This is
the shape `ac chat` runs; see `chat.yaml` (and `examples/chat.yaml` for a runnable copy).

## Gotchas

- **Inline `{ ... }` maps + `${...}` need quotes.** In an inline flow mapping the
  `}` in `${input.x}` closes the map early. Either quote the value
  (`input: {x: "${input.x}"}`) or use block form (preferred):
  ```yaml
  input:
    x: ${input.x}
  ```
- **AGENT `output:` may be any shape** — a bare `str`/`Literal[...]` keeps it a text
  producer; a record/number/bool/list switches it to structured generation (schema-checked
  at the write boundary, `retries:`-capped self-correction).
- **Prompts see only LOCAL inputs.** Inside `prompt:` you may reference only names
  the node declares in its own `input:` block, written bare (`${name}`). Pool refs
  (`${input.x}`, `${other.output}`) go in the `input:` block first.
- **No `edges:` block, no per-node `id:`, no body wrappers.** The graph is inferred
  from `${...}` references; a node body is flat.
- **`call:` resolves defs-first, else a `uses:` alias** (a sibling file by name on
  the search path). `alias@v1` adds a version guard.
- **MODEL nodes aren't wired** — `kind: model` parses but running one raises. Use
  `code` for deterministic compute.
- **`loop` bodies must be `'a -> 'a`.** The body's `output:` shape must EQUAL the
  loop's `input:` (carried record) and read only a subset of its names — checked at
  build (names) and load (types). Give EXACTLY ONE driver: `while:` (pre-check,
  0+ runs), `until:` (post-check / do-while, 1+ runs), or `times: N` (fixed count).
  `max:` is required for `while:`/`until:` but REDUNDANT+REJECTED with `times:`.
  Every `while:`/`until:` ref must name a carried field (a typo is rejected at load,
  not read as falsy), and `max:`/`times:` must be a plain integer `>= 1`.
- **Node-local `asserts:` reading `${output}` are POST checks** — they fire once the
  node's value is committed, and fail the run loudly on a false/raising expr. This
  includes a `call` node: its POST asserts may read `${output}` **and** the call's
  declared inputs (`${name}`), like a leaf node. `map` nodes reject node-local
  `asserts:` at load time — assert a `map`'s result with a flow-level/downstream check.
- **Flow-level `asserts:` read `input.X` (boundary, before any node) or `node.output`
  (post, after the terminal).** `${system.X}` is **not** allowed — it is ambient (no
  producer node to draw the check's dependency edge from), so a `system` ref is a
  LoadError; put a system-dependent check in a body node's `asserts:` or a `case`.
  A flow's `asserts:` are lowered onto its synthesized `start`/`end` boundary nodes and
  fire on the ordinary node path (`start` post-coercion, `end` post-terminal), so a
  called `defs:`/`uses:` child's asserts fire identically to the top-level flow's.

## Model selection — the `llm_config` cascade

Model fields resolve **per field, most-specific wins**. An agent fills only the fields
it leaves unset from the layer outside it. Precedence (most specific first):

1. the agent's own `llm_config:`
2. the enclosing (sub)flow's `llm_config:`, then each parent flow outward
3. the CLI `--provider` / `--model` flags
4. process provider/model defaults, read exactly once when the run initializes

Set flow-wide defaults with a top-level `llm_config:`; override one field per node;
opt a node out of the authored outer layers with `inherit: false` (own dict plus
the snapshotted process provider/model defaults; no flow/parent/CLI layers). The
static tree is resolved up front, so later environment changes do not alter the
effective selection. See [`llm-config-cascade.yaml`](templates/llm-config-cascade.yaml).

```yaml
llm_config: {provider: anthropic, temperature: 0.2}   # flow layer
nodes:
  a: {kind: agent, prompt: hi, llm_config: {model: claude-opus-4-8}}        # fills `model`
  b: {kind: agent, prompt: hi, llm_config: {provider: openai, model: gpt-5.5, inherit: false}}
```

## Per-node config — `env:`

Any node may carry an `env:` mapping of **static** config keys its own `run()` reads
(the engine never interprets `env`). A flow-level `env:` is the default for every node;
each node overrides individual keys. Merge is **static at build time** (`{**flow_env,
**node_env}`, node wins), **flow-local** (a child flow does NOT inherit the parent's
flow env), literals only (no `${...}`).

```yaml
env: {max_tool_iterations: 150}         # flow default for every node
nodes:
  r: {kind: agent, prompt: hi, tools: [search], env: {max_tool_iterations: 300}}  # node override
```

Keys in use: `max_tool_iterations` (agent) — tool-calling loop turns before an
`AgentLoopError` (default `-1` = no cap; set a positive int to bound it).

## Validate

```bash
ac run <flow>.yaml --input k=v          # loads, then runs (prompts for missing required inputs)
```
From Python: `load_flow(text, search_paths=[flow_dir])` loads + compiles without a
model. A flow with only `code`/`tool` nodes runs with no provider; any `agent` node
needs a provider/model configured.
