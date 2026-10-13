# Reference — composing flows

Deep-dive companion to [`SKILL.md`](SKILL.md). The skill is the authoring
workflow; this is the lookup sheet (operators, contexts, type forms, recipes,
gotchas) plus the [`templates/`](templates/) index. The authoritative grammar is
[`docs/syntax.md`](../../../docs/syntax.md); runnable end-to-end flows are in
[`examples/`](../../../examples).

Conceptual vocabulary: [`docs/agent.md`](../../../docs/agent.md).

## Templates (copy one as a starting point)

Each file in [`templates/`](templates/) is a minimal, loadable flow for one shape.
Copy, rename, and edit.

| Template | Shape it shows |
|----------|----------------|
| [`minimal.yaml`](templates/minimal.yaml) | one AGENT node, `str` in/out |
| [`compact.yaml`](templates/compact.yaml) | the same one-node flow in compact form (no `nodes:` map) |
| [`pipeline.yaml`](templates/pipeline.yaml) | AGENT node → CODE node (typed record) — deterministic post-processing |
| [`inline-code.yaml`](templates/inline-code.yaml) | inline `code:` source — a bare body run in-process (no `module:function`) |
| [`typed_output.yaml`](templates/typed_output.yaml) | AGENT node with a record `output:` — structured generation + `retries:` |
| [`branching.yaml`](templates/branching.yaml) | classify → `case` route → `\|` join |
| [`tool-use.yaml`](templates/tool-use.yaml) | a TOOL node (no LLM) feeding an AGENT node |
| [`human-in-loop.yaml`](templates/human-in-loop.yaml) | a blocking `human_input` gate (typed approve/revise answer) |
| [`human-questions.yaml`](templates/human-questions.yaml) | a `human_input` gate with multiple questions — static `questions:` + `adaptive_questions:` |
| [`wait-until.yaml`](templates/wait-until.yaml) | an absolute UTC WAIT ordered before downstream CODE |
| [`child-summarize.yaml`](templates/child-summarize.yaml) | a reusable CHILD flow |
| [`call-child.yaml`](templates/call-child.yaml) | `call` a sibling flow once (via `uses:`) |
| [`expr-and-call.yaml`](templates/expr-and-call.yaml) | `${a + b}` arithmetic in a binding + inline `call(...)` over an in-file `defs:` callee |
| [`map-fanout.yaml`](templates/map-fanout.yaml) | `map` a child over a list and collect in source order |
| [`inline-composites.yaml`](templates/inline-composites.yaml) | provider-free inline CALL, MAP, and LOOP child bodies (no `defs:`) |
| [`loop.yaml`](templates/loop.yaml) | `loop` a body until a predicate goes false — chat-shaped blocking turn |
| [`chat.yaml`](templates/chat.yaml) | a conversational REPL — HUMAN_INPUT → exit CASE → AGENT/CODE nodes, with flow-owned exit, the shape `yian chat` runs |
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

`and`, `or`, `not`, and `in` are reserved operators, so none can be the root
name of a reference. A dotted field may use those names (`${record.output.in}`).

Nesting a ref is allowed: `${a:-${b:-"lit"}}`. The `:-`/`:?` RHS is an
expression, so a string default must be **quoted** (`${x:-"today"}`; bare `today`
reads a variable). A whole-string `${ref}` resolves to the **typed value**;
embedded in surrounding text it is **stringified**. A child-flow call is the
whole-value `call(...)` directive (below), never a `${flow(args)}` span.

Child-flow output typing retains an exact or partial direct-reference contract, projects
known partial-record fields exactly, and proves comparisons, membership, and
`not`/`and`/`or` as `bool`. Unsupported computed
results stay unknown rather than inheriting one referenced operand's type. Each
declared output retains optional field-local evidence, so an opaque field is never
encoded as actual `None`, MAP can distinguish an opaque result from no declaration,
and LOOP still checks proven siblings. A `${null}` source may fill a nullable sink.
For LOOP, those proven siblings and the carried field-name set remain enforced even
when `while:` stops before the first body run or a durable run is replayed. A field
the LOOP body omits takes its type from its seed binding.

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
every authored node `input:` mapping participates, including HUMAN_INPUT context
and LOOP carried-record seeds. Nesting is allowed (`call(a, x=call(b, y=1))`).
It is recognized **only** as the entire trimmed value. The old `${flow(args)}`
(a flow call inside braces) is a load error: hoist an embedded flow call to a named
node; split a coalesce-of-calls into
per-node outputs. Pure builtins (`${upper(x)}`) stay legal inside `${}`.

This is distinct from an **inline child body**. `call(f, ...)` invokes an existing
named child flow and synthesizes the CALL node around that invocation. An explicit
`kind: call`/`map`/`loop` node with `nodes:` defines a one-off child-flow body on
the node itself.

## Inline CALL / MAP / LOOP child bodies

A composite node may replace its named `call:` target with a private body:

```yaml
each:
  kind: map
  over: ${input.rows}
  input:
    row: ${item}
    name: ${item.name}
  nodes:
    collect:
      kind: code
      input:
        row: ${input.row}
        name: ${input.name}
      output:
        row: Row
        name: str
      code: |
        return inputs
  output: ${collect.output}
```

Contract:

- Exactly one of named `call:` or inline `nodes:` is required. Inline `nodes:`
  is non-empty and requires the `output:` key; a named call must not also declare
  that child output. CALL may set it to null and return `None`; MAP/LOOP require
  their usual output shape.
- Outer `input:` keys generate the private child's parameters. The child cannot
  capture parent inputs/nodes or `${item}`; pass them explicitly through outer
  `input:`. Its own inputs/nodes and the run-global `${system.today}`,
  `${system.now}`, and `${system.run_id}` remain available.
- Generated parameter types are strict structural proofs. YAML
  null/bool/int/float/string values, closed records, non-empty homogeneous lists
  (with int/float widening), and
  lone direct references retain exact structure. Record field walks require
  proven fields and carry any root/intermediate nullability into the result.
  An optional proven element `Type` admits actual null list items; a specialized
  list keeps its storage discriminator and round-trips those items as `None`.
  Cross-flow bindings accept strict-to-nullable elements but reject the reverse.
  Unsupported whole-value computations, empty/heterogeneous lists, producers
  without an available static output type, or incomplete list element types fail
  at the authored binding; use a typed named `defs:` body as the escape hatch.
- MAP's `${item...}` provider is demand-driven. If no outer argument is a lone
  whole-value item reference, `over:` need not be statically inspectable. The
  first such item read proves `over:` is a list with an exact element type;
  later reads reuse that proof.
- Only a string at the outer binding value is scanned as a template. Strings
  nested inside a literal mapping/list remain literal, including `${...}` and
  `$$` text.
- Nesting is recursive. Every source occurrence has a distinct private child,
  including YAML alias/merge occurrences.
- Existing fields do not change ownership: loop drivers configure the outer
  composite; `env:` remains node-local; CALL/LOOP asserts keep their existing
  behavior and MAP asserts stay invalid. `parallel:` remains an accepted outer
  MAP field but is currently inert; run-level workers control concurrency.
  There is no inline child flow-level `llm_config:`/`env:`/`asserts:` block, and
  `llm_config:` remains invalid on composite nodes. The enclosing flow's LLM
  cascade reaches body AGENT nodes exactly as for an unconfigured named child;
  configure an individual AGENT node or factor a `defs:` child when it needs its own
  supported flow-level surface.

## Type forms

Python typing vocabulary. Scalars: `str`, `int`, `float`, `bool`, `date`,
`datetime`, `object`, `None`.

| Form | Example | Notes |
|------|---------|-------|
| list | `list[str]` | |
| nullable | `Optional[str]` | may be `null` |
| default-fill | `lookback: int = 30` | filled when the input is omitted |
| enum | `Literal[go, no_go, wait]` | one of these tags |
| bounded integer | `int[1, 12]` | inclusive integer endpoints |
| bounded float | `float[0.0, 1.0]` | inclusive finite float endpoints |
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

An explicit `Optional[int] = null` is a present default, unlike no `=`. A null
default requires a nullable or `None` type, and an empty `=` is invalid. Defaults
are YAML literals validated at load and are never string-coerced: use `int = 30`,
not `int = "30"`, and quote date strings (`date = "2026-01-01"`). Only supplied
boundary strings may be coerced. A compatible default keeps its authored payload,
so `float = 30` remains an integer.

Bounds require two ordered numeric literals and exclude booleans. Integer bounds
retain arbitrary precision. Float bounds must be finite; integer endpoints used
in a float domain must convert without rounding. Bounds compose with other
type forms and constrain defaults, supplied values, and generated schemas.
Cross-flow wiring requires the source domain to fit inside the destination;
an unbounded source does not prove a bounded destination safe.

## Recipes

**Give an AGENT a local prompt setting.** Add `variables:` beside `input:`, with
an entry such as `num_sentences: int[1, 12] = 4`, then use `${num_sentences}` in
the prompt. Each variable requires a validated typed default; nullable variables
also need an explicit default. Variable and input names cannot overlap. Default
strings stay literal, and each invocation gets a fresh copy of all defaults.
Ownership follows the authored AGENT path plus name across repeated executions.
A variable named `temperature` does not configure the model; use `llm_config:`.

**Get a structured or numeric value out of an AGENT node.** Declare the typed shape directly
as the node's `output:` — a `Literal[...]`, record, non-string scalar, or list switches
the AGENT node to **structured generation** (the engine derives a schema and the model emits a
conforming value, retried up to `retries:` times on deviation; see `typed_output.yaml`).
Use a downstream `code` node only when you need deterministic post-processing of the
value.

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
that value is committed as its ordinary node output. See `wait-until.yaml`.

**Ask the human.** For a *guaranteed* gate use a `human_input` node (it blocks its
worker until the host answers). For "ask only if the model decides it needs to",
give an AGENT node `tools: [ask_user]`. Both calls stay inside their live
node execution. A `human_input` gate can present multiple-choice questions
(static `questions:`), have an LLM compose them
(`adaptive_questions:`, which desugars to a synthetic composer AGENT node + gate at load), or read
them from an upstream node (`questions: ${name}`); the answer is then a record
keyed by each question's `header`. The adaptive desugar owns the `__questions`
input name, so authors cannot bind that name on an adaptive gate.

These are live authoring surfaces: the AGENT node has no authored `entry:`/`controls:`
fields, HUMAN_INPUT has no control fields, and WAIT has no event-watch form. Each
completes through the ordinary node-output path in the active run.

**Reuse a child flow.** Factor the repeated work into its own flow file, bind it with
`uses: <alias>: <filename>`, and `call:` it (once) or `map:` it (per list element).
A CALL record supports `${node.output.field}`; MAP returns a list at
`${node.output}`, so project fields in a downstream CODE/MAP step. If the body is
one-off, keep the CALL/MAP/LOOP node and put its closed `nodes:` + `output:` body
there instead; do not create a throwaway `defs:` entry.

**Loop until done.** Use a `loop` node to re-run a body under one of three drivers,
threading a carried record `'a -> 'a` (the body's `output:` returns a subset of
the carried fields; each omitted field keeps its value). The `input:` is the SEED
carried record. Pick EXACTLY ONE driver:
`while: not done` is a PRE-check over the carried record (0+ runs, stop when the
predicate goes false); `until: done` is a POST-check / do-while (1+ runs, stop
when the predicate becomes TRUE); `times: N` runs exactly N times with no
predicate. Predicates use the canonical bare form (no `${}`). `max:` is an optional runaway guard
for `while:`/`until:` (without it the loop runs until its predicate ends it) — but
REDUNDANT and REJECTED with `times:`. A body that blocks
(a `human_input` leaf) makes the loop a live chat REPL — the same run threads each
answered turn. Each continuing iteration PRUNES the previous iteration's body nodes and
the nodes grown under them by a nested CALL, MAP, or LOOP, so the live graph stays about
one iteration, whatever the body, unless a nested node is still unfinished (that subtree
stays live); the run log still keeps every iteration. The node budget (10,000 live nodes)
is checked before pruning; it limits one iteration's size, not the iteration count. See
`loop.yaml`.

**Conversational REPL (LOOP + human_input).** A chat is a `loop` whose body blocks at
`ask` (`human_input`), then a CASE routes exact `/exit` and `/quit` values to an exit
record before `reply` (`agent`). The normal branch appends the message and reply to the
transcript. Output bindings rejoin those branches into the carried 2-field record
`{transcript: str, exited: bool}` (`'a -> 'a`). Typed exit is flow-owned; only terminal
EOF is normalized by the bundled host. The loop body's `output:` may return only the
fields that change; a field it omits keeps its value. A lone multi-ref coalesce has no
inferable type, so the template uses adjacent defaulted spans for the `str` field and
compares the optional exit field with `true` to prove `bool`. This is the shape
`yian chat` runs; see `chat.yaml` (and `examples/chat.yaml` for a runnable copy).

## Gotchas

- **Inline `{ ... }` maps + `${...}` need quotes.** In an inline flow mapping the
  `}` in `${input.x}` closes the map early. Either quote the value
  (`input: {x: "${input.x}"}`) or use block form (preferred):
  ```yaml
  input:
    x: ${input.x}
  ```
- **An AGENT node's `output:` may be any shape** — only a bare, untagged `str` keeps it a text
  producer; a `Literal[...]`/record/non-string scalar/list switches it to structured
  generation (schema-checked at the write boundary, `retries:`-capped self-correction).
- **Prompt names are local.** Inside `prompt:` use names from the node's own
  `input:` block, written bare (`${name}`); AGENT prompts also see local `variables:`. Pool refs
  (`${input.x}`, `${other.output}`) go in the `input:` block first.
- **No `edges:` block, no per-node `id:`, no body wrappers.** `${...}` references
  infer data edges; `depends_on:` / `runs_after:` add ordering-only edges. A node
  body is flat.
- **`output: {...}` always returns a record.** `output: ${x}` returns the value
  as is; `output: {order: ${x}}` returns `{order: x}` (a caller reads
  `${child.output.order}`), and `output: {}` returns `{}`. No `output:` returns
  `None`.
- **`call:` resolves defs-first, else a `uses:` alias** (a sibling file by name on
  the search path). `alias@v1` adds a version guard.
- **No `kind: model`** — removed; serve ML inference behind a host tool and call it
  with `kind: tool`. Use `code` for deterministic compute.
- **`loop` bodies return a subset of the carried record.** The body's `output:`
  returns at least one of the loop's `input:` (carried record) fields, and no others;
  each omitted field keeps its previous value. The body reads only a subset of the
  carried names. Names are checked at build; each returned field must keep its carried
  type (checked at load). Omitted keeps a field; a returned null sets a nullable field
  to null. A literal `${null}` is typed `None`, so it does not fit an `Optional[str]`
  seed; return a nullable-typed producer instead. A body may return one field
  (`output: {done: ...}`); an empty `output: {}`, or a reference to an empty record,
  is rejected. A predicate that reads only fields the body never returns cannot
  change; give such a loop `max:`.
  Give EXACTLY ONE driver: `while:` (pre-check, 0+ runs), `until:` (post-check /
  do-while, 1+ runs), or `times: N` (fixed count). `max:` is optional for
  `while:`/`until:` but REDUNDANT+REJECTED with `times:`.
  Every `while:`/`until:` ref must name a carried field (a typo is rejected at load,
  not read as falsy), and `max:` (when given) and `times:` must be a plain integer `>= 1`.
- **Node-local `asserts:` reading `${output}` are POST checks** — they fire once the
  node's value is committed, and fail the run loudly on a false/raising expr. This
  includes a `call` node: its POST asserts may read `${output}` **and** the call's
  declared inputs (`${name}`), like a leaf node. `map` nodes reject node-local
  `asserts:` at load time — assert a `map`'s result with a flow-level/downstream check.
- **Flow-level `asserts:` read `input.X` (boundary, before any node) or `node.output`
  (post, after the terminal).** An exact unqualified input name is shorthand (`age`
  means `input.age`) and takes precedence over a same-named node; dotted input reads
  remain explicit (`input.profile.age`). `${system.X}` is **not** allowed — it is
  ambient (no producer node to draw the check's dependency edge from), so a `system`
  ref is a LoadError; put a system-dependent check in a body node's `asserts:` or a `case`.
  A flow's `asserts:` are lowered onto its synthesized `start`/`end` boundary nodes and
  fire on the ordinary node path (`start` post-coercion, `end` post-terminal), so a
  called `defs:`/`uses:` child's asserts fire identically to the top-level flow's.

## Model selection — the `llm_config` cascade

Use `yian auth openai` to select subscription or API-key authentication. Both use
`provider: openai` and the same Yian execution and memory paths. A flow's
`llm_config:` cannot set `api_key` or `base_url`; a Python host passes the key in
the run's `llm_config` (it needs `provider`) and again to
`resume_flow`/`result.backward` as `api_key=key`. It is never recorded, and it
reaches every AGENT of that provider, `inherit: false` ones included. A
`base_url` beside the key (https, or http for local hosts) is where that key is
sent; the log records it on those agents and resume/backward reuse it.

Model fields resolve **per field, most-specific wins**. An AGENT node fills only the fields
it leaves unset from the layer outside it. Precedence (most specific first):

1. the AGENT node's own `llm_config:`
2. the enclosing flow's `llm_config:`, then each parent flow outward
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

Any node may carry an `env:` mapping of **static** config keys its own
implementation reads (the engine never interprets `env`). A flow-level `env:`
is the default for every node;
each node overrides individual keys. Merge is **static at build time** (`{**flow_env,
**node_env}`, node wins), **flow-local** (a child flow does NOT inherit the parent's
flow env), literals only (no `${...}`).

```yaml
env: {max_tool_iterations: 150}         # flow default for every node
nodes:
  r: {kind: agent, prompt: hi, tools: [search], env: {max_tool_iterations: 300}}  # node override
```

Keys in use: `max_tool_iterations` (AGENT node) — tool-calling loop turns before an
`AgentLoopError` (default `-1` = no cap; set a positive int to bound it).

## Agent memory — `long_term_memory:`

`long_term_memory:` is valid on a full flow (`nodes:` form, root file or `defs:`
entry) or an AGENT node. Each entry names one memory and its type:

```yaml
long_term_memory:                  # on a flow
  notes: episodic                  # short form: name: type
  lessons:                         # long form: only type and config
    type: reflective
    config: {max_items: 100, recall_limit: 12}
nodes:
  answer:
    kind: agent
    long_term_memory: {chat: session}
```

`long_term_memory: episodic` is sugar for `{episodic: episodic}`. `null` equals
omission. Names and types match `[A-Za-z_][A-Za-z0-9_-]*`; `name`, `type`, and
`config` are reserved. Built-in config keys: `episodic` `max_episodes`,
`recall_limit`; `reflective` `max_items`, `recall_limit`; `session`
`history_budget_chars`, `recent_turns`; `profile` `max_chars`,
`input_budget_chars`. Other types are custom and built by a Python host. The YAML
contains no credentials, storage locations, or Python imports.

A memory's key is its name prefixed by the `uses:` aliases on its import path
(`notes`, `lib.notes`, `b.d.notes`). One key declared at several sites of one file
is one memory; libraries and repeated aliases stay independent. An AGENT's stack is
each enclosing flow's keys, outermost first, through CALL, MAP, and LOOP, then its
own; each key appears once and contributes one recall section. MAP items and LOOP
iterations reuse that stack. A child's declaration never binds to a parent's
memory. Other node kinds and anonymous inline children cannot carry the
declaration.

Errors are reported at the declaration (a multi-site key names its first site and
lists the others as notes). At load: the old `{name: ..., config: ...}` form or
`{}`; a bad name or type name; unknown long-form keys or non-mapping config;
`storage`, `store`, or `path` as a config key; `api_key` anywhere; a built-in
config key outside the allowlist; one key with different type or config; two keys
equal under `casefold()`; a memory-bearing alias that does not match the name
pattern; a flow-level session; a session key with more than one site; an AGENT
with two sessions. `build_memories` adds: a custom type with no supplied value or
factory, `factories` naming a built-in type, a non-`Memory` value, config on a
supplied key, `storage` for a supplied or custom key, an undeclared key, a rejected
config, `root` without a safe flow `id`, and two built-in keys on one storage file.

`yian run` builds every built-in memory in RAM, or with `--memory-root DIR` at
`DIR/<flow id>/<scope...>/<name>.sqlite` (`.md` for profile), with repeatable
`--memory-storage KEY=PATH` overriding one key. It refuses a flow with a
custom type, even on `--resume`. `yian chat` and its `run_flow` tool supply no
long-term memory, so declared memories stay off and session conversations are not
kept. Use the same storage on unfinished resume; paths
never belong in YAML or run logs, and terminal inspection opens no backend. A
library host passes `memory=build_memories(flow, root=, storage=, supplied=,
factories=)` to `run_flow`; Python's `memory=None` remains dormant.
For API session history, declare one `session` memory on the AGENT. Its key has
exactly one declaring AGENT, whose call sites all share the conversation.
Repeated MAP/LOOP executions share it and queue in admission order, not guaranteed
MAP input order. Use non-session memory for independent MAP items. History advances
automatically during forward, independently of `learnable` or backward. Current
instructions and tool permissions still apply; restoring history never executes
old tool calls. Explicit API backward self-feedback can become committed notes
for the conversation through the key's learn/session commit/recall lifecycle,
without another model call or YAML selector. Notes are recalled from
the run-opening snapshot and do not rewrite raw conversation history. No-feedback
backward adds no session notes. Summary settings are `history_budget_chars=32_000`
and `recent_turns=4`, as constructor keywords or YAML config. The current
model summarizes older context without deleting raw history or truncating recent
whole turns. Budget is soft; oversized valid summaries are not retried or cut.
Episodic, reflective, and session backends optionally persist through host-only
`storage=path`, opened on first use; `None` remains process-local. SQLite
ownership lasts until backend close. Closing rejects active/queued work, but idle
run handles do not block it. Storage paths never belong in YAML.
`learnable=True` independently retains execution; forward runs no memory
learners and commits no learned mutations.
Call `result.backward(feedback)` explicitly after forward returns, and inspect
its separate outcome. Each memory key gets one learn call, even with `None`
feedback, without AGENT backward model work. Terminal resume is silent
inspection; each explicit `backward()` call is a new whole attempt, even after an
earlier attempt succeeded or committed. The CLI builds declared built-in
memory but has no backward command and never enables learning implicitly.

## Validate

```bash
yian run <flow>.yaml --inputs base.json --input k=v
```
Explicit JSON inputs load first; repeatable `--input` flags override individual
keys but remain a string/scalar surface. With an interactive stdin (TTY), `yian`
prompts for declarations still missing and keeps four outcomes distinct: an explicit
value, `Set null`, omit/use-default when allowed, and cancellation. Nullable
boolean/`Literal[...]` widgets put `Set null` in their value select. A null-capable
open scalar/list/object first offers omit/default when allowed, `Set null`, and
`Enter value`; Enter opens the existing scalar or typed-JSON editor. Optional blank
there returns to the palette instead of omitting. A direct non-null-capable
optional/defaulted editor still blanks to omission. The null-only `None` type offers
`Set null` and no Enter action.

`Set null` supplies a present `None`; text `null` for a string input and a `Literal`
tag named `null` remain ordinary strings. The structured editor rejects root JSON
`null`, so choose `Set null`; nested null remains valid where the nested contract
accepts it. Invalid JSON or values stay in the editor for correction. Cancelling the
palette or editor cancels the run. Without a TTY, missing optional/defaulted
declarations remain absent, and missing required declarations fail with exit code `2`
before setup. Supply explicit null and native structures through `--inputs`; `--input`
remains a string/scalar surface.

From Python: `load_flow(text, search_paths=[flow_dir])` loads + compiles without a
model. A flow with only `code`/`tool` nodes runs with no provider; any AGENT node
needs a provider/model configured.
