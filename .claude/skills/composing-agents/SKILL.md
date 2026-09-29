---
name: composing-agents
description: Use when authoring an Agent Composer flow — writing or editing a flow YAML, building an agent workflow, wiring nodes (agent / code / tool / case / human_input / wait / call / map), or answering "how do I write a flow that …". Trigger on "create an agent", "write a flow", "compose a workflow", "add a node to my flow", "how do I branch / loop / call another flow", or any `composer run` flow-authoring request.
---

# Composing an agent flow

In Agent Composer, an **agent** is a composable typed transformation represented
by a node, a flow, or a child flow. A **flow is a function**: a typed `input:`, a
graph of `nodes:`, and a typed `output:`. The concrete **AGENT node**
(`kind: agent`) is the LLM-backed leaf. `${...}` references infer data edges;
`depends_on:` and `runs_after:` add ordering-only edges. The human owns the
structure; the LLM only fills the AGENT node leaves.

Conceptual vocabulary: [`docs/agent.md`](../../../docs/agent.md). Full authoring
reference: [`docs/syntax.md`](../../../docs/syntax.md). Working flows:
[`examples/`](../../../examples). This skill is the authoring workflow + the
parts that bite.

**Companions:** [`reference.md`](reference.md) — operators, the three expression
contexts, type forms, recipes, and gotchas. [`templates/`](templates/) — minimal
loadable starter flows for each shape (single AGENT node, AGENT node → CODE node pipeline,
branch+join, tool use, human gate, named and inline `call`/`map`/`loop`
composition); copy one and edit.

## The workflow — author a flow in order

1. **State the function.** Write the one-line job, then its **signature**: what
   typed values come in (`input:`) and what comes out (`output:`). If you can't
   name the input and output types, the design isn't done.
2. **Name reusable types** in `typedefs:` (enums, records, aliases).
3. **Decompose into leaf nodes.** Each node is one step: an LLM call (`agent`),
   deterministic code (`code`), a registered tool (`tool`), a branch (`case`), a
   blocking human gate (`human_input`), a blocking UTC deadline (`wait`), a child-flow call
   (`call`/`map`), or an iterated body (`loop`). A reusable child lives in
   `defs:`/`uses:`; a one-off child may put `nodes:` + `output:` directly on
   CALL/MAP/LOOP. Prefer many small typed nodes over one mega-prompt.
4. **Wire by reference, not by edges.** A node consumes another by putting
   `${producer.output}` in its `input:` block. That reference *is* the edge.
5. **Bind the flow `output:`** to the terminal node(s).
6. **Add `asserts:`** for invariants that must hold (boundary or post checks).
7. **Validate, then run** (see "Validate & run").

## The flow skeleton

```yaml
id: my-flow                  # stable identifier (no spaces)
name: my_flow                # display name
description: One line.       # optional

typedefs:                    # optional — named/composed types
  Verdict: Literal[go, no_go, wait]

input:                       # the parameters — typed
  question: str
  audience: Optional[str]    # nullable; omitted -> null
  lookback: int = 30         # default-fill when omitted

nodes:                       # the body — a map keyed by node id (no per-node id:, no wrappers)
  step_a:
    kind: agent
    input: { question: ${input.question} }
    output: str
    prompt: "Answer: ${question}"

output: ${step_a.output}     # one value, OR a multi-field object (see below)

asserts:                     # optional — boolean invariants
  - step_a.output != ""
```

There is **no `edges:` block**, no `__start__`/`__end__`, no per-node `id:`, and
no body wrappers — a node body is flat.

Node IDs are the keys under `nodes:`. They cannot contain `.`, `#`, or `/`:
`.` separates the node name from `output` and its field path in a reference,
while `#` and `/` are reserved for generated runtime identities. In compact
mode, the flow `id` also names its node and follows this rule.

### Compact mode — when the flow is a single node

For the "one flow, one node" case, drop the `nodes:` map and the redundant
`output: ${node.output}` line — put the node `kind:` and its fields at the top
level. The flow `id:` names the node; the flow `input:` is the node signature
(auto-wired by name); the flow `output:` is the node's output type, re-exported.

```yaml
id: hello
name: hello
input:
  name: str                  # auto-wired into the node as ${input.name}
output: str                  # the node's output TYPE — also the flow output
kind: agent
prompt: "Write a warm one-sentence greeting to ${name}."
```

Allowed only for the **leaf kinds** (`agent`, `code`, `model`, `tool`,
`human_input`) — `case`/`call`/`map`/`loop` need the full `nodes:` form. See
[`templates/compact.yaml`](templates/compact.yaml).

## References — how you wire values

| Write | Means |
|-------|-------|
| `${input.X}` | field `X` of the flow input |
| `${node.output}` | node `node`'s whole (typed) value |
| `${node.output.field}` | dot into an object value |
| `${name}` (bare) | a local input inside `prompt:` or `case` `when:`; AGENT prompts also see their own `variables:` |

A whole-string `${ref}` resolves to the **typed value**; embedded in surrounding
text it is **stringified**.

Prompt templates are strict: a missing reference or any span whose final value is
null raises, including an embedded span. Binding templates stay non-strict—a whole
null remains typed null and an embedded null contributes empty text.

**Operators inside `${...}`** — one expression grammar, everywhere:

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
reads a variable). A child-flow call is the whole-value `call(...)` directive
(see below), never a `${flow(args)}` span.

For child-flow output contracts, a direct ref preserves its exact or partial contract;
selecting a known partial-record field yields an exact type. A comparison/membership or
`not`/`and`/`or` result is proven `bool`. This permits a
LOOP body field such as `exited: ${ask.output == "/exit"}` without a CODE fold.
Other computed result types remain unknown unless an explicit typed node supplies them.
In LOOP records, an unknown field does not disable enforcement of proven siblings,
including when `while:` stops before the first iteration or a run is replayed.

## Types

Python typing vocabulary. Scalars: `str`, `int`, `float`, `bool`, `date`,
`datetime`, `object`, `None`. Forms: `list[X]`, `Optional[X]`, `Literal[...]` (an
enum of tags), and named **records**. `Optional[X]` (nullable) and `= default`
(omission-fill) are orthogonal.

Use `int[1, 12]` or `float[0.0, 1.0]` for inclusive numeric bounds. Supply two
ordered literal endpoints. Integer bounds may be arbitrary precision; float
bounds must be finite, and integer endpoints in a float domain must convert
without rounding. Booleans are invalid.
Bounds compose with lists, records, aliases, and nullable types. Defaults and
supplied values are checked; cross-flow sources must fit the destination domain.

An explicit `Optional[int] = null` is a present default, distinct from no `=`;
null is invalid for a non-nullable type, and an empty `=` is invalid. The default
RHS is a YAML literal validated at load and never string-coerced: write
`int = 30`, not `int = "30"`, and quote date strings (`date = "2026-01-01"`).
Only values supplied at a flow boundary may undergo string coercion. Compatible
defaults keep their authored payload, so `float = 30` remains an integer.

```yaml
typedefs:
  Basket: list[str]                    # alias (aliases compose)
  Decision: Literal[go, no_go, wait]   # enum — one of these tags
  Signal:                              # record — fields recurse
    score: float
    note: Optional[str]
```

## Node kinds (the closed set — you compose these, you don't define new ones)

### `agent` — the LLM-backed AGENT node
```yaml
classify:
  kind: agent
  mode: plain                  # "plain" = one call; "tool_calling" (default) = a tool loop
  input:
    text: ${input.text}        # bindings — these infer the data edges
  output: Literal[positive, neutral, negative]
  prompt: |-
    Classify the sentiment of: ${text}
    Answer with exactly one of: positive, neutral, negative.
```
**An AGENT node's `output:` may be any shape.** Only a bare, untagged `str` keeps the
node a text producer. A `Literal[...]` enum, record, non-string scalar, or list
switches it to **structured generation** — the engine derives a schema from
`output:` and the model emits a conforming value (native structured output, or a
JSON fallback for providers that lack it), validated at the write boundary and
retried up to `retries:` times (default 2) on deviation. A root `Literal[...]`
returns its bare tag to the flow.
Use a downstream `code`/`model` node only for deterministic post-processing. To give
an AGENT node ordinary tools: `tools: [tool_id, ...]` (requires `mode: tool_calling`).
A node may pin its model via `llm_config:`; otherwise the process defaults captured
once when the run initializes apply. `inherit: false` skips authored flow/parent/CLI
layers but still receives those provider/model defaults.
OpenAI subscription and API-key connections both use `provider: openai`;
select the method and default model with `composer auth openai`.
OpenAI is the default provider. Environment defaults override saved defaults;
model defaults apply only to their matching provider. Grant ordinary Composer
tools explicitly; retired `provider: codex`, `sandbox`, `cwd`, and `builtin_tools`
settings are rejected.
Tune a node's non-model behavior with a static `env:` block (flow-level default +
per-node override, node wins): e.g. `env: {max_tool_iterations: 300}` bounds an
AGENT's tool-calling loop (default `-1` = no cap). `retries` bounds typed-output
correction, not tool, transport, authentication, or stop failures.

### Agent-owned variables — `variables:`

Use typed defaults for local prompt settings, alongside the AGENT node's inputs:

```yaml
writer:
  kind: agent
  variables:
    num_sentences: int[1, 12] = 4
    tone: Literal[concise, detailed, technical] = concise
  input:
    topic: ${input.topic}
  prompt: Write ${num_sentences} ${tone} sentences about ${topic}.
  output: str
```

Every variable needs an explicit, valid default, including nullable variables.
Names cannot overlap inputs. Defaults are literal values; `${...}` inside a
default string stays text. Each invocation receives a fresh copy and repeated
prompt references read the same local value. Ownership follows the authored
AGENT path plus name across MAP items and LOOP iterations. These declarations
create no data edges. Model settings still belong in `llm_config:` even when a
variable has the same name, such as `temperature`.

### Agent memory — `long_term_memory:`

Put `long_term_memory: NAME` on a full flow (`nodes:` form) or an AGENT node to
declare long-term memory and select a named host policy. The object form is
`long_term_memory: {name: NAME, config: {...}}`; config is public behavior only,
so never author credentials, tenant/backend locations, or Python imports there.
Flow frames are inherited by every AGENT node in that flow, and an AGENT node
may add its own local frame. Each frame writes only itself. MAP/LOOP executions
reuse the frames authored on their flows and AGENT nodes rather than creating
per-item/per-iteration frames.
Other node kinds and anonymous inline child flows cannot carry this declaration.
`composer run` automatically hosts declared `episodic`, `reflective`, and `session`
policies; custom policies, including `ProfileMemory`, require Python injection.
Library hosts enable recall
with `run_flow(..., memory={NAME: implementation})`; `memory=None` stays dormant.
For API conversation continuity, declare `long_term_memory: session`; Python hosts
inject `SessionMemory()` as `"session"`;
frame config is empty. All policies contribute through one memory context:
SessionMemory supplies a structured transcript, while ordinary memories supply
notes/instructions. Preserve transcript roles and tool results; do not flatten
history into a note. The innermost frame contributing a transcript owns one
conversation per authored AGENT leaf; other frames still provide recall. Repeated
calls queue, including MAP items and LOOP iterations. MAP does not fork history
or promise input-index admission order; use non-session memory for independent
items. Session history updates automatically during forward, without learning.
Explicit API backward self-feedback can become committed SessionMemory notes for
future calls through the same policy, without another model call or YAML option.
Notes stay scoped to the owning frame and authored leaf; recall is pinned when
the run opens. Host-only `SessionMemory(history_budget_chars=32_000, recent_turns=4)`
controls API summaries of older context; do not put these settings in YAML.
Raw history and recent whole turns stay intact, even above the soft budget.
Episodic, reflective, and session backends accept optional `storage=path` for local
SQLite durability; the default `None` remains process-local. Never put a storage
path in YAML. A file has one owner until backend close; active work or queued
turns block close, idle run handles do not. In `composer run`, repeat
`--memory-storage POLICY=PATH` to persist selected declared policies; others stay
process-local and lose state on exit. Supply the same stores on unfinished
resume; terminal inspection opens none. CLI activation does not run backward
learning.
Independently, `learnable=True` retains
execution for a later explicit `result.backward(feedback)` call; forward returns
before frame learning or learned-mutation commits. `result.backward()` with no external feedback still
runs every enabled frame learner, while unreached AGENT backward model calls are
skipped. The separate outcome never changes forward values or updates local
variables. Terminal `resume_flow` inspects without learning; it does not retry a
backward attempt. See the [Python API](../../../docs/api.md#explicit-backward-feedback)
for feedback shapes and repeated backward attempts.

### `code` — deterministic Python
`code:` is either a **`module:function`** reference (import + call) or **inline source** —
a bare body that reads the `inputs` dict and `return`s a value (the engine wraps it as
`def main(inputs)` and runs it in-process; same one-dict convention, so a body promotes to
a reference by copy-paste). See [`templates/inline-code.yaml`](templates/inline-code.yaml).
```yaml
verdict:
  kind: code
  input:
    s: ${score.output.signal}
  output: Signal               # one typed value; an AGENT node supports this too
  code: pkg.mod:fn             # a reference — OR an inline `code: |` body that returns
```

### `tool` — a registered tool (no LLM)
```yaml
news:
  kind: tool
  tool_id: get_facts           # a registered or per-run tool id
  input:                       # untyped — each value is a ${...} ref or a literal
    symbol: ${input.topic}
    limit: 10
```
Register the tool in Python so the id resolves:
```python
from agent_composer.tools import register_tool

@register_tool("get_facts")
def get_facts(symbol: str, limit: int = 10) -> list[dict]:
    "Fetch recent facts for a symbol."   # docstring = the description the model sees
    ...
```
Or give it to one run only: `run_flow(flow, inputs, tools={"get_facts": get_facts})`.
That mapping is the run's whole tool set: registered tools are hidden. To keep
them, pass `tools={**registered_tools(), "get_facts": get_facts}`.
Note `tool` uses `tool_id:` + `input:` (a tool declares no `output:`). (`model` —
`kind: model`, `model_id:` — is a parseable kind but the serving seam isn't wired
yet; running one raises.)

### `case` — branching (routes only; no `input:`)
Exactly one branch runs; the others skip (their refs resolve to null). **Join the
branches back with a `|` coalesce.**
```yaml
route:
  kind: case
  on: ${synth.output}          # simple form: match a Literal value
  cases:
    - when: go
      then: go_brief
    - when: no_go
      then: no_go_brief
  else: more_info_brief        # required unless the Literal cases are exhaustive
```
Searched form (first true `when:` wins, no `on:`): `when: "score.output >= 0.5"`.
Routing on a `Literal` is **exhaustiveness-checked** — omitting a tag with no
`else:` is a compile error.

### `human_input` — a guaranteed human gate
**Always** calls the run's `ask_user` handler and blocks its worker until the
host returns a typed answer (validated against `output:`). It stays inside the
live node call and returns the answer as an ordinary node output.
```yaml
approve:
  kind: human_input
  input: { plan: ${draft.output} }    # context the prompt may reference (bare ${plan})
  prompt: |-
    Here is the plan:

    ${plan}

    Approve as-is, or revise? (approve / revise)
  output: Approval                    # a typed answer, e.g. a Literal enum
```
Instead of a typed `prompt:`+`output:`, a gate can present **questions** (1–4,
AskUserQuestion-shaped) — the answer is a record keyed by each `header`; `output:`
then defaults to `object`. `prompt:` is optional; `questions:` and
`adaptive_questions:` are mutually exclusive. The gate itself never chooses or
calls an LLM: `adaptive_questions:` is LOAD-TIME sugar that lowers to a
synthetic composer AGENT node + gate; the gate only calls the host's installed
interaction tool.
```yaml
ask:                                  # static questions list
  kind: human_input
  input: { proj: ${input.proj} }      # ${proj} renders into question text
  questions:
    - question: "Which framework for ${proj}?"
      header: Framework               # answer key -> {Framework: <label>}
      options: [ {label: React, description: a lib}, {label: Vue, description: a fw} ]
    - { question: "Any notes?", header: Notes }   # no options -> free-text
ask2:                                 # LLM-composed questions (desugared at load)
  kind: human_input
  input: { ctx: ${research.output} }
  adaptive_questions: { prompt: "Design 1-3 questions for: ${ctx}", retries: 3 }
```
(On an adaptive gate, `__questions` is reserved for the generated composer wire;
do not declare it in `input:`.)
Questions may also come from an upstream node: `questions: ${qs}`.

### `wait` — block until an absolute UTC deadline
```yaml
settle:
  kind: wait
  until: ${input.as_of}
  # order a downstream node after it with depends_on: [settle]
```
`until:` resolves to an ISO `date` or `datetime`. Date and naive datetime values
are UTC; aware values are normalized to UTC. A past/current deadline returns
immediately. Otherwise WAIT holds one worker until the remaining wall-clock
duration elapses, then returns `None` as an ordinary node output.

### `call` / `map` — composition over child flows
A child is itself a flow. `call` runs it once; `map` runs it per list element
(`${item}` is the current element). Name a reusable child with `call:`:
```yaml
uses:
  research-one: 03-research-one   # bind an external sibling flow to a local alias

nodes:
  each:
    kind: map
    over: ${input.topics}         # a list[T]
    call: research-one            # node value is list[U]
    input: { topic: "${item}" }   # per-element call args (sink-bind the child's params)
```
`call:` resolves **defs-first, else a `uses:` alias** (an external flow on the
search path, by filename; `alias@v1` adds a version guard). A CALL's record fields
are available as `${one.output.field}`; a MAP returns a list at `${each.output}`,
so project element fields in a downstream CODE/MAP step. `parallel:` is accepted
but currently inert; run-level `num_workers` controls MAP concurrency.

For a one-off child, replace `call:` with a non-empty `nodes:` body and add the
required `output:` key:

```yaml
each:
  kind: map
  over: ${input.topics}
  input:
    topic: ${item}                # generates child parameter `topic`
  nodes:
    normalize:
      kind: code
      input:
        topic: ${input.topic}
      output: str
      code: |
        return inputs["topic"].strip()
  output: ${normalize.output}
```

Use exactly one of `call:` or `nodes:`. The `output:` key is required; CALL may
set it to null and return `None`, while MAP/LOOP require their usual output
shape. The inline child cannot capture the parent pool: pass parent values and
MAP's item through the outer `input:`, then read generated `${input.X}`
parameters and child-node outputs inside. Run-global `${system.today}`,
`${system.now}`, and `${system.run_id}` remain available. Types are inferred
strictly from YAML null/bool/int/float/string values, structural containers, or
direct typed refs; refs preserve structure/nullability, and only a lone
whole-value `${item...}` lazily requires a proven `over:` element type. An
optional element admits actual null list items; a specialized list keeps its
storage kind and preserves `None` through lossless round-trips. Cross-flow
bindings accept strict-to-nullable elements but reject nullable-to-strict. If
inference cannot prove a type, write a named `defs:` child with an explicit
signature. Strings nested in literal containers stay literal and are not
reference-scanned. Other
fields keep their outer-node meaning (LOOP drivers, `env:`, and CALL/LOOP
asserts; MAP asserts remain invalid); `parallel:` remains accepted but inert.
Inline child flow configuration is not a separate surface, and `llm_config:`
remains invalid on composite nodes. The enclosing flow's LLM cascade still
reaches body AGENT nodes. See
[`templates/inline-composites.yaml`](templates/inline-composites.yaml).

### `loop` — iterate a body until a predicate goes false (the engine's `while`)
Threads ONE carried record from each turn into the next: the body maps the carried
record to the NEXT one (`'a -> 'a`), and the loop re-runs while `while:` holds.
```yaml
turn:
  kind: loop
  call: chat_turn              # the body (a def or a `uses:` file)
  input:                       # the SEED carried record — defines 'a
    messages: []
    exited: false
  while: not ${exited}         # pre-check over the carried record; `not` OUTSIDE ${...}
  max: 100                     # REQUIRED runaway guard (LoopMaxExceeded)
```
`loop` accepts the same inline `nodes:` + `output:` form instead of `call:`; its
generated parameters are the carried `input:` fields, so the same `'a -> 'a`
contract still applies.
The body's `output:` shape must **equal** the carried record and read only a
**subset** of its names (`'a -> 'a`) — names are checked at build, and statically
proven body/seed field pairs are compared at load. Proven output fields are always
enforced at runtime and replay. The
node's value is the final carried record. A body that blocks (a `human_input` leaf)
makes the loop a live chat REPL: the same run threads each answered turn. Pick
**exactly one**
driver: `while:` (pre-check, 0+ runs), `until:` (post-check / do-while, 1+ runs,
stop when the predicate becomes true), or `times: N` (fixed count). `max:` is
required for `while:`/`until:` but **redundant and rejected** with `times:`.

#### Conversational REPL — the `loop` shape used by `composer chat`
A chat is a `loop` whose body blocks each turn at `ask` (`human_input`), then a
`case` routes exact `/exit` and `/quit` values to an exit record before the
`reply` AGENT node. The normal branch builds the next transcript; the body's output
bindings rejoin the exit record or normal transcript into the carried
`{transcript: str, exited: bool}` record. Typed exit is flow-owned — a host must
not intercept it — while terminal EOF may be normalized to `/exit`. **Gotcha:**
the body's `output:` must EQUAL the carried record. A lone multi-ref coalesce has
no inferable type; adjacent defaulted spans preserve `str`, and comparing the
optional exit field with `true` proves the carried `bool`. See
[`templates/chat.yaml`](templates/chat.yaml).

## Run-ordering without data (`depends_on` / `runs_after`)
Both gate a node on another **settling** even when no value flows. `depends_on:
[x]` also **co-skips** the dependent if `x` skipped; `runs_after: [x]` orders only
(the dependent still runs). Use these for `wait`, side-effecting tools, etc.

## The three expression contexts (one grammar)
One `${...}` grammar; the context decides what happens to the result:
- **Bindings** (`input:`/`output:` values): **evaluated** to a typed value — refs,
  literals, arithmetic (`+ - * / % **`), list literals, comparisons, `:-`/`:?`,
  `|`, and pure builtins. A child-flow call is the whole-value `call(...)`
  directive, not a `${...}` span.
- **`when:` / `asserts:`**: **tested** as a boolean — `== != < <= > >=`,
  `in`/`not in`, `and`/`or`/`not`, parens, arithmetic operands. Bare or
  `${...}`-wrapped are equivalent. A flow-level `asserts:` reads `input.X`
  (boundary, before any node) or `node.output` (post, after the terminal). An
  exact unqualified flow-input name is shorthand (`age` means `input.age`) and
  wins over a same-named node; keep fields explicit as `input.profile.age`.
  `${system.X}` is **not** allowed there (it is ambient; a `system` ref is a
  LoadError — move the check into a body node).
- **Prompts**: free text with embedded `${...}` spans (each stringified).

> Bindings wire, conditions test, nodes compute.

## Inline `call(...)` — a flow call as a binding's whole value
`call(f, arg=${ref}, k=lit)` written as a binding's **whole value** (or a `case`
`then:`/`else:` target) desugars at load into an anonymous `call` node; the host
becomes `${<synth>.output}`. Every authored node `input:` mapping participates,
including HUMAN_INPUT context and LOOP carried-record seeds. Keyword args only;
nesting is allowed (`call(a, x=call(b, y=1))`). It is recognized **only** as the
entire trimmed value — a `call(` mid-value or inside `${...}` is literal text. The old
`${flow(args)}` form (a flow call *inside* braces) is a **load error**: hoist an
embedded flow call to a named `call` node, and split a coalesce-of-calls
(`${a() | b()}`) into per-node outputs. Pure builtins (`${upper(x)}`) stay legal
inside `${}`.

Do not confuse this with an inline child body: `call(f, ...)` invokes an already
named child flow and synthesizes a CALL node; `kind: call` + `nodes:` defines a
one-off child's body on that node.

## Prompt names are local
Inside a `prompt:` use names from the node's own `input:` block, written bare
(`${name}`). An AGENT prompt also sees its own `variables:`. Pool references (`${input.x}`,
`${other.output}`) go in the `input:` block — bind there, then use the local name
in the prompt.

## Effects from inside an AGENT node — `ask_user`
A `tool_calling` AGENT node granted `tools: [ask_user]` can synchronously ask the human
**only if the model decides it needs to**. The tool blocks inside the same live
model/tool loop and its returned answer becomes the ordinary tool result. For a
*guaranteed* gate use `human_input`; for "ask only if needed" use `ask_user`.
`ask_user` on an AGENT node returns its answer to the same live tool loop; HUMAN_INPUT and
WAIT return ordinary node outputs from their live calls. The host passes the handler
per run: `run_flow(..., tools={"ask_user": handler})`. `composer run` and `composer chat` do
this for you; a run without a handler fails any node that asks. There is no authored
`entry:`/`controls:` field on an AGENT node or event-mode WAIT.

## Validate & run
- **Quick validate** (loads + compiles, no model): `composer run <flow>.yaml` will load
  first; or from Python `load_flow(text, search_paths=[flow_dir])`.
- **Run:** `composer run <flow>.yaml [--inputs file.json] [--input k=v]...`. JSON inputs
  load first; repeatable `--input` flags override individual keys but remain a
  string/scalar surface. With an interactive stdin (TTY), every declaration still
  missing is prompted. Keep four outcomes distinct: a value, `Set null`,
  omit/use-default, and cancellation. Nullable boolean and `Literal[...]` selects
  include `Set null`. Null-capable open scalar/list/object inputs first show
  omit/default when allowed, `Set null`, and `Enter value`; Enter opens the existing
  scalar or typed-JSON editor. Optional blank there returns to the palette rather
  than omitting, while a direct non-null-capable optional/defaulted editor still
  blanks to omission. The null-only `None` type has `Set null` and no Enter action.
  Treat text `null` for a string input or a `Literal` tag `null` as an ordinary
  string. Keep root JSON `null` invalid in the structured editor—choose `Set null`
  instead—while allowing nested null where the nested contract accepts it.
  Cancellation cancels the run.
  Without a TTY, missing optional/defaulted declarations stay absent. Missing
  required declarations fail with exit code `2` before provider or run-log setup;
  explicit null and native structures require `--inputs`, while ordinary scalars
  may use the string-valued `--input` or `--inputs`.
  `human_input` and `ask_user` are separate live-run interactions, unchanged by
  the initial-input omission policy. The host broker reads their answers from its
  input stream while the run remains live. WAIT does not prompt and also remains
  in the live run.
- A flow with only `code`/`tool` nodes runs with **no provider**; any `agent` node
  needs a provider/model configured (see `docs/installation.md`).

## Authoring checklist
- [ ] `input:` and `output:` are fully typed; every type used is a scalar/form or a `typedefs:` entry.
- [ ] Every data dependency uses `${...}` in `input:`; ordering-only dependencies use `depends_on:` or `runs_after:`.
- [ ] Every `agent` `output:` is typed — bare `str` for free text, or a
  `Literal[...]`/other supported shape for structured generation.
- [ ] Every prompt references local inputs or, for an AGENT, its own declared variables.
- [ ] Every variable has a typed default and a name distinct from local inputs.
- [ ] Every `long_term_memory:` declaration is on a flow or `agent`, selects a host-registered memory policy by name, and contains only public behavior config.
- [ ] Each `case` has an `else:` (or its `Literal` cases are exhaustive); branches are joined with `|`.
- [ ] `tool` ids are registered or passed per run with `tools=`; a `code` `module:function` is importable (or an inline `code: |` body `return`s a value); `call`/`uses:` targets resolve on the search path.
- [ ] Every CALL/MAP/LOOP has exactly one child form: named `call:`, or non-empty inline `nodes:` plus an `output:` key; inline parent/item dependencies cross through outer `input:`.
- [ ] A child-flow call is written `call(f, ...)` as a whole value, never `${flow(args)}`; string `:-` defaults and `:?` messages are quoted.
- [ ] It loads cleanly (`composer run` / `load_flow`) before you wire a model.
