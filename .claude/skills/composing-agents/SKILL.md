---
name: composing-agents
description: Use when authoring an Agent Composer flow — writing or editing a flow YAML, building an agent workflow, wiring nodes (agent / code / tool / case / human_input / wait / call / map), or answering "how do I write a flow that …". Trigger on "create an agent", "write a flow", "compose a workflow", "add a node to my flow", "how do I branch / loop / call another flow", or any `ac run` flow-authoring request.
---

# Composing an agent workflow

An Agent Composer **flow is a function**: a typed `input:`, a graph of `nodes:`,
and a typed `output:`. You never draw edges — the engine **infers the graph from
the `${...}` references** between nodes. The human owns the structure; the LLM
only fills the leaf boxes.

Full reference: [`docs/syntax.md`](../../../docs/syntax.md). Working flows:
[`examples/`](../../../examples). This skill is the authoring workflow + the parts
that bite.

**Companions:** [`reference.md`](reference.md) — operators, the three expression
contexts, type forms, recipes, and gotchas. [`templates/`](templates/) — minimal
loadable starter flows for each shape (single agent, agent→code pipeline,
branch+join, tool use, human gate, `call`/`map` composition); copy one and edit.

## The workflow — author a flow in order

1. **State the function.** Write the one-line job, then its **signature**: what
   typed values come in (`input:`) and what comes out (`output:`). If you can't
   name the input and output types, the design isn't done.
2. **Name reusable types** in `typedefs:` (enums, records, aliases).
3. **Decompose into leaf nodes.** Each node is one step: an LLM call (`agent`),
   deterministic code (`code`), a registered tool (`tool`), a branch (`case`), a
   blocking human gate (`human_input`), a blocking UTC deadline (`wait`), a child-flow call
   (`call`/`map`), or an iterated body (`loop`). Prefer many small typed nodes over
   one mega-prompt.
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
| `${name}` (bare) | **only inside a `prompt:` or a `case` `when:`** — that node's own declared input |

A whole-string `${ref}` resolves to the **typed value**; embedded in surrounding
text it is **stringified**.

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

Nesting a ref is allowed: `${a:-${b:-"lit"}}`. The `:-`/`:?` RHS is an
expression, so a string default must be **quoted** (`${x:-"today"}`; bare `today`
reads a variable). A child-flow call is the whole-value `call(...)` directive
(see below), never a `${flow(args)}` span.

## Types

Python typing vocabulary. Scalars: `str`, `int`, `float`, `bool`, `date`,
`datetime`, `object`, `None`. Forms: `list[X]`, `Optional[X]`, `Literal[...]` (an
enum of tags), and named **records**. `Optional[X]` (nullable) and `= default`
(omission-fill) are orthogonal.

```yaml
typedefs:
  Basket: list[str]                    # alias (aliases compose)
  Decision: Literal[go, no_go, wait]   # enum — one of these tags
  Signal:                              # record — fields recurse
    score: float
    note: Optional[str]
```

## Node kinds (the closed set — you compose these, you don't define new ones)

### `agent` — an LLM leaf
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
**An AGENT's `output:` may be any shape.** A bare `str` or a `Literal[...]` enum
(the model answers with one tag) keeps the agent a text producer. A record,
`float`, `int`, `bool`, or list switches it to **structured generation** — the
engine derives a schema from `output:` and the model emits a conforming value
(native structured output, or a JSON fallback for providers that lack it), validated
at the write boundary and retried up to `retries:` times (default 2) on deviation.
Use a downstream `code`/`model` node only for deterministic post-processing. To give
an agent ordinary tools: `tools: [tool_id, ...]` (requires `mode: tool_calling`).
A node may pin its model via `llm_config:`; otherwise the process defaults captured
once when the run initializes apply. `inherit: false` skips authored flow/parent/CLI
layers but still receives those provider/model defaults.
Tune a node's non-model behavior with a static `env:` block (flow-level default +
per-node override, node wins): e.g. `env: {max_tool_iterations: 300}` bounds an
agent's tool-calling loop (default `-1` = no cap).

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
  output: Signal               # unlike an agent, ANY type
  code: pkg.mod:fn             # a reference — OR an inline `code: |` body that returns
```

### `tool` — a registered tool (no LLM)
```yaml
news:
  kind: tool
  tool_id: get_facts           # a TOOL_REGISTRY key
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
**Always** calls the host-owned `ask_user` tool and blocks its worker until the
host returns a typed answer (validated against `output:`). It stays inside the
live node call; it does not create a resumable checkpoint.
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
compose-agent + gate; the gate only calls the host's installed interaction tool.
```yaml
ask:                                  # (A) static questions list
  kind: human_input
  input: { proj: ${input.proj} }      # ${proj} renders into question text
  questions:
    - question: "Which framework for ${proj}?"
      header: Framework               # answer key -> {Framework: <label>}
      options: [ {label: React, description: a lib}, {label: Vue, description: a fw} ]
    - { question: "Any notes?", header: Notes }   # no options -> free-text
ask2:                                 # (B) LLM composes them (desugars at load)
  kind: human_input
  input: { ctx: ${research.output} }
  adaptive_questions: { prompt: "Design 1-3 questions for: ${ctx}", retries: 3 }
```
(Form (C): read a list from an upstream node — `questions: ${qs}`.)

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
duration elapses, then returns `None`; it does not create a checkpoint.

### `call` / `map` — composition over child flows
A child is itself a flow file. `call` runs it once; `map` runs it per list element
(`${item}` is the current element).
```yaml
uses:
  research-one: 03-research-one   # bind an external sibling flow to a local alias

nodes:
  each:
    kind: map
    over: ${input.topics}         # a list[T]
    call: research-one            # node value is list[U]
    parallel: true
    input: { topic: ${item} }     # per-element call args (sink-bind the child's params)
```
`call:` resolves **defs-first, else a `uses:` alias** (an external flow on the
search path, by filename; `alias@v1` adds a version guard). Reference the callee's
object fields downstream as `${each.output.field}`.

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
The body's `output:` shape must **equal** the carried record and read only a
**subset** of its names (`'a -> 'a`) — names checked at build, types at load. The
node's value is the final carried record. A body that blocks (a `human_input` leaf)
makes the loop a live chat REPL: the same run threads each answered turn. Pick
**exactly one**
driver: `while:` (pre-check, 0+ runs), `until:` (post-check / do-while, 1+ runs,
stop when the predicate becomes true), or `times: N` (fixed count). `max:` is
required for `while:`/`until:` but **redundant and rejected** with `times:`.

#### Conversational REPL — a chat as a `loop` (what `ac chat` runs)
A chat is a `loop` whose body blocks each turn at `ask` (`human_input`), then a
`case` routes exact `/exit` and `/quit` values to an exit record before the
`reply` agent. The normal branch builds the next transcript; defaulted output
bindings rejoin the exit record or normal transcript into the carried
`{transcript: str, exited: bool}` record. Typed exit is flow-owned — a host must
not intercept it — while terminal EOF may be normalized to `/exit`. **Gotcha:**
the body's `output:` must EQUAL the carried record. A lone multi-ref coalesce has
no inferable type; adjacent defaulted spans preserve `str`, and the lone
defaulted exit span preserves `bool`. See [`templates/chat.yaml`](templates/chat.yaml).

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
  (boundary, before any node) or `node.output` (post, after the terminal) —
  `${system.X}` is **not** allowed there (it is ambient; a `system` ref is a
  LoadError — move the check into a body node).
- **Prompts**: free text with embedded `${...}` spans (each stringified).

> Bindings wire, conditions test, nodes compute.

## Inline `call(...)` — a flow call as a binding's whole value
`call(f, arg=${ref}, k=lit)` written as a binding's **whole value** (or a `case`
`then:`/`else:` target) desugars at load into an anonymous `call` node; the host
becomes `${<synth>.output}`. Keyword args only; nesting allowed
(`call(a, x=call(b, y=1))`). It is recognized **only** as the entire trimmed
value — a `call(` mid-value or inside `${...}` is literal text. The old
`${flow(args)}` form (a flow call *inside* braces) is a **load error**: hoist an
embedded flow call to a named `call` node, and split a coalesce-of-calls
(`${a() | b()}`) into per-node outputs. Pure builtins (`${upper(x)}`) stay legal
inside `${}`.

## Prompts see only LOCAL inputs
Inside a `prompt:` you may reference only names the node declares in its own
`input:` block, written bare (`${name}`). Pool references (`${input.x}`,
`${other.output}`) go in the `input:` block — bind there, then use the local name
in the prompt.

## Effects from inside an agent — `ask_user`
A `tool_calling` agent granted `tools: [ask_user]` can synchronously ask the human
**only if the model decides it needs to**. The tool blocks inside the same live
agent loop and its returned answer becomes the ordinary tool result. For a
*guaranteed* gate use `human_input`; for "ask only if needed" use `ask_user`.

## Validate & run
- **Quick validate** (loads + compiles, no model): `ac run <flow>.yaml` will load
  first; or from Python `load_flow(text, search_paths=[flow_dir])`.
- **Run:** `ac run <flow>.yaml --input k=v [--inputs file.json]`. Missing required
  inputs are prompted; `human_input` and `ask_user` requests are served by the
  host while the run remains live. WAIT also remains in the live run. A temporary
  generic `Pause`/checkpoint path remains only for artificial compatibility until
  issue #117 removes it.
- A flow with only `code`/`tool` nodes runs with **no provider**; any `agent` node
  needs a provider/model configured (see `docs/installation.md`).

## Authoring checklist
- [ ] `input:` and `output:` are fully typed; every type used is a scalar/form or a `typedefs:` entry.
- [ ] Every node consumes upstream values via `${...}` in its `input:` (the edges are inferred).
- [ ] Every `agent` `output:` is typed — `str`/`Literal[...]` for text, or a record/number/bool/list for structured generation.
- [ ] Every prompt references only the node's own local input names.
- [ ] Each `case` has an `else:` (or its `Literal` cases are exhaustive); branches are joined with `|`.
- [ ] `tool` ids are registered; a `code` `module:function` is importable (or an inline `code: |` body `return`s a value); `call`/`uses:` targets resolve on the search path.
- [ ] A child-flow call is written `call(f, ...)` as a whole value, never `${flow(args)}`; string `:-` defaults and `:?` messages are quoted.
- [ ] It loads cleanly (`ac run` / `load_flow`) before you wire a model.
