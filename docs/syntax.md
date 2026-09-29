# Flow syntax

A flow is a **function**: a typed `input:`, a graph of `nodes:`, and a typed
`output:`. `${...}` references infer data edges; `depends_on:` and `runs_after:`
add ordering-only edges. The human owns the structure; the LLM only fills the
AGENT node leaves. See [Agent](agent.md) for the conceptual model and terminology;
this page is the YAML reference.

## The flow shape

A flow file is Docker-Compose-shaped: metadata scalars at the top, then the
interface and body as top-level sections.

```yaml
id: momentum               # stable identifier
name: momentum             # display name
description: ...           # optional, one line
typedefs: { ... }          # optional — named/composed types
input:   { ... }           # parameters — typed
nodes:   { ... }           # body — a map keyed by node id
output:  { ... }           # return — bindings (one value, or a multi-field object)
asserts: [ ... ]           # optional — boolean invariants
```

There is **no `edges:` block**, no `__start__`/`__end__`, no per-node `id:`, and
no body wrappers — a node body is flat.

Node IDs are the keys under `nodes:`. They cannot contain `.`, `#`, or `/`:
`.` separates the node name from `output` and its field path in a reference,
while `#` and `/` are reserved for generated runtime identities. In compact
mode, the flow `id` also names its node and follows this rule.

### Compact mode — when the flow *is* one node

The common case is "one flow, one node." Writing a full `nodes:` map plus a
redundant `output: ${greet.output}` wiring step for it is noise, so a flow whose
body is a single node can be written **inline**: drop the `nodes:` map and put the
node's `kind:` and its fields at the top level.

```yaml
# hello.yaml — the compact form
id: hello                  # names BOTH the flow and its single node
name: hello
input:
  name: str                # the node signature — auto-wired by name
output: str                # the node's output TYPE — re-exported as the flow output
kind: agent
prompt: |-
  Write a short, warm one-sentence greeting addressed to ${name}.
```

This desugars to the canonical one-node flow below before compile — same IR, same
behavior:

```yaml
id: hello
name: hello
input:
  name: str
nodes:
  hello:                   # keyed by the flow id
    kind: agent
    input:
      name: ${input.name}  # each flow input auto-wired by name
    output: str
    prompt: |-
      Write a short, warm one-sentence greeting addressed to ${name}.
output: ${hello.output}    # the single node's output, re-exported
```

Rules:

- The flow `input:` is the node's signature — each parameter is auto-wired into the
  node by name (`name` → `${input.name}`), so you refer to it bare in the prompt.
- The flow `output:` is the node's output **type**; the flow returns that node's
  output (no explicit `output: ${...}` line).
- Any other field (`prompt:`, `tools:`, `llm_config:`, node-local `asserts:`, …)
  is the node body.
- Allowed only for the **value-producing leaf kinds** — `agent`, `code`, `model`,
  `tool`, `human_input`. `case`/`call`/`map` reference other nodes a one-node flow
  has none of, so they need the full `nodes:` form.

## References — naming a value

Everywhere you wire data you use a `${...}` reference:

| Write | Means |
|-------|-------|
| `${input.X}` | field `X` of the flow's input |
| `${node.output}` | node `node`'s whole value |
| `${node.output.field}` | dot into an object value |
| `${name}` / bare `name` | a local input in a node's `prompt:` or a `case` `when:`; AGENT prompts also see their own declared `variables:` |

A whole-string `${ref}` resolves to the **typed value**; embedded in surrounding
text it is stringified.

Prompt templates are strict: a missing reference or any span whose final value is
null raises, even when the span is embedded in text. Binding templates retain their
non-strict behavior—a whole null remains a typed null, while an embedded null
contributes empty text.

!!! important "Prompt names are local"
    Inside a `prompt:` use names from the node's own `input:` block, written
    bare, like `${name}`. An AGENT prompt also sees its own `variables:`. Pool references
    (`${input.x}`, `${other.output}`) belong in the `input:` block, not the
    prompt. Bind it there, then refer to the local name in the prompt.

### Operators inside `${...}`

A `${...}` is **one expression**. The same grammar works everywhere `${...}`
appears — bindings, conditions, prompts. Inside it you may write:

| Form | Meaning |
|------|---------|
| `${a + b}`, `${a * b + 1}` | arithmetic — `+ - * / % **`, unary minus |
| `${a == b}`, `${x in [1, 2]}` | comparisons — `== != < <= > >=`, `in` / `not in` |
| `${a and b}`, `${not a}` | boolean — `and` / `or` / `not` |
| `${[a, b, c]}` | a list literal (elements are themselves expressions) |
| `${upper(name)}`, `${join(items, ", ")}` | a pure builtin call (with dotted access on the result: `${fn(x).field}`) |
| `${X:-default}` | value, else `default` if absent |
| `${X:?"msg"}` | required — fail with the literal `msg` if absent (quote the message) |
| `${a \| b \| c}` | first present among peers (n-ary coalesce — for branch joins) |
| `$$` | a literal `$` (the scanner's universal escape — outside a span; in a prompt `$$` renders a single `$`) |

The operator words `and`, `or`, `not`, and `in` are reserved and cannot be the
root name of a reference. They may still name a dotted field, for example
`${record.output.in}`.

The `:-` default and `:?` message RHS are themselves expressions, so a **string**
default must be quoted: `${input.as_of:-"today"}`, not `:-today` (bare `today`
would read a *variable* named `today`). Nesting a ref is allowed:
`${a:-${b:-"lit"}}`.

A whole-string `${expr}` resolves to the **typed value** (a number stays a
number, a list a list); embedded in surrounding text it is stringified.

!!! note "Flow calls are not expressions"
    A `${...}` computes over values — it does **not** invoke a flow. To call a
    child flow inline, use the `call(...)` directive (see [`call` / `map`](#call-map-composition-over-child-flows)),
    not `${flow(args)}` (which is a load error).


## Types

The type vocabulary is Python typing. Scalars: `str`, `int`, `float`, `bool`,
`date`, `datetime`, `object`, `None`. Containers/forms: `list[X]`, `Optional[X]`,
`Literal[...]` (an enum of tags), and named **records** (dataclass-style).

Numeric domains may add inclusive bounds: `int[1, 12]` or `float[0.0, 1.0]`.
Both endpoints are required and the lower endpoint must not exceed the upper.
Integer bounds can be any size. Float bounds must be finite; integer endpoints
used in a float domain must convert without rounding. Booleans and computed
endpoint expressions are invalid. Bounds work inside aliases, records, lists,
and `Optional[...]`.
Defaults and supplied values must satisfy them. Cross-flow wiring requires the
source domain to fit inside the destination domain; an unbounded source cannot
establish that guarantee for a bounded destination.

Name reusable types in `typedefs:`:

```yaml
typedefs:
  Ticker: str                          # alias
  Basket: list[Ticker]                 # aliases compose
  Decision: Literal[go, no_go, wait]   # enum — one of these tags
  Signal:                              # record — fields recurse
    score: float
    note: Optional[str]                # nullable field
```

`Optional[X]` (nullable) and a `= default` (omission-fill) are orthogonal. A
required input has neither; `lookback: int = 30` defaults when omitted; and
`Optional[date]` with no default may be omitted, resolving to null. An explicit
`Optional[int] = null` is a present default, distinct from having no `=`. A null
default for a non-nullable type is invalid, as is an empty `=`.

Typed storage for `list[Optional[X]]` accepts actual null items because its
refined element type is nullable. When `X` has a specialized list storage kind,
those items remain `None` without lowering that kind to `list[Any]`. Cross-flow
bindings compare element contracts too: `list[X]` may fill
`list[Optional[X]]`, but the nullable source cannot fill the strict sink.

The default RHS is a YAML literal validated against the declared type while the
flow loads. Stored defaults are never string-coerced: write `int = 30`, not
`int = "30"`; a date string must likewise be quoted, such as
`date = "2026-01-01"`. Compatible payloads retain their authored representation,
so `float = 30` remains the integer `30`. String coercion applies only to values
supplied at a flow boundary.

## Per-node configuration — `env:`

Any node may carry an `env:` block — a mapping of **static** config keys the node's
own implementation reads. It is the generic knob for "tune this node's behavior"
that isn't data flow (`input:`) or model selection (`llm_config:`). Keys are plain
strings; each key's meaning and type is the consuming node's concern (the engine
never interprets `env`; compose stores the merged mapping on the node for its own
implementation to read).

A flow can set a default `env:` for **every** node under it; each node can override
individual keys. On build the two merge per key, **node wins**:

```yaml
env:                            # flow layer — default for every node
  max_tool_iterations: 150
nodes:
  researcher:
    kind: agent
    prompt: Investigate ${topic}.
    tools: [search]
    env:
      max_tool_iterations: 300  # this node overrides the flow default
```

The merge is **static, at build time** (`{**flow_env, **node_env}`) — there is no
runtime cascade and no per-field resolution across parent flows. It is also
**flow-local**: a called child flow does *not* inherit its parent's flow `env:` —
it has its own. Values must be literals (no `${...}` refs).

Keys in use today:

| Key | Node kind | Meaning |
|-----|-----------|---------|
| `max_tool_iterations` | `agent` | Model turns the tool-calling loop runs before it gives up with an `AgentLoopError`. Default `-1` (**no cap** — run until a final answer). Set a positive int to bound it; `0` and other negatives are rejected. |

## Agent memory — `long_term_memory:`

`long_term_memory:` declares long-term memory on a full flow (the ordinary
`nodes:` form) or an AGENT node. Its value selects a named host policy. The
declaration contains public behavior only; the Python host supplies the matching
implementation at run time.

```yaml
id: research
name: research
long_term_memory:
  name: reflective
  config:
    max_items: 100
    recall_limit: 12
input:
  topic: str
nodes:
  investigate:
    kind: agent
    long_term_memory: reflective
    input: {topic: "${input.topic}"}
    output: str
    prompt: Investigate ${topic}.
output: ${investigate.output}
```

The short form is `long_term_memory: reflective`. The mapping form accepts
exactly `name:` and an optional literal `config:` mapping; `name` selects the
host policy. Do not put credentials,
tenant IDs, Python imports, backend paths, or persistence locations in YAML;
those belong to the host-created memory object.

A flow declaration creates one flow-owned frame inherited by every AGENT node
executed in that flow. A declaration on an AGENT node creates a second, local
frame. Recall stacks those sections from the outer flow inward, while each frame
learns and writes only its own state. CALL enters the called flow's frames. MAP
items and LOOP iterations do not create frames of their own; repeated executions
contribute occurrences to the same static frame authored on the flow or AGENT
node. `long_term_memory:` is invalid on other node kinds and on anonymous inline
CALL/MAP/LOOP bodies; use a named `defs:` flow when the child itself needs a flow
frame.

`composer run` automatically supplies declared `episodic`, `reflective`, and `session`
policies. They are process-local unless the host flag
`--memory-storage POLICY=PATH` selects a durable store; storage never belongs in
YAML. Custom policies, including `ProfileMemory` (which needs a document store),
require Python host injection. For Python calls, declarations
remain dormant unless the host passes a `memory={...}` registry to `run_flow`,
keyed by the selected policy names. All memories contribute through one context:
ordinary memories supply notes or instructions, and SessionMemory supplies a
structured session transcript alongside any learned notes.
For conversation continuity, declare `long_term_memory: session`; Python hosts
inject `SessionMemory()` under `"session"`. It accepts no frame config. The host can set
`SessionMemory(history_budget_chars=32_000, recent_turns=4)` for API context
summaries; these settings do not belong in YAML and do not truncate raw history.
The innermost frame contributing a transcript owns a separate conversation for each
authored AGENT leaf; other frames still provide recall. Repeated calls queue in
admission order, including LOOP iterations and MAP items. MAP does not fork
history or guarantee input-index admission order; use non-session memory when
items should be independent. History updates automatically during forward,
without enabling learning. Explicit API backward learning can commit self-feedback
as notes for future runs through the same memory policy; it requires no additional
YAML selector. See [SessionMemory](api.md#sessionmemory-foundation).

Independently, `learnable=True` retains
accepted execution for an explicit `result.backward(feedback)` call after forward
returns. Forward does not run frame learners or commit learned mutations; session
history updates are separate. A call with no external feedback still
presents history to every enabled memory frame; backward outcomes stay separate
from the forward result. See [the Python API](api.md#explicit-backward-feedback).
The `composer run` CLI activates memory but does not expose backward or learning
commands. See [CLI memory](cli.md#memory-policies-and-storage).

## Node kinds

Every node has a `kind:`. The set is closed — you compose flows from these, you
do not define new kinds.

### `agent` — the LLM-backed AGENT node

```yaml
classify:
  kind: agent
  input:
    text: ${input.text}        # bindings — these infer the data edges
  output: Literal[positive, neutral, negative]
  prompt: |-
    Classify the sentiment of: ${text}
    Answer with exactly one of: positive, neutral, negative.
```

An AGENT node's `output:` may be any declared shape. Only a bare, untagged `str`
keeps the node a **text producer**. A `Literal[...]` enum, record, non-string scalar,
or list switches the node to **structured generation**: the engine derives a
schema from the declared `output:` and asks the model to emit a conforming value
(via the provider's native structured output, or a JSON prompt-injection fallback
for providers that lack it). A root `Literal[...]` is constrained internally as a
single `value` field and returned to the flow as the bare tag. The result is
validated while its canonical success record is prepared, like every other node
output.

```yaml
extract:
  kind: agent
  input:
    text: ${input.text}
  output:                       # a record shape -> structured generation
    name: str
    score: int
  prompt: |-
    Extract the person's name and a 0-10 score from: ${text}
```

If the model deviates from the schema, the engine feeds the error back and retries
up to `retries:` times (default 2):

```yaml
extract:
  kind: agent
  retries: 3                    # extra self-correction attempts (default 2)
  output: {name: str, score: int}
  prompt: ...
```

A node can pin its own provider/model; otherwise the process defaults snapshotted
when the run initializes apply (see [Installation](installation.md)).

#### `llm_config` — the model-selection cascade

Model selection cascades **per field, most-specific wins**. Each field (provider,
model, temperature, …) is resolved independently: an AGENT node fills only the fields
it leaves unset from the layer outside it. Precedence, most specific first:

1. the AGENT node's own `llm_config:`
2. the enclosing flow's `llm_config:`, then each parent flow outward
3. the CLI `--provider` / `--model` flags (`composer run … --provider anthropic`)
4. process provider/model defaults, read exactly once when the run initializes

Process defaults use environment overrides, then saved authentication choices,
then the provider's built-in fallback. OpenAI is the default provider. An unset
model uses the selected provider's default; it never inherits another provider's
saved or environment model. An explicit model in an authored layer still follows
the per-field precedence above.

A flow can set defaults for every AGENT node under it with a top-level `llm_config:`:

```yaml
llm_config:                     # flow layer — every AGENT node inherits these
  provider: anthropic
  temperature: 0.2
nodes:
  drafter:
    kind: agent
    prompt: Draft a summary.
    llm_config:
      model: claude-opus-4-8    # fills the one field the flow leaves unset
```

The CLI flags are the *outermost authored* layer — they fill gaps, they do **not**
override an AGENT node or flow that set the field. The run resolves the complete
static tree of AGENT nodes up front, so later environment changes cannot alter a model selection.
To ignore the authored outer layers, set `inherit: false` in a node's
`llm_config:`. The node then uses its own dict plus the snapshotted process
provider/model defaults; it ignores enclosing flow, parent-flow, and CLI layers:

```yaml
  grader:
    kind: agent
    prompt: Grade it.
    llm_config:
      provider: openai
      model: gpt-5.5
      inherit: false            # own config + process defaults; no flow/CLI layers
```

#### OpenAI authentication

Run `composer auth openai` to select ChatGPT subscription sign-in or an OpenAI API key,
then choose the default model. Both use `provider: openai` and the ordinary
`plain` or `tool_calling` modes, explicit Composer tool grants, typed output,
SessionMemory, and backward learning.

The retired `provider: codex`, `sandbox`, `cwd`, and `builtin_tools` settings
must be removed when migrating. Native thread references cannot be resumed as
message histories; use fresh SessionMemory. Composer tools execute on the host.
See [authentication setup](installation.md#openai-authentication).

#### Agent-owned variables

Declare local prompt settings in `variables:`. Every entry needs a type and an
explicit default:

```yaml
id: summarize
name: summarize
input:
  topic: str
nodes:
  writer:
    kind: agent
    variables:
      num_sentences: int[1, 12] = 4
      temperature: float[0.0, 1.0] = 0.4
      tone: Literal[concise, detailed, technical] = concise
      instruction: str = "Summarize clearly"
    input:
      topic: ${input.topic}
    prompt: ${instruction}. Write ${num_sentences} ${tone} sentences about ${topic}.
    output: str
output: ${writer.output}
```

Variable names must be unique and cannot overlap the node's input names. Defaults
are validated while loading, including bounds and enum choices. Nullable variables
still require an explicit default, such as `Optional[str] = null`; a quoted empty
string is a valid `str` default. Default strings are literal text, so `${...}`
inside a default is not evaluated as a binding.

Each invocation starts with a fresh copy of these defaults. Repeated `${name}`
references read the same local value. A variable belongs to its authored AGENT
path and name: MAP items and LOOP iterations of that node share its declaration,
while distinct authored callsites have distinct owners. Variables introduce no
data dependencies or flow outputs. A variable named `temperature` is prompt data;
configure model temperature through `llm_config:`.

### `code` / `model` / `tool` — the other leaves

These are the non-LLM computational leaves — a Python callable, an ML model, or
a registered tool. CODE and MODEL consume `input:` bindings and may declare a
typed `output:`. TOOL instead accepts an untyped `input:` map defined by the
registered tool and rejects `output:`; its return value is still bound as the
node's `output` for downstream references.

A `code` node's `code:` field is one of two shapes:

**Reference** — a `module:function` token: import and call it in-process.

```yaml
verdict:
  kind: code
  input:
    s: ${score.output.signal}
  output: str
  code: pkg.mod:fn             # module:function
```

**Inline** — real source shipped with the flow. Write a **bare body** (no `def`) that
reads the node's inputs via the `inputs` dict and **`return`s** a value; the engine
wraps it as `def main(inputs):` and runs it **in-process**:

```yaml
verdict:
  kind: code
  input:
    rating: ${score.output.rating}
    label:  ${score.output.label}
  output: str
  code: |
    lean = "positive" if inputs["rating"] >= 0 else "negative"
    return f"{inputs['label']}: {lean}"
```

Inline notes:

- **Same one-dict convention as reference mode** — inline reads `inputs["x"]` exactly as
  a `module:function` callable receives `inputs`, so a body promotes to a reference by
  copy-paste. A body **must `return`** a value (a body with no `return` is rejected at
  load).
- **In-process, no isolation — only run flows you trust.** The trust model is
  *author == operator*: inline `code:` executes arbitrary Python **with your privileges**,
  the same capability the reference form already has. There is **no sandbox**, so loading a
  flow you didn't author — one written by someone else, or generated by an agent — runs
  *their* code as *you* (remote code execution). Keep flow YAML operator-authored and
  version-controlled. A runaway body is not killed, only **logged** by CODE's watchdog once it
  overruns a soft time budget; a killable, sandboxed runtime (and an opt-in gate for
  untrusted sources) is planned.
- **Output must be JSON-representable** (`str` / `int` / `float` / `bool` / `list` /
  `object` / typedefs); a non-serializable return — even nested — fails at the node.
- **Imports** resolve against the process `sys.path` (cwd + `PYTHONPATH`), same as the
  reference form. A dotted token with no colon (`pkg.mod.helper`) is treated as a likely
  typo'd reference and rejected at load with a "did you mean `pkg.mod:helper`?" hint.
- **`file:` / folder / repo sources are planned**, not yet available.

### `case` — branching

A `case` node **routes only** — it has no `input:`. Exactly one branch runs; the
others are skipped (their references resolve to null). Join the branches back
with a coalesce.

Simple form — match a value:

```yaml
route:
  kind: case
  on: ${synth.output}          # a Literal value
  cases:
    - when: go
      then: go_brief
    - when: no_go
      then: no_go_brief
  else: more_info_brief        # required unless the cases are exhaustive
```

Searched form — first true `when:` wins (a boolean expression, no `on:`):

```yaml
gate:
  kind: case
  cases:
    - when: "score.output.signal >= 0.5"
      then: bullish
  else: cautious
```

Routing on a `Literal` is **exhaustiveness-checked**: omitting a tag with no
`else:` is a compile error. Join the branches:

```yaml
output: ${bullish.output | cautious.output}
```

### `human_input` — a deterministic human gate

A `human_input` node **always** invokes the host-owned `ask_user` interaction at a
fixed point and blocks its worker until the human answers. The answer is
validated against the declared `output:` before the flow continues. This is one
live node call; the invocation stays active until the answer arrives.

```yaml
approve:
  kind: human_input
  input:
    plan: ${draft.output}      # context the prompt may reference (bare ${plan})
  prompt: |-
    Here is the plan:

    ${plan}

    Approve as-is, or revise? (approve / revise)
  output: Approval             # a typed answer, e.g. a Literal enum
```

The same invocation stays live until the host returns an answer. Instead of (or
alongside) a `prompt:`, a gate may carry **questions** —
AskUserQuestion-shaped multiple-choice/free-text prompts. Each question is
`{question, header, options:[{label, description}], multi_select}`; `options`
(omit ⇒ free-text) and `multi_select` (default `false`) are optional. 1–4
questions, headers unique. The host always offers a free-text **"Other"** escape.
The gate's answer is a **record keyed by header** — `{header: label}`, or
`{header: [labels]}` for a `multi_select` question — so `output:` defaults to
`object`. `prompt:` is optional once a node has questions; `questions:` and
`adaptive_questions:` are **mutually exclusive**.

**Static questions** — a literal list (with `${...}` templating from `input:`):

```yaml
ask:
  kind: human_input
  input: { proj: ${input.proj} }
  questions:
    - question: "Which framework for ${proj}?"   # ${proj} renders from input
      header: Framework                          # answer key -> {Framework: <label>}
      options:
        - { label: React, description: A component library. }
        - { label: Vue, description: A progressive framework. }
      multi_select: false        # optional, default false
    - question: "Any notes for the build?"       # no options -> free-text
      header: Notes
  # output omitted -> defaults to object: {Framework: ..., Notes: ...}
```

**Adaptive questions** — an LLM composes the questions from context. The block
**desugars at load** into a synthetic composer AGENT node (`<node>__compose`, output
`list[Question]`) wired into the gate; the runtime gate never calls an LLM.
The input name `__questions` is reserved for that generated wire and cannot be
authored on an adaptive gate.

```yaml
ask:
  kind: human_input
  input: { ctx: ${research.output} }
  adaptive_questions:
    prompt: "Design 1-3 questions with options for: ${ctx}"  # required (LLM brief)
    mode: plain                  # optional, default plain
    llm_config: { model: ... }   # optional — the composer's provider/model
    retries: 3                   # optional — self-correction attempts
```

**Upstream questions** — read the list from an author-written upstream node:

```yaml
ask:
  kind: human_input
  input: { qs: ${composer.output} }   # composer.output is a list[Question]
  questions: ${qs}
```

### `wait` — block until an absolute UTC deadline

```yaml
settle:
  kind: wait
  until: ${input.as_of}
  # downstream nodes order after it with depends_on: [settle]
```

`until:` must resolve to an ISO `date` or `datetime`. A date means midnight UTC;
a naive datetime is UTC; an offset-aware datetime is normalized to UTC. A deadline
at or before the current instant returns immediately. Otherwise WAIT occupies one
worker until the remaining wall-clock duration elapses, then returns `None`; it does
not split the run into another execution segment. Engine/host shutdown wakes the
worker cooperatively.

### `call` / `map` — composition over child flows

`call` runs a child flow once; `map` runs it over a list (`${item}` is the
current element). A reusable child is named by `call:`: resolution is
`defs:` first, then a declared `uses:` alias.

```yaml
each:
  kind: map
  over: ${input.tickers}       # a list[T]
  call: child_flow             # node value is list[U]
  input:
    ticker: ${item}
```

The `input:` map contains the child's arguments. A `call` re-exports the child's
output (`None` when the child declares no output value); a `map` requires one
child output value and returns `list[U]` in `over:` order.
`parallel:` is accepted but currently inert; MAP concurrency is controlled by
the run's `num_workers`, while collection order remains the `over:` order.

#### Inline child bodies — skip a one-off `defs:` entry

When a CALL, MAP, or LOOP body is used only once, put the child graph directly
on the composite node. This is load-time sugar for a private child flow; the
output uses the same compiled-child execution path as a named `defs:` body, while
keeping occurrence-local identity and diagnostics.

```yaml
nodes:
  clean_one:
    kind: call
    input:                         # arguments become the child parameters
      text: ${input.text}
    nodes:                         # the private child body
      clean:
        kind: code
        input:
          text: ${input.text}      # reads the generated child parameter
        output: str
        code: |
          return inputs["text"].strip()
    output: ${clean.output}        # child return binding

  clean_each:
    kind: map
    over: ${input.texts}
    input:
      text: ${item}                # current item becomes a child parameter
    nodes:
      clean:
        kind: code
        input:
          text: ${input.text}
        output: str
        code: |
          return inputs["text"].strip()
    output: ${clean.output}
```

Inline-body rules:

- Write **exactly one** of `call:` or `nodes:`. Inline `nodes:` must be a
  non-empty mapping and requires `output:`; `output:` is not valid beside a
  named `call:`. The key itself is required: CALL may use `output: null` and
  return `None`, while MAP requires one output value and LOOP requires its
  carried-record shape.
- The outer composite `input:` is still the argument map. Its keys become the
  anonymous child's typed input parameters, and the body reads them as
  `${input.<key>}`.
- The child is closed to its parent's pool: it cannot capture parent inputs,
  parent nodes, or MAP's `${item}`. Pass those values through the outer `input:`
  map, then read the generated inputs and child nodes inside. The run-global
  `${system.today}`, `${system.now}`, and `${system.run_id}` values remain
  available as in every flow.
- Parameter types are inferred strictly. YAML null/bool/int/float/string values,
  closed literal records, non-empty homogeneous literal lists, and direct typed
  references (`${input.x}`, `${producer.output.field}`, or MAP `${item.field}`)
  retain their structural type. A nullable record path produces an optional
  parameter. If a type cannot be proved, loading fails; use a named `defs:` child
  with an explicit `input:` signature.
- MAP proves the `over:` element type only when an outer argument is a lone
  whole-value `${item...}` reference. An opaque `over:` therefore remains legal
  when the inline body does not need the item. When item typing is requested,
  `over:` must have an exact proven list element type. That proof retains
  element nullability, so an optional element admits actual null items and
  preserves them as `None` in typed storage.
- Strings nested inside literal lists or mappings stay literal. For example,
  `payload: {text: "${input.text}"}` passes the characters `${input.text}`; nested
  containers are not scanned for references.
- Existing composite configuration keeps its existing owner and meaning:
  LOOP drivers still configure the outer composite, `env:` stays on that node
  (inner body nodes may set their own `env:` normally), CALL/LOOP node asserts
  keep their normal semantics, and MAP node-local asserts remain invalid.
  `parallel:` remains an accepted outer MAP field but is currently inert.
  `llm_config:` remains invalid on a composite node; the enclosing flow's LLM
  cascade still descends into body AGENT nodes exactly as it does for an unconfigured
  named child. There is no separate inline-child
  `llm_config:`/`env:`/`asserts:` surface; configure individual body AGENT nodes or use
  `defs:` when the child needs its own supported flow-level configuration or
  assertions.

Inline bodies may contain further CALL/MAP/LOOP nodes, including further inline
bodies. Each occurrence remains a separate lexical child.

#### Inline `call(...)` — a flow call as a binding's whole value

Declaring a whole `call` node for a one-off child call is verbose. The `call(...)`
**directive** is sugar: written as a binding's **whole value**, it desugars at
load into an anonymous `call` node, and the host binding is rewritten to reference
that node's output. This applies to every authored node `input:` mapping, including
HUMAN_INPUT context bindings and LOOP carried-record seeds.

```yaml
# these two are equivalent —
news: call(enrich, topic=${input.topic})   # inline directive (sugar)

# desugars to:
__call_0: { kind: call, call: enrich, input: { topic: ${input.topic} } }
news: ${__call_0.output}
```

Rules:

- **Whole-value only.** `call(...)` is recognized only when it is the *entire*
  trimmed value (a binding value or a `case` `then:`/`else:` route target). A
  `call(` in the *middle* of a value, or inside a `${...}` span, is literal text.
- **Keyword args only** — `call(f, x=1, note=${a.output})`. Each arg value is a
  `${...}` expression or a literal (`window=30` binds the int `30`). A positional
  arg is a load error.
- **Nesting is allowed** — `call(outer, x=call(inner, y=1))`.
- **A flow call inside `${...}` is retired.** The old `${flow(args)}` form (a flow
  call *inside* braces) is a load error that points at the `call(...)` directive.
  An **embedded** flow call (`pe=${relevance(t=${x})}` — a call inside surrounding
  text or a `${}` span) must be **hoisted** to a named `call` node and referenced
  by output. A **coalesce of calls** (`${a(x=1) | b(y=2)}`) has no whole-value form
  and is a load error — hoist each call to its own node, then coalesce the outputs.
  (Pure builtins like `${upper(x)}` stay legal inside `${}` — only *flow* callees
  moved out.)

This directive and an inline child body solve different verbosity problems.
`call(named, x=...)` is a whole-value binding that invokes an already named
`defs:`/`uses:` child flow and synthesizes the surrounding CALL node. A
`kind: call` node with `nodes:` defines that one-off child body in place. It
is invalid to combine `call:` and `nodes:` on the same node.

### `loop` — re-run a body under a predicate or a fixed count

`loop` runs a child flow (the **body**) over and over, threading one **carried
record** from each iteration into the next. It is the engine's `while`/`do-while`/
`for`: the body maps the carried record to the *next* carried record (`'a -> 'a`),
and the loop re-runs it under one of three drivers — a pre-check predicate
(`while:`), a post-check predicate (`until:`), or a fixed count (`times:`).

```yaml
turn:
  kind: loop
  call: chat_turn              # the body flow (a def or a file)
  input:                       # the SEED carried record
    messages: []
    exited: false
  while: not exited            # pre-check predicate, over the carried record
  max: 100                     # required runaway guard
```

A one-off LOOP body can be inline too:

```yaml
repeat:
  kind: loop
  times: 3
  input:
    n: 0
  nodes:
    bump:
      kind: code
      input:
        n: ${input.n}
      output:
        n: int
      code: |
        return {"n": inputs["n"] + 1}
  output: ${bump.output}
```

Here the generated child signature is `n: int`, and its record output preserves
the LOOP contract `'a -> 'a`.

- **`input:`** is the seed carried record — its field names and types define `'a`.
- **`call:` / `nodes:`** names a reusable body or defines a one-off body inline.
  The body's `output:` must be the **same shape** as
  the carried record (`'a -> 'a`), and the fields the body reads (`${input.X}`)
  must be a subset of the carried names. This contract is checked twice: field
  **names** at build, and each statically proven body field against a statically
  proven seed source at load. Every proven output field is enforced at runtime and
  replay even when its seed was opaque at load. Each output field's type is inferred
  from its binding shape (see [expressions](expressions.md)): a lone direct ref preserves
  an exact or partial contract, a known partial-record field is exact, a proven comparison
  or boolean expression is `bool`, and a
  **concatenating template** (literal text around spans) is `str`. Therefore a body may
  write `exited: ${ask.output == "/exit"}` directly, and a carried transcript may grow
  through a plain template binding, without a CODE fold node. If a computed field's
  type is not proven, its declared name is still checked but its type stays opaque;
  opaque never means a concrete `None` value and does not erase type checks for proven
  sibling fields. LOOP enforces the closed field set and every proven sibling on normal
  iterations, zero-iteration `while:` commits, and durable replay. A literal `${null}`
  is compatible with a nullable carried field, including inside a typed list.
- **Exactly one** of `while:` / `until:` / `times:` per loop node selects the
  driver. Zero or more than one is a load-time error (`exactly one of
  while:/until:/times: is required`).

**`while:` — pre-check (0+ runs).** A predicate evaluated on the carried record
*before* each iteration; 0 iterations run if the seed already fails it. It is a
record-scoped boolean over bare refs — every ref must name a **carried record
field** (a typo'd name is rejected at load, not silently read as falsy) — written
in the canonical bare form: `while: not exited`. **`max:` is required.**

**`until:` — post-check / do-while (1+ runs).** Same record-scoped predicate
syntax as `while:` (bare refs, no `${}`), but checked *after*
each iteration: the body always runs at least once, and the loop **continues
while the predicate is FALSE and stops the moment it becomes TRUE**. **`max:` is
required.**

```yaml
retry:
  kind: loop
  call: attempt
  input:
    ok: false
  until: ok                    # post-check; runs once, then stops when ok is true
  max: 5                       # required runaway guard
```

**`times: N` — fixed count.** The body runs exactly `N` times, with no predicate.
`N` must be a plain integer `>= 1`. **`max:` is redundant here and REJECTED** — the
count already bounds the loop, so supplying both is a load-time error (`max: is
redundant with times:`).

```yaml
poll:
  kind: loop
  call: step
  input:
    n: 0
  times: 3                     # exactly 3 runs; do NOT also give max:
```

- **`max:`** (for `while:`/`until:`) is a **required** runaway guard — a plain
  integer `>= 1`: if the loop would run more than `max` iterations the run fails
  loudly (`LoopMaxExceeded`).

The node's value is the final carried record (committed under the loop node's id
once the loop stops).

A body may itself block (e.g. a `human_input` leaf): the worker stays inside that
live iteration until the host answers, then the same run advances into the next
iteration. This is the shape a chat REPL takes.

A long loop stays within the engine's node budget: once an iteration commits its
carried record forward, that iteration's expanded body is **pruned** from the live
graph, so only one iteration is resident at a time (a thousand-turn loop costs the
same as a two-turn one).

## Effects from inside an AGENT node — the `ask_user` tool

A `tool_calling` AGENT node can be granted the ordinary `ask_user` tool. Unlike
`human_input` (which always asks), `ask_user` is *model-chosen*: the node calls it
**only if** it decides it needs a fact it cannot supply. The tool blocks inside
the same live model/tool loop, then its returned answer is fed back as the tool result.

```yaml
assistant:
  kind: agent
  mode: tool_calling           # the loop — required to call tools
  tools: [ask_user]            # explicitly grant the capability
  input:
    request: ${input.request}
  output: str
  prompt: |-
    The user asks: ${request}
    If a detail essential to a good answer is missing, call ask_user ONCE to get it.
    Otherwise answer directly.
```

For a *guaranteed* gate use `human_input`; for "ask only if needed" use
`ask_user`.

All three authorable blocking surfaces are live calls: `ask_user` returns the
human's answer to the same model/tool loop, HUMAN_INPUT returns it as the node's
output, and WAIT returns `None` at its deadline.
There is no authored `entry:`/`controls:` surface on an AGENT node and no event-mode WAIT.

## Expression contexts

One `${...}` grammar serves every context — the difference is what each context
*does* with the result, not what it can parse:

- **Bindings** (`input:` / `output:` values): the expression is **evaluated** to a
  typed value. Refs, literals, arithmetic (`+ - * / % **`), list literals,
  comparisons, `:-`/`:?`, `\|`, and pure builtins all work. To invoke a child flow
  as a binding's whole value, use the `call(...)` directive — not a `${...}` span.
- **`when:` / `asserts:`**: the expression is **tested** as a boolean. Same
  operators. Write conditions in the canonical **bare** form (`a.output > 5`) — no
  `${}`. Braces on a ref (`${a.output} > 5`) or around the whole expression
  (`${a.output > 5}`) load and evaluate identically (an expression field is already
  in expression mode, so the `${}` wrapper is a redundant no-op), but bare is the
  form used throughout the docs and seeds.
- **Prompts**: free text with embedded `${...}` spans (each stringified into the
  text); spans may use the full grammar, but reference only the node's own declared
  inputs.

> Bindings wire, conditions test, nodes compute.


## Asserts

`asserts:` are boolean invariants; a false (or raising) one fails the run loudly.
A top-level `asserts:` runs over `input.X` (a boundary check, before any node)
or `node.output` (a post check, after the terminal). An unqualified single name
matching a declared flow input is shorthand (`age >= 18` means `input.age >= 18`)
and wins over a same-named node; node results still require `node.output`. The
shorthand does not extend through fields: write `input.profile.age`, not
`profile.age`. `${system.X}` is **not**
allowed in a flow-level assert — it is ambient (there is no producer node to draw
the check's dependency edge from), so it raises a `LoadError` at load time. If you
need a system-dependent check, put it in a body node's `asserts:` or a `case`.

A flow-level post assert reads its producer over an **optional** edge (the assert
must not deadlock the run when its producer sits behind an untaken `case` branch).
So a producer that is **skipped** resolves to `null` for the assert: a check like
`${maybe.output} == null` **silently passes** when `maybe` is skipped, rather than
failing. Assert on a value you know is produced, or gate the check behind the same
branch as its producer.

```yaml
asserts:
  - synth.output in ["go", "no_go", "needs_more_info"]
```

A node may also carry its own `asserts:`. A node-local assert is **PRE** if it
reads only the node's inputs (checked before the node runs) and **POST** if it
reads `output` (checked once the node's value is committed). Both fail the run
loudly, exactly like a flow-level assert.

```yaml
nodes:
  classify:
    kind: agent
    prompt: ...
    output: str
    asserts:
      - output in ["go", "no_go"]   # POST — reads the node's own output
```

This holds for a `call` node too: its POST `asserts:` fire when the call's value
is committed, and may read `${output}` **and** the call's declared inputs
(`${name}`), matching leaf-node semantics. (`map` nodes still reject node-local
`asserts:` at load time — assert a `map`'s result with a flow-level or downstream
check instead.)

## Next

- [Examples](examples.md) — these constructs in working flows.
- [Python API](api.md) — run and resume flows from code.
