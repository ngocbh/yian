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
| [`branching.yaml`](templates/branching.yaml) | classify → `case` route → `${route.output}` |
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
| [`chat.yaml`](templates/chat.yaml) | a conversational REPL — HUMAN_INPUT → CASE (exit routes to a `!Turn` value entry, else a reply call), with flow-owned exit, the shape `yian chat` runs |
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
| `${[a, b]}`, `${upper(x)}` | list literal / pure builtin call (null or wrong-typed args fail: `${upper(x :- "")}`) |
| `${Buy(action="buy", qty=n)}`, `${Buy(x)}` | constructor: build a typedef record from exactly these fields, or check and narrow `x` (like `!Buy ${x}`) |
| `${X:-default}` | value, else `default` if absent |
| `${X:?"msg"}` | required — fail with the literal `msg` if absent (quote the message) |
| `${a \| b \| c}` | first present among peers (e.g. the branches of a routing-only `case`) |
| `$$` | a literal `$` (universal escape — in a prompt `$$` renders a single `$`) |

Arithmetic is Python's: `+` concatenates text and lists, `*` repeats them, `%` formats text
(`${"%.2f" % p}`), a bool is 0/1 (but a CASE `on:` never equates a bool with a number). No
cost limits; an operator Python can't complete (overflow) fails like an expression error.

`and`, `or`, `not`, and `in` are reserved operators, so none can be the root
name of a reference. A dotted field may use those names (`${record.output.in}`).

Inside an expression, name values directly and group with parentheses:
`${a :- (b :- "lit")}` or `${a | b | "lit"}`. A `${...}` inside an expression
(`${a :- ${b}}`, `when: "${a.output} > 5"`) is a load error that shows the rewrite.
The `:-`/`:?` RHS is an expression, so a string default must be **quoted**
(`${x:-"today"}`; bare `today` reads a variable). A whole-string `${ref}` resolves
to the **typed value**; embedded in surrounding text it is **stringified**. A
child-flow call is the whole-value `call(...)` directive (below), never a
`${flow(args)}` span. Arithmetic on a value that is null when a `case` skips its
node is a load error at that binding or searched `when:`
(`output 'q': ${order.output.qty + 1} does arithmetic on order.output.qty, which is null
when node 'order' is skipped; give it a default with :-, e.g. (order.output.qty :- <default>)`).

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

A tag (`name: !T value`) replaces the inferred type with its declared one: a parent
sees a child's declared output type, an inline parameter or omitted LOOP field takes
the declared type, and the binder checks the value at runtime. `${...}` nested in a
slot's mappings and lists is evaluated, and each field keeps its inferred type.

## The three expression contexts (one grammar)

One `${...}` grammar; the context decides what happens to the result:

| Context | Where | What it does |
|---------|-------|--------------|
| **Bindings** | `input:` / `output:` values | **evaluated** to a typed value — refs, literals, arithmetic, lists, `:-` / `:?` / `\|`, pure builtins. Child-flow call = `call(...)` directive. A tag (`name: !T value`) declares the type; nested `${...}` in mappings and lists is evaluated. A builtin called with arguments it doesn't take is a load error. |
| **Conditions** | `when:` / `asserts:` | **tested** as a boolean: `== != < <= > >=`, `in`/`not in`, `and`/`or`/`not`, parens, arithmetic operands. Canonical **bare** form (no `${}`); one whole `${...}` loads and evaluates identically; a `${...}` inside the expression is a load error, as is a call to a function other than a builtin, `isinstance`, or a `typedefs:` constructor `T(...)`, or to a builtin with arguments it doesn't take. |
| **Prompts** | `prompt:` text | free text with embedded `${...}` spans (each stringified); a builtin called with arguments it doesn't take is a load error |

> Bindings wire, conditions test, nodes compute. A heavy transform still belongs
> in a `code` node — but simple arithmetic/coalesce in a binding is fine.

## Inline `call(...)` directive

A child-flow call written as a binding's **whole value** (or a `case`
`then:`/`else:` target) — `call(f, arg=${ref}, k=lit)` — desugars at load into an
anonymous `call` node; the host becomes `${<synth>.output}`. Keyword args only,
each at most once; a bare arg is one plain scalar read with the flow's YAML value
rules (`x=1e3` is a float, `x=None` text), and a quoted arg is text;
every authored node `input:` mapping participates, including HUMAN_INPUT context
and LOOP carried-record seeds. Nesting is allowed (`call(a, x=call(b, y=1))`). The synth
node runs only when its host node can: it carries the host's `case` routes, `depends_on:`,
and `runs_after:` (inner calls of a nested branch call carry the branch), and waits for the
host's other inputs, skipping whenever a skipped input skips the host (a `:-` default only
waits). Calls in the same node don't wait on each other unless one reads the other, and a
nested call also carries its outer call's other arguments (in
`call(f, x=call(g), y=${a.output})`, `g` waits for `a`).
An **unquoted** value starting with `call(` must be exactly one well-formed
directive, else a load error names the problem (unbalanced parens, trailing text,
missing comma, positional or repeated keyword, a keyword without a name, a bare
value YAML can't build like `d=2026-13-45`). A
**quoted** value is text, so `"call(...)"` is the escape for literal text; a quoted
call holding `${...}` is a load error. The old `${flow(args)}`
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
  at the authored binding. Declare the type with a tag
  (`n: !int ${input.a + input.b}`, `tags: !list(str) []`): the parameter takes
  the declared type and the binder checks the value at runtime. A typed named
  `defs:` body also works.
- MAP's `${item...}` provider is demand-driven. If no outer argument is a lone
  whole-value item reference, `over:` need not be statically inspectable. The
  first such item read proves `over:` is a list with an exact element type;
  later reads reuse that proof.
- `${...}` nested inside a mapping or list argument is evaluated, and each field
  keeps its type; `$$` is a literal `$` at any depth. A nested `call(...)` is a
  load error.
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

Scalars: `str`, `int`, `float`, `bool`, `date`, `datetime`, `object`, `Any`, `None`. Forms
use parentheses; bracket spellings are a load error that shows the new one. `Any` is any
JSON value except null (`any` is a load error); `object` is a JSON object (text keys,
JSON values).

| Form | Example | Notes |
|------|---------|-------|
| list | `list(str)` | |
| any JSON value | `Any`, `list(Any)`, `list(Optional(Any))` | `Any` fills only `Any`; tag or `isinstance` to narrow it |
| nullable | `Optional(str)` | may be `null` |
| enum | `Literal(go, no_go, wait)` | one of these tags |
| bounded integer | `int(1..12)`, `int(0..)`, `int(..100)` | inclusive range; either end may be open, not both |
| bounded float | `float(0.0..1.0)`, `float(0.0..)` | inclusive finite range; either end may be open, not both |
| mapping | `dict(str, int)` | |
| alias | `Basket: list(str)` | aliases compose |
| record | (see below) | fields recurse |
| union | `Union(Buy, Sell)` | one of several records, told apart by a tag field or by shape |
| required input | `qty: !int` | a flow input: the tag with no value |
| default-fill | `lookback: !int 30` | filled when the input is omitted |
| typed slot | `qty: !int ${size.output.qty}` | a binding, flow output, or variable declaring its value's type (see below) |

```yaml
typedefs:
  Signal:
    score: float
    note: Optional(str)
```

Typedefs and node/compact-flow `output:` declarations are written as plain types,
with no tag. Every slot is typed with a tag.

`Optional(X)` (nullable) and a default (omission-fill) are **orthogonal** —
nullable says the value may be null; default says what to use when the input
isn't supplied at all. `note: !Optional(str)` may be omitted and gives null;
`note: !Optional(str) null` has a present null default.

A null default requires a nullable or `None` type. Defaults are literal YAML
validated at load: `${...}` inside is text, and nothing is coerced: use `!int 30`,
not `!int "30"`, and quote date strings (`!date "2026-01-01"`). Only strings supplied
to the top-level flow (CLI, JSON inputs, `run_flow`) may be coerced; a `call`, `map`, or
`loop` edge is not a flow boundary. A compatible default keeps its authored payload,
so `!float 30` remains an integer.

A `Union(...)` lists records. Give every member a **tag field**: the same field name
declared as a single-value `Literal`, with a different value in each member. Members
with no tag load only when their shapes can't overlap.

```yaml
typedefs:
  Buy:
    action: Literal(buy)
    qty: int
  Sell:
    action: Literal(sell)
    price: float
  Order: Union(Buy, Sell)
```

`Buy | Sell` and `Union(Buy, None)` are load errors with hints (`use Union(Buy, Sell)`,
`use Optional(Union(...))`). A bad union fails at its typedef even if nothing uses it.

**Typed slots.** A slot's type is a YAML tag on its value: `name: !T value`, or
`name: !T` with no value (an empty tagged scalar, not null). Slots are flow inputs,
node `input:` bindings (LOOP seeds, HUMAN_INPUT context, CALL/MAP/inline-body
arguments), flow `output:` fields and a single `output: !T ${x}`, agent
`variables:` (always with a value), and records or lists nested in any of these.
`output: !T {...}` is one typed value. Inside a tag there are no commas: name a
multi-argument type (`Union`, a multi-value `Literal`, `dict(K, V)`) in `typedefs:`
and tag by name. A tag ends at a space, so `!Union(A, B)` is a load error that says
to name it; `[ ] { } ,` in a tag are load errors. Inside `{...}`, quote an unquoted
multi-argument type or use block style, and put a space after a tag before `,`/`}`.
Tags only check: no fill-in, no conversion. Quoted values stay text (`!int "5"`
fails), and values follow YAML 1.2 core rules plus unquoted dates (`!str 30` fails with a hint to quote; `!bool yes` fails: `yes` is text, write `true`). A
record missing a required single-value tag field fails, naming the field. Load
rejects a value whose inferred type is disjoint from the declared one; narrower or
unknown types are checked at runtime, failing the consumer (END for a flow output)
with `binding.type`. Downstream sees the declared type. Declared types are in the
flow fingerprint. Tags anywhere else (typedefs, node `output:`, other fields, keys,
`<<:` sources) are load errors; tags under `x-*` keys are ignored, and `!!` tags keep
their YAML meaning.

**Old forms** are load errors that show the new spelling: `name: T = value`,
`name: T = default` inputs and variables, untagged inputs (`qty: int` -> `qty: !int`),
inline record input types (`cfg: {a: int}` -> name the record, `cfg: !Cfg`; `!object`
is an unstructured mapping), and brackets anywhere. `note: a = b` (no type on the
left) is plain text.

NaN, infinity, and complex numbers (`${(0 - 8) ** 0.5}`) are not yian values. A
binding whose value holds one at any depth fails its node with `binding.evaluate`;
a temporary inside an expression is fine. Keep the value in range or return null
explicitly: a `case` routed to a value entry `no_value: !Optional(float) null`, or a
`code` node returning `None`. An int of any size is a value and is recorded
exactly; an int literal in the YAML may have any number of digits, and text (a
template, a prompt, `join`, `%s`/`%d` and the other text and integer `%` conversions)
writes every digit; `%e`/`%f`/`%g`/`%c` on an int too large for a float or character
fail as in Python.

A value nests at most 100 levels, counting each record and list. A binding or MAP
`over` list nested deeper fails `binding.evaluate` and a node output fails
`contract.output`, naming the path; `${[x]}`, a record flow output, or a MAP's result
list around a 100-level value is one level too deep. A run input may nest 100 levels.
An expression nests at most 100 levels as well (each call, list, unary operator,
comparison, `|`/`:-`/`:?`, and operator chain is one); a deeper one is a load error at
its field. A flow file's YAML nests at most 200 levels (each mapping and list is one);
deeper is a located load error.

Bounds are ordered numeric literals with at most one end left open (`int(0..)`,
`int(..100)`; `int(..)` is an error) and exclude booleans. Integer bounds
retain arbitrary precision. Float bounds must be finite; integer endpoints used
in a float domain must convert without rounding. Bounds compose with other
type forms and constrain defaults, supplied values, and generated schemas.
Cross-flow wiring requires the source domain to fit inside the destination;
an unbounded source (or an open end) does not prove a bounded destination safe,
and a plain `str` does not prove a `Literal(...)` destination safe (its tags must
be a subset).
A tag (`mode: !Mode ${input.s}`, with `Mode: Literal(a, b)` in `typedefs:`) accepts
either and checks the value at run time.

## Recipes

**Give an AGENT a local prompt setting.** Add `variables:` beside `input:`, with
an entry such as `num_sentences: !int(1..12) 4`, then use `${num_sentences}` in
the prompt. Each variable requires a tag and a validated default; nullable variables
also need an explicit default. Variable and input names cannot overlap. Default
strings stay literal, and each invocation gets a fresh copy of all defaults.
Ownership follows the authored AGENT path plus name across repeated executions.
A variable named `temperature` does not configure the model; use its provider block in `llm_config:`.

**Get a structured or numeric value out of an AGENT node.** Declare the typed shape
directly as the node's `output:` — a `Literal(...)`, record, union, non-string scalar, or
list switches the AGENT node to **structured generation** (the engine derives a schema and
the model emits a conforming value, retried up to `retries:` times on deviation; see
`typed_output.yaml`). `Optional(X)` lets the model answer null at the root, in a list
element, or in a record field; a root `Optional(str)` stays plain text and never returns
null. Use a downstream `code` node only when you need deterministic post-processing of the
value.

**Branch and rejoin.** A `case` takes one branch; the other targets skip, and so do
nodes that plainly read them. A searched `case` whose `when:` reads a node only under
a default that keeps a binding running (`(score.output :- 0) >= 0.5`) still runs when
that node is skipped. Read the taken branch's result with `${route.output}`
(see `branching.yaml`); tag it where you read it (`!Turn ${route.output}`). A branch
names a node or calls a flow; for a fixed or computed result, route to a value entry
(`approve: !str approve`, a `!object` or typed record). A value in a branch, or a
`case` `output:`, is a load error that shows the rewrite. A `case` whose result is used
needs an `else:`, every tag of a non-nullable `Literal` `on:`, or `when: true` and
`when: false` on a non-nullable `bool` `on:`; routing on a `Literal` node output or flow
input is always exhaustiveness-checked. A routing-only `case` takes none when nothing
matches and there is no `else:`. A flow output field that reads only nodes a `case` may
skip returns `null` when they were skipped (typed `Optional(T)`, as if `:- null` were
written), in root, `defs:`, and inline bodies; a join over an exhaustive `case` never
does. A LOOP body field like that sets the carried field to null. Templates and
comparisons keep their null behavior; arithmetic needs a `:-` default.

**Branch on a union.** Route with `on: ${d.output.action}` (the tag field). Each
`when:` target, nodes chained to it by plain refs or `depends_on`, and inline `call(...)`s
lifted out of it (which carry its route and wait for its inputs), read its member's
fields (`${d.output.qty}` under `when: buy`); `else:` sees the unlisted members. Outside a
branch only fields every member has can be read. For an untagged union, use a searched
`case` with `when: isinstance(d.output, Text)`. Return a member field from the flow by
joining branch nodes (`${b.output | s.output}`); flow outputs are never narrowed. A
searched `case` whose `when:` reads a node only under such a default doesn't pass that
node's narrowing to its branches.

**Order without data flow.** When node B must run after A but consumes no value
from it (a `wait`, a side-effecting tool), use `depends_on: [a]` (co-skips B if A
skipped) or `runs_after: [a]` (orders only; B still runs).

**Wait until an absolute deadline.** Give `wait` a `date` (`2026-05-21`) or
`datetime` (`2026-05-21T14:30[:00][Z|+07:00]`) via `until:`. Dates mean midnight
UTC, naive datetimes are UTC, and aware datetimes are normalized to UTC. WAIT
blocks one worker and returns `None` at the deadline; that value is committed as its
ordinary node output. See `wait-until.yaml`.

**Ask the human.** For a *guaranteed* gate use a `human_input` node (it blocks its
worker until the host answers). For "ask only if the model decides it needs to",
give an AGENT node `tools: [ask_user]`. Both calls stay inside their live
node execution. A `human_input` gate can present multiple-choice questions
(static `questions:`), have an LLM compose them
(`adaptive_questions:`, which desugars to a synthetic composer AGENT node + gate at load;
the composer carries the gate's `case` routes, `depends_on:`, and `runs_after:`, and waits
for the gate's other inputs), or read
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
one iteration, whatever the body; the run log still keeps every iteration. The next
iteration starts once the body output is ready, so a body node the output doesn't read
(a log/notify/save step) can overlap it; it stays live until it finishes, a later
iteration prunes it, and the run waits for it before it ends. The node budget (10,000 live nodes)
is checked before pruning; it limits one iteration's size, not the iteration count. See
`loop.yaml`.

**Conversational REPL (LOOP + human_input).** A chat is a `loop` whose body blocks at
`ask` (`human_input`), then a CASE picks the carried 2-field record
`{transcript: str, exited: bool}` (`'a -> 'a`). Exact `/exit` and `/quit` route to
`exited: !Turn {...}`, a value entry that keeps the transcript and sets `exited: true`;
`else:` calls a `respond` def whose `reply` (`agent`) and fold CODE node append the
message and reply. The body's output is `!Turn ${route.output}`; the tag is required, as
it types whichever branch ran as the carried record. Typed exit is flow-owned; only
terminal EOF is normalized by the bundled host. The loop body's `output:` may return
only the fields that change; a field it omits keeps its value. This is the shape
`yian chat` runs; see `chat.yaml` (and `examples/chat.yaml` for a runnable copy).

## Gotchas

- **Type a slot with a tag, not `T = value`.** `qty: int = ${x}` is a load error that
  shows `` `qty: !int ${x}` ``; `note: a = b` (no type on the left) is plain text.
  A multi-argument type in a tag is a load error: name it in `typedefs:` and tag by
  name (`!Decision`). Prefer block-style mappings for tagged values; inside `{...}`
  put a space after a tag before `,` or `}`. A value starting with `call(` is a
  directive (quoted, it is text).
- **No secrets in the file.** Under a secret name (`password`, `secret`,
  `client_secret`, `private_key`, `access_token`, `refresh_token`, `api_key`,
  `passwd`, or a name ending in `_api_key`, `_password`, `_secret`, matched across
  case, camelCase, and `-`/`_`) a value must be `${...}` or `null`. A literal,
  including `''`, a `$${...}` escape, or a nested `{password: x}`, fails at load:
  `node 'login': input 'password' is a secret name, so its value can't be written in
  the flow file; pass it as an input (${input.password}) or from the host`. In
  `env:` and defaults `${...}` stays text, so pass a credential through a node
  `input:` from a flow input, or from the host.
- **Inline `{ ... }` maps + `${...}` need quotes.** In an inline flow mapping the
  `}` in `${input.x}` closes the map early. Either quote the value
  (`input: {x: "${input.x}"}`) or use block form (preferred):
  ```yaml
  input:
    x: ${input.x}
  ```
- **An AGENT node's `output:` may be any shape** — only a bare, untagged `str` keeps it a text
  producer; a `Literal(...)`/record/union/non-string scalar/list switches it to structured
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
  `None`. Every key in a flow must be text, field names included: an unquoted
  `true`, `null`, `1`, or date key is a load error, so quote it (`'true': ${x}`).
  A scalar literal (`output: 5`, `x: 5`) is returned as written.
- **`call:` resolves defs-first, else a `uses:` alias** (a sibling file by name on
  the search path). `alias@v1` adds a version guard.
- **No `kind: model`** — removed; serve ML inference behind a host tool and call it
  with `kind: tool`. Use `code` for deterministic compute.
- **A `tool` node's `input:` names are its tool's arguments; `output:` is optional.**
  When declared, every returned value is checked against it. At run start each
  static TOOL node is checked against the run's tool: unknown or missing inputs,
  and input or output types that can never fit, raise `ValueError`. No coercion:
  a `"10"` literal into an `int` argument fails.
- **`loop` bodies return a subset of the carried record.** The body's `output:`
  returns at least one of the loop's `input:` (carried record) fields, and no others;
  each omitted field keeps its previous value. The body reads only a subset of the
  carried names. Names are checked at build; each returned field must keep its carried
  type (checked at load). Omitted keeps a field; a returned null sets a nullable field
  to null. A literal `${null}` is typed `None`, so it does not fit an `Optional(str)`
  seed; return a nullable-typed producer instead. A body may return one field
  (`output: {done: ...}`); an empty `output: {}`, or a reference to an empty record,
  is rejected. A predicate that reads only fields the body never returns cannot
  change, so such a loop is a load error unless it has `max:` (it then fails if it
  reaches the cap).
  Give EXACTLY ONE driver: `while:` (pre-check, 0+ runs), `until:` (post-check /
  do-while, 1+ runs), or `times: N` (fixed count). `max:` is optional for
  `while:`/`until:` but REDUNDANT+REJECTED with `times:`.
  Every `while:`/`until:` ref must name a carried field, and a field read below it must
  exist on that field's type (a typo is rejected at load, not read as falsy; a field only
  some union members have needs an `isinstance(r, T) and ...` part before the read), and
  `max:` (when given) and `times:` must be a plain integer `>= 1`.
- **Node-local `asserts:` reading `${output}` are POST checks** — they fire once the
  node's value is committed, and fail the run loudly on a false/raising expr. This
  includes a `call` node: its POST asserts may read `${output}` **and** the call's
  declared inputs (`${name}`), like a leaf node. `map` nodes reject node-local
  `asserts:` at load time — assert a `map`'s result with a flow-level/downstream check.
  A `call`/`loop` node's own asserts can't use `isinstance` (they run on the child's
  `end`); assert inside the child flow.
- **`isinstance(x, T)` takes a bare type name** — a `typedefs:` name or
  `str`/`int`/`float`/`bool`/`date`/`datetime`/`object` (not `Any`). `list(...)`/
  `Optional(...)` need an alias. Constructors `${T(...)}` take only typedef names
  (`${int(x)}` is an unknown function). It is not a prompt function: compute it in an
  input binding. WAIT `until:` and MAP `over:` reject it.
- **A member-only union field fails to load outside its branch.** `${d.output.qty}`
  where only `Buy` has `qty` must sit under `when: buy` (or an `isinstance` case).
  A `when:` value that isn't a tag of the union is a load error (``case node 'route':
  `when: buyy` is not a tag of `on: ${d.output.action}` (`Literal(buy, sell)`); use one
  of: buy, sell``), and inline child bodies are typed before narrowing.
- **Flow-level `asserts:` read `input.X` (boundary, before any node) or `node.output`
  (post, after the terminal).** An exact unqualified input name is shorthand (`age`
  means `input.age`) and takes precedence over a same-named node; dotted input reads
  remain explicit (`input.profile.age`). `${system.X}` is **not** allowed — it is
  ambient (no producer node to draw the check's dependency edge from), so a `system`
  ref is a LoadError; put a system-dependent check in a body node's `asserts:` or a `case`.
  A flow's `asserts:` are lowered onto its synthesized `start`/`end` boundary nodes and
  fire on the ordinary node path (`start` post-binding, `end` post-terminal), so a
  called `defs:`/`uses:` child's asserts fire identically to the top-level flow's.

## Model selection — the `llm_config` cascade

Save a provider's credential and model with `yian auth <provider>` (`yian auth`
lists them; `--default` picks the default provider; `--remove` deletes one). For
OpenAI it selects subscription or API-key authentication; both use
`provider: openai` and the same Yian execution and memory paths. A saved login's key
beats the provider's key variable; a host `api_key` beats both. A flow's
`llm_config:` cannot set `api_key`, nor `base_url` except beside `provider: azure`
(the required Azure resource endpoint; `model` is the deployment name, and no
Azure endpoint or deployment environment variable is read); a Python host passes the key in
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
4. process provider/model defaults, read exactly once when the run initializes:
   `YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL`, then the saved login

There is no built-in provider or model. An unset model uses the selected provider's
own default, never another provider's. An agent that no layer gives a provider and
model fails the run before any node runs, naming the agent and the three fixes
(`llm_config`, `--provider`/`--model`, `yian auth <provider>`).

Set flow-wide defaults with a top-level `llm_config:`; override one field per node;
opt a node out of the authored outer layers with `inherit: false` (a real boolean; own dict plus
the snapshotted process provider/model defaults; no flow/parent/CLI layers). The
static tree is resolved up front, so later environment changes do not alter the
effective selection. See [`llm-config-cascade.yaml`](templates/llm-config-cascade.yaml).

```yaml
llm_config: {provider: anthropic, anthropic: {max_tokens: 8000}}   # flow layer
nodes:
  a: {kind: agent, prompt: hi, llm_config: {model: claude-opus-4-8}}        # fills `model`
  b: {kind: agent, prompt: hi, llm_config: {provider: openai, model: gpt-5.5, inherit: false}}
```

### Provider blocks

The top level of `llm_config:` holds only `provider`, `model`, and `inherit:` (plus the
host-only `api_key`/`base_url`). Every request setting goes in a block named after a
provider. A node uses the block of its resolved provider; the other blocks are inert but
still checked at load. A block mirrors the request body of the API yian calls (same
names, nesting, and values) and is sent as written; the load check names the block,
the key path, and what is allowed, without echoing the value.

| Block | API (write fields as its reference does) |
| --- | --- |
| `openai` | OpenAI Responses (`reasoning: {effort: high}`, `max_output_tokens`, `text: {verbosity: low}`); on a ChatGPT subscription, re-checked at run start without `temperature`/`max_output_tokens` |
| `azure` | Azure OpenAI Chat Completions (`reasoning_effort`, `max_completion_tokens`) |
| `xai`, `deepseek`, `qwen`/`qwen-cn`, `glm`/`glm-cn`, `minimax`/`minimax-cn`, `openrouter`, `vllm` | that vendor's Chat Completions (e.g. `qwen: {enable_thinking: true}`, `deepseek: {thinking: {type: enabled}}`) |
| `anthropic` | Messages (`max_tokens`, `thinking: {type: adaptive}`, `output_config: {effort: high}`) |
| `google` | Gemini `generateContent`, REST camelCase (`generationConfig: {thinkingConfig: {thinkingLevel: HIGH}, maxOutputTokens: 8192}`) |
| `ollama` | `/api/chat`: `think`, `options` (`temperature`, `num_ctx`, ...), `keep_alive`, `logprobs`, `top_logprobs`; no default `think` |

Reserved, because yian builds them (setting one is a load error saying so): model,
messages/`input`/`contents`, `system`/`instructions`/`systemInstruction`, tools and
tool choice, `parallel_tool_calls`, structured output (`response_format`, `text.format`,
`output_config.format`, `responseSchema`, `responseMimeType`, Ollama `format`),
streaming, conversation state (`previous_response_id`, `conversation`, Responses
`store`), and the reply count (`n`, `candidateCount`).

Blocks cascade per block field, one level deep: a node's `anthropic: {output_config:
{effort: low}}` keeps the flow's `anthropic: {max_tokens: 8000}`, but replaces the
flow's whole `output_config`. An omitted field is not sent (Anthropic `max_tokens` is
the one default). A limit of one model is left to the provider, whose error replaces a
silent drop: newer Claude models refuse `temperature`, so don't set it for them.

| Retired key | Block form |
| --- | --- |
| `anthropic_effort` | `anthropic: {output_config: {effort: …}}` |
| `openai_reasoning_effort` | `openai: {reasoning: {effort: …}}` |
| `google_thinking_level` | `google: {generationConfig: {thinkingConfig: {thinkingLevel: …}}}` |
| `temperature` | `<provider>: {temperature: …}`; google `generationConfig.temperature`; ollama `options.temperature` |

## Per-node config — `env:`

Any node may carry an `env:` mapping of **static** config keys its own
implementation reads (the engine never interprets `env`). A flow-level `env:`
is the default for every node;
each node overrides individual keys. Merge is **static at build time** (`{**flow_env,
**node_env}`, node wins), **flow-local** (a child flow does NOT inherit the parent's
flow env; a multi-node `defs:` entry sets its own `env:` beside its `nodes:`),
literals only (no `${...}`), and no literal under a secret name (see Gotchas).

```yaml
env: {max_tool_iterations: 150}         # flow default for every node
nodes:
  r: {kind: agent, prompt: hi, tools: [search], env: {max_tool_iterations: 300}}  # node override
```

Keys in use: `max_tool_iterations` (AGENT node) — tool-calling loop turns before an
`AgentLoopError` (default `-1` = no cap; set a positive int to bound it; any other
value fails at load). Tool calls in the last allowed turn's reply are not run.

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
`input_budget_chars`. Values are integers of at least 1 (`recent_turns` at least
0), checked at load; episodic and reflective limits may be `null`. Other types are custom and built by a Python host. The YAML
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
The `--inputs` file must be UTF-8 text holding one JSON object; a file that is not
UTF-8, not valid JSON, or not an object fails with exit code `2`. Explicit JSON
inputs load first; repeatable `--input` flags override individual
keys but remain a string/scalar surface. With an interactive stdin (TTY), `yian`
prompts for declarations still missing and keeps four outcomes distinct: an explicit
value, `Set null`, omit/use-default when allowed, and cancellation. Nullable
boolean/`Literal(...)` widgets put `Set null` in their value select. A null-capable
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
