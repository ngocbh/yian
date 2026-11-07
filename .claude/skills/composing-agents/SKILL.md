---
name: composing-agents
description: 'Use when writing or editing a yian flow (a YAML file run with `yian run`): wiring agent, code, tool, case, call, map or loop nodes, asking a person, waiting on an inbox or a time, or answering "how do I write a flow that ...". Trigger on "write a flow", "create an agent", "compose a workflow", "add a node to my flow", "how do I branch / loop / call another flow".'
---

# Writing yian flows

A flow is a YAML file that works like a typed function: typed `input:`, a graph of `nodes:` keyed
by id, and an `output:`. A node reads another with `${node.output}`, and that reference is the
edge; yian works out the order. This skill gives the workflow and the rules authors most often get
wrong. The docs at https://yian.readthedocs.io/ have every rule; [reference.md](reference.md) is a
lookup sheet; [templates/](templates/) holds a flow that loads for each common shape.

## Workflow

1. **State the signature.** One line on the job, then the typed inputs and the output. Name
   records, enums and unions under `typedefs:`.
2. **Start from a template** (index in [reference.md](reference.md#templates)). Copy it beside
   your flow and edit it.
3. **Write one node per step.** A model call is an `agent`; deterministic work is `code` (not a
   prompt); a host tool is `tool`; a branch is `case`; a child flow is `call`, `map` or `loop`.
   Prefer several small typed nodes to one large prompt.
4. **Check.** `yian check` loads the flow, checks every type and reference, and runs nothing.
   Each error names its place, and most show the rewrite. Fix and repeat until it prints `ok`.
5. **Run.** Agents need a provider and model: run `yian auth <provider>` once, or pass
   `--provider` and `--model`. A flow with only `code` and `tool` nodes needs neither.

```console
$ yian check flow.yaml
flow.yaml: ok
$ yian run flow.yaml --input topic=tides
$ yian run flow.yaml --inputs inputs.json
```

`--input k=v` is text, converted when the input is an `int`, `float` or `bool`; pass lists,
records and `null` in a JSON file with `--inputs`. See [CLI](https://yian.readthedocs.io/cli/) for every flag.

## A flow

```yaml
id: triage-ticket
name: triage ticket
description: Classify a support ticket, then page someone or draft a reply.
typedefs:
  Severity: Literal(low, high)
input:
  ticket: !str
  team: !str support                  # a default
nodes:
  severity:
    kind: agent
    input:
      ticket: ${input.ticket}         # bind first; the prompt reads the local name
    output: Severity                  # a non-str type asks for structured output
    prompt: Classify this ticket's severity as low or high. ${ticket}
  route:
    kind: case
    on: ${severity.output}
    cases:
      - when: high
        then: page
      - when: low
        then: reply
  page: !str "Paged the ${input.team} on-call."   # a value entry
  reply:
    kind: agent
    input:
      ticket: ${input.ticket}
    output: str
    prompt: Write a short, friendly reply to this ticket. ${ticket}
  words:
    kind: code
    input:
      text: ${route.output}           # the output of whichever branch ran
    output: int
    code: |
      return inputs.text.split(/\s+/).filter(Boolean).length;
output:
  severity: ${severity.output}
  action: ${route.output}
  words: ${words.output}
```

| Kind | Use it for | Page |
|---|---|---|
| `agent` | A model call, with optional tools, skills and steering | [Agents](https://yian.readthedocs.io/agents/) |
| `code` | A TypeScript or JavaScript body, or a `./file.ts:fn` export | [Code and tools](https://yian.readthedocs.io/code-and-tools/) |
| `tool` | A host tool, or the built-ins `ask`, `take` and `sleep` | [Code and tools](https://yian.readthedocs.io/code-and-tools/), [Inboxes and waiting](https://yian.readthedocs.io/inboxes/) |
| `case` | Take one branch, skip the others | [Control flow](https://yian.readthedocs.io/control-flow/) |
| `call`, `map`, `loop` | Run a child flow once, per list item, or repeatedly | [Control flow](https://yian.readthedocs.io/control-flow/) |
| a tagged entry | A fixed or computed value, such as `count: !int 3` | [Flows and nodes](https://yian.readthedocs.io/flows/#values) |

## Rules that most often go wrong

### Types and tags

- **Slots take a tag; type definitions don't.** A slot holds a value: a flow input
  (`qty: !int`), a binding (`n: !int ${x.output}`), a value entry, a flow output
  (`output: !Order ${x.output}`). A type definition is plain text: a `typedefs:` entry and the
  `output:` of an `agent`, `code` or `tool` node. Mixing them up is a load error.
- **Every flow input has a tag**, with an optional default: `lookback: !int 30`,
  `note: !Optional(str)` (omitted is null).
- **A tag ends at the first space and holds no `,`.** Name a type with several arguments in
  `typedefs:` and tag with the name: `Order: Union(Buy, Sell)`, then `!Order`. One-argument
  types work inline: `!Optional(list(str))`.
- **Quote a type with a `,` inside `{...}`.** YAML splits a flow mapping at each comma, so
  write `Msg: {role: "Literal(user, assistant)", text: str}` or the record in block style.
- **Spellings:** parentheses, not brackets (`Optional(str)`, `int(1..12)`); `Any`, not `any`;
  `Union(A, B)`, not `A | B`; `Optional(Order)`, not `Union(Order, None)`.
- **Union members must differ by a tag field or by shape.** A tag field is the same field as a
  one-value `Literal` in each member, such as `action: Literal(buy)`; shapes such as `{text: str}`
  and `{num: int}` can't overlap. A field only some members have can be read only after narrowing:
  on a `case` branch over the tag, or after `isinstance(x, Buy)` or `x.action == "buy"`. See
  [Narrowing](https://yian.readthedocs.io/types/#narrowing).
- **A tag checks; it never converts.** Unquoted YAML follows the core schema: `yes` is text
  (write `true`), `!str 30` fails (quote it: `!str "30"`), `!int "5"` fails.
- **Tag what can't be inferred**, such as an empty list: `seen: !list(str) []`.

### Wiring and prompts

- **Node ids are identifiers**: `fetch_data`, not `fetch-data`. `input`, `output`, `system`,
  `item` and the expression keywords are taken.
- **A node's `input:` is a mapping of names.** `input: ${other.output}` is a load error; bind
  each name: `n: ${other.output.n}`.
- **Prompts read local names only.** `prompt:` and `system_prompt:` see the node's own `input:`
  names and `variables:`, bare: bind `topic: ${input.topic}`, then write `${topic}`. Writing
  `${input.topic}` in a prompt is a load error. An `ask` node is the exception: its `prompt:` and
  `questions:` sit in its `input:`, so they read `${draft.output}` directly.
- **Order without data** with `depends_on: [a]` (skipped when `a` is skipped) or
  `runs_after: [a]` (runs either way). Both take a list.
- **Compact form** (the node's fields at the top level, no `nodes:`) is only for a flow whose
  body is one `agent`, `code` or `tool` node. See
  [Compact form](https://yian.readthedocs.io/flows/#compact-form).

### Expressions and null

- **A lone `${...}` is a typed value; inside text it is written as text.** Conditions
  (`when:`, `asserts:`, `while:`, `until:`) are bare expressions: `when: score.output >= 0.5`.
  A `${...}` inside an expression is a load error: write `${a :- b}`, not `${a :- ${b}}`.
- **Quote text in an expression**: `${x :- "none"}`. A bare `none` is null, and other bare
  words read a value by that name.
- **A `.` read on a value that may be null is a load error**: an `Optional` value, or a node a
  `case` may skip. Write `${x?.qty :- 0}`, or check first: `x != null and x.qty > 0`.
- **A null in a prompt fails the node.** Give it a default: `${note :- "none"}`.
- **Arithmetic is numeric.** `+` also joins two texts or two lists; `*` and `%` on text fail
  (put the value in a template instead); `/` always gives a float.
- **Prompts may call only `join`, `upper`, `lower` and `render_as_json`.** Compute `len(...)`,
  `isinstance(...)` and constructors in a binding.
- **Quote YAML that YAML would parse itself**: a slice or comprehension
  (`names: "${[m.text for m in input.msgs]}"`), and any `${...}` inside a `{...}` flow mapping
  (`input: {x: "${input.x}"}`). Block style avoids most of this.
- **A child flow call is `call(f, k=v)` as a binding's whole value**, never `${f(...)}`. A
  quoted `"call(f)"` is text. See [Expressions](https://yian.readthedocs.io/expressions/).

### Branching with `case`

- **A `case` only routes**: no `input:`, no `output:`. A `then:` or `else:` names a node, a
  value entry (for a fixed result, such as `approve: !str approve`), or an unquoted `call(...)`.
  A literal in a branch is a load error that shows the value entry to write.
- **Read the result with `${route.output}`**, as a whole binding or inside text, not inside an
  expression or condition. Tag it where one type is needed: `output: !Turn ${route.output}`.
- **A `case` whose result is read must be exhaustive**: an `else:`, every tag of a `Literal`
  `on:`, or `when: true` and `when: false` on a `bool`.
- **`on:` holds a reference** (`on: ${input.kind}`), and each `when:` is a value of its type.
  Without `on:`, each `when:` is a condition and the first true one wins.
- **Skipped branches spread.** A node that reads a skipped node whole is skipped too; inside
  text the value is empty; with `:-` it runs on the default; a flow output reads null. See
  [When a node is skipped](https://yian.readthedocs.io/flows/#when-a-node-is-skipped).

### Child flows: `call`, `map`, `loop`

- **`call:` names a `defs:` entry first, then a `uses:` alias** (`uses: {summarize:
  child-summarize}` loads `child-summarize.yaml` beside the flow). A ref is a path relative to the
  importing file, without the extension (only `.yaml` files are found); `@<version>` must equal
  the child's quoted `version:` text (`clean: ../common/clean@1`).
- **Write exactly one of `call:` and an inline body** (`nodes:` plus `output:`). An inline body
  can't see its parent's inputs, nodes or `${item}`: pass them through the node's `input:`, and
  read them inside as `${input.<name>}`.
- **Each argument must fit the child's input type**, checked at load and never converted: a `str`
  doesn't fill an `int`, and an `Optional` value doesn't fill a strict input. Tag a value you
  know is set to check it at run time: `n: !int ${input.n}`.
- **`map`** runs the child per item of `over:` (a list), with the item as `${item}`, and returns
  a list in order. A `map` can't have `asserts:`.
- **`loop`** takes exactly one driver: `while:` (checked before each run), `until:` (after each
  run) or `times: N`. `max:` guards `while:` and `until:`, and is a load error with `times:`.
  The body reads carried fields as `${input.<name>}`, and the condition reads them bare
  (`while: not done`). Without `max:`, a condition that doesn't end the loop at once must
  read a carried field the body returns. The body returns only carried fields, at least one; a field it leaves out
  keeps its value. See [Repeat with loop](https://yian.readthedocs.io/control-flow/#repeat-with-loop).

### Code nodes

- **Bodies are TypeScript or JavaScript**: the body of an async function that reads `inputs.x`
  and ends with `return`. Return `null` for no value; `undefined` fails the node.
- **A file reference starts with `./` or `../`**: `code: ./text.ts:countWords`, resolved against
  the flow's folder. The file is `.ts`, `.js` or `.mjs`, and the path has no whitespace.
  `module:function` without `./`, or a reference that breaks these rules, fails at load.
- **Only erasable TypeScript**: annotations, `as`, interfaces and type aliases; `enum` and
  `namespace` fail at load. A body runs on a worker thread and is not a sandbox.

### Agents

- **`output:` defaults to `str`.** Any other type (a record, `Literal`, list, number) asks the
  model for that type and checks the reply, retrying up to `retries:` (default 2). It can't
  hold `file` or `None`.
- **`tools:` is always a list** of tool or group names, such as an MCP server's name, or
  `{group: [tool, ...]}` for some tools of a group. `ask_user` lets the model ask a person when
  it chooses; for a question that must be asked, use an `ask` node.
- **A group's tools show to the model as `<group>_<tool>`.** A bare tool name finds a group's tool
  only when one group has it; when two do, grant `{github: [search]}`. A tool in two granted
  groups is offered once, under the first grant's name.
- **An agent a host may interrupt needs an `Optional(...)` output.** An interrupted agent's
  output is null; with `str` or any other type it fails with `agent.interrupted`. This applies
  to every agent with `steer_inbox:`.
- **`llm_config:` holds `provider`, `model` and, on an agent, `inherit:`.** Set `provider` and
  `model` together: an agent that sets only `provider:` keeps an outer `model:`, even one for
  another provider. Request settings go in a block named after the provider, written as that API's
  request body: `anthropic: {max_tokens: 8000}`, `openai: {reasoning: {effort: high}}`. See [Models
  and providers](https://yian.readthedocs.io/models/).
- **No `session` memory.** Carry a conversation as a `loop` field and compact it with a `call`;
  see `templates/chat.yaml`. `long_term_memory:` was removed: it is a load error.

### Waiting: `ask`, `take`, `sleep`

- **An inbox is an input typed `!Inbox(T)`**, required, with no default. Read it only whole, as
  `${input.notes}`: in a `take`'s `from:`, an agent's `steer_inbox:`, or an `Inbox` input of a
  `call` or `loop` child. An inbox nothing reads is a load error.
- **`take` and `sleep` have fixed outputs**, so `output:`, `asserts:` and `retries:` on them are
  load errors. A `take` gives `{events: list(T), closed: bool}`.
- **Durations need a unit**: `90s`, `30m`, `24h`, `2d`. `timeout: 86400` fails.
- **An `ask` with a `timeout:`** gives null when nobody answers in time if its `output:` is
  `Optional(...)`; otherwise it fails with `tool.timeout`.

See [Inboxes and waiting](https://yian.readthedocs.io/inboxes/).

### Secrets

Never put a credential in a flow file or an input: the run log records both. Keep it in the
environment: `yian auth` saves a provider login, an MCP server's config reads `${VAR}`, and a code
node or host tool reads the variable itself. `api_key`, `headers`, `organization` and `project` in
`llm_config:` are load errors, and so is `base_url` outside `provider: azure`.

## Before you hand the flow over

- `yian check` prints `ok`.
- Every node reads what it needs through `${...}` in `input:`, and the flow `output:` names the
  results the caller wants.
- Every `case` whose result is read is exhaustive, and every interruptible agent has an
  `Optional(...)` output.
- No secret is in the file.
