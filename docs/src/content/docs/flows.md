---
title: "Flows and nodes"
description: "The shape of a flow file: inputs, outputs, nodes and how they are ordered."
---

A flow is a YAML file that works like a function: it takes typed inputs, runs a graph of nodes,
and returns an output. This page explains the shape of the file and how its nodes fit together.

## A first flow

This flow adds tax to a price. The `total` node runs a line of code; `summary` is a value that
reads it.

```yaml title="order-total.yaml" check=runnable
id: order-total
name: Order total
description: Add tax to a price and describe the result.
input:
  price: !float
  rate: !float 0.2
nodes:
  total:
    kind: code
    input:
      price: ${input.price}
      rate: ${input.rate}
    output: float
    code: |
      return inputs.price * (1 + inputs.rate);
  summary: !str "Total with tax: ${total.output}"
output:
  total: ${total.output}
  summary: ${summary.output}
```

```console
$ yian run order-total.yaml --input price=10
{"summary":"Total with tax: 12","total":12}
```

There is no list of edges. `summary` reads `${total.output}`, so yian runs `total` first.

## The file shape

A flow file is a mapping of these top-level keys:

| Key | Required | What it holds |
|---|---|---|
| `id` | yes | A stable identifier for the flow |
| `name` | yes | A display name |
| `description` | no | One line on what the flow does |
| `version` | no | Any version text, such as `v2` |
| `input` | no | The flow's inputs, each with a type (see [Inputs](#inputs)) |
| `nodes` | yes | The body: nodes and values, keyed by id (see [Nodes](#nodes)) |
| `output` | no | What the flow returns (see [Outputs](#outputs)) |
| `asserts` | no | Conditions on inputs and results (see [Asserts](#asserts)) |
| `typedefs` | no | Named types (see [Types](/types/)) |
| `defs` | no | Child flows defined in this file (see [Control flow](/control-flow/)) |
| `uses` | no | Child flows from other files, by alias (see [Import a flow](/control-flow/#import-a-flow-from-another-file-with-uses)) |
| `system` | no | `paths:`, a list of folders, relative to this file, where `uses:` also looks after this file's folder |
| `llm_config` | no | Model defaults for every agent (see [Models and providers](/models/)) |
| `env` | no | Settings for every node in this flow, such as an agent's `max_tool_iterations`; a child flow doesn't inherit them (see [Agents](/agents/)) |

Any other key is a load error, so a typo such as `nodez:` fails loudly. Keys that start with
`x-` are the exception: yian ignores them, so use them for notes or to hold YAML anchors.
Run `yian check flow.yaml` to see a flow's load errors without running it.

```yaml
id: weekly-report
name: Weekly report
x-owner: data-team
nodes:
  title: !str Weekly report
output: ${title.output}
```

Every mapping key in the file must be text. Quote a key YAML would read as a boolean, number or
null (`"true":`). A mapping can't repeat a key, and a repeated node id is a load error.

Names that start with `__` belong to yian, which generates them. A node id, an input of a flow or
a node, a flow output or an agent variable can't start with `__`.

## Inputs

Each input is a name and a [YAML tag](/types/#yaml-tags) that gives its type, with an optional
default after it:

```yaml
id: inputs-demo
name: inputs demo
input:
  qty: !int                  # required
  tags: !list(str) []        # optional; defaults to []
  note: !Optional(str)       # optional; null when omitted
nodes:
  count: !int ${input.qty}
output: ${count.output}
```

| You write | Meaning |
|---|---|
| `qty: !int` | Required |
| `qty: !int 5` | Optional. An omitted `qty` is `5` |
| `note: !Optional(str)` | Optional. An omitted `note` is null |
| `note: !Optional(str) null` | The same, with the null written out |

A default is a literal, checked against the type when the flow loads. A `${...}` inside a
default stays text. A null default needs an `Optional` type.

Nodes read an input as `${input.qty}`. From the command line, pass inputs with
`--input qty=3` or a JSON file with `--inputs`:

```console
$ yian run report.yaml --input qty=3 --input note="first run"
$ yian run report.yaml --inputs inputs.json
```

Text from `--input` becomes an `int`, `float` or `bool` when the input has that type (see
[Types](/types/#inputs-from-the-command-line)).

Two kinds of input take no value. An inbox (`notes: !Inbox(str)`) gets messages while the run
goes on (see [Inboxes and waiting](/inboxes/)). A memory (`lessons: !FileMemory`) is bound by
`yian run`, to one it keeps per project, or by the program that runs the flow (see
[Memory](/memory/#bind-a-memory)).

## Nodes

Each entry under `nodes:` is either a node (a mapping with `kind:`) or a value (a tagged YAML
value). Its key is its id.

### Node ids

A node id is an identifier: ASCII letters, digits and `_`, not starting with a digit. Write
`fetch_data`, not `fetch-data`, so that `${fetch_data.output}` can read it. These ids are taken:

- `input`, `output`, `system` and `item`, which begin `${...}` paths;
- `inputs` and `outputs`, and any name starting with `__`;
- the expression keywords `and`, `or`, `not`, `in`, `true`, `false`, `null`, `none` and `inf`.

### Node kinds

| Kind | What it does | Page |
|---|---|---|
| `agent` | Asks a model, which may call tools | [Agents](/agents/) |
| `code` | Runs a TypeScript function | [Code and tools](/code-and-tools/) |
| `tool` | Calls a tool, such as a host tool or `ask`, `take` or `sleep` | [Code and tools](/code-and-tools/), [Inboxes and waiting](/inboxes/) |
| `case` | Sends the run down one branch | [Control flow](/control-flow/) |
| `call` | Runs a child flow once | [Control flow](/control-flow/) |
| `map` | Runs a child flow once per list item | [Control flow](/control-flow/) |
| `loop` | Runs a child flow again and again | [Control flow](/control-flow/) |

### Common fields

| Field | Kinds | Meaning |
|---|---|---|
| `kind` | all | The node kind |
| `input` | all but `case` | A mapping of names to the values the node reads |
| `output` | `agent`, `code`, `tool` | The type of the node's result, written as a type with no tag |
| `asserts` | all but `case` and `map` | Conditions on the node's inputs and result (see [Asserts](#asserts)) |
| `depends_on` | all | Node ids to run after; this node is skipped when one is skipped |
| `runs_after` | all | Node ids to run after; this node still runs when one is skipped |
| `node_name` | all | A display title for the node |
| `env` | all | Settings for this node; they override the flow's `env:` key by key |

A node's `input:` is a mapping. Each value is a literal, a `${...}` expression, text with
`${...}` inside it, or a record or list that holds any of these:

```yaml
id: inputs-shapes
name: inputs shapes
input:
  topic: !str
nodes:
  brief:
    kind: code
    input:
      topic: ${input.topic}                 # one value, typed
      heading: "Notes on ${input.topic}"    # text
      limit: 3                              # a literal
      options: {tone: plain, tags: ["${input.topic}", draft]}
    output: str
    code: |
      return `${inputs.heading} (${inputs.limit})`;
output: ${brief.output}
```

`input: ${other.output}` is a load error: bind each name you need, such as
`n: ${other.output.n}`.

### Values

A tagged entry is a value. Other nodes read it as `${id.output}`, like a node's result:

```yaml
id: values-demo
name: values demo
input:
  name: !str
nodes:
  count: !int 3
  greeting: !str Hello, ${input.name}
  doubled: !int ${count.output * 2}
  settings: !object {retries: 2, verbose: false}
output:
  greeting: ${greeting.output}
  doubled: ${doubled.output}
  retries: ${settings.output.retries}
```

The tag is the value's type: a literal is checked when the flow loads, and a computed value
when it runs. An entry with neither `kind:` nor a tag is a load error that shows the tag to add.

## Outputs

`output:` says what the flow returns.

| You write | The flow returns |
|---|---|
| `output: ${total.output}` | That one value |
| `output: {total: "${total.output}", note: "${note.output}"}` | A record with those fields (quote each `${...}` inside `{...}`) |
| `output: !float ${total.output}` | One value, checked against the type |
| `output: 5` | The literal as written |
| no `output:` | null |

A field of a record output may carry a tag too, such as `size: !Size ${size.output}`. A caller
reads a field of a child's record output as `${child.output.total}`.

## Compact form

A flow whose body is one `agent`, `code` or `tool` node can drop `nodes:` and put the node's
fields at the top level. Only `id`, `name`, `description`, `version` and `typedefs` stay with the
flow; every other key, `asserts:` included, belongs to the node:

<!-- yian-test {"replies": ["Hello, Ada! Lovely to meet you."]} -->
```yaml title="hello.yaml" check=runnable
id: hello
name: hello
input:
  name: !str
output: str
kind: agent
prompt: Write a short, warm greeting for ${name}.
```

```console
$ yian run hello.yaml --input name=Ada
Hello, Ada! Lovely to meet you.
```

- The flow `id:` also names the node, so it must be an identifier.
- Each flow input is passed to the node under its own name, so the prompt reads `${name}`.
- `output:` is the node's output type, and the flow returns the node's result.
- `case`, `call`, `map` and `loop` need the full `nodes:` form, and so do `uses:` and `defs:`.

## Dependencies and ordering

yian works out the order from the data. A node that reads `${total.output}` runs after `total`,
and nodes that don't read each other may run at the same time. A cycle, such as two nodes that
read each other, is a load error.

To order nodes without passing data, list ids in `depends_on:` or `runs_after:`.

### When a node is skipped

A `case` runs only one branch, so the nodes on the other branches are skipped. What happens to a
node that reads a skipped one depends on how it reads it:

| The reader | Result |
|---|---|
| Reads it as a whole value, `${a.output}` | Skipped too |
| Reads it inside text, `"got ${a.output}"` | Runs; the skipped value is empty text |
| Reads it with a default, `${a.output :- "none"}` | Runs with the default |
| Lists it in `depends_on:` | Skipped too |
| Lists it in `runs_after:` | Runs |
| Is the flow's `output:` | Reads null |

```yaml title="skips.yaml" check=runnable
id: skips
name: skips
input:
  go: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: input.go
        then: a
  a: !str ran
  whole: !str ${a.output}
  text: !str "got ${a.output}"
  fallback: !str ${a.output :- "default"}
  after:
    kind: code
    runs_after: [a]
    output: str
    code: |
      return "after ran";
output:
  a: ${a.output}
  whole: ${whole.output}
  text: ${text.output}
  fallback: ${fallback.output}
  after: ${after.output}
```

```console
$ yian run skips.yaml --input go=false
{"a":null,"after":"after ran","fallback":"default","text":"got ","whole":null}
```

## Asserts

An assert is a condition that must hold; a false one fails the run with `assertion.failed`.
Write it as a bare [expression](/expressions/).

A flow's `asserts:` read its inputs, checked before any node runs, or node results, checked at
the end. A lone input name is short for `input.<name>`, so `qty > 0` means `input.qty > 0`;
write `input.order.qty` for a field.

A node's `asserts:` read its own input names, checked before it runs, or `output`, checked after:

```yaml
id: asserts-demo
name: asserts demo
input:
  qty: !int
asserts:
  - qty > 0
nodes:
  double:
    kind: code
    input:
      n: ${input.qty}
    output: int
    code: |
      return inputs.n * 2;
    asserts:
      - n < 1000
      - output > n
output: ${double.output}
```

- A `map` node can't have asserts; put them in its child flow.
- A flow's asserts can't read `system.today` or other `system` values; check those in a node.
- A flow assert that reads a node a `case` skipped sees null.
