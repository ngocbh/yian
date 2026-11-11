---
title: "Control flow"
description: "Branch, repeat and reuse work with case, loop, map, call and defs."
---

This page explains how a flow branches, repeats and reuses work. Four node kinds do it:

| Kind | What it does | Its value |
|---|---|---|
| `case` | Takes one branch and skips the others | The output of the branch that ran |
| `call` | Runs a child flow once | The child's output |
| `map` | Runs a child flow once per list item | A list of the child's outputs, in list order |
| `loop` | Runs a child flow again and again, passing a record from one run to the next | The final record |

A child flow is a `defs:` entry, a flow imported with [`uses:`](#import-a-flow-from-another-file-with-uses),
or a body written inline on the node.

## Branch with `case`

A `case` node routes. It has no `input:` and no `output:`. It picks one branch; the nodes on the
other branches are skipped, and so is every node that reads a skipped node without a default
or names it in `depends_on:`.

### Match a value

With `on:`, each `when:` is a value compared with `on:`. The first match wins, and `else:` runs
when nothing matches:

```yaml title="triage.yaml" check=runnable
id: triage
name: triage
typedefs:
  Severity: Literal(low, high)
input:
  severity: !Severity
nodes:
  route:
    kind: case
    on: ${input.severity}
    cases:
      - when: high
        then: page
    else: queue
  page: !str page the on-call
  queue: !str add to the queue
output: ${route.output}
```

```console
$ yian run triage.yaml --input severity=high
page the on-call
```

`page` and `queue` are value entries: nodes that hold a fixed or computed value (see
[Flows and nodes](/flows/)).

| Rule | Example |
|---|---|
| `on:` holds a reference | `on: ${input.n}`; `on: input.n` is a load error |
| A `when:` is a value of the `on:` type | `when: 1` on an `int`, `when: "1"` on a `str` |
| Ints and floats compare as numbers | `when: 2` matches `2.0` |
| A `when:` the `on:` type can't hold is a load error, even with `else:` | `when: true` on an `int`; `when: nogo` on `Literal(go, no_go)` |

### Test conditions

Without `on:`, each `when:` is a condition. The first one that is true wins:

```yaml
id: grade
name: grade
input:
  score: !int
nodes:
  gate:
    kind: case
    cases:
      - when: input.score >= 90
        then: top
      - when: input.score >= 50
        then: pass
    else: fail
  top: !str top marks
  pass: !str passed
  fail: !str try again
output: ${gate.output}
```

A condition skips the `case` when it reads a skipped node, unless the read has a default:
`(score.output :- 0) >= 0.5` still routes when `score` is skipped. Conditions use the
[expression](/expressions/) syntax, and `and` and `or` stop early, so
`x.output != null and x.output.k > 1` reads `x.output.k` only when `x.output` is set.

A condition or an `on:` can read an agent's new messages: `len(reply.new_messages) > 1` is true
when agent `reply` added more than its reply, such as a tool call.

### Branch targets

A `then:` or `else:` names what runs on that branch:

- a sibling node, as an unquoted word (`then: reply`);
- a value entry, which is also a node (`then: approve`);
- a child flow call, as an unquoted `call(...)` (`else: call(respond, text=${input.text})`).

Anything else, such as quoted text, a number or `null`, is a load error: put the value in a
value entry and name that. A word that names no node is a load error that suggests the nearest
node.

A node that names the `case` itself in `depends_on:` or `runs_after:` isn't on a branch: it runs
whichever branch is taken.

### Read the result

`${route.output}` is the output of the branch that ran, and `${route.output.field}` is one field
of it. Use it as a whole binding or inside text. It can't appear inside an expression, a `:-`
default, a `when:` or `on:`, or an assert.

The branches may give different types. To check the result as one type, tag the read:
`output: !Turn ${route.output}`.

A `case` whose result is read must be exhaustive: some branch must always run. That means it has
an `else:`, or its `on:` is a `Literal` with every tag listed, or a `bool` with `when: true` and
`when: false`. Otherwise the flow fails to load:

```yaml check=fails
id: pick
name: pick
input:
  x: !str
nodes:
  gate:
    kind: case
    on: ${input.x}
    cases:
      - when: a
        then: one
  one: !str one
output: ${gate.output}
```

```text
case node 'gate' produces a value, so it must be exhaustive: add an `else:` (on: ${input.x} is str, not a Literal or bool)
```

A `case` that only routes may match nothing; then no branch runs. Routing on a `Literal` input or
node output with no `else:` must still list every tag.

### Skipped nodes in the flow output

A flow output that reads only nodes a `case` may skip is `null` when they were skipped, and its
type becomes `Optional`. Here, with `approve=false`, the output is
`{"order":null,"qty":null}`:

```yaml
id: orders
name: orders
input:
  approve: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: ${input.approve}
        then: order
    else: rejected
  order:
    kind: code
    output:
      qty: int
    code: |
      return { qty: 2 };
  rejected: !str no
output:
  order: ${order.output}
  qty: ${order.output.qty}
```

Elsewhere, read a field of a node that may be skipped through `?.` and give it a default:
`${(order.output?.qty :- 0) + 1}`. A plain field read or arithmetic there is a load error that
shows this fix.

### Narrow a union

When `on:` reads the tag field of a [union](/types/) of records, each branch sees only the
members its `when:` names, so it can read their fields:

```yaml check=fragment
route:
  kind: case
  on: ${decide.output.action}   # decide.output is Buy, Sell or Hold
  cases:
    - when: buy
      then: place_buy           # sees decide.output as Buy
    - when: sell
      then: place_sell          # sees decide.output as Sell
  else: log_hold                # sees Hold
```

A condition narrows its branch too: `when: isinstance(decide.output, Buy)` or
`when: lookup.output != null`.

## Reuse a flow with `defs` and `call`

`defs:` holds named child flows. Each entry has the same shape as a flow: `input:`, `nodes:` and
`output:`. A `call` node runs one with arguments in `input:`; a `map` node runs one per item of
`over:`, with the item as `${item}`:

```yaml title="greet_team.yaml" check=runnable
id: greet_team
name: greet_team
input:
  lead: !str
  team: !str                    # names separated by commas
defs:
  greet:
    input:
      name: !str
    nodes:
      line:
        kind: code
        input:
          name: ${input.name}
        output: str
        code: return `Hi, ${inputs.name}!`
    output: ${line.output}
nodes:
  names:
    kind: code
    input:
      team: ${input.team}
    output: list(str)
    code: return inputs.team.split(",").map((name) => name.trim())
  first:
    kind: call
    call: greet
    input:
      name: ${input.lead}
  rest:
    kind: map
    over: ${names.output}
    call: greet
    input:
      name: ${item}
output:
  lead: ${first.output}
  team: ${rest.output}
```

```console
$ yian run greet_team.yaml --input lead=Ada --input team="Bo, Cy"
{"lead":"Hi, Ada!","team":["Hi, Bo!","Hi, Cy!"]}
```

- `call:` names a `defs:` entry first, then a [`uses:`](#import-a-flow-from-another-file-with-uses)
  alias. Any other name is a load error, and so is a `defs:` name or alias that is also a node id.
- `input:` lists the child's arguments, one per name. A whole `${...}` record there is a load
  error.
- Each argument must fit the child's input type, checked at load. Nothing is converted: a `str`
  doesn't fit an `int` input, and an `Optional` value doesn't fit a strict one. Tag a value you
  know is set: `n: !int ${input.n}`.
- A `call` returns the child's output, or `null` when the child has no `output:`.
- A `map` needs a child with an `output:`, and returns a list in `over:` order.
- `over:` must be a list, checked at load, or when the node runs if its type can't be proven.
  `--num-workers` sets how many items run at once; `parallel:` is accepted and has no effect.

```text
call node 'c': binding 'n' — child expects 'int', source is 'str'
```

### Import a flow from another file with `uses:`

`uses:` names child flows kept in other files, each under an alias that `call:` takes like a
`defs:` name. Here `app/report.yaml` imports `common/clean.yaml`:

```yaml title="common/clean.yaml"
id: clean
name: clean
version: "1"
input:
  text: !str
nodes:
  trimmed:
    kind: code
    input:
      text: ${input.text}
    output: str
    code: return inputs.text.trim()
output: ${trimmed.output}
```

```yaml title="app/report.yaml" check=runnable
id: report
name: report
input:
  note: !str
uses:
  clean: ../common/clean@1
nodes:
  tidy:
    kind: call
    call: clean
    input:
      text: ${input.note}
output: ${tidy.output}
```

```console
$ yian run app/report.yaml --input note="  Shipped.  "
Shipped.
```

- A ref is the child's path without its extension, relative to the importing file's folder:
  `../common/clean` loads `../common/clean.yaml`. Only `.yaml` files are found, so a ref that
  ends in `.yaml` and a `.yml` file both fail with `not found`.
- `@<version>` pins the child: its `version:` must be the same text, or the flow doesn't load.
  Quote a number there (`version: "1"`). A ref without `@` takes any version.
- `yian run` and `yian check` look in the importing file's folder, then in each `system.paths`
  folder, and the first file found wins. A host's `loadFlow` looks in its `searchPaths` instead,
  then in `system.paths` joined onto the first of them (see [Embedding](/embedding/#run-a-flow)).
- A child's own refs resolve from its folder and its own `system.paths`, never the importer's.
- Under `yian check --root` (see [yian check](/cli/#yian-check)), a ref resolves only from the
  importing file's folder, and a flow that sets `system.paths` is refused. A `../` ref fails only
  when its file is outside the root: the example above loads with `--root .` but not with
  `--root app`.
- Files that import each other in a cycle fail to load, and so does a `hub:` ref.

Pinning a version the child doesn't declare fails at load:

```yaml title="app/pinned.yaml" check=fails
id: pinned
name: pinned
uses:
  clean: ../common/clean@2
nodes:
  tidy:
    kind: call
    call: clean
    input:
      text: draft
output: ${tidy.output}
```

```text
uses: ref pinned ../common/clean@2 but the resolved file declares version: "1" (file-per-version: name a file that declares "2")
```

### Call a flow inside a binding

To call a child once without writing a `call` node, write `call(...)` as a binding's whole value.
The loader turns it into a `call` node for you:

```yaml check=fragment
brief:
  kind: code
  input:
    notes: call(enrich, topic=${input.topic})
  output: str
  code: return `Brief on ${inputs.notes}`
```

- Arguments are keywords only: `call(f, x=1, note=${a.output})`. Calls may nest:
  `call(outer, x=call(inner, y=1))`.
- The call runs only when the node it is written in can run: on that node's `case` branches,
  after its `runs_after:` nodes, and once its other inputs are ready.
- A quoted value is text, never a call: `note: "call(f)"` is the text `call(f)`.
- A flow call inside `${...}` is a load error, and in the middle of other text `call(...)` is
  plain text. Give the call its own node and read that node's output.

## Repeat with `loop`

A `loop` runs its body again and again. `input:` is the seed of a carried record. Each run of the
body reads the record and returns the fields that change; a field the body leaves out keeps its
value.

```yaml title="countdown.yaml" check=runnable
id: countdown
name: countdown
input:
  start: !int
nodes:
  count:
    kind: loop
    input:
      n: ${input.start}
      seen: !list(int) []
    while: n > 0
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: return inputs.n - 1
    output:
      n: ${step.output}
      seen: ${input.seen + [input.n]}
output: ${count.output.seen}
```

```console
$ yian run countdown.yaml --input start=3
[3,2,1]
```

Exactly one driver decides how often the body runs:

| Driver | Runs | Checked | `max:` |
|---|---|---|---|
| `while: <cond>` | 0 or more times | before each run | optional |
| `until: <cond>` | 1 or more times | after each run; stops once it is true | optional |
| `times: N` | exactly N times (N ≥ 1) | none | a load error |

- A condition reads carried fields by bare name (`while: not done`, `until: tries >= 3`). A name
  or field the record doesn't have is a load error.
- `max:` is a runaway guard. A loop that would run more than `max` times fails with
  `loop.max_exceeded`. Without `max:`, a loop runs until its condition ends it.
- A condition that reads only fields the body never returns can't change, so it is a load error
  unless you give `max:`. So is one that reads no carried field and doesn't end the loop at once
  (`until: 1 > 2`).
- The body returns at least one carried field, and only carried fields. Each must fit the field's
  carried type, which its seed gives. Over a tagged seed, a returned field whose type differs
  only by allowing `null` still loads, as in the example below. Any other mismatch is a load error,
  tagged or not.
- Returning `null` sets the field to `null`; leaving it out keeps its value.
- A `while:` that may be false at the start runs the body zero times, and the loop then outputs
  its seeds. So each literal seed of a returned field must fit the body's type for that field,
  unless `while:` is true on the literal seeds (`cursor: null` under `while: cursor == null`).
- `input:` lists each carried field. Each seed is checked against the body's input like a `call`
  argument. An empty list seed needs a tag, as `seen: !list(int) []` shows.
- `${count.output}` is the final record, and `${count.output.n}` one field of it.

A tagged seed lets the body return its type plus `null`. Here the seed `last: !int 0` is tagged,
so the body's `Optional(int)` loads, inline or as a `call:` body. A `null` it returns fails the
next run of the body with `contract.input`. After the last run it stays in the loop's output,
whose field reads `Optional(int)`:

```yaml title="poll.yaml" check=runnable
id: poll
name: poll
input:
  tries: !int
nodes:
  poll:
    kind: loop
    input:
      n: 0
      tries: ${input.tries}
      last: !int 0
    while: n < tries
    nodes:
      read:
        kind: code
        input:
          n: ${input.n}
        output: Optional(int)
        code: |
          return inputs.n < 2 ? inputs.n * 10 : null
    output:
      n: ${input.n + 1}
      last: ${read.output}
output: ${poll.output.last}
```

```console
$ yian run poll.yaml --input tries=3
null
```

With an untagged seed, `last: 0`, the same flow doesn't load:

```yaml title="poll_untagged.yaml" check=fails
id: poll_untagged
name: poll_untagged
input:
  tries: !int
nodes:
  poll:
    kind: loop
    input:
      n: 0
      tries: ${input.tries}
      last: 0
    while: n < tries
    nodes:
      read:
        kind: code
        input:
          n: ${input.n}
        output: Optional(int)
        code: |
          return inputs.n < 2 ? inputs.n * 10 : null
    output:
      n: ${input.n + 1}
      last: ${read.output}
output: ${poll.output.last}
```

```text
loop node 'poll': the body returns 'last' as Optional(int), but the carried field is typed int from its seed; declare the seed's type: `last: !Optional(int) 0`
```

A tag allows only that `null`. Any other wider type is a load error, tagged or not: under
`mode: !Mode a`, a body that returns a `str` doesn't load.

```yaml title="mode.yaml" check=fails
id: mode
name: mode
typedefs:
  Mode: Literal(a, b)
nodes:
  pick:
    kind: loop
    input:
      n: 0
      mode: !Mode a
    while: n < 2
    nodes:
      read:
        kind: code
        output: str
        code: return "b"
    output:
      n: ${input.n + 1}
      mode: ${read.output}
output: ${pick.output.mode}
```

```text
loop node 'pick': the body returns 'mode' as str, but the carried field is declared Literal(a, b); declare the seed's type: `mode: !str "a"`
```

A `while:` null check guards the body. Under `while: cursor != null`, a seed or returned field
typed `Optional(str)` fits a body input typed `str`, since the body never runs with a null
`cursor`. `until:` and `times:` guard nothing.

A loop may run any number of times. The run log keeps every run (see
[How yian works](/how-it-works/)).

A body may wait for a message, a time or a person; see [Inboxes and waiting](/inboxes/).

## Write a body inline

A `call`, `map` or `loop` used once can hold its body on the node: `nodes:` and `output:` instead
of `call:`.

```yaml check=fragment
loud:
  kind: map
  over: ${names.output}
  input:
    word: ${item}               # the body reads it as ${input.word}
  nodes:
    shout:
      kind: code
      input:
        word: ${input.word}
      output: str
      code: return inputs.word.toUpperCase()
  output: ${shout.output}
```

- Write exactly one of `call:` and `nodes:`. An inline body needs `output:`; `output:` beside a
  named `call:` is a load error.
- The node's `input:` becomes the body's inputs, read as `${input.<name>}`.
- The body can't read its parent's inputs, nodes or `${item}`. Pass what it needs through
  `input:`. `${system.today}`, `${system.now}` and `${system.run_id}` work everywhere.
- Each input's type is inferred from its binding. When it can't be proven, such as for an empty
  list, the load fails; add a tag (`seen: !list(int) []`) or use a `defs:` entry.
- `env:` and the loop driver stay on the outer node. An inline body has no `llm_config:` or flow
  `asserts:`; use a `defs:` entry when it needs them.

## Nesting limit

Each `call`, `map` or `loop` body is one level, however it is written. Bodies nest at most 50
levels; `YIAN_MAX_CALL_NESTING` can lower the limit (see [Reference](/reference/)). A deeper
flow fails to load.
