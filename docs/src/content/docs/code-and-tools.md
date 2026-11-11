---
title: "Code and tools"
description: "CODE and TOOL nodes: inline TypeScript bodies, ./file.ts:fn references, the worker, and tool calls."
---

This page explains the two nodes that run without a model: a `code` node runs your own
TypeScript or JavaScript, and a `tool` node calls one tool your host supplies. Both are
deterministic steps: same inputs, same work, no model in the loop.

## CODE nodes

A `code` node binds its `input:`, runs code over those values, and returns one value. Its
`output:` declares the type that value must have.

| Field | Required | What it holds |
|---|---|---|
| `code` | yes | An inline body, or a `./file.ts:fn` reference |
| `input` | no | The values the code reads, as bindings |
| `output` | no | The type of the returned value; yian checks it |
| `asserts` | no | Checks on the result (see [Expressions](/expressions/)) |

### Inline bodies

Any `code:` text that isn't a file reference is an inline body. It is the body of an async
function: it reads its inputs from `inputs` and ends with `return`. TypeScript type annotations
are allowed and are stripped before the body runs.

```yaml title="word_count.yaml" check=runnable
id: word_count
name: word_count
input:
  text: !str
nodes:
  count:
    kind: code
    input:
      text: ${input.text}
    output: int
    code: |
      const words: string[] = inputs.text.split(/\s+/).filter(Boolean);
      return words.length;
output: ${count.output}
```

```console
$ yian run word_count.yaml --input text="the quick brown fox"
4
```

- `inputs` holds only the node's declared inputs. Reading any other name throws, and `inputs`
  can't be changed.
- The body may use `await`, including `await import("node:path")` for a module.
- Returning nothing (`undefined`) fails the node; return `null` for no value.
- A body that doesn't parse fails at load, at the line of the error. The body isn't run at load.

### File references

Move a body into a file once it grows, or when several flows share it. A `code:` value of the
form `./path.ts:name` (or `../`, and `.js` or `.mjs`) calls the named export of that file. The
path resolves against the flow file's directory.

```yaml check=fragment
count:
  kind: code
  input:
    text: ${input.text}
  output: int
  code: ./text.ts:countWords
```

```ts
// text.ts, next to the flow file
export function countWords(inputs: { text: string }): number {
  return inputs.text.split(/\s+/).filter(Boolean).length;
}
```

The export gets the same `inputs` record an inline body gets, so a body moves into a file
unchanged. At load, yian checks that the file exists; it doesn't import it until the node runs.
A value that breaks this form fails at load: `text:countWords` or `text.ts:countWords` (no `./`),
an absolute path, another extension, or a path with a space.

### Values in and out

| yian type | Your code sees | What you return |
|---|---|---|
| `int` | `number`, or `bigint` past 2^53 | a safe-integer `number`, or a `bigint` |
| `float` | `number`, or `bigint` for an int past 2^53 | a `number` |
| `str`, `bool` | `string`, `boolean` | the same |
| `date`, `datetime` | an ISO `string` | an ISO `string` |
| `None` | `null` | `null` |
| `list(T)` | an array | an array |
| record, `object`, `dict(...)` | a plain object | a plain object |
| `Any` | any of these | any of these |
| `file` | a `FileRef` | a `FileRef` |

The return value is checked against `output:`. A `Date`, a class instance, a getter or a function
in the result fails the node with `contract.output`. Anything the code throws fails the node with
`node.execute`; for an inline body, the error points at the body line that threw.

### The worker

Inline bodies and file references run on a worker thread, one call at a time per thread, so a
slow CODE node doesn't block the rest of the run.

- Idle workers are reused. A module is imported once per thread and cached there, so an edit to
  a referenced file takes effect in a new process.
- Globals and module state may survive between calls on the same thread. Don't rely on it either
  way.
- `console.log` and the other `console` calls print as usual, or go to a host's
  [`onConsole`](/embedding/#runflow-options).
- Don't leave timers or other work running after you return. A late error from it fails
  the call its thread is running then, or prints an error line when the thread is idle; either
  way the thread is replaced.
- A worker is **not a sandbox**. Code shares the process's privileges, files and environment
  variables. Only run flows whose code you trust.

### Bun and Node

yian strips the types from inline bodies itself, so inline bodies behave the same on both
runtimes. Write types that can be erased: annotations, `as`, interfaces and type aliases. An
`enum`, a `namespace` or a constructor parameter property fails at load.

A referenced `.ts` file is imported by the runtime. Bun compiles it fully. Node strips its types
the same way yian does, so the same erasable-only rule applies there. Use a `.js` or `.mjs` file
for plain JavaScript on either runtime.

## TOOL nodes

A `tool` node calls one tool by id, with its `input:` as the tool's arguments. There is no model
in the loop. Your host supplies the tools (see [Agents](/agents/#tools) for where tools come
from), so the flow names a tool and the run looks it up.

```yaml check=fragment
quote:
  kind: tool
  tool_id: get_quote
  retries: 3
  input:
    symbol: ${input.symbol}
  output:
    symbol: str
    price: float
```

| Field | Required | What it holds |
|---|---|---|
| `tool_id` | yes | A tool name, `{group: tool}` (a group's or a [memory](/memory/)'s tool), or a built-in: `take`, `sleep` or `ask` |
| `input` | no | The tool's arguments, by the tool's own parameter names |
| `output` | no | The type the result must have; without it any JSON value or file passes |
| `retries` | no | Extra attempts after a failed call (default 0); not for `take`, `sleep`, `ask` |
| `asserts` | no | Checks on the result |
| `adaptive_questions` | no | `ask` only (see [Inboxes and waiting](/inboxes/)) |

The built-ins `take`, `sleep` and `ask` read an inbox, wait until a time and ask a person; they
are covered on [Inboxes and waiting](/inboxes/).

### Checks before the run starts

When a run starts, yian checks every TOOL node against the tool it names, including nodes in
branches and child flows that may never run. The run doesn't start when:

- the run's tools have no tool with that id (the error suggests a close match), or two groups hold
  a tool of that bare name;
- the node passes an argument the tool doesn't take, or leaves out a required one;
- an argument, literal or `${...}`, can never fit the tool's parameter type, such as a `str` for
  an `int`;
- the declared `output:` can never match what the tool says it returns.

Nothing is converted for you: pass `10`, not `"10"`, to an `int` parameter.

### Retries

`retries: N` calls the tool up to N more times after a failure. Before each retry the node waits
between half and all of 1, 2, 4 seconds, and so on, up to 30 seconds. Each attempt is recorded
in the run log, so a resumed run keeps counting from where it stopped.

| What the tool throws | Retried |
|---|---|
| any error | yes |
| `PermanentToolError`, or `ReportedToolError` with `permanent: true` | no |
| `TransientToolError` with `retryAfter` | yes, after that many seconds (still capped) |
| a failed argument or `output:` check | no |

The default is 0, so a tool runs once unless you opt in: a retried call repeats any side effect
the failed attempt already had. `YIAN_TOOL_RETRIES` changes the default;
`YIAN_TOOL_RETRY_DELAY_SECONDS` (1) and `YIAN_TOOL_RETRY_MAX_DELAY_SECONDS` (30) change the waits.
When every attempt fails, the error lists each attempt.
