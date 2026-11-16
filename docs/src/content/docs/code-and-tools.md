---
title: "Code and tools"
description: "CODE and TOOL nodes: inline TypeScript bodies, ./file.ts:fn references, the worker, tool calls, and tool modules."
---

This page explains the two nodes that run without a model: a `code` node runs your own
TypeScript or JavaScript, and a `tool` node calls one tool your host or a
[tool module](#tool-modules) supplies. Both are deterministic steps: same inputs, same work, no
model in the loop.

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
    output: !int
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
path resolves against the flow file's real folder, symlinks followed.

```yaml check=fragment
count:
  kind: code
  input:
    text: ${input.text}
  output: !int
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
from), so the flow names a tool and the run looks it up. A `call` node does the same with `call:`
in place of `tool_id:`, and so does `call(get_quote, ...)` in a binding or a `case` branch (see
[Call a tool](/control-flow/#call-a-tool)). A `map` calls a tool once per item, and a `loop` once
per turn (see [Map over a tool](/control-flow/#map-over-a-tool) and
[Loop over a tool](/control-flow/#loop-over-a-tool)).

```yaml check=fragment
quote:
  kind: tool
  tool_id: get_quote
  retries: 3
  input:
    symbol: ${input.symbol}
  output:
    symbol: !str
    price: !float
```

| Field | Required | What it holds |
|---|---|---|
| `tool_id` | yes | A tool name, `{group: tool}` (a group's, a [memory](/memory/)'s or a [tool module](#tool-modules)'s tool), or a built-in: `take`, `sleep` or `ask` |
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

- the run's tools have no tool with that id (the error suggests the closest name the file
  defines and a close tool, the closer first), or two groups hold a tool of that bare name;
- the node passes an argument the tool doesn't take, or leaves out a required one;
- an argument, literal or `${...}`, can never fit the tool's parameter type, such as a `str` for
  an `int`;
- the declared `output:` can never match what the tool says it returns. As in JSON Schema, an
  object in the tool's output schema allows other keys unless it says
  `additionalProperties: false`.

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

## Tool modules

A tool module is a TypeScript or JavaScript file whose exports are tools. A flow imports it with
`uses:`, and its tools work as [host tools](/agents/#host-tools) do: a `tool` or `call` node calls
one, and an agent grants them. A module needs no runtime import of yian, so it runs wherever the
flow runs, under a compiled `yian` too.

```ts title="crm.ts"
import type { ToolOptions } from "@ngocbh/yian/tools";

const TOOL_ERROR = Symbol.for("yian.tool_error");
const customers: Record<string, string> = { "7": "Ada" };

export const lookup = {
  name: "lookup",
  description: "Find a customer by id.",
  parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
  execute: async (args) => {
    const name = customers[String(args.id)];
    if (name === undefined) throw Object.assign(new Error("no such customer"), { [TOOL_ERROR]: ["reported"] });
    return { id: args.id, name };
  },
} satisfies ToolOptions;

export const LIMIT = 10; // not a tool
```

```yaml title="lookup.yaml" check=runnable
id: lookup
name: lookup
input:
  id: !str
uses:
  crm: ./crm.ts
nodes:
  found:
    kind: tool
    tool_id: {crm: lookup}
    input:
      id: ${input.id}
    output: {id: !str, name: !str}
output: ${found.output.name}
```

```console
$ yian run lookup.yaml --input id=7
Ada
```

### Exports and aliases

A named export is a tool when it is a `tool()` value, or a plain object with an `execute`
function: `tool()`'s options (`name`, `description`, `parameters`, `execute`, and optionally
`output`, `isConcurrencySafe` and `resumable`; see [Host tools](/agents/#host-tools)). Every
other export, and `default`, is ignored.

- A tool's `name` must equal its export's name.
- A plain object missing a field, holding a mistyped one or a key `tool()` doesn't take (such as
  `resumeable`), and an export that can't be read, are refused.
- A plain object's `execute` and `isConcurrencySafe` run with the object as `this`.
- For the types, write `import type { ToolOptions }` as the whole statement. Node keeps
  `import { type ToolOptions }` as an import, which fails where yian isn't installed.
- Write an ES module: Node and Bun read a CommonJS file's exports differently.

| `uses:` entry | Imports | A `tool` node calls | An agent's model sees |
|---|---|---|---|
| `crm: ./crm.ts` | Every tool the file exports, as a group | `tool_id: {crm: lookup}` | `crm_lookup`, `crm_create`, ... |
| `quote: ./tools.ts:quote` | One export, as one tool | `tool_id: quote` | `quote` |

- An agent grants a group whole (`tools: [crm]`) or in part (`{crm: [lookup]}`), and a
  one-export alias by its name. A tool that two aliases of one file reach is offered once.
- A ref is a relative path, `./` or `../`, to a `.ts`, `.js` or `.mjs` file, resolved from the
  importing file's real folder, never from `system.paths`. It takes no `@version`, and under a
  `codeRoot` or `yian check --root` it must stay inside, as a CODE file must.
- A `call`, `map` or `loop` node names a module's tool as `tool_id:` does: `call: {crm: lookup}`,
  or `call: quote`; in a binding, `call(crm.lookup, ...)` or `call(quote, ...)`. Each calls the
  module's tool even when the run has a host tool or group of the alias's name.

### When a module runs

Loading a flow and `yian check` never import a module; they check that its file exists. Each run
start and each resume (each `advance`, for a host) imports the modules the flow's nodes reach,
each once, on a thread of its own, and lists their tools. The threads end when the run ends,
stops or waits.

- A module's top-level code runs at every start and resume, so keep it cheap and free of side
  effects.
- One thread serves every call to its module, interleaved; a call that blocks the thread stalls
  only that module's calls.
- The thread is **not a sandbox**, as CODE's worker isn't.
- The run refuses to start, naming the file (from the root flow's folder) and the export, when
  an import throws or outlasts `YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS` (30), a one-export alias
  names an export that is missing or isn't a tool, or a group's file has no tools, a refused
  export or a tool named like a built-in:

```text
crm.ts export 'find' holds a tool named 'lookup'; rename one
```

A run records the tools of the modules its agents grant, and a resume refuses a module whose tools
changed. A changed `execute` runs as written.

### Calls

A module's tool is checked, called and retried as a host tool is, and its validators,
`isConcurrencySafe` and `execute` run on the module's thread.

- `ctx` holds `callId`, `attempt`, `runId`, `node`, `toolName` (the export's name) and `signal`.
  `ctx.context` is `undefined`: the host's app object can't reach the thread.
- What `console` prints during a call goes where the calling node's console goes: printed, or to
  a host's [`onConsole`](/embedding/#runflow-options). Lines printed at import, or after the call
  ended, go to the process's `console`.
- An error marked `reported` is a result an agent's model reads; any other error fails the agent.
  A `tool` node retries as [Retries](#retries) says. Mark a kind without importing yian through
  the `Symbol.for("yian.tool_error")` key, which yian's own error classes carry too;
  `["reported", "permanent"]` is both.

```ts check=fragment
const TOOL_ERROR = Symbol.for("yian.tool_error");
throw Object.assign(new Error("no such customer"), { [TOOL_ERROR]: ["reported"] });            // the model reads it
throw Object.assign(new Error("quantity must be positive"), { [TOOL_ERROR]: ["permanent"] });  // TOOL retries stop
throw Object.assign(new Error("rate limited"), { [TOOL_ERROR]: ["transient"], retryAfter: 5 }); // retried after 5 s
```

- A result must be a yian value, as a CODE node's must. A `Date`, a class instance or a function
  in it fails the node with `node.execute`, never retried, even for a tool without `output` (a
  host tool's such result fails a TOOL node with `contract.output`).
- A tool may return `{ [Symbol.for("yian.pending")]: true }`, which is `pending()`, to answer the
  call later. That works only under a host that passes
  [`completesCalls: true`](/embedding/#pending-host-calls); elsewhere, `yian run` and `yian chat`
  included, the call fails: `tool 'crm_lookup' returned pending(), but this run's host doesn't
  complete calls`.

### Exits and restarts

When a module's thread exits (`process.exit`, or an error nothing catches, such as one a timer
throws), every call it was running fails with `tool.module_exited`:
`crm.ts's thread exited (code 3); the call's outcome is unknown`. An agent's model reads it, unless
the thread exited while checking the result; a `tool` node fails without a retry, since the call
may have run.

The next call starts a fresh thread, which imports the module again. When that import fails or
outlasts the timeout, or the module's tools differ from the ones the run started with, the call
and every later one fail their nodes with `tool.module_restart`, until the run is resumed.
