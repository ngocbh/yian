---
title: "Embedding yian"
description: "Run flows from your own TypeScript program."
---

This page is for host developers: you load and run flows from your own TypeScript program, keep
runs on disk, move messages and answers into them, and watch what they do. Install the package as
[Get started](/get-started/) shows. It runs on Node 22.18 or later and Bun 1.4 or later.

## Entry points

| Import | What it's for |
|---|---|
| `@ngocbh/yian` | Load and run flows: `loadFlow`, `runFlow`, `advance`; the host calls `send`, `withdraw`, `close`, `complete`, `interrupt`, `cancel`, `pendingCalls` and their errors `DeliveryRefused` and `RunBusyError`; host tools (`tool`, `ToolContext`, `pending`, the tool errors); memory (`buildMemories`, `MemoryRegistry`, storage classes); `LoadError`, `Failure`, `Diagnostic` |
| `@ngocbh/yian/run` | Everything about runs: `runFlow`, `advance`, the host calls and their errors, plus `RunResult`, `runLogPath`, `inspectRun`, `loadRecordedFlow`, the types `RunFlowOptions`, `AdvanceOptions`, `EventHook` and `StreamHook`; for hosts that show runs, `LiveChannel`, `iterRunRecords`, `foldRunLog`, `agentCalls`, `agentCall`, `updateRunIndex` and `readRunIndex` |
| `@ngocbh/yian/tools` | Tools without the run API: `registerTool`, `registeredTools`, `invokeTool`, `mcpServer` and the `Mcp*` errors, `pending`, `isPending`, and the ask handler types `AskHandler` and `AskRequest` |
| `@ngocbh/yian/skills` | `loadSkills`, `loadSkillDir`, `defaultSkillRoots`, `Skill`, `SkillError` |
| `@ngocbh/yian/platform` | yian's file lock and SQLite wrapper, for hosts that lock files the same way: `lockFile`, `waitForLock`, `isLocked`, `FileLockError`, `openSqlite`, `sqliteErrorCode` |
| `@ngocbh/yian/llm_clients` | Provider setup: `LLMConfig`, `SUPPORTED_LLM_PROVIDERS`, `listModels`, `AuthStore`, `AuthError` |
| `@ngocbh/yian/code-worker` | Import it only for its side effect, as a bundle entry (see [Bundling](#bundling)) |

On Bun each import loads the TypeScript source; on Node it loads the built JavaScript. The event
classes aren't exported: narrow on `event.type` (see [Events](#events)).

## Run a flow

`loadFlow` reads YAML text into a flow, and `runFlow` runs it. This flow prices an order with a
host tool and formats the total in a CODE node:

```yaml title="total.yaml"
id: total
name: total
input:
  qty: !int
  price: !float
nodes:
  quote:
    kind: tool
    tool_id: quote
    input:
      qty: ${input.qty}
      price: ${input.price}
  label:
    kind: code
    input:
      total: ${quote.output}
    output: str
    code: |
      return `total: ${inputs.total}`;
output: ${label.output}
```

```ts check=fragment
import { readFile } from "node:fs/promises";

import { loadFlow, runFlow, tool } from "@ngocbh/yian";

const quote = tool({
  name: "quote",
  description: "Price an order.",
  parameters: {
    type: "object",
    properties: { qty: { type: "integer" }, price: { type: "number" } },
    required: ["qty", "price"],
  },
  execute: ({ qty, price }) => Number(qty) * Number(price),
});

const flow = await loadFlow(await readFile("total.yaml", "utf8"), {
  searchPaths: ["."],
  sourceLabel: "total.yaml",
});
const result = await runFlow(flow, { qty: 3, price: 2.5 }, { tools: { quote } });
console.log(result.status, result.output); // succeeded total: 7.5
```

`loadFlow` takes `searchPaths` (where `uses:` targets are found; put the flow's own folder first),
`sourceLabel` (the name errors show), `baseDir` (where `./file.ts:fn` references resolve),
`codeRoot` (a folder every CODE file must stay inside) and `env`. A broken flow rejects with
`LoadError`, whose `diagnostic` points into the source. A CODE reference outside `codeRoot` is
named as the flow wrote it (`'../x.ts:fn' is outside the code root`), with no absolute path, and
reads the same whether or not its file exists, so you can show it to the flow's author.

A flow that fails still resolves, with `status: "failed"`. `runFlow` rejects for problems in your
program or setup: a bad option, a tool, provider or model the flow needs but the run lacks, an MCP
server that doesn't start, a bad `YIAN_*` value, a callback that throws, a log it can't write.

### `RunResult`

| Field | Holds |
|---|---|
| `status` | `"succeeded"`, `"failed"`, `"cancelled"`, or `"waiting"` when every node left is waiting (see [Waiting runs](#waiting-runs)) |
| `output` | The flow's output on success, else `null` |
| `failure` | A `Failure` (`code`, `message`, `nodeId`, ...) on failure, else `null`; `flow.diagnose(failure)` adds source frames |
| `runId` | The run's id |
| `log` | The log path you gave, or `null` |
| `input` | The inputs the run recorded |
| `events` | The events of this call |
| `waits` | What a waiting run waits on: `node`, `callId`, `tool`, `inbox`, `deadline`, `settled` |
| `unread` | At the end of a run, the messages no take read, as send seqs by inbox |
| `crashedAttempts` | Node attempts a resume ran again after a crash, by node id |

`result.backward(feedback)` runs the backward pass on a run started with `learnable: true`; see
[Learning](/learning/).

### `runFlow` options

| Option | Default | Meaning |
|---|---|---|
| `log` | `null` | A new log file. `null` keeps the run in memory; an existing file is never overwritten |
| `runId` | a fresh id | The run's id, `system.run_id` in the flow |
| `numWorkers` | `8` | How many nodes run at once; `0` runs them one at a time |
| `tools` | the registered tools | Tools by id. Passing it replaces the registered set, so list every tool the flow uses |
| `skills` | none | A `Map` of skills by name, from `loadSkills` |
| `memory` | none | One backend per declared memory key, usually from `buildMemories` |
| `llmConfig` | none | The outermost `llm_config` layer (see [Models and providers](/models/)). Its `api_key` is never recorded |
| `context` | `undefined` | Your app object; tools read it as `ctx.context`. Never recorded |
| `onEvent`, `onStream` | none | Event callbacks (see [Events](#events)) |
| `live` | none | A `LiveChannel` a viewer follows |
| `signal` | none | An `AbortSignal` that stops the run |
| `env` | `process.env` | A function `(name) => string \| undefined` that yian reads instead of `process.env` |
| `learnable` | `false` | Keep the run for a later `result.backward` |
| `parent` | none | The run that started this one, recorded in the log |

`scheduler`, `random` and `models` exist for yian's own tests; leave them unset.

## Keep runs on disk

A run with a `log` writes every step to that file, so it can survive a crash, wait for days, and
take messages from other processes. Give each run a folder of its own: `runLogPath(runsDir, runId)`
makes `<runsDir>/<runId>/` and returns the log path inside it. The run keeps its lock file and
media files beside the log.

`advance({ log })` moves a run on from its log:

| The run | What `advance` does |
|---|---|
| Finished | Returns the recorded result and writes nothing |
| Crashed or stopped | Replays the log, then runs the unfinished nodes again |
| Waiting, with an answer or deadline in | Wakes those nodes and runs on |
| Waiting, with nothing new | Writes nothing and returns `waiting` again |

`advance` runs the flow the log recorded, not the file on disk, with the log's inputs, run id and
model settings. Pass again what the log doesn't hold: `tools`, `skills`, `memory`, `context`, and
the API key (`apiKey`, `headers`) of a run started with `llmConfig.api_key`.

The log doesn't hold `codeRoot` either. Pass `codeRoot` again on every `advance` and every host
call below; a call without it doesn't confine the recorded flow's CODE files at all.
[How yian works](/how-it-works/#crash-recovery) explains replay and what can run twice.

Only one writer holds a log at a time. A call on a log another process holds, or an `advance` of a
run your process is already moving, rejects with `RunBusyError`. While a run waits, its process
keeps the run's state in memory, so the next call on the same log skips the replay.

## Events

`onEvent` sees each event after its step is safely in the log, plus the model and tool calls of
agents as they happen. `onStream` sees each `StreamChunk` of streamed text (`nodeId`, `chunk`,
`final`). Both are called in order; a throw, or a rejected promise, stops the run with that error.

```ts check=fragment
import type { EventHook } from "@ngocbh/yian/run";

const onEvent: EventHook = (event) => {
  if (event.type === "NodeSucceeded") console.log("done:", event.nodeId);
  else if (event.type === "RunFailed") console.log("failed:", event.failure.code);
};
await runFlow(flow, { qty: 3, price: 2.5 }, { tools: { quote }, onEvent });
```

| Events | When |
|---|---|
| `RunStarted`, `RunResumed`, `RunSucceeded`, `RunFailed`, `RunWaiting`, `RunCancelled` | The run starts, resumes, ends or starts waiting |
| `NodeStarted`, `NodeSucceeded`, `NodeFailed`, `NodeRouted`, `NodeSkipped`, `NodeExpanded` | A node runs, a `case` picks a branch, a node is skipped, a `map` or `loop` grows its nodes |
| `NodeWaiting`, `NodeWoken`, `CallClosed` | A node waits on a call, wakes, or a call closes without an answer |
| `NodeAttemptResumed` | A resume runs a crashed node again |
| `ModelCallStarted`, `ModelCallFinished`, `ToolCallStarted`, `ToolCallFinished`, `ReplayDegraded` | Inside an agent; not in the log |

Event fields are camelCase. `event.toJSON()` gives the snake_case form; it keeps flow values as
they are (ints as `bigint`), so write it with `logJson` from `@ngocbh/yian/run`, not
`JSON.stringify`.

To show a run live to a viewer, pass a `LiveChannel` as `live`. `live.subscribe(fn)` gets each
log line as it is written and each model call's text chunks, as plain JSON frames you can send
anywhere; `live.follow(fn)` starts with a snapshot, for a viewer that joins late.

## Waiting runs

A run stops with `status: "waiting"` when every node that can still run waits on something: a
`take` of an inbox, a `sleep`, an `ask`, or a host tool that answers later. The log holds
everything, and the lock is released. [Inboxes and waiting](/inboxes/) covers these nodes from
the flow's side.

### Inbox calls

Your program reaches a waiting run by its log path:

| Call | Returns | What it does |
|---|---|---|
| `send(log, inbox, value)` | the send's seq | Adds a message to an `Inbox(T)` input; `value` must fit `T` |
| `withdraw(log, inbox)` | `[{seq, value}]` | Takes back every message nothing has read yet |
| `close(log, inbox)` | the seq, or `null` if already closed | Refuses later sends; takes still read what is queued |
| `complete(log, callId, result)` | the seq, or `null` for the same answer twice | Answers a pending host call (see [Pending host calls](#pending-host-calls)) |
| `interrupt(log, inbox)` | `{seq, nodes, calls}` | Stops the agents steered by that inbox |
| `cancel(log)` | the open host calls it closed | Ends the run as `cancelled` |
| `pendingCalls(log)` | `PendingCall[]` | Lists open host calls; reads without the lock |

Apart from `cancel`, which ends the run, none of them moves the run on: call `advance` next. If
your process is running the run right now, the call goes straight to it instead. This flow waits for notes:

```yaml title="notes.yaml"
id: notes
name: notes
input:
  notes: !Inbox(str)
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: ${input.notes}
      take: all
output: ${heard.output.events}
```

```ts check=fragment
import { randomUUID } from "node:crypto";

import { advance, close, loadFlow, runFlow, send } from "@ngocbh/yian";
import { runLogPath } from "@ngocbh/yian/run";

const runId = randomUUID();
const log = runLogPath("./runs", runId);
const first = await runFlow(flow, {}, { log, runId });
console.log(first.status, first.waits.map((wait) => `${wait.node} (${wait.tool})`));
// waiting [ "heard (take)" ]

await send(log, "notes", "buy milk");
await send(log, "notes", "call Ada");
await close(log, "notes");
const done = await advance({ log });
console.log(done.status, done.output); // succeeded [ "buy milk", "call Ada" ]
```

A refused call appends nothing and throws `DeliveryRefused`, whose `reason` says why:

| `reason` | Meaning |
|---|---|
| `ended` | The run has finished |
| `not_started` | The run hasn't started yet |
| `unknown_inbox` | The flow has no such inbox input |
| `inbox_closed` | The inbox is closed |
| `invalid_value` | The value doesn't fit the inbox's type or the call's output type |
| `too_large` | The message is over `YIAN_MAX_EVENT_BYTES` (1 MiB by default) |
| `unknown_call` | No open call has that id |
| `not_host_call` | The id names a take or a sleep |
| `call_settled` | The call already has an answer, timed out or was closed |

### Pending host calls

A host tool can answer later: its handler returns `pending()`, yian records the call, and the
node waits. The run needs a `log`; without one the node fails, since nothing could answer it.
An `ask` works the same way: pass a handler under the `ask` key of `tools`. A flow with an `ask`
and no handler is refused before it starts (`no ask handler for this run`).

```yaml title="approve.yaml"
id: approve
name: approve
input:
  item: !str
nodes:
  answer:
    kind: tool
    tool_id: ask
    input:
      prompt: Approve ${input.item}?
      timeout: 2d
    output: Optional(str)
output: ${answer.output :- "no answer"}
```

```ts check=fragment
import { advance, complete, pending, pendingCalls, runFlow } from "@ngocbh/yian";
import type { AskRequest } from "@ngocbh/yian/tools";

const ask = (request: AskRequest) => {
  queueForReview(request.callId, request.question); // your code
  return pending();
};
const first = await runFlow(flow, { item: "the refund" }, { log, tools: { ask } });
// first.status is "waiting"

// Later, in this process or another one:
const [call] = pendingCalls(log); // { callId, node, tool: "ask", arguments, deadline, ... }
await complete(log, call!.callId, { output: "yes" });
const done = await advance({ log, tools: { ask } });
console.log(done.status, done.output); // succeeded yes
```

- `complete` takes `{ output }`, checked against the node's output type, or `{ error: "..." }`,
  which fails the call.
- `request.callId` stays the same when a resumed run asks again, so use it to drop duplicates.
  An `ask` also gets `deadline` and the `extra` inputs your handler declares (pass
  `{ handler, inputs }` instead of a function to declare them).
- When a deadline passes, `advance` times the call out: `null` for an `Optional(...)` output,
  else the node fails with `tool.timeout`. yian doesn't wake the run itself; call `advance` after
  the earliest `deadline`.
- An agent's `ask_user` tool goes to the same `ask` handler.

### Interrupt and cancel

```ts check=fragment
const stopped = await interrupt(log, "notes"); // { seq, nodes, calls }
const closed = await cancel(log);              // the open host calls it closed
```

`interrupt` stops the agents whose `steer_inbox:` is that inbox (see
[Agents](/agents/#steering-and-interrupts)). Each one's reply is cut: an `Optional(...)` output
succeeds with `null`, any other fails with `agent.interrupted`. A later `complete` of the open
calls it closes is refused. An agent that starts reading the inbox
soon after is stopped too, before its first model call. `interrupt` appends nothing
(`seq: null`) when the run is waiting for the next message on the inbox, or is already stopping.
Queued steers stay; `withdraw` takes them back.

`cancel` ends the run with no output and no learning, and closes its open calls. If your process
is running the run, its running nodes are stopped at once. To stop a run you may want to resume,
abort its `signal` instead: the run records a stop, rejects with the signal's reason, and
`advance` runs the stopped nodes again.

## Tools, skills and memory

- **Tools.** `tool()` builds a tool from a JSON Schema or a Standard Schema validator (zod,
  valibot, ArkType); see [Agents](/agents/#host-tools). A handler gets `(args, ctx)`. `ctx` holds
  `callId` (the same on every retry and resume, so use it as an idempotency key), `attempt`,
  `runId`, `node`, `context` (your app object) and `signal` (aborted when the call must end).
- **MCP servers.** `mcpServer({ command, args })` or `mcpServer({ url, headers })` describes a
  server; pass it as a `tools` entry and its tools become one group. yian opens it for the run
  and closes it after, unless you opened it yourself with `await server.open()`. It needs the
  optional peer package `@modelcontextprotocol/client`.
- **Skills.** `loadSkills(defaultSkillRoots())` reads the Claude Code and Codex skill folders
  into the `Map` that `skills` takes.
- **Memory.** `buildMemories(flow, { root, storage })` builds a backend for each memory the flow
  declares, as `yian run`'s memory flags do ([Learning](/learning/#storage)). Pass the registry as
  `memory`, and `close()` it when you are done. `supplied` passes ready backends by key, and
  `factories` builds custom memory types by type name.

## Inspect runs

| Call | Gives |
|---|---|
| `inspectRun(log)` | Where a run stands, without the lock: `status` (`running`, `succeeded`, `failed`, `cancelled`, `waiting`, `stopped`, `crashed`), `inFlight`, `waits`, `ready`, `wakeAt`. Pass `{ flow }` (from `loadRecordedFlow`) for an exact `status` |
| `loadRecordedFlow(log)` | The flow the log recorded |
| `iterRunRecords(log)` | The log's records, one by one, while another process may still be writing |
| `foldRunLog(log)` | Each record with the value it wrote |
| `agentCalls(log)`, `agentCall(log, { nodeId, n })` | The agents' model calls, and one call's full request and reply |
| `updateRunIndex(runsDir)`, `readRunIndex(runsDir)` | An `index.sqlite` with one row per run folder, which you can delete and rebuild |

`inspectRun(log).wakeAt` is the earliest deadline of a waiting run: a host that keeps runs
waiting calls `advance` then.

## Environment and stopping

- `env` is a function `(name) => string | undefined`. `loadFlow`, `runFlow`, `advance`,
  `inspectRun`, `buildMemories` and the host calls read through it instead of `process.env`:
  provider keys, `HOME`, and every `YIAN_*` setting. Runs with different `env`s don't share
  settings. A resume refuses `YIAN_*` values that differ from the recorded ones.
- `signal` stops `runFlow` or `advance`. An already-aborted signal rejects before anything is
  created.

## Bundling

A CODE node runs on a worker thread loaded from yian's worker module. If you bundle yian (with
`bun build`, or esbuild with ESM output), ship the worker as a second entry, a one-line file at
`nodes/code/worker.ts` below the folder that holds your entry:

```ts check=fragment
// src/nodes/code/worker.ts, next to src/main.ts
import "@ngocbh/yian/code-worker";
```

Otherwise every CODE call fails with `CODE worker failed: Cannot find module ...`. With
`bun build --compile`, pass both entries:

```console
$ bun build --compile --no-compile-autoload-dotenv --no-compile-autoload-bunfig src/main.ts src/nodes/code/worker.ts --outfile out/host
```

A `./file.ts:fn` CODE file is read from disk when the run needs it, so it can't see packages
bundled into the binary.

## Testing without a model

yian doesn't export a testing entry point or a scripted model yet. You can still test a host
offline with what the package exports:

- Flows without agents (CODE, TOOL, `case`, `loop`, inboxes) need no model. Pass fake tools in
  `tools`, and drive waits with `send`, `complete` and `advance` as above.
- For agents, point a provider at a small local server. Ollama needs no key and reads its address
  from `OLLAMA_BASE_URL` in `env`, so a test can answer its chat calls with canned replies.

```yaml title="greet.yaml"
id: greet
name: greet
input:
  name: !str
nodes:
  hello:
    kind: agent
    input:
      name: ${input.name}
    prompt: Greet ${name} in one short sentence.
    output: str
output: ${hello.output}
```

```ts check=fragment
import { afterAll, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

import { loadFlow, runFlow } from "@ngocbh/yian";

// Answers each call to Ollama's /api/chat with the next reply, as one NDJSON line.
const replies = ["Hello, Ada!"];
const server = createServer((request, response) => {
  request.resume();
  const content = replies.shift() ?? "";
  response.writeHead(200, { "content-type": "application/x-ndjson" });
  response.end(JSON.stringify({ message: { role: "assistant", content }, done: true, done_reason: "stop" }) + "\n");
});
await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
afterAll(() => server.close());

test("greet answers with the canned reply", async () => {
  const { port } = server.address() as AddressInfo;
  // Only these variables exist for the run: no saved logins, no YIAN_* settings.
  const vars: Record<string, string> = { OLLAMA_BASE_URL: `http://127.0.0.1:${port}`, HOME: "/nonexistent" };
  const env = (name: string) => vars[name];
  const flow = await loadFlow(await readFile("greet.yaml", "utf8"), { env });
  const result = await runFlow(flow, { name: "Ada" }, { env, llmConfig: { provider: "ollama", model: "stub" } });
  expect(result.output).toBe("Hello, Ada!");
});
```

Each reply line is one JSON object, and the line with `done: true` ends the reply. To have the
agent call a tool, put Ollama's `tool_calls` list in the line's `message`.
