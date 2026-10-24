# yian (TypeScript)

The TypeScript implementation of yian ([ngocbh/yian#736][issue-736]), the npm package `@ngocbh/yian`
(`yian` is taken on npm). The Python package in `src/yian/` stays until the cut-over. Until then the
guides in `docs/` describe Python yian, with a "Changed (TS yian):" note wherever TS yian differs.
This README gives the TypeScript shape of the API: names, option objects and types. For what each
option does, follow its link to [`docs/api.md`][docs/api.md]; the flow YAML is in
[`docs/syntax.md`][docs/syntax.md].

## Install

TS yian runs on Node >= 22.18 or Bun >= 1.4.

```sh
npm install @ngocbh/yian                          # or: bun add @ngocbh/yian
npx yian --help                                   # the CLI of that install
npx @ngocbh/yian --help                           # the CLI, without an install
npm install @modelcontextprotocol/client@2.3.1    # only to open MCP servers
```

The first release isn't published yet; until then, `npm pack` in `ts/` makes the package to install.

- Under Bun, the entry points resolve to the TypeScript sources in `src/` (the `bun` export
  condition); elsewhere to the built JavaScript in `dist/src/`, with its `.d.ts` types.
- The MCP SDK is an optional peer dependency, loaded only when a server opens.
- Single-file binaries need neither Node nor Bun: `yian-darwin-arm64`, `yian-darwin-x64`,
  `yian-linux-x64` and `yian-linux-arm64`. None is published yet; build them from a clone with
  `bun run build:bin` in `ts/` (into `ts/bin/`). They bundle the MCP SDK and read no `bunfig.toml`.
  Run from a clone with `bun src/cli/index.ts`, yian sees the `.env` Bun loads from the working
  directory; the binaries load none.
- `yian --licenses` prints yian's notice, Bun's, and the licenses of the packages the binaries
  bundle.
- `yian --version` prints the package version.

## Example

A flow with a TOOL node, served by a host tool, and a CODE node; neither calls a model.

```yaml
# total.yaml
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

```ts
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
  execute: ({ qty, price }) => (qty as number) * (price as number),
});

const flow = await loadFlow(await readFile("total.yaml", "utf8"), {
  searchPaths: ["."],
  sourceLabel: "total.yaml",
});
const result = await runFlow(flow, { qty: 3, price: 2.5 }, {
  tools: { quote },
  onEvent: (event) => console.log(event.type),
});
console.log(result.status, result.output); // succeeded total: 7.5
```

## Entry points

Only the names below are the supported surface. The other exports on `/run` (`hostInputs`,
`runStartInputs`, `runStartLlm`, `LlmConfigsCheck`, `checkRecordedLlmFields`, `skillGrants`,
`replayRunLog`, `replayCompiledRunLog`), on `/llm_clients` (`chatModelFromConfig`,
`defaultModelAccess`, `ModelAccess`, `callCredentials`, `scrubCallCredentials`) and on `/platform`
are internal: they serve the CLI and yian's own code, and may change without notice.

| Import | Names |
|---|---|
| `@ngocbh/yian` | `loadFlow`, `ComposedFlow`, `CompiledFlow`, `LoadError`, `FlowValidationError`; `runFlow`, `advance`; `Failure`, `Diagnostic`; `VariablePool`, `evaluateWhen`, `ExpressionError`, `registerTemplateFn`; `tool`, `FunctionTool`, `ToolContext`, `pending`, `PermanentToolError`, `TransientToolError`, `ReportedToolError`, `ToolAttemptsFailed`; `buildMemories`, `MemoryRegistry`, `SQLiteStorage`, `checkStoragePath`, `MarkdownStore` (type), `FileMarkdownStore`, `InMemoryMarkdownStore`, `MarkdownVersionMismatch`, `markdownVersion`, `ProfileMemory` |
| `@ngocbh/yian/run` | `runFlow`, `RunFlowOptions`, `advance`, `AdvanceOptions`, `loadRecordedFlow`, `RunResult`, `BackwardOutcome`, `Steering`, `EventHook`, `StreamHook`, `runLogPath`, `inspectRun`, `RunInspection`, `RunStatus`; for hosts that show runs, `updateRunIndex`, `readRunIndex`, `RunIndexRow`, `RunIndexStatus`, `RunIndexBusyError`, `iterRunRecords`, `IterRunRecordsOptions`, `RunLogEntry`, `runRecordOffsets`, `RunLogChangedError`, `RunLogSchemaError`, `staticNodes`, `staticPathOf`; for hosts that read run-log values, `TypedValue`, `FileRef`, `logJson`, `parseJson`, `JSONDecodeError`, `StorageKind`, `Value`, `ValueRecord`, `JsonValue`, `JsonObject`, `JsonMode` |
| `@ngocbh/yian/tools` | `registerTool`, `registeredTools`, `invokeTool`, `ToolEntry`, `AskHandler`, `AskRequest`, `AskTool`, `AskUserHandler`, `AskUserRequest`, `pending`, `isPending`, `PendingMarker`, `TAKE`, `SLEEP`, `ASK`; `mcpServer`, `McpServer`, `McpServerOptions`, `McpError`, `McpConnectError`, `McpToolError`, `McpRequestError`, `McpInvalidRequestError`, `McpTimeoutError`, `McpTransportError`, `McpServerClosed`, `McpCallCancelled`, `isMcpError`; `ToolSet`, `CheckedCall`, `InvokeOptions`, `invokeToolTrusted`, `resolveTools`, `TOOL_REGISTRY`; also `tool`, `FunctionTool`, `StandardSchemaV1`, `StandardIssue`, `StandardResult`, `ToolContext`, the tool errors, `ToolCheckError`, `TOOL_ERROR`, `toolErrorKinds`, `ToolErrorKind`, `isPermanentToolError`, `isTransientToolError`, `isReportedToolError`, `isToolCheckError` |
| `@ngocbh/yian/skills` | `loadSkills`, `loadSkillDir`, `defaultSkillRoots`, `Skill`, `SkillError` |
| `@ngocbh/yian/platform` | for hosts that lock files and open SQLite as yian does: `lockFile`, `waitForLock`, `isLocked`, `FileLock`, `FileLockError`, `FileLockReason`; `openSqlite`, `Sqlite`, `SqliteStatement`, `SqliteParam`, `SqliteValue`, `SqliteRow`, `sqliteErrorCode` |
| `@ngocbh/yian/llm_clients` | for hosts that set up providers: `LLMConfig`, `LLM_CONFIG_KEYS`, `SUPPORTED_LLM_PROVIDERS`, `LLMProvider`, `ValidationError`, `Issue`, `azureResourceEndpoint`; `listModels`, `ListModelsOptions`, `ModelListError`; `AuthStore`, `AuthStoreOptions`, `AuthState`, `ProviderSelection`, `Credentials`, `ApiKeyCredentials`, `OAuthCredentials`; `OpenAIAuth`, `AuthRequest`, `AuthRequestOptions`, `HttpResponse`, `AuthError` |

`Steering`, `EventHook`, `StreamHook`, `RunFlowOptions`, `AdvanceOptions`, `RunInspection`,
`RunStatus`, `IterRunRecordsOptions`, `RunLogEntry`, `StorageKind`, `Value`, `ValueRecord`,
`JsonValue`, `JsonObject`, `JsonMode`, `ToolEntry`, `AskHandler`, `AskRequest`, `AskTool`,
`AskUserHandler`, `AskUserRequest`, `PendingMarker`, `CheckedCall`, `InvokeOptions`, `McpServerOptions`, `StandardSchemaV1`, `StandardIssue`,
`StandardResult`, `ToolErrorKind`, `MarkdownStore`, `FileLock`, `FileLockReason`, `Sqlite`,
`SqliteStatement`, `SqliteParam`, `SqliteValue`, `SqliteRow`, `LLMProvider`, `Issue`,
`ListModelsOptions`, `AuthStoreOptions`, `AuthState`, `Credentials`, `AuthRequest` and
`AuthRequestOptions` are types only.

## Differences from Python yian

Each difference is a "Changed (TS yian):" note in `docs/`; these shape every call.

- **camelCase.** Functions, options and fields are camelCase (`load_flow` is `loadFlow`, `run_id` is
  `runId`, `backward_outcome` is `backwardOutcome`), and keyword arguments become one options
  object. The JSON forms stay snake_case: an event's `toJSON()`, run logs, CLI documents. Flow
  values keep their own keys.
- **Promises.** `loadFlow`, `runFlow`, `advance`, `result.backward`, `buildMemories`,
  `invokeTool` and `McpServer`'s `open` and `close` return Promises; their errors are rejections.
  `inspectRun`, `iterRunRecords`, `runRecordOffsets`, `updateRunIndex`, `readRunIndex`, `loadSkills`
  and `loadSkillDir` are synchronous.
- **Numbers.** Ints in yian values are `bigint` and floats are `number`: `result.output`,
  `result.input` and event payloads hold `3n`, not `3`. Values a host passes in (`inputs`,
  `llmConfig`, feedback, a tool's result) go through host intake: a whole `number` within ±(2^53−1)
  becomes an int (`{n: 3}` holds `3n`), any other number a float, and `Infinity` is a float; NaN, a
  `Date`, a `Map`, an accessor or a Proxy is refused with `TypeCheckError` (see [Types → host
  intake][docs/typing.md#two-constructors-infer-vs-check]). Tool arguments and CODE inputs arrive
  the other way: an int within ±(2^53−1) as a `number`, a larger one as a `bigint`.
- **`signal`.** `runFlow` and `advance` take an `AbortSignal`; aborting it stops the run, which
  rejects with the signal's reason. A tool's `ctx.signal` aborts when its call must end early.
- **Tools.** Every tool is a `FunctionTool`, often built with `tool()` from a raw JSON Schema or a
  Standard Schema v1 validator (zod 4, ArkType; valibot through `toStandardJsonSchema`). A plain
  function is not a tool. `execute(args, ctx)` takes the arguments first and may be async.
- **CODE nodes** are TypeScript or JavaScript, a reference `./file.ts:fn` or an inline body that
  reads `inputs`, and run on worker threads, so aborting the run (the `signal`, or one Ctrl-C of the
  CLI) ends even a body that never yields (see [`code` /
  `tool`][docs/syntax.md#code--tool--the-other-leaves] in the syntax guide). A host that bundles
  yian (esbuild, webpack, `bun build`) must ship the worker as its own entry at
  `nodes/code/worker.js` beside the bundle file, or every CODE call fails.
- **Errors.** Python's `ValueError` is a `RangeError` or `TypeError`, and every yian error sets
  `error.name` to its class. Some errors a call can throw aren't exported (`TypeCheckError`,
  `MemorySetupError`, `DefaultsError`, `DefaultsMismatchError`): check `error.name`.
  Tool errors, `FunctionTool` and `McpServer` are recognized by brand, not `instanceof`, so ones
  from another copy of yian work.
- **Storage.** Run logs (schema 16) and memory stores (SQLite envelope version 3) are not compatible
  with Python yian's.

## Loading a flow

```ts
loadFlow(text: string, options?: {
  searchPaths?: readonly string[] | null;
  sourceLabel?: string;            // default "<flow>"
  baseDir?: string | null;
  codeRoot?: string | null;
  childResolver?: ((path: string, version: string | null) => ComposedFlow | Promise<ComposedFlow>) | null;
  env?: (name: string) => string | undefined;
}): Promise<ComposedFlow>
```

See [`load_flow`][docs/api.md#load_flow]. `baseDir` is the directory a CODE node's `./file.ts:fn`
reference resolves against (default: the first search path, else the current directory). With
`codeRoot`, a reference whose file's real path is outside the real path of `codeRoot` is a
`LoadError`; an inline `code:` body isn't confined. A malformed
flow rejects with `LoadError` (a `RangeError`), whose `diagnostic` is a `Diagnostic` located in the
source.

A `ComposedFlow` holds `compiled` (the `CompiledFlow` graph), `version`, `name`, `description`,
`originResolver` and `importDepth`; `flow.diagnose(failure)` returns a `Diagnostic` with source
frames for a run's `Failure` (see [Diagnosing a failed run][docs/api.md#diagnosing-a-failed-run]).
`FlowValidationError` (a `RangeError`, with `errors`, a `readonly string[]`) reports a failed graph
check.

## Running a flow

```ts
runFlow(flow: ComposedFlow, inputs: Record<string, unknown>, options?: RunFlowOptions): Promise<RunResult>
```

A failed flow resolves with `status: "failed"`; host, callback and log errors reject. See
[`run_flow`][docs/api.md#run_flow] for the options:

| Option | Type | Default |
|---|---|---|
| `log` | a new JSONL path, never overwritten | `null`: a process-local log |
| `runId` | `string` | minted |
| `numWorkers` | `number`; `0` runs serially | `8` |
| `onEvent` | `EventHook`, `(event) => unknown` | `null` |
| `onStream` | `StreamHook`, `(chunk) => unknown` | `null` |
| `live` | `LiveChannel` (see [Following a run live](#following-a-run-live)) | `null` |
| `llmConfig` | record, the outer `llm_config` layer | `null` |
| `memory` | `Map` of memory key to backend, usually a `MemoryRegistry` | `null` |
| `learnable` | `boolean`: keep the run for `result.backward` | `false` |
| `tools` | record of tool id to `ToolEntry`; replaces the registry | `null`: the registered tools |
| `skills` | `Map<string, Skill>` | `null` |
| `steering` | `Steering` | `null` |
| `context` | the host's app object, as `ctx.context` | `undefined` |
| `signal` | `AbortSignal` | `null` |
| `env` | `(name: string) => string \| undefined` | `null`: the process environment |
| `scheduler`, `random`, `models` | for tests (see [`env`, `signal` and `scheduler`](#env-signal-and-scheduler)) | real clock, `Math.random`, real models |

A callback that throws, or returns a promise that rejects, stops the run, which rejects with that
error.

Each run gets a folder of its own: a run's folder is the folder that holds its `log`, and the files
the run writes beside the log (the writer's lock, `<log>.lock`, and media files under
`media/<sha256>.<ext>`) go only there. Pass
`log: runLogPath(runsDir, runId)` with the same `runId` to get `<runsDir>/<runId>/log.jsonl`;
`runLogPath` makes the new run folder with mode 0700 (the runs folder too, when missing), creates
no file, throws `EEXIST` when anything already holds the run folder's path, and throws a
`RangeError` for a run id that isn't one folder name. A host that passes a `log` path of its own
must keep one run per folder; Yian doesn't check it. The CLI uses `~/.yian/runs`.

### `RunResult`

| Field | Type |
|---|---|
| `status` | `"succeeded"` or `"failed"` |
| `output` | the flow's value on success, else `null` |
| `failure` | `Failure \| null` |
| `input` | the inputs `RunStarted` kept |
| `runId` | `string` |
| `log` | the path as given, or `null` |
| `events` | this call's post-durable events |
| `crashedAttempts` | `ReadonlyMap<string, number>`: attempts a resume retried after a crash, by node id |
| `backwardOutcome`, `learning` | the last observed backward attempt, else `null` |

`await result.backward(feedback, { apiKey, headers, onEvent })` runs one backward attempt over a
`learnable: true` run and resolves to a `BackwardOutcome` (`status`, `feedback`, `failure`,
`learning`). See [`RunResult`][docs/api.md#runresult] and [Explicit backward
feedback][docs/api.md#explicit-backward-feedback].

### Events and stream chunks

`onEvent` receives each observer event after its record is durable, plus the transient in-loop AGENT
events; `onStream` receives `StreamChunk`s. Every event has a `type` naming its class and camelCase
fields. The classes aren't exported: narrow on `event.type` (the event type is
`Parameters<EventHook>[0]`).

- Run: `RunStarted`, `RunResumed`, `RunSucceeded` (`output`), `RunFailed` (`failure`).
- Node: `NodeStarted`, `NodeSucceeded`, `NodeRouted`, `NodeSkipped`, `NodeFailed`, `NodeExpanded`,
  `NodeAttemptResumed`.
- In-loop AGENT, never recorded: `ModelCallStarted`, `ModelCallFinished`, `ToolCallStarted`,
  `ToolCallFinished`, `ReplayDegraded`.
- Outside a run: `ModelCallRetried`, and a learner's `ReplayDegraded`, reach `result.backward`'s and
  `buildMemories`' `onEvent`, never a run's.
- `StreamChunk`: `nodeId`, `key`, `chunk`, `final`, `n`, `index`. An AGENT's model call events and
  chunks name the step the call becomes: `nodeId` + `n` is its `model_call` step's id (each attempt
  is its own step), and `index` counts the step's chunks from 0. Tool events carry their reply's `n`.

`event.toJSON()` gives the snake_case form that
[`conformance/schema/events.schema.json`][conformance/schema/events.schema.json] describes. Flow
values inside are left as they are (ints as `bigint`, `Infinity`), so `JSON.stringify` needs a
replacer for them. Meanings: the `on_event` and `on_stream` items of
[`run_flow`][docs/api.md#run_flow].

### Following a run live

```ts
const live = new LiveChannel();
const stop = live.subscribe((frame: LiveFrame) => send(logJson(frame)));
await runFlow(flow, inputs, { log, live });
```

A `LiveChannel` (exported by `@ngocbh/yian/run`, with `LiveChannelOptions`, `LiveFrame`,
`LiveFollowFrame`, `LiveSnapshot` and `LiveCall`)
is a run's live channel, passed as `runFlow`'s or `advance`'s `live`. Its frames are plain JSON
with the log's snake_case keys, so a host can send them anywhere with `logJson`:

- `{type: "record", seq, line}`: a line exactly as the log appended it, sent once the append
  returned, `run_started` and every `node_step` included; decode it with the log's own reader. A
  resumed run sends only the records it appends.
- `{type: "start", node, n, model, purpose, started_at}`, `{type: "chunk", node, n, index, text}`
  and `{type: "end", node, n, error}`: one AGENT model call attempt, named by the step it becomes
  (`node` + `n`, as `StreamChunk` and the call events name it), its text chunks numbered by `index`
  from 0. `error` is the error's name, as on `ModelCallFinished`; a failed attempt's record, with
  its scrubbed message, comes first.

Records never reach `onEvent`, and frames never enter the log. `subscribe(fn)` returns the function
that stops it. The stream is ephemeral, so a subscriber never affects the run: one that throws is
dropped and reported to `new LiveChannel({ onSubscriberError: (error, frame) => ... })`, a yian
warning by default, and the frame still reaches the others. `snapshot()` returns
`{last_seq, calls: [{node, n, chunks, next_index}]}`: the seq of the last record (`-1` before any)
and each call in flight with its chunks so far. The channel keeps one buffer per call, so calls of
parallel nodes (a MAP's items) stay apart; a call leaves it when its step record is appended (a
failed attempt's comes before its `end` frame), when its node settles (a stop), or when a
`node_failed` or `run_stopped` record ends the run.

`follow(fn)` is how a late viewer joins: it calls `fn` with `{type: "snapshot", snapshot}`, then
with every later frame (a `LiveFollowFrame`), and returns the function that stops it. The snapshot
and the subscription are taken in one synchronous turn, so every frame after the snapshot is new:
records continue after `last_seq`, and each call's chunks after its `next_index`. The viewer reads
the log up to `last_seq` with `iterRunRecords` (this may be async), then applies the frames that
came meanwhile, and follows.

### Following a live run from another process

Yian ships no transport: the host carries frames over its own channel, and Yian gives it both ends.

1. The writing host calls `live.follow(fn)` and sends each frame, as one `logJson` line, on its
   channel.
2. The serving host keeps a `LiveView` per run: `LiveView.from(iterRunRecords(log), snapshot)`,
   then `view.apply(frame)` for each later frame, including those that came while it read the log.
3. For a new viewer, the serving host takes `view.lastSeq` and `view.calls()` (`{node, n, text,
   nextIndex}` each) and registers the viewer for later frames in the same synchronous turn; the
   viewer gets the log up to that seq and those calls, then the frames.
4. `apply` returning `false` is a gap (a record or chunk went missing): stop the old `follow`,
   ask `follow` again and rebuild the view.

`LiveView` (exported with `LiveViewCall`) never reads files. `from` reads only each record's `seq`
and ignores records past `last_seq`, which the frames bring again; it throws `RangeError` unless the
records run from seq 0 to `last_seq`. Frames must arrive in the order they were sent: a duplicate
`start` delivered after its call's step record would bring the call back. `apply` folds one frame:
a record at or below `lastSeq`, the start of a call in flight, or a chunk below its call's
`nextIndex` is a duplicate and changes nothing; a chunk of a call not in flight is a gap. A call
goes when its step record arrives or when it ends with an error, and every call goes on
`node_failed` or `run_stopped`, which end the run. A run that ends without either (a host fault,
or its process gone) sends nothing more, so the host drops the view when the run ends. Each call
has its own buffer, keyed by `(node, n)`, so a MAP's parallel calls stay apart, and a late
snapshot lists every call in flight.

## Recovering and inspecting runs

```ts
advance(options: AdvanceOptions): Promise<RunResult>
loadRecordedFlow(log: string, options?: { env?: (name: string) => string | undefined; codeRoot?: string | null }): Promise<ComposedFlow | null>
foldRunLog(log: string, options?: { env?: (name: string) => string | undefined; flow?: ComposedFlow | null; codeRoot?: string | null }): Promise<FoldedRun>
inspectRun(log: string, options?: { flow?: ComposedFlow | null; env?: (name: string) => string | undefined }): RunInspection
updateRunIndex(runsDir: string, options?: { timeoutMs?: number }): void
readRunIndex(runsDir: string, options?: { timeoutMs?: number }): RunIndexRow[]
```

`advance` takes `log` (required) and `runFlow`'s `numWorkers` (`null` keeps the recorded one),
`onEvent`, `onStream`, `live`, `memory`, `tools`, `skills`, `steering`, `context`, `signal`, `env`,
`scheduler`, `random` and `models`, plus `loadFlow`'s `codeRoot`, and `apiKey` and `headers` for
a run started with `llmConfig.api_key`. It runs the flow `RunStarted` recorded, compiled from the
recorded flow files; `flow` is only for a run whose flow was built in code (`RunStarted.flow` is
`null`).
`loadRecordedFlow` compiles a log's recorded flow the same way, for a host that needs it first;
both follow each recorded file's `refs` to the files it uses, never the disk.
With `codeRoot`, both refuse a recorded CODE reference whose file's real path is now outside
it, as `loadFlow` does, with `LoadError`; `advance` refuses before it appends anything.
`foldRunLog` takes `codeRoot` too and refuses the same way before its replay.
The specs of the tools and skills the AGENTs grant are recorded, and an unfinished run's must
match the ones passed; tool implementations, memory and context aren't recorded: pass the ones
the run had. A resumed AGENT replays its recorded model calls and tool results instead of making
them again; a recorded call the resumed host would ask differently rejects the resume and leaves
the run resumable. A log another yian version started resumes with a warning. See [Durable recovery
with `resume_flow`][docs/api.md#durable-recovery-with-resume_flow].

`inspectRun` reads a log without its lock: `status` (a `RunStatus`: `"running"`, `"succeeded"`,
`"failed"`, `"stopped"` or `"crashed"`), `inFlight`, `crashedAttempts`, `runId`, `schema`,
`flowFingerprint`, `learnable`, `lastSeq`, `startedAt`, `endedAt` and `failure`. Pass `flow` for an
exact status. See [Reading a run log with
`inspectRun`][docs/api.md#reading-a-run-log-with-inspectrun].

For hosts that show runs: `iterRunRecords(log, { afterOffset, runId, afterSeq, kinds, payloads })`
streams a log another process may be writing, yielding `RunLogEntry`s and throwing
`RunLogChangedError` when the file no longer continues what was read ([Following a run
log][docs/api.md#following-a-run-log-with-iterrunrecords]); `staticNodes(flow)` lists `[path, node]`
for every node of a flow and its child flows, and `staticPathOf(nodeId)` maps a grown node's id onto
that path ([Naming a run's nodes][docs/api.md#naming-a-runs-nodes-with-staticnodes]). A log of
another schema throws `RunLogSchemaError`. Every pool write is a `pool_commit` record: the run's
system values and inputs right after `run_started`, and each node's output before the
`node_succeeded` that names its seq (`commit`); `runRecordOffsets(log)` maps each seq to its line's
byte offset, so a host reads the commit a success names with `iterRunRecords(log, { afterOffset })`.
A crash between a commit and its success leaves an orphan the resume ignores (the node reruns); a
log that ends inside its run start can't be resumed. See [Authoritative run
logs][docs/api.md#authoritative-run-logs].

A derived commit holds no value, only `from`: a non-top `__end__`'s, the stopping LOOP driver's,
and the commits an expansion writes for its child STARTs before its `node_expanded` (which holds no
seed). Each pool name the binding read maps to the seq of the `pool_commit` that wrote it, and a
replay rebuilds the value. `foldRunLog(log)` replays a log through the run reducer, as resume does,
and returns `{flow, records}`: each record (`entry`, as `iterRunRecords` reads it) with `value`, the
`TypedValue` an accepted `pool_commit` writes (held, or rebuilt from its `from`), or `null` for an
orphan commit and any other record. A host reads a LOOP iteration's value, `<loop>#k/__end__`, this
way (`PoolCommitRecord` is exported from `@ngocbh/yian/run` too):

```ts
const { records } = await foldRunLog(log);
const turn = records.find(({ entry }) => {
  const record = entry.envelope.record;
  return record instanceof PoolCommitRecord && record.key === "chat#3/__end__";
});
const history = turn?.value?.toObject();
```

A non-derived `pool_commit` payload is flat, `{node_id, key, kind, value}`. A host decodes its value with
`TypedValue.fromWire({ kind, value })` (an int is a `bigint`, a file a `FileRef`), writes it back as
the log does with `logJson(typed.toWire())`, and reads that text with `parseJson(text, "codec")`,
which throws `JSONDecodeError` on text yian doesn't read:

```ts
const typed = TypedValue.fromWire({ kind: payload.kind, value: payload.value });
const text = logJson(typed.toWire()); // {"kind":…,"value":…}, the value as the log writes it
const back = TypedValue.fromWire(parseJson(text, "codec"));
```

See [Reading values from a run log][docs/api.md#reading-values-from-a-run-log].

`updateRunIndex(runsDir)` keeps `<runsDir>/index.sqlite`, one row per run folder
`<runsDir>/<run_id>/log.jsonl`: run id, flow file, status, start and end times, parent run, last
seq, result (the top-level END's value as `logJson` text) and token sums over every `model_call`
step, failed attempts included. It reads each log on from the byte offset its row reached, reads a
truncated or replaced log again from the start, and serializes builders in one `BEGIN IMMEDIATE`
transaction that waits at most `timeoutMs` (default 5000) and then throws `RunIndexBusyError`.
`readRunIndex(runsDir)` returns the rows, with `running` in place of a `stopped` or `crashed`
status while a writer holds the run's lock. The index may lag the logs, never leads them, and can
be deleted and rebuilt. See [The run index][docs/api.md#the-run-index-with-updaterunindex].

## Tools

```ts
tool({ name, description, parameters, output?, parametersJsonSchema?, outputJsonSchema?, execute, isConcurrencySafe? }): FunctionTool
new FunctionTool({ name, description, parametersJsonSchema, execute, isConcurrencySafe?, parametersValidator?, outputJsonSchema?, outputValidator? })
```

`parameters` and `output` take a raw JSON Schema or a Standard Schema v1 validator; with a validator
that has no Standard JSON Schema interface, pass its JSON Schema beside it. `execute(args, ctx)`
returns the result or a Promise of it. `ctx` is a `ToolContext<T>`: `callId`, `attempt`, `toolName`,
`runId`, `node`, `context` (the run's `context` option) and `signal`. See [Tool
context][docs/api.md#tool-context] and [Typed tools][docs/api.md#typed-tools].

A tool controls retries and what the model reads with its errors ([TOOL
retries][docs/api.md#tool-retries]): `new PermanentToolError(message)` stops retries,
`new TransientToolError(message, { retryAfter: 5 })` asks for a retry after that many seconds, and
`new ReportedToolError(message, { permanent })` is read by the calling AGENT's model as the call's
result. `toolErrorKinds(error)` returns the `ToolErrorKind`s (`"permanent"`, `"transient"`,
`"reported"`, `"check"`) an error's `TOOL_ERROR` brand holds, and the `is*ToolError` guards read the
same brand; `ToolCheckError` is thrown when one of yian's own checks of a call fails. A TOOL node
whose retries ran out fails with a `ToolAttemptsFailed` listing every attempt (its `cause` is the
last attempt's error).

The tools a run sees: without `tools`, the process registry; with `tools`, exactly that record.

- `registerTool(toolId, tool, { descriptionOverride })` registers a `FunctionTool` (its `name` must
  equal `toolId`) in `TOOL_REGISTRY`; `registeredTools()` returns the registry as a record, and
  `resolveTools(ids)` looks registered tools up. `invokeTool(toolId, args)` checks and calls a
  registered tool; `invokeToolTrusted(toolId, args)` checks it against the full schema,
  host-injected fields included.
- `new ToolSet(tools, { context })` holds one run's tools (the registry when `tools` is omitted),
  and `set.forRun(runId, { signal })` binds a copy to a run. `await set.validate(toolId, args)`
  returns a `CheckedCall` that `set.invoke(call, options)` runs and `set.isConcurrencySafe(call)`
  reads; `set.invoke(toolId, args, options)` and `set.invokeTrusted(toolId, args, options)` validate
  first. `InvokeOptions` holds `node`, `callId`, `attempt` and `signal`.
- A `ToolEntry` is a `FunctionTool`, an array of them (a [tool group][docs/api.md#tool-groups]), an
  `McpServer` (a group of its tools), or, under the key `ask` (`ask_user` is its old name), an
  `AskHandler` `(request, ctx) => answer` or an `AskTool` `{ handler, inputs, required? }` that
  declares the extra inputs a TOOL `ask` may pass ([Blocking human
  interaction][docs/api.md#blocking-human-interaction]). The one handler serves the AGENT's
  `ask_user` tool and the TOOL built-in `ask`, and gets an `AskRequest`: `question`, `options`,
  `questions`, `answerSchema`, `title`, `nodeId`, `callId`, `deadline` and `extra`. A flow that uses
  registered tools and asks passes `{ ...registeredTools(), ask: handler }`.
- `take`, `sleep`, `ask`, `ask_user` and `load_skill` are reserved: `registerTool`, a `tools` key
  (other than the ask slot), a group tool and an MCP server can't take one. `take` and `sleep` are
  engine built-ins a TOOL node reaches with `tool_id:`.
- `pending()` returns the marker a host handler returns when its answer comes later;
  `isPending(value)` reads its brand. `ToolSet` passes the marker through without an output
  check.

### MCP servers

```ts
mcpServer({ command, args?, env?, cwd?, timeout?, connectTimeout?, label?, concurrencySafe? }): McpServer
mcpServer({ url, headers?, timeout?, connectTimeout?, label?, concurrencySafe? }): McpServer
```

The server starts only when opened: `await server.open({ signal })`, then `server.tools`, and
`await server.close()` (or `await using`). An unopened server passed in `tools` is opened for that
run and closed when it ends. Timeouts are in seconds. Errors are `McpError` subclasses (`isMcpError`
checks the brand): `McpConnectError`, `McpToolError`, `McpRequestError`, `McpInvalidRequestError`,
`McpTimeoutError`, `McpTransportError`, `McpServerClosed`, `McpCallCancelled`. Requires the optional
peer `@modelcontextprotocol/client`. See [MCP servers][docs/api.md#mcp-servers].

## Memory

`await buildMemories(flow, { root, storage, supplied, factories, env, models, onEvent })` builds one
backend for each memory key the flow declares and resolves to a `MemoryRegistry`. `root` is the
directory for per-flow memory files, `storage` a record of a file per built-in key, `supplied` a
record of ready backends (the registry doesn't close them), and `factories` a record of constructors
for custom types, `(config) => backend`, possibly async. `env`, `models` and `onEvent` reach the
reflective and profile learners (`onEvent` gets their `ModelCallRetried` and `ReplayDegraded`).

A `MemoryRegistry` is a read-only `Map` of memory key to backend, with `types` (the declared type of
each key) and `close()`; pass it as `runFlow`'s `memory`, and close it when done
(`await using registry = await buildMemories(flow)`). See [Building memories with
`build_memories`][docs/api.md#building-memories-with-build_memories]. A host memory is any object
with `openRun({ snapshot })` and `reset()`; its methods may return promises.

Storage, for hosts that build memories themselves ([Optional durable
memory][docs/api.md#optional-durable-memory]): `new SQLiteStorage(path, backend, { createParent })`
is a single-owner SQLite state file, and `checkStoragePath(path)` returns the absolute path it would
use or throws. A `MarkdownStore` (profile memory's document) has `read(version?)`, returning
`[text, version]` (or a promise of it), and `write(text, { baseVersion, provenance })`, returning
the new version (or a promise of it); `new FileMarkdownStore(path, { createParent })` and
`new InMemoryMarkdownStore(text)` implement it, a stale `baseVersion` throws
`MarkdownVersionMismatch`, and `markdownVersion(text)` is the `"sha256:<hex>"` version of a text.
A store's versions are opaque strings, and `read(v)` returns `v`. It must keep every version it
hands out, since resume and backward read the recorded one; an unknown version throws `RangeError`.

`new ProfileMemory(store, { llmConfig, maxChars, inputBudgetChars, env, createModel, onRetry })`
keeps one profile document in any `MarkdownStore`. Pass it in `runFlow`'s `memory` `Map`, or in
`buildMemories`' `supplied` for a declaration without YAML config.

## Skills

```ts
import { defaultSkillRoots, loadSkills } from "@ngocbh/yian/skills";

const skills = loadSkills(defaultSkillRoots({ cwd: process.cwd() }));
await runFlow(flow, inputs, { skills });
```

`loadSkills(roots, { skipInvalid, warn })` returns a `Map<string, Skill>` sorted by name, an earlier
root winning a name; `loadSkillDir(directory)` reads one skill;
`defaultSkillRoots({ cwd, home, env })` lists the Claude Code and Codex roots, highest precedence
first. `new Skill({ name, description, instructions, directory })` builds one in code; a bad skill
throws `SkillError`. See [Skills][docs/api.md#skills].

## Steering

`steering` is an object with `interruptSignal(nodeId)`, returning an `AbortSignal` the host aborts
to interrupt that AGENT node, and `drain(nodeId)`, returning the queued steer messages (or a promise
of them). It reaches AGENT nodes authored `steerable: true`; pass the same inbox to `advance`.
See [Steering][docs/api.md#steering].

## Provider setup

`@ngocbh/yian/llm_clients` gives hosts that set up providers (a settings page, a login command) the
pieces `yian auth` uses.

```ts
import { ApiKeyCredentials, AuthStore, listModels } from "@ngocbh/yian/llm_clients";

const models = await listModels("anthropic", { apiKey }); // a refused key throws ModelListError
const store = new AuthStore();
await store.save("anthropic", new ApiKeyCredentials(apiKey), models![0]!);
console.log(store.selection()?.model);
```

- `LLMConfig.fromHost(data)` checks a JSON `llm_config` through the host intake `runFlow` uses
  and throws `ValidationError` (a `RangeError` whose `issues` are `Issue`s: `loc`, `type`,
  `msg`), or `TypeCheckError` (a `RangeError`, not exported) for a value that isn't a yian value,
  such as NaN. Its messages never hold a value, but they name record keys as written, at any
  depth (an unknown key, a block field, a header name), so a host that must not echo input names
  shows its own fixed text. `LLM_CONFIG_KEYS` holds the top-level keys.
- `listModels(provider, { apiKey, baseUrl, timeout, headers, organization, project, signal,
  scheduler, request })` resolves to the model ids the provider's host offers, sorted and without
  duplicates, or `null` for `azure`, which has no listing. It reads no environment, so a host can
  try a key before it saves it. A `signal` stop rejects with `signal.reason`; every other failure
  is a `ModelListError` with a short fixed message (`HTTP 401`, `timed out`, ...) and no `cause`;
  its `status` is the HTTP status of an `HTTP <status>` failure, else `null`.
- `new AuthStore(path?, options?)` is the private saved-login file: `path`, else `YIAN_AUTH_FILE`,
  else `~/.config/yian/auth.json` (under `XDG_CONFIG_HOME` when set). `selection(provider?)`
  returns the provider's nonsecret `ProviderSelection`, or the default's, or `null`; `read()`
  returns every selection and the default. `save(provider, credentials, model, { endpoint,
  organization, project, makeDefault })` commits an `ApiKeyCredentials`, an `OAuthCredentials` or
  `null` together with its selection. `credentials(provider, credentialId, { refresh })` returns
  the saved credentials, refreshing an expiring subscription token under the store's lock.
  `delete()` removes the whole store under that lock, so it waits for a refresh in flight, and
  keeps the lock file; it creates nothing when neither the store nor its lock file exists (a
  missing directory too). It throws `AuthError` when the directory isn't private or the lock wait
  passes `lockTimeoutMs` (30 s by default).
- `new OpenAIAuth({ request, scheduler, signal, requestTimeoutMs })` signs in to a ChatGPT
  subscription: `deviceLogin(displayCode, { timeoutMs })` and `browserLogin(openUrl, {
  awaitCallback, timeoutMs })` resolve to `OAuthCredentials`, `refresh(credentials)` renews them,
  and `subscriptionModels(credentials, { clientVersion })` lists the subscription's models. A
  failed request or login throws `AuthError`, whose `status` is the HTTP status of a failed
  authentication request, else `null`.
- One `AuthRequest`, `(method, url, { headers, timeoutMs, jsonBody, form, signal })` resolving to an
  `HttpResponse(status, data)`, serves as both `OpenAIAuth`'s and `listModels`' `request`
  (`listModels` also passes `bodyLimit` and `scheduler`). Unset, both use yian's sender, which
  never follows a redirect.
- `SUPPORTED_LLM_PROVIDERS` lists the provider names; `azureResourceEndpoint(url)` strips an
  `/openai/v1` or `/openai` suffix from an Azure portal URL, leaving the resource endpoint.

## Errors and failure codes

A failed run's `result.failure` is a `Failure`: `code`, `message`, `nodeId`, `trace`, `causeType`
(for debugging only) and `traceback`, with `toJSON()`. `code` is one of the codes in [Failure
codes][docs/api.md#failure-codes]; TS yian adds `agent.output_invalid`, `provider.rate_limit`,
`provider.quota`, `provider.unavailable`, `provider.context_length`, `provider.refused` and
`condition.evaluate`, which Python reports as `node.execute` (or, for an `asserts:` condition,
`assertion.failed`). A failed run is terminal; branch on `code`, never on `causeType`.

A `Diagnostic` (from `LoadError.diagnostic` or `flow.diagnose(failure)`) holds `code`, `message`,
`frames` (each with `label`, `text`, `range` and `note`), `notes`, `nodeId`, `causeType` and
`traceback`.

Host-side errors reject the call instead: `LoadError`, `FlowValidationError` and `ExpressionError`
(all `RangeError`s), `TypeError` and `RangeError` for bad options or inputs, the `Mcp*` errors, the
signal's reason, and the unexported errors named in [Differences from Python
yian](#differences-from-python-yian). Each function's TSDoc lists what it throws.

## Expressions and values

`evaluateWhen(expression, pool, types?)` evaluates a boolean `when:`/assert expression against a
`VariablePool`, the run's typed value store ([Types →
`VariablePool`][docs/typing.md#variablepool--the-runs-typed-value-store]), and throws
`ExpressionError` for a bad expression.
`registerTemplateFn(name, { params: [{ name, default? }], call(...args) })` adds or replaces a
function `${...}` expressions can call; each `default` goes through host intake.

## `env`, `signal` and `scheduler`

- `env` is a function `(name: string) => string | undefined`. `loadFlow`, `runFlow`, `advance`,
  `inspectRun`, `buildMemories` and `defaultSkillRoots` read the environment through it instead of
  `process.env`: connections, credentials, `HOME`, and every `YIAN_*` default and limit. Concurrent
  calls with different `env`s each keep their own values. A bad `YIAN_*` value rejects with
  `DefaultsError`; a resume whose recorded values differ, with `DefaultsMismatchError`. See
  [Defaults and limits][docs/api.md#defaults-and-limits] and [`docs/defaults.md`][docs/defaults.md].
- `signal` is the host's stop for `runFlow`, `advance` and `McpServer.open`. An aborted run
  appends a stop record, lets the nodes in flight wind down and rejects with the signal's reason;
  `advance` retries the stopped nodes. An already-aborted signal rejects before anything is
  created.
- `scheduler` (`{ now(), sleep(ms, signal) }`), `random` (a draw in [0, 1) for backoff jitter) and
  `models` (the model builder AGENT nodes use) are for tests and conformance: virtual time, fixed
  jitter and scripted models. Hosts leave them unset.

## File locks and SQLite

`@ngocbh/yian/platform` gives hosts the exclusive file lock and the SQLite wrapper yian uses itself.

```ts
import { openSqlite, waitForLock } from "@ngocbh/yian/platform";

using lock = await waitForLock("/srv/app/app.lock", { timeoutMs: 5000, private: true });
const db = openSqlite("/srv/app/app.sqlite");
db.exec("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)");
db.prepare("INSERT OR REPLACE INTO kv VALUES (?, ?)").run("a", "1");
db.close();
```

The lock is a SQLite exclusive transaction on a 0-byte file, created with mode 0600 when missing (an
existing file keeps its mode). The kernel frees it when the process dies; the file is never
unlinked, and no sibling file appears. Bun and Node processes see each other's locks. It is POSIX
only (`private` reads `process.getuid()`). Never open a lock file any other way: on Node, closing
any descriptor of a locked file drops the process's lock.

- `lockFile(path, { private })` takes the lock without waiting and returns a `FileLock`: call
  `release()`, or hold it with `using`. The lock isn't re-entrant: a second `lockFile` in the
  holding process is busy too. OS errors (`ENOENT` for a missing directory) and SQLite errors
  (`SQLITE_NOTADB` for a foreign file) propagate unchanged; any other failure throws
  `FileLockError`, whose `reason` is:
  - `"busy"`: the lock is held. Busy can also show up briefly while another contender or an
    `isLocked` probe is mid-attempt, so to ask "is another daemon running?", call `waitForLock`
    with a small `timeoutMs`.
  - `"unsafe"`: the path isn't a regular file, or, with `private`, has more than one link.
  - `"not-private"`: with `private`, the file isn't owned by this user, or has group or other mode
    bits set.
  - `"changed"`: the path was replaced or removed while locking.
- `waitForLock(path, { timeoutMs, signal, private })` retries every 100 ms until its sleeps add up
  to `timeoutMs` (default `Infinity`), then rejects with `FileLockError` `"busy"`; `0` tries once.
  An abort rejects with `signal.reason`, a `NaN` or negative `timeoutMs` with `RangeError`, and any
  other `lockFile` failure propagates at once.
- `isLocked(path)` says whether another connection holds the lock. A missing path, a non-regular
  file and a foreign (non-SQLite) file are `false`, and nothing is created. Any other open failure
  (no read permission, say) throws, so a held lock is never reported free. Like busy, `true` can
  show up briefly while a `lockFile` call is mid-attempt.

`openSqlite(path, { readonly })` opens a database file (creating it unless `readonly`) through
`bun:sqlite` or `node:sqlite`. A `Sqlite` has `exec(sql)`, `prepare(sql)` (cached per SQL text) and
`close()` (a second call does nothing); a `SqliteStatement` has `all`, `get` (`undefined` for no
row) and `run`. Parameters are positional `?`s: a safe-integer `number` binds as INTEGER, any other
`number` as REAL. Integers read back as `number`, and one outside the safe range throws
`RangeError`.

Foreign keys are off and double-quoted string literals are accepted, as in Python's sqlite3. Two
runtime differences remain: passing some but too few positional parameters throws on Bun but binds
NULL on Node (passing none binds NULL on both), and the drivers' error objects differ, so read the
extended result code with `sqliteErrorCode(error)` (`& 0xff` for the primary code).

## CLI

The `yian` command (`npx yian`, or a binary) is described in [`docs/cli.md`][docs/cli.md];
`yian --help` lists its commands, and `yian --version` prints the version. yian's warnings (an
unknown model, TLS checks off, ...) reach a library host through `process.emitWarning`
(`process.on("warning")`); the CLI prints them as `warning:` lines.

[conformance/schema/events.schema.json]: https://github.com/ngocbh/yian/blob/main/conformance/schema/events.schema.json
[docs/api.md]: https://github.com/ngocbh/yian/blob/main/docs/api.md
[docs/api.md#authoritative-run-logs]: https://github.com/ngocbh/yian/blob/main/docs/api.md#authoritative-run-logs
[docs/api.md#blocking-human-interaction]: https://github.com/ngocbh/yian/blob/main/docs/api.md#blocking-human-interaction
[docs/api.md#building-memories-with-build_memories]: https://github.com/ngocbh/yian/blob/main/docs/api.md#building-memories-with-build_memories
[docs/api.md#defaults-and-limits]: https://github.com/ngocbh/yian/blob/main/docs/api.md#defaults-and-limits
[docs/api.md#diagnosing-a-failed-run]: https://github.com/ngocbh/yian/blob/main/docs/api.md#diagnosing-a-failed-run
[docs/api.md#durable-recovery-with-resume_flow]: https://github.com/ngocbh/yian/blob/main/docs/api.md#durable-recovery-with-resume_flow
[docs/api.md#explicit-backward-feedback]: https://github.com/ngocbh/yian/blob/main/docs/api.md#explicit-backward-feedback
[docs/api.md#failure-codes]: https://github.com/ngocbh/yian/blob/main/docs/api.md#failure-codes
[docs/api.md#following-a-run-log-with-iterrunrecords]: https://github.com/ngocbh/yian/blob/main/docs/api.md#following-a-run-log-with-iterrunrecords
[docs/api.md#load_flow]: https://github.com/ngocbh/yian/blob/main/docs/api.md#load_flow
[docs/api.md#mcp-servers]: https://github.com/ngocbh/yian/blob/main/docs/api.md#mcp-servers
[docs/api.md#naming-a-runs-nodes-with-staticnodes]: https://github.com/ngocbh/yian/blob/main/docs/api.md#naming-a-runs-nodes-with-staticnodes
[docs/api.md#optional-durable-memory]: https://github.com/ngocbh/yian/blob/main/docs/api.md#optional-durable-memory
[docs/api.md#reading-a-run-log-with-inspectrun]: https://github.com/ngocbh/yian/blob/main/docs/api.md#reading-a-run-log-with-inspectrun
[docs/api.md#reading-values-from-a-run-log]: https://github.com/ngocbh/yian/blob/main/docs/api.md#reading-values-from-a-run-log
[docs/api.md#run_flow]: https://github.com/ngocbh/yian/blob/main/docs/api.md#run_flow
[docs/api.md#runresult]: https://github.com/ngocbh/yian/blob/main/docs/api.md#runresult
[docs/api.md#skills]: https://github.com/ngocbh/yian/blob/main/docs/api.md#skills
[docs/api.md#steering]: https://github.com/ngocbh/yian/blob/main/docs/api.md#steering
[docs/api.md#the-run-index-with-updaterunindex]: https://github.com/ngocbh/yian/blob/main/docs/api.md#the-run-index-with-updaterunindex
[docs/api.md#tool-context]: https://github.com/ngocbh/yian/blob/main/docs/api.md#tool-context
[docs/api.md#tool-groups]: https://github.com/ngocbh/yian/blob/main/docs/api.md#tool-groups
[docs/api.md#tool-retries]: https://github.com/ngocbh/yian/blob/main/docs/api.md#tool-retries
[docs/api.md#typed-tools]: https://github.com/ngocbh/yian/blob/main/docs/api.md#typed-tools
[docs/cli.md]: https://github.com/ngocbh/yian/blob/main/docs/cli.md
[docs/defaults.md]: https://github.com/ngocbh/yian/blob/main/docs/defaults.md
[docs/syntax.md]: https://github.com/ngocbh/yian/blob/main/docs/syntax.md
[docs/syntax.md#a-list-prompt]: https://github.com/ngocbh/yian/blob/main/docs/syntax.md#a-list-prompt
[docs/syntax.md#code--tool--the-other-leaves]: https://github.com/ngocbh/yian/blob/main/docs/syntax.md#code--tool--the-other-leaves
[docs/typing.md#two-constructors-infer-vs-check]: https://github.com/ngocbh/yian/blob/main/docs/typing.md#two-constructors-infer-vs-check
[docs/typing.md#variablepool--the-runs-typed-value-store]: https://github.com/ngocbh/yian/blob/main/docs/typing.md#variablepool--the-runs-typed-value-store
[issue-736]: https://github.com/ngocbh/yian/issues/736

## Developing

Notes for working on `ts/`. Python yian in `src/yian/` is the reference the port mirrors until
the cut-over. Parity status on 2026-10-22: the whole shared conformance suite passes on Bun and on
Node 22, and so does every unit test; a change keeps both green.

### Layout

- `src/<pkg>/<mod>.ts` mirrors `src/yian/<pkg>/<mod>.py`.
- `tests/<area>/<x>.test.ts` mirrors `tests/<area>/test_<x>.py`.
- Relative imports use `.ts` specifiers; `tsc` rewrites them to `.js` in `dist/`.
- No `Bun.*` or `bun:*` outside `src/platform.ts`.

### Conventions

- Every yian error class sets `override name = "<Class>"`: hosts and the conformance runner
  read `error.name`.
- Sort strings with `compareCodeUnits` from `src/diagnostics.ts` (UTF-16 order); the "src sorts
  with a comparator" lint bans a bare `.sort()` or `.toSorted()`.
- Every event in `src/events.ts` has a `type` naming its class and a `toJSON()` giving its
  snake_case JSON form (`../conformance/schema/events.schema.json`); flow values inside are left
  as they are, so write the form with yian's JSON codec, not `JSON.stringify` (it turns infinity
  into null and throws on a bigint).
- Read limits and settings with `current()` from `src/defaults.ts`. Read the environment only
  through an injected `Env`: `processEnv` in `defaults.ts` is the one place that touches
  `process.env`, and a lint checks it. Unit tests read the process environment's `YIAN_*`
  variables, so run them with none set; the test files that set limits reset them with
  `install(BUILTIN)` or `forget()`.
- `src/platform.ts` holds the runtime-specific APIs: `runtime`, SQLite (`openSqlite`,
  `sqliteErrorCode`), the exclusive file lock (`lockFile`, `waitForLock`, `isLocked`), time
  (`Scheduler`, `realScheduler`, `virtualScheduler`, `SLEEP_CHUNK_MS`) and HTTP (`safeFetch`
  over an injectable `Send`, which never follows a redirect, throws `RedirectError` for a 3xx
  and sends `USER_AGENT`; every request goes through it). The lock and SQLite names are public
  through `@ngocbh/yian/platform` (see [File locks and SQLite](#file-locks-and-sqlite)), so their
  behavior is API. Code that waits takes an injected `Scheduler`, so tests and conformance run on
  `virtualScheduler`. `stripTypes` strips a CODE body's types with amaro on both runtimes, keeping
  every line number, and returns text: the caller builds the function.
- Host values enter through `fromHost`; values reaching `buildValueAs` are yian values.
- A value becomes text with `writeAsText` (the text form) or `writeAsJson` (`render_as_json`),
  never `String()` or `JSON.stringify`; messages show values with `valueRepr`.
- A `float` slot given an int stores kind `int`, and a `list[float]` item may be an int
  (`bigint`): code reading a float slot handles both (#535, `int-in-float`).

### YAML

- In `src/`, only `src/compose/source/` imports `yaml` (tests may import it).
- It parses with `FLOW_OPTIONS` (YAML 1.2, failsafe schema), so the library resolves no
  scalar; `scalars.ts` applies yian's table.
- `composeFlowDocument` resolves aliases and applies the nesting and node limits.
- Ints are `bigint`, dates stay text, `.inf` is a float, and `.nan` and an overflowing float
  are refused.
- In a flow document, a local type tag never changes how its scalar reads (`!float 3` is `3n`);
  `parseYamlLiteral` refuses local tags.
- `SourceDocument.parse` reads lone CRs as LF, composes once, lexes the CST for property
  positions, and keeps every node as a frozen definition with text keys; it reports the first
  `nonTextKey` and `repeatedKey` for Compose to refuse.
- `SourceCursor` navigates a document's values, which are frozen; `value` returns a copy.
- `SourceCatalog` turns an opaque origin into a diagnostic frame by replaying the occurrence
  path it kept.

### Compose

- `src/compose/helpers/errors.ts` holds `LoadError`, the one error a load throws. It carries one
  `Diagnostic`, whose code is always `compose.load`, and is frozen.
- Build it with a `LoadDiagnostics` from the document: `LoadDiagnostics.fromDocument(document)`,
  then `.error(message, { range, cause })`; `.prepend` returns a fresh error with an outer frame
  before the inner ones.
- Messages keep Python's wording: names and source text go through `reprText`, lists of names
  and values through the text form (`writeAsText`).
- Every value written in a flow file goes through `checkWrittenValue`
  (`src/compose/helpers/value_nesting.ts`) before anything reads it: it refuses a value nested
  past the value limit and, a TS-only rule, text or a record key holding a lone surrogate, as a
  located `LoadError`. `checkEnvValues` does the same for `env:`, and also refuses a value that
  holds itself through an alias.
- `src/compose/helpers/` reads documents through `compose/source` and never imports `yaml`.
- `src/compose/helpers/tags.ts` reads every slot through its tags (`readTaggedValue`) and
  records each tag it reads in the load's ledger; `rejectUnreadTags` refuses the rest.
- `src/compose/helpers/types.ts` reads `output:` types and `input:`/`variables:` declarations:
  the tag is the type and the value after it the default, checked with `checkWrittenValue`
  first.
- `src/compose/helpers/narrowing_facts.ts` and `condition_guards.ts` hold the narrowing inside
  conditions: what a check proves at each read, impossible checks and narrowed reads.
  `output_contracts.ts` types output bindings, and `isinstance_types.ts` resolves the type names
  of `isinstance` and constructors. These helpers return messages and never throw a `LoadError`.
- Kind readers live in `src/compose/nodes/<kind>.ts`. YAML counts (`retries:`) are checked as safe
  integers and stored as numbers; LOOP's `times:` and `max:` stay raw on the descriptor and are
  checked and converted when the LOOP builds. An `llm_config:` layer holding `${...}` values is a
  `ValueTemplate` (a `Binding` for one whole-record value), read before the node builds. The CODE
  reader compiles nothing at build. The awaited load step, `checkFlowCode`, which the loader
  awaits once after assembly (each def's CODE nodes first, in written order, then the root's),
  checks every CODE node of the flow and its child flows: an inline body's syntax and a
  reference's file, refusing a Python-style `module:function` and, when the load has a
  `codeRoot`, a reference whose file's real path is outside it, running no code. A CODE node's
  `baseDir`, the directory its reference resolves against, comes from `BuildContext.baseDir`:
  `loadFlow`'s `baseDir`, else its first search path, else the current directory, for the root,
  its defs and its inline bodies alike; a file read through the search path uses its own
  directory. WAIT keeps an `until:` date or datetime as the text written, after checking it
  parses.
- `src/compose/loader.ts` is the front door: `loadFlow` reads every `uses:` file (through a
  `ChildResolver`, sync or async) before the flow assembles, and returns a Promise.
- `src/compose/index.ts` re-exports Python's `__all__` only, with the loader's `ChildResolver`.
- `src/compose/validate/` holds the validate phase, the cross-node checks over the assembled flow.
  Outside it only `src/compose/build/` imports it: the pipeline runs the passes, and `boundary.ts`
  takes the `LoweredAsserts` type from `asserts.ts`. Inside it, a pass imports another validate
  module only for its shared helpers (such as `Narrowing` from `narrowing.ts` and `LocalInput` from
  `typed_bindings.ts`).

### Skills

- `src/skills/` is a leaf: it holds `Skill`, the `load_skill` tool and the system-message
  listing, and imports only `defaults`, `diagnostics`, `typesys` and `llm`. Its `index.ts`
  never re-exports the Compose loaders.
- `src/compose/skills.ts` (`loadSkills`, `loadSkillDir`, `defaultSkillRoots`) reads `SKILL.md`
  directories and parses their frontmatter with `parseYamlLiteral`.
- Reads are synchronous. Skill files are read as strict UTF-8, with CRLF and lone CR read as LF.

### Tools

- `src/tools/` imports no engine or front-end layer (`nodes`, `compile`, `compose`, `runtime`,
  `run`, `cli`); `index.ts` is its public face.
- A host tool is a `FunctionTool` (`execute(args, ctx)`, sync or async); a run's ask handler is
  wrapped into two, `ask_user` and `ask`, so a `ToolSet` holds only `FunctionTool`s.
- Tool errors, `FunctionTool`, `McpServer` and the MCP errors are recognized by `Symbol.for`
  brands, never by `instanceof`, so a tool built against another copy of yian works.
  `toolErrorKinds`, the tool error guards (`isPermanentToolError`, ...) and `isMcpError` read
  the error brands; the registry checks the `FunctionTool` and `McpServer` brands.
- Arguments enter through `fromHost` and reach `execute` through `toHost`: fresh arrays and
  records, ints within ±(2^53−1) as `number`, larger ones as `bigint`.
- `tool()` builds a `FunctionTool` from Standard Schema v1 validators (with the Standard JSON
  Schema interface) or raw JSON Schemas. yian depends only on those interfaces; zod, valibot and
  ArkType are dev dependencies for the tests.
- `validate` checks a call (JSON Schema, then the validator once) and returns a `CheckedCall`;
  `invoke(call)` runs it without checking the arguments again and `isConcurrencySafe(call)` reads
  it; both see what the validator returned, or the checked arguments when there is no validator.
  `invoke(toolId, args)` validates first.
- `invoke` returns the tool's result raw, awaited, with `undefined` read as `null`, unless the
  tool declares an output: then it checks the result and returns the checked value in host form.
  A `pending()` marker (a `Symbol.for` brand) comes back unchanged and unchecked.
  The callers run value intake on it at their own boundaries (`contract.output`).
- Intake (`fromHost`) refuses getters, setters and Proxies, so no host code runs while it reads
  a value and the value checked is the value recorded.
- A `FunctionTool`'s `parametersJsonSchema` must be JSON within the nesting limit, since the
  model sees it; tools still receive and may return a real `Infinity`.
- `tools/mcp.ts` holds everything MCP: yian's own stdio transport (the server in its own process
  group, lossless decoding of big ints, the message cap, the stderr tail) and the HTTP path,
  which sends every request through `safeFetch`. The SDK (`@modelcontextprotocol/client` 2.3.1)
  is an optional peer dependency, loaded only when a server opens.
- The MCP tests run a fake server (`tests/fixtures/mcp/fake_server.ts`) over stdio and as an
  in-process HTTP server.

### LLM layer

- `src/llm/` is a leaf: it imports only `defaults`, `platform` and `typesys/json` (the layer
  ratchet checks this).
- Messages are `yian.message.v1` (`../conformance/schema/message.schema.json`). Provider payloads
  are opaque JSON, holding the versioned replay records of design 7.5 (`replayRecord`,
  `readReplayRecord` in `src/llm/messages.ts`); a record that doesn't match the request is
  left out, never an error. Every reasoning part carries a replay record; a user message's
  `author` is rendered per adapter (`withAuthorWrapper`, `escapeAuthorTagsInJson` in
  `src/llm/body.ts`).
- Every model call streams. A call ends with exactly one `Finish`, its last event, or a thrown
  `ProviderError`, whose `kind` is a stable error kind (failure codes are assigned above
  `llm/`) and whose `retryable` is set when it is built. Any other error is a defect and is
  never retried.
- Every request goes through `safeFetch` in `src/platform.ts` (no redirects, `User-Agent`, one
  warning per process when `NODE_TLS_REJECT_UNAUTHORIZED=0`, read through `processEnv`); code
  that needs a fake or a cassette injects a `Send`, never a global `fetch`.
- An adapter is a `WireProtocol` (`src/llm/executor.ts`), one file per wire format in
  `src/llm/protocols/`: it builds the body from a `Request` and turns SSE or NDJSON frames into
  events. `RouteModel` sends it over a `Route` (`src/llm/route.ts`, with an auth object from
  `src/llm/auth.ts`), one attempt per call, under yian's first-response and idle timers
  (`src/llm/transport.ts`). An adapter's `ProviderError` holds the provider's text uncapped;
  `RouteModel` scrubs it of the route's credentials, then caps it. `shortRequest` is for model
  listing and sign-in.
- `scrubError(error, redact)` in `src/llm/errors.ts` rewrites an error's `message`, `stack`,
  `cause` chain and `AggregateError.errors` in place with a text redactor (`llm/` can't import
  `llm_clients/`, so callers pass `(text) => redactKeys(text, secrets)`); it never throws.
- Retries happen above the call: `callWithRetries` retries a retryable `ProviderError`, before or
  after partial output, up to `MODEL_RETRIES` times, calling `onRetry` before each wait.
- `src/llm/structured.ts` chooses how a structured call is sent per route, model and schema (a
  tool, an output format, or a prompt), and parses the reply into a JSON object the caller
  validates; `src/llm/json_schema.ts` holds the schema transforms, and adapters render
  `Request.outputFormat`.
- `ScriptedModel` in `src/llm/testing.ts` scripts model replies for tests. Cassettes
  (`yian.cassette.v1`) replay HTTP exchanges as a `Send` (`CassetteReplay`); `CassetteRecorder`
  records them with every credential redacted by value, and refuses to save one that still holds
  a credential. Record only when `recordMode(processEnv)` says so (`RECORD=1`, never under `CI`).

### LLM clients

- `src/llm_clients/` holds `LLMConfig`, the provider block decoders (one field-spec table per
  API, the key walk, the value pass), the connection-setting checks and the key-variable table.
- Only `connection.ts`, `factory.ts` and `auth/store.ts` read the environment, through an `Env` parameter
  (default `processEnv`); the rest reads none. Limits come from `current()`.
- `connection.ts`: selection (`ConnectionSnapshot`, read once from `YIAN_DEFAULT_PROVIDER`,
  `YIAN_DEFAULT_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` and the saved logins), the connections
  recorded on each keyed agent, and the credentials a call may send (`connectionCredentials`, and
  `callCredentials` for the scrub set).
- `factory.ts`: `chatModelFromConfig` resolves an `LLMConfig` (a host record goes through
  `fromHost` first) to a `RouteModel` on its provider's route: Python's key, base URL and header
  precedence, then one route builder per family (OpenAI Responses and the ChatGPT subscription, the
  Chat Completions vendors, Azure, Anthropic, Gemini, Ollama). Secret connection-header values ride
  on `ExtraHeadersAuth`, so `RouteModel` scrubs them; OpenAI's organization and project ids go in
  `Route.headers`. Replay records carry `account`, the first 16 hex digits of SHA-256 of the key
  sent (OpenAI, Anthropic, name-routed Responses deployments), of the ChatGPT account id, or of the
  Azure resource on an AD token. The unknown-model warning goes through `onWarning`.
- `model_list.ts`: `listModels` lists a provider's model ids over `auth/http.ts`'s `request`, with
  Python's per-provider paging and reasons (`ModelListError`); it reads no environment.
- `index.ts`: the package's exports, as Python's `__init__.py`, and the `./llm_clients` entry (see
  [Provider setup](#provider-setup)).
- `layers.ts`: resolves one agent's `llm_config` layers into its effective config, and checks
  layers and complete selections. It reads no environment or saved settings.
- `redact.ts`: `redactKey`/`redactKeys` remove keys and their masked echoes (`sk-abc****wxyz`)
  from provider text. `connection.ts`'s `scrubCallCredentials(error, config, {env, sent})` scrubs
  an error of everything `callCredentials` lists plus what the route sent.
- `model_catalog.ts`: Python's static model catalog (`MODEL_OPTIONS`, `getKnownModels`);
  `validators.ts`: `validateModel`, whether a model is in the catalog (any model for ollama,
  openrouter and vllm).
- `auth/common.ts`: `AuthError`, the credential classes and `ProviderSelection`.
- `auth/store.ts`: `AuthStore`, the saved-login file in Python's format. Reads are sync. `save`,
  `remove`, `setDefault` and `delete` (the whole store) are async and hold `<path>.lock` through
  `waitForLock`; `credentials` is async and takes the lock only to refresh, so `delete` waits for a
  refresh in flight. Nothing unlinks the lock file: a waiter and a newcomer would then lock
  different files. The lock is a SQLite lock that Python's `flock` doesn't see. Node and Bun have
  no `dir_fd`, so private-path checks use `lstat`, an `O_NOFOLLOW` open and `fstat`. Parallel
  `credentials` calls in one process that refresh the same login share one refresh (success or
  failure); the file lock still serializes processes.
- `auth/http.ts`: `request`, Python's `HttpResponse` over `shortRequest`: one total timeout,
  redirects never followed (a 3xx comes back with `null` data), the body capped.
- `auth/openai.ts`: `OpenAIAuth`, browser and device sign-in, refresh and the model catalogs.
  It takes a `Scheduler` and an `AbortSignal` instead of Python's clock and stop callbacks; a
  login's deadline is one timer that aborts its signal. The browser callback listener is
  `node:net` on `127.0.0.1:1455`.
- `auth/index.ts`: the `auth` package's exports, as Python's `auth/__init__.py`.
- Values are yian values (ints are `bigint`), so a host `llm_config` goes through
  `LLMConfig.fromHost` first.
- `LLMConfig` holds the key, the header values and the blocks in `#private` fields. The key and
  headers are never dumped; the blocks are read through `blockFor` and `dump`.
- Lookups by a user key use `Object.hasOwn`. `PROVIDER_BLOCKS` and the schema tables have no
  prototype, so an inherited name such as `toString` or `__proto__` is never an entry.
- Validation errors are `ValidationError` with pydantic-shaped `issues` (`loc`, `type`, `msg`);
  for a `value_error`, `error` holds the validator's own text.

### Nodes

- The carriers in `src/nodes/contracts.ts` (`Binding`, `ValueTemplate`, `Parameter`, the
  contracts) are frozen. To change an origin, rebuild: `mapOrigins` returns a new carrier, and
  `rewriteOriginCarrier` rebuilds a whole graph; pass it one `seen` map so a carrier reached twice
  comes back as one shared object.
- Supplied text becomes a value only at the run boundary, in `coerceParam` (one input) and
  `coerceInputs` (the flow's supplied inputs). `bindParams` never converts text: a CALL, MAP or
  LOOP edge is not a flow boundary.
- For a parameter with `coerceStrings`, text for a `float` input that spells an int reads as that
  int (`"2"` is `2n`); `"2.0"` reads as a float, and `inf` and `-inf` as infinity.
- `coerceInputs` and `applyDefaults` take and return plain records; `checkInputNames` checks one.
- `bindParams(params, wiring, pool, {item, types, labels, collector, sourceResolver})` takes the
  wiring as a `ReadonlyMap` (`name -> source`) and returns a fresh plain record of own data
  properties, copying each value. A `BindingCollector` records each input's route at the same
  read, and `freeze()` returns a frozen `BindingCapture` whose `boundInputs` is a fresh copy on
  every read.
- `src/nodes/base.ts` holds `Node`. `run(inputs, caps)` is async and resolves to one `Output`,
  `Route` or `Grow`; a kind's `forward(inputs, caps)` may return the result or a promise. The
  capabilities are one `NodeCaps` object (`signal`, `scheduler`, `random`, `runDir`, `bindItem`,
  `steps`, `tools`, and AGENT's fields below). `tools` is the run's `ToolSet`, for a node that declares
  `needsTools` (TOOL, HUMAN_INPUT and AGENT). `signal` is
  optional in the type; the runtime's evaluator (`src/runtime/eval_node.ts`) gives every node its
  own. `scheduler` (the clock and cancellable sleep a node's waits run on; `realScheduler` when
  absent), `random` (a draw in [0, 1) for backoff jitter; `Math.random` when absent) and `runDir`
  (the run's folder, where its files go) go to every node when `evalNode` or `FlowEngine` is given
  them; `runFlow` and `advance` give `runDir` the log's folder, and none without a log.
- `BackwardContext` and `BackwardResult` deep-copy and deep-freeze their values once, in the
  constructor: read them, and copy before you change one (unlike `BindingCapture.boundInputs`,
  which gives a fresh copy per read).
- A compile-owned node copy is shallow (same prototype, own fields copied). Code that changes a
  copy reassigns a field and never mutates a carrier, array, record or map in place. Node classes
  use no ES `#` members (a copy never ran the constructor, so it has no private brand) and no
  instance field that captures `this` (an arrow-function field or a bound callback would stay
  bound to the original); use TS `private` and prototype methods.
- Each kind lives in `src/nodes/<kind>/node.ts` (start, end, case, call, map, loop, code, tool,
  human_input, wait) as a `Node` subclass; import it from that file. There is no per-kind index,
  and, as Python's `yian.nodes`, `src/nodes/index.ts` exports no kind. START's, END's, CASE's,
  CALL's, MAP's, LOOP's, CODE's, TOOL's, HUMAN_INPUT's and WAIT's bodies run.
- AGENT lives in `src/nodes/agent/node.ts`, and its `llm_config` values in
  `src/nodes/agent/llm_values.ts`. The node holds its data, checks and load-time hooks; `forward`
  runs an AGENT with its tools and skills (below), and `backward` (`backward.ts`, Python's
  `nodes/agent/backward.py`) makes one tool-free call of the recorded model through the
  `BackwardOptions` `models` the backward driver passes (consumer `"backward"`). Every AGENT has
  `needsSteps`: it records its model calls and tool results as steps (below), and a call-time
  agent (`callTime` set) also records its config as an `llm_config` step.
  `llmValueTypes` is an array of `[path, type]` entries, not a map keyed by path. AGENT is the node
  the remaining `NodeCaps` fields serve (`skills`, `emitStream`, `emitEvent`, `captureRender`,
  `evaluationTrace`, `openMemoryContext`, and `steering`, a `Steering` inbox given only to an agent
  authored `steerable: true`).
- AGENT `forward` (`node.ts` → `loop.ts` `runAgent` → `structured.ts` `generateStructured` or
  `live.ts` `callModel`) reaches models only through `caps.models`, a `ModelAccess`
  (`defaultModelAccess(env)` in `src/llm_clients/factory.ts`; tests pass one returning a
  `ScriptedModel`). Every call streams through `callWithRetries` and publishes
  `ModelCallStarted`, chat `StreamChunk`s and `ModelCallFinished` per `attempt`. The AGENT's own
  model errors fail with their `provider.*` code, spent `retries:` corrections with
  `agent.output_invalid`. With tools or skills it runs the tool loop: consecutive calls whose tool
  is concurrency-safe (`isConcurrencySafe`, and `load_skill`) run together as a group, and an
  unsafe call runs alone. Changed (TS yian): there is no memory transcript, so the model reads only
  this start's messages and nothing is recorded into a memory (Python's `nodes/agent/memory.py`
  records each call into a session memory's turn and summarizes old turns).
- Changed (TS yian): an AGENT `prompt:` may be a list of messages (see [A list
  prompt][docs/syntax.md#a-list-prompt]). `AgentNode.prompt` is the template text or a
  `PromptItem[]` (`{role: "user" | "assistant", text}` or `{splice: <name>}`); compose reads the
  list and ranges each item at `siteKey("field", "prompt[<i>]")`. A splice takes a built-in
  `Message` (sent with all its parts) or a `{role, text}` record, or a list of them. `forward`
  renders it into `AgentRunContext.messages` (a text prompt is one user message) and
  `promptText`, the last message's text, which recall and evidence read; the rendered list must
  pass `checkModelConversation` before any model is built, and the loop sends
  `[system, ...ctx.messages]`. A chat carries its history as a LOOP field and splices it before
  the new user message:

  ```yaml
  prompt:
    - ${history}          # a !list(Message) input, or !list(Msg) with Msg = {role: Literal(user, assistant), text: str}
    - user: ${message}
  ```
- `src/nodes/agent/messages.ts` holds the built-in `Message` value (`Message`, `MessagePart`,
  `checkMessage`), its conversion to and from the model message (`toModelMessage`,
  `fromModelMessage`, through `llm/messages.ts`' strict `yian.message.v1` reader), and
  the conversation shape every model route can send: `checkModelConversation` over model
  messages (the render's check) and the public `checkConversation` over `Message` values, which
  converts them first. The type itself is
  written in yian's type syntax in `typesys/types.ts` (`builtinType`), which `resolveType` falls
  back to when a flow's registry lacks the name.
- An AGENT writes its steps through `AgentSteps` (`src/nodes/agent/context.ts`), before acting
  on them: each model call, chat or structured, as a `model_call` step once it returns (`callModel`
  in `live.ts`; a failed attempt too, its error scrubbed as the node's failure is; a call a stop cut
  writes nothing), and each tool result as a `tool_result` step as that call finishes, in finish
  order (`runGroup` in `loop.ts`). A reply's media goes to `<run folder>/media/` and the step and
  the next request hold its file form; `callModel` sends the provider the bytes inline
  (`fileToInline`). `request_sha256` (`requestSha256`) hashes the request with the steers left
  out. A tool call's idempotency key is `<run>/<node>/<n>/<index>`, `n` its reply's step.
  Model-retry waits run on `caps.scheduler` and their backoff jitter comes from `caps.random`
  (before, the real clock and `Math.random`). A resumed AGENT replays these steps (`AgentSteps`'
  replay cursor over `NodeSteps.prior`): `callModel` takes each recorded reply back in place of
  its call once its purpose and `request_sha256` match the rebuilt request (else a
  `CapabilityFault`, nothing written), `replayToolCalls` gives each recorded tool result back
  without running the call and closes a call with none as `crashed` (`TOOL_OUTCOME_UNKNOWN`), a
  recorded failure fails the node again, and failed attempts after the last call make the first
  live call wait the recorded `retry_in` and continue the attempt count (`callWithRetries`'
  `prior`). The host inbox (steers, the interrupt) is read only once the replay ends; the steer
  rules sit together in `context.ts`. Each reason the built model gives
  for leaving out replay state (`onReplayDegraded`, passed to `ModelAccess.createModel`) reaches
  `emitEvent` once per invocation, as the in-loop `ReplayDegraded`.
- A CODE `code:` value is a reference (`./file.ts:fn`, also `.js` or `.mjs` and `../` paths,
  resolved against the node's `baseDir`) or an inline TS/JS body. The constructor classifies it
  by shape into `mode` and compiles or imports nothing. Each call runs on a worker thread
  (`src/nodes/code/worker.ts`), reused when idle, that the node terminates when `caps.signal`
  aborts. A failing inline body's `CodeError` stack names its frames after the node's
  `sourceFile` (the load's source label, from `BuildContext.sourceFile`; `<inline-code>` by
  default) and the body's own line; `original` keeps the runtime's `yian-code:<id>` frames.
- A host that bundles yian (esbuild, webpack, `bun build`) must ship the worker as its own entry
  at `nodes/code/worker.js` relative to the bundle file, else every CODE call fails
  `CODE worker failed: Cannot find module ...`. `bun build --compile` takes
  `./src/nodes/code/worker.ts` as a second entry, with both entries under `src/`.
- `code` may also be `{builtin: "identity", field}` (`mode: "builtin"`), the form a tagged value
  entry lowers to.
- WAIT's `until` takes a date or datetime string only: dates are text in TS yian.
- Counts on nodes (LOOP `times`, `maxIters` and `iteration`, END `n`, TOOL `retries`) are
  safe-integer `number`s, not `bigint`. END keeps its mode in public `mode`, `names` and `n`
  fields, and `EndNode.list` replaces Python's `list_`. A CASE branch is a frozen `Case`: an
  origin rewrite rebuilds it.
- `typeToSchema(type)` in `src/nodes/agent/structured.ts` returns the JSON Schema pydantic gives
  for an AGENT `output:`, with ints as `bigint`, or `null` for a bare `str`. It differs where
  pydantic's output is an accident of Python: keys are sorted at every level but a `properties`
  map (pydantic leaves a field named `properties` or `default` unsorted); `properties` and
  `$defs` follow JS key order, so integer-like names come first; a `$defs` key turns every code
  point outside `[A-Za-z0-9_-]` into `_` (pydantic keeps only the text after the last `.`), and
  the `title` stays raw; a clashing key is `<key>__1`, `<key>__2` (pydantic writes a
  module-qualified name); identical schemas share one def; fields starting with `_` are kept
  (pydantic drops them).

### Memory

- Changed (TS yian): no session memory; carry history as a LOOP field. A `session` memory, on a
  flow or an AGENT, is a load error ("session memory was removed; carry the history as a loop
  field"), and so is a `MemoryDecl` of type `session` built in code. An AGENT's model reads only
  its own start's messages: the system prompt with the recalled memory sections, the prompt's
  messages (a list prompt can splice the carried history), then its steers, replies and tool
  results. There is no memory transcript, no `history_summary` model
  call and no `YIAN_SESSION_*` setting.
- `src/memory/contracts.ts` holds the memory identities that compile records (`DeclSite`,
  `MemoryDecl`, `MemorySpec`, `MemoryStack`, `MemoryScope`), the declaration check
  `checkMemoryDeclaration`, the setup and operation failures, recorded snapshots, and the host
  memory objects (`Memory`, `MemorySession`, `isMemory`), the learning result carriers
  (`MemoryMutation`, `LearningFailure`, `PolicyCommitOutcome`, `LearningResult`), and the
  evidence carriers (`EvaluationEvidence`, `MemorySection`, `ModelTextInteraction`,
  `ModelValueInteraction`, `ToolCallInteraction`, `ToolResultInteraction`), and the learning
  inputs (`MemoryEpisodeStatus`, `MemoryEpisode`, `LearnContext`) and the recall carriers
  (`RecallContext`, `MemoryContent`, `MemoryContext`, `MemoryRecallError`). A `MemoryContent` holds
  only notes, and a `MemoryContext` only its sections.
  `src/memory/index.ts` re-exports the public names; the mutation batch check `validateMutations`
  and the feedback snapshot `freezeFeedback` are exported from `memory/contracts.ts` only, as
  Python keeps `_validate_mutations` and `_freeze_feedback` private.
- The learning result carriers are frozen and hold their nested carriers by reference (checked by
  exact class), where Python copies them.
- The evidence carriers are frozen too and hold their nested carriers the same way. Their JSON
  values (bound inputs, a model value, tool arguments) are deep-frozen copies under the value
  number rule: NaN is refused, `±Infinity` kept and `-0` copied as `0`. They have no memory
  nesting limit.
- Episodes and learn contexts are frozen as well. An episode's JSON (inputs, output, failure) and
  a learn context's feedback are deep-frozen copies under the same value number rule, with no
  memory nesting limit; a context keeps its episodes by reference. `freezeFeedback` takes a memo
  shared across calls, so a subtree repeated across feedback items is one frozen copy.
- Changed: memory JSON (a mutation payload, a snapshot envelope's export, learn feedback, a stored
  state) has no depth limit (Python bounds it at the value nesting limit plus 28). It is copied
  with an explicit stack, not recursion, so a deep value can't overflow the call stack.
- A config or snapshot is a deep-frozen copy whose keys are inserted in code-unit order; JS still
  lists integer-like keys (`"9"`, `"10"`) first, in numeric order.
- The declarations are frozen, so `rewriteMemoryDeclOrigins` returns rebuilt declarations and
  leaves its inputs unchanged; pass it one `seen` map for a whole graph.
- An int in a config is a `bigint`; a `number` is a float, so a built-in key refuses `5` and takes
  `5n`.
- A host memory is any object with `openRun({ snapshot })` and `reset()` (`isMemory` checks it, as
  Python's `isinstance(value, Memory)`). Every method of a host memory interface (`Memory`,
  `MemorySession`, `MarkdownStore.read/write`) may return a value or a promise, and yian always
  awaits it, as for tools; the `snapshot` read and the engine's `EvaluationTrace` stay
  synchronous. A run session has no `with` form: the caller calls `close()` in `finally`.
- Changed: `RecallContext.inputs` follows the value number rule (NaN refused, `±Infinity` kept,
  `-0` read as `0`).
- `src/memory/storage/sqlite.ts` holds `SQLiteStorage`, the single-owner SQLite state file, opened
  through `platform.ts`'s `openSqlite`. Changed: the envelope is version 3 (Python's is 2), so a
  Python file fails `storage.version`; `±Infinity` is kept with the codec's `$float` tag; there is
  no fork or thread guard. It also holds `checkStoragePath` and `realStoragePath`, a port of
  Python's non-strict `os.path.realpath`.
- `src/memory/storage/markdown.ts` holds the `MarkdownStore` interface, the async
  `FileMarkdownStore` and the sync `InMemoryMarkdownStore`. `read` returns `[text, version]`; an
  unknown version or non-UTF-8 text throws `RangeError`, a stale base `MarkdownVersionMismatch`,
  and unreachable storage a Node system error. The document, kept versions and write log are
  byte-compatible with Python's, pinned by `conformance/fixtures/markdown_store.json`, which
  Python yian wrote. The `.<name>.lock` file is a SQLite lock that Python's `flock` doesn't see.
- `src/memory/episodic.ts` holds `EpisodicMemory`, with optional SQLite storage opened at the
  first operation. Changed: the constructor takes an options object (`maxEpisodes`,
  `recallLimit`, `storage`, `createParent`) whose limits are integral `number`s or `bigint`s;
  snapshots are deep-frozen; `recall` is async and awaits `rewriteQuery`, which may return a
  promise; recall text and entry ids use yian's canonical JSON (`logJson`); an episode holding
  `±Infinity` fails `learn` as `episodic.invalid_context`; there is no `with` form (call `close()`
  in `finally`) and no fork check.
- `src/memory/reflective.ts` holds `ReflectiveMemory`, which keeps atomic lessons and asks one
  structured model call to propose lesson mutations. Changed: the constructor takes an options
  object (`llmConfig`, `maxItems`, `recallLimit`, `storage`, `createParent`, `env`, `createModel`,
  `onRetry`); `createModel(config)` builds the learner's model (default `chatModelFromConfig`);
  `recall` and `learn` are async, and `rewriteQuery` may be async; snapshot and stored ints are
  `bigint`s; there is no `with` form and no fork check.
- `src/memory/profile.ts` holds `ProfileMemory`, which keeps one markdown document in a
  `MarkdownStore` and asks one plain model call to rewrite it from feedback. Changed: the
  constructor takes the store and an options object (`llmConfig`, `maxChars`, `inputBudgetChars`,
  `env`, `createModel`, `onRetry`); `openRun`, `read`, `write`, `learn` and `commit` are async;
  a Node system error from the store is `profile.store_unavailable` and a `RangeError`
  `profile.store_invalid`; lengths count code points; the snapshot's `version` is `1n`; there is
  no `with` form.
- `src/memory_host.ts` holds `MemoryRegistry` and `buildMemories`, which build the memories a flow
  declares for one host. It sits at the top of `src/`, not under `memory/`, because it reads the
  compiled memory table, and nothing under `memory/` imports it. Changed: `buildMemories` and
  `close()` are async; the registry and its `types` are read-only `Map` subclasses; `storage`,
  `supplied` and `factories` are plain records; a factory gets `toHost(config)` and may be async;
  `env` reaches reflective and profile memory; `models` builds their learners' models (node
  `__memory__`, consumer `memory:<key>`) and `onEvent` receives their `ModelCallRetried` and
  `ReplayDegraded` events; `await using` closes the registry.

### Runtime

- `src/runtime/common.ts` holds `NodeState`, the scheduler states runtime modules share.
- `src/runtime/contracts.ts` holds the forward captures (`BackwardCapture`, `ForwardCapture`) and
  the backward outcomes (`BackwardAttribution`, `AttributedBackwardResult`, `MemoryProposal`,
  `BackwardFailure`, `BackwardOutcome`). They are frozen: JSON values are deep-frozen copies made
  once by the constructor, with no copy on read, and frozen carriers are kept by reference.
- `validateRoute` and `validateSource` check a `ValueRoute` or `SourceRef` at the durable capture
  boundary; `validateRoute(route, { deep: false })` skips the children and snapshot already
  checked, so a tree built bottom-up checks each node once.
- Route and source numbers (sequences, sizes, spans, path indexes) are safe-integer `number`s,
  never `bigint`.
- A `SourceRef` compares by identity: key maps and sets by `sourceKey(ref)`.
- `src/runtime/log.ts` holds the run records, `RecordEnvelope` and its codec (`encodeRecord`,
  `decodeRecord`), the canonical helpers (`canonicalTyped`, `canonicalSeed`,
  `encodePool`, `decodePool`), and the payload helpers later runtime modules share
  (`routePayload`, `routeFromPayload`, `executionScopePayload`, `evidencePayload`). Records and
  envelopes are frozen: a record's JSON values are deep-frozen canonical copies made once by its
  constructor, with no copy on read, and its frozen carriers are kept by reference. The one
  exception is a capture's `evaluatedKeys`, a `Set` that can't be frozen and is read-only by
  contract. Counters
  (`seq`, `numWorkers`, `attemptId`, indexes) are safe-integer `number`s, written as JSON ints. A
  capture or feedback nested deep enough to overflow the stack throws the engine's `RangeError`,
  because `ForwardCapture` and backward feedback freeze recursively.
- `log.ts` also holds the logs. `MemoryRunLog` keeps one run's records in the process;
  `MemoryRunLog.fromRecords` reopens a detached prefix. `JsonlRunLog.create(path, runId)` makes a
  new 0600 file and `JsonlRunLog.resume(path)` reopens one, validating its complete prefix before
  it truncates an incomplete tail. Both logs keep the canonical round trip of every record, and
  `close()` refuses later appends but keeps the records readable.
- Each append stamps the envelope's `at` (UTC, milliseconds, `toISOString` form) from the log's
  `clock` option (epoch ms, default `Date.now`); `runFlow`, `advance` and a result's
  `backward` pass their `scheduler`'s clock, so tests and conformance get virtual times. An
  envelope the engine prepares has `at: null` and encodes without it; the engine's prepare,
  append and replay checks compare records without `at`, and every log reader refuses a line
  without it. Times aren't ordered; `seq` is. Schema 10 (before `at`) is retired, not migrated.
- Changed: a `JsonlRunLog` holds a SQLite exclusive lock (`lockFile` in `platform.ts`) on
  `<path>.lock` until `close()`, where Python flocks the log itself. The kernel frees the lock when
  the process dies, and the lock file is never deleted. A busy lock throws `run log <path> is
  locked by another writer (is the run still active?)` with the `FileLockError` as `cause`. The
  lock sits beside the log's resolved path, so symlinks to a log share it; a hard link doesn't
  (Python's flock covered hard links too), so never run two writers through hard links.
- Each append writes at the known prefix end and fsyncs. A failed write or fsync truncates back to
  the prefix and poisons the log: later appends throw `run log <path> refuses appends after a
  failed append` (cause: the original error), and a failed rollback throws an `AggregateError`. A
  record refused before any write doesn't poison the log.
- For inspection, `iterRunRecords(path, { afterOffset, runId, afterSeq, kinds, payloads })`
  streams the complete lines without the lock and without changing the file, in 64 KiB reads
  (memory bounded by the largest line), yielding `{ seq, offset, end, runId, at, kind, nodeId,
  envelope }`; it stops at an unterminated last line, and with `payloads: false` checks only the
  envelope and node id. A host saves the last entry's `{ end, runId, seq }` and passes them back
  as `afterOffset`, `runId` and `afterSeq`. `RunLogChangedError` (a `RangeError`) says the file is
  shorter than where the reader got to, or, given `runId`/`afterSeq`, doesn't continue the saved
  record (the offset no longer ends a line, or the next record isn't seq `afterSeq + 1` of that
  run): read from 0 again. It may see a line a concurrent writer later rolls back.
  `readRunLog(path)` collects its envelopes; `isRunLogLocked(path)` probes the resolved log's
  `.lock`.
- Changed: a schema the reader can't read throws `RunLogSchemaError` (a `RangeError`, `schema`
  the number named), where Python raises `ValueError`; the decoders keep its class when they add
  the line to the message.
- `src/runtime/run_index.ts` (TS-only) holds the run index, `<runs>/index.sqlite`: `updateRunIndex`
  folds each run folder's log into one row from the byte offset the row reached
  (`iterRunRecords`), keeping the run scan (`RunScan`: plain JSON data, folded one record at a
  time by `foldRunScan`, read by `scanStatus`) in a `scan` column. `inspectRun`'s record scan is a
  loop over the same `foldRunScan`, so both read a status by one set of rules. `readRunIndex` puts
  `running` on top while a writer holds a run's lock. One `BEGIN IMMEDIATE` transaction with a busy
  timeout serializes builders. Never imported by the reducer or the engine.
- `src/runtime/trace.ts` holds the retained trace of a learnable run (`RetainedTrace`: accepted
  sources, occurrences, current writers, boundaries and the terminal). `prepareTrace` checks one
  record against the trace and returns a frozen `TraceDelta`; the caller applies it
  (`delta.apply()`) only after the log accepts the record, so a refused append leaves the trace
  unchanged. Sources are keyed by `sourceKey`. Typed values (`SourceEntry.valueJson`,
  `TraceTerminal.outputJson`, the system values) are held as canonical log text (`logJson` of the
  typed wire) and decoded on each read, so every read gets a fresh value and equality is text
  equality. An occurrence holds its envelope's round trip `decodeRecord(encodeRecord(envelope))`.
- `src/runtime/learning.ts` holds the pure backward transfer (`seedBackward`,
  `contextualTransfers`, `projectRootBackward`; `ContextCache` materializes each route context
  once per pass), the attributed validation `validateAttributedBackward`, the attempt state and
  its reducer (`BackwardAttempt`, `RuntimeBackwardState`, `reduceBackwardRecord`), the owner
  contexts (`projectOwnerContexts`, `OwnerContextError`), the episode outcome (`episodeOutcome`,
  `failureJson`) and the commit ids (`memoryCommitId`). It invokes no callbacks or backends. The
  integers in the JSON it builds (sequences, indexes, `parents`) are `bigint`s, since a `number`
  is a float there.
- Changed: `memoryCommitId` digests the `logJson` text of its input, not Python's sorted compact
  `json.dumps`; see `docs/api.md` for where the two differ.
- `src/runtime/state.ts` holds the run state (`RunState`) and its reducer. Each `prepare*` function
  (`prepareRunStarted`, `prepareInitialCommit`, `prepareNodeStarted`, `prepareNodeStep`,
  `preparePoolCommit`, `prepareNodeSucceeded`, `prepareNodeRouted`, `prepareNodeFailed`,
  `prepareExpanded`, `prepareRunResumed`, `prepareRunStopped`, `prepareRecord`) validates one
  transition without mutating the state and returns a frozen `PreparedTransition`: the canonical record, the run id and sequence it expects,
  and a frozen plan. `commitPrepared(log, state, prepared)` appends the record, checks what the log
  returned, then applies the plan, with no `await` in between, so nothing interleaves between the
  append and the apply. `commitInitialPool` commits the run start's commits (the system values in
key order, then the START inputs) after `RunStarted`; a node's `pool_commit` waits in
`pendingCommits` until its `node_succeeded` names it, and a resume drops it. A derived commit
(`isDerivedCommit`: a `derived` node other than the top-level START and END) holds `from` and no
value; its success rebuilds the value (`rebuildCommitValue`: `rebind` and the node's pure
`forward`, from `runtime/eval_node.ts`'s `rebuildDerived`), reading each named value from the pool
through `poolResolver`, which refuses a seq that isn't its name's producer. A spawner's expansion
first writes one commit per START of its fragment (`expansionCommits`, `prepareExpansionCommit`),
each under the child START's id with the spawner's `from`; `node_expanded` admits them (each gets
the value its START binds over its constant wiring) and the child START's success names its commit
instead of writing one. Replay rebuilds the seed from that `from` before the prune applies
(`rebuildExpansion`). `producers` maps each
pool name (a store key, or `system.<name>`) to the seq of the accepted `pool_commit` that last wrote
it: set when a commit is applied, deleted with its pool entry, rebuilt by replay, so it never grows
past the pool. A `derived` node (START, END, LOOP, CALL, MAP) evaluates with binding capture in
every run; without a retained trace its sources come from `resolveFromPool(path, pool, producers,
runId)`, and the route stays in memory with the outcome and is dropped after the commit. A node step
carries `n`, the node's step count plus one (steps are kept across a resume until the node
settles), and a step with any other `n` is refused live and on replay. Frontier order:
START is ready after the START inputs' commit; a success walks its
  target's outgoing edges in order and appends each successor when its last edge from the target
  settles and its join is ready; a route settles its edges depth first in stored order, skips the
  heads whose joins are dead (following each skipped node's output redirect) and appends the heads
  that become ready in that order; a resume puts the retried nodes first, in dispatch order, ahead
  of what was ready; a stop changes only the sequence and `lastStopSequence`. `crashedAttempts`
  counts, per node, the attempts a resume retried that no stop covered, using `lastStopSequence`
  (the sequence of the last stop since the last resume). The first route whose skips reach END
  derives the `graph.terminal_unreachable` failure (naming the first required END input it skipped);
  it ends the run once nothing is ready or in flight, unless a node failure ends it first; a node
  failure ends the run at once. A bad output, an unknown handle or a bad commit target from a node
  throws `NodeDataError` (`original` holds the underlying error), which fails the node, not the run.
  A growth (`prepareExpanded`) copies the spawner with `copyGraph`, calls the injected builder with
  the canonical seed, copies the fragment it returns, validates it against the live graph (ids,
  wiring, contracts, edges, reachability, redirects, the node budget, depth) and decides which
  requested or earlier-deferred prune ids retire now, with their grown subtrees, and which stay live
  and deferred while they, their grown subtrees or the work that reads them are unsettled; the
  commit then installs the fragment and retires the pruned nodes in place, after the append. A bad
  seed, fragment or prune from the node throws `GrowthRejected` (`original` holds the underlying
  error), which fails the node; a builder or host fault stays raw. Replaying a `NodeExpandedRecord`
  rebuilds the fragment and requires the recorded digest, and throws the underlying error, never
  `GrowthRejected`. `replayRecords(state, envelopes, {growBuilder, onPrepared})` is the recovery fold
(`onPrepared` sees each record prepared, before it applies): it folds
  a complete historical prefix through the same `prepare*` and `commitPrepared` path against a
  read-only `HistoricalLogCursor` that matches each record byte for byte instead of writing it, so
  the replayed state, crashed counts included, is exactly the live one; a corrupt record throws
  the underlying `RangeError` or `TypeError`, never `NodeDataError`. After the terminal, the
  backward and memory-commit records reduce through `reduceBackwardRecord`
  (`sameRuntimeBackwardState` compares the prepared and recomputed attempt state at commit).
- `src/runtime/eval_node.ts` holds the evaluator. `evalNode(node, flow, pool, options)` binds the
  node's inputs from the pool through the flow's wiring (or a per-run override), builds the
  capabilities the node's traits ask for, runs it, and resolves to an `EvaluatedOutcome`: one
  `Output`, `Route`, `Grow` or `EvaluationFailure` (a `Failure` classified by the stage that threw:
  binding, assertion, body, result), with the evidence and captures a learnable run asks for. A stop
  (the node's aborted signal), a capability's host fault (the first one wins, even if the node caught
  it), an unexpected binder fault and malformed setup reject instead. `MemoryContextOpener` is the
  memory runtime's admission callback, `(scope, request, { signal }) => Promise<MemoryContext>`.
- `src/runtime/engine.ts` holds `FlowEngine`, the only writer of a `RunState` while it runs. `new
  FlowEngine(state, log, { growBuilder, numWorkers, onEvent, onStream, memoryContextOpener, tools,
  skills, steering, live })` borrows a state and log already initialized to the same sequence; `run()`
  starts the head of `ready`, evaluates it, commits the outcome (a refused output, route or growth
  becomes a `contract.*` failure) and publishes each record's observers to `onEvent`, until the state
  has a result, and resolves to those observers. A node failure is a result; any other error aborts
  the run signal and rejects `run()`, as does a promise a callback returns that rejects before the
  result. A `needsSteps` node's `record` appends and applies each step before it returns. Serial mode
  (`numWorkers: 0`) awaits each node inline; pooled mode keeps at most `numWorkers` evaluations in
  flight and commits their outcomes in settle order from a FIFO completion queue (a worker never
  rejects); a `RunFailed` is published before the remaining workers are awaited. The `signal` option
  is the host's stop: each node gets a child signal as `caps.signal`; a host stop or a failing host
  callback records one `RunStoppedRecord` when requested, then nothing starts, later outcomes are
  discarded, every running node is awaited, and `run()` rejects with the initiating error. Each
  evaluation runs in a fresh `AsyncResource`, so `AsyncLocalStorage` stores set inside a node stay in
  it. `resumedAttemptEvents(state, record)` builds the `NodeAttemptResumed` events a resume publishes
  after `RunResumed`. With `live`, each model call's `ModelCallStarted`, text chunks and
  `ModelCallFinished` also go to the channel, and a node's calls leave it when the node settles.
- `src/runtime/live.ts` holds `LiveChannel` (see [Following a run live](#following-a-run-live)):
  the engine-only `start`, `chunk`, `end` and `settle` update the calls in flight, one per
  `(node, n)`, then send the frame to each subscriber, dropping and reporting one that throws.
  `wrap(log)` gives the run boundary a log whose appends also reach the channel (a `node_step`
  drops its call), its `last_seq` starting at the records the log already holds.
- `src/runtime/memory_runtime.ts` holds `MemoryRuntime`: `open` validates the table, origins,
  registry and snapshots, opens one session per key in key order and captures the snapshots;
  `openContext(scope, recall, { signal })` recalls each scope key outer to local, keeps non-empty
  sections, checks the signal, and resolves to the context; `proposeLearning`/`commitLearning`;
  `openContext`, `proposeLearning` and `commitLearning` refuse a closed runtime ("memory runtime is
  closed"); `close()` closes in reverse key order once, noting close failures on a pending error
  (`notes`), and a later call waits for the first and never throws.
- Changed: a backend's `MemoryOperationError` is a learning failure whatever its `cause`: the cause
  chain isn't searched for host controls (TS has no such error classes; stops are signals), and any
  other backend error escapes unchanged.
- Changed: the registry is a `Map` of key to backend (Python accepts any mapping).
- Changed: a session's `commit` may resolve to `undefined` or `null`.
- Changed: a restored snapshot must equal the recorded one by value and type (`1` and `1.0` differ,
  where Python's `==` treats them equal).
- `src/runtime/backward.ts` holds
  `computeBackward(trace, rootFeedback, { cache, models, onEvent })`: one reverse walk over the
  accepted occurrences that gathers each occurrence's messages, calls its retained backward handler
  once with `{ models, onEvent }` (awaited; it may return a promise), validates the whole result
  before routing input feedback, and returns the root projection with every occurrence's attributed
  result (unreached ones uninvoked); a handler's throw is a `backward.execute`
  `BackwardComputeError` and a refused result `backward.result`, while trace faults escape
  unattributed. `driveBackward(state, log, feedback, { openMemory, models, onEvent })` runs one new
  explicit attempt when a host calls it: it prepares its first record (an unfinished
  `BackwardFinished` for an unfinished earlier attempt, else `BackwardStarted`) as a gate, opens the
  memory runtime (only when memories are enabled; the opener may be async) and checks it against the
  recorded table and snapshots, then commits that record (and `BackwardStarted` after an unfinished
  one), computes, projects owner contexts, stages every key's proposal, commits `BackwardReady`,
  requests and observes each key's commit in table order, and commits `BackwardFinished`; each
  record is prepared and committed with no await in between, and the runtime is always closed,
  keeping the initiating error.
- Changed: a handler's throw is `backward.execute` whatever it is (a `CapabilityFault` included; no
  host-control unwrapping).
- Changed: a drive finishes only its own attempt: the attempt id is captured once, so a drive
  overtaken by another while it awaits (its opener, a handler, a learner or a commit) is refused
  after that await and appends nothing further (Python re-reads the current attempt, which is the
  same while synchronous).
- `src/run.ts` (`yian/run`) holds `runFlow`, `advance`, `loadRecordedFlow`, `inspectRun` and
  `RunResult`. `runFlow` takes the inputs and `llmConfig` through host intake, checks the run's
  tools, skills and model configs, opens its memory sessions and MCP servers, creates the log,
  recording the flow files, granted tool specs and skills, and drives a `FlowEngine` to a result,
  then closes what it opened. `advance` locks an existing log, compiles the flow it recorded,
  replays it on a private copy and drives the rest (a finished log returns its result and appends
  nothing; an unfinished one first checks the granted tool specs and skills). `inspectRun` reads a log's status, in-flight nodes, crashed counts,
  run id, schema, flow fingerprint, learnable flag, last seq, start time (`startedAt`), a finished
  run's end time (`endedAt`, its last record's `at`) and a failed run's failure without taking
  its lock; a shrink under the read while a writer holds the lock (a rolled-back append)
  reads the log once more. `staticNodes(flow)` lists every static node path of a flow and its
  child flows with its node, so a host can name a logged node's kind; `staticPathOf(nodeId)`
  (`nodes/ids.ts`, beside `mapCallsite`) maps a grown node's id onto that path by dropping each
  segment's MAP element `#<i>` and LOOP iteration `~<k>` suffixes. `run.ts` re-exports
  `staticPathOf`, `iterRunRecords`, `RunLogChangedError` and `RunLogSchemaError` for hosts, and,
  for hosts that read run-log values, `TypedValue`, `FileRef`, `logJson`, `parseJson` and
  `JSONDecodeError`.
- Value equality: each carrier class has one `same*` helper in the module that owns it
  (`sameFailure` in `diagnostics.ts`, `sameFeedback` and `sameSource` in `expr/lineage.ts`,
  `sameBackwardResult` in `nodes/base.ts`, the backward carriers' in `runtime/contracts.ts`, the
  memory carriers' in `memory/contracts.ts`, `sameSourceEntry` in `runtime/trace.ts`,
  `sameRuntimeBackwardState` in `runtime/learning.ts`). Every `same*` takes non-null values; compare
  a nullable pair with `sameOptional(a, b, same)` from `nodes/contracts.ts`. `memory/contracts.ts`
  writes its null checks inline, since it imports only `diagnostics` and `defaults`. A
  `BackwardFailure`'s cause compares by value, never `===`.

### Compile

- `src/compile/model.ts` holds the graph (`Flow`, `CompiledFlow`, `Edge`, `FlowOutput`), the
  growth-metadata check and `copyGraph`; `origins.ts` the origin traversal and rewrites;
  `execution_layout.ts` the static execution scopes; `validation.ts` the cycle, path and type
  checks; `manifest.ts` `flowManifest`, `flowFingerprint` (SHA-256 hex over the manifest's
  `canonicalJson`, RFC 8785) and `expansionDigest`. `src/compile/index.ts` re-exports Python's
  `__all__` only.
- `src/compile/memory_layout.ts` resolves the memory key table and every reachable node's
  `MemoryScope` (`resolveMemoryLayout`) and stamps them all or nothing (`applyMemoryLayout`).
- `src/compile/expand.ts` holds the pure growth builders: `cloneChild` and `buildGrow`, which
  turn a CALL, MAP or LOOP seed into a `Flow` fragment.
- `src/compile/llm_cascade.ts` holds the `llm_config` cascade: the load-time check for a value a
  child flow's agent would inherit (`findInheritedLlmValue`), run-start resolution
  (`resolveLlmCascade`, `resolveRunLlm`), replay of a recorded map (`applyEffectiveLlmMap`), the
  call-time helpers (`knownCallTimeProviders`, `callTimeSelection`, `inFlightLlmConfigs`, ...)
  and `bindHostCredentials`. Each callsite's child flow is a fresh `copyGraph` copy, and a node's
  config is replaced with a new record (`copyValue` for deep copies), never changed in place, since
  a copied flow's nodes share their records with the original. The maps that go into or come from
  the run log (effective, call-time, in-flight) are records; the label, subject, provider and
  base-URL lookups are `Map`s.
- A `Flow`'s `nodes` and `wiring` are `Map`s: they keep insertion order, so node and wiring order
  is Python's (a record would list index-like keys first and could collide with `__proto__`).
- `copyGraph` is the one graph copier (Python's `copy.deepcopy` of a flow or node). Each call uses
  a fresh memo, so sharing and cycles carry over; a node copy is shallow; a flow reached through a
  node field is copied once; each flow copy gets fresh `nodes`, `edges` and `wiring` containers
  and, for a `CompiledFlow`, a rebuilt adjacency index.
  Carriers and payloads stay shared, so change a copy by reassigning a field or, on a
  `CompiledFlow`, through `addSubgraph`/`removeSubgraph`.
- `rewriteFlowOrigins` and `rebaseFlowOrigins` return a rewritten copy and never touch the input;
  a payload stays shared, except that the records and lists around a nested binding are rebuilt.
- Every manifest value is a typed wrapper: an int `{"type": "int", "value": "<digits>"}`, a float
  its IEEE-754 big-endian hex (`inf` is `7ff0000000000000`, `-0` is written as `0`), a mapping's
  entries sorted by key, a `FileRef` its `{uri, mime, name}` mapping. NaN is refused (V8 and
  JavaScriptCore write its bits differently). Counts (TOOL and AGENT `retries`, LOOP `iteration`
  and `maxIters`, END `n`, indexes) are raw JSON ints and must be safe integers; AGENT `retries` is
  written even at 0, TOOL's only when non-zero; LOOP `times` is an int wrapper.
- A node's or flow's fields are its own enumerable keys; a field the manifest doesn't know throws.
  `kind` is emitted; trait fields are ignored: the kind fixes them.
- A CODE node contributes `{"inline": body}`, `{"ref": "./x.ts:fn"}` or
  `{"builtin": "identity", "field": "x"}`; `baseDir`, `sourceFile`, `mode` and its origins aren't
  identity. An AGENT contributes its authored fields (`own_llm_config`, not `llmConfig`);
  `llmConfig`, `llmConfigLabel`, `wiresLlmConfig`, `llmValueTypes`, `callTime` and
  `hostCredentials` aren't identity.
- In `typesCompatible` and `typeRelation`, an open numeric end is `null`.
- Compare positions and scopes with `sameExecutionPosition`/`sameExecutionScope`, output contracts
  with `sameOutputContract`, and arrays with `sameItems` (`src/nodes/contracts.ts`).

### CLI

- `src/cli/terminal.ts` holds `Terminal`, the host terminal a run prompts on: `askUser` (the
  run's ask handler), `askText`, `askChoice`, `promptInputs` (the missing flow inputs) and
  `close`. Prompts take turns in one FIFO and draw on stderr; `onStop` hears Ctrl-C or the end of
  input, which close the terminal, and `onActive` hears a turn take and give back the terminal.
  A call the closed terminal ends throws `TerminalStopped`, or the signal's reason once it
  aborted (`askUser`), or returns null.
- It reads keys with raw-mode `node:readline` and has no dependency: it imports only `node:`
  modules and yian's own.
- `src/cli/index.ts` is the `yian` entry: one command table (specs as data, handlers imported
  when their command runs), the `YIAN_<NAME>` defaults check, usage errors (exit 2, nothing on
  stdout), and an exit that waits for the last write (Bun cuts piped output short otherwise).
- `src/cli/utils.ts` holds the argument parser (`parseCommand` over `node:util` `parseArgs`, with
  Click's path checks and messages), `CliIo` (the streams and `Env` a command gets; tests pass
  fakes), `CommandEnv` (a prompted key is recorded there, never in `process.env`), `printJson`
  (one `yian.cli.v1` document; ints must be `bigint`, a `number` is written as a float), and the
  key prompts `ensureEnvKey`, `ensureApiKey`, `confirmOllamaEndpoint`.
- `src/cli/run.ts` holds `yian run`, `yian check`, `yian skills check` and `yian status`, wired to
  `loadFlow`, `runFlow` (fresh), `loadRecordedFlow` and `advance` (`--resume`, which takes no
  flow file and runs the recorded flow, with the recorded-key checks),
  `loadSkillDir`, `inspectRun`, and `updateRunIndex` and `readRunIndex` (`yian status` with no
  log; `yian run` updates the index of `~/.yian/runs` when a run there ends); the diagnostic boxes, the banner, progress on stderr and
  `stopOnSignals` (SIGINT, SIGTERM and SIGHUP stop a run; exit 130).
- A stop aborts the run's controller with a `HostStop` (`utils.ts`) naming the signal; Ctrl-C at
  a prompt is a `HostStop("SIGINT")`.
- A required provider key that is missing and that no prompt supplied throws `MissingKeyError`
  (`utils.ts`); `run` reports it as `run.setup` before the log exists.
- `run --json` runs in a child of the same runtime whose fd 1 is the parent's stderr and whose
  document goes out on fd 3; the parent forwards a stop as SIGUSR2 (`index.ts`).
- `src/cli/mcp_config.ts` reads `mcpServers` files into unopened MCP servers (`userMcpConfig`,
  `loadMcpConfig`). `src/cli/memory.ts` builds the memory registry from `--memory-root` and
  `--memory-storage` (`parseMemoryStorage`, `buildCliMemories`).
- `src/cli/auth.ts` holds `yian auth` (list, setup through `Terminal` prompts, `--default`,
  `--remove`, whole-file removal, device sign-in as JSON lines), `yian models`, `yian providers`
  and `yian config check`. `auth` and `models` take an optional `AuthDeps` (the OpenAI service, the
  listing, the model factory, the terminal, the browser opener, the signal source and the
  scheduler), which tests replace with fakes. Ctrl-C stops `yian auth` (SIGTERM too for the JSON
  device form); deadlines run on the scheduler, never on `AbortSignal.timeout`.
- The conformance runner runs `cli` cases by spawning `src/cli/index.ts` (built: `dist/src/cli/
  index.js`) with a preloaded hook that routes `fetch` to a stub server.

### Commands

Run from `ts/`:

```sh
bun install --frozen-lockfile
bun run typecheck
bun run test                  # or one file: bun test ./tests/engine/layer_direction.test.ts
bun run test:node             # the same tests on Node, with the `node` on PATH
bun run build
YIAN_TEST_NODE=<node> bun test ./tests/engine/platform.test.ts   # after a build: the Node checks
bun run build:bin             # the four binaries into bin/; YIAN_BIN_TARGETS="darwin-arm64" for a subset
YIAN_TEST_BINARY=<absolute path to a binary> bun test tests/packaging.test.ts   # also run that binary
YIAN_UPDATE_LICENSES=1 bun test tests/packaging.test.ts   # rewrite src/cli/licenses.ts
```

`bun test` reads the preload `tests/conftest.ts` from `bunfig.toml`, which applies only when
`bun test` starts in `ts/`. `test:node` needs Node >= 22.18: it loads `tests/node_runner.ts`, which
runs the `bun:test` files on `node:test`, then the same preload.

The preload gives each test a fresh dir inside one `yian-test-*` dir per process under `$TMPDIR`,
removed when the run ends (on Bun by the preload's `afterAll`, on Node at exit). A killed run, or a
Bun run stopped by `--bail`, leaves that one dir.

`tests/packaging.test.ts` checks `src/cli/licenses.ts`, the text `yian --licenses` prints, against
`NOTICE`, Bun's license, the license files of the packages the binaries bundle (read from
`node_modules`) and those of the packages copied into their `dist/`. After a dependency change,
rewrite it with `YIAN_UPDATE_LICENSES=1`. Bun's section and the copied packages' sections are kept
from the current file unless `YIAN_BUN_LICENSE=<Bun's LICENSE.md>` (run under that Bun) or
`YIAN_VENDORED_LICENSES=<dir>` (each npm package extracted as `<dir>/<any>/package/`) is set; set
them when a Bun upgrade or a newly copied package changes those sections.

### Conformance

`tests/conformance.ts` runs the cases in `../conformance/cases/` (see `../conformance/README.md`).
Run from `ts/`:

```sh
bun tests/conformance.ts                                  # every case, on Bun
bun tests/conformance.ts --group defaults --kind defaults # a selection
bun run build && node dist/tests/conformance.js           # every case, on Node
```

Flags, each repeatable (values of one flag are alternatives; different flags all apply):
`--group G`, `--kind K`, `--case <group>/<name>`, `--rules R` (cases whose `rules` hold R),
`--diagnostic-code C` (`load` cases expecting a diagnostic with code C). `--kind`, `--rules` and
`--diagnostic-code` never hide a case file that can't be read; `--group` and `--case`, read from
its path, still select or skip it. `--crash-sweep` also gives each eligible `run` case a crash
sweep: the run is cut after each record but the last, resumed, and checked. `--write-logs` (for
development; it needs `--case`) writes each selected `run`, `resume` or `backward` case's golden
log, `<name>.log.jsonl` beside the case, which later runs compare byte for byte.

The runner runs every kind. A `run`, `resume` or `backward` case's `host` may hold `tools`,
`human`, `jitter`, `models`, `memory`, `steering` and `stop`, and its `expect` may hold `memory`; a
case using `host.skills` fails before it is set up.
