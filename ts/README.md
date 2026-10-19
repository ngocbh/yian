# yian (TypeScript)

The TypeScript implementation of yian, being ported under #736. It is not usable yet; the
Python package in `src/yian/` stays the reference until the two reach parity.

- npm name: `@ngocbh/yian` (`yian` is taken on npm).
- Runtimes: Node >= 22.18 and Bun >= 1.4.

## Layout

- `src/<pkg>/<mod>.ts` mirrors `src/yian/<pkg>/<mod>.py`.
- `tests/<area>/<x>.test.ts` mirrors `tests/<area>/test_<x>.py`.
- Relative imports use `.ts` specifiers; `tsc` rewrites them to `.js` in `dist/`.
- No `Bun.*` or `bun:*` outside `src/platform.ts`.

## Conventions

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
  and sends `USER_AGENT`; every request goes through it). Never open a lock file
  any other way: on Node, closing any descriptor of a locked file drops the process's lock. Code
  that waits takes an injected `Scheduler`, so tests and conformance run on `virtualScheduler`.
  `stripTypes` strips a CODE body's types with amaro on both runtimes, keeping every line
  number, and returns text: the caller builds the function.
- Host values enter through `fromHost`; values reaching `buildValueAs` are yian values.
- A value becomes text with `writeAsText` (the text form) or `writeAsJson` (`render_as_json`),
  never `String()` or `JSON.stringify`; messages show values with `valueRepr`.
- A `float` slot given an int stores kind `int`, and a `list[float]` item may be an int
  (`bigint`): code reading a float slot handles both (#535, `int-in-float`).

## YAML

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

## Skills

- `src/skills/` is a leaf: it holds `Skill`, the `load_skill` tool and the system-message
  listing, and imports only `defaults`, `diagnostics`, `typesys` and `llm`. Its `index.ts`
  never re-exports the Compose loaders.
- `src/compose/skills.ts` (`loadSkills`, `loadSkillDir`, `defaultSkillRoots`) reads `SKILL.md`
  directories and parses their frontmatter with `parseYamlLiteral`.
- Reads are synchronous. Skill files are read as strict UTF-8, with CRLF and lone CR read as LF.

## Tools

- `src/tools/` imports no engine or front-end layer (`nodes`, `compile`, `compose`, `runtime`,
  `run`, `cli`); `index.ts` is its public face.
- A host tool is a `FunctionTool` (`execute(args, ctx)`, sync or async); a run's `ask_user`
  handler is wrapped into one, so a `ToolSet` holds only `FunctionTool`s.
- Tool errors, `FunctionTool`, `McpServer` and the MCP errors are recognized by `Symbol.for`
  brands, never by `instanceof`, so a tool built against another copy of yian works.
  `toolErrorKinds`, the tool error guards (`isPermanentToolError`, ...) and `isMcpError` read
  the error brands; the registry checks the `FunctionTool` and `McpServer` brands.
- Arguments enter through `fromHost` and reach `execute` through `toHost`: fresh arrays and
  records, ints within ±(2^53−1) as `number`, larger ones as `bigint`.
- `invoke` returns the tool's result raw, awaited, with `undefined` read as `null`; the callers
  run value intake on it at their own boundaries (`contract.output`).
- A `FunctionTool`'s `parametersJsonSchema` must be JSON within the nesting limit, since the
  model sees it; tools still receive and may return a real `Infinity`.

## LLM layer

- `src/llm/` is a leaf: it imports only `defaults`, `platform` and `typesys/json` (the layer
  ratchet checks this).
- Messages are `yian.message.v1` (`../conformance/schema/message.schema.json`). Provider payloads
  are opaque JSON, holding the versioned replay records of design 7.5 (`replayRecord`,
  `readReplayRecord` in `src/llm/messages.ts`); a record that doesn't match the request is
  left out, never an error.
- Every model call streams. A call ends with exactly one `Finish`, its last event, or a thrown
  `ProviderError`, whose `kind` is a stable error kind (failure codes are assigned above
  `llm/`) and whose `retryable` is set when it is built. Any other error is a defect and is
  never retried.
- Every request goes through `safeFetch` in `src/platform.ts` (no redirects, `User-Agent`); code
  that needs a fake or a cassette injects a `Send`, never a global `fetch`.
- An adapter is a `WireProtocol` (`src/llm/executor.ts`), one file per wire format in
  `src/llm/protocols/`: it builds the body from a `Request` and turns SSE or NDJSON frames into
  events. `RouteModel` sends it over a `Route` (`src/llm/route.ts`, with an auth object from
  `src/llm/auth.ts`), one attempt per call, under yian's first-response and idle timers
  (`src/llm/transport.ts`). An adapter's `ProviderError` holds the provider's text uncapped;
  `RouteModel` scrubs it of the route's credentials, then caps it. `shortRequest` is for model
  listing and sign-in.
- Retries happen above the call: `callWithRetries` retries a retryable `ProviderError`, before or
  after partial output, up to `MODEL_RETRIES` times, calling `onRetry` before each wait.
- `ScriptedModel` in `src/llm/testing.ts` scripts model replies for tests. Cassettes
  (`yian.cassette.v1`) replay HTTP exchanges as a `Send` (`CassetteReplay`); `CassetteRecorder`
  records them with every credential redacted by value, and refuses to save one that still holds
  a credential. Record only when `recordMode(processEnv)` says so (`RECORD=1`, never under `CI`).

## LLM clients

- `src/llm_clients/` holds `LLMConfig`, the provider block decoders (one field-spec table per
  API, the key walk, the value pass), the connection-setting checks and the key-variable table.
- It reads no environment variable itself; its limits come from `current()`.
- Values are yian values (ints are `bigint`), so a host `llm_config` goes through `fromHost` first.
- `LLMConfig` holds the key, the header values and the blocks in `#private` fields. The key and
  headers are never dumped; the blocks are read through `blockFor` and `dump`.
- Lookups by a user key use `Object.hasOwn`. `PROVIDER_BLOCKS` and the schema tables have no
  prototype, so `in` and indexing by a user key are safe too.
- Validation errors are `ValidationError` with pydantic-shaped `issues` (`loc`, `type`, `msg`);
  for a `value_error`, `error` holds the validator's own text.

## Nodes

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
  capabilities are one `NodeCaps` object (`signal`, `bindItem`, `steps`). `signal` is optional
  in the type; the runtime's evaluator gives every node its own (the evaluator is not ported
  yet).
- `BackwardContext` and `BackwardResult` deep-copy and deep-freeze their values once, in the
  constructor: read them, and copy before you change one (unlike `BindingCapture.boundInputs`,
  which gives a fresh copy per read).
- A compile-owned node copy is shallow (same prototype, own fields copied). Code that changes a
  copy reassigns a field and never mutates a carrier, array, record or map in place. Node classes
  use no ES `#` members (a copy never ran the constructor, so it has no private brand) and no
  instance field that captures `this` (an arrow-function field or a bound callback would stay
  bound to the original); use TS `private` and prototype methods.

## Memory

- `src/memory/contracts.ts` holds the memory identities that compile records (`DeclSite`,
  `MemoryDecl`, `MemorySpec`, `MemoryStack`, `MemoryScope`), the declaration check
  `checkMemoryDeclaration`, the setup and operation failures, recorded snapshots, and the host
  memory objects (`Memory`, `MemorySession`, `isMemory`). `src/memory/index.ts` re-exports the
  public names.
- A config or snapshot is a deep-frozen copy whose keys are inserted in code-unit order; JS still
  lists integer-like keys (`"9"`, `"10"`) first, in numeric order.
- The declarations are frozen, so `rewriteMemoryDeclOrigins` returns rebuilt declarations and
  leaves its inputs unchanged; pass it one `seen` map for a whole graph.
- An int in a config is a `bigint`; a `number` is a float, so a built-in key refuses `5` and takes
  `5n`.
- A host memory is any object with `openRun({ snapshot })` and `reset()` (`isMemory` checks it, as
  Python's `isinstance(value, Memory)`); both are synchronous, and a session has no `with` form:
  the caller calls `close()` in `finally`.
- The recall, learning evidence, learning and session conversation contracts, the session's
  recall, learn and commit methods, and the built-in backends are not ported yet.

## Commands

Run from `ts/`:

```sh
bun install --frozen-lockfile
bun run typecheck
bun run test                  # or one file: bun test ./tests/engine/layer_direction.test.ts
bun run build
YIAN_TEST_NODE=<node> bun test ./tests/engine/platform.test.ts   # after a build: the Node checks
```

## Conformance

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
its path, still select or skip it.
