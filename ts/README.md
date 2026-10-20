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

## Compose

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
  reader compiles nothing at load yet: checking an inline body's syntax and `return`, and a
  reference's file, waits for an awaited load step. A CODE node's `baseDir`, the directory its
  reference resolves against, comes from `BuildContext.baseDir` (`null` by default). WAIT keeps an
  `until:` date or datetime as the text written, after checking it parses.

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
- `tool()` builds a `FunctionTool` from Standard Schema v1 validators (with the Standard JSON
  Schema interface) or raw JSON Schemas. yian depends only on those interfaces; zod, valibot and
  ArkType are dev dependencies for the tests.
- `validate` checks a call (JSON Schema, then the validator once) and returns a `CheckedCall`;
  `invoke(call)` runs it without checking the arguments again and `isConcurrencySafe(call)` reads
  it; both see what the validator returned, or the checked arguments when there is no validator.
  `invoke(toolId, args)` validates first.
- `invoke` returns the tool's result raw, awaited, with `undefined` read as `null`, unless the
  tool declares an output: then it checks the result and returns the checked value in host form.
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

## LLM clients

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
- `index.ts`: the package's exports, as Python's `__init__.py`.
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
  `remove` and `setDefault` are async and hold `<path>.lock` through `waitForLock`; `credentials`
  is async and takes the lock only to refresh. Node and Bun have no `dir_fd`, so private-path
  checks use `lstat`, an `O_NOFOLLOW` open and `fstat`. Parallel `credentials` calls in one
  process that refresh the same login share one refresh (success or failure); the file lock
  still serializes processes.
- `auth/http.ts`: `request`, Python's `HttpResponse` over `shortRequest`: one total timeout,
  redirects never followed (a 3xx comes back with `null` data), the body capped.
- `auth/openai.ts`: `OpenAIAuth`, browser and device sign-in, refresh and the model catalogs.
  It takes a `Scheduler` and an `AbortSignal` instead of Python's clock and stop callbacks; a
  login's deadline is one timer that aborts its signal. The browser callback listener is
  `node:net` on `127.0.0.1:1455`.
- `auth/index.ts`: the `auth` package's exports, as Python's `auth/__init__.py`.
- Values are yian values (ints are `bigint`), so a host `llm_config` goes through `fromHost` first.
- `LLMConfig` holds the key, the header values and the blocks in `#private` fields. The key and
  headers are never dumped; the blocks are read through `blockFor` and `dump`.
- Lookups by a user key use `Object.hasOwn`. `PROVIDER_BLOCKS` and the schema tables have no
  prototype, so an inherited name such as `toString` or `__proto__` is never an entry.
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
  capabilities are one `NodeCaps` object (`signal`, `bindItem`, `steps`, `tools`, and AGENT's
  fields below). `tools` is the run's `ToolSet`, for a node that declares `needsTools` (TOOL,
  HUMAN_INPUT and AGENT). `signal` is
  optional in the type; the runtime's evaluator gives every node its own (the evaluator is not
  ported yet).
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
  and, as Python's `yian.nodes`, `src/nodes/index.ts` exports no kind. START's and END's bodies
  run. The other kinds hold their data, checks and load-time hooks (origins, setup and condition
  checks, MAP's `over` and WAIT's `until` bindings, HUMAN_INPUT's prompt sites, backward layouts,
  LOOP `respawn`); their `forward` throws "not ported yet".
- AGENT lives in `src/nodes/agent/node.ts`, and its `llm_config` values in
  `src/nodes/agent/llm_values.ts`. The node holds its data, checks and load-time hooks; `forward`
  and `backward` are not ported yet. `callTime` is an accessor over an own field, so setting it
  also sets `needsSteps`, and a shallow copy carries both. `llmValueTypes` is an array of
  `[path, type]` entries, not a map keyed by path. AGENT is the node the remaining `NodeCaps`
  fields serve (`skills`, `emitStream`, `emitEvent`, `captureRender`, and `steering`, a `Steering`
  inbox given only to an agent authored `steerable: true`).
- A CODE `code:` value is a reference (`./file.ts:fn`, also `.js` or `.mjs` and `../` paths,
  resolved against the node's `baseDir`) or an inline TS/JS body. The constructor classifies it
  by shape into `mode` and compiles or imports nothing.
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
- `compatibility(config)` in `src/nodes/agent/memory.ts` gives the provider and message format,
  `yian.message.v1` (Python's is `yian.langchain-message.v1`). TS yian never reads a Python memory
  store.

## Memory

- `src/memory/contracts.ts` holds the memory identities that compile records (`DeclSite`,
  `MemoryDecl`, `MemorySpec`, `MemoryStack`, `MemoryScope`), the declaration check
  `checkMemoryDeclaration`, the setup and operation failures, recorded snapshots, and the host
  memory objects (`Memory`, `MemorySession`, `isMemory`), the learning result carriers
  (`MemoryMutation`, `LearningFailure`, `PolicyCommitOutcome`, `LearningResult`), and the
  evidence carriers (`EvaluationEvidence`, `MemorySection`, `ModelTextInteraction`,
  `ModelValueInteraction`, `ToolCallInteraction`, `ToolResultInteraction`), and the learning
  inputs (`MemoryEpisodeStatus`, `MemoryEpisode`, `LearnContext`), the recall carriers
  (`RecallContext`, `MemoryContextRequest`, `MemoryContent`, `MemoryContext`, `MemoryRecallError`)
  and the session conversation contracts (`Conversation*`, `TranscriptSource`, `ConversationTurn`).
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
  `MemorySession`, `TranscriptSource`, `ConversationTurn`, `MarkdownStore.read/write`) may return a
  value or a promise, and yian always awaits it, as for tools; property reads (`snapshot`, `view`),
  `ConversationTurn.interrupt()` and the engine's `EvaluationTrace` stay synchronous. A session has
  no `with` form: the caller calls `close()` in `finally`.
- `TranscriptSource.openTurn(request, { signal })` resolves to an admitted turn (Python's
  `open_turn` and `__enter__`); the stop is that `AbortSignal` (no `CheckStopped`), and
  `ConversationTurn.close(error?)` replaces `__exit__`.
- `src/memory/conversation.ts` holds the conversation fold and its codecs and digests;
  `src/memory/summary.ts` holds whole-turn selection and the summary prefix proof. Neither is
  re-exported by `memory/index.ts`, as Python's `memory/__init__.py` doesn't import them.
- Changed: conversation JSON and `RecallContext.inputs` follow the value number rule (NaN refused,
  `±Infinity` kept, `-0` read as `0`); Python refuses non-finite numbers in conversation JSON.
- Conversation counters are safe-integer `number`s; wire JSON writes them as ints (`bigint`s) and
  the decoders check them.
- Conversation and summary digests are SHA-256 of `logJson` text. They equal Python's for ints and
  text and differ only for floats, astral key order and records shaped like the codec's
  `$float`/`$record` tags; TS stores are read only by TS.
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
- `src/memory/session.ts` holds `SessionMemory`, its run handle `Session` and its `Turn`, with
  optional SQLite storage opened at the first operation. Changed: the constructor takes an options
  object (`historyBudgetChars`, `recentTurns`, `storage`, `createParent`) whose settings are
  integral `number`s or `bigint`s; `openTurn` (which waits
  for the FIFO lease) and `prepareHistory` are async, everything else is synchronous; a stop is
  an `AbortSignal`; `close(error?)` never rethrows `error`, and appends a stop or cleanup failure
  to `error.notes`; snapshots are deep-frozen; `-0` equals `0` in a compatibility descriptor; there
  is no `with` form (call `close()` in `finally`) and no fork check; a waiter sees a poisoned
  store at its next wake-up, not by polling.
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

## Runtime

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
  `decodeRecord`, schema 10), the canonical helpers (`canonicalTyped`, `canonicalSeed`,
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
- For inspection, `readRunLog(path)` reads the complete prefix without the lock and without
  changing the file, so it may see a line a concurrent writer later rolls back;
  `isRunLogLocked(path)` probes the resolved log's `.lock`.
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
  (`prepareRunStarted`, `prepareNodeStarted`, `prepareNodeStep`, `prepareNodeSucceeded`,
  `prepareNodeRouted`, `prepareNodeFailed`, `prepareRunResumed`, `prepareRunStopped`,
  `prepareRecord`) validates one transition without mutating the state and returns a frozen
  `PreparedTransition`: the canonical record, the run id and sequence it expects, and a frozen plan.
  `commitPrepared(log, state, prepared)` appends the record, checks what the log returned, then
  applies the plan, with no `await` in between, so nothing interleaves between the append and the
  apply. Frontier order: START is ready after `RunStarted`; a success walks its target's outgoing
  edges in order and appends each successor when its last edge from the target settles and its join
  is ready; a route settles its edges depth first in stored order, skips the heads whose joins are
  dead (following each skipped node's output redirect) and appends the heads that become ready in
  that order; a resume puts the retried nodes first, in dispatch order, ahead of what was ready; a
  stop changes only the sequence and `lastStopSequence`. `interruptedAttempts` counts, per node,
  the attempts a resume retried that no stop covered, using `lastStopSequence` (the sequence of the
  last stop since the last resume). The first route whose skips reach END derives the
  `graph.terminal_unreachable` failure (naming the first required END input it skipped); it ends the
  run once nothing is ready or in flight, unless a node failure ends it first; a node failure ends
  the run at once. A bad output, an unknown handle or a bad commit target from a node throws
  `NodeDataError` (`original` holds the underlying error), which fails the node, not the run.
  Growth, the recovery fold (`replayRecords`) and the backward records throw `not ported yet
  (T4-5)`.
- Value equality: each carrier class has one `same*` helper in the module that owns it
  (`sameFailure` in `diagnostics.ts`, `sameFeedback` and `sameSource` in `expr/lineage.ts`,
  `sameBackwardResult` in `nodes/base.ts`, the backward carriers' in `runtime/contracts.ts`, the
  memory carriers' in `memory/contracts.ts`, `sameSourceEntry` in `runtime/trace.ts`). Every
  `same*` takes non-null values; compare a nullable pair with `sameOptional(a, b, same)` from
  `nodes/contracts.ts`. `memory/contracts.ts` writes its null checks inline, since it imports
  only `diagnostics` and `defaults`. A `BackwardFailure`'s cause compares by value, never `===`.

## Compile

- `src/compile/model.ts` holds the graph (`Flow`, `CompiledFlow`, `Edge`, `FlowOutput`), the
  growth-metadata check and `copyGraph`; `origins.ts` the origin traversal and rewrites;
  `execution_layout.ts` the static execution scopes; `validation.ts` the cycle, path and type
  checks; `manifest.ts` `flowManifest`, `flowFingerprint` (SHA-256 hex over the manifest's
  `canonicalJson`, RFC 8785) and `expansionDigest`. `src/compile/index.ts` re-exports Python's
  `__all__` only.
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
  `{"builtin": "identity", "field": "x"}`; `baseDir`, `mode` and its origins aren't identity. An
  AGENT contributes its authored fields (`own_llm_config`, not `llmConfig`); `llmConfig`,
  `llmConfigLabel`, `wiresLlmConfig`, `llmValueTypes`, `callTime` and `hostCredentials` aren't
  identity.
- In `typesCompatible` and `typeRelation`, an open numeric end is `null`.
- Compare positions and scopes with `sameExecutionPosition`/`sameExecutionScope`, output contracts
  with `sameOutputContract`, and arrays with `sameItems` (`src/nodes/contracts.ts`).

## CLI

- `src/cli/terminal.ts` holds `Terminal`, the host terminal a run prompts on: `askUser` (the
  run's `ask_user` handler), `askText`, `askChoice`, `promptInputs` (the missing flow inputs) and
  `close`. Prompts take turns in one FIFO and draw on stderr; `onStop` hears Ctrl-C or the end of
  input, which close the terminal, and `onActive` hears a turn take and give back the terminal.
  A call the closed terminal ends throws `TerminalStopped`, or the signal's reason once it
  aborted (`askUser`), or returns null.
- It reads keys with raw-mode `node:readline` and has no dependency: it imports only `node:`
  modules and yian's own.

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
