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
- `ScriptedModel` in `src/llm/testing.ts` scripts model replies for tests.

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
