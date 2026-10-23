# The yian conformance suite

Language-neutral test cases that pin what a flow does, shared by the Python implementation
(`tests/test_conformance.py`) and the TypeScript one (`ts/tests/conformance.ts`). The two
implementations share no run-log format, so a case states its expectations as data: load
diagnostics, the run's output, its run log projected to a *trace*, observer events, and so on.
Each runner turns a case into calls on its own implementation and compares the results.

## Layout

```
conformance/
  README.md                       # this file: the case format and the runner contract
  cases/<group>/<name>.yaml       # one case per file; its id is <group>/<name>
  cases/<group>/<name>.log.jsonl  # optional golden TS run log (TS-only; see Running the suite)
  fixtures/                       # files cases share: CODE helpers, golden profiles, stub data
  schema/                         # JSON Schemas of the documents yian gives hosts
```

## Reading a case file

A case file is YAML 1.2 core schema: `true`/`false`, `null`/`~`, integers (decimal, `0o` octal,
`0x` hex), floats, and text. Nothing else is implicit: a date is text, and there are no merge
keys or `yes`/`no` booleans. A tag (`!!str`, `!!timestamp`, `!x` or a bare `!`) is an error. A
duplicate key or a mapping key that isn't text is an error. `1` is an int and `1.0` a float.

Values in a case (everything except flow text, `files`, `python_files`, `stdin`, `read` and `http`)
are read in yian's JSON codec form: `{$float: inf}` and `{$float: -inf}` are the floats infinity and
minus infinity, and `{$record: X}` is the record `X`. A record that is exactly `{$float: inf}` or
`{$float: -inf}`, or whose only key is `$record`, is written inside `$record`. A bare `.inf` or
`.nan` in a value is an error.

The format is closed: a key this file doesn't define is an error. The format changes only by
additions to this file.

## The case file

```yaml
kind: run                  # load | run | resume | backward | expr | value | defaults | fingerprint | cli | json
rules: [text-form]         # TS-era rules the expectation depends on; absent or empty means shared
pending: "#391"            # the TS runner expects this case to fail until that issue lands
python_skip: why           # the Python runner can't run this case at all
flow: |                    # the flow, in TS form (TS CODE bodies)
  id: greet
  ...
python_flow: |             # the flow for the Python runner, when it differs
files: {}                  # more files in the case directory: relative path -> text
python_files: {}           # the files for the Python runner, when they differ
env: {}                    # environment variables: name -> text
inputs: {name: Ada}        # run inputs
run: {}                    # host arguments of the run
host: {}                   # the host, as data
expect: {}                 # the expected results
python_expect: {}          # the Python expectation, when the rules changed it
```

Each kind adds a few keys of its own (below). `kind` and `expect` are required, and so are the
keys and results each kind requires (Kinds, below); `python_expect` needs `rules`.

### Which runner runs what

- A runner runs `flow` and checks `expect`. The Python runner uses `python_flow`,
  `python_files` and `python_expect` instead when they are present.
- The Python runner skips a case with `python_skip`, and a case with `rules` but no
  `python_expect`.
- `pending` is for the TS runner: it reports the case as an expected failure, and fails if the
  case passes, so the marker is removed in the change that makes it pass. The Python runner
  ignores `pending`; a case Python can't pass uses `python_skip`.
- The Python runner skips `json` cases: they check TS yian's own JSON reader and encoders.

## The case directory and environment

Every case runs in a new temporary directory, the *case directory*:

- `flow` (or `python_flow`) is written to `flow.yaml`, and each `files` (or `python_files`) entry to
  its relative path with mode 0600; `fixtures/` is a copy of `conformance/fixtures/`. The case
  directory, and every directory the runner makes for `files`, is 0700 (yian's auth store refuses
  anything others can read; its own checks are pinned by `tests/test_provider_auth.py`, not by
  cases). A TS CODE reference resolves relative to the file that holds it, so `flow.yaml`, or a
  `files` entry at the top of the case directory, names a shared helper `./fixtures/<file>.ts:<fn>`.
- The flow is loaded from the text of `flow.yaml` with the source label `flow.yaml`, resolving
  `uses:` and `call:` paths in the case directory. Frame labels are what the loader gives:
  `flow.yaml` for the case's flow, and a `uses:` child's file name (`child.yaml` for
  `lib/child.yaml`).
- yian's own variables come only from `env` (every kind but `cli`, below): every `YIAN_*`
  variable and every variable whose name ends in `_API_KEY`, `_BASE_URL`, `_ENDPOINT` or
  `_API_VERSION` is unset first, and `YIAN_AUTH_FILE` names a file in the case directory that
  doesn't exist, unless `env` sets them. Values in `env` are text.
- **Values keep their kinds.** A runner hands case values to the implementation as yian values
  with the kinds the case file gives them: `inputs`, stub tool results, human answers,
  `value`, `values` and `item`. A TS run still takes `inputs`, a tool result and an answer
  through host intake (design 3.2.1), as it takes any host's, so a whole float reaches it as an
  int: `2.0` is the int `2` (`2.5` stays a float).
- **Secrets** are written as `<key:N>` (N a number) in `run`, `resume`, `backward`, `env`, `host`,
  `stdin` and `http`. The runner replaces each with the text `sk-conformance-N` and fails the case
  if one appears in the trace, events, stream, tool calls, output, the raised error's message,
  stdout, stderr or a run log file.
- **Time.** The TS runner's runs use a virtual clock that starts at the real current time and
  moves only by the waits, so waits take no time. The Python runner waits for real, so a case
  keeps its waits short.

## Kinds

Keys marked required must be present. In a `load`, `expr`, `value`, `defaults`, `fingerprint` or
`json` case, `expect` and `python_expect` hold the results under Checks: exactly one group where
it says "or". A `run`, `resume` or `backward` case's `expect` has exactly one of `raises` and
`output`.

| Kind | Keys | Checks |
|---|---|---|
| `load` | `flow` (required), `files`, `env` | `expect.load` |
| `run` | `flow` (required), `files`, `env`, `inputs`, `run`, `host` | `expect.raises`, or `expect.output`; then `trace`, `events`, `stream`, `tool_calls`, `memory` when present |
| `resume` | as `run`, plus `host.crash_after_record` or `host.stop`, and `resume` | as `run`, for the resumed run |
| `backward` | as `run`, plus `backward` | as `run`, plus `expect.backward` |
| `expr` | exactly one of `expr` and `template` (required), `values`, `item`, `mode`, `typedefs`, `env` | `expect.value` or `expect.raises` |
| `value` | `type` and `value` (required), `typedefs`, `env` | `expect.value` and `expect.kind`, or `expect.raises` |
| `defaults` | `env` | `expect.defaults` or `expect.raises` |
| `fingerprint` | `flow` and `other` (required), `files`, `env` | `expect.fingerprint` |
| `cli` | `flow`, `files`, `env`, `args`, `stdin`, `signal`, `http` | `expect.exit`, then `stdout`, `stderr`, `json`, `json_lines`, `exists` when present |
| `json` | exactly one of `read` (with `mode`) and `write` (with `form`) | `expect.value` (a read), `expect.text` (a write), or `expect.raises` |

### `load`

Loads the flow. `expect.load` is `ok`, or `{diagnostics: [<diagnostic>, ...]}` in order (Python
reports one).

### `run`

Loads the flow and runs it with `inputs` (absent: `{}`), the `run` arguments and the host. The
runner always gives the run a log file in the case directory, and projects it to the trace.

| `run` key | Meaning | Absent |
|---|---|---|
| `run_id` | the run id | the implementation makes one (a placeholder in expectations) |
| `num_workers` | the worker bound; `0` is serial | the implementation's default, 8 |
| `learnable` | keep learning evidence | `false` |
| `llm_config` | the run's `llm_config` layer (its `headers` included) | none |
| `context` | `present`: tools get a host context object | none |

`expect.output` is `{status: succeeded, output: <value>}`, the output in data form (a file as
`{uri, mime, name}`), or `{status: failed, failure: <failure>}`. `expect.raises` is the host
error the run raised instead (the log may not exist). When the run raised before creating its
log, its trace is `[]`.

With `host.memory` (below), its `before` runs go first; `tool_calls` lists their calls too, and
every other result is the main run's.

### `resume`

For serial runs (`num_workers: 0`). A `resume` case gives one of `host.crash_after_record`, a
record number or `end`, and `host.stop` (below). A number `k` cuts the first run after record `k`
(0-based, so records `0`..`k` are kept): every later append is refused without writing, the run
is abandoned, and the runner checks that the cut happened. Each runner's cut still runs the run's
own cleanup (closing the log, and memory and MCP close calls; in TS, `runFlow`'s), which a real
crash would not. `end` lets the first run run to its end (a first run that raises fails the case),
so the resume is a finished run's, read-only. With `host.stop`, the first run must reject with the
stop, and the resume continues the stopped run. Then the flow is
resumed from the log with the `resume` arguments `num_workers` (absent: the recorded value),
`api_key`, `headers` and `context` (`context` as in `run`). The expectations are for the resumed
run: its output, the trace of the whole log, the events and stream of the resume call, and
`tool_calls` over both runs. With `host.memory`, the `before` runs come first (their calls are in
`tool_calls` too), and the cut run and the resume each get the memories built afresh, so the
resume sees what the cut run saved before the cut.

### `backward`

Runs the flow (`run.learnable: true`), then calls backward once per entry of `backward`
(required), a non-empty list of `{feedback, api_key, headers}` (only `feedback` required).
`expect.backward` has one entry per call: the `outcome` of the `backward_finished` record the
call appended, projected like the trace (nested nulls kept, so `failure: null` stays), or
`{raises: <raises>}` when the call raised. The trace includes the backward records. With
`host.memory`, a `before` run's backward calls aren't in `expect.backward`; the main run's
memories are closed after its last backward call, and `expect.memory` is read after that.

### `expr`

Evaluates one expression, or one template.

| Key | Meaning |
|---|---|
| `expr` | an expression as written inside `${...}` |
| `template` | instead of `expr`: a template with `${...}` spans |
| `values` | the names in scope; a dotted reference walks into them |
| `item` | the MAP item in scope (`item`, `item.path`), when present (not with a strict `template`) |
| `mode` | `binding` (the default): a missing reference is null, and a template is a binding value (one whole span gives its value; embedded spans give text, null as empty); `condition`: the `expr` is evaluated as a `when:` condition (either spelling, `a > 1` or `${a > 1}`) and gives a bool, with no `item`; `strict`: a missing reference is an error, and a template is a prompt rendered with the AGENT and HUMAN_INPUT prompt rules |
| `typedefs` | a `typedefs:` mapping as in a flow, for `isinstance` and constructors (not with a strict `template`) |

`expect.value` is the result; `expect.raises` the error.

### `value`

Checks a value against a type at a write boundary. `type` is type text as in a flow (`list(int)`,
`float(0.0..)`), `typedefs` the flow's `typedefs:`, `value` the value. `expect.value` is the
stored value and `expect.kind` its storage kind (`int`, `list[str]`, `object`, ...); or
`expect.raises`, with its `path`.

### `defaults`

Reads yian's defaults from `env`. `expect.defaults` maps entry names (as in `docs/defaults.md`,
`MAX_TOTAL_NODES`) to their values; entries not listed aren't compared, and an entry the
implementation doesn't define fails the case. `expect.raises` is the error for a bad variable.

### `fingerprint`

Loads `flow` twice and `other` once. The two loads of `flow` must have the same fingerprint, and
`expect.fingerprint` says whether `other`'s is the `same` or `different`. A case never states a
hash. `python_flow` and `python_files` are not keys of a `fingerprint` case: both runners load
`flow` and `other` as written, so a pair with CODE bodies is in TS form and carries
`rules: [code]`.

### `cli`

Runs the `yian` command with `args` (the arguments after `yian`) in the case directory, with
`HOME` set to an empty `home/` directory there, `PATH`, `COLUMNS=1000`, the `env` variables
and no other variables (the Python runner adds `PYTHONPATH`, to find yian, and `no_proxy=*`,
so a system proxy doesn't take requests meant for the stub; the runner doesn't set
`YIAN_AUTH_FILE`, so unless `env` does, its default under `HOME` applies), and `stdin` (absent:
empty; never a terminal). `expect.exit` is the exit code. In stdout and stderr the case
directory's path, and its resolved real path, are
replaced by `<tmp>`, and spaces at line ends are removed; then `stdout` and `stderr` are
compared exactly when present. `json` is stdout read as one JSON document, and `json_lines`
stdout read as one JSON document per non-empty line, both in codec form. A command that names a
generated path (`yian run` names its new run log on stderr) is checked without `stderr`, or
given the path (`--log`).

A `json` document, and each `json_lines` line, is checked as parsed (before the codec decoding)
against `conformance/schema/cli.schema.json` before it is compared; a document that doesn't
match fails the case. Runners validate with JSON Schema 2020-12; the TS runner uses
`ts/tests/schema.ts`, the small validator the event and message schema tests share, which
implements the keywords these schemas use.

`expect.exists` maps relative paths under the case directory (not absolute, no `..` part) to
`true` or `false`: whether each exists after the command (a dangling symlink exists).

`signal: {name: SIGINT, after: <text>}` interrupts the command: the runner starts it, sends the
signal once `after` appears in what it has written to stdout or stderr so far, then waits for it
to exit. `name` is `SIGINT` or `SIGTERM`. The case fails if the command exits before `after`
appears, if 60 seconds pass first, or if the command dies from the signal instead of exiting
(yian handles both and exits 130). The command starts with both signals at their default
disposition: a shell can start background jobs with SIGINT ignored, so a runner resets them in
the child (libuv resets dispositions; a runner on another spawn checks its own).

`http` is a stub HTTP server: a list of exchanges, answered in order.

```yaml
http:
  - request: {method: POST, url: "https://auth.openai.com/api/accounts/deviceauth/usercode",
              json: {client_id: app_EMoamEEZ73f0CkXaXp7hrann}}
    response: {status: 200, json: {device_auth_id: dev-1, user_code: ABCD-1234, interval: 1}}
  - request: {method: GET, url: "https://gateway.example/v1/models",
              headers: {authorization: "Bearer <key:1>"}}
    response: {status: 200, json: {data: [{id: b}, {id: a}]}, delay: 0.0}
    repeat: true        # only on the last exchange: it answers zero or more requests
```

- `request` takes `method` and `url` (required; the full URL, query included), `headers` (names
  compared case-insensitively; each listed one must be sent with that value, others are
  ignored) and at most one of `json` (the body read as JSON, compared exactly) and `form` (the
  urlencoded body as a mapping of text, compared exactly). `response` takes `status`
  (required, an int from 100 to 599), `json` (absent: an empty body) and `delay` (seconds
  before answering, a finite int or float ≥ 0, cut short when the runner stops the server).
  `repeat: true` is allowed only on the last exchange.
- `http` is plain JSON: runners don't decode it (a `$float` or `$record` mapping is just a
  mapping), but they fill its secrets, so the secret check covers stub tokens and device ids
  written as `<key:N>`.
- A `cli` case makes no HTTP request beyond those its `http` lists. The runner checks every
  request yian's auth and model-listing transport makes against `http` (an empty list when the
  case has none): it starts one server on 127.0.0.1 for every `cli` case and routes each such
  request to it, whatever its origin, with the original URL. How it routes is the runner's: the
  Python runner wraps yian's request function in the command's process and sends the original
  URL in an `x-conformance-url` header, which the request check ignores; a TS runner hooks its
  own HTTP seam.
- An exchange counts as requested once its request has been read, before its `delay`. A request
  that doesn't match the next exchange, or comes after the last one (unless that one has
  `repeat`), is answered 500 with no body and fails the case whatever the exit code; the failure
  names what differs, with filled secrets written `<key:N>`. After the command exits, the runner
  stops the server and waits for its handlers, then fails the case if an exchange without
  `repeat` was never requested.

### `json`

Reads JSON text with yian's lossless reader, or writes a value with one of yian's canonical
encoders (design #736, sections 3.2.1, 3.2.5, 3.2.12). `read` is the JSON text and `mode` is
`codec` (yian's codec form is decoded: data yian wrote) or `plain` (nothing is decoded: model
replies, tool arguments, `--inputs`); `expect.value` is the value read. `write` is a value
(in codec form, like every case value) and `form` is `canonical` (RFC 8785, for
fingerprints) or `log` (a run-log line); `expect.text` is the text written. A `log` text is
also read back in `codec` mode and must give `write` again. `expect.raises` is the error
instead. In `codec` mode a bare `{"$record": X}` whose `X` isn't an object (e.g.
`{"$record": 5}`) reads as a record holding that one key (the case
`json/read_codec_record_of_scalar` pins it). `json` cases are TS-only.

## The host, as data

A case can't hold code, so its host is declared under `host`.

- **`tools`**: stub tools by name, each `{parameters: <JSON Schema>, description: <text>,
  concurrency_safe: <bool>, results: <script>}` (`description` and `concurrency_safe`
  optional). With `concurrency_safe: true`, the tool's `isConcurrencySafe` returns true for every
  call, so an AGENT may run its calls together (see
  [`docs/api.md`](../docs/api.md#tool-context)). The Python runner ignores it, since one call at
  a time is a valid schedule. A script is a list of entries, taken one per call
  in order, or a mapping from node id to such a list, taken by the calling node (for tools that
  several nodes call at once). An entry is `{return: <value>}` or `{raise: {class: <name>,
  message: <text>, retry_after: <seconds>}}` (`retry_after`, a finite non-negative number of
  seconds, only for `TransientToolError`). The classes are `PermanentToolError`,
  `TransientToolError`, `ReportedToolError`, `ToolCheckError`, `TimeoutError` and
  `ConnectionError`; any other name raises a plain error with that class name.
  A call after the last entry fails the case before any result is compared, since it is the
  case's own error; an entry left unused fails it after the comparisons.
- **`human`**: `{<node id>: [<answer>, ...]}`, the answers the run's `ask_user` tool gives to the
  calling node, in order; the same rules when they run out or are left over. A case with `human`
  doesn't declare a tool named `ask_user`.
- **`jitter`**: the random draws in `[0, 1)` retry backoffs take, TOOL retries and AGENT model
  retries alike, in the order they are drawn: one per retry that waits a backoff, even a zero one,
  and none for a `retry_after` wait (a provider's retry-after, for a model retry). The same rules
  apply. Model-retry draws are TS-only: the Python runner gives `jitter` to TOOL retries alone, so
  a case that draws for a model retry is `python_skip`.
  Without `jitter`, draws are the implementation's own, and the case doesn't depend on them.
- **`models`**: scripted model calls, `{<node id>: {<consumer>: [<entry>, ...]}}`; each call
  takes the next entry of its node and consumer. The consumers are `agent` (an AGENT's calls),
  `structured_retry` (a structured call's attempts after its first, which keeps the outer
  consumer), `compose` (the calls of the synthetic composer AGENT `<gate>__compose`, under that
  node id), `backward`, `summary` (the AGENT's history summaries, under the AGENT's node id)
  and `memory:<key>` (a reflective or profile memory's learner, under the node id
  `__memory__`; `<key>` is the memory key and may hold `/`). An entry is
  `{reply: <message>}`, a yian.message.v1 assistant message, with an optional `usage`; or
  `{chunks: [<chunk>, ...]}` whose last chunk is `{finish: <reason or null>}`; or
  `{chunks: [...], raise: {class, message, status}}` for a failure after the chunks
  (`chunks: []`: before any output; `status` optional, the provider's HTTP status). The call
  fails with an error of class `class` and message `message`: as for a stub tool, `TimeoutError`
  and `ConnectionError` are those classes, and any other name is a plain error with that class
  name.
  In TS, a `raise` with `status` is the provider error an HTTP reply with that status and body
  would give (its kind read from the status and the message, as `kindForStatus` reads a body);
  its class name is TS's own, so a case with such a raise doesn't pin `ModelCallFinished.error`.
  Without `status`, it is a plain error named `class`, never retried.
  TS seats a call from the call itself, since its model is built per node: the node id is the
  one the model was built for, and the consumer is the one it was built for when it has one
  (`ModelAccess.createModel`'s `consumer`: `backward`, or `memory:<key>` under `__memory__`),
  else `summary` when the request's first message is a system message starting `Summarize the
  historical conversation` (the AGENT's summary prompt), else `structured_retry` when the
  request's last message is a user message starting `Your previous output was invalid:` (the
  correction a structured call adds after an invalid reply), else `compose` for a node id ending
  in `__compose` and `agent` for any other. Every TS model call streams, so no `chunks` entry is
  summed; the text chunks of an entry build one text part, ahead of its tool calls.
  Any entry may add `request: {messages: [<message>, ...], tools: [<tool spec>, ...], schema:
  <JSON Schema>}` (Python's schema:
  `convert_to_openai_tool(schema)["function"]["parameters"]`), which the runner compares
  exactly with the call's, failing the case when they differ (`tools` absent: none bound;
  `schema` present: the call asked for output matching it, Python's native structured call).
  In TS, `schema` is the native structured output's schema, a structured call lowered to a tool
  binds that tool in `tools`, and numbers compare by value, since a JS number doesn't keep `1`
  and `1.0` apart.
  A request's `messages` and tool specs are yian's own and shared; its `schema`, any text built
  from it (a fallback call's schema instruction) and a model call's `structured_*` `purpose` in
  `events` record how Python lowers a structured call, so a case that depends on them carries
  `rules: [llm]` (harvested cases are tagged before they are committed).
  A tool spec is `{name, description, parameters}` (Python:
  `convert_to_openai_tool(tool)["function"]`), in the bound order. A reply to a `schema` call is
  one text part holding the JSON value in the schema's carrier form (a wrapped scalar is
  `{"value": ...}`, a list `{"items": ...}`).
  A chunk is `{text: <delta>}`, `{tool_call: {index, id, name, arguments}}` (`id`
  and `name` when the delta carries them; `arguments` is text), `{usage: <usage>}` or
  `{finish: <reason or null>}`. A usage is `{input_tokens, output_tokens, total_tokens}` plus
  optional `cache_read_tokens`, `cache_creation_tokens` and `reasoning_tokens`. A `reply`
  served to a streamed call streams as one chunk; a `chunks` entry served to an unstreamed call
  is summed. The script replaces the provider below config resolution: the runner still
  resolves and checks each call's config, so a `models` case needs no provider key unless it
  tests credentials. The Python runner resolves no credentials from the config, so it doesn't
  check a pinned saved login that changed or was removed. In `run`, `resume` and `backward`
  cases every model call is scripted, so a call with no script fails the case at once instead of
  reaching a provider; calls that run out and leftovers fail as above.
- **`memory`**: `{before: [<run>, ...]}` turns memory on (`{}` with no earlier runs). For each
  run (each `before` run, the main run, and for a `resume` case the cut run and the resume) the
  runner builds the flow's memories from their declarations, fresh per run, rooted at
  `<case dir>/memory` (Python: `build_memories(flow, root=...)`), passes them to the run, and
  closes them after that run's backward calls. An error building them is the case's `raises`
  (e.g. `MemorySetupError`), except in a `before` run (below); in a `resume` case, the cut run's
  error skips the resume. A `before` entry is `{inputs, run, backward}`, the case's own shapes
  (`run.run_id` required; `backward` needs `run.learnable: true`); the `before` runs go in
  order, before the main run, over the same memory root. Their outcomes aren't compared, but a
  raise (a run's, a backward call's, or building their memories) fails the case. Scripts are
  shared by all runs in order; `tool_calls` lists every run's calls; `trace`, `events`,
  `stream`, `output` and `backward` are the main run's. A profile is seeded by a `files` entry
  at its storage path, `memory/<flow id>/<scope...>/<name>.md`.
  In TS, the memories are `buildMemories(flow, {root, models, onEvent})` with the runner's
  scripted model access; the events `onEvent` gets (the learners' `ModelCallRetried` and
  `ReplayDegraded`, TS-only) are kept apart and never compared with `expect.events`.
- **`steering`**: `{<node id>: [<action>, ...]}`, the host inbox a steerable AGENT polls (Python:
  the `steering` argument of `run_flow` and `resume_flow`). The node id is the occurrence id, as
  in `tool_calls`' `node` (`loop#0/chat` inside a LOOP or MAP). An action is
  `{after: <point>, steer: [<text>, ...]}`, which queues steer messages, or
  `{after: <point>, interrupt: true}`, which sets the node's interrupt. An action without `after`
  is queued before the run. The points:
  - `{model_call: n}`: once the node's `n`th `agent` model call has finished (its last chunk
    taken); not when the call raises;
  - `{model_call: n, chunk: k}`: once chunk `k` of that call has been delivered; `k` counts the
    entry's `chunks` list, text or not (a `reply` streams as one chunk);
  - `{tool_call: i}`: once the node's `i`th tool call in call order has returned or raised,
    counted as in `expect.tool_calls` (`ask_user` included). Stub tools return at once, so calls
    finish in call order.

  Counts start at 1. In a `resume` case they run across the cut run and the resume, as the
  scripts do. Steering acts in the main run (and its resume), never in a `host.memory` `before`
  run, whose calls aren't counted. Steers are drained once; an interrupt stays set. An action
  whose point is never reached fails the case after the comparisons, like a leftover entry.
  Steering in a `resume` case is specified but not yet pinned by a case here.

  The AGENT reads its inbox only at fixed places: steers and the interrupt at the top of each
  loop iteration (just before a model call), the interrupt before each later chunk of a
  streamed reply and before each tool call. A polled inbox can't act "before" a point, so:
  - a steer before model call `n` is `after: {model_call: n-1}`, and one before model call 1
    has no `after`;
  - an interrupt before tool call `i` is `after: {tool_call: i-1}`, or `after: {model_call: n}`
    for the first call of reply `n`. In TS this holds only when call `i` isn't in the same group
    as call `i-1`: the calls of `concurrency_safe` stub tools run together, so a group's later
    calls have already started when an earlier one returns, and they finish;
  - an interrupt after chunk `k` is `after: {model_call: n, chunk: k}`.

  TS reads the interrupt before each later **text** chunk, before each tool call's check,
  before each group, and before failing at a call whose check failed. So a case that cuts mid-stream needs a text chunk after its point (an
  interrupt before a `tool_call`, `usage` or `finish` chunk keeps the reply whole in TS, and its
  calls are skipped with brackets), or a TS `expect`.

  An interrupted AGENT succeeds with its text so far (`""` before any reply): mid-stream it keeps
  the chunks shown and ends the stream with a final `""` chunk, and on the last chunk it keeps the
  whole reply. The tool calls it skips finish with error `interrupted`. Only an AGENT authored
  `steerable: true` (with `str` output) reads the inbox.
- **`stop`**: `{<node id>: {after: <point>}}`, a stop of the run (TS only; the Python runner
  doesn't support it yet). The points and their counting are `steering`'s, and `after` is
  required. When the run reaches the point, the runner aborts the run's `signal` with a
  `ConformanceStop` error, so the run rejects with it: a `run` case expects `raises: {class:
  ConformanceStop, message: conformance stop}`, and its trace ends with `run_stopped`. A stop
  whose point is never reached fails the case after the comparisons, like a leftover entry.
- **`crash_after_record`**: `resume` cases only (above).

`expect.tool_calls` lists, per tool that was called, each call in order as `{args, call_id,
attempt, node}`.
`ask_user` calls from `human` are listed too; their `args` are the request in data form
(`question`, `questions`, `answer_schema`, `title`, `node_id`, `options`; nulls left out).
HUMAN_INPUT never sends `options`; a model's `ask_user` call can.

`expect.memory` (it needs `host.memory`) is `{<key>: <text>}`: after the main run and its
backward calls, the runner builds the memories afresh and reads each listed key's whole recall
text (Python: `open_run().recall(None)`, a `MemoryContent`'s `text`). Its keys, like `<key>` in
`memory:<key>`, are memory keys as the `memory_commit_*` records name them. A key the flow
doesn't declare fails the case. The Python runner also fails a session key, whose whole recall is
empty by contract; TS yian has no session memory.

The format reserves these host keys; a runner that doesn't support one yet fails the case that
uses it:

- **`skills`**: skill directories, given as `files`.

Not in the format yet (each comes as an addition when a case needs it): file values in
`values`/`item`, output schemas for stub tools, `files` for the `other` flow of a fingerprint
case, a clock for `system.now`, and the content of a file after a `cli` command (such as
which logins an auth file holds), so no case pins a removal or the token a refresh saves.

## Forms

**Comparison** is exact and kind-strict: `true`, `1`, `1.0` and `"1"` all differ; mappings are
compared by keys and values, ignoring key order; lists in order. One exception: in values a host
receives (`expect.output`'s `output`, event data, stream chunks and `tool_calls`), an int and a
float compare by value (`2` equals `2.0`), because a TS host receives both as a JS `number`. The
trace keeps the kinds. Kinds are about yian values: the format's own numbers (ranges, paths,
attempts, sequence numbers, `defaults` values, `exit`) are ints, and `retry_in` is a float.

**Sorting**, wherever this file sorts, is by UTF-16 code units.

**Diagnostic**: `{code, message, frames, notes}`. A frame is `{label, range: [start_line,
start_col, end_line, end_col], note}`, 1-based with an exclusive end column. `notes`, `frames`
and a frame's `note` are left out when empty.

**Messages.** In `expect.load` diagnostics and in `cli` `json`/`json_lines` documents, an expected
mapping that leaves out `message` doesn't compare it. Cases use this for text the YAML parser
writes (a syntax error), for text yian doesn't write (a validation library's message, a CODE
syntax error the runtime reports), and for `cli` documents, whose message text isn't part of
the contract.

**Failure**: a node or run failure in diagnostic form, as `diagnose(failure)` resolves it, plus
the failing node: `{code, message, node, frames, notes}`, with `node` null for a run-level
failure.

**Raises**: `{class, message}`, plus `path` for a `TypeCheckError` (a list of field names and
list indexes, `[]` at the root). `class` is the error's class name, read through the error-class
table.

**Error-class table.** Class names are compared through one table, used for `raises` and for the
error type in TOOL steps. It maps each implementation's name to the case name; the Python names
are the case names.

- The tool classes `PermanentToolError`, `TransientToolError`, `ReportedToolError`,
  `ToolCheckError`, `TimeoutError` and `ConnectionError`, and yian's own classes `LoadError`,
  `FlowValidationError`, `ExpressionError`, `RequiredError`, `ConstructorTypeError`,
  `TypeCheckError`, `DefaultsError` and `DefaultsMismatchError`, have the same name in both
  implementations. `IntTooLongError` is Python-only (TS has no digit limit), so a case holding it
  carries `rules: [numbers]`.
- In `raises`, a Python class whose name starts with `_` is private, and its case name is its
  nearest public base class's (`_NoUnionMember` is `TypeCheckError`).
- TS's `RangeError` is the case's `ValueError`: TS throws it wherever Python raises `ValueError`
  at a yian boundary. A JS-internal `RangeError`, such as a stack overflow, reads as `ValueError`
  too.
- TS's `RunLogSchemaError`, the `RangeError` a run log of a schema TS can't read throws, is the
  case's `ValueError`, which Python raises there.
- Any other class compares by its name as given, a stub's declared class included.
- A row mapping a TS class to a different case name is added by the change that introduces that
  TS class; rows are only added.

**Data form** of an event, stream chunk or `ask_user` request: a mapping of its fields by their
snake_case names, at every depth of yian's own structures (never inside a value the flow
produced, such as an output or an answer); fields that are null are left out at the top level
only; lists and tuples as lists, sets as sorted lists, a failure in failure form, a file as
`{uri, mime, name}`.
The `attempt` of `ModelCallStarted`, `ModelCallFinished` and `StreamChunk` is left out when it
is 1 (a null one already is); cases never write `attempt: 1`, and the TS runner's check refuses
a case that does.

- An **event** is `[<EventName>]` or `[<EventName>, <data form>]`, for example
  `[NodeSucceeded, {node_id: hello, output: hi Ada}]` and `[RunStarted]`.
- A **stream chunk** is its data form.

## The trace

The trace is the run log projected to what both implementations must agree on. Each record
becomes one entry `[<kind>, <node>, <payload>]`, in log order. `node` is null for a run-level
record. In a payload, a top-level field that is null is left out (nested values are kept as
they are, except as the rules below say), and an empty payload is left out.

| Record | Node | Payload fields |
|---|---|---|
| `run_started` | null | `learnable`, `num_workers`, `defaults`, `effective_llm_by_static_path`, `call_time_llm`, `parent` |
| `run_resumed` | null | `retry_node_ids`, `num_workers` |
| `run_stopped` (TS only) | null | `reason` (the reason's kind), `node_ids` |
| `node_started` | `node_id` | none |
| `node_step` | `node_id` | `step` |
| `pool_commit` (TS only) | `node_id` | `key` (only when it differs from the node), `value` (left out for `system.now` and `system.today`) |
| `node_succeeded` | `node_id` | `commit` (the seq of the `pool_commit` it accepts), `forward_capture` |
| `node_routed` | `node_id` | `handle`, `terminal_failure`, `forward_capture` |
| `node_expanded` | `spawner_id` | `seed`, `prune_ids`, `forward_capture` |
| `node_failed` | the failure's node | `failure`, `forward_capture` |
| `memory_commit_requested`, `_succeeded`, `_failed` | null | `memory`, `commit_id`, `attempt_id`, `mutations`, `failure` |
| `backward_started` | null | `attempt_id`, `feedback` |
| `backward_ready` | null | `attempt_id`, `root_result`, `attributed_results`, `proposals` |
| `backward_finished` | null | `attempt_id`, `outcome`, `attributed_results` |

For example:

```yaml
- [run_started, null, {learnable: false, num_workers: 0, defaults: {MAX_TOTAL_NODES: 20000}}]
- [pool_commit, __start__, {key: system.now}]
- [pool_commit, __start__, {key: system.run_id, value: r1}]
- [pool_commit, __start__, {key: system.today}]
- [pool_commit, __start__, {value: {q: x}}]
- [node_started, fetch]
- [node_step, fetch, {step: {event: failed, attempt: 1, error: {type: TransientToolError, message: busy}, retry_in: 0.5}}]
- [pool_commit, fetch, {value: F1}]
- [node_succeeded, fetch, {commit: 7}]
- [node_failed, place, {failure: {code: node.execute, message: rejected, node: place, frames: [{label: flow.yaml, range: [12, 3, 12, 8]}]}}]
```

- Every write to the variable pool is a `pool_commit` (TS schema 13): the run start writes the
  system values (keys `system.now`, `system.run_id`, `system.today`, in key order) and then the
  START inputs right after `run_started`, and a node's output is a `pool_commit` before the
  `node_succeeded` that names its seq. A commit no success names (a crash between the two) stays
  in the log and its node reran. The run start's times (`system.now`, `system.today`) leave out
  their `value`.
- Values are plain values: a `pool_commit`'s `value` is recorded typed (`kind` and `value`), and
  is written without the kind tag (a file as `{uri, mime, name}`); every other value (seeds,
  captures, feedback) is recorded plain already.
- `defaults`, `effective_llm_by_static_path`, `call_time_llm` and `parent` are left out when
  empty or absent.
- Dropped: the envelope (`schema`, `run_id`, `seq`); a `pool_commit`'s `kind`; from
  `run_started`, `flow_fingerprint`, `start_wiring_override`, `memory_snapshots`, `memories`,
  `memory_stacks`, and the bulky `flow` (the flow files), `skills`, `tools` and `yian_version`;
  `topology_digest`; and every `evidence` field (the model and tool evaluation
  evidence, whose shape is implementation-specific).
- Failures: `node_failed.failure`, `node_routed.terminal_failure` and a backward failure's
  `cause` (when it is a node failure, with `trace`) are in failure form. A backward failure
  (`backward_finished.outcome.failure`) leaves out `cause_type` and `traceback`. Nothing else
  is removed, and values the flow produced or received are never rewritten, except by
  placeholders. A memory record's `failure` is a memory operation failure, `{stage, code,
  message, retryable}`, kept as recorded.
- In `forward_capture`, a route (and each route nested in its `children` and `snapshot`)
  leaves out `expression` and `result_type`, because they are text each implementation spells
  its own way (a literal's JSON text, markers for values JSON can't hold); its structure stays.
  It also leaves out every other field equal to its empty value (null, `[]`, `0`, `""`), except
  `kind`, which always stays, and its `value_json` text is read as JSON (codec form). The
  capture leaves out `evaluated_keys`, `items`, `input_names` and `variable_names` when they
  are `[]`, and `node_data` when it is null or `{}`; `bound_inputs` is kept as recorded, even
  when empty. Each `items` entry is `{index, capture}`, and its `capture` is projected by these
  same rules. `node_data.effective_llm_config` is projected as below; everything else is kept
  as recorded (sources, spans). A route's `spans` are offsets in code points into its result
  text (the text its `value_json` holds).
- An agent's LLM config (each value of `run_started.effective_llm_by_static_path` and of
  `run_started.call_time_llm.agents`, the config of an `effective_llm_config` step, and a
  `forward_capture`'s `node_data.effective_llm_config`) keeps its `llm_config` fields (keys
  not starting with `_`), `_header_names`, `_host_key`, and `_connection` cut to `method`,
  `source` and `endpoint` (key hashes and store paths are dropped); any other `_` key is
  dropped.
- `call_time_llm` keeps only `agents`: its `selection` is dropped, because its fields are
  internal and its effect is recorded in each call-time agent's `effective_llm_config` step.
- The backward records' nested structures are kept as recorded, except the backward failure
  above.
- In TOOL steps, `retry_in` is a float, and a `failed` step's error type goes through the
  error-class table.

**Python reference.** The trace rules at tag `conformance-trace-python` are the last ones both
implementations share. After that the TS trace shows the TS log as written; a shared case whose
expectation changes is regenerated with `--write-expect` and gets `rules: [run-log]`, which the
Python runner skips.

**Placeholders.** Ids an implementation makes up are replaced by `<id:1>`, `<id:2>`, ...: a run id
the case didn't give, and the values of `exchange_id`, `commit_id`, `mutation_id` and `note_id`
fields. (This list grows only by fields no committed case holds yet.) The generated run id is the
run's own id. Every occurrence inside a text value is replaced, so a call id `<run id>/fetch`
becomes `<id:1>/fetch`; mapping keys are left alone. The ids are collected from every
projection, expected or not, and numbered in order of first appearance (by position, within one
text), walking the `before` runs' traces first, run by run in order (their ids are numbered,
but their entries are never compared), then the trace, events, stream, tool calls (tools in
sorted order), output, backward outcomes and `expect.memory` texts, with lists in order and
mapping keys in sorted order; projections added later are walked after `expect.memory`. An
episode's `entry_id` equals a `mutation_id`, so recall text gets the same placeholder. Since
replacement is textual, the TS runner's id source makes ids that can't occur in other text
(UUIDs, for example). A `raises` expectation is compared as is; a case whose error names the
run id gives `run_id`.

**Pooled runs.** When the run's `num_workers` isn't 0, the trace, events, stream and each tool's
calls are first grouped: run-level entries first, then each node's entries, nodes in sorted
order, each group keeping its order, and each `node_succeeded` leaves out its `commit` (the seq of
the `pool_commit` just before it in its node's group, which depends on how the parallel nodes
interleave). Placeholders are numbered and lists compared after that.
Expectations are written grouped. A `resume` case's results are grouped by the resumed run's
`num_workers` (`resume.num_workers`, else the recorded one). A `before` run's trace is grouped
when its own `num_workers` isn't 0.
An entry's node is its trace `node`; an event's `node_id` (a `NodeFailed`'s failure node; run
events are run-level); a chunk's `node_id`; a tool call's `node`.

## Rule ids

A case lists in `rules` the TS-era rules its expectation depends on (design #736, section 3.2).

| Id | Rule |
|---|---|
| `numbers` | int is unbounded and kept apart from float; lossless JSON; host number intake; duplicate keys refused; `YIAN_*` values stop at 2^53-1 |
| `int-in-float` | an int filling a float slot stays an int at any depth, and an int spelled as text for a `float` input reads as that int; Python widens it to a float |
| `arithmetic` | the operator table: `/` gives a float, `%` takes the left sign, no repetition or `%` formatting, division by zero |
| `compare` | equality across kinds, ordering, membership |
| `text-form` | how a value becomes text: JSON-style values, JS number spelling |
| `canonical` | RFC 8785 fingerprints and the log's canonical JSON |
| `columns` | 1-based UTF-16 columns; line breaks only at LF, CRLF and CR |
| `messages` | values, lists of names and quoted keys in messages spelled in the text form |
| `text` | character counting, ASCII lexer classes, whitespace, `$` at the end of text only |
| `yaml` | YAML 1.2 as the TS parser reads it |
| `dates` | dates and times as text, `system.now` |
| `infinity` | infinity is a float value |
| `llm` | how the TS LLM layer builds requests |
| `code` | TS CODE bodies and their failures |
| `failure-codes` | decision 36's TS-only failure codes |
| `run-log` | the TS run log redesign: the trace shows the TS log as written |

## Groups

A case's group is its directory under `cases/`.

| Group | Kind(s) | Covers |
|---|---|---|
| `defaults` | defaults | the defaults table, parse rules, messages |
| `json` | json (TS-only) | lossless reader, canonical encoders, RFC 8785 vectors, log number form |
| `values` | value | building values, kinds, files, infinity and NaN, nesting, surrogates, `-0`, host intake |
| `types` | value, load | type text, ranges, typedefs, type nesting |
| `yaml` | load | YAML 1.2 scalars, dates, merge keys, non-text keys, limits, syntax errors, line breaks |
| `expr` | expr | parsing, nesting, arithmetic, comparison, membership, truthiness |
| `templates` | expr | the scanner, `$$`, the text form, builtins, prompt strictness |
| `load-core` | load | flow shape, references, wiring, outputs, compact mode, `uses:`/imports |
| `load-types` | load | tags, typed bindings, constructors, `isinstance`, narrowing |
| `load-case` | load | CASE branches, exhaustiveness, value entries, skips |
| `load-loop` | load | LOOP seeds, predicates, partial updates |
| `load-map-call` | load | CALL, MAP, inline bodies, inline `call(...)`, call nesting |
| `load-agent` | load | AGENT fields, the `llm_config` cascade, provider blocks, skills, memory declarations |
| `load-leaves` | load | TOOL, CODE, HUMAN_INPUT, WAIT, asserts |
| `fingerprint` | fingerprint | same and different pairs, determinism |
| `run-core` | run | START and END, bindings, outputs, failures, events, stops |
| `run-case`, `run-loop`, `run-map-call` | run | control flow and growth |
| `run-leaves` | run | TOOL (retries, call ids), CODE, HUMAN_INPUT, WAIT |
| `run-agent` | run | AGENT with scripted models: plain, structured, tool loop, steering, interrupts, retries |
| `run-memory` | run | episodic, reflective and profile memories |
| `resume` | resume | recovery, TOOL attempts across a resume, interrupted-attempt counts, re-supplied keys |
| `backward` | backward | explicit backward feedback and learning |
| `cli` | cli | non-interactive `yian run`, the `--json` commands, `yian status --json` |

## Running the suite

```bash
PYTHONPATH=src python -m pytest -q tests/test_conformance.py              # every case
PYTHONPATH=src python -m pytest -q tests/test_conformance.py -k run-core  # one group
CONFORMANCE_CASES=/tmp/cases PYTHONPATH=src python -m pytest -q tests/test_conformance.py  # cases under /tmp/cases
```

Each case is one test named by its id. A skipped case says why.

```bash
cd ts && bun tests/conformance.ts                                  # every case, on Bun
cd ts && bun tests/conformance.ts --group defaults --kind defaults # a selection
cd ts && bun run build && node dist/tests/conformance.js           # every case, on Node
```

The TS runner's flags, each repeatable, are `--group`, `--kind`, `--case <group>/<name>`,
`--rules` and `--diagnostic-code`. It prints one `FAIL` line per failing case and a summary,
counts a failing `pending` case as expected, fails a case whose kind it doesn't run yet, and exits
1 when any case fails. On every `run` and `resume` case with a log it also checks the log's invariants
(in-flight nodes within `num_workers`, nothing but backward and memory records after the terminal
outcome, `inspectRun` agreeing with a replay, a finished log resumed read-only). `--crash-sweep`
also cuts each serial `run` case that expects `output`, isn't `pending`, and has no `host.models`,
`host.steering`, `host.memory` or `host.stop` after every record but the last, resumes it, and
checks the invariants, plus the output and node outcomes when the case has no `host.human` or
`host.jitter` and its tool scripts only return. A resume adds records, so in those outcomes a
capture source's `execution_seq` is compared by the kind and node of the record it names, and a
success's `commit` by the trace fields of the `pool_commit` it names. A node cut right after its `pool_commit` runs again, so what it took from the host scripts goes
back from its last record before the commit (its steps replay). A cut inside the run start
(before the START inputs' commit) can't be resumed: the sweep expects the resume to refuse it
(`run log <path> ends inside its run start; start the run again`) and leave the log as it is.

A `run`, `resume` or `backward` case may have a **golden log**, `<name>.log.jsonl` beside it: the
TS run log the case writes (for a `resume` case, the whole log after the resume; for a `backward`
case, with its backward records), pinned byte for byte. Before
the comparison, every `traceback` value, each record's time (`at`), the seeded clock values
(the `value` of the `system.now` and `system.today` commits), the recorded flow's folder (`run_started`'s
`flow.dir`) and the yian version (`yian_version`) are replaced by `"<masked>"`, since the times
come from the virtual clock, which starts at the real time, the folder is the case's temporary one,
and the version changes with each release; each line is then written again with the log's own
encoder, which gives back the line as written. When the TS log format changes, regenerate the
golden logs with `--write-logs`, naming each case that has one, and check the diff holds only the
change. Bun and Node write the same bytes. A golden log goes
with a serial case that sets `run.run_id`, makes no random ids (no memory; an AGENT case only when
its scripted replies give every tool call id), and holds no file paths; the error texts in it are
a stub's or the runner's own (the stop's `conformance stop`).
`--write-logs` (for development; it needs `--case`) writes the golden log of each selected `run`,
`resume` and `backward` case instead of comparing it. The Python runner ignores golden logs.

```bash
cd ts && bun tests/conformance.ts --write-expect --group run-core      # regenerate a group's expectations
cd ts && bun tests/conformance.ts --write-expect --case run-core/x     # or one case's
```

`--write-expect` reruns each selected `run`, `resume` and `backward` case and, for each of
`trace`, `events`, `stream` and `tool_calls` that the case already has and whose computed value
differs, writes the computed value back into `expect`, then adds `run-log` to `rules` (a new `rules`
goes right after `kind`). It prints a `wrote <id>: <keys>` line for each case it changes. Values
are written in the codec form above, after placeholders, so a float keeps its fraction (`2.0`), an
int stays exact at any size, and a made-up id stays `<id:N>`. Only the lines of `rules` and
`expect` change, and in them the parts that already held their value keep their text, comments
included. It never writes `output`, `raises`, `flow` or `host`, or any other key: a case whose
`output` or `raises` differs (or that fails another way) is reported as a failure and left as it
is, and a `pending` case is never written. Review the diff like a golden log: every changed line
holds only the change that made the expectation differ.
