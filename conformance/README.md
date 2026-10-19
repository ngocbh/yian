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
  cases/<group>/<name>.log.jsonl  # optional pinned TS run log (TS-only)
  fixtures/                       # files cases share: CODE helpers, golden profiles, stub data
  schema/                         # JSON Schemas of the documents yian gives hosts
```

## Reading a case file

A case file is YAML 1.2 core schema: `true`/`false`, `null`/`~`, integers (decimal, `0o` octal,
`0x` hex), floats, and text. Nothing else is implicit: a date is text, and there are no merge
keys or `yes`/`no` booleans. A tag (`!!str`, `!!timestamp`, `!x` or a bare `!`) is an error. A
duplicate key or a mapping key that isn't text is an error. `1` is an int and `1.0` a float.

Values in a case (everything except flow text, `files`, `python_files`, `stdin` and `read`) are
read in yian's JSON codec form: `{$float: inf}` and `{$float: -inf}` are the floats infinity and
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

- `flow` (or `python_flow`) is written to `flow.yaml`, and each `files` (or `python_files`)
  entry to its relative path; `fixtures/` is a copy of `conformance/fixtures/`. A TS CODE
  reference resolves relative to the file that holds it, so `flow.yaml`, or a `files` entry at
  the top of the case directory, names a shared helper `./fixtures/<file>.ts:<fn>`.
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
  `value`, `values` and `item`. (So `2.0` reaches a TS run as a float; a TS runner passes these
  without the host number intake of design 3.2.1.)
- **Secrets** are written as `<key:N>` (N a number) in `run`, `resume`, `backward`, `env`, `host`
  and `stdin`. The runner replaces each with the text `sk-conformance-N` and fails the case if one
  appears in the trace, events, stream, tool calls, output, the raised error's message, stdout,
  stderr or the run log file.
- **Time.** The TS runner uses a virtual scheduler, so waits take no time. The Python runner
  waits for real, so a case keeps its waits short.

## Kinds

Keys marked required must be present. In a `load`, `expr`, `value`, `defaults`, `fingerprint` or
`json` case, `expect` and `python_expect` hold the results under Checks: exactly one group where
it says "or". A `run`, `resume` or `backward` case's `expect` has exactly one of `raises` and
`output`.

| Kind | Keys | Checks |
|---|---|---|
| `load` | `flow` (required), `files`, `env` | `expect.load` |
| `run` | `flow` (required), `files`, `env`, `inputs`, `run`, `host` | `expect.raises`, or `expect.output`; then `trace`, `events`, `stream`, `tool_calls` when present |
| `resume` | as `run`, plus `host.crash_after_record` and `resume` | as `run`, for the resumed run |
| `backward` | as `run`, plus `backward` | as `run`, plus `expect.backward` |
| `expr` | exactly one of `expr` and `template` (required), `values`, `item`, `mode`, `typedefs`, `env` | `expect.value` or `expect.raises` |
| `value` | `type` and `value` (required), `typedefs`, `env` | `expect.value` and `expect.kind`, or `expect.raises` |
| `defaults` | `env` | `expect.defaults` or `expect.raises` |
| `fingerprint` | `flow` and `other` (required), `files`, `env` | `expect.fingerprint` |
| `cli` | `flow`, `files`, `env`, `args`, `stdin`, `signal` | `expect.exit`, then `stdout`, `stderr`, `json`, `json_lines` when present |
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

### `resume`

For serial runs (`num_workers: 0`). `host.crash_after_record: k` (required) cuts the first run
after record `k` (0-based, so records `0`..`k` are kept): every later append is refused without
writing, the run is abandoned, and the runner checks that the cut happened. The Python runner's
cut still runs the run's own cleanup (closing the log; later, memory and MCP close calls), which
a real crash would not. Then the flow is resumed from the log with the `resume` arguments
`num_workers` (absent: the recorded value), `api_key`, `headers` and `context` (`context` as in
`run`). The expectations are for the resumed run: its output, the trace of the whole log, the
events and stream of the resume call, and `tool_calls` over both runs.

### `backward`

Runs the flow (`run.learnable: true`), then calls backward once per entry of `backward`
(required), a non-empty list of `{feedback, api_key, headers}` (only `feedback` required).
`expect.backward` has one entry per call: the `outcome` of the `backward_finished` record the
call appended, projected like the trace (nested nulls kept, so `failure: null` stays), or
`{raises: <raises>}` when the call raised. The trace includes the backward records.

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
and no other variables (the Python runner adds `PYTHONPATH`, to find yian; the runner doesn't
set `YIAN_AUTH_FILE`, so unless `env` does, its default under `HOME` applies), and `stdin`
(absent: empty; never a terminal). `expect.exit` is the exit code. In stdout and stderr the case directory's path, and its resolved real path, are
replaced by `<tmp>`, and spaces at line ends are removed; then `stdout` and `stderr` are
compared exactly when present. `json` is stdout read as one JSON document, and `json_lines`
stdout read as one JSON document per non-empty line, both in codec form. A command that names a
generated path (`yian run` names its new run log on stderr) is checked without `stderr`, or
given the path (`--log`).

A `json` document, and each `json_lines` line, is checked as parsed (before the codec decoding)
against `conformance/schema/cli.schema.json` before it is compared; a document that doesn't
match fails the case. Runners use a JSON Schema 2020-12 validator (in TS a dev-only dependency,
shared with the event and message schema tests).

`signal: {name: SIGINT, after: <text>}` interrupts the command: the runner starts it, sends the
signal once `after` appears in what it has written to stdout or stderr so far, then waits for it
to exit. The case fails if the command exits before `after` appears, if 60 seconds pass first,
or if the command dies from the signal instead of exiting (yian handles SIGINT and exits 130).
The command starts with SIGINT's default disposition: a shell can start background jobs with it
ignored, so a runner resets it in the child (libuv resets dispositions; a runner on another
spawn checks its own). SIGINT is the only signal so far.

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
  results: <script>}` (`description` optional). A script is a list of entries, taken one per call
  in order, or a mapping from node id to such a list, taken by the calling node (for tools that
  several nodes call at once). An entry is `{return: <value>}` or `{raise: {class: <name>,
  message: <text>, retry_after: <seconds>}}` (`retry_after` only for `TransientToolError`). The
  classes are `PermanentToolError`, `TransientToolError`, `ReportedToolError`, `ToolCheckError`,
  `TimeoutError` and `ConnectionError`; any other name raises a plain error with that class name.
  A call after the last entry fails the case before any result is compared, since it is the
  case's own error; an entry left unused fails it after the comparisons.
- **`human`**: `{<node id>: [<answer>, ...]}`, the answers the run's `ask_user` tool gives to the
  calling node, in order; the same rules when they run out or are left over. A case with `human`
  doesn't declare a tool named `ask_user`.
- **`jitter`**: the random draws in `[0, 1)` TOOL retry backoffs take, in order: one per retry
  that waits a backoff, even a zero one, and none for a `retry_after` wait. The same rules apply.
  Without `jitter`, draws are the implementation's own, and the case doesn't depend on them.
- **`models`**: scripted model calls, `{<node id>: {<consumer>: [<entry>, ...]}}`; each call
  takes the next entry of its node and consumer. The consumers are `agent` (an AGENT's calls),
  `structured_retry` (a structured call's attempts after its first, which keeps the outer
  consumer), `compose` (the calls of the synthetic composer AGENT `<gate>__compose`, under that
  node id) and `backward`; `summary` and `memory:<key>` are reserved. An entry is
  `{reply: <message>}`, a yian.message.v1 assistant message, with an optional `usage`; or
  `{chunks: [<chunk>, ...]}` whose last chunk is `{finish: <reason or null>}`; or
  `{chunks: [...], raise: {class, message, status}}` for a failure after the chunks
  (`chunks: []`: before any output; `status` optional, the provider's HTTP status). The call
  fails with an error of class `class` and message `message`: as for a stub tool, `TimeoutError`
  and `ConnectionError` are those classes, and any other name is a plain error with that class
  name.
  Any entry may add `request: {messages: [<message>, ...], tools: [<tool spec>, ...], schema:
  <JSON Schema>}` (Python's schema:
  `convert_to_openai_tool(schema)["function"]["parameters"]`), which the runner compares
  exactly with the call's, failing the case when they differ (`tools` absent: none bound;
  `schema` present: the call asked for output matching it, Python's native structured call).
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
- **`crash_after_record`**: `resume` cases only (above).

`expect.tool_calls` lists, per tool that was called, each call in order as `{args, call_id,
attempt, node}`.
`ask_user` calls from `human` are listed too; their `args` are the request in data form
(`question`, `questions`, `answer_schema`, `title`, `node_id`, `options`; nulls left out).
HUMAN_INPUT never sends `options`; a model's `ask_user` call can.

The format reserves these host keys; a runner that doesn't support one yet fails the case that
uses it:

- **`steering`** and **`stop`**: a steer, interrupt or stop at a point: before model call `n` of
  a node, after chunk `k` of model call `n`, or before tool call `i`.
- **`memory`**: memory backends, their initial contents, and `expect.memory` for the contents
  after.
- **`skills`**: skill directories, given as `files`.

Not in the format yet (each comes as an addition when a case needs it): file values in
`values`/`item`, output schemas for stub tools, `files` for the `other` flow of a fingerprint
case, a clock for `system.now`, signals other than SIGINT, and stub HTTP servers for `cli`
commands that call providers.

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
- Any other class compares by its name as given, a stub's declared class included.
- A row mapping a TS class to a different case name is added by the change that introduces that
  TS class; rows are only added.

**Data form** of an event, stream chunk or `ask_user` request: a mapping of its fields by their
snake_case names, at every depth of yian's own structures (never inside a value the flow
produced, such as an output or an answer); fields that are null are left out at the top level
only; lists and tuples as lists, sets as sorted lists, a failure in failure form, a file as
`{uri, mime, name}`.

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
| `run_started` | null | `inputs` (the run's inputs as recorded), `learnable`, `num_workers`, `defaults`, `effective_llm_by_static_path`, `call_time_llm` |
| `run_resumed` | null | `retry_node_ids`, `num_workers` |
| `run_stopped` (TS only) | null | `reason` (the reason's kind), `node_ids` |
| `node_started` | `node_id` | none |
| `node_step` | `node_id` | `step` |
| `node_succeeded` | `node_id` | `value`, `commit_target` (only when it differs from the node), `forward_capture` |
| `node_routed` | `node_id` | `handle`, `terminal_failure`, `forward_capture` |
| `node_expanded` | `spawner_id` | `seed`, `prune_ids`, `forward_capture` |
| `node_failed` | the failure's node | `failure`, `forward_capture` |
| `memory_commit_requested`, `_succeeded`, `_failed` | null | `memory`, `commit_id`, `attempt_id`, `mutations`, `failure` |
| `backward_started` | null | `attempt_id`, `feedback` |
| `backward_ready` | null | `attempt_id`, `root_result`, `attributed_results`, `proposals` |
| `backward_finished` | null | `attempt_id`, `outcome`, `attributed_results` |

For example:

```yaml
- [run_started, null, {inputs: {q: x}, learnable: false, num_workers: 0, defaults: {MAX_TOTAL_NODES: 20000}}]
- [node_started, fetch]
- [node_step, fetch, {step: {event: failed, attempt: 1, error: {type: TransientToolError, message: busy}, retry_in: 0.5}}]
- [node_succeeded, fetch, {value: F1}]
- [node_failed, place, {failure: {code: node.execute, message: rejected, node: place, frames: [{label: flow.yaml, range: [12, 3, 12, 8]}]}}]
```

- Values are plain values: `run_started.inputs` and `node_succeeded.value` are recorded
  typed, and are written without the kind tag (a file as `{uri, mime, name}`); every other
  value (seeds, captures, feedback) is recorded plain already.
- `defaults`, `effective_llm_by_static_path` and `call_time_llm` are left out when empty or
  absent.
- Dropped: the envelope (`schema`, `run_id`, `seq`); from `run_started`, `flow_fingerprint`,
  `initial_pool` (beyond the inputs), `start_wiring_override`, `memory_snapshots`, `memories` and
  `memory_stacks`; `topology_digest`; and every `evidence` field (the model and tool evaluation
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

**Placeholders.** Ids an implementation makes up are replaced by `<id:1>`, `<id:2>`, ...: a run id
the case didn't give, and the values of `exchange_id`, `commit_id`, `mutation_id` and `note_id`
fields. (This list grows only by fields no committed case holds yet.) The generated run id is the
run's own id. Every occurrence inside a text value is replaced, so a call id `<run id>/fetch`
becomes `<id:1>/fetch`; mapping keys are left alone. The ids are collected from every
projection, expected or not, and numbered in order of first appearance (by position, within one
text), walking the trace, then the events, stream, tool calls (tools in sorted order), output and
backward outcomes, with lists in order and mapping keys in sorted order; projections added later
are walked after the backward outcomes. Since replacement is textual, the TS runner's id source
makes ids that can't occur in other text (UUIDs, for example). A `raises` expectation is compared
as is; a case whose error names the run id gives `run_id`.

**Pooled runs.** When the run's `num_workers` isn't 0, the trace, events, stream and each tool's
calls are first grouped: run-level entries first, then each node's entries, nodes in sorted
order, each group keeping its order. Placeholders are numbered and lists compared after that.
Expectations are written grouped.
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
| `messages` | values in messages spelled in the text form |
| `text` | character counting, ASCII lexer classes, whitespace, `$` at the end of text only |
| `yaml` | YAML 1.2 as the TS parser reads it |
| `dates` | dates and times as text, `system.now` |
| `infinity` | infinity is a float value |
| `llm` | how the TS LLM layer builds requests |
| `code` | TS CODE bodies and their failures |

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
| `run-memory` | run | session, episodic, reflective and profile memories |
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
1 when any case fails.
