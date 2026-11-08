---
title: "Reference"
description: "Defaults and limits, diagnostic and error codes, and run log records."
---

This page collects the values you look up: defaults and limits, environment variables, where
yian keeps its files, failure and error codes, and the records of a run log.

## Defaults and limits

Every default and limit yian builds in can be changed with an environment variable: `YIAN_` plus
the entry's name. `YIAN_MAX_TOTAL_NODES=20000` raises the node budget from 10,000 to 20,000.

```console
$ YIAN_MAX_TOTAL_NODES=20000 yian run brief.yaml --input topic=climate
```

Each entry is in one of three groups:

| Group | What it bounds |
|---|---|
| limit | A safety limit of the language, such as how deep a value may nest, or how much of a value an error message shows |
| setting | How a node behaves when the flow says nothing. Where a flow can set it (an AGENT's or TOOL's `retries:`), the flow's value wins |
| input cap | The largest input yian reads from a file, a server or the user |

### Setting a value

- A value is a plain decimal integer: `0`, or digits with no leading zero and an optional leading `-`.
  Spaces, `_`, `+`, leading zeros, `-0` and exponents are refused.
- An empty variable counts as unset.
- An entry with no maximum takes values up to 9,007,199,254,740,991.
- yian reads the variables once per process, the first time it needs a value. Changing the
  environment after that has no effect. A host can pass its own environment per load or run
  instead (see [Embedding yian](/embedding/)).

A bad value is an error that names the variable, the value and the range. The CLI prints it and
exits 1, before the command does anything else:

```text
YIAN_MAX_VALUE_NESTING="abc" is not an integer; use an integer from 1 to 100, or unset it
YIAN_MAX_VALUE_NESTING="150" is out of range; use an integer from 1 to 100, or unset it
```

### Runs keep the values they started with

A run records, in its `run_started` record, the values that differ from the built-in ones.
Resuming the run needs the same values. A difference stops the resume with
one line per variable, saying what to set:

```text
YIAN_MAX_TOTAL_NODES is 20000 here, but this run started with 5000; set YIAN_MAX_TOTAL_NODES=5000 to resume it
YIAN_AGENT_RETRIES is 4 here, but this run started with the default 2; unset YIAN_AGENT_RETRIES to resume it
```

A log that recorded a variable this version doesn't know, or a value outside its range, can't be
resumed.

### All defaults and limits

An entry whose valid values say "can only be lowered" has its default as its maximum.

| Variable | Group | Default | Valid values | Meaning |
| --- | --- | ---: | --- | --- |
| `YIAN_MAX_VALUE_NESTING` | limit | 100 | 1 to 100 (can only be lowered) | Deepest nesting a value may have; also bounds expression nesting |
| `YIAN_MAX_TYPE_NESTING` | limit | 100 | 1 to 125 | Deepest nesting a type may have, counting record levels |
| `YIAN_MAX_AGENT_OUTPUT_NESTING` | limit | 16 | 1 to 16 (can only be lowered) | Deepest nesting an AGENT `output:` type may have (each record and list is a level) |
| `YIAN_MAX_YAML_NESTING` | limit | 200 | 1 to 200 (can only be lowered) | Deepest nesting of a YAML document yian reads |
| `YIAN_MAX_YAML_NODES` | limit | 100,000 | 1 or more | Most nodes a YAML document yian reads may expand to, counting each alias as a copy |
| `YIAN_MAX_CALL_NESTING` | limit | 50 | 1 to 50 (can only be lowered) | Deepest chain of call, map, and loop bodies a flow may nest, and of uses: files it may import |
| `YIAN_MAX_TOTAL_NODES` | limit | 10,000 | 1 or more | Most live nodes a run's graph may grow to |
| `YIAN_MAX_SKIP_WORLDS` | limit | 10,000 | 1 or more | Most branch combinations the skip check explores before giving up |
| `YIAN_MAX_SHOWN_VALUE_CHARS` | limit | 200 | 2 or more | Longest rendering of a value or key an error message shows |
| `YIAN_MAX_FAILURE_CHARS` | limit | 10,000 | 1 or more | Longest message or traceback a failure records; a longer one is cut and says how much was cut |
| `YIAN_AGENT_RETRIES` | setting | 2 | 0 or more | Extra attempts an AGENT gets to correct an invalid structured reply when the node sets none |
| `YIAN_AGENT_MAX_TOOL_ITERATIONS` | setting | -1 | 1 or more, or -1 for no cap | AGENT tool-call turns when the node sets none; -1 is no cap |
| `YIAN_TOOL_RETRIES` | setting | 0 | 0 or more | Extra attempts a TOOL node gets after a failed tool call when the node sets none |
| `YIAN_TOOL_RETRY_DELAY_SECONDS` | setting | 1 | 0 or more | Base wait in seconds before a TOOL node's first retry; each later wait doubles, up to the cap |
| `YIAN_TOOL_RETRY_MAX_DELAY_SECONDS` | setting | 30 | 0 or more | Longest wait in seconds before a TOOL node retries a failed tool call |
| `YIAN_MEMORY_LOCK_TIMEOUT_SECONDS` | setting | 30 | 0 or more | Longest wait in seconds for a memory's lock |
| `YIAN_MEMORY_MAX_CHANGE_IDS` | setting | 10,000 | 1 or more | Most change ids a memory keeps to answer a repeated change |
| `YIAN_MEMORY_MAX_UNDO` | setting | 100 | 0 or more | Most changes a memory can undo |
| `YIAN_FILE_MEMORY_MAX_FILE_CHARS` | setting | 100,000 | 1 or more | Most characters one file of a FileMemory may hold |
| `YIAN_FILE_MEMORY_MAX_TOTAL_CHARS` | setting | 1,000,000 | 1 or more | Most characters all files of a FileMemory may hold |
| `YIAN_FILE_MEMORY_MAX_FILES` | setting | 1,000 | 1 or more | Most files a FileMemory may hold |
| `YIAN_FILE_MEMORY_MAX_DEPTH` | setting | 4 | 0 or more | Most folders deep a FileMemory file may sit |
| `YIAN_FILE_MEMORY_MAX_RECALL_CHARS` | setting | 8,000 | 1 or more | Most characters of a FileMemory's recall |
| `YIAN_FILE_MEMORY_MAX_READ_CHARS` | setting | 20,000 | 1 or more | Most characters one FileMemory list, read or search result shows |
| `YIAN_FILE_MEMORY_MAX_RESULTS` | setting | 100 | 1 or more | Most entries a FileMemory's `list` shows, and lines its `search` shows |
| `YIAN_VECTOR_MEMORY_MAX_ITEMS` | setting | 20,000 | 1 or more | Most items a VectorDBMemory may hold |
| `YIAN_VECTOR_MEMORY_MAX_K` | setting | 50 | 1 or more | Most items a VectorDBMemory's `search` returns and its `list` shows |
| `YIAN_VECTOR_MEMORY_MAX_TEXT_CHARS` | setting | 8,000 | 1 or more | Most characters of one VectorDBMemory item's text and fields, of a retire's reason, or of a query |
| `YIAN_VECTOR_MEMORY_MAX_RECALL_CHARS` | setting | 4,000 | 1 or more | Most characters of a VectorDBMemory's recall |
| `YIAN_MCP_CONNECT_TIMEOUT_SECONDS` | setting | 30 | 1 or more | Longest wait in seconds for an MCP server to start and list its tools |
| `YIAN_MCP_CALL_TIMEOUT_SECONDS` | setting | 300 | 1 or more | Longest wait in seconds for one MCP tool call |
| `YIAN_MODEL_RETRIES` | setting | 6 | 0 or more | Extra attempts a model call gets after a transport failure (a network error, 5xx, 429 or timeout) |
| `YIAN_MODEL_FIRST_RESPONSE_TIMEOUT_SECONDS` | setting | 240 | 1 or more | Longest wait in seconds from sending a model request to the first bytes of its response |
| `YIAN_MODEL_IDLE_TIMEOUT_SECONDS` | setting | 240 | 1 or more | Longest wait in seconds between bytes of a streamed model response |
| `YIAN_REQUEST_TIMEOUT_SECONDS` | setting | 30 | 1 or more | Longest wait in seconds for a short request (model listing, sign-in calls) |
| `YIAN_EMBEDDING_TIMEOUT_SECONDS` | setting | 60 | 1 or more | Longest wait in seconds for one embeddings request |
| `YIAN_EMBEDDING_BATCH_ITEMS` | setting | 64 | 1 or more | Most texts sent in one embeddings request; more go in several requests |
| `YIAN_MAX_AUTH_RESPONSE_BYTES` | input cap | 1,048,576 | 1 or more | Largest response body read from a sign-in server |
| `YIAN_MAX_AUTH_STORE_BYTES` | input cap | 1,048,576 | 1 or more | Largest auth store file read or written |
| `YIAN_MAX_OAUTH_CALLBACK_BYTES` | input cap | 16,384 | 1 or more | Largest request header block read from the sign-in callback |
| `YIAN_MAX_MODEL_LIST_BYTES` | input cap | 16,777,216 | 1 or more | Largest model listing read from a provider |
| `YIAN_MAX_MODEL_LIST_PAGES` | input cap | 100 | 1 or more | Most pages read from a provider's model listing |
| `YIAN_MAX_MCP_MESSAGE_BYTES` | input cap | 16,777,216 | 1 or more | Largest message read from an MCP server: a stdio line, an SSE event or an HTTP reply body |
| `YIAN_MAX_MODEL_EVENT_BYTES` | input cap | 16,777,216 | 1 or more | Largest event read from a model's reply stream: an SSE event's data or one NDJSON line |
| `YIAN_MAX_EMBEDDING_REPLY_BYTES` | input cap | 33,554,432 | 1 or more | Largest reply read from an embeddings request |
| `YIAN_MAX_EVENT_BYTES` | input cap | 1,048,576 | 1 or more | Largest event sent to a run's inbox, host call's pending step or completed answer, in run-log JSON bytes |
| `YIAN_MAX_BASE_URL_CHARS` | input cap | 2,048 | 1 or more | Longest provider base URL accepted |
| `YIAN_MAX_CONNECTION_HEADERS` | input cap | 32 | 1 or more | Most extra request headers a provider connection may send |
| `YIAN_MAX_HEADER_VALUE_CHARS` | input cap | 8,192 | 1 or more | Longest connection header name or value accepted |
| `YIAN_MAX_ORGANIZATION_CHARS` | input cap | 256 | 1 or more | Longest OpenAI organization or project id accepted |
| `YIAN_MAX_SKILL_FILE_BYTES` | input cap | 262,144 | 1 or more | Largest SKILL.md or bundled skill file returned to a model |
| `YIAN_MAX_SKILL_NAME_CHARS` | input cap | 64 | 1 or more | Longest skill name accepted |
| `YIAN_MAX_SKILL_DESCRIPTION_CHARS` | input cap | 1,024 | 2 or more | Longest skill description shown in the skill listing |
| `YIAN_MAX_QUESTIONS` | input cap | 4 | 1 or more | Most questions an `ask` may put to the user at once |
| `YIAN_MAX_QUESTION_OPTIONS` | input cap | 6 | 1 or more | Most options a question to the user may offer |
| `YIAN_MAX_QUESTION_OPTION_CHARS` | input cap | 80 | 2 or more | Longest question option shown to the user |
| `YIAN_MAX_KEY_PROMPTS` | input cap | 3 | 1 or more | Times the CLI asks for an API key before giving up |

- TOOL retries wait `YIAN_TOOL_RETRY_DELAY_SECONDS`, then twice that, and so on, up to
  `YIAN_TOOL_RETRY_MAX_DELAY_SECONDS`. Each wait is picked at random between half and all of its
  step.

## Environment variables

yian reads these variables:

| Variable | What it does |
|---|---|
| `YIAN_<NAME>` | Changes one default or limit; there is one per row of the [table above](#all-defaults-and-limits) |
| `YIAN_DEFAULT_PROVIDER` | The provider for agents that set none, ahead of the saved default login |
| `YIAN_DEFAULT_MODEL` | The model for agents that set none, ahead of the saved login's model. Ignored unless a default provider is set |
| `YIAN_DEFAULT_EMBEDDING` | The embedding model, as `provider/model`, a new VectorDB memory is made with when `yian run` gets no `--embedding`, ahead of the one `yian config embedding` saved |
| `YIAN_AUTH_FILE` | The saved-logins file to use (see [Files and folders](#files-and-folders)) |
| `HOME` | Your home folder, for `~/.yian`, `~/.config` and the skill folders. When unset or empty, yian asks the system |
| `XDG_CONFIG_HOME` | The folder that holds `yian/auth.json`, in place of `~/.config` |
| `NO_COLOR` | Set to anything but empty, the CLI prints no colour |
| `SSH_CONNECTION` | When set, `yian auth openai` offers device sign-in first |

Provider API keys and base URLs, such as `ANTHROPIC_API_KEY`, are listed on
[Models and providers](/models/). `yian providers` prints each provider's key variable.

## Files and folders

| Path | What it holds |
|---|---|
| `~/.config/yian/auth.json` | Your saved provider logins and default embedding model. `$YIAN_AUTH_FILE`, or `$XDG_CONFIG_HOME/yian/auth.json`, takes its place. Written with mode `0600`, beside a `.lock` file, in a folder that is yours with mode `0700` and not a symbolic link |
| `~/.yian/mcp.json` | MCP servers `yian run` reads, when a flow needs a tool yian doesn't have |
| `~/.yian/runs/<run_id>/log.jsonl` | A run's log, written by `yian run` and `yian chat` |
| `~/.yian/runs/<run_id>/log.jsonl.lock` | The lock the process writing the log holds |
| `~/.yian/runs/<run_id>/media/` | Media files the run keeps, named by their SHA-256 |
| `~/.yian/runs/index.sqlite` | The run index `yian status` lists. It is rebuilt from the logs, so you can delete it |
| `~/.yian/memory/<project>/<flow id>/<input>` | The default memory `yian run` keeps for a flow's FileMemory input, a folder (see [CLI](/cli/#memory)) |
| `~/.yian/memory/<project>/<flow id>/<input>.db` | The default store `yian run` keeps for a VectorDBMemory input |
| `~/.claude/skills`, `~/.agents/skills` | Your skills, read when an agent grants skills |
| `.claude/skills`, `.agents/skills` | Project skills, from the current folder up to the git root |
| `<memory folder>/.yian/` | A FileMemory's identity, journal, lock and temp files (see [Memory](/memory/#filememory)) |
| `<store path>`, `<store path>.lock` | A VectorDBMemory's SQLite store and its lock, with mode `0600` (see [Memory](/memory/#vectordbmemory)) |

`~/.yian`, `~/.yian/runs` and each run folder have mode `0700`; a log has mode `0600`. Logs are
not encrypted and can hold prompts, replies and tool data, so keep them private.

## Failure codes

A failed run has one failure, and its `code` is one of these. A failed run is final: resuming its
log prints the same failure, so "retry" below means a new run.

| Code | When | What to do |
|---|---|---|
| `binding.evaluate` | A node input's `${...}` binding or default can't be evaluated, or a required input is unbound | Fix the flow or its inputs |
| `binding.type` | A tagged binding (`!int ${...}`) holds a value that breaks its tag | Fix the flow or its inputs |
| `contract.input` | A value given to a node input breaks its declared type, or a MAP `over:`, a `sleep`'s `until:` or an AGENT's `llm_config` value has the wrong shape | Fix the flow or its inputs |
| `assertion.failed` | An `asserts:` condition is false | Treat the result as rejected |
| `condition.evaluate` | A LOOP `while:` or `until:`, a CASE `when:`, or an `asserts:` condition can't be evaluated | Fix the flow or its inputs |
| `node.execute` | A node's body failed and no finer code fits: a CODE error, a tool error, a model error such as a bad request or an unknown model | Read the message; a retry can help with a passing error |
| `contract.node_result` | A node returned something that isn't an output, a route or a growth | Report a bug in that node |
| `contract.output` | A node's or the flow's output doesn't match its declared type | Fix the flow, or the code or tool that made the value |
| `contract.route` | A node returned a route handle that isn't a non-empty string | Report a bug in that node |
| `contract.growth` | A CALL, MAP or LOOP would grow the graph past `YIAN_MAX_TOTAL_NODES` | Bound the fan-out, or raise the limit |
| `graph.terminal_unreachable` | A route skipped every path to the flow's output | Fix the graph |
| `loop.max_exceeded` | A `while:` or `until:` LOOP with `max: M` would start its body run M+1 | Raise `max:`, or treat it as the loop ending unmet |
| `agent.max_tool_iterations` | An AGENT reached its `max_tool_iterations` and its last reply still asked for tools | Raise the cap, or simplify the task |
| `agent.output_invalid` | An AGENT's structured reply still breaks its `output:` type after its `retries:` | Loosen the type, or improve the prompt |
| `agent.interrupted` | An AGENT was interrupted and its output type doesn't allow null | Declare `output: Optional(...)` and route on null |
| `tool.timeout` | An `ask` timed out with no answer and its output type doesn't allow null | Declare `output: Optional(...)`, or give it longer |
| `provider.auth` | The provider refused the agent's credentials, or yian couldn't supply them | Fix the login (`yian auth <provider>`) or the key |
| `provider.rate_limit` | The provider's rate limit (HTTP 429) outlasted yian's model-call retries | Retry later, or run fewer calls at once |
| `provider.quota` | The account's quota or credit is used up | Add credit or change plan |
| `provider.unavailable` | The provider was unreachable or failing (a 5xx, a timeout, a dropped connection) past yian's retries | Retry later, or switch provider |
| `provider.context_length` | The request is longer than the model's context window | Shorten the input, or pick a larger model |
| `provider.refused` | The model or the provider's content filter declined to answer | Change the input |

A `loop.max_exceeded` failure is on the LOOP node. A failure inside a loop body keeps its own
code, and a `times:` loop never raises it.

## CLI error codes

A `--json` command that fails without its own result prints an error document whose
`error.code` is one of these. `yian check` and `yian skills check` use some of them for each item.

| Code | When |
|---|---|
| `compose.load` | The flow doesn't load, or its file can't be read |
| `expr.nullable_field_read` | The flow doesn't load because it reads a field of a value that may be null. The message names the fix |
| `compose.outside_root` | `yian check --root`: a flow or a `uses:` import is outside the root |
| `defaults.invalid` | A `YIAN_*` variable holds a value it doesn't take (any command) |
| `run.setup` | The run couldn't start or resume: skills or MCP setup, a missing or changed provider key, the run log (missing, damaged, locked), or an input that fails its check. Also a log or index `yian status` can't read |
| `run.cancelled` | You cancelled the run at a prompt, input ended there, or the host cancelled the run |
| `auth.failed` | `yian auth` failed: options that don't go together, an unusable file, a login that isn't saved, or a failed sign-in or model check |
| `models.failed` | `yian models` couldn't list the models. `error.status` is the HTTP status, or `null` |
| `models.timeout` | The listing took longer than `--timeout`. `error.status` is `null` |
| `skill.invalid` | `yian skills check`: a directory holds no valid `SKILL.md` |
| `memory.conflict` | `yian memory`: a file or item changed since it was read, so the edit or undo changed nothing |
| `memory.refused` | `yian memory`: the change isn't allowed, such as an item that doesn't exist, an undo of a change no longer kept, or a reset while a run is running |
| `memory.cancelled` | `yian memory erase`, `reset` or `edit --delete`: you said no |
| `memory.failed` | `yian memory` failed: the memory can't be read or isn't one, or the editor failed. A memory that isn't there yet is a usage error, with no document |

## Run log records

A run log is a JSON Lines file: one record per line, appended and never rewritten. Each line is
compact JSON with its keys sorted:

```json
{"at":"2026-10-27T08:47:40.404Z","kind":"node_started","payload":{"node_id":"greet"},"run_id":"31fc75ac4c0640de867d47f59f928ea3","schema":29,"seq":8}
```

| Key | What it holds |
|---|---|
| `schema` | The record format, `29`. yian reads only this one. A log of schema 10 to 28 reads as `run log schema N predates schema 29 and is not supported`, any other as `unsupported run-record schema: N` |
| `run_id` | The run's id |
| `seq` | The record's place in the log, from 0 |
| `at` | When the record was appended, in UTC with milliseconds. Use `seq`, not `at`, for order |
| `kind` | The record kind, below |
| `payload` | The record's data |

| Kind | What it records |
|---|---|
| `run_started` | The run begins: its flow files and code root, settings, model choices, granted tools and skills, and the `YIAN_*` values that differ from the built-in ones |
| `run_resumed` | A resume begins: the unfinished nodes it retries, and its worker count |
| `run_stopped` | The host stopped the run while nodes were running. The run can be resumed |
| `run_waiting` | Every unfinished node waits, and the run holds no worker |
| `run_cancelled` | The host cancelled the run. It is over |
| `run_failed` | The run failed: nodes still wait, but nothing can reach its output |
| `node_started` | A node starts |
| `node_step` | One step a running node recorded, such as a model call, which a retry of the node reads back |
| `node_waiting` | A node waits on calls: host calls, a take or a sleep |
| `node_woken` | A waiting node runs again, naming the calls that timed out |
| `pool_commit` | A value written to the run: an input, a node's output, or a message an inbox received |
| `node_succeeded` | A node succeeded, naming the `pool_commit` that holds its output |
| `node_routed` | A node, such as a CASE, chose a route |
| `node_expanded` | A CALL, MAP or LOOP added its child nodes to the graph |
| `node_failed` | A node failed, with its failure. The run is over |
| `inbox_closed` | An inbox was closed; it takes no more messages |
| `inbox_withdrawn` | Messages nobody had read yet were taken back from an inbox |
| `interrupted` | The agents reading an inbox were interrupted |

`yian status` reads these records to say where a run stands. See [How yian works](/how-it-works/)
for how yian replays a log to recover a run.
