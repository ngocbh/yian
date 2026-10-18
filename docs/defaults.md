# Defaults and limits

Every default and limit yian builds in is listed in the [table](#all-defaults-and-limits)
below, and each one can be changed with an environment variable: `YIAN_` plus the
entry's name. `YIAN_MAX_TOTAL_NODES=20000` raises the node budget from 10,000 to
20,000. The entries fall in three groups:

- **limit**: a language safety limit, such as how deep a value may nest or how many
  live nodes a run may grow to, or a bound on what yian writes in a message, such as
  how much of a value an error message shows;
- **setting**: the default for how a node or memory behaves. Where a flow can write
  the setting (an AGENT's or TOOL's `retries:`, its `env: {max_tool_iterations: ...}`,
  a memory's `config:`), the written value wins and the variable changes only what
  applies when the flow sets none. The rest, such as the TOOL retry delays and the
  reflective prompt budgets, have no flow-level form, so only the variable changes them;
- **input cap**: the largest input yian reads from a file, a server, or the user.

The Python names are in `yian.defaults`: `current()` returns this process's values as
a frozen `Defaults` with one lower-case attribute per entry
(`current().max_total_nodes`), and `ENTRIES` lists each entry with its range.

Changed (TS yian): `current()` comes from `ts/src/defaults.ts` and has one camelCase key
per entry (`current().maxTotalNodes`). It reads the process environment (`processEnv`);
`read(env)` takes any `Env` accessor and returns the values it sets.

## Setting a value

Set the variable before the process starts:

```bash
YIAN_MAX_TOTAL_NODES=20000 yian run flow.yaml --input topic=climate
```

yian reads every variable once per process, the first time it needs a value, and
keeps the result. The CLI reads them when it starts; in Python, `load_flow`,
`run_flow`, and `resume_flow` read them before doing any work. Changing `os.environ`
after that has no effect, so a Python host sets the variables before its first yian
call. An empty variable counts as unset.

A value must be a plain decimal integer: `0`, or digits with no leading zero and an
optional `-`, at most 18 digits. Spaces, `_`, `+`, leading zeros, `-0`, and exponents
are refused. A malformed or out-of-range value raises `DefaultsError` (a
`ValueError`) naming the variable, the value, and the range; the CLI prints it and
exits 1:

```text
YIAN_MAX_VALUE_NESTING='abc' is not an integer; use an integer from 1 to 100, or unset it
YIAN_MAX_VALUE_NESTING='150' is out of range; use an integer from 1 to 100, or unset it
```

An entry with no maximum accepts any value of up to 18 digits. A failed read is not
kept: every later call raises again until the variable is fixed.

Changed (TS yian): the message quotes the value as JSON text
(`YIAN_MAX_VALUE_NESTING="abc" is not an integer; use an integer from 1 to 100, or unset it`),
and a long value is cut by code points. An entry with no maximum accepts values up to
9,007,199,254,740,991 (JavaScript's largest safe integer), not any value of up to 18 digits.
`DefaultsError` and `DefaultsMismatchError` are `RangeError`s.

## Runs record the values they started with

A run's `RunStarted` record keeps the values that differ from the built-in ones in
its payload's `defaults` key, keyed by entry name and sorted, for example
`"defaults": {"MAX_TOTAL_NODES": 20000}`.

A run with every built-in value writes no `defaults` key, so its log is byte for
byte what it was before the field existed, and a log without the key counts as all
built-in values. Every entry is recorded, input caps included.

Resuming a run (`resume_flow`, `yian run --resume`) and replaying it for backward
learning compare the recorded values with this process's values before checking the
flow fingerprint. Some values change the loaded flow (`YIAN_AGENT_RETRIES` is baked
into each AGENT and `YIAN_TOOL_RETRIES` into each TOOL), so checking them first names
the variable instead of failing on the flow fingerprint. The comparison is by
effective value: a recorded value equal to the built-in one matches an unset
variable. A difference raises `DefaultsMismatchError` (a `ValueError`) with one line
per variable:

```text
YIAN_MAX_TOTAL_NODES is 20000 here, but this run started with 5000; set YIAN_MAX_TOTAL_NODES=5000 to resume it
YIAN_MAX_TOTAL_NODES is 20000 here, but this run started with the default 10000; unset YIAN_MAX_TOTAL_NODES to resume it
```

A log that recorded an entry this version of yian doesn't know can't be resumed by
it (`this run started with YIAN_<NAME>, which this version of yian does not know`),
and neither can a log that recorded a value outside the entry's range
(`this run recorded an invalid value ...`).

The node budget is checked by the run's state, which reads the recorded value
(`RunState.run_defaults`), never the process's, so replaying a log depends only on
the log.

## Derived bounds

Two bounds follow `YIAN_MAX_VALUE_NESTING` and have no variable of their own:

- **Expression nesting** equals value nesting (100 by default); see
  [expression nesting limit](expressions.md#expression-nesting-limit).
- **Memory JSON nesting** is value nesting plus 28 levels (128 by default), room for
  the episode and feedback wrappers around a run value. Its error ends
  `(derived from YIAN_MAX_VALUE_NESTING)`. The bound also guards stored memory, so a
  memory store written under a higher value limit may not load under a lower one.

Changed (TS yian): memory JSON nesting doesn't exist. Stores are read without a depth
limit, and stored values are checked against value nesting.

## Why the nesting limits have maxima

yian walks values, types, YAML documents, and chains of flow bodies recursively, and
Python's stack is finite. Each nesting limit stops a deep input with a located error
before it can raise `RecursionError`, and its maximum is the deepest level shown to
load, run, and resume with stack to spare:

- `YIAN_MAX_VALUE_NESTING` can only be lowered (its maximum is its default, 100). A
  learnable run logs a CALL binding's expression as a route two JSON levels deep per
  call, and the run log's encoder refuses to write past 255 levels, so a 125-level
  call expression could not be logged. Changed (TS yian): yian's JSON writer and reader
  don't recurse, so the log encoder reason is gone; the limit stays for the value walkers
  not yet ported.
- `YIAN_MAX_CALL_NESTING` can only be lowered (maximum 50). Loading, running, and
  resuming walk a chain of bodies at about 10 frames per level; 50 levels load, run,
  and resume within Python's default recursion limit, and deeper chains are not
  shown to fit. A chain of `uses:` imports is held to the same limit, since loading
  each file recurses too; the deepest accepted load measured, 50 imports whose last
  file calls a 50-level chain, peaks near 735 frames on Python 3.10 and 3.12.
  Changed (TS yian): the loader, run, and resume are not yet ported, so the maximum stays
  until they are measured.
- `YIAN_MAX_YAML_NESTING` can only be lowered (maximum 200): composing and walking a
  document 200 levels deep stays within Python's default recursion limit.
  Changed (TS yian): the `yaml` package composes about 500 levels on Node but 5000 on
  Bun's main thread; the limit keeps the deepest accepted document the same on both.
- `YIAN_MAX_TYPE_NESTING` may be raised to 125, the depth proven for type text: type
  text nested 125 levels still parses, and text nested 150 levels does not. A record
  type in `output:` counts its record levels against the same limit. Copies of a flow
  share its types, so record depth adds no stack at run setup. The limit also bounds
  each typedef's levels, the length of a chain of typedefs naming each other, and a
  type with its typedef names expanded. Changed (TS yian): the TS type walkers handle at
  least 500 levels on Node and Bun; the maximum stays 125 until the walkers not yet ported
  are measured.
- `YIAN_MAX_AGENT_OUTPUT_NESTING` can only be lowered (maximum 16). Building a model's
  structured reply schema is recursive. The costliest shape is an `Optional(Union(...))`
  field at each level, about 28 frames per level on Python 3.12 and more on 3.11. With
  300 frames below Python's default recursion limit, under 50 nested calls, that shape
  passes 18 levels on Python 3.11 and 23 on 3.12 through every structured-output path.
  The maximum is the smallest passing depth minus 2, a margin for Python versions and
  library releases that weren't measured. Under the deepest accepted load (50 `uses:`
  imports whose last file calls a 50-level chain), an AGENT at 16 levels builds its
  schema well above the load's peak. Changed (TS yian): the schema builder is not yet
  ported; the maximum also stays because providers limit the nesting of a structured-output
  schema, so a deeper `output:` type is of no use.

Changed (TS yian): the limits keep the deepest accepted input the same on Node and Bun,
whose stacks differ. Each maximum is re-measured once every recursive walker is ported.

`YIAN_MAX_YAML_NODES` has no maximum, since counting a document's nodes takes no stack;
raising it raises load time and run-log size in proportion.

## Integer digits

`YIAN_MAX_INT_DIGITS` defaults to 4,300, Python's own limit on the decimal digits of
an int converted to or from text. A set value may not exceed the interpreter's limit
(`sys.get_int_max_str_digits()`, when the interpreter has one), since yian could not
honour more. The default is not checked against it, so with a lower
`PYTHONINTMAXSTRDIGITS` the interpreter's limit applies first. An int with more digits
is not a yian value: it fails at the first boundary that sees it, and an int literal
in a flow file is a load error; see [Types → Integer size
limit](typing.md#integer-size-limit).

Changed (TS yian): `YIAN_MAX_INT_DIGITS` doesn't exist; ints have no size limit.

## TS yian only

These entries exist only in TS yian; Python yian neither reads nor records them.

| TS yian variable | Group | Default | Valid values | Meaning |
| --- | --- | ---: | --- | --- |
| `YIAN_MODEL_RETRIES` | setting | 6 | 0 or more | Extra attempts a model call gets after a transport failure (a network error, 5xx, 429 or timeout) |
| `YIAN_MODEL_FIRST_RESPONSE_TIMEOUT_SECONDS` | setting | 240 | 1 or more | Longest wait in seconds from sending a model request to the first bytes of its response |
| `YIAN_MODEL_IDLE_TIMEOUT_SECONDS` | setting | 240 | 1 or more | Longest wait in seconds between bytes of a streamed model response |
| `YIAN_REQUEST_TIMEOUT_SECONDS` | setting | 30 | 1 or more | Longest wait in seconds for a short request (model listing, sign-in calls) |
| `YIAN_MAX_MCP_MESSAGE_BYTES` | input cap | 16,777,216 | 1 or more | Largest message read from an MCP server |

A value of 300 or more for either model timeout takes effect only on Bun: Node's fetch
has its own 300-second limits. TS yian has no restart limit (a host that resumes runs
automatically sets its own) and no int digit limit.

## All defaults and limits

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
| `YIAN_MAX_INT_DIGITS` | limit | 4,300 | 1 to the interpreter's int digit limit | Most decimal digits an int may have to be a yian value |
| `YIAN_MAX_SHOWN_VALUE_CHARS` | limit | 200 | 2 or more | Longest rendering of a value or key an error message shows |
| `YIAN_AGENT_RETRIES` | setting | 2 | 0 or more | Extra attempts an AGENT gets to correct an invalid structured reply when the node sets none |
| `YIAN_AGENT_MAX_TOOL_ITERATIONS` | setting | -1 | 1 or more, or -1 for no cap | AGENT tool-call turns when the node sets none; -1 is no cap |
| `YIAN_TOOL_RETRIES` | setting | 0 | 0 or more | Extra attempts a TOOL node gets after a failed tool call when the node sets none |
| `YIAN_TOOL_RETRY_DELAY_SECONDS` | setting | 1 | 0 or more | Base wait in seconds before a TOOL node's first retry; each later wait doubles, up to the cap |
| `YIAN_TOOL_RETRY_MAX_DELAY_SECONDS` | setting | 30 | 0 or more | Longest wait in seconds before a TOOL node retries a failed tool call |
| `YIAN_MCP_CONNECT_TIMEOUT_SECONDS` | setting | 30 | 1 or more | Longest wait in seconds for an MCP server to start and list its tools |
| `YIAN_MCP_CALL_TIMEOUT_SECONDS` | setting | 300 | 1 or more | Longest wait in seconds for one MCP tool call |
| `YIAN_EPISODIC_MAX_EPISODES` | setting | 100 | 1 or more | Episodes an episodic memory keeps when it sets no limit |
| `YIAN_EPISODIC_RECALL_LIMIT` | setting | 10 | 1 or more | Episodes an episodic memory recalls when it sets no limit |
| `YIAN_REFLECTIVE_MAX_ITEMS` | setting | 100 | 1 or more | Lessons a reflective memory keeps when it sets no limit |
| `YIAN_REFLECTIVE_RECALL_LIMIT` | setting | 20 | 1 or more | Lessons a reflective memory recalls when it sets no limit |
| `YIAN_REFLECTIVE_MAX_LESSON_CHARS` | setting | 8,000 | 256 or more | Characters of current lessons shown to the reflection model |
| `YIAN_REFLECTIVE_MAX_EPISODE_CHARS` | setting | 20,000 | 256 or more | Characters of the episode shown to the reflection model |
| `YIAN_REFLECTIVE_MAX_LIMIT_CHARS` | setting | 256 | 256 or more | Characters of the memory limits shown to the reflection model |
| `YIAN_SESSION_HISTORY_BUDGET_CHARS` | setting | 32,000 | 1 or more | Characters of history a session memory sends to the model |
| `YIAN_SESSION_RECENT_TURNS` | setting | 4 | 0 or more | Recent turns a session memory always keeps whole |
| `YIAN_PROFILE_MAX_CHARS` | setting | 32,000 | 1 or more | Characters a profile memory may hold |
| `YIAN_PROFILE_INPUT_BUDGET_CHARS` | setting | 24,000 | 1 or more | Characters of run material sent to a profile memory's learner |
| `YIAN_MAX_AUTH_RESPONSE_BYTES` | input cap | 1,048,576 | 1 or more | Largest response body read from a sign-in server |
| `YIAN_MAX_AUTH_STORE_BYTES` | input cap | 1,048,576 | 1 or more | Largest auth store file read or written |
| `YIAN_MAX_OAUTH_CALLBACK_BYTES` | input cap | 16,384 | 1 or more | Largest request header block read from the sign-in callback |
| `YIAN_MAX_MODEL_LIST_BYTES` | input cap | 16,777,216 | 1 or more | Largest model listing read from a provider |
| `YIAN_MAX_BASE_URL_CHARS` | input cap | 2,048 | 1 or more | Longest provider base URL accepted |
| `YIAN_MAX_CONNECTION_HEADERS` | input cap | 32 | 1 or more | Most extra request headers a provider connection may send |
| `YIAN_MAX_HEADER_VALUE_CHARS` | input cap | 8,192 | 1 or more | Longest connection header name or value accepted |
| `YIAN_MAX_ORGANIZATION_CHARS` | input cap | 256 | 1 or more | Longest OpenAI organization or project id accepted |
| `YIAN_MAX_SKILL_FILE_BYTES` | input cap | 262,144 | 1 or more | Largest SKILL.md or bundled skill file returned to a model |
| `YIAN_MAX_SKILL_NAME_CHARS` | input cap | 64 | 1 or more | Longest skill name accepted |
| `YIAN_MAX_SKILL_DESCRIPTION_CHARS` | input cap | 1,024 | 2 or more | Longest skill description shown in the skill listing |
| `YIAN_MAX_QUESTIONS` | input cap | 4 | 1 or more | Most questions a HUMAN_INPUT node may ask at once |
| `YIAN_MAX_QUESTION_OPTIONS` | input cap | 6 | 1 or more | Most options a question to the user may offer |
| `YIAN_MAX_QUESTION_OPTION_CHARS` | input cap | 80 | 2 or more | Longest question option shown to the user |
| `YIAN_MAX_KEY_PROMPTS` | input cap | 3 | 1 or more | Times the CLI asks for an API key before giving up |

Episodic and reflective memories record their limits in each snapshot and refuse a
changed request, so their defaults apply only to a store that has no recorded limits
yet.
