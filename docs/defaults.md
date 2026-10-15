# Defaults and limits

Every default and limit yian builds in is listed in the [table](#all-defaults-and-limits)
below, and each one can be changed with an environment variable: `YIAN_` plus the
entry's name. `YIAN_MAX_TOTAL_NODES=20000` raises the node budget from 10,000 to
20,000. The entries fall in three groups:

- **limit**: a language safety limit, such as how deep a value may nest or how many
  live nodes a run may grow to;
- **setting**: the default for a setting a flow author can write. A value written in
  the flow wins; the variable changes only what applies when the flow sets none (an
  AGENT's `retries:`, its `env: {max_tool_iterations: ...}`, a memory's `config:`);
- **input cap**: the largest input yian reads from a file, a server, or the user.

The Python names are in `yian.defaults`: `current()` returns this process's values as
a frozen `Defaults` with one lower-case attribute per entry
(`current().max_total_nodes`), and `ENTRIES` lists each entry with its range.

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
into each AGENT), so checking them first names the variable instead of failing on the
flow fingerprint. The comparison is by effective value: a recorded value equal to the
built-in one matches an unset variable. A difference raises `DefaultsMismatchError` (a
`ValueError`) with one line per variable:

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

## Why the nesting limits have maxima

yian walks values, types, YAML documents, and chains of flow bodies recursively, and
Python's stack is finite. Each nesting limit stops a deep input with a located error
before it can raise `RecursionError`, and its maximum is the deepest level shown to
load, run, and resume with stack to spare:

- `YIAN_MAX_VALUE_NESTING` can only be lowered (its maximum is its default, 100). A
  learnable run logs a CALL binding's expression as a route two JSON levels deep per
  call, and the run log's encoder refuses to write past 255 levels, so a 125-level
  call expression could not be logged.
- `YIAN_MAX_CALL_NESTING` can only be lowered (maximum 50). Loading, running, and
  resuming walk a chain of bodies at about 10 frames per level; 50 levels load, run,
  and resume within Python's default recursion limit, and deeper chains are not
  shown to fit. A chain of `uses:` imports is held to the same limit, since loading
  each file recurses too; the deepest accepted load measured, 50 imports whose last
  file holds an uncalled 50-level `defs:` chain, peaks near 670 frames on Python 3.12.
- `YIAN_MAX_YAML_NESTING` can only be lowered (maximum 200): composing and walking a
  document 200 levels deep stays within Python's default recursion limit.
- `YIAN_MAX_TYPE_NESTING` may be raised to 125, the depth proven for type text: type
  text nested 125 levels still parses, and text nested 150 levels does not. A record
  type in `output:` counts its record levels against the same limit, but one near 125
  levels runs with less stack headroom.

## Integer digits

`YIAN_MAX_INT_DIGITS` defaults to 4,300, Python's own limit on the decimal digits of
an int converted to or from text. A set value may not exceed the interpreter's limit
(`sys.get_int_max_str_digits()`, when the interpreter has one), since yian could not
honour more. The default is not checked against it, so with a lower
`PYTHONINTMAXSTRDIGITS` the interpreter's limit applies first. An int with more digits
is not a yian value: it fails at the first boundary that sees it, and an int literal
in a flow file is a load error; see [Types → Integer size
limit](typing.md#integer-size-limit).

## All defaults and limits

An entry whose valid values say "can only be lowered" has its default as its maximum.

| Variable | Group | Default | Valid values | Meaning |
| --- | --- | ---: | --- | --- |
| `YIAN_MAX_VALUE_NESTING` | limit | 100 | 1 to 100 (can only be lowered) | Deepest nesting a value may have; also bounds expression nesting |
| `YIAN_MAX_TYPE_NESTING` | limit | 100 | 1 to 125 | Deepest nesting a type may have, counting record levels |
| `YIAN_MAX_YAML_NESTING` | limit | 200 | 1 to 200 (can only be lowered) | Deepest nesting of a YAML document yian reads |
| `YIAN_MAX_CALL_NESTING` | limit | 50 | 1 to 50 (can only be lowered) | Deepest chain of call, map, and loop bodies a flow may nest, and of uses: files it may import |
| `YIAN_MAX_TOTAL_NODES` | limit | 10,000 | 1 or more | Most live nodes a run's graph may grow to |
| `YIAN_MAX_SKIP_WORLDS` | limit | 10,000 | 1 or more | Most branch combinations the skip check explores before giving up |
| `YIAN_MAX_INT_DIGITS` | limit | 4,300 | 1 to the interpreter's int digit limit | Most decimal digits an int may have to be a yian value |
| `YIAN_AGENT_RETRIES` | setting | 2 | 0 or more | Extra attempts an AGENT gets to correct an invalid structured reply when the node sets none |
| `YIAN_AGENT_MAX_TOOL_ITERATIONS` | setting | -1 | 1 or more, or -1 for no cap | AGENT tool-call turns when the node sets none; -1 is no cap |
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
