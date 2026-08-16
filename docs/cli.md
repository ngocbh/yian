# The `ac` CLI

`ac` runs a flow file: it loads the YAML, gathers inputs for a fresh run or
recovers them from a durable log, drives the flow to a terminal state, and
prints the output.

```console
ac run FLOW.yaml [--inputs inputs.json] [--input k=v]... [--log PATH] [-w N] [--provider NAME] [--model NAME] [--quiet] [--verbose] [--engine-trace]
ac run FLOW.yaml --resume PATH [-w N] [--quiet] [--verbose] [--engine-trace]
```

## Durable logs and recovery

Every fresh CLI run writes an authoritative JSONL log and reports its path on
stderr, including under `--quiet`:

```console
# fresh log: ~/.agent_composer/runs/<run_id>.jsonl
ac run flow.yaml --input topic=climate

# choose another new path (it must not already exist)
ac run flow.yaml --input topic=climate --log ./climate.jsonl

# recover a terminal or unfinished prefix with the same authored flow
ac run flow.yaml --resume ./climate.jsonl
```

The CLI creates `~/.agent_composer` and its `runs/` directory with mode `0700`;
the runtime creates each log exclusively with mode `0600`. A log can retain
canonical inputs, node outputs (including model, tool, and human data), resolved
model configuration, and failure tracebacks. Treat it as sensitive data: do not
commit or publish it.

Recovery reads the original inputs, run identity, and resolved model map from the
log. `--resume` therefore rejects `--input`, `--inputs`, `--provider`, `--model`,
and `--log` before input prompts or provider setup. An already terminal log is
read-only and prints its recorded result. An unfinished log appends a resume
segment and retries any node with a durable start but no durable outcome.

Expected host-side failures while creating or recovering a log (for example a
corrupt, mismatched, or locked log) are reported as concise errors. Pass
`--engine-trace` when debugging the engine to include the Python traceback.

`ac chat` uses the same default directory and reports one new path for its initial
run. If a failed turn restarts the immutable chat flow from the last completed
transcript, that restart gets a separate log rather than reusing the failed run.

## Supplying inputs

A fresh flow declares typed `input:` fields. Explicit inputs layer in this order
(later overrides earlier):

1. **`--inputs file.json`** — a JSON object of `{ "key": value, ... }`.
2. **`--input k=v`** — one input, repeatable. Values arrive as strings and are
   coerced, where supported, against each input's declared type at the run boundary.
   A default applied by START was already validated at load and is not string-coerced.

When stdin is an interactive TTY, `ac` then prompts for each declaration still
missing.

```console
# all from flags
ac run examples/decision-brief.yaml --input question="Adopt a monorepo?" --input audience="execs"

# from a JSON file, with one override
ac run flow.yaml --inputs base.json --input audience="execs"

# in an interactive terminal, supply nothing and get prompted for each missing input
ac run examples/hello.yaml
```

### The "Running flow" banner

At the start of every run (unless `--quiet`), `ac` prints a boxed banner to
**stderr** identifying what it is about to run — the flow's `name:`, its
`version:` (if set), and its `description:` — so you can confirm you picked the
right file:

```console
╭──────────────────────────────────────────────────────────╮
│ Running flow: debate  (version: v2)                        │
│ Description: Frame a question, argue both sides, then decide. │
╰──────────────────────────────────────────────────────────╯
```

A flow with no `name`/`description`/`version` prints no banner.

### The interactive prompt

When stdin is an interactive TTY, `ac` prompts for each declared input still
missing. The widget follows its resolved type:

- a required `bool` → a yes/no confirm,
- an optional/defaulted `bool` → a select containing `↩ Omit input` or
  `↩ Use declared default`, followed by `True` and `False`,
- a required `Literal[...]` → a select list of the allowed tags,
- an optional/defaulted `Literal[...]` → the same list with the omit/default action
  preselected and each explicit tag labelled `Set value: ...`,
- anything else → a free-text entry, initially blank.

Choosing the omit/default action, or leaving optional/defaulted free text blank,
does not submit a value. The key stays absent: START applies the exact validated
default payload when present, while an optional input without a default remains
absent. The CLI never stringifies a default into an override. A typed answer remains
an ordinary explicit override and follows run-boundary coercion and validation.

Each prompt's label carries the input's canonical resolved **type**, a required
(`*`) / `optional` mark, and any **default** — e.g. `topic (str) *`,
`as_of (Optional[date]) [optional]`, `window (int) [default: 30]`. For the
ISO-8601 `date` / `datetime` scalars — whose accepted string form isn't obvious —
a label with no default also shows an **example** value (`e.g. 2026-05-21`,
`e.g. 2026-05-21T14:30`); when a default is present it already shows the format,
so the example is dropped. An explicit null default is shown as `[default: None]`.
Whitespace-only optional/defaulted free text is also omission. To supply explicit
null, whitespace-only strings, or structured list/object values, use `--inputs`
JSON. Cancelling a prompt cancels the run.

!!! note
    Initial input prompts need a real terminal (TTY). In a non-interactive context
    (a CI job, a piped command, `srun` without a pty), missing optional/defaulted
    declarations stay absent so START applies defaults or retains optional absence.
    If required declarations are missing, `ac` reports all of them and exits with
    code `2` before provider or run-log setup; supply them through `--input` or
    `--inputs`.

## Blocking interactions

`human_input` and an agent's authored `ask_user` tool are blocking interactions
inside live node calls. These terminal interactions also require an interactive
TTY; the initial-input omission policy above does not make them non-interactive.
`ac` runs the flow behind a main-thread terminal broker: the node's worker waits,
the broker collects the answer, and the answer returns into that same live node
call. A `human_input` always asks; an agent calls `ask_user` only when its model
chooses to.

```console
ac run examples/human-approval.yaml --input task="Plan a team offsite"
#   → draft is written
#   → CLI asks: "Approve it as-is, or send it back to be revised? (approve / revise)"
#   → you type: approve
#   → run continues to completion and prints the kickoff message
```

An authored `wait` also stays in the live run: it occupies one worker until its
absolute UTC deadline, then returns `None`. It does not ask the terminal for an
early release or split execution into a later continuation.

In the bundled composer chat, Ctrl-C also reaches a WAIT inside the `run_flow`
flow-op tool. The nested run uses a one-way child stop signal: host cancellation
flows into it, but an ordinary nested failure remains a `FAILED` tool result and
does not cancel the surrounding conversation.

## Output and progress

- The terminal output is rendered as Markdown when it is a non-empty string,
  otherwise printed as-is (e.g. a multi-field object).
- **Per-node progress** is printed to **stderr** as the flow advances. On a real
  terminal each running node shows an animated spinner that is rewritten in place
  as a green `✓ node_id` on success, or a red `✗ node_id` (with the error on the
  next line) on failure. Off a terminal (a pipe, CI) only the final `✓`/`✗` line
  per node is printed. Nodes that fan out run concurrently, so several spinners
  can be live at once.
- Pass `--quiet` / `-q` to suppress progress entirely.
- Pass `--verbose` / `-v` to also print **each node's output** under its check as
  it finishes — handy for seeing the intermediate values a multi-node flow
  produces, not just the final result.
- The durable log path is always reported, even under `--quiet`, so an interrupted
  run remains discoverable.

## Concurrency (`--num-workers`)

Fresh runs default to single-threaded execution: ready nodes run one at a time in
a deterministic order. Pass `--num-workers N` / `-w N` to run with a worker pool
of size `N`, so **independent** ready nodes — the parallel arms of a fan-out —
execute concurrently:

```console
# the debate diamond argues both sides at once instead of one after the other
ac run examples/debate.yaml --input question="Adopt a monorepo?" --num-workers 4
```

- A fresh run with no worker flag uses `0`, the single-threaded deterministic
  drain. A resumed run with no flag preserves the latest count in its log.
- On resume, an explicit `-w N` records `N` for the new segment and changes only
  future scheduling; it does not rewrite historical work.
- `>=1` spawns that many workers. Nodes are pure executors and a single-writer
  dispatcher owns all state, so the **terminal output is the same** whatever the
  worker count — only the intra-run scheduling changes (and wall-clock time, when a
  flow has parallel branches that each do real work, e.g. separate LLM calls).
- Each blocking HUMAN_INPUT, agent `ask_user`, or future-deadline WAIT occupies one worker;
  independent ready nodes use any remaining workers and wait when all are occupied.

The dispatcher appends every node start and outcome to the selected authoritative
log before publishing the progress event that the CLI renders.

## Exit codes

| Status | Exit code |
|--------|-----------|
| Flow succeeded | `0` |
| Flow failed | `1` (error printed to stderr) |
| Flow failed to compile (bad YAML) | `1` (located error printed to stderr) |
| Run-log creation or recovery failed | `1` (concise error printed to stderr) |
| Run cancelled at the input prompt | `1` |
| Invalid CLI input (including a forbidden resume combination or required inputs missing without a TTY) | `2` |
| Run interrupted with Ctrl-C during execution | `130` |

## Compile errors

When a flow fails to **load/compile** (an unknown reference, a cycle, a bad field,
a malformed `uses:`), `ac` does not dump an engine Python traceback — it shows
**where in your `.yaml` it broke** as a boxed source frame: the `.yaml` around the
fault with line numbers, the offending line highlighted (`❱`), the panel titled
`file:line`, and the message below. (Same shape as a Python traceback's code box.)

```console
$ ac run tests/seeds/errors/e01-undeclared-ref.yaml
╭─ e01-undeclared-ref.yaml:19 ─────────────────────────────────────────────────────────────────────╮
│   14     input:                                                                                  │
│   15       topic: ${input.topic}                                                                 │
│   16     output: float                                                                           │
│   17     prompt: "Rate ${topic} in [-1, 1]; output only the number."                             │
│   18                                                                                             │
│ ❱ 19 output: ${scor.output}     # <-- typo: no node named `scor` (it's `score`)                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
flow has unresolved references:
  flow output from: reference ${scor.output} uses unknown namespace 'scor'
```

Pass `--engine-trace` to **also** print the engine's Python traceback under the
located error — for debugging the engine itself, not your flow.

An error that spans **several** places (e.g. a **cycle**, which implicates every node
in the loop) highlights *all* of them, widens the frame to cover both ends, and prints a
"why" legend naming each dependency edge that closes the loop — so you can see not just
*which* nodes form the cycle but *which references* create it:

```console
$ ac run tests/seeds/errors/e02-cycle.yaml
╭─ e02-cycle.yaml:12,18 ───────────────────────────────────────────────────────────────────────────╮
│    7                                                                                             │
│    8 input:                                                                                      │
│    9   seed: str                                                                                 │
│   10                                                                                             │
│   11 nodes:                                                                                      │
│ ❱ 12   a:                                                                                        │
│   13     kind: agent                                                                             │
│   14     input:                                                                                  │
│   15       x: ${b.output}          # a depends on b ...                                          │
│   16     output: str                                                                             │
│   17     prompt: "Echo ${x}."                                                                    │
│ ❱ 18   b:                                                                                        │
│   19     kind: agent                                                                             │
│   20     input:                                                                                  │
│   21       x: ${a.output}          # ... and b depends on a -> cycle                             │
│   22     output: str                                                                             │
│   23     prompt: "Echo ${x}."                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
flow has a cycle involving ['a', 'b']; flows must be acyclic
  ↳ a depends on b (a.input.x)
  ↳ b depends on a (b.input.x)
```

## Runtime errors

A failure that happens **while the flow runs** — a node raises, a `code` node
returns the wrong type, an `:?` required reference is missing at its use site —
is shown like a compile error when authored source is available: boxed `.yaml`
frames with the message below. The innermost frame points at the **precise line
the failure originates from** — the
input binding, the assert expression, the input declaration, a node's `output:`
declaration — not just the node header. So a runtime error reads like a compile error: you see *exactly* where in
the flow it broke, not an engine traceback:

```console
$ ac run tests/seeds/errors/e07-required-missing.yaml --input topic=climate   # as_of omitted
╭─ e07-required-missing.yaml:23 ───────────────────────────────╮
│   18 nodes:                                                  │
│   19   report:                                               │
│   20     kind: agent                                         │
│   21     input:                                              │
│   22       topic: ${input.topic}                             │
│ ❱ 23       as_of:  ${input.as_of:?as_of is required ...}     │
│   24     output: str                                         │
╰──────────────────────────────────────────────────────────────╯
as_of is required for the report
```

Runtime failures carry an ordered trace of opaque authored origins. The loaded
Compose front end resolves that trace through its source catalog, and the CLI sends
the resulting diagnostic through the same renderer used for load errors. A referenced
`code` callable therefore points at its `code:` scalar; inline code can point at the
exact authored line that raised. The CLI does not infer a location from a runtime node
id or fall back by node kind. If an origin cannot be presented, the semantic failure
message remains available without a source frame.

For a failure inside a CALL, MAP, or LOOP child flow, the ordered trace already
contains the outer invocation sites followed by the precise child operation. The
front end resolves each origin against the currently loaded source, so nested and
external flows render in outermost-to-innermost order without walking the compiled
child graph or reusing coordinates retained by the runtime.

Failures with **no node behind them** are still boxed at their precise line: a
false boundary/post `assert:` boxes the offending `asserts:` expression, an
input that can't be coerced to its declared type boxes that input's declaration,
and a node whose value fails its declared `output:` Shape boxes that `output:`.

## Next

- [Flow syntax](syntax.md) — write your own flows.
- [Examples](examples.md) — walk the shipped flows.
