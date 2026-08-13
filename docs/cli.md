# The `ac` CLI

`ac` runs a flow file: it loads the YAML, gathers the inputs, drives the flow to
a terminal state, and prints the output.

```console
ac run FLOW.yaml [--input k=v]... [--inputs inputs.json] [--quiet] [--verbose] [--num-workers N] [--engine-trace]
```

## Supplying inputs

A flow declares typed `input:` fields. You can supply them three ways, and they
layer in this order (later overrides earlier):

1. **`--inputs file.json`** — a JSON object of `{ "key": value, ... }`.
2. **`--input k=v`** — one input, repeatable. Values arrive as strings and are
   coerced to each input's declared type at the run boundary.
3. **Interactive prompt** — any *required* input still missing is prompted for.

```console
# all from flags
ac run examples/decision-brief.yaml --input question="Adopt a monorepo?" --input audience="execs"

# from a JSON file, with one override
ac run flow.yaml --inputs base.json --input audience="execs"

# supply nothing — get prompted for each required input
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

When a required input is missing, `ac` prompts for it. The widget follows each
input's declared type:

- a `bool` input → a yes/no confirm,
- a `Literal[...]` enum → a select list of the allowed tags,
- anything else → a free-text entry.

Each prompt's label carries the input's declared **type**, a required (`*`) /
`optional` mark, and any **default** — e.g. `topic (str) *`,
`as_of (Optional[date]) [optional]`, `window (int) [default: 30]`. For the
ISO-8601 `date` / `datetime` scalars — whose accepted string form isn't obvious —
a label with no default also shows an **example** value (`e.g. 2026-05-21`,
`e.g. 2026-05-21T14:30`); when a default is present it already shows the format,
so the example is dropped. An optional input left blank is skipped (its default
applies). Cancelling a prompt (Ctrl-C / Esc) cancels the run.

!!! note
    The prompt needs a real terminal (TTY). In a non-interactive context
    (a CI job, a piped command, `srun` without a pty) supply every required
    input via `--input` / `--inputs` instead, or the run will abort.

## Blocking interactions

`human_input` and an agent's authored `ask_user` tool are blocking interactions,
not suspension points. `ac` runs the flow behind a main-thread terminal broker:
the node's worker waits, the broker collects the answer, and the answer returns
into that same live node call. A `human_input` always asks; an agent calls
`ask_user` only when its model chooses to.

```console
ac run examples/human-approval.yaml --input task="Plan a team offsite"
#   → draft is written
#   → CLI asks: "Approve it as-is, or send it back to be revised? (approve / revise)"
#   → you type: approve
#   → run continues to completion and prints the kickoff message
```

An authored `wait` also stays in the live run: it occupies one worker until its
absolute UTC deadline, then returns `None`. It does not ask the terminal for an
early release and does not create a checkpoint.

In the bundled composer chat, Ctrl-C also reaches a WAIT inside the `run_flow`
flow-op tool. The nested run uses a one-way child stop signal: host cancellation
flows into it, but an ordinary nested failure remains a `FAILED` tool result and
does not cancel the surrounding conversation.

The CLI still contains a compatibility loop for a custom/artificial node that
directly returns the generic `Pause` result. It can answer legacy human-shaped or
scheduled reasons; a bare external-event reason remains paused. No authorable
AGENT, HUMAN_INPUT, or WAIT node enters this path, and issue #117 removes it.

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

## Concurrency (`--num-workers`)

By default the engine runs single-threaded: ready nodes execute one at a time in a
deterministic order. Pass `--num-workers N` / `-w N` to run with a worker pool of
size `N`, so **independent** ready nodes — the parallel arms of a fan-out — execute
concurrently:

```console
# the debate diamond argues both sides at once instead of one after the other
ac run examples/debate.yaml --input question="Adopt a monorepo?" --num-workers 4
```

- `0` (the default) is the single-threaded, deterministic drain.
- `>=1` spawns that many workers. Nodes are pure executors and a single-writer
  dispatcher owns all state, so the **terminal output is the same** whatever the
  worker count — only the intra-run scheduling changes (and wall-clock time, when a
  flow has parallel branches that each do real work, e.g. separate LLM calls).
- Each blocking HUMAN_INPUT, agent `ask_user`, or future-deadline WAIT occupies one worker;
  independent ready nodes use any remaining workers and wait when all are occupied.
- The dormant artificial-pause API preserves the configured worker count on a live
  resume. A durable compatibility resume uses the checkpointed count unless its caller
  explicitly overrides it.

## Exit codes

| Status | Exit code |
|--------|-----------|
| Flow succeeded | `0` |
| Flow failed | `1` (error printed to stderr) |
| Flow failed to compile (bad YAML) | `1` (located error printed to stderr) |
| Artificial compatibility pause left unresolved | `1` |
| Run cancelled at the input prompt | `1` |

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
is shown the same way as a compile error: a boxed `.yaml` frame, with the message
below. The frame points at the **precise line the failure originates from** — the
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

When the exact line can't be pinned (e.g. a `code` callable that raises somewhere
inside its body), the frame falls back to the **best line for the node's kind** —
a code node's `code:` field — and then to the node header. A failure with no node
behind it *and* no resolvable line prints the plain `run failed: <message>` line.

Failures with **no node behind them** are still boxed at their precise line: a
false boundary/post `assert:` boxes the offending `asserts:` expression, an
input that can't be coerced to its declared type boxes that input's declaration,
and a node whose value fails its declared `output:` Shape boxes that `output:`.

## Next

- [Flow syntax](syntax.md) — write your own flows.
- [Examples](examples.md) — walk the shipped flows.
