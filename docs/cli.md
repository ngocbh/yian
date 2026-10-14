# The `yian` CLI

`yian` runs a flow file: it loads the YAML, gathers inputs for a fresh run or
recovers them from a durable log, drives the flow to a terminal state, and
prints the output.

```console
yian run FLOW.yaml [--inputs inputs.json] [--input k=v]... [--log PATH] [--memory-root DIR] [--memory-storage KEY=PATH]... [-w N] [--provider NAME] [--model NAME] [--quiet] [--verbose] [--engine-trace]
yian run FLOW.yaml --resume PATH [--memory-root DIR] [--memory-storage KEY=PATH]... [-w N] [--quiet] [--verbose] [--engine-trace]
```

Commands below that use `examples/...` assume a repository clone and run from
its root; those files are not installed with the Python package.

A flow's AGENT nodes need a provider and model. There is no built-in default: they
come from the flow's `llm_config`, then `--provider`/`--model`, then
`YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL`, then the login saved by
`yian auth <provider>`. See
[Choosing a provider and model](installation.md#choosing-a-provider-and-model).

## Authentication

```console
yian auth
yian auth PROVIDER [--model MODEL] [--api-key-env NAME] [--endpoint URL] [--method subscription|api-key] [--login browser|device]
yian auth PROVIDER --default
yian auth PROVIDER --remove
```

`yian auth` with no provider lists the saved logins: each provider, its kind
(`api-key`, `subscription`, or `none`), its model, an Azure endpoint, and the
default. It notes when `YIAN_DEFAULT_PROVIDER` or `YIAN_DEFAULT_MODEL` overrides the
saved choice, and never prints a secret.

`yian auth PROVIDER` sets up one provider (`openai`, `anthropic`, `google`, `azure`,
`ollama`, `vllm`, `xai`, `deepseek`, `qwen`, `qwen-cn`, `glm`, `glm-cn`, `minimax`,
`minimax-cn`, or `openrouter`). It saves the credential and the model together:

- `--api-key-env NAME` names the variable holding the API key; the default is the
  provider's own (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...). An unset variable
  gives a hidden prompt in a terminal. `ollama` takes no key; `vllm`'s is optional
  unless `--api-key-env` names its variable.
- `--endpoint URL` (azure only) is the resource endpoint, such as
  `https://<resource>.openai.azure.com/`; `--model` is then the deployment. Azure
  also needs `OPENAI_API_VERSION` set.
- `--method` and `--login` (openai only) choose ChatGPT subscription or API-key
  access, and browser or device sign-in. Device login works over SSH.
- `--model MODEL` names the model; a terminal prompts when it is omitted. Off a
  terminal, supply the model; for openai, the method and login mode; for azure,
  `--endpoint`; and set the key variable for a provider that needs a key.

The model is accepted when the provider's model listing has it; otherwise (an alias
the listing does not show, or Azure, which has no listing) one short request to the
model must succeed. With a base-URL variable such as `ANTHROPIC_BASE_URL` set,
`yian auth` skips the listing and checks the model with one request. OpenAI accepts
only a model in the account's catalog. Setup changes nothing until sign-in and the
model check succeed; a failed or cancelled setup keeps the previous login. The
provider becomes the default only when no default is saved yet.

`--default` makes an already-saved provider the default without signing in again.
`--remove` deletes that provider's saved login; removing the default leaves no
default. Neither takes other options.

A host `api_key` beats a saved login, and a saved login beats the provider's key
variable, so a saved OpenAI subscription wins over an ambient `OPENAI_API_KEY`.
`YIAN_DEFAULT_PROVIDER` and `YIAN_DEFAULT_MODEL` beat the saved default provider and
model. Changing an account or billing method requires running setup again. See
[Saved logins](installation.md#saved-logins-yian-auth).

## Durable logs and recovery

Before execution, a fresh invocation reports its selected JSONL path on stderr,
including under `--quiet`. The file becomes authoritative once runtime
initialization durably commits `RunStarted`:

```console
# fresh log: ~/.yian/runs/<run_id>.jsonl
yian run flow.yaml --input topic=climate

# choose another new path (it must not already exist)
yian run flow.yaml --input topic=climate --log ./climate.jsonl

# recover a terminal or unfinished prefix with the same authored flow
yian run flow.yaml --resume ./climate.jsonl
```

The CLI creates `~/.yian` and its `runs/` directory with mode `0700`;
the runtime creates each log exclusively with mode `0600`. A log can retain
canonical inputs, node outputs (including model, tool, and human data), resolved
model configuration, and failure tracebacks. Treat it as sensitive data: do not
commit or publish it.

Recovery reads the original inputs, run identity, and resolved model map from the
log, including each agent's connection identity (a saved login or the provider's key
variable) and endpoint, without credentials. Re-authentication cannot silently
redirect an unfinished run to a new account.
`--resume` therefore rejects `--input`, `--inputs`, `--provider`, `--model`,
and `--log` before input prompts or provider setup. After repairing any incomplete
tail, an already terminal log gets no new records and prints its recorded result.

Before an unfinished log appends its resume segment, `--resume` checks the
credentials of every recorded model configuration, as a fresh run does. Every saved
login an agent recorded, of any provider, must still be the same login. One that
was replaced by running `yian auth PROVIDER` again, removed, or whose method changed
fails, naming `yian auth PROVIDER`. A recorded OpenAI environment key fails when
`OPENAI_API_KEY` is missing or differs from the key the run started with. Every
other agent's key variable is checked as well. In a terminal it prompts for a
missing key; a typed OpenAI key must be the one the run started with. Otherwise it
exits 1: for a missing or changed key it asks you to set the key and resume again,
and a replaced or removed saved login needs a new run. A failed check appends
nothing, so the log stays resumable. A run started from Python with
`llm_config.api_key` can only be resumed from Python, with
`resume_flow(..., api_key=...)`.

An eligible unfinished log appends a resume segment and retries nodes with a durable
start but no durable outcome. Requests and tool actions may repeat; the run log neither reconstructs
session memory nor provides exactly-once execution. See the
[Python API recovery contract](api.md#durable-recovery-with-resume_flow).

Expected host-side failures while creating or recovering a log (for example a
corrupt, mismatched, or locked log) are reported as concise errors that name the
log path, plus the line and seq of a damaged record. Pass
`--engine-trace` when debugging the engine to include the Python traceback.

`yian chat` uses the same default directory and reports one new path for its initial
run. If a failed turn restarts the immutable chat flow from the last completed
transcript, that restart gets a separate log rather than reusing the failed run.
A restart happens only when the failed attempt answered at least one prompt with
a typed line. An attempt that fails before that (for example, a flow whose first
step fails, or a failure after terminal EOF) ends the session: `yian chat` prints
`chat failed: <reason>` to stderr and exits 1.

`yian chat` and its `run_flow` tool supply no long-term memory: memories a
flow declares stay off, and session conversations are not kept.

## Memory and storage

`yian run` builds every memory the flow declares, using
`yian.build_memories`. Without flags, each built-in memory
(`episodic`, `reflective`, `session`, `profile`) lives in RAM for that one
invocation. A flow without declarations opens no memory.

```console
# Keep every memory under ./mem/<flow id>/
yian run flow.yaml --input question="First question" --memory-root ./mem
yian run flow.yaml --input question="Follow-up" --memory-root ./mem

# Place one memory explicitly; others still follow --memory-root (or RAM)
yian run flow.yaml --memory-root ./mem --memory-storage lib.notes=./notes.sqlite

# Recover an unfinished run with the same storage
yian run flow.yaml --resume ./run.jsonl --memory-root ./mem
```

- `--memory-root DIR` stores memory key `notes` at
  `DIR/<flow id>/notes.sqlite`, and key `lib.notes` (imported through
  `uses: {lib: ...}`) at `DIR/<flow id>/lib/notes.sqlite`. A profile memory
  uses `.md`. Directories are created on first use. The flow `id:` must be
  usable as one directory name.
- `--memory-storage KEY=PATH` (repeatable) places one built-in memory at PATH.
  It overrides `--memory-root` for that key. PATH must be a local file path,
  not `:memory:` or a `file:` URI. A bad path shape, or two memories on one
  file (case ignored), exits 1 before the run starts. The directory must already exist; otherwise the run fails when that
  memory first opens (`storage.io`, or `profile.store_unavailable` for a
  profile).
- These are bad parameters (exit 2): a malformed option, a repeated key, a key
  the flow does not declare, and a key whose type is custom.
- A flow that declares a custom memory type cannot run from the CLI.
  `yian run` exits 1 with a setup error before it asks for inputs or
  writes a log. This also applies to `--resume`, even of a finished run.
  Build such a flow from Python with `build_memories(..., factories=...)` or
  `supplied=...`.

Nothing is opened before the run starts. Each `.sqlite` file has one owner at
a time: a second `yian run` on the same file fails with `storage.busy`
until the first exits. A profile's `.md` file has no single owner; writers take
turns through a lock file beside it. The files are unencrypted and can contain
prompts, replies, and learned notes. Use a local filesystem, not a shared
network one. Two flow files with the same `id:` share a directory under one
root. On case-insensitive filesystems, ids that differ only by case do too.
Changing a memory's type makes its old file fail with `storage.kind`. Changing
an episodic or reflective limit against an existing file fails with
`<type>.config_conflict`. Only removing the file or using the stored limits
clears it; a reset keeps the stored limits.

Session history advances automatically during execution. This does not enable
learning: `yian run` neither sets `learnable=True` nor calls `backward()`.

Both flags are accepted with `--resume`, and must name the storage the run
used. A terminal resume only prints the recorded result and opens no memory
(the registry is still built, so flag and custom-type errors still apply).
Run logs do not reconstruct session history; an interrupted request can be
repeated against the latest stored history.

## Skills

`yian run` supplies the Claude Code and Codex skills installed on your machine. An
AGENT uses those its `skills:` field grants (see
[Agent skills](syntax.md#agent-skills-skills)). When any AGENT in the flow grants
skills, the CLI reads these folders, and the first folder that has a skill name wins:

1. `~/.claude/skills`
2. `~/.agents/skills`
3. `.claude/skills` and then `.agents/skills` in the current directory, then in
   each parent directory up to the git repository root. These start at the
   directory you run `yian` in, not at the flow file's directory.

Your personal skills win over project skills of the same name, as in Claude Code.
A skill with a broken or unreadable `SKILL.md` is skipped with a `warning:` line on
stderr. So is a name with `_` or uppercase letters: skill names use lowercase
letters, digits, and hyphens. A flow that grants a skill none of these folders has
fails before the run starts. A flow without `skills:` reads no skill folders.
Resuming with `--resume` reads the folders again, so keep the skills unchanged
until the run finishes.

## Supplying inputs

A fresh flow declares typed `input:` fields. Explicit inputs layer in this order
(later overrides earlier):

1. **`--inputs file.json`** — a JSON object of `{ "key": value, ... }`. This is
   the non-interactive surface for native list/object values and explicit null.
2. **`--input k=v`** — one input, repeatable. This is a string/scalar surface:
   values arrive as strings and are coerced, where supported, against scalar input
   types at the run boundary (see [syntax.md](syntax.md) for the accepted `bool`
   spellings). It does not parse list/object JSON or explicit null. An `Any` input
   stays text (`--input x=3` gives `"3"`); pass a number or a structure in `--inputs`.
   A default applied by START was already validated at load and is not
   string-coerced.

An `int` input may have any number of digits, from `--inputs`, `-i`/`--input`, or the
interactive prompt.

Every key must name a declared input: an unknown name (such as a typo) is a bad
parameter (exit 2) that lists the declared inputs. When stdin is an interactive
TTY, `yian` then prompts for each declaration still missing.

```console
# all from flags
yian run examples/decision-brief.yaml --input question="Adopt a monorepo?" --input audience="execs"

# from a JSON file, with one override
yian run flow.yaml --inputs base.json --input audience="execs"

# in an interactive terminal, supply nothing and get prompted for each missing input
yian run examples/hello.yaml
```

### The "Running flow" banner

At the start of every run (unless `--quiet`), `yian` prints a boxed banner to
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

When stdin is an interactive TTY, `yian` prompts for each declared input still
missing. The widget follows its resolved type:

- a required non-nullable `bool` → a yes/no confirm,
- any other `bool` → one select containing omit/default when allowed, `Set null`
  when nullable, and `True` / `False`,
- a `Literal(...)` → one select containing its tags plus omit/default and
  `Set null` when applicable; explicit tags in a nullable/omittable select are
  labelled `Set value: ...`,
- a null-only `None` → omit/default when allowed plus `Set null`, with no value
  editor,
- a null-capable open scalar, list, object, or record → an action palette with
  omit/default when allowed, `Set null`, and `Enter value`,
- a non-null-capable list → a direct JSON-array text editor,
- a non-null-capable object or record → a direct JSON-object text editor,
- any other non-null-capable scalar → the existing free-text editor.

`Enter value` opens the existing scalar editor or the typed JSON editor. List/object
answers are parsed as JSON and validated recursively against the complete resolved
type, including list elements and record fields. Invalid JSON, the wrong root shape,
or a nested type mismatch is shown inside the editor and the same question remains
open for correction.

Across initial prompts, the CLI keeps four outcome meanings distinct: an explicit
value; `Set null`; omit/use-default; and cancellation. Each widget offers only the
outcomes allowed by its declaration. `Set null` submits a present `None`, so it shadows
a declared default and undergoes normal type validation. Omission keeps the key absent:
START applies the exact validated default payload when present, while an optional input
without a default remains absent. Cancelling at either the action palette or editor
cancels the run.

For a null-capable open input, optional blank/whitespace after `Enter value` returns
to a fresh action palette; it does **not** silently choose omission. A direct
non-null-capable optional/defaulted text or JSON editor retains the old behavior:
blank/whitespace omits the key. The CLI never stringifies a default into an override.
Any explicit typed answer follows ordinary run-boundary coercion and validation.

Each prompt's label carries the input's canonical resolved **type**, a required
(`*`) / `optional` mark, and any **default** — e.g. `topic (str) *`,
`as_of (Optional(date)) [optional]`, `window (int) [default: 30]`. For the
`date` / `datetime` scalars — whose accepted string form isn't obvious
([accepted forms](typing.md#valuekind-the-closed-vocabulary)) — a label with no
default also shows an **example** value (`e.g. 2026-05-21`,
`e.g. 2026-05-21T14:30`); when a default is present it already shows the format,
so the example is dropped. An explicit null default is shown as `[default: None]`.

The text `null` in a string editor and a `Literal` tag named `null` remain ordinary
strings, distinct from `Set null`. A structured editor still rejects root JSON
`null`; choose the palette's `Set null` action instead. Nested JSON null remains
valid exactly where the nested contract accepts it. To supply explicit null or a
native list/object without a TTY, use `--inputs`; `--input` remains a string/scalar
surface. An optional whitespace-only string value likewise needs an explicit input
source rather than the prompt's blank path.

!!! note
    Initial input prompts need a real terminal (TTY). In a non-interactive context
    (a CI job, a piped command, `srun` without a pty), missing optional/defaulted
    declarations stay absent so START applies defaults or retains optional absence.
    If required declarations are missing, `yian` reports all of them and exits with
    code `2` before provider or run-log setup. Native list/object values and explicit
    null must be supplied through `--inputs`; ordinary required scalars may use
    `--input` or `--inputs`.

## Blocking interactions

`human_input` and an AGENT node's authored `ask_user` tool are blocking interactions
inside live node calls. They are separate from initial input gathering and are
unchanged by its non-TTY omission policy. `yian` runs the flow behind a main-thread
terminal broker: the node's worker waits, the broker reads the answer from the
host input stream, and the answer returns into that same live node call. A
`human_input` always asks; an AGENT node calls `ask_user` only when its model chooses
to. When the model passes `options`, the prompt is a choice list with an "Other"
entry for a typed answer.

```console
yian run examples/human-approval.yaml --input task="Plan a team offsite"
#   → draft is written
#   → CLI asks: "Approve it as-is, or send it back to be revised? (approve / revise)"
#   → you type: approve
#   → run continues to completion and prints the kickoff message
```

An authored `wait` also stays in the live run: it occupies one worker until its
absolute UTC deadline, then returns `None`. It does not ask the terminal for an
early release or split execution into a later continuation.

In the bundled yian chat, Ctrl-C also reaches a WAIT inside the `run_flow`
flow-op tool. The nested run uses a one-way child stop signal: host cancellation
flows into it, but an ordinary nested failure remains a `FAILED` tool result and
does not cancel the surrounding conversation.

## Output and progress

- The terminal output is rendered as Markdown when it is a non-empty string,
  otherwise printed as-is (e.g. a multi-field object). An int prints every digit,
  however long, here and in `--verbose` node output.
- **Per-node progress** is printed to **stderr** as the flow advances. On a real
  terminal each running node shows an animated spinner that is rewritten in place
  as a green `✓ node_id` on success, or a red `✗ node_id` (with the error on the
  next line) on failure. Off a terminal (a pipe, CI) only the final `✓`/`✗` line
  per node is printed. With a worker pool, fan-out nodes can run concurrently, so
  several spinners can be live at once.
- Pass `--quiet` / `-q` to suppress progress entirely.
- Pass `--verbose` / `-v` to also print **each node's output** under its check as
  it finishes — handy for seeing the intermediate values a multi-node flow
  produces, not just the final result.
- Once an invocation reaches run-log setup, the path is reported even under
  `--quiet`, so an interrupted run remains discoverable.

## Concurrency (`--num-workers`)

Fresh runs default to a pool of eight workers, so **independent** ready nodes
execute concurrently. Pass `--num-workers N` / `-w N` to choose a different pool
size, or `-w 0` to execute ready nodes serially in a deterministic order:

```console
# limit the debate flow to four concurrent workers
yian run examples/debate.yaml \
  --input question="Adopt a monorepo?" --num-workers 4
```

- A fresh run with no worker flag uses `8`. A resumed run with no flag preserves
  the latest count in its log.
- On resume, an explicit `-w N` records `N` for the new segment and changes only
  future scheduling; it does not rewrite historical work.
- `>=1` sets the maximum concurrent worker count. Nodes cannot mutate graph state, and a
  single-writer dispatcher commits their outcomes. The worker count changes only
  ready-work scheduling, not the graph or output binding; effectful nodes may
  still be nondeterministic.
- Each blocking HUMAN_INPUT, `ask_user` call from an AGENT node, or future-deadline WAIT occupies one worker;
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
| Memory setup failed (a custom memory type, a storage path that is not a local file, or a flow `id:` that cannot be a directory name) | `1` (concise error printed to stderr) |
| Run cancelled at the input prompt, or input closed at a resume key prompt | `1` |
| Resume key check failed (a missing, changed, or unavailable provider key) | `1` (error printed to stderr) |
| Invalid CLI input (including a forbidden resume combination, a bad `--memory-root` or `--memory-storage` flag, an `--inputs` file that is not UTF-8 text, not valid JSON, or not a JSON object, an input the flow does not declare, or required inputs missing without a TTY) | `2` |
| Run interrupted with Ctrl-C during execution or at a resume key prompt | `130` |

## Compile errors

When a flow fails to **load/compile** (an unknown reference, a cycle, a bad field,
a malformed `uses:`), `yian` does not dump an engine Python traceback — it shows
**where in your `.yaml` it broke** as a boxed source frame: the `.yaml` around the
fault with line numbers, the offending line highlighted (`❱`), the panel titled
`file:line`, and the message below. (Same shape as a Python traceback's code box.)

```console
$ yian run tests/seeds/errors/e01-undeclared-ref.yaml
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
$ yian run tests/seeds/errors/e02-cycle.yaml
╭─ e02-cycle.yaml:12,18 ───────────────────────────────────────────────────────────────────────────╮
│    7                                                                                             │
│    8 input:                                                                                      │
│    9   seed: !str                                                                                │
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
$ yian run tests/seeds/errors/e07-required-missing.yaml --input topic=climate   # as_of omitted
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
