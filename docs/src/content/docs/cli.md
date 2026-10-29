---
title: "CLI"
description: "Every yian command and flag."
---

This page lists every `yian` command and flag, what each command prints, and its exit codes.
`yian <command> --help` shows the same flags in your terminal.

## Commands

| Command | What it does |
|---|---|
| [`yian run`](#yian-run) | Start a flow run, or resume one from its run log, and print its output |
| [`yian chat`](#yian-chat) | Chat with a flow: each line you type goes to its inbox input |
| [`yian check`](#yian-check) | Load flows without running them and report whether they load |
| [`yian status`](#yian-status) | Report where a run stands, or list your runs |
| [`yian auth`](#yian-auth) | Set up, list, choose the default among, or remove provider logins |
| [`yian providers`](#yian-providers) | List the supported providers and their API-key variables |
| [`yian models`](#yian-models) | List the models a login or an API key offers |
| [`yian config check`](#yian-config-check) | Check an `llm_config` layer read from stdin |
| [`yian skills check`](#yian-skills-check) | Check skill directories |

`yian` on its own also takes these options:

| Option | What it does |
|---|---|
| `--help` | Show the commands and exit |
| `--version` | Show the version and exit |
| `--licenses` | Show the licenses of yian and of what its binaries bundle, and exit |

Before a command reads its flags, yian checks every `YIAN_*` variable in the environment (see
[Defaults and limits](/reference/#defaults-and-limits)). A bad value prints its error and exits 1.

## yian run

`yian run` loads a flow, asks for any inputs still missing, runs it, and prints its output. Every
run writes a durable run log, so you can resume a run that stopped. On a terminal, a flow with an
inbox input becomes a chat (see [Chatting on a terminal](#chatting-on-a-terminal)).

```console
$ yian run brief.yaml --input topic=climate
$ yian run brief.yaml --inputs inputs.json --input words=200 --quiet
$ yian run --resume ~/.yian/runs/4f1c/log.jsonl
```

| Flag | Value | What it does |
|---|---|---|
| `FLOW` | path | The flow `.yaml` to run. Leave it out with `--resume` |
| `-i`, `--input` | `k=v` | One input. Repeat it for more |
| `--inputs` | file | A JSON file of inputs |
| `--inbox` | name | On a terminal, the inbox input to send typed lines to, when the flow has several |
| `--log` | path | Write the run log here instead of the default location. The path must not exist |
| `--resume` | path | Resume the run recorded in this run log, with the flow it recorded |
| `-q`, `--quiet` | | Print no banner and no per-node progress |
| `-v`, `--verbose` | | Also print each node's output as it finishes |
| `-w`, `--num-workers` | int | How many nodes may run at once. Default 8 for a fresh run; `0` runs nodes one at a time |
| `--provider` | name | Provider for agents that set none |
| `--model` | name | Model for agents that set none |
| `--memory-root` | dir | Keep every built-in memory under `DIR/<flow id>/` |
| `--memory-storage` | `KEY=PATH` | Keep one memory at `PATH`. Repeat it for more |
| `--mcp-config` | file | Read MCP servers from this file too. Repeat it for more |
| `--engine-trace` | | On a failure, also print the engine's stack trace |
| `--json` | | Print one JSON document on stdout (see [Machine-readable output](#machine-readable-output)) |

`--provider` and `--model` fill a gap: an agent's own `llm_config` and the flow's win over them.
See [Models and providers](/models/) for the whole order.

### Inputs

Inputs come from three places, later ones winning:

1. `--inputs FILE`: a JSON object of input names to values. Use it for lists, records and
   `null`.
2. `--input k=v`: one value as text. yian reads the text as the input's declared type, so
   `--input times=2` fills an `int` input. An `Any` input keeps the text as is.
3. A prompt, for each input still missing, when stdin is a terminal.

A name the flow doesn't declare, or a value for an inbox input, is a usage error; an undeclared name
lists the declared inputs. Off a terminal,
missing optional inputs stay unset (their defaults apply), and missing required inputs are a usage
error that names them all.

<!-- yian-test {"replies": ["Hi Ada! Hi Ada!"]} -->
```yaml check=runnable title="greet.yaml"
id: greet
name: greet
input:
  name: !str
  times: !int
output: str
kind: agent
prompt: Greet ${name} ${times} times, in one line.
```

```console
$ yian run greet.yaml --input name=Ada --input times=2
Hi Ada! Hi Ada!
```

Each input prompt shows the type and whether it is required or has a default, such as
`topic (str) *` or `window (int) [default: 30]`. A list or record opens a JSON editor; a nullable
input offers `Set null`. Ctrl-C or the end of input cancels: `run cancelled`, exit 1.

### Run logs and resuming

A fresh run writes its log to `~/.yian/runs/<run_id>/log.jsonl`, in a folder of its own, and
prints `Run log: PATH` on stderr before it starts, even with `--quiet`. With `--log PATH`, choose a
path in a folder no other run uses: that folder is the run's folder, and the run keeps its lock
file and its `media` folder beside the log. A model call whose reply holds media fails if `media`
is a symlink, another user's, or writable by others.

The log holds the run's inputs, node outputs, model replies and errors. Treat it as private data.
yian creates `~/.yian` and the run folders with mode `0700` and the log with mode `0600`.

`--resume PATH` continues the run recorded in that log. It runs the flow files the run started
with, not the files on disk now, so an edited flow needs a new run. It takes no `FLOW`, `--input`,
`--inputs`, `--provider`, `--model` or `--log`: the log holds all of those. It does take `-w`,
`--memory-root`, `--memory-storage` and `--mcp-config`; give the memory flags the run used.

- A finished run prints its recorded result again and runs nothing.
- An unfinished run retries the nodes that started but never finished. A model call or tool call
  that was cut off may run twice.
- Before it appends anything, the resume checks each provider key or saved login the run
  recorded. A missing key prompts in a terminal; a changed key, a replaced login, or a key that
  came from a host stops the resume with exit 1 and leaves the log as it was.
- The `YIAN_*` values must match the ones the run started with. Each one that differs is named,
  with the setting to use (see [Defaults and limits](/reference/#defaults-and-limits)).
- Without `-w`, a resume keeps the recorded worker count.

### Waiting runs

A run can end waiting: on an inbox `take`, a `sleep` whose time hasn't come, or a host call. Then
`yian run` prints what it waits on and exits 0:

```text
run waiting: review (take)
continue with: yian run --resume /home/ada/.yian/runs/4f1c/log.jsonl
```

`--resume` moves a waiting run on once a call it waits on is answered or past its deadline;
otherwise it appends nothing and prints the same lines again. It doesn't wait for a deadline
itself. An `ask` never waits in `yian run`: it prompts at the terminal. See
[Inboxes and waiting](/inboxes/).

### Chatting on a terminal

When stdin and stderr are a terminal, `yian run FLOW` chats with a flow that has an `Inbox(str)`
or `Inbox(Message)` input, as [`yian chat`](#yian-chat) does: each line you type goes to the inbox,
and the output of each node named `reply` prints on stdout. It chats when all of these hold:

- The flow has one such input, or `--inbox NAME` picks one of several.
- The flow declares no memory and grants or calls no MCP tool.
- The run is fresh, with no `--resume`, `--json` or `--memory-storage`.

Otherwise the run goes as above, and a `take` on the inbox leaves it waiting. A chat takes
`--input`, `--inputs`, `--log`, `-w`, `--provider`, `--model` and `--engine-trace`, and shows no
banner or per-node progress.

### Memory

Without flags, each built-in memory a flow declares lives in RAM for one run.

- `--memory-root DIR` keeps memory `notes` at `DIR/<flow id>/notes.sqlite`, and `lib.notes`, from a
  `uses:` import, at `DIR/<flow id>/lib/notes.sqlite`. A profile memory uses `.md`. Folders are
  created on first use. The flow needs an `id:`.
- `--memory-storage KEY=PATH` puts one memory at `PATH` and wins over `--memory-root` for that key.
  The folder must exist.
- A key the flow doesn't declare, a repeated key, or a custom memory type's key is a usage error. A
  flow that declares a custom memory type can't run from the CLI; run it from your own program
  (see [Embedding yian](/embedding/)). `yian run` never runs the [backward pass](/learning/).

### Skills and MCP servers

When an agent in the flow grants skills, `yian run` and `yian chat` read skill folders, and the
first folder that has a name wins:

1. `~/.claude/skills`, then `~/.agents/skills`;
2. `.claude/skills`, then `.agents/skills`, in the current folder and each parent up to the git
   root.

An invalid skill is skipped with a `warning:` line. A flow that grants a skill none of the folders
has fails before it starts.

When a flow names a tool that isn't built in, `yian run` reads MCP servers from
`~/.yian/mcp.json`, when it exists, then from each `--mcp-config` file; a later entry with the same
name replaces an earlier one. It starts only the servers the flow needs and stops them when the run
ends. See [Agents](/agents/) for the file format.

### Output and progress

- The output goes to stdout: a string as it is, any other value in text form (`true`, `[1,"a"]`,
  `{"k":"v"}`).
- The banner (the flow's name, version and description) and per-node progress go to stderr. Each
  node shows a spinner, then `✓ node` or `✗ node` with the error below. Each line ends with how
  long the node has run, counting up while it runs and fixed once it ends (`✓ frame [5s]`,
  `⠋ debate [1m05s]`). Off a terminal, only the finished lines print.
- Warnings print as `warning: <message>` lines on stderr.
- Colour is used only on a terminal, and never when `NO_COLOR` is set.
- On a terminal, the output, errors and warnings lose their escape sequences and control
  characters (all but newlines and tabs), so text from a model or tool can't move the cursor, set
  the clipboard or add a link. Piped or redirected output keeps every byte.

A load error shows where the flow broke: the lines around it, the line marked `❱`, and the message
below. A runtime error points at the line it came from the same way. Add `--engine-trace` only to
debug yian itself.

### Stopping a run

A signal (SIGINT, SIGTERM, SIGHUP) or Ctrl-C at an `ask` prompt stops a run: `cancelled by user`,
exit 130. Once the log exists, `yian run --resume` continues it. A second signal ends at once.

## yian chat

`yian chat` starts a run of a flow with an `Inbox(str)` or `Inbox(Message)` input and sends each
line you type to it.

```console
$ yian chat chat.yaml
$ yian chat support.yaml --inbox questions --input team=billing
$ yian chat --resume ~/.yian/runs/4f1c/log.jsonl
```

| Flag | Value | What it does |
|---|---|---|
| `FLOW` | path | The flow `.yaml` to chat with. Leave it out with `--resume` |
| `--inbox` | name | The inbox input to send lines to, when the flow has several |
| `-i`, `--input` | `k=v` | Another input. Repeat it for more |
| `--inputs` | file | A JSON file of the other inputs |
| `--log` | path | Write the run log here instead of the default location. The path must not exist |
| `--resume` | path | Continue the chat recorded in this run log, with the flow it recorded |
| `-w`, `--num-workers` | int | How many nodes may run at once (default 8; `0` for one at a time). A resume keeps the recorded count unless you give it |
| `--provider` | name | Provider for agents that set none |
| `--model` | name | Model for agents that set none |
| `--engine-trace` | | On a failure, also print the engine's stack trace |

- Each time the run waits, a `> You:` prompt shows on stderr. Your line goes to the inbox and the
  run moves on. The output of every node named `reply` prints on stdout as it finishes, cleaned
  on a terminal as `yian run`'s output is.
- An `Inbox(Message)` gets your line as a user message with one text part.
- `/exit`, `/quit` or Ctrl-D closes the inbox; once the flow finishes: `session ended`, exit 0.
- Ctrl-C while a reply runs interrupts it, and the prompt comes back. A second Ctrl-C within 2
  seconds, or Ctrl-C at the prompt, cancels the run: `chat cancelled`, exit 130.
- SIGTERM, SIGHUP, or Ctrl-C at an `ask` prompt stops the run: `chat stopped; the run is in LOG`
  and `continue with: yian chat --resume LOG` (with `--inbox NAME` when the flow has several), exit
  130.
- On a terminal, Esc takes back the lines the run hasn't read yet into the prompt and stops the
  reply. It never cancels the run.
- A line the inbox refuses prints `not sent: REASON`, and the chat goes on.
- While the prompt is up, the chat wakes the run when a `sleep` or a timeout is due.
- A failed run prints its error and exits 1; `yian status LOG` shows where it stands.

`--resume LOG` picks up a chat that stopped, or that was waiting for your next line when it ended.
It runs the flow the log recorded, as `yian run --resume` does, so you leave out `FLOW`, `-i`,
`--inputs`, `--provider`, `--model` and `--log`. Pass `--inbox` again if the flow has several.

- The chat first moves the run on. A reply the stop cut runs again and prints; replies printed
  before don't print again. Then the `> You:` prompt comes back as usual.
- If the inbox was closed before the stop, no prompt comes back: the chat waits for the run to
  finish.
- Before the run calls a model, the chat checks the provider keys the run started with, as
  `yian run --resume` does. A missing, changed or declined key prints why and exits 1.
- A run that has ended can't be continued: `chat failed: the run in LOG has already ended
  (STATUS); start a new chat`, exit 1, and nothing is written to the log.

`yian chat` reads skills as `yian run` does (see [Skills and MCP servers](#skills-and-mcp-servers)),
once when it starts, so a skill edited during the chat takes effect in the next one. It opens no MCP
servers and builds no memories. See [Inboxes and waiting](/inboxes/) for how a flow reads its inbox.

## yian check

`yian check` loads each flow and runs none of its code. Use it in CI or before you run a flow.

```console
$ yian check brief.yaml review.yaml
brief.yaml: ok
review.yaml: ok
```

| Flag | Value | What it does |
|---|---|---|
| `FLOWS...` | paths | The flow files to check |
| `--root` | dir | Load each flow, its `uses:` imports and its CODE files only from inside this folder |
| `--json` | | Print one JSON document |

A flow that doesn't load prints its path and the located error on stderr, and the command exits 1.

With `--root DIR`, each flow and each `uses:` import must be inside `DIR` (`compose.outside_root`
otherwise), a `uses:` ref loads only from the importing file's folder, and a flow that sets
`system.paths` is refused. Each `./file.ts:fn` CODE reference must resolve inside `DIR` too,
symlinks followed, as with `loadFlow`'s `codeRoot`; one outside fails with `compose.load`.

With `--json`, `flows` holds one result per argument, in order. A flow that loads gives its `id`,
`name`, `description` and `inputs`, each input with its `type`, `required` and `has_default`; one
that doesn't gives `diagnostics`:

```json
{"format":"yian.cli.v1","flows":[{"description":null,"id":"brief","inputs":[{"has_default":false,"name":"topic","required":true,"type":"str"}],"name":"brief","ok":true,"path":"brief.yaml"}]}
```

## yian status

`yian status LOG` reports where the run in a run log stands. It only reads the log, so it works
while the run is going.

```console
$ yian status ~/.yian/runs/4f1c/log.jsonl
status: crashed
in flight: draft
crashed attempts: draft=1
```

| Status | Meaning |
|---|---|
| `running` | A process is writing the log now |
| `succeeded`, `failed`, `cancelled` | The run ended |
| `waiting` | Every unfinished node waits on a take, a sleep or a host call |
| `stopped` | A signal or the host stopped the run; `--resume` continues it |
| `crashed` | The process ended mid-run; `--resume` continues it |

With `--json`, the document also has `ready` (a waiting node can be woken now) and `wake_at` (the
earliest deadline it waits on, or `null`). A log it can't read exits 1.

`yian status` with no `LOG` lists the runs in `~/.yian/runs`, oldest first: run id, status, start
time and flow file. It first updates the run index, `~/.yian/runs/index.sqlite`, which is rebuilt
from the logs and safe to delete. With `--json`, each run also has its end time, result and token
counts.

A run whose log yian can't read shows `unreadable at seq N` as its status: N is the first record
it couldn't read, such as a damaged line. The update warns once, and again only after the log
changes. `yian status LOG` says what is wrong. With `--json`, such a run has `unreadable` with
`seq` and `reason` (`null` for other runs), and its other fields describe the records before seq N.

## yian auth

`yian auth` saves the provider login a run uses when no `llm_config` names one. See
[Models and providers](/models/) for how the login is chosen.

```console
$ yian auth
$ yian auth anthropic --model claude-sonnet-4-5
$ yian auth openai --method subscription --login device
$ yian auth openai --default
$ yian auth anthropic --remove
```

| Flag | Value | What it does |
|---|---|---|
| `PROVIDER` | name | The provider to set up, make the default, or remove. Leave it out to list your logins |
| `--model` | name | The model to save (for azure, the deployment name). Prompts when left out |
| `--api-key-env` | name | The variable holding the API key. Default: the provider's own, such as `ANTHROPIC_API_KEY` |
| `--endpoint` | URL | Azure only: the resource endpoint, such as `https://NAME.openai.azure.com/` |
| `--method` | `subscription` or `api-key` | OpenAI only: sign in with a ChatGPT subscription or use an API key |
| `--login` | `browser` or `device` | OpenAI subscription only: how to sign in. `device` works over SSH |
| `--organization` | id | OpenAI API key only: the organization id sent with each request |
| `--project` | id | OpenAI API key only: the project id sent with each request |
| `--header` | `NAME=VARIABLE` or `NAME` | An extra request header saved with the key, its value read from `VARIABLE` or a hidden prompt. Repeat it for more |
| `--default` | | Make an already-saved provider the default |
| `--remove` | | Delete the provider's login. Without a provider, delete the whole `--auth-file` |
| `--auth-file` | path | Use this file instead of `$YIAN_AUTH_FILE` or the default (see [Files and folders](/reference/#files-and-folders)) |
| `--json` | | For `--remove` and for OpenAI device sign-in only |

Setup changes nothing until sign-in and a check of the model succeed. The first provider you save
becomes the default. Listing never prints a secret. Off a terminal, give every value the setup
would ask for.

An OpenAI API key is checked by listing the models at `OPENAI_BASE_URL` when it is set, else at
OpenAI's API, so the key goes only to the host runs use.

## yian providers

`yian providers` lists each supported provider, the login kinds `yian auth` can save for it, and
the variable its API key is read from.

```console
$ yian providers
openai: api-key, subscription (OPENAI_API_KEY)
xai: api-key (XAI_API_KEY)
deepseek: api-key (DEEPSEEK_API_KEY)
qwen: api-key (DASHSCOPE_API_KEY)
qwen-cn: api-key (DASHSCOPE_CN_API_KEY)
glm: api-key (ZHIPU_API_KEY)
glm-cn: api-key (ZHIPU_CN_API_KEY)
minimax: api-key (MINIMAX_API_KEY)
minimax-cn: api-key (MINIMAX_CN_API_KEY)
openrouter: api-key (OPENROUTER_API_KEY)
vllm: api-key, none (VLLM_API_KEY)
ollama: none
anthropic: api-key (ANTHROPIC_API_KEY)
google: api-key (GOOGLE_API_KEY or GEMINI_API_KEY)
azure: api-key (AZURE_OPENAI_API_KEY)
```

With `--json`, `providers` holds each one's `name`, `methods` and `api_key_env` (`null` for none).

## yian models

`yian models` lists the models a login or a key offers.

```console
$ yian models --auth-file ~/.config/yian/auth.json
$ printf '%s' "$OPENROUTER_API_KEY" | yian models --provider openrouter
```

| Flag | Value | What it does |
|---|---|---|
| `--auth-file` | path | List the models of the OpenAI subscription login saved in this file |
| `--provider` | name | List the models this provider's host offers for the key on stdin |
| `--base-url` | URL | With `--provider`: list this host instead of the provider's own |
| `--timeout` | seconds | Time for the whole listing, above 0 (default 8) |
| `--json` | | Print one JSON document |

Give exactly one of `--auth-file` and `--provider`. The key comes from stdin, never from the
command line or the environment; an empty stdin, or a terminal, sends none. Without `--base-url`,
`--provider openai` lists `OPENAI_BASE_URL` when it is set, the host runs use. Azure has no model
listing. A failed listing exits 1.

## yian config check

`yian config check` reads an `llm_config` layer as a JSON object on stdin, key included, and checks
it as a run would. Values never appear in its messages.

```console
$ cat layer.json | yian config check --json
{"format":"yian.cli.v1","errors":[{"field":"openai.temperature","message":"Input should be a valid number"}],"ok":false}
```

Without `--json` it prints `ok`, or one `FIELD: MESSAGE` line per error on stderr. It exits 0 when
the layer is valid, 1 when it has errors, and 2 when stdin isn't a JSON object.

## yian skills check

`yian skills check` loads the `SKILL.md` in each directory.

```console
$ yian skills check skills/brief skills/review
skills/brief: ok (brief)
skills/review: ok (review)
```

An invalid skill prints its error on stderr, and the command exits 1. With `--json`, `skills` holds
one result per directory, in order. See [Agents](/agents/) for the skill format.

## Machine-readable output

With `--json`, a command prints one JSON object and a newline on stdout, and nothing else there.
Prompts, the banner, progress and warnings still go to stderr. OpenAI device sign-in is the one
exception: it prints JSON lines (`code`, `waiting`, then `done` or `error`).

| Command | Document keys |
|---|---|
| `yian run` | `status`, then `output` (succeeded), `failure` (failed), or `waits` and `wake_at` (waiting) |
| `yian check` | `flows` |
| `yian status LOG` | `status`, `in_flight`, `crashed_attempts`, `ready`, `wake_at` |
| `yian status` | `runs` |
| `yian skills check` | `skills` |
| `yian config check` | `ok`, `errors` |
| `yian providers` | `providers` |
| `yian models` | `models` (`null` for azure) |
| `yian auth --remove` | `ok` |

- Every document has `"format": "yian.cli.v1"`, and its keys are snake_case.
- Keys and codes are only ever added within `yian.cli.v1`. Ignore keys you don't know, and treat an
  unknown error code as a generic error.
- Message text can change; codes and keys don't.
- A float always has a fraction or an exponent (`2.0`); an int never does.
- During `yian run --json`, anything else a node writes to stdout goes to stderr.

```json
{"format":"yian.cli.v1","failure":{"code":"node.execute","frames":[{"label":"fail.yaml","range":[12,7,12,57]}],"message":"Error: no greeting for Ada","node":"greet"},"status":"failed"}
```

A `failure` is a diagnostic plus `node`, the failing node or `null`. A diagnostic is `code`,
`message`, `frames` and `notes`; a frame is `label`, `range` and `note`, and `range` is
`[start_line, start_col, end_line, end_col]`, 1-based with an exclusive end column. Empty
`frames`, `notes` and `note` are left out. See [Failure codes](/reference/#failure-codes).

A command that fails without its own result prints `{"format": "yian.cli.v1", "error": ...}`
instead. `error.code` is one of the [CLI error codes](/reference/#cli-error-codes).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. Also a run that ends waiting, and `yian status` whatever the status |
| `1` | A reported problem: a failed run, a flow that doesn't load, a setup error, a bad `YIAN_*` value, a cancelled input prompt, a failed check, login or listing. With `--json`, its document is on stdout |
| `2` | A usage error: an unknown flag, a missing argument, an undeclared input or a value for an inbox input, missing required inputs off a terminal, or flags that don't go together. The message is on stderr and there is no document |
| `130` | Stopped by a signal, or by Ctrl-C during the run. There is no document |

Any other exit, a crash included, prints no document; treat it as an internal error.

## Troubleshooting

| Problem | Try |
|---|---|
| A flow won't load | `yian check FLOW` shows the located error without running anything |
| A run stopped or crashed | `yian status LOG` shows where it stands; `yian run --resume LOG` continues it |
| A run waits | `yian status LOG --json` shows `ready` and `wake_at`; resume once it is ready or due |
| You lost a log path | `yian status` lists every run in `~/.yian/runs` |
| A model setting is refused | Check the layer with `yian config check` before you pass it to a run |
| A skill isn't found | `yian skills check DIR` shows why it doesn't load |
| A resume names a `YIAN_*` variable | Set or unset it as the message says, then resume again |
