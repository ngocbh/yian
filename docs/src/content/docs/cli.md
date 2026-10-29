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
| [`yian chat`](#yian-chat) | Chat with yian's assistant, which finds, reads, checks, writes and runs flows for you |
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
| `--inbox` | name | On a terminal, the inbox input to send typed lines to, when the flow has several (see [Chatting on a terminal](#chatting-on-a-terminal)) |
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
or `Inbox(Message)` input: each line you type goes to the inbox, and the output of each node named
`reply` prints on stdout, its escape sequences dropped and its control characters shown. Only
stdin and stderr are checked, so with stdout redirected it still chats and the replies go to the
file. It chats when all of these hold:

- The flow has one such input, or `--inbox NAME` picks one of several.
- The flow declares no memory and grants or calls no MCP tool.
- There is no `--json`, `--memory-root` or `--memory-storage`; with `--resume`, see below.

Otherwise the run goes as above, and a `take` on the inbox leaves it waiting. To chat with a flow
from a script or a pipe, drive it from your own program with `send` and `advance` (see
[Inbox calls](/embedding/#inbox-calls)). When the rest holds
but `--inbox NAME` names none of the flow's inbox inputs, the command is a usage error. A chat takes
`--input`, `--inputs`, `--log`, `-w`, `--provider`, `--model` and `--engine-trace`, and shows no
banner or per-node progress.

```console
$ yian run chat.yaml
$ yian run support.yaml --inbox questions --input team=billing
```

- Each time the run waits, a `> You:` prompt shows on stderr. Your line goes to the inbox and the
  run moves on. The replies are cleaned on a terminal as `yian run`'s output is.
- An `Inbox(Message)` gets your line as a user message with one text part.
- `/exit`, `/quit` or Ctrl-D closes the inbox; once the flow finishes: `session ended`, exit 0.
- Ctrl-C while a reply runs interrupts it, and the prompt comes back. A second Ctrl-C within 2
  seconds, or Ctrl-C at the prompt, cancels the run: `chat cancelled`, exit 130.
- SIGTERM, SIGHUP, or Ctrl-C at an `ask` prompt stops the run: `chat stopped; the run is in LOG`
  and `continue with: yian run --resume LOG` (with `--inbox NAME` when the flow has several), exit
  130.
- Esc takes back the lines the run hasn't read yet into the prompt and stops the reply. It never
  cancels the run.
- A line the inbox refuses prints `not sent: REASON`, and the chat goes on.
- While the prompt is up, the chat wakes the run when a `sleep` or a timeout is due.
- A failed run prints its error and exits 1; `yian status LOG` shows where it stands.

On a terminal, `yian run --resume LOG` picks up a chat that stopped, or that was waiting for your
next line when it ended, when the flow the log recorded is one it would chat with. It runs that
flow, so you leave out `FLOW`, `-i`, `--inputs`, `--provider`, `--model` and `--log`. Pass
`--inbox` again if the flow has several.

- The chat first moves the run on. A reply the stop cut runs again and prints; replies printed
  before don't print again. Then the `> You:` prompt comes back as usual.
- If the inbox was closed before the stop, no prompt comes back: the chat waits for the run to
  finish.
- Before the run calls a model, the chat checks the provider keys the run started with, as
  `yian run --resume` does. A missing, changed or declined key prints why and exits 1.

A chat reads skills once, when it starts, so a skill edited during the chat takes effect in the
next one. See [Inboxes and waiting](/inboxes/) for how a flow reads its inbox.

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

When an agent in the flow grants skills, `yian run` reads skill folders, and the first folder that
has a name wins:

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

`yian chat` opens a chat with yian's assistant. It helps you find, read, check, write and run the
flows in the folder you start it in, and it reads yian's docs to answer questions about writing
flows. Give it a flow to start the chat about that flow: at every turn the assistant sees the
flow's text and whether it loads, as they were when the chat opened. It reads the file again to
see your later changes. A flow outside the folder is only that snapshot, since the tools can't
reach it; the chat says so when it opens. Run `yian chat` from the flow's folder to work on it.

```console
$ yian chat
$ yian chat hello.yaml
```

| Flag | Value | What it does |
|---|---|---|
| `FLOW` | path | A flow `.yaml` you want help with |
| `--log` | path | Write the chat's run log here instead of the default location. The path must not exist |
| `--provider` | name | Provider for the assistant and the flows it runs, when they set none |
| `--model` | name | Model for the assistant and the flows it runs, when they set none |
| `--engine-trace` | | On a failure, also print the engine's stack trace |

The assistant is an agent, so it needs a provider and model: set a default with `yian auth`, or pass
`--provider` and `--model`. A line on stderr names the model and the flow in focus. When that flow
has an inbox input, a second line says to chat with the flow itself through `yian run FLOW` (see
[Chatting on a terminal](#chatting-on-a-terminal)). The prompt, `/exit`, Ctrl-C and Esc work as
there, and the assistant's replies print on stdout.

When stdin, stdout and stderr are all a terminal, the chat draws its own look instead: a box at the
top names the model, how it is reached (an API key or a ChatGPT login), the folder and the flow in
focus. The reply streams in as it is written, with its Markdown (bold, italic, code, headings,
lists, code blocks) drawn. Each tool call shows as a `⏺` line with its result under it, and so does
each event of a flow the assistant runs. A spinner runs while the assistant works, and you type in
a framed box. Set `NO_COLOR`, or redirect any of the three, to get the plain lines above.

The box stays up under the spinner while the assistant works. Press Enter there to send the line at
once: the assistant reads it before its next model call, so it can change course mid-reply. Ctrl-C
and Esc work as they do between replies. An approval card or a question takes the box's place
until you answer it.

A paste of more than 3 lines or 800 characters goes into the box as one token, such as
`[Pasted text #1 +40 lines]`, numbered within the line. The cursor steps over a token, and
Backspace, Delete or Ctrl-U removes it whole. On Enter the assistant gets the whole text pasted,
tabs and all, while the line printed above shows the tokens. Esc puts a line it takes back into
the box with its tokens; after `/exit`, the line it prints shows them too. Text you type that looks
like a token stays text. The chat remembers tokens only while it runs: after `yian run --resume`,
Esc puts a line sent before the stop back as plain text, or as one token
(`[Pasted text #1 +40 lines]`) when it is as long as a paste that collapses.

If your terminal drops the end of a paste, the box keeps waiting for it. Once the paste has
paused for half a second, Esc ends it and puts in what arrived, and Ctrl-C drops it.

Under the box, the chat lists the flows the assistant started that are still running or waiting,
one row each, such as `r1  color.yaml  ● asks · 12s`, and the hint under the box adds `↓ runs`
when there is room. A flow drops off the list once it ends; its result stays in the chat. The
list's keys:

| Key | What it does |
|---|---|
| ↓ | From the box's last line, moves into the list; then moves to the next row |
| ↑ | Moves to the row above; from the top row, back to the box |
| Esc | In the list, goes back to the box. It takes nothing back |
| Enter | Opens the flow's view |

A flow's view fills the screen with the flow, its run id, where it stands and its log, then what
it has done so far, as `yian run` draws it, and the question it waits on, if any. It updates as
the flow goes on. It only shows the flow: you still answer through the assistant. Press Esc to go
back to the chat, which kept going meanwhile. When the chat needs you (an approval card or a
question), the view says so in its last line, and Esc takes you to it.

### What the assistant can do

| Tool | What it does |
|---|---|
| `list_flows` | Lists the `.yaml` and `.yml` files under the folder, skipping `node_modules` and dot-folders |
| `read_flow` | Reads a flow file, up to 256 KiB |
| `check_flow` | Loads a flow without running it, as `yian check` does, and lists its inputs |
| `write_flow` | Writes a flow file that loads, once you approve it |
| `read_docs` | Reads a page of these docs |
| `start_flow` | Starts a flow in the background, once you approve it |
| `flow_status`, `wait_flow` | Read where a flow it started stands, now or once it changes |
| `answer_flow` | Answers a question a flow it started asks, with your answer |
| `message_flow`, `close_inbox` | Send a message to, or close, an inbox input of a flow it started |
| `stop_flow` | Cancels a flow it started |
| `report_issue` | Helps you report something yian did wrong (see [Reporting a problem](#reporting-a-problem)) |

It also has the `composing-agents` skill, yian's guide to writing flows. Every path
stays inside the folder the chat started in: a path that leaves it, through `..`, an absolute path
or a symlink, is refused. A flow it checks or starts must find its `uses:` flows and code files
inside the folder too.

### Approvals

Before a write or a start, the chat shows a card on the terminal and waits for your choice:

- `write_flow` shows the change as a diff (on a terminal, framed, with line numbers, removed lines
  red and added lines green). Approve writes the file; Reject leaves it alone.
- `start_flow` shows the inputs and what the flow can do: its node kinds, the tools and skills its
  agents get, and the code files it runs. Approve starts it; "Approve this flow for this chat"
  also starts it again later without asking, until the flow or its code files change.

Off a terminal nothing is approved, and the assistant is told so. `report_issue` starts its flow
without a card: that flow's own card comes before anything leaves your machine.

### Flows the assistant runs

A flow the assistant starts runs in the background while you chat. The chat tells the assistant
what each one does: the questions it asks, what its nodes named `reply` say, a wait for a message,
and its output or failure. The assistant passes a question on to you and sends your answer back,
and it sends your messages to a flow that reads an inbox. A flow's `ask` never prompts on the
terminal: its question goes through the assistant.

- Each flow's run log is `~/.yian/runs/<chat run id>-r<n>/log.jsonl`, where `r1`, `r2`, ... number
  the flows the chat started.
- What the flows' `code` nodes print goes to a file beside the chat's log, named after it:
  `log.child-output.log` beside `log.jsonl`.
- A flow's agents use `--provider` and `--model` as yian run's do, and the keys they need must be set
  before it starts; the chat doesn't prompt for one. Skills are read from the skill folders as
  [`yian run`](#skills-and-mcp-servers) reads them, from the chat's folder.

After Ctrl-C or Esc, news from the flows waits for your next line and reaches the assistant just
before it. Each waiting event shows above the prompt as
`(child event waiting; sent with your next message: ...)`.

### Reporting a problem

When yian does something unexpected, the assistant offers to report it. `report_issue` starts a
flow bundled with yian, in the background like any flow the assistant runs. The flow drafts the
report (what you ran, what you expected, what happened, the error, the yian version and platform,
and the run log's path) and asks you, through the assistant, for what it lacks. It gives a run
log's path, never its contents. Then it replaces what looks like a secret with `[redacted]`: API
keys and tokens of the usual shapes (`sk-`, `ghp_`, `github_pat_`, `hf_`, long random strings),
`Authorization`, `Bearer` and cookie values, values after a name like `password=`, `API_KEY=` or
`--token`, and a URL's `user:password@`. Your home folder becomes `~`.

Last, a card shows the whole report and the ways to file it that work on your machine:

| Choice | What it does |
|---|---|
| File it with gh | Runs `gh issue create`, when the `gh` CLI is installed and logged in |
| File it with GITHUB_TOKEN | Makes one GitHub API call with the `GITHUB_TOKEN` in your environment. yian reads the token itself: the assistant never sees it, and no log holds it |
| Open a prefilled link in the browser | Gives a link to GitHub's new-issue page with the report filled in. A body too long for a link is cut, and says so |
| Save it only | Keeps the report, to send another way |
| Cancel | Files and saves nothing |

Save it only is preselected. When the card takes the input box's place while you type, it ignores
keys for a moment, so an Enter meant for the box files nothing. Every choice but Cancel also saves
the report to `~/.yian/reports/<time>.md`. The issue tracker
is private for now, so gh and the token file only for people with access to it; the link and the
saved file work for anyone.

Never paste a password or token into the chat: the model reads it, and the chat's run log keeps
it. If you pasted one, revoke it.

### Leaving the chat

When the chat ends, the flows still running are cancelled. Each flow left waiting prints its log
and how to go on:

| It waits for | What to do |
|---|---|
| A timer (a `sleep` or a timeout) | `yian run --resume LOG` continues it |
| An answer to a question, or a message | It can't be continued outside the chat yet |

Ctrl-C while the chat ends stops every flow at once, cancelled or not, and the chat still prints
where each one is: a flow stopped while it ran prints `run rN was stopped while it ran` and its log.

The chat's own run can't be resumed: `yian run --resume` on its log exits 1. It tells a chat's log
by its flow, `id: yian-chat` in a file named `chat.yaml`, so a flow of yours with both is refused
too.

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
