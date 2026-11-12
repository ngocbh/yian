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
| [`yian memory`](#yian-memory) | List the memories runs keep, and show, undo, edit, erase, reset, re-embed or compact one |
| [`yian auth`](#yian-auth) | Set up, list, choose the default among, or remove provider logins |
| [`yian providers`](#yian-providers) | List the supported providers and their API-key variables |
| [`yian models`](#yian-models) | List the models a login or an API key offers |
| [`yian config check`](#yian-config-check) | Check an `llm_config` layer read from stdin |
| [`yian config embedding`](#yian-config-embedding) | Save the embedding model new VectorDB memories are made with |
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
| `--log` | path | Write the run log here instead of the default location. The path must not exist, and its folder must |
| `--resume` | path | Resume the run recorded in this run log, with the flow it recorded |
| `-q`, `--quiet` | | Print no banner and no per-node progress |
| `-v`, `--verbose` | | Also print each node's output as it finishes |
| `-w`, `--num-workers` | int | How many nodes may run at once. Default 8 for a fresh run; `0` runs nodes one at a time |
| `--provider` | name | Provider for agents that set none |
| `--model` | name | Model for agents that set none |
| `--mcp-config` | file | Read MCP servers from this file too. Repeat it for more |
| `--memory` | `NAME=PATH` | Bind memory input `NAME` to this folder, or `.db` store. Repeat it for more (see [Memory](#memory)) |
| `--embedding` | `provider/model` | The embedding model a new VectorDB memory is made with (see [Memory](#memory)) |
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

A name the flow doesn't declare is a usage error that lists the declared inputs; so is a value for
an inbox input. Off a terminal, missing optional inputs stay unset (their defaults apply), and
missing required inputs are a usage error that names them all.

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
path in a folder no other run uses: the run keeps its lock file and media files beside the log.

The log holds the run's inputs, node outputs, model replies and errors. Treat it as private data.
yian creates `~/.yian` and the run folders with mode `0700` and the log with mode `0600`.

`--resume PATH` continues the run recorded in that log. It runs the flow files the run started
with, not the files on disk now, so an edited flow needs a new run. It takes no `FLOW`, `--input`,
`--inputs`, `--provider`, `--model`, `--embedding` or `--log`: the log holds all of those. It does
take `-w`, `--mcp-config` and `--memory`.

- A finished run prints its recorded result again and runs nothing.
- An unfinished run retries the nodes that started but never finished. A model call or tool call
  that was cut off may run twice.
- Before it appends anything, the resume checks each provider key or saved login the run
  recorded. A missing key prompts in a terminal; a changed key, a replaced login, or a key that
  came from a host stops the resume with exit 1 and leaves the log as it was.
- The `YIAN_*` values must match the ones the run started with. Each one that differs is named,
  with the setting to use (see [Defaults and limits](/reference/#defaults-and-limits)).
- Without `-w`, a resume keeps the recorded worker count.
- Give the same `--memory` flags again: an input without one is bound to the default memory the run
  recorded. A missing memory, or another one, stops the resume. A crash resume that finds a memory
  changed since the run started says so, and the run keeps the recall it started with.

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

On a terminal, `yian run FLOW` chats with a flow that has an `Inbox(str)` or `Inbox(Message)`
input: each line you type goes to the inbox, and each node named `reply` prints its output. It
chats when stdin and stderr are a terminal, the flow has one such input (or `--inbox NAME` picks
one), it uses no MCP tool, and there is no `--json`. Otherwise a `take` on the inbox leaves the run
waiting; from a script, use [`send` and `advance`](/embedding/#inbox-calls).

```console
$ yian run chat.yaml
```

| Key | What it does |
|---|---|
| Enter | Sends the line |
| Esc | Takes back the lines the flow hasn't read yet, and stops the reply |
| Ctrl-C | Stops the reply. A second Ctrl-C within 2 seconds, or one at the prompt, cancels the run |
| `/exit`, `/quit` or Ctrl-D | Ends the chat once the flow has read your lines |

Lines you type or paste while the flow replies wait, in order, for the next prompt.

A chat that stops prints `continue with: yian run --resume LOG`; run it on a terminal to go on.

### Memory

`yian run` binds each [memory](/memory/) input. An input with no `--memory` flag gets its default
memory, kept per project: `~/.yian/memory/<project>/<flow id>/<input>`, a folder, or a `.db` file
for a VectorDBMemory. `<project>` is the flow file's folder name and a hash of its path, so each
project keeps its own. The first run makes the memory and says so on stderr:

```text
memory lessons: support-29f8843f/answer/lessons at /home/ada/.yian/memory/support-29f8843f/answer/lessons (new)
```

- `--memory NAME=PATH` binds input `NAME` to a folder of your own, or a VectorDBMemory input to a
  `.db` file. Inputs given one path share one memory.
- A default needs the flow's `id:` to be a plain name; otherwise bind each memory with `--memory`.
  An `Optional` memory input stays unbound without a flag.
- A FileMemory folder can't be `/`, your home folder, or a folder that holds `~/.yian`: binding
  adopts a folder's files, and a reset deletes them.
- A new VectorDBMemory needs an embedding model: `--embedding provider/model`, else
  `YIAN_DEFAULT_EMBEDDING`, else the one [`yian config embedding`](#yian-config-embedding) saved.
  An existing store keeps its own; [`yian memory reembed`](#yian-memory) changes it.
- `yian chat` binds the memories of the flows its assistant runs to their defaults, and takes no
  `--memory`.

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
name replaces an earlier one. It starts the servers the flow names, and for a bare tool name the
servers in file order until one lists it, and stops them when the run ends. See
[Agents](/agents/#mcp-servers) for the file format and which servers start.

### Output and progress

- The output goes to stdout: a string as it is, any other value in text form (`true`, `[1,"a"]`,
  `{"k":"v"}`).
- The banner (the flow's name, version and description) and per-node progress go to stderr. Each
  node shows a spinner, then `✓ node` or `✗ node` with the error below. Each line ends with how
  long the node has run (`✓ frame [5s]`). Off a terminal, only the finished lines print. When more
  nodes run than the screen has rows, the last spinner line counts the rest (`… and 17 more
  running`). A `console` line from a CODE body prints on its own line above the spinners.
- Warnings print as `warning: <message>` lines on stderr.
- Colour is used only on a terminal, and never when `NO_COLOR` is set.
- On a terminal, escape sequences and control characters other than newlines and tabs
  are dropped from model, tool and run output, and BIDI embeddings, overrides and isolates show as
  their code points (`\u{202e}`). In text from a flow file, a file name or a run log (the banner, a
  flow's source in an error, paths, run and node ids, `yian config check` lines) control
  characters show instead (`^[` for ESC, `^G` for BEL), and BIDI and other invisible characters
  as their code points. A CODE body's `console` output prints as it is. Piped or redirected output
  keeps every byte.

A load error shows where the flow broke: the lines around it, the line marked `❱`, and the message
below. A span of more than 10 lines shows its first and last 4 lines around a `⋮ N lines` row, and
a line wider than the terminal (200 columns when stderr isn't one) is cut with `…` around the error.
A runtime error points at the line it came from the same way. Add `--engine-trace` only to debug
yian itself.

### Stopping a run

A signal (SIGINT, SIGTERM, SIGHUP) or Ctrl-C at an `ask` prompt stops a run: `cancelled by user`,
exit 130. Once the log exists, `yian run --resume` continues it. A second signal ends at once.
With `--json`, killing `yian` in a way it can't catch (SIGKILL, an out-of-memory kill) stops the
run too; if it hasn't stopped 5 seconds later, its process ends and the run reads `crashed`.

## yian chat

`yian chat` opens a chat with yian's assistant. It finds, reads, checks, writes and runs the flows
in the folder you start it in, and answers questions from these docs. Give it a flow to talk about
that flow; start it from the flow's folder, since it reaches no file outside it.

```console
$ yian chat
$ yian chat hello.yaml
```

| Flag | Value | What it does |
|---|---|---|
| `FLOW` | path | A flow `.yaml` you want help with |
| `--log` | path | Where to write the chat's run log. The path must not exist, and its folder must |
| `--provider` | name | Provider for the assistant and its flows, when they set none |
| `--model` | name | Model for the assistant and its flows, when they set none |
| `--resume` | | Picks an earlier chat in this folder and goes on from it |
| `--continue` | | Goes on from the latest chat in this folder |
| `--engine-trace` | | On a failure, also print the engine's stack trace |

It needs a model: set one with `yian auth`, or pass `--provider` and `--model`. A chat going on from
an earlier one keeps the model that chat chose with `/model`, unless you pass `--provider` or
`--model`. A chat whose `--log` is outside `~/.yian/runs` can't be resumed. After its first
reply, a new chat names itself in a few words with one more call to the reply's model, its input
cut to 2,000 characters; `/resume` lists the chats by
name and searches their names and first lines as you type, as `/model` and `/provider` search
theirs. Off a terminal, or with `NO_COLOR`, it prints plain lines and numbered lists.

| Key | What it does |
|---|---|
| Enter | Sends the line. During a reply, the assistant reads it before its next step |
| Esc | Takes back the lines the assistant hasn't read yet, and stops the reply |
| Ctrl-C | Stops the reply. A second Ctrl-C within 2 seconds, or one at the prompt, ends the chat |
| `/` | Lists the commands |
| `/model` | Picks the model of the next replies, and of the flows the assistant starts after it, from your logins |
| `/provider` | Picks a provider, signing in to it as `yian auth` does when it has no saved login, then one of its models. A key set in the environment is saved only if you say so |
| `/resume` | Ends this chat, as `/exit` does, and goes on from an earlier chat in this folder in a new one; its waiting flows stay as they are |
| `/exit`, `/quit` or Ctrl-D | Leaves the chat |
| Ctrl-L | Rerenders the screen |
| ↓, ↑ | Move between the box and the list of running flows under it |
| Enter on a flow | Opens its view: its progress, its log and any question it waits on |
| Esc in the list or a view | Goes back to the box |

A long paste shows as a token such as `[Pasted text #1 +40 lines]`; the assistant gets all of it.

### What the assistant can do

| Tool | What it does |
|---|---|
| `list_flows`, `read_flow`, `read_docs` | Lists the flows in the folder, reads one, or reads a page of these docs |
| `check_flow` | Loads a flow or a draft without running it, as `yian check` does |
| `write_flow` | Writes a flow that loads, once you approve |
| `start_flow` | Starts a flow or a draft in the background, once you approve |
| `flow_status`, `wait_flow`, `answer_flow`, `message_flow`, `close_inbox`, `stop_flow` | Follows, answers, messages or cancels a flow it started |
| `report_issue` | Reports a problem with yian (see [Reporting a problem](#reporting-a-problem)) |
| `author_flow` | Builds or reworks a flow with you (see [Building and reviewing flows](#building-and-reviewing-flows)) |
| `review_flow` | Reviews a flow for security, reliability, cost and clarity |

It also has the `composing-agents` skill, yian's guide to writing flows.

### Approvals

A write shows its diff on a card, and a start shows the inputs and what the flow can do; each
waits for you. "Approve this flow for this chat" lets it start again without asking until the flow
or its code changes. Off a terminal nothing is approved. `report_issue`, `author_flow` and
`review_flow` start without a card, but anything they write, run or file asks on its own.

### Flows the assistant runs

A flow the assistant starts runs in the background. Its questions reach you through the
assistant, which sends your answers back. The keys its agents need must already be set, and its
memory inputs get their default memories (see [Memory](#memory)). Its log is
`~/.yian/runs/<chat run id>-r<n>/log.jsonl`.

### Building and reviewing flows

For a new flow or a major change the assistant runs `author_flow`, which asks you through the
assistant and goes step by step; small edits it makes with `write_flow`.

1. **Goal**: what the flow is for, its inputs and output, and what "done" means.
2. **Contract**: types, nodes in outline, sample inputs and the path, which you confirm.
3. **Draft**: written with the `composing-agents` skill.
4. **Check**: fixes load errors, three tries at most.
5. **Review**: runs `review_flow`.
6. **Try**: runs the draft on the samples once you approve; skipped after a high-severity security
   finding or for a flow with an inbox input.
7. **Refine**: you refine again (five rounds at most), write it, or stop.
8. **Write**: `write_flow`, with its card.

`review_flow` lists findings, most severe first, each with a severity, the node, the problem and a
fix, in four areas: security, reliability, cost and clarity.

### Reporting a problem

When yian does something unexpected, the assistant offers to report it. `report_issue` drafts the
report, asks you for what's missing, and replaces what looks like a secret with `[redacted]`. A
card then shows the report and the ways to file it that work on your machine:

| Choice | What it does |
|---|---|
| File it with gh | Runs `gh issue create` |
| File it with GITHUB_TOKEN | Files it with your `GITHUB_TOKEN`; the assistant never sees it |
| Open a prefilled link in the browser | Gives a filled-in link to GitHub's new-issue page |
| Save it only | Saves the report only |
| Cancel | Files and saves nothing |

Save it only is preselected. Every choice but Cancel saves the report to
`~/.yian/reports/<time>.md`. Never paste a password or token into the chat: the model reads it
and the run log keeps it. If you did, revoke it.

### Leaving the chat

When the chat ends, running flows are cancelled. A flow left waiting prints its log:

- waiting on a timer (a `sleep` or a timeout): `yian run --resume LOG` continues it;
- waiting on an answer or a message: it can't be continued outside the chat.

The chat itself can't be resumed.

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
| `--root` | dir | Load each flow, its `uses:` imports, its CODE files and the files it includes only from inside this folder |
| `--json` | | Print one JSON document |

A flow that doesn't load prints its path and the located error on stderr, and the command exits 1.

With `--root DIR`, each flow and each `uses:` import must be inside `DIR` (`compose.outside_root`
otherwise), a `uses:` ref loads only from the importing file's folder, resolved as `yian run`
resolves it (an absolute ref too), and a flow that sets
`system.paths` is refused. Each `./file.ts:fn` CODE reference and each file an
[`!include`](/agents/#prompts-in-files) reads must resolve inside `DIR` too, symlinks followed, as
with `loadFlow`'s `codeRoot`; one outside fails with `compose.load`.

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
| `unstarted` | The process ended before the run's inputs were recorded; `--resume` refuses it, so delete the log and start the run again |

With `--json`, the document also has `ready` and `wake_at`. `ready` is `true` when a waiting node
can wake now: every call it waits on is settled, or an interrupt stops it. `wake_at` is the
earliest deadline a waiting node waits on, or `null`. Once `wake_at` has passed, `--resume` runs
on too, though `ready` stays `false`. A log it can't read exits 1.

`yian status` with no `LOG` lists the runs in `~/.yian/runs`, oldest first: run id, status, start
time and flow file. It first updates the run index, `~/.yian/runs/index.sqlite`, which is rebuilt
from the logs and safe to delete; a damaged one is rebuilt with a warning. While another `yian`
holds the index longer than 5 seconds, it lists what the index holds, after `warning: run index is
being updated; listing may be stale`. `yian run` adds only its own run to the index. With `--json`,
each run also has its end time, result (`null` unless it succeeded), token counts and `memories`,
each memory it was bound to by input, identity and type.

A run whose log yian can't read from record N on shows `unreadable at seq N`; `yian status LOG`
says what is wrong. So does a run folder named for another run, such as a copy of one. With
`--json`, it has `unreadable: {seq, reason}` (`null` for other runs).

## yian memory

`yian memory` reads and changes the memories `yian run` keeps. Name a memory as `FLOW NAME`, the
default memory of input `NAME` of the flow file `FLOW`, or with `--at PATH`, a memory's folder or
`.db` file. A change it makes is yours: history shows it as `user` and your user name.

```console
$ yian memory list
$ yian memory show answer.yaml lessons
$ yian memory history answer.yaml lessons --limit 5
$ yian memory undo answer.yaml lessons 2001c567d78a491facd383297e13fc6b/add
$ yian memory edit --at ./memory/lessons --file notes/math.md
$ yian memory erase answer.yaml lessons MEMORY.md --text 555-0100
$ yian memory edit triage.yaml cases --add "Ask for the order id first." --fields '{"title": "refunds", "tags": ["billing"]}'
$ yian memory reembed triage.yaml cases --embedding openai/text-embedding-3-large
```

| Command | What it does |
|---|---|
| `list` | Lists the memories under `~/.yian/memory`, each with its identity, type and location |
| `show` | Shows where a memory is, its size and its recall. `--all` also prints every file, or every item |
| `history` | Lists the last changes (`--limit`, 20 by default), newest first, each with its id and author |
| `undo CHANGE` | Undoes the change with that id |
| `edit` | Opens a file (`--file`, the index by default) in `$EDITOR`, and applies what you save. A VectorDBMemory takes `--add TEXT`, `--id N` with `--text TEXT`, or `--delete N` instead; `--fields JSON` gives a typed item's fields |
| `erase TARGET` | Erases a file, or with `--text` that text in it, or a VectorDBMemory's item `TARGET`, from the memory and its history |
| `reset` | Empties a memory and its history, keeping its identity |
| `reembed` | Embeds a VectorDBMemory's items again with the `--embedding` model, which the store then keeps |
| `compact` | Rebuilds a VectorDBMemory's file, giving back the space deleted items took |

- `erase` and `edit --delete` first list the runs on this machine that used the memory, and `reset`
  the unfinished ones; then each asks. `--yes` doesn't ask, and off a terminal it is required. A
  `no` exits 1.
- `edit --id N` and `--delete N` take `--if-version V`, the item's version as `show` gave it; a
  newer one is a conflict.
- An erase doesn't reach run logs, which can hold copies of the text: delete those yourself.
- `reset` refuses while a run is running on the memory. A waiting, stopped or crashed run resumes
  with the emptied memory.
- An edit or undo that conflicts with a change made meanwhile changes nothing and exits 1; a
  FileMemory edit keeps your text in a copy and says where.
- Every command takes `--json`.

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
| `--remove` | | Delete the provider's login. Without a provider, delete the whole `--auth-file`, its default embedding model included |
| `--auth-file` | path | Use this file instead of `$YIAN_AUTH_FILE` or the default (see [Files and folders](/reference/#files-and-folders)). Its folder must be yours, mode `0700` |
| `--json` | | For `--remove` and for OpenAI device sign-in only |

Setup changes nothing until sign-in and a check of the model succeed. The first provider you save
becomes the default. Listing never prints a secret. Off a terminal, give every value the setup
would ask for.

An OpenAI API key is checked against `OPENAI_BASE_URL` when it is set.

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
| `--timeout` | seconds | Time for the whole listing, above 0 (default 8, or `YIAN_MODEL_LIST_TIMEOUT_SECONDS`) |
| `--json` | | Print one JSON document |

Give exactly one of `--auth-file` and `--provider`. The key comes from stdin, never from the
command line or the environment; an empty stdin, or a terminal, sends none. Without `--base-url`,
`--provider openai` uses `OPENAI_BASE_URL` when it is set. Azure has no model
listing. A failed listing exits 1.

## yian config check

`yian config check` reads an `llm_config` layer as a JSON object on stdin, key included, and checks
it as a run would if no flow or agent overrides its provider or model. Values never appear in its
messages.

```console
$ cat layer.json | yian config check --json
{"format":"yian.cli.v1","errors":[{"field":"openai.temperature","message":"Input should be a valid number"}],"ok":false}
```

Without `--json` it prints `ok`, or one `FIELD: MESSAGE` line per error on stderr. It exits 0 when
the layer is valid, 1 when it has errors, and 2 when stdin isn't a JSON object.

## yian config embedding

`yian config embedding` saves the embedding model a new VectorDB memory is made with when
`yian run` gets no `--embedding`. It goes in the saved-logins file, beside your logins.

```console
$ yian config embedding openai/text-embedding-3-small
saved the default embedding model openai/text-embedding-3-small
```

The provider must be `openai` or an OpenAI-compatible one, such as `openrouter` or `vllm`; its
calls use your saved API-key login or key. `YIAN_DEFAULT_EMBEDDING=provider/model` takes the saved
model's place. With no model, the command prints the default and where it comes from. yian versions
from before this command can't read a saved-logins file that holds a default embedding model.

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
| `yian memory list` | `memories` |
| `yian memory show` | `identity`, `type`, `location`, `version`, `recall`, then `characters` and `files`, or a store's `count`, `retired` and, with `--all`, `items` |
| `yian memory history` | `changes` |
| `yian memory undo`, `yian memory edit` | `change`, `version`, a store edit's `item`, and a `--delete`'s `runs` |
| `yian memory erase` | `runs`, `version`, `disabled_undos` |
| `yian memory reset` | `identity`, `runs` |
| `yian memory reembed`, `yian memory compact` | `identity`, and `embedding` for `reembed` |
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
- `yian memory show --all --json` writes its items as it reads them. If a read fails part way,
  the unfinished document's line ends, and the error document follows on the last line.

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
| `70` | A crash: an internal error, a bug in yian. Its stack is on stderr and there is no document |
| `130` | Stopped by a signal, or by Ctrl-C during the run. There is no document |

Any other exit prints no document; treat it as an internal error.

## Troubleshooting

| Problem | Try |
|---|---|
| A flow won't load | `yian check FLOW` shows the located error without running anything |
| A run stopped or crashed | `yian status LOG` shows where it stands; `yian run --resume LOG` continues it |
| A run waits | `yian status LOG --json` shows `ready` and `wake_at`; resume once it is ready or due |
| You lost a log path | `yian status` lists every run in `~/.yian/runs` |
| A memory input needs an embedding model | Pass `--embedding provider/model`, or save one with `yian config embedding` |
| An agent wrote a wrong note | `yian memory history FLOW NAME` lists the changes; `yian memory undo` takes one back |
| A model setting is refused | Check the layer with `yian config check` before you pass it to a run |
| A skill isn't found | `yian skills check DIR` shows why it doesn't load |
| A resume names a `YIAN_*` variable | Set or unset it as the message says, then resume again |
| `Credential storage at F is malformed` | Restore F from a backup, or delete its logins with `yian auth --remove --auth-file F` and log in again. A run that used F resumes only after a restore |
