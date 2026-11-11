---
title: "Get started"
description: "Install yian, set up a model provider, and run your first flow."
---

This page takes you from installing yian to running your first flow and chatting with one.

## Install

yian is the npm package `@ngocbh/yian`. It runs on Node 22.18 or later, or Bun 1.4 or later.

Add it to a project:

```console
$ npm install @ngocbh/yian
```

With Bun, run `bun add @ngocbh/yian` instead.

To run flows from the terminal, you need the `yian` command. Pick one way:

| Way | Command | Notes |
|---|---|---|
| Install it globally | `npm i -g @ngocbh/yian` | Puts `yian` on your PATH |
| Run it once with npm | `npx @ngocbh/yian` | Downloads the package if needed |
| Run it once with Bun | `bunx --bun @ngocbh/yian` | Without `--bun`, bunx runs it with Node |
| Install from GitHub | `curl -fsSL https://raw.githubusercontent.com/ngocbh/yian/main/install.sh \| bash` | Clones yian into `~/.local/share/yian` and runs it with Bun, installing Bun if needed. Run it again to update; `uninstall.sh` removes it |
| Install a clone of the repo | `./install.sh` | Runs the checkout with Bun, installing Bun if needed. `./uninstall.sh` removes it |

```console
$ npx @ngocbh/yian --version
```

:::caution
Always spell out the package name. Where yian isn't installed, plain `npx yian` or `bunx yian`
fetches a different, unrelated package called `yian`.
:::

The rest of the docs write the command as `yian`. Run `yian --help` to list the commands.

To use tools from MCP servers, install the MCP client package where yian is installed: in the
same project, or globally (`npm i -g`) next to a global yian.

```console
$ npm install @modelcontextprotocol/client@2.3.1
```

## Set up a model provider

yian has no built-in model. Each agent step needs a provider (such as `anthropic` or `openai`)
and a model. The simplest way is to save a login once:

```console
$ yian auth anthropic
```

`yian auth` asks for the API key with a hidden prompt, unless the provider's key variable
(here `ANTHROPIC_API_KEY`) is set. It then asks which model to use, checks it, and saves both.
The first provider you set up becomes the default. Pass `--model` to skip the model prompt:

```console
$ yian auth anthropic --model claude-sonnet-4-5
$ yian auth ollama --model llama3.2:3b
```

Useful commands:

| Command | What it does |
|---|---|
| `yian providers` | Lists the providers, the login kinds each takes, and its API-key variable |
| `yian auth` | Lists your saved logins and the default |
| `yian auth openai --default` | Makes a saved provider the default |
| `yian auth openai --remove` | Deletes a saved login |

You can also skip the saved login and set the provider and model for one run with
`--provider` and `--model`, or in your shell with `YIAN_DEFAULT_PROVIDER` and
`YIAN_DEFAULT_MODEL`. A flow can name its own model too.
[Models and providers](/models/) covers all of these, including which one wins, ChatGPT
subscription logins, Azure, local models and custom endpoints.

## Write a flow

Save this as `hello.yaml`:

<!-- yian-test {"replies": ["Hello, Ada! It's lovely to meet you."]} -->

```yaml title="hello.yaml" check=runnable
id: hello
name: hello
description: Greet someone with a short, friendly one-liner.
input:
  name: !str
output: str
kind: agent
prompt: |-
  Write a short, warm one-sentence greeting addressed to ${name}.
  Just the greeting, no preamble.
```

```console
$ yian run hello.yaml --input name=Ada
Hello, Ada! It's lovely to meet you.
```

This flow is a single agent step, so it fits in one block:

| Key | Meaning |
|---|---|
| `id` | Required. The flow's identifier; in this one-step form it also names the step |
| `name` | Required. A display name |
| `description` | Says what the flow does (optional) |
| `input` | The flow's inputs and their types. `name: !str` is a required string |
| `output` | The type of the result. `str` is free text |
| `kind: agent` | The step is an agent, backed by an LLM |
| `prompt` | What the agent is asked. `${name}` reads the `name` input |

Flows with more than one step list them under `nodes:`; see [Flows and nodes](/flows/).

To check a flow without running it or calling a model, use `yian check`:

```console
$ yian check hello.yaml
```

## Run a flow

`yian run` starts the flow and prints its result:

```console
$ yian run hello.yaml --input name=Ada
```

- The result goes to stdout, as text for a `str` and as JSON for anything else.
- Progress for each step, and the path of the run log, go to stderr.
- On a terminal, `yian run` asks for each input you didn't pass. You can skip optional ones.
- Each run is logged under `~/.yian/runs/`. `yian status` lists those runs.

Common flags:

| Flag | What it does |
|---|---|
| `--input k=v`, `-i k=v` | Sets one input. Repeat it for more inputs |
| `--inputs FILE` | Reads the inputs from a JSON file |
| `--provider`, `--model` | Sets the provider and model for agents that set none |
| `--quiet`, `-q` | Hides per-step progress |
| `--verbose`, `-v` | Also prints each step's output as it finishes |
| `--log PATH` | Writes the run log to `PATH` instead of `~/.yian/runs/` |
| `--resume PATH` | Continues the run logged at `PATH`, for example after a crash or a wait |
| `--json` | Prints one JSON document with the result or the error |

A run that stops to wait, for example for a person's answer, prints the
`yian run --resume` line that continues it. The [CLI](/cli/) page lists every flag.

## Chat with yian's assistant

`yian chat` opens a chat with yian's assistant. Ask it to find, explain, check, write or run the
flows in the current folder. It asks before it writes a file or runs a flow, and it needs the
provider login you set up above.

```console
$ yian chat
$ yian chat hello.yaml
```

Give it a flow to talk about that flow. Type `/exit` or press Ctrl-D to leave. See
[`yian chat`](/cli/#yian-chat).

## Chat with a flow

A flow can hold a conversation itself. It takes an input of type `Inbox(str)` or `Inbox(Message)`,
and on a terminal `yian run` sends each line you type to it. Whatever a step named `reply` answers
is printed.

Save [`chat.yaml` and `compact_history.yaml`](/examples/#chatyaml) from the Examples page in
one folder, then run:

```console
$ yian run chat.yaml
```

Type a line and press Enter to send it; `/exit` or Ctrl-D ends the chat.
[Chatting on a terminal](/cli/#chatting-on-a-terminal) lists the other keys.

Use `--inbox NAME` when the flow has more than one inbox input. Inboxes are covered in
[Inboxes and waiting](/inboxes/).

## Next steps

- Read complete flows on the [Examples](/examples/) page.
- Learn the file shape in [Flows and nodes](/flows/).
- Give agents tools in [Agents](/agents/).
