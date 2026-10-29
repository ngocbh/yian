---
title: "Agents"
description: "Agent nodes: prompts, models, tools, structured output, steering and memory."
---

This page explains `agent` nodes, the steps that call a model. You write the prompt and the
output type; yian builds the request, runs any tool calls the model makes, and checks the reply.
To choose the provider and model, see [Models and providers](/models/).

## A first agent

<!-- yian-test {"replies": ["The customer is happy with the fast delivery."]} -->

```yaml title="summarize.yaml" check=runnable
id: summarize
name: summarize
input:
  review: !str
nodes:
  summary:
    kind: agent
    input:
      review: ${input.review}
    output: str
    prompt: Summarize this review in one sentence. ${review}
output: ${summary.output}
```

```console
$ yian run summarize.yaml --input review="Arrived in two days, works great."
The customer is happy with the fast delivery.
```

An agent without `tools:` or `skills:` makes one model call, plus one for each reply it
corrects. With them, it runs a tool loop: the model calls tools, reads their results, and calls
again until it gives a final answer.

| Field | What it holds |
|---|---|
| `input` | The values the prompts read, as bindings |
| `prompt` | The user message: a template, or a list of messages |
| `system_prompt` | The system message; replaces yian's default instruction |
| `variables` | Typed local values the prompts read |
| `output` | The reply's type (default `str`) |
| `retries` | Extra attempts to correct a reply that doesn't fit `output:` (default 2) |
| `asserts` | Checks on the result (see [Expressions](/expressions/)) |
| `tools` | Tool and tool group names the model may call |
| `skills` | Skill names, or `all` |
| `steer_inbox` | An `Inbox` input whose messages steer the agent while it runs |
| `long_term_memory` | Memories the agent recalls and learns into |
| `llm_config` | Provider, model and request settings (see [Models and providers](/models/)) |
| `env` | `max_tool_iterations`, the tool loop's turn cap |

## Prompts

`prompt:` and `system_prompt:` are templates. They read the node's own `input:` names and
`variables:` by bare name, as `${review}`, never `${input.review}`; the [Expressions](/expressions/)
page covers what goes inside `${...}`. An unknown name fails at load.

```yaml
id: brief
name: brief
input:
  topic: !str
nodes:
  writer:
    kind: agent
    input:
      topic: ${input.topic}
    variables:
      sentences: !int(1..12) 3
      tone: !str plain
    system_prompt: You write short briefs for busy readers.
    prompt: Write ${sentences} ${tone} sentences about ${topic}.
    output: str
output: ${writer.output}
```

- `system_prompt:` replaces yian's default instruction; it doesn't add to it. Every model call
  of the node starts with it.
- Each `variables:` entry is a type tag and a value, checked at load. Variables can't share a
  name with an input, and can't be named `__llm_config` or `__steer_inbox`.

### A list prompt

To send a conversation as real turns, give `prompt:` a list. Each item is either `${name}`
alone, naming an input that holds a message or a list of messages, or a one-key `user:` or
`assistant:` message whose text is a template.

```yaml
id: reply
name: reply
input:
  history: !list(Message)
  message: !str
nodes:
  answer:
    kind: agent
    input:
      history: ${input.history}
      message: ${input.message}
    output: str
    prompt:
      - ${history}
      - user: ${message}
output: ${answer.output}
```

`Message` is the built-in message type (see [Types](/types/)); a record with `role` (`user` or
`assistant`) and `text` works too. The last item must be a user message. An agent's `${answer.new_messages}` holds the
messages its call added (replies, tool calls and their results), so a chat loop can append them
to its history.

## Structured output

An `output:` of `str` or `Optional(str)` keeps the reply as free text. Any other type, such as a
record, a list, a number or a `Literal(...)`, asks the model for a value of that type and checks
the reply against it.

<!-- yian-test {"replies": ["{\"name\": \"Ada\", \"score\": 9}"]} -->

```yaml title="extract.yaml" check=runnable
id: extract
name: extract
input:
  text: !str
output:
  name: str
  score: int(0..10)
kind: agent
prompt: Give the person's name and a 0-10 score for how happy they sound. ${text}
```

```console
$ yian run extract.yaml --input text="Ada loved the new release."
{"name":"Ada","score":9}
```

- yian uses the provider's structured output when it has one, and a JSON instruction otherwise.
- A reply that doesn't fit gets a correction and another try, up to `retries:` times. Then the
  node fails with `agent.output_invalid`.
- `output:` can't hold `file` or `None`, and nests at most 16 levels. Build such a value in a
  [CODE node](/code-and-tools/).

## Tools

`tools:` grants tools by name. The model sees each tool's name, description and argument schema,
and decides when to call it.

```yaml check=fragment
researcher:
  kind: agent
  input:
    question: ${input.question}
  tools: [github, ask_user]
  env:
    max_tool_iterations: 20
  prompt: Answer ${question}. Ask the user only if a detail is missing.
  output: str
```

`tools:` is always a list. A name is a single tool or a **tool group**, such as all the tools of
one MCP server. The run fails before it starts when a granted name is missing.

| Where tools come from | How |
|---|---|
| `ask_user` | Built in. The model asks the person running the flow a question, with optional answer choices. `yian run` asks in the terminal; a host must pass an `ask` handler in its tools |
| MCP servers | `yian run` reads them from config files (below). A server's name grants all its tools; a tool's own name grants just that tool |
| Host tools | A program passes them to the run (below) |

- Consecutive calls in one reply run together when their tools say that is safe. Results reach
  the model in call order.
- A tool that throws `ReportedToolError` doesn't fail the node: the model reads the message as
  the call's result and carries on. MCP tool errors work this way. Any other error fails the node.
- `max_tool_iterations` caps the model turns. The default, `-1`, means no cap. At the cap, a
  reply that still asks for tools fails the node with `agent.max_tool_iterations`. Set it in the
  node's `env:` or the flow's `env:`.

### MCP servers

`yian run` reads `~/.yian/mcp.json` when it exists, then each `--mcp-config FILE`. A later
file's entry replaces an earlier one of the same name.

```console
$ yian run research.yaml --mcp-config ./mcp.json --input question="Which issues are stale?"
```

The file uses the `mcpServers` format other MCP clients use:

```json
{
  "mcpServers": {
    "files": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "${HOME}/notes"]
    },
    "github": {
      "url": "https://mcp.example.com/github",
      "headers": { "Authorization": "Bearer ${GITHUB_TOKEN}" }
    }
  }
}
```

| Key | Meaning |
|---|---|
| `command`, `args`, `env`, `cwd` | A stdio server started as a subprocess |
| `url`, `headers` | A streamable HTTP server |
| `type` | Optional: `stdio`, `http` or `streamable-http`; must match `command` or `url` |
| `disabled` | `true` turns the server off, even one an earlier file defines |

- An entry has exactly one of `command` and `url`. SSE servers aren't supported.
- A key set to `null` counts as absent.
- A server can't share a name with a registered tool or a built-in: `take`, `sleep`, `ask`,
  `ask_user`, `load_skill`.
- In `command`, `args`, `cwd`, `url` and the values of `env` and `headers`, `${VAR}` reads an
  environment variable, `${VAR:-default}` gives a fallback, and `$$` is a literal `$`. An unset
  variable with no fallback is an error, so keep tokens in the environment, not the file.
- A stdio server gets only a minimal environment (`PATH`, `HOME` and a few more) plus its `env`.
- Servers start only when the flow needs their tools, and stop when the run ends. The config
  files are read only when the flow names a tool yian doesn't already have.
- MCP needs the optional package `@modelcontextprotocol/client` 2.3.1 installed beside yian.

### Host tools

A program that runs flows passes its own tools in `runFlow`'s `tools` option. Build a tool with
`tool()` from a JSON Schema or a Standard Schema validator (zod, valibot, ArkType):

```ts
import { loadFlow, runFlow, tool } from "@ngocbh/yian";

const search = tool({
  name: "search",
  description: "Search the docs and return the best matching passage.",
  parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  execute: async ({ query }) => lookup(String(query)),
});

const result = await runFlow(flow, { question }, { tools: { search, toolbox: [a, b] } });
```

A key holding one tool must match its name; a key holding a list is a tool group. The
`tools` option replaces the run's tools, so pass everything the flow names. See
[Embedding yian](/embedding/) for tool context, pending calls and MCP from a program.

## Skills

A skill is a folder holding a `SKILL.md` file: YAML front matter with a `description` and an optional
`name` (default: the folder's name), then Markdown instructions. Other files in the folder come with it. This is the format Claude
Code and Codex use.

```yaml check=fragment
reviewer:
  kind: agent
  skills: [code-review, release-notes]
  input:
    diff: ${input.diff}
  prompt: Review this change. ${diff}
  output: str
```

The model sees only each skill's name and description. When it needs one, it calls the built-in
`load_skill` tool, which returns the instructions, or a file in the skill's folder such as
`references/checklist.md`. Files must be UTF-8 text of at most 256 KiB; paths outside the folder
and hidden files are refused, including through symlinks. Against another process swapping
folders in a skill folder while it's read, this holds on Linux with `/proc` and with Bun on
macOS; elsewhere (Node on macOS, Windows) it is best effort. A hard link to a hidden file reads like any other file, so
don't keep hard links to secrets in a skill folder.

- `skills: [name, ...]` grants those skills; `skills: all` grants every skill the run has.
- Names use lowercase letters, digits and single hyphens, up to 64 characters.
- The run fails before it starts when a granted skill is missing.

`yian run` reads skills from these folders; the first folder with a name wins:

1. `~/.claude/skills`, then `~/.agents/skills`;
2. `.claude/skills`, then `.agents/skills`, in the current directory, then in each parent up to
   the git root (outside a git repo, the current directory only).

A broken skill is skipped with a warning. Check a skill folder with `yian skills check`:

```console
$ yian skills check skills/brief
skills/brief: ok (brief)
```

A host passes skills in `runFlow`'s `skills` option, a map built with `loadSkills(roots)` from
`@ngocbh/yian/skills`.

## Steering and interrupts

`steer_inbox:` lets someone add to an agent's task while it runs, like typing a follow-up to a
coding agent. It names one of the flow's `Inbox` inputs:

```yaml
id: researcher
name: researcher
input:
  question: !str
  notes: !Inbox(str)
nodes:
  worker:
    kind: agent
    input:
      question: ${input.question}
    steer_inbox: ${input.notes}
    prompt: Research ${question} and answer it.
    output: Optional(str)
output: ${worker.output :- "(stopped)"}
```

- **Steers.** Before each model call, the agent reads everything queued in the inbox and sends it
  with that request as a user message. A steer that arrives during the final answer gets one more
  model call, within the turn cap. An agent with [structured output](/agents/#structured-output) and
  no tools or skills reads its steers once, before its only call.
- **Render.** `steer_inbox: {from: ${input.notes}, render: "From the owner: ${event}"}` wraps each
  message in a template. The template reads only `${event}`.
- **Interrupts.** A host can interrupt every agent reading an inbox. The agent stops at its next
  safe point: before a model call, while a model call waits or thinks before its reply starts
  (the request is aborted at once), between streamed chunks, or before a tool call. A running tool call
  finishes. In `yian chat`, Ctrl-C or Esc during a turn does this.
- **The result.** An interrupted agent's output is null. With an `Optional(...)` output it
  succeeds, so the flow can route on null as above. Any other output fails the node with
  `agent.interrupted`.

The host sends steers and interrupts through the run log; see [Inboxes and waiting](/inboxes/)
and [Embedding yian](/embedding/).

## Memory

`long_term_memory:` gives an agent, or a whole flow, memories that last across runs. Each entry
names a memory and its type: `episodic`, `reflective` or `profile`.

```yaml
id: support
name: support
long_term_memory:
  notes: episodic
input:
  ticket: !str
nodes:
  answer:
    kind: agent
    input:
      ticket: ${input.ticket}
    long_term_memory:
      lessons: reflective
    prompt: Answer this support ticket. ${ticket}
    output: str
output: ${answer.output}
```

Before its first model call, the agent recalls its memories, the flow's first, then its own,
into its system message. Memories learn only when the host runs the backward pass after the
run. See [Learning](/learning/) for the memory types, feedback and storage.
