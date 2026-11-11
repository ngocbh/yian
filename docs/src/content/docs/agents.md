---
title: "Agents"
description: "Agent nodes: prompts, models, tools, structured output and steering."
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
| `tools` | The tools the model may call: tools, tool groups and [memory](/memory/) inputs by name, or `{group: [tool, ...]}` |
| `skills` | Skill names, or `all` |
| `steer_inbox` | An `Inbox` input whose messages steer the agent while it runs |
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
- Each `variables:` entry is a type tag and a value, checked at load, or an `!include(T)` that
  reads both from a file (see [Prompts in files](#prompts-in-files)). Variables can't share a
  name with an input or start with `__`.

### Prompts in files

To keep a long prompt in its own file, read it with `!include` and a path relative to the flow
file's folder.

```md title="prompts/writer.md"
Write ${sentences} ${tone} sentences about ${topic}.
```

```yaml title="brief_file.yaml"
id: brief_file
name: brief_file
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
    prompt: !include prompts/writer.md
    output: str
output: ${writer.output}
```

- The file's text is the value, as if written there in quotes: a `${...}` in it is checked at
  load, and its final newline is kept. Its line endings read as LF. The text isn't YAML, so an
  included file can't include another.
- `!include` works on any text value. It fails at load on a key, a list or a mapping, and where
  a type goes: `input:` and `variables:` declarations, a node's `output:` type and `typedefs:`.
- The file must be UTF-8 text of at most `YIAN_MAX_INCLUDE_BYTES` (see
  [Reference](/reference/#all-defaults-and-limits)), and a load with a root folder, such as
  `yian check --root`, refuses one outside it. A missing or unreadable file fails at load, at the
  `!include`.
- A run records the text, so a resume uses the text the run started with, whatever the file
  holds by then.

To send a file exactly as written, read it with `!include(str)`. Its text is never a template:
`$$`, `${...}` and its line endings stay as they are, so a prompt an optimizer rewrites, or one
that writes math or shell, reaches the model unchanged. `!include(T)` reads a file as a value of
type `T` wherever a tag goes, as if `!T` and the value were written there. That reads few-shot
examples or a table into a variable:

```md title="prompts/tutor.md"
Write math as $$x^2$$. Shell paths start at ${HOME}.
```

```json title="examples.json"
["Q: 2+2? A: 4", "Q: 3+3? A: 6"]
```

```yaml title="tutor.yaml"
id: tutor
name: tutor
input:
  question: !str
nodes:
  tutor:
    kind: agent
    input:
      question: ${input.question}
    variables:
      examples: !include(list(str)) examples.json
    system_prompt: !include(str) prompts/tutor.md
    prompt: "Examples:\n${join(examples, '\n')}\n\nQ: ${question}"
    output: str
output: ${tutor.output}
```

- A `T` whose every value but null is text reads the file's text, as `str` does: a typedef
  naming `str`, `Optional(str)` or a `Literal`. The text is then checked against `T`.
- Any other `T` reads the file as one YAML value, so JSON works too. It can't hold a tag, so it
  can't include another file, nor an anchor, an alias, a `<<:` merge key or a repeated key. Its
  strings are never templates.
- The value is checked against `T` at load, as a literal after `!T` is. A file that doesn't hold
  one such value, or holds one that doesn't fit, fails at the tag; the error names the file, the
  line in it and the place in the value, such as `examples[2]`.
- `${examples}` writes a variable's value as it is, without reading it as a template again.
- A `T` that reads text works on any text value, as `!include` does, and gives the text exactly:
  `prompt:`, `system_prompt:`, a list prompt's messages, `code:`, `description:` and the like.
  It fails at load where the flow writes a condition, an expression or a template of its own:
  `when:`, `on:`, `while:`, `until:`, `times:`, `max:`, `asserts:`, a map's `over:`,
  `steer_inbox:` and its `render:`, and an ask's `adaptive_questions.prompt`. A case's `then:` or
  `else:` takes it in its own entry in `nodes:`.
- A `T` that reads YAML goes only where a tag goes.
- Its path, the root folder, the size cap and the record work as for `!include`.

To build a system prompt from several files, read each into a variable and join them in the
prompt; `rstrip` drops a file's final newline. The files' `$$` and `${...}` reach the model as
written:

```yaml check=fragment
variables:
  system_md: !include(str) prompts/system.md
  skill_main: !include(str) skills/main.md
system_prompt: "${rstrip(system_md)}\n\n## Skill: main\n\n${skill_main}"
```

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
to its history, and a `case` can route on them.

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
- A `date` or `datetime` field tells the model the [form](/types/) yian accepts, such as
  `2026-10-14` or `2026-10-14T10:30:00Z`.
- A reply that doesn't fit gets a correction and another try, up to `retries:` times. Then the
  node fails with `agent.output_invalid`.
- A record reply keeps only the fields its type declares, and drops any other field without a
  correction. A [union](/types/#unions) member is stricter: a field it, or a record inside it,
  doesn't declare gets a correction and another try.
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

`tools:` is always a list. A **tool group** holds several tools, such as all the tools of one MCP
server. Each entry grants one of these:

| Entry | Grants | The model sees |
|---|---|---|
| `ask_user` | One tool | `ask_user` |
| `github` | A whole group | `github_search`, `github_create_issue`, ... |
| `{github: [search]}` | Some tools of a group | `github_search` |

- A group's tools show as `<group>_<tool>`, so two groups may each have a `search`.
- A bare tool name, such as `search`, finds a group's tool when only one group has it, and keeps
  its own name. When two groups have one, grant it as `{github: [search]}`.
- A tool in two granted groups is offered once, under the first grant's name.
- The run fails before it starts when a granted name is missing or held by two groups, a group
  lacks a listed tool, or two granted tools would show one name.

| Where tools come from | How |
|---|---|
| `ask_user` | Built in. The model asks the person running the flow a question, with optional answer choices. `yian run` asks in the terminal; a host must pass an `ask` handler in its tools |
| MCP servers | `yian run` reads them from config files (below). Each server is a tool group named after it |
| Host tools | A program passes them to the run (below) |

- Consecutive calls in one reply run together when their tools say that is safe. Results reach
  the model in call order.
- A tool that throws `ReportedToolError` doesn't fail the node: the model reads the message as
  the call's result and carries on. MCP tool errors work this way. Any other error fails the node.
- `max_tool_iterations` caps the model turns. The default, `-1`, means no cap. At the cap, a
  reply that still asks for tools runs none of them, and the model answers once more. Set it in
  the node's `env:` or the flow's `env:`.

These mistakes of the model, and the cap, don't fail the node. The call never reaches a tool,
and the model reads an error result instead:

| The model | It reads |
|---|---|
| Sends arguments that aren't valid JSON | `Arguments for <tool> are not valid JSON; send them as one JSON object.` |
| Calls a tool it wasn't granted | `Unavailable tool: <tool>` |
| Sends arguments the tool's schema or validator refuses | The check's message |
| Still calls tools at `max_tool_iterations: N` | `Tool-call limit reached (N turns). No more tool calls will run; give your final answer now.` |

The model's reply after the cap is the last. If it still calls tools, they get the same result,
and its text is the output. A structured agent then makes its structured call.

A reply that gives two tool calls the same id, or whose arguments nest deeper than
`YIAN_MAX_VALUE_NESTING`, still fails the node.

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
- A model sees a server's tools as `<server>_<tool>`, so name servers with letters, digits, `_`
  and `-`. The run fails before it starts on a name a model can't call, or one past 64 characters.
- In `command`, `args`, `cwd`, `url` and the values of `env` and `headers`, `${VAR}` reads an
  environment variable, `${VAR:-default}` gives a fallback, and `$$` is a literal `$`. An unset
  variable with no fallback is an error, so keep tokens in the environment, not the file.
- A stdio server gets only a minimal environment (`PATH`, `HOME` and a few more) plus its `env`.
- A result's text that holds half a character, as text cut inside an emoji does, reaches the model
  with `�` in its place. Structured content that isn't a yian value is reported as a failed call.
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

A tool that passes `ctx.callId` to its service as an idempotency key can set `resumable: true`.
When a crash or a stop cuts off an agent's call to it before the result is recorded, the resumed
agent sends the call again, with the same `ctx.callId`, instead of telling the model the outcome
is unknown. Leave it unset (`false`) for any tool that could act twice.

A key holding one tool must match its name; a key holding a list is a tool group. Two groups may
hold tools of one name, but two tools that could show one name, such as a tool `github_search` and
group `github`'s `search`, are refused. The `tools` option replaces the run's tools, so pass
everything the flow names. See
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
and hidden files are refused, including through symlinks. A refused call doesn't fail the node:
the model gets the error and carries on.

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
  safe point: before a model call, while it waits for a reply, between streamed chunks, or before
  a tool call. A running tool call finishes. In `yian run`'s chat, Ctrl-C or Esc during a turn
  does this.
- **The result.** An interrupted agent's output is null. With an `Optional(...)` output it
  succeeds, so the flow can route on null as above. Any other output fails the node with
  `agent.interrupted`.

The host sends steers and interrupts through the run log; see [Inboxes and waiting](/inboxes/)
and [Embedding yian](/embedding/).

## Memory

An agent keeps notes across runs in a memory, a flow input such as `lessons: !FileMemory`.
`${lessons}` in a prompt renders its recall, and `tools: [{lessons: [read, append]}]` grants its
tools, which the model sees as `lessons_read` and `lessons_append`. See [Memory](/memory/).
