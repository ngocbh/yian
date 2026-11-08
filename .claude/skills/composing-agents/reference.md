# Reference: writing yian flows

A lookup sheet for [SKILL.md](SKILL.md). Each section links the docs page with the full rules.

## Templates

Each file in [templates/](templates/) loads with `yian check`. Copy one and edit it.

| Template | Shape |
|---|---|
| [minimal.yaml](templates/minimal.yaml) | One agent, text in and out |
| [compact.yaml](templates/compact.yaml) | The same flow in compact form, with no `nodes:` |
| [typed_output.yaml](templates/typed_output.yaml) | An agent with a record `output:` (structured output, `retries:`) |
| [pipeline.yaml](templates/pipeline.yaml) | An agent drafts text; a code node parses it into a record |
| [inline-code.yaml](templates/inline-code.yaml) | An inline TypeScript `code:` body |
| [branching.yaml](templates/branching.yaml) | Classify, route with `case`, read `${route.output}` |
| [tool-use.yaml](templates/tool-use.yaml) | A host tool node feeding an agent |
| [human-in-loop.yaml](templates/human-in-loop.yaml) | An `ask` node with a typed answer |
| [human-questions.yaml](templates/human-questions.yaml) | An `ask` with `questions:` and `adaptive_questions:` |
| [sleep-until.yaml](templates/sleep-until.yaml) | A `sleep` until a time, then a node ordered after it |
| [child-summarize.yaml](templates/child-summarize.yaml) | A child flow that the next two call |
| [call-child.yaml](templates/call-child.yaml) | `call` a sibling file once through `uses:` |
| [map-fanout.yaml](templates/map-fanout.yaml) | `map` a sibling file over a list |
| [expr-and-call.yaml](templates/expr-and-call.yaml) | Arithmetic in a binding, and an inline `call(...)` of a `defs:` entry |
| [inline-composites.yaml](templates/inline-composites.yaml) | Inline `call`, `map` and `loop` bodies, no model |
| [loop.yaml](templates/loop.yaml) | A `loop` that asks a person each turn until `bye` |
| [chat.yaml](templates/chat.yaml) | A chat on an `Inbox(Message)`: `take`, a steerable reply, a history compacted each turn |
| [compact_history.yaml](templates/compact_history.yaml) | Fold older turns into a rolling summary once the history passes a budget |
| [memory.yaml](templates/memory.yaml) | An agent that reads a memory's recall and appends what it learns |
| [llm-config-cascade.yaml](templates/llm-config-cascade.yaml) | Flow `llm_config:`, a node override, a value, `inherit: false` |
| [node-env-config.yaml](templates/node-env-config.yaml) | Flow `env:` with a node override (`max_tool_iterations`) |

`call-child.yaml`, `map-fanout.yaml` and `chat.yaml` load their `uses:` siblings from the same
folder, so copy those too.

## The file

Top-level keys ([Flows and nodes](https://yian.readthedocs.io/flows/#the-file-shape)): `id` and
`name` (required), `description`, `version`, `typedefs`, `input`, `nodes` (required), `output`,
`asserts`, `defs`, `uses`, `system` (`paths:` for `uses:`), `llm_config`, `env`. Any other key is
a load error, except `x-` keys, which yian ignores.

| Flow `output:` | Returns |
|---|---|
| `${total.output}` | That value |
| `{total: "${total.output}", note: "${note.output}"}` | A record (quote each `${...}` in a `{...}`) |
| `!float ${total.output}` | One value, checked against the type |
| none | null |

Fields every node takes: `kind`, `input` (not on `case`), `asserts` (not on `case` or `map`),
`depends_on`, `runs_after`, `node_name`, `env`. `output` is a type on `agent`, `code` and `tool`.

## Node fields

**agent** ([Agents](https://yian.readthedocs.io/agents/)): `input`, `prompt` (a template, or a
list of messages), `system_prompt` (replaces the default instruction), `variables` (tagged
values the prompts read, such as `tone: !str plain`), `output`, `retries`, `tools` (tool, group
and memory input names, or `{group: [tool, ...]}`; a group's tools show as `<group>_<tool>`), `skills`
(names or `all`), `steer_inbox`, `llm_config`, `env` (`max_tool_iterations`).

**code** ([Code and tools](https://yian.readthedocs.io/code-and-tools/)): `code` (an inline body
or `./file.ts:fn`), `input`, `output`. The body reads `inputs.<name>` and returns a value.

**tool** ([Code and tools](https://yian.readthedocs.io/code-and-tools/#tool-nodes)): `tool_id`
(a tool name or `{group: tool}`), `input` (the tool's own argument names), `output`, `retries`
(default 0). The run fails before it starts when the tool is missing or an argument can never
fit.

**case** ([Control flow](https://yian.readthedocs.io/control-flow/#branch-with-case)): `on`
(optional), `cases` (a list of `{when, then}`), `else`. A `when:` or `on:` may read an agent's
`new_messages` (`when: ${len(reply.new_messages) > 1}`).

**call / map / loop** ([Control flow](https://yian.readthedocs.io/control-flow/)): `call` or
inline `nodes` plus `output`; `input` (the child's arguments, or the loop's seed record); `map`
adds `over`; `loop` adds one of `while`, `until`, `times`, and optional `max`.

**ask** ([Inboxes and waiting](https://yian.readthedocs.io/inboxes/#ask-a-person)):
`tool_id: ask`, `input` with one of `prompt` or `questions` (1 to 4, each with `question`, a
unique `header`, optional `options` (labels unique within the question) and `multi_select`),
optional `timeout`; or
`adaptive_questions: {prompt: ...}` beside `input`. A `questions:` answer is a record keyed by
`header`. Other inputs reach the host's handler, which must declare them.

**take**: `tool_id: take`, `input` with `from: ${input.<inbox>}`, optional `take:` (`one`, the
default, or `all`) and optional `timeout:`.
**sleep**: `tool_id: sleep`, `input` with exactly one of `until:` (a date or datetime, UTC when
it has no offset) or `for:` (a duration). Its output is null.

## References

[Expressions](https://yian.readthedocs.io/expressions/#references)

| Write | Reads |
|---|---|
| `input.qty` | A flow input |
| `total.output`, `total.output.amount` | A node's result, a field of it |
| `route.output` | The branch of `case` `route` that ran |
| `item` | The current item, inside a `map` body's arguments |
| `draft.new_messages` | The messages agent `draft` added: replies, tool calls and results |
| `system.today`, `system.now`, `system.run_id` | The run's date, time (UTC) and id |
| `qty` (bare) | A node's own input, in its prompts and asserts |
| `output` | A node's own result, in its asserts |

## Operators and null

From loosest to tightest: `a | b | c` (first non-null), `a :- b` (default), `a :? "msg"` (fail
with `msg` when null), `or`, `and`, `not`, comparisons (`== != < <= > >= in`, `not in`), `+ -`,
`* / %`, unary `-`, `**`, then `xs[a:b]`, calls and `.field`. Comparisons don't chain.

| Write | Gives |
|---|---|
| `x?.name` | `x.name`, or null when `x` is null |
| `x :- "none"` | `x`, or `"none"` when `x` is null |
| `(order.output?.qty :- 0) + 1` | Arithmetic on a node a `case` may skip |
| `xs[1:]`, `xs[:-2]` | A slice; no single index, no step |
| `[m.text for m in ms if m.role == "user"]` | A comprehension; more `for` and `if` may follow |
| `isinstance(x, Buy)` | Whether `x` is a valid `Buy`; narrows the reads after it |
| `Buy(x)`, `Buy(action="buy", qty=2)` | `x` checked as a `Buy`; a new `Buy` record |
| `$$` | A literal `$` |

False values are `false`, `null`, `0`, `0.0`, `""`, `[]` and an empty record. Builtins: `len`
(bindings and conditions only), `upper`, `lower`, `join(value, sep="\n")`,
`render_as_json(value, indent=2)`. A null argument fails the call: `${upper(name :- "")}`. See
[Null rules](https://yian.readthedocs.io/expressions/#null-rules).

## Types

[Types](https://yian.readthedocs.io/types/)

| Type | Notes |
|---|---|
| `str`, `int`, `float`, `bool` | `int` is any size; an int fits a `float` |
| `date`, `datetime` | Text such as `2026-10-14`, `2026-10-14T10:30:00Z` |
| `int(1..12)`, `int(0..)`, `float(0.0..1.0)` | Inclusive ranges; one end may be open |
| `object`, `dict(K, V)` | Any JSON object; `K` and `V` must name known types |
| `Any` | Any JSON value but null; fills only an `Any` slot unless tagged or narrowed |
| `None`, `Optional(X)` | Null; `X` or null |
| `list(X)` | Holds null only if `X` does: `list(Optional(int))` |
| `Literal(a, b)` | One of these texts |
| `Union(A, B)` | One of these records, told apart by a tag field or by shape |
| `file` | Fields `uri`, `mime`, `name` |
| `Message` | Built in: `role` (`user`, `assistant`, `tool`) and `parts` (text, tool calls, ...) |
| `Inbox(X)` | A flow input whose events arrive while the run goes on |
| `FileMemory` | A flow input the host binds to a memory; see [Memory](https://yian.readthedocs.io/memory/) |

Records are closed: an extra field fails the check (an agent's record reply drops it instead; a
union member's reply gets a correction). Every field is required except an `Optional(...)` one.
A typedef can't take a built-in name such as `Message`.

```yaml
typedefs:
  Buy:
    action: Literal(buy)
    qty: int
  Sell:
    action: Literal(sell)
    price: float
  Order: Union(Buy, Sell)
```

Into a child's input (`call`, `map`, `loop`), `list(int)` fits `list(Optional(int))`, a record
fits `object`, and `Literal(small)` fits `Literal(small, large)`. `str` doesn't fit a `Literal`,
`Optional(str)` doesn't fit `str`, and a union doesn't fit one of its members: tag the binding to
check it at run time, or narrow it first. A tagged loop seed allows only an added `null`: under
`acc: !int 0` the body may return `Optional(int)`, and a `null` fails the next run of the body.
Any other wider return (`str` over `!Mode a`) is a load error, tagged or not.

## Models

[Models and providers](https://yian.readthedocs.io/models/). Each agent field takes the first
value set: the agent's `llm_config:`, then each enclosing flow's outward, then
`--provider`/`--model` (or the host's), then `YIAN_DEFAULT_PROVIDER` or the saved default
login. `inherit: false` on an agent skips the flow and CLI layers.

```yaml
llm_config:
  provider: anthropic
  model: claude-sonnet-4-5
  anthropic: {max_tokens: 8000, thinking: {type: adaptive}}
  openai: {reasoning: {effort: high}}   # used only by agents that resolve to openai
```

A field may come from a value (`model: ${input.model}`); null leaves it unset. A value that an
agent in a child flow would inherit must read the root flow's inputs; otherwise pass it down as
an input.

## Chat pattern

`templates/chat.yaml` is the shape for a conversation. The flow takes
`owner_messages: !Inbox(Message)`, and a `loop` carries `{summary, history, closed}` plus the
inbox. Each turn:

1. `heard`, a `take: all` on the inbox, waits for the next messages; beside it, `compacted`
   calls `compact_history.yaml`, which folds older turns into the summary once the history
   passes a budget.
2. A condition `case` sends a non-empty `events` to `answered`, a `call` of `respond`.
3. In `respond`, the agent `reply` reads the history as a list `prompt:`, takes the inbox as
   `steer_inbox:` and has `output: Optional(str)`; the def's `output:` is
   `${[input.said + reply.new_messages]}`, this turn as one list.
4. The turn returns `history: ${compacted.output.history + (answered.output :- [])}` and
   `closed: ${heard.output.closed}`; the loop runs `while: not closed`.

The host ends the chat by closing the inbox (`/exit` in `yian run`'s chat). Try it with
`yian run chat.yaml` from the templates folder.

## Where to look next

| Need | Page |
|---|---|
| Every default and limit, failure codes | [Reference](https://yian.readthedocs.io/reference/) |
| Running from TypeScript, host tools, pending calls | [Embedding yian](https://yian.readthedocs.io/embedding/) |
| Run logs, resume, replay | [How yian works](https://yian.readthedocs.io/how-it-works/) |
| Runnable example flows | [Examples](https://yian.readthedocs.io/examples/) |
