# Agent model-context documentation design

## Goal

Rewrite `docs/agent.md` so a newcomer first understands an agent as a boundary
plus behavior, then sees the exact request Agent Composer constructs for an
LLM-backed AGENT node. Keep the page conceptual, concise, and faithful to the
current implementation.

## Page structure

Keep the existing first two paragraphs verbatim and remove the Mermaid diagram
that merely repeats them. Replace “What an agent contains” with “What an agent
is composed of”:

- the **boundary** is the named, typed input and output;
- the **behavior** is either one atomic node operation or a flow's fixed
  workflow graph;
- a child flow has the same boundary-and-graph shape inside another flow;
- the concrete AGENT node is the atomic LLM-backed behavior.

Retain the compact, multi-node, and child-flow YAML examples after this
conceptual progression, shortening repeated explanations where possible.

## Model-request diagram

Draw the model request rather than pretending the provider exposes one flat
prompt. Inside the request, show three parallel fields:

1. `messages`, whose order Agent Composer controls:
   - `SystemMessage [system prompt | outer-flow memory | nested child-flow
     memory ... | AGENT-local memory]`;
   - `HumanMessage [prompt rendered from the node's bound inputs]`;
   - later assistant tool calls and `ToolMessage` results when the tool loop
     makes another call.
2. `tools`, containing schemas resolved from explicitly granted `tools:` ids.
3. `response schema`, derived from a structured `output:` contract; a bare
   string output has no structured schema.

All three feed the provider/API serializer, which owns their eventual token or
control representation. The diagram must not claim that tool descriptions or
the response schema occur before or after memory in a token sequence.

## YAML and memory mapping

Explain that `input:` binds local values but does not independently dump them
into model messages; only values rendered through `prompt:` become the initial
human message. `mode:`, `llm_config:`, `retries:`, and `env:` control execution
rather than adding message content.

Show memory as zero or more frames ordered from the outermost memory-owning flow
through nested child flows, followed by an optional AGENT-local frame. A
`memory_policy:` declaration identifies a frame; the Python host must inject the
implementation. Recall is snapshot-isolated and automatic for the AGENT node.
Post-terminal learning affects later runs, never a later node in the same run.

Current docs must remain honest about missing surfaces: the system instruction
is framework-owned today, with authored `system_prompt:` tracked in #259. There
is no current `skills:` surface; Codex/Claude skill reuse and project-tool
discovery are tracked in #260. Put these in a short planned-extensions note,
not inside current YAML examples.

## Validation

- Load every YAML fence in `docs/agent.md` with `load_flow`.
- Build the site with `mkdocs build --strict`.
- Run focused memory/AGENT-context tests to validate factual claims.
- Run the full test suite before opening the pull request.
