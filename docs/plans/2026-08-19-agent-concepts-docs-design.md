# Agent concepts documentation design

## Goal

Give a reader with no prior Agent Composer knowledge one clear path from the
concept of an agent, through its YAML representation, to the framework that
loads and runs it. Make each documentation page own one question and use one
vocabulary throughout.

## Canonical model

**Agent** is the conceptual umbrella, not a Python class or YAML kind. An agent
is a composable transformation from a named input record to an output value.
Agent Composer can represent that transformation as one node, a flow of nodes,
or a child flow used inside another flow.

Implementation terms remain precise:

- A **node** is one schedulable step in a flow.
- A **flow** packages a graph behind one input and output boundary.
- A **root flow** is the flow passed to `run_flow`.
- A **child flow** is a flow applied inside another by CALL, MAP, or LOOP.
  “Subflow” is introduced once as a common synonym, not used as a second type.
- An **AGENT node** or **`kind: agent`** is the LLM-backed node kind. `AgentNode`
  names only its Python class.
- A **run** is one invocation of a root flow, including expanded child flows,
  under one run log.

The definition is deliberately broader than “LLM agent.” It lets the same
typed, compositional model describe an LLM leaf, deterministic node, composite
flow, and nested flow. It does not claim that all of those are the same runtime
type.

## Agent anatomy

Every agent has a boundary and behavior:

- a named input record;
- an output value, which may be a scalar, list, record, or `None`;
- behavior implemented by one node or a graph of nodes.

An agent may additionally compose capabilities and policy:

- a task prompt and model configuration on an AGENT node;
- explicit tool grants on a tool-calling AGENT node;
- a memory policy on a flow or AGENT node;
- assertions and execution configuration;
- child agents represented by child flows.

These are optional. A CODE node has no task prompt; CASE routes instead of
producing a value; a flow can exist without memory; and prompts/tools belong to
the concrete AGENT node rather than every conceptual agent.

Nodes are **pure at the graph-state boundary**: they can perform external
effects, but they cannot mutate the variable pool, scheduler, or graph. The
author fixes the graph. LLMs generate values and may choose among granted tool
calls, but they do not rewrite the flow structure.

## Page ownership

The documentation answers questions in this order:

1. **Home — Why?** The problem, the fixed-graph approach, one short example,
   and routes into the docs.
2. **Agent — What?** The conceptual definition, anatomy, composition levels,
   canonical terminology, and three YAML representations.
3. **Architecture — How?** The end-to-end path from YAML to `RunResult`, major
   ownership boundaries, and links to deeper internals.
4. **Flow syntax — How do I author it?** The complete YAML reference.
5. **Compose — How is YAML compiled?** Parsing, lowering, wiring, validation,
   and source provenance.
6. **The engine — How does a run advance?** Scheduling, records, reduction,
   failure, recovery, growth, and learning.
7. **Runtime nodes — What does the engine execute?** `Node`, `NodeResult`, and
   traits.
8. **Python API — How does a host invoke and extend it?** Loading, running,
   tools, memory implementations, and diagnostics.

The Agent page is first under Internals, immediately before Architecture. Pages
may repeat a one-sentence local definition, but detailed explanations link to
their owner rather than being copied.

## Agent page

`docs/agent.md` is self-contained and concise:

1. one-sentence definition and a node/flow/child-flow composition diagram;
2. anatomy: boundary, behavior, and optional capabilities;
3. terminology table;
4. compact single-node YAML;
5. multi-node flow YAML;
6. named child-flow YAML through `defs:` and CALL;
7. fixed-structure and graph-boundary rules;
8. link to Architecture for execution and Syntax for the full reference.

The examples teach scale, not every field. Syntax remains the field-level
reference.

## Architecture page

`docs/architecture.md` becomes a map rather than a second Compose/Engine manual:

```text
YAML text
  -> SourceDocument (parse once)
  -> Compose (validate and lower)
  -> ComposedFlow / CompiledFlow
  -> run initialization and private graph
  -> append-record/apply-reducer engine loop
  -> primary output or expected failure
  -> optional post-terminal learning
  -> RunResult
```

The page keeps the important distinctions between source presentation,
source-neutral semantic IR, and record-authoritative runtime state. It links to
Compose for compiler passes, Engine for transition/recovery details, and Runtime
nodes for the execution contract. It calls the `VariablePool` the **variable
pool** or **typed value store**, reserving **memory** for host-injected agent
memory.

Host/control/log faults are noted as propagated exceptions rather than being
misrepresented as a failed `RunResult`. Child flows are described as graph
expansions inside the same run, never child engines or child run logs.

## Consistency pass

The rewrite applies these replacements where they describe framework concepts:

- conceptual “LLM-only agent” definitions -> umbrella definition plus AGENT-node
  qualification;
- workflow -> flow, except broad product prose;
- subflow/sub-flow/callable -> child flow;
- shared memory/run memory -> variable pool or graph state;
- pure function -> pure at the graph-state boundary;
- agent prompt -> task prompt;
- tools on an agent -> tool grants on an AGENT node;
- generic result -> node outcome, flow output, or `RunResult`.

The public docs, root README, engine design README, engine skill, and
flow-authoring skill receive the same vocabulary. This is wording and
information architecture only; runtime, YAML, and API behavior do not change.

## Verification

- Load the three Agent-page YAML shapes with the current loader.
- Search edited sources for the known conflicting phrases and inconsistent
  subflow spelling.
- Run `mkdocs build --strict` in the `agentcomposer` environment.
- Review the rendered navigation and cross-links.
- Have an independent reviewer compare the result with this design and issue
  #257 before merge.
