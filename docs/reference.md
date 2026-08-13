# API Reference

Auto-generated from the source docstrings. For a narrative walkthrough with
examples, see [Python API](api.md); this page is the exhaustive symbol reference.

## Top-level package

These are exported from `agent_composer`.

::: agent_composer.load_flow

::: agent_composer.run_flow

::: agent_composer.ComposedFlow

::: agent_composer.CompiledFlow

::: agent_composer.FlowEngine

::: agent_composer.VariablePool

::: agent_composer.evaluate_when

::: agent_composer.LoadError

::: agent_composer.FlowValidationError

::: agent_composer.ExpressionError

## Run / resume

The run-loop primitives live in `agent_composer.run`.

::: agent_composer.run.RunResult

::: agent_composer.run.resume_flow

::: agent_composer.run.resume_command

## Nodes

Every node implements one contract — a pure function of its bound input record —
and reports its kind through the closed [`NodeKind`][agent_composer.nodes.base.NodeKind]
vocabulary. A node returns one of the closed sum
`Output | Route | Pause | Grow` and never touches the variable pool.

::: agent_composer.nodes.base.NodeKind

::: agent_composer.nodes.base.Node

::: agent_composer.nodes.base.Output

::: agent_composer.nodes.base.Route

::: agent_composer.nodes.base.Pause

::: agent_composer.nodes.base.Grow

### Node kinds

The authorable leaf kinds plus the internal-only drivers and boundaries the loader
synthesizes.

::: agent_composer.nodes.agent.node.AgentNode

::: agent_composer.nodes.code.node.CodeNode

::: agent_composer.nodes.model.node.ModelNode

::: agent_composer.nodes.tool.node.ToolNode

::: agent_composer.nodes.case.node.CaseNode

::: agent_composer.nodes.case.node.Case

::: agent_composer.nodes.human_input.node.HumanInputNode

::: agent_composer.nodes.wait.node.WaitNode

::: agent_composer.nodes.start.node.StartNode

::: agent_composer.nodes.end.node.EndNode

::: agent_composer.nodes.call.node.CallNode

::: agent_composer.nodes.map.node.MapNode

### Parameter binding

The node/flow split: a node declares its signature as
[`ParamDecl`][agent_composer.nodes.binding.ParamDecl]s (no source); the flow owns
the wiring, and [`bind_params`][agent_composer.nodes.binding.bind_params] joins them
into the typed input record handed to the node.

::: agent_composer.nodes.binding.ParamDecl

::: agent_composer.nodes.binding.bind_params

::: agent_composer.nodes.binding.BindingError

## Tools

The core ships no domain tools; a host registers its own into the shared registry.

::: agent_composer.tools.register_tool

::: agent_composer.tools.resolve_tools

::: agent_composer.tools.install_ask_user

::: agent_composer.tools.invoke_tool

::: agent_composer.tools.invoke_tool_trusted

## LLM configuration

How an AGENT node selects and builds its chat model.

::: agent_composer.llm_clients.LLMConfig

::: agent_composer.llm_clients.model_from_config

::: agent_composer.llm_clients.create_llm_client

::: agent_composer.llm_clients.BaseLLMClient

The provider/model defaults used when an `LLMConfig` leaves them unset are read
from the environment.

::: agent_composer._settings.default_llm_provider

::: agent_composer._settings.default_llm_model

## Engine events

The two-tier event vocabulary. Node-level events are produced by `Node.run()`;
run-level events are streamed to the caller by `FlowEngine.run()`.

::: agent_composer.events.NodeStarted

::: agent_composer.events.StreamChunk

::: agent_composer.events.NodeSucceeded

::: agent_composer.events.NodeFailed

::: agent_composer.events.NodeExpanded

::: agent_composer.events.PauseRequested

::: agent_composer.events.RunStarted

::: agent_composer.events.RunResumed

::: agent_composer.events.RunSucceeded

::: agent_composer.events.RunFailed

::: agent_composer.events.RunPaused

::: agent_composer.events.RunAborted
