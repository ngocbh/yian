# API Reference

Auto-generated from the source docstrings. For a narrative walkthrough with
examples, see [Python API](api.md); this page is the exhaustive symbol reference.

## Top-level package

These are exported from `yian`.

::: yian.load_flow

::: yian.run_flow

::: yian.Failure

::: yian.ComposedFlow

::: yian.CompiledFlow

::: yian.VariablePool

::: yian.evaluate_when

::: yian.LoadError

::: yian.FlowValidationError

::: yian.ExpressionError

## Run and recovery

The run-loop primitives live in `yian.run`.

::: yian.run.RunResult

::: yian.run.resume_flow

::: yian.run.Steering

::: yian.runtime.BackwardOutcome

::: yian.runtime.BackwardFailure

::: yian.memory.LearningResult

::: yian.memory.LearningFailure

::: yian.memory.PolicyCommitOutcome

## Session memory foundation

The optional turn contracts and backend support API-backed AGENT
message conversation history through host injection into `run_flow`. Run logs
do not rebuild history. Explicit API backward self-feedback can be committed as
notes for future recall. API context summaries preserve raw history and recent
whole turns. Host-only `storage=path` optionally preserves complete backend state
across restarts and opens on first use; `storage=None` remains process-local. See
[durable memory](api.md#optional-durable-memory). `yian run` builds declared
built-in memories; `--memory-root DIR` or `--memory-storage KEY=PATH` keeps them
on disk. It does not run backward learning. See
[SessionMemory foundation](api.md#sessionmemory-foundation) for ownership,
history updates, retry, and reset semantics.

Import the concrete backend from `yian.memory.session`. The
`yian.memory` facade remains contracts-only; `MemorySession` is the
existing run-handle protocol, not the concrete `SessionMemory` backend.

::: yian.memory.session.SessionMemory

### Optional capabilities

Each memory's recall returns text or `MemoryContent`. Rich content can supply a
transcript source alongside notes; the evaluator admits at most one transcript,
from a session memory declared on that AGENT, into one transient context.

::: yian.memory.MemoryContent

::: yian.memory.MemoryContextRequest

::: yian.memory.MemoryContext

::: yian.memory.TranscriptSource

::: yian.memory.ConversationTurn

### Immutable conversation carriers

::: yian.memory.ConversationInvocation
    options:
      merge_init_into_class: true

::: yian.memory.ConversationRequest
    options:
      merge_init_into_class: true

::: yian.memory.ConversationItem
    options:
      merge_init_into_class: true

::: yian.memory.ConversationEffectState
    options:
      merge_init_into_class: true

::: yian.memory.ConversationCompletion
    options:
      merge_init_into_class: true

::: yian.memory.ConversationApplied
    options:
      merge_init_into_class: true

::: yian.memory.ConversationView
    options:
      merge_init_into_class: true

::: yian.memory.ConversationStamp
    options:
      merge_init_into_class: true

::: yian.memory.ConversationAdmitted
    options:
      merge_init_into_class: true

::: yian.memory.ConversationEffectRequested
    options:
      merge_init_into_class: true

::: yian.memory.ConversationEffectObserved
    options:
      merge_init_into_class: true

::: yian.memory.ConversationClosed
    options:
      merge_init_into_class: true

### Pure fold and wire codecs

::: yian.memory.conversation.advance_conversation

::: yian.memory.conversation.conversation_content_digest

::: yian.memory.conversation.checkpoint_digest

::: yian.memory.conversation.conversation_view_to_json

::: yian.memory.conversation.conversation_view_from_json

::: yian.memory.conversation.conversation_checkpoint_to_json

::: yian.memory.conversation.conversation_checkpoint_from_json

## Building memories

`build_memories` reads a flow's memory table and builds one backend per memory
key. See [building memories](api.md#building-memories-with-build_memories).

::: yian.memory_host.build_memories

::: yian.memory_host.MemoryRegistry

## Profile memory

Import the concrete backend from `yian.memory.profile` and stores from
`yian.memory.storage`. See [profile memory](learning.md#profile-memory).

::: yian.memory.profile.ProfileMemory

::: yian.memory.storage.markdown.MarkdownStore

::: yian.memory.storage.markdown.FileMarkdownStore

::: yian.memory.storage.markdown.MarkdownVersionMismatch

::: yian.memory.storage.InMemoryMarkdownStore

## Compose source provenance

Compose owns immutable source views separately from the source-neutral engine.
Literal string scalars can expose exact constructed-line ranges without another
YAML parse; other scalar forms retain a whole-value fallback.

::: yian.compose.source.SourceCursor.scalar_source

::: yian.compose.source.SourceCursor.scalar_style

::: yian.compose.source.ScalarSource

## Defaults and limits

Every predefined default and limit, each overridable by a `YIAN_<NAME>` environment
variable. See [Defaults and limits](defaults.md) for the table.

::: yian.defaults

## Runtime authority

The lower-level runtime surface is deliberately scoped to
`yian.runtime`. `FlowEngine` borrows an initialized matching
`RunState`/`RunLog` pair; normal callers should prefer `run_flow` and
`resume_flow`, which own log lifecycle and replay.

::: yian.runtime.FlowEngine

::: yian.runtime.RunState

::: yian.runtime.RunStartData

::: yian.runtime.RunLog

::: yian.runtime.MemoryRunLog

::: yian.runtime.JsonlRunLog

::: yian.runtime.prepare_run_started

::: yian.runtime.commit_prepared

### Durable run records

The log stores only this closed transition vocabulary. Run success and failure
observers are derived by reducing these records; there is no separate forward
terminal record. Backward attempts have their own Started/Ready/Finished records.
Logs use schema 9. A log written with schema 7 or 8 is rejected on read with
"run log schema N predates schema 9 and is not supported". Accepted bytes are
preserved when new records extend a prefix. Run logs hold no conversation
history; that belongs to the memory backend.

::: yian.runtime.log.RecordEnvelope

::: yian.runtime.log.RunStartedRecord

::: yian.runtime.log.RunResumedRecord

::: yian.runtime.log.NodeStartedRecord

::: yian.runtime.log.NodeStepRecord

::: yian.runtime.log.NodeSucceededRecord

::: yian.runtime.log.NodeRoutedRecord

::: yian.runtime.log.NodeExpandedRecord

::: yian.runtime.log.NodeFailedRecord

::: yian.runtime.log.BackwardStartedRecord

::: yian.runtime.log.BackwardReadyRecord

::: yian.runtime.log.BackwardFinishedRecord

::: yian.runtime.log.MemoryCommitRequestedRecord

::: yian.runtime.log.MemoryCommitSucceededRecord

::: yian.runtime.log.MemoryCommitFailedRecord

::: yian.runtime.contracts.ForwardCapture

::: yian.runtime.contracts.MemoryProposal

::: yian.runtime.contracts.AttributedBackwardResult

## Nodes

Every node implements one graph-state-isolated contract: it receives a bound
input record and explicitly granted capabilities, then returns one of the closed
sum `Output | Route | Grow` without touching the variable pool. Each node
declares one member of the closed
[`NodeKind`][yian.nodes.base.NodeKind] vocabulary. Failure is a raised
exception, not a node-result variant.

::: yian.nodes.base.NodeKind

::: yian.nodes.base.Node

::: yian.nodes.base.BackwardContext

::: yian.nodes.base.BackwardResult

::: yian.nodes.base.validate_backward_result

::: yian.nodes.base.Output

::: yian.nodes.base.Route

::: yian.nodes.base.Grow

::: yian.nodes.contracts.NodeSteps

### Semantic IR carriers

These source-neutral frozen values give binding, parameter, assertion, and value
contract provenance one canonical home. Origins are optional for Python-authored
graphs. Compiled wiring, assertions, and node parameters use these carriers;
`Node.output_contract` is canonical too. Compose output derivation retains exact or
partial carriers through `compose/helpers/output_contracts.py`; record assembly
uses `PartialRecordContract` when any field is opaque, retaining the exact key set and
proven siblings through child signatures, CALL/LOOP, validation, runtime, replay,
origins, and manifests.

::: yian.nodes.contracts.Binding

::: yian.nodes.contracts.Parameter

::: yian.nodes.contracts.Assertion

::: yian.nodes.contracts.ValueContract

::: yian.nodes.contracts.PartialRecordContract

::: yian.nodes.contracts.OutputContract

::: yian.nodes.contracts.ContractOrigins

::: yian.nodes.contracts.ValuePath

::: yian.nodes.contracts.as_binding

::: yian.nodes.contracts.binding_source

### Node kinds

The authorable leaf, control, and composition kinds plus the internal-only START
and END boundaries synthesized by the loader.

::: yian.nodes.agent.node.AgentNode

::: yian.nodes.code.node.CodeNode

::: yian.nodes.tool.node.ToolNode

::: yian.nodes.tool.node.ToolAttemptsFailed

::: yian.nodes.case.node.CaseNode

::: yian.nodes.case.node.Case

::: yian.nodes.human_input.node.HumanInputNode

::: yian.nodes.wait.node.WaitNode

::: yian.nodes.start.node.StartNode

::: yian.nodes.end.node.EndNode

::: yian.nodes.call.node.CallNode

::: yian.nodes.map.node.MapNode

::: yian.nodes.loop.node.LoopNode

### Parameter binding

The node/flow split: a node declares its signature as
[`Parameter`][yian.nodes.contracts.Parameter] values (no source); the
flow owns canonical `Binding` sources, and
[`bind_params`][yian.nodes.binding.bind_params] joins them into the
typed input record handed to the node. A bound null shadows a parameter default
but remains subject to its contract; it is accepted only when the parameter is
untyped or its contract accepts null (a nullable or `None`-kind type).

::: yian.nodes.binding.bind_params

::: yian.nodes.binding.BindingError

## Tools

The core ships no domain tools; a host registers its own into the shared registry
or passes the whole set for one run with `run_flow(..., tools=...)`. A list value
in that mapping is a tool group an AGENT can grant by name. `ToolContext`,
`FunctionTool`, `function_tool`, `PermanentToolError`, and `TransientToolError` are
also exported from `yian`.

::: yian.tools.ToolContext

::: yian.tools.FunctionTool

::: yian.tools.function_tool

::: yian.tools.PermanentToolError

::: yian.tools.TransientToolError

::: yian.tools.ToolCheckError

::: yian.tools.register_tool

::: yian.tools.registered_tools

::: yian.tools.ToolSet

::: yian.tools.resolve_tools

::: yian.tools.invoke_tool

::: yian.tools.invoke_tool_trusted

## Skills

Host-supplied `SKILL.md` skills. The host builds the mapping passed as `skills=`.

::: yian.skills.Skill

::: yian.skills.SkillError

::: yian.compose.load_skills

::: yian.compose.default_skill_roots

## LLM configuration

How an AGENT node selects and builds its chat model.

::: yian.llm_clients.LLMConfig

The provider blocks of `LLMConfig`, one class per API, each linking the reference it
mirrors:

::: yian.llm_clients.provider_blocks
    options:
      members:
        - OpenAIResponsesBlock
        - OpenAISubscriptionBlock
        - AzureChatBlock
        - XAIChatBlock
        - DeepSeekChatBlock
        - QwenChatBlock
        - QwenCNChatBlock
        - GLMChatBlock
        - GLMCNChatBlock
        - MiniMaxChatBlock
        - MiniMaxCNChatBlock
        - OpenRouterChatBlock
        - VLLMChatBlock
        - AnthropicMessagesBlock
        - GeminiGenerateContentBlock
        - OllamaChatBlock
        - PROVIDER_BLOCKS
        - block_schema
        - merge_request_body

::: yian.llm_clients.model_from_config

There is no built-in provider or model. An `LLMConfig` that leaves them unset takes
`YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL`, then the login saved by `yian auth`;
with neither, `model_from_config` raises `ValueError` ("No model is set; ...").

::: yian.llm_clients.create_llm_client

::: yian.llm_clients.BaseLLMClient

::: yian.llm_clients.list_models

::: yian.llm_clients.ModelListError

## Observer events

Transition observers are published only after their corresponding record has been
appended and reduced; they are not the durable authority. `StreamChunk` is the
callback-only exception: it is delivered during evaluation and never appended to
the run log. The in-loop AGENT observers below are likewise callback-only;
`IN_LOOP_EVENTS` names them.

::: yian.events.NodeStarted

::: yian.events.StreamChunk

::: yian.events.NodeSucceeded

::: yian.events.NodeRouted

::: yian.events.NodeSkipped

::: yian.events.NodeFailed

::: yian.events.NodeExpanded

::: yian.events.TokenUsage

::: yian.events.ModelCallStarted

::: yian.events.ModelCallFinished

::: yian.events.ToolCallStarted

::: yian.events.ToolCallFinished

::: yian.events.IN_LOOP_EVENTS

::: yian.events.RunStarted

::: yian.events.RunResumed

::: yian.events.RunSucceeded

::: yian.events.RunFailed
