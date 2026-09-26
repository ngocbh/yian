# API Reference

Auto-generated from the source docstrings. For a narrative walkthrough with
examples, see [Python API](api.md); this page is the exhaustive symbol reference.

## Top-level package

These are exported from `agent_composer`.

::: agent_composer.load_flow

::: agent_composer.run_flow

::: agent_composer.Failure

::: agent_composer.ComposedFlow

::: agent_composer.CompiledFlow

::: agent_composer.VariablePool

::: agent_composer.evaluate_when

::: agent_composer.LoadError

::: agent_composer.FlowValidationError

::: agent_composer.ExpressionError

## Run and recovery

The run-loop primitives live in `agent_composer.run`.

::: agent_composer.run.RunResult

::: agent_composer.run.resume_flow

::: agent_composer.runtime.BackwardOutcome

::: agent_composer.runtime.BackwardFailure

::: agent_composer.memory.LearningResult

::: agent_composer.memory.LearningFailure

::: agent_composer.memory.PolicyCommitOutcome

## Session memory foundation

The optional turn contracts and backend support API-backed AGENT
message conversation history through host injection
into `run_flow`. Historical
conversation RunLog wrappers remain readable as inert metadata; they do not
rebuild history. Explicit API backward self-feedback can be committed as notes
for future recall. API context summaries preserve raw history and recent whole
turns. Host-only `storage=path` optionally preserves complete backend state across
restarts; `storage=None` remains process-local. See
[durable memory](api.md#optional-durable-memory). `ac run` activates declared built-in
policies; optional `--memory-storage POLICY=PATH` preserves selected policies across
CLI processes. It does not run backward learning. See
[SessionMemory foundation](api.md#sessionmemory-foundation) for ownership,
history updates, retry, and reset semantics.

Import the concrete backend from `agent_composer.memory.session`. The
`agent_composer.memory` facade remains contracts-only; `MemorySession` is the
existing run-handle protocol, not the concrete `SessionMemory` backend.

::: agent_composer.memory.session.SessionMemory

### Optional capabilities

These extend participating implementations without changing the mandatory
`Memory`, `MemorySession`, or `MemoryFrame` protocols.

::: agent_composer.memory.ConversationFrame

::: agent_composer.memory.ConversationTurn

### Immutable conversation carriers

::: agent_composer.memory.ConversationKey
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationInvocation
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationRequest
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationItem
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationEffectState
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationCompletion
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationApplied
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationView
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationStamp
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationAdmitted
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationEffectRequested
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationEffectObserved
    options:
      merge_init_into_class: true

::: agent_composer.memory.ConversationClosed
    options:
      merge_init_into_class: true

### Pure fold and wire codecs

::: agent_composer.memory.conversation.advance_conversation

::: agent_composer.memory.conversation.conversation_content_digest

::: agent_composer.memory.conversation.checkpoint_digest

::: agent_composer.memory.conversation.conversation_view_to_json

::: agent_composer.memory.conversation.conversation_view_from_json

::: agent_composer.memory.conversation.conversation_checkpoint_to_json

::: agent_composer.memory.conversation.conversation_checkpoint_from_json

## Compose source provenance

Compose owns immutable source views separately from the source-neutral engine.
Literal string scalars can expose exact constructed-line ranges without another
YAML parse; other scalar forms retain a whole-value fallback.

::: agent_composer.compose.source.SourceCursor.scalar_source

::: agent_composer.compose.source.ScalarSource

## Runtime authority

The lower-level runtime surface is deliberately scoped to
`agent_composer.runtime`. `FlowEngine` borrows an initialized matching
`RunState`/`RunLog` pair; normal callers should prefer `run_flow` and
`resume_flow`, which own log lifecycle and replay.

::: agent_composer.runtime.FlowEngine

::: agent_composer.runtime.RunState

::: agent_composer.runtime.RunStartData

::: agent_composer.runtime.RunLog

::: agent_composer.runtime.MemoryRunLog

::: agent_composer.runtime.JsonlRunLog

::: agent_composer.runtime.prepare_run_started

::: agent_composer.runtime.commit_prepared

### Durable run records

The log stores only this closed transition vocabulary. Run success and failure
observers are derived by reducing these records; there is no separate forward
terminal record. Backward attempts have their own Started/Ready/Finished records.
New writes use schema 8. Readers retain schemas 7 and 8 with unchanged validation
for the thirteen pre-existing kinds; conversation records require schema 8.
Historical accepted bytes are preserved when new records extend an older prefix.
The four conversation wrappers are inert compatibility metadata. Runtime validates
their unfinished-start ownership and scope but retains no conversation projection,
restores no memory, and grants no external effects.

::: agent_composer.runtime.log.RecordEnvelope

::: agent_composer.runtime.log.RunStartedRecord

::: agent_composer.runtime.log.RunResumedRecord

::: agent_composer.runtime.log.NodeStartedRecord

::: agent_composer.runtime.log.NodeSucceededRecord

::: agent_composer.runtime.log.NodeRoutedRecord

::: agent_composer.runtime.log.NodeExpandedRecord

::: agent_composer.runtime.log.NodeFailedRecord

::: agent_composer.runtime.log.BackwardStartedRecord

::: agent_composer.runtime.log.BackwardReadyRecord

::: agent_composer.runtime.log.BackwardFinishedRecord

::: agent_composer.runtime.log.MemoryCommitRequestedRecord

::: agent_composer.runtime.log.MemoryCommitSucceededRecord

::: agent_composer.runtime.log.MemoryCommitFailedRecord

::: agent_composer.runtime.log.ConversationAdmittedRecord
    options:
      merge_init_into_class: true

::: agent_composer.runtime.log.ConversationEffectRequestedRecord
    options:
      merge_init_into_class: true

::: agent_composer.runtime.log.ConversationEffectObservedRecord
    options:
      merge_init_into_class: true

::: agent_composer.runtime.log.ConversationClosedRecord
    options:
      merge_init_into_class: true

::: agent_composer.runtime.contracts.ForwardCapture

::: agent_composer.runtime.contracts.FrameProposal

::: agent_composer.runtime.contracts.AttributedBackwardResult

## Nodes

Every node implements one graph-state-isolated contract: it receives a bound
input record and explicitly granted capabilities, then returns one of the closed
sum `Output | Route | Grow` without touching the variable pool. Each node
declares one member of the closed
[`NodeKind`][agent_composer.nodes.base.NodeKind] vocabulary. Failure is a raised
exception, not a node-result variant.

::: agent_composer.nodes.base.NodeKind

::: agent_composer.nodes.base.Node

::: agent_composer.nodes.base.BackwardContext

::: agent_composer.nodes.base.BackwardResult

::: agent_composer.nodes.base.validate_backward_result

::: agent_composer.nodes.base.Output

::: agent_composer.nodes.base.Route

::: agent_composer.nodes.base.Grow

### Semantic IR carriers

These source-neutral frozen values give binding, parameter, assertion, and value
contract provenance one canonical home. Origins are optional for Python-authored
graphs. Compiled wiring, assertions, and node parameters use these carriers;
`Node.output_contract` is canonical too. Compose output derivation retains exact or
partial carriers through `compose/helpers/output_contracts.py`; multi-output assembly
uses `PartialRecordContract` when any field is opaque, retaining the exact key set and
proven siblings through child signatures, CALL/LOOP, validation, runtime, replay,
origins, and manifests.

::: agent_composer.nodes.contracts.Binding

::: agent_composer.nodes.contracts.Parameter

::: agent_composer.nodes.contracts.Assertion

::: agent_composer.nodes.contracts.ValueContract

::: agent_composer.nodes.contracts.PartialRecordContract

::: agent_composer.nodes.contracts.OutputContract

::: agent_composer.nodes.contracts.ContractOrigins

::: agent_composer.nodes.contracts.ValuePath

::: agent_composer.nodes.contracts.as_binding

::: agent_composer.nodes.contracts.binding_source

### Node kinds

The authorable leaf, control, and composition kinds plus the internal-only START
and END boundaries synthesized by the loader.

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

::: agent_composer.nodes.loop.node.LoopNode

### Parameter binding

The node/flow split: a node declares its signature as
[`Parameter`][agent_composer.nodes.contracts.Parameter] values (no source); the
flow owns canonical `Binding` sources, and
[`bind_params`][agent_composer.nodes.binding.bind_params] joins them into the
typed input record handed to the node. A bound null shadows a parameter default
but remains subject to its contract; it is accepted only when the parameter is
untyped or its contract accepts null (a nullable or `None`-kind type).

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

## Observer events

Transition observers are published only after their corresponding record has been
appended and reduced; they are not the durable authority. `StreamChunk` is the
callback-only exception: it is delivered during evaluation and never appended to
the run log.

::: agent_composer.events.NodeStarted

::: agent_composer.events.StreamChunk

::: agent_composer.events.NodeSucceeded

::: agent_composer.events.NodeRouted

::: agent_composer.events.NodeSkipped

::: agent_composer.events.NodeFailed

::: agent_composer.events.NodeExpanded

::: agent_composer.events.RunStarted

::: agent_composer.events.RunResumed

::: agent_composer.events.RunSucceeded

::: agent_composer.events.RunFailed
