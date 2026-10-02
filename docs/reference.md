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

::: agent_composer.run.Steering

::: agent_composer.runtime.BackwardOutcome

::: agent_composer.runtime.BackwardFailure

::: agent_composer.memory.LearningResult

::: agent_composer.memory.LearningFailure

::: agent_composer.memory.PolicyCommitOutcome

## Session memory foundation

The optional turn contracts and backend support API-backed AGENT
message conversation history through host injection into `run_flow`. Run logs
do not rebuild history. Explicit API backward self-feedback can be committed as
notes for future recall. API context summaries preserve raw history and recent
whole turns. Host-only `storage=path` optionally preserves complete backend state
across restarts and opens on first use; `storage=None` remains process-local. See
[durable memory](api.md#optional-durable-memory). `composer run` builds declared
built-in memories; `--memory-root DIR` or `--memory-storage KEY=PATH` keeps them
on disk. It does not run backward learning. See
[SessionMemory foundation](api.md#sessionmemory-foundation) for ownership,
history updates, retry, and reset semantics.

Import the concrete backend from `agent_composer.memory.session`. The
`agent_composer.memory` facade remains contracts-only; `MemorySession` is the
existing run-handle protocol, not the concrete `SessionMemory` backend.

::: agent_composer.memory.session.SessionMemory

### Optional capabilities

Each memory's recall returns text or `MemoryContent`. Rich content can supply a
transcript source alongside notes; the evaluator admits at most one transcript,
from a session memory declared on that AGENT, into one transient context.

::: agent_composer.memory.MemoryContent

::: agent_composer.memory.MemoryContextRequest

::: agent_composer.memory.MemoryContext

::: agent_composer.memory.TranscriptSource

::: agent_composer.memory.ConversationTurn

### Immutable conversation carriers

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

## Building memories

`build_memories` reads a flow's memory table and builds one backend per memory
key. See [building memories](api.md#building-memories-with-build_memories).

::: agent_composer.memory_host.build_memories

::: agent_composer.memory_host.MemoryRegistry

## Profile memory

Import the concrete backend from `agent_composer.memory.profile` and stores from
`agent_composer.memory.storage`. See [profile memory](learning.md#profile-memory).

::: agent_composer.memory.profile.ProfileMemory

::: agent_composer.memory.storage.markdown.MarkdownStore

::: agent_composer.memory.storage.markdown.FileMarkdownStore

::: agent_composer.memory.storage.markdown.MarkdownVersionMismatch

::: agent_composer.memory.storage.InMemoryMarkdownStore

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
Logs use schema 9. A log written with schema 7 or 8 is rejected on read with
"run log schema N predates schema 9 and is not supported". Accepted bytes are
preserved when new records extend a prefix. Run logs hold no conversation
history; that belongs to the memory backend.

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

::: agent_composer.runtime.contracts.ForwardCapture

::: agent_composer.runtime.contracts.MemoryProposal

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

The core ships no domain tools; a host registers its own into the shared registry
or passes the whole set for one run with `run_flow(..., tools=...)`. A list value
in that mapping is a tool group an AGENT can grant by name.

::: agent_composer.tools.register_tool

::: agent_composer.tools.registered_tools

::: agent_composer.tools.ToolSet

::: agent_composer.tools.resolve_tools

::: agent_composer.tools.invoke_tool

::: agent_composer.tools.invoke_tool_trusted

## Skills

Host-supplied `SKILL.md` skills. The host builds the mapping passed as `skills=`.

::: agent_composer.skills.Skill

::: agent_composer.skills.SkillError

::: agent_composer.compose.load_skills

::: agent_composer.compose.default_skill_roots

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
the run log. The in-loop AGENT observers below are likewise callback-only;
`IN_LOOP_EVENTS` names them.

::: agent_composer.events.NodeStarted

::: agent_composer.events.StreamChunk

::: agent_composer.events.NodeSucceeded

::: agent_composer.events.NodeRouted

::: agent_composer.events.NodeSkipped

::: agent_composer.events.NodeFailed

::: agent_composer.events.NodeExpanded

::: agent_composer.events.TokenUsage

::: agent_composer.events.ModelCallStarted

::: agent_composer.events.ModelCallFinished

::: agent_composer.events.ToolCallStarted

::: agent_composer.events.ToolCallFinished

::: agent_composer.events.IN_LOOP_EVENTS

::: agent_composer.events.RunStarted

::: agent_composer.events.RunResumed

::: agent_composer.events.RunSucceeded

::: agent_composer.events.RunFailed
