"""Kind-blind authoritative run state, logs, and the engine drain.

Runtime executes compiled topology by preparing one record, appending it, and only
then applying its state transition. Graph-growth construction arrives as an
injected callable from the run boundary, so runtime interprets no node kind and
owns no compile-time expansion. Run-scoped memory orchestration consumes only
source-neutral frame contracts and injected policy sessions; its learning intents
follow the same append-before-apply reducer path. Pooled drives own their workers
through a full join and signal cooperative stop before that join when dispatch
exits abnormally.
Optional conversation admission captures accepted invocation identity before
worker submission. The evaluator binds that identity with authored scope and owns
the lazy turn lifetime through wrapped execution and cleanup; memory owns history,
and nodes own provider effects.
No transcript or live lease enters a worker result or the run-state reducer.
Accepted forward captures and paired producer reads follow the same reducer
for live execution and replay; private backward handlers stay outside records.
Shared forward/backward contracts live below the log and driver;
they keep reverse-operation status separate from forward execution status.
The internal reverse fold walks accepted outcomes once without writing state,
logs, or memory. Trace-owned terminal authority and context reconstruction are
shared below that fold; durable attribution references values instead of copying
materialized downstream histories.
Legacy conversation wrappers remain readable as inert compatibility metadata.
State validates their unfinished invocation and compiled memory scope, then advances
record sequencing without retaining conversations, restoring memory, or granting effects.

Knows about:   events, memory contracts and wire codecs, typesys, nodes, and compile
               topology/identity/validation contracts.
Never imports: compile.expand, compose, the package-root run boundary, CLI, or provider
               SDKs.
"""

from agent_composer.runtime.contracts import BackwardFailure, BackwardOutcome, ForwardCapture
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.log import JsonlRunLog, MemoryRunLog, RunLog
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_run_started,
)

__all__ = [
    "BackwardFailure",
    "BackwardOutcome",
    "FlowEngine",
    "ForwardCapture",
    "JsonlRunLog",
    "MemoryRunLog",
    "RunLog",
    "RunStartData",
    "RunState",
    "commit_prepared",
    "prepare_run_started",
]
