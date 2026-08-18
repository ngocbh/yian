"""Kind-blind authoritative run state, logs, and the engine drain.

Runtime executes compiled topology by preparing one record, appending it, and only
then applying its state transition. Graph-growth construction arrives as an
injected callable from the run boundary, so runtime interprets no node kind and
owns no compile-time expansion. Run-scoped memory orchestration consumes only
source-neutral frame contracts and injected policy sessions; its learning intents
follow the same append-before-apply reducer path. Pooled drives own their workers
through a full join and signal cooperative stop before that join when dispatch
exits abnormally.

Knows about:   events, memory contracts, typesys, nodes, and compile
               topology/identity/validation contracts.
Never imports: compile.expand, compose, the package-root run boundary, CLI, or provider
               SDKs.
"""

from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.log import JsonlRunLog, MemoryRunLog, RunLog
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_run_started,
)

__all__ = [
    "FlowEngine",
    "JsonlRunLog",
    "MemoryRunLog",
    "RunLog",
    "RunStartData",
    "RunState",
    "commit_prepared",
    "prepare_run_started",
]
