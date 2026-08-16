"""Agent Composer — a generic runner for a *workflow of nodes*.

Owns its execution runtime (a single-writer dispatcher over a fixed worker pool),
a typed serializable variable pool, append-only run-log recovery, and blocking
host-owned human interaction. AGENT, HUMAN_INPUT, and WAIT remain live and return
ordinary outputs. See the project README for status.

Layers, leaf-to-root (a package imports only DOWN/leftward — never up, never in a cycle):

    diagnostics  <-  events
          ^
          └────────  nodes  <-  compile  <-  compose
                       ^  ^
          typesys + expr  └──────  llm_clients      (independent leaves feeding nodes)

    _state.py + compile execution contracts  <-  runtime
    compose + compile.expand + runtime  <-  run.py

`diagnostics.py` is the dependency-free source-neutral value spine and owns
`SourceRange`; `events` imports semantic failures from it but does not re-export
that Compose-facing coordinate type. `_state.py` is an internal dependency-free
scheduler-state contract shared by compile and runtime.
Other leaves import nothing upward: `_state`, `typesys` (the typed value pool),
`expr` (`${...}` evaluation), and `llm_clients` (provider client wrappers +
`LLMConfig` — the AGENT node's model seam). `nodes` are the node kinds — spawners return canonical growth seeds and
never import compile; the synthesized boundary kinds own their
reserved ids (`StartNode.ID == "__start__"`, `EndNode.ID == "__end__"`). `compile` is the
compiled `CompiledFlow` IR + representation-neutral validation checkers, and owns the
kind-aware `build_grow` constructor (and re-exports the boundary ids as
`START_ID`/`END_ID`); `compose` reads Compose-shaped YAML into a
runnable `CompiledFlow`; runtime consumes compile topology, identity, and validation
contracts but never imports `compile.expand`.
`run` (the package-root `run.py`) is the top-level run boundary: it consumes `compose`'s
`ComposedFlow`, injects `compile.expand.build_grow`, and drives the runtime engine, so it
sits above all three and is imported by none.
This package facade resolves the public names below lazily so importing a leaf
submodule does not bootstrap the whole engine.
Nothing here imports a DB or a server. Durable authority is limited to the small
runtime `RunLog` seam (memory or locked JSONL); remote host coordination lives outside.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Static aliases keep documentation and type tooling aware of the lazy facade.
if TYPE_CHECKING:
    from agent_composer.compile.model import CompiledFlow
    from agent_composer.compile.validation import FlowValidationError
    from agent_composer.compose import ComposedFlow, LoadError, load_flow
    from agent_composer.diagnostics import Failure
    from agent_composer.expr.expressions import ExpressionError, evaluate_when
    from agent_composer.run import run_flow
    from agent_composer.typesys.pool import VariablePool


# Public root name -> (defining module, attribute), imported on first access.
_EXPORTS: dict[str, tuple[str, str]] = {
    "ExpressionError": ("agent_composer.expr.expressions", "ExpressionError"),
    "FlowValidationError": (
        "agent_composer.compile.validation",
        "FlowValidationError",
    ),
    "Failure": ("agent_composer.diagnostics", "Failure"),
    "LoadError": ("agent_composer.compose", "LoadError"),
    "ComposedFlow": ("agent_composer.compose", "ComposedFlow"),
    "CompiledFlow": ("agent_composer.compile.model", "CompiledFlow"),
    "VariablePool": ("agent_composer.typesys.pool", "VariablePool"),
    "evaluate_when": ("agent_composer.expr.expressions", "evaluate_when"),
    "load_flow": ("agent_composer.compose", "load_flow"),
    "run_flow": ("agent_composer.run", "run_flow"),
}

__all__ = [
    "ExpressionError",
    "FlowValidationError",
    "Failure",
    "LoadError",
    "ComposedFlow",
    "CompiledFlow",
    "VariablePool",
    "evaluate_when",
    "load_flow",
    "run_flow",
]


def __getattr__(name: str) -> Any:
    """Load one declared package-root export when callers first request it."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive attribute discovery."""
    return sorted({*globals(), *__all__})
