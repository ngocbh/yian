"""Yian — a generic runner for a *workflow of nodes*.

Owns its execution runtime (a single-writer dispatcher over a fixed worker pool),
a typed serializable variable pool, append-only run-log recovery, and blocking
host-owned human interaction. AGENT, HUMAN_INPUT, and WAIT remain live and return
ordinary outputs. See the project README for status.

Layers, leaf-to-root (a package imports only DOWN/leftward — never up, never in a cycle):

    json_codec                                      (stdlib only; below every package)
    diagnostics  <-  events
          ^
          └────────  nodes  <-  compile  <-  compose
                       ^  ^
          typesys + expr  └──────  llm_clients      (independent leaves feeding nodes)

    secret_names  <-  compose                      (stdlib-only leaf)

    compile execution contracts  <-  runtime
    compose + compile.expand + runtime  <-  run.py
    compile + compose + memory  <-  memory_host.py  <-  run.py, cli

`json_codec.py` imports only the standard library and converts values holding ints of
any size to and from JSON text, and to plain text, without changing
`sys.set_int_max_str_digits`; any layer may import it. `diagnostics.py` is the dependency-free source-neutral value spine and owns
`SourceRange`; `events` imports semantic failures from it but does not re-export
that Compose-facing coordinate type. `secret_names.py` is the stdlib-only list of
names whose literal value is a credential a flow file must not hold, kept in one
place so compose today and memory later read the same list. Other leaves import nothing upward:
`typesys` (the typed value pool), `expr` (`${...}` evaluation), and `llm_clients` (provider client wrappers +
`LLMConfig` — the AGENT node's model seam), and `skills` (host-supplied `SKILL.md` values,
confined bundled-file reads, and the AGENT's built-in `load_skill` tool; `compose.skills`
reads skill folders from disk). `nodes` are the node kinds — spawners return canonical growth seeds and
never import compile; the synthesized boundary kinds own their
reserved ids (`StartNode.ID == "__start__"`, `EndNode.ID == "__end__"`). `compile` is the
compiled `CompiledFlow` IR + representation-neutral validation checkers, and owns the
kind-aware `build_grow` constructor (and re-exports the boundary ids as
`START_ID`/`END_ID`); `compose` reads Compose-shaped YAML into a
runnable `CompiledFlow`; runtime consumes compile topology, identity, and validation
contracts but never imports `compile.expand`. `runtime.common` owns the runtime
scheduler-state vocabulary used by its reducer.
`run` (the package-root `run.py`) is the top-level run boundary: it consumes `compose`'s
`ComposedFlow`, injects `compile.expand.build_grow`, and drives the runtime engine, so it
sits above all three and is imported by none.
`memory_host` (the package-root `memory_host.py`) builds the memories a flow declares
(`build_memories`) and owns their lifetime in a `MemoryRegistry`. It reads the compiled
memory table and constructs backends; `run.py` imports it only to check a registry's
types, and the CLI calls it for `yian run`.
This package facade resolves the public names below lazily so importing a leaf
submodule does not bootstrap the whole engine.
Nothing here imports a DB or a server. Durable authority is limited to the small
runtime `RunLog` seam (memory or locked JSONL); remote host coordination lives outside.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Static aliases keep documentation and type tooling aware of the lazy facade.
if TYPE_CHECKING:
    from yian.compile.model import CompiledFlow
    from yian.compile.validation import FlowValidationError
    from yian.compose import ComposedFlow, LoadError, load_flow
    from yian.diagnostics import Failure
    from yian.expr.expressions import ExpressionError, evaluate_when
    from yian.memory_host import MemoryRegistry, build_memories
    from yian.run import run_flow
    from yian.typesys.pool import VariablePool


# Public root name -> (defining module, attribute), imported on first access.
_EXPORTS: dict[str, tuple[str, str]] = {
    "ExpressionError": ("yian.expr.expressions", "ExpressionError"),
    "FlowValidationError": (
        "yian.compile.validation",
        "FlowValidationError",
    ),
    "Failure": ("yian.diagnostics", "Failure"),
    "LoadError": ("yian.compose", "LoadError"),
    "ComposedFlow": ("yian.compose", "ComposedFlow"),
    "CompiledFlow": ("yian.compile.model", "CompiledFlow"),
    "VariablePool": ("yian.typesys.pool", "VariablePool"),
    "evaluate_when": ("yian.expr.expressions", "evaluate_when"),
    "load_flow": ("yian.compose", "load_flow"),
    "run_flow": ("yian.run", "run_flow"),
    "build_memories": ("yian.memory_host", "build_memories"),
    "MemoryRegistry": ("yian.memory_host", "MemoryRegistry"),
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
    "build_memories",
    "MemoryRegistry",
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
