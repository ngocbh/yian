"""Agent Composer — a generic runner for a *workflow of nodes*.

Owns its execution runtime (a single-writer dispatcher over a fixed worker pool),
a typed serializable variable pool, append-only run-log recovery, and blocking
host-owned human interaction. AGENT, HUMAN_INPUT, and WAIT remain live and return
ordinary outputs. See the project README for status.

Layers, leaf-to-root (a package imports only DOWN/leftward — never up, never in a cycle):

    events  <-  typesys  <-  nodes  <-  compile  <-  compose
                            ^  ^
                expr  ──────┘  └──────  llm_clients      (both leaves, imported by nodes upward)

    _state.py + compile execution contracts  <-  runtime
    compose + compile.expand + runtime  <-  run.py

`_state.py` is an internal dependency-free scheduler-state contract shared by
compile and runtime.
Leaves import nothing else in the layer: `events`, `_state`, `typesys` (the typed
value pool), `expr` (`${...}` evaluation), and `llm_clients` (provider client wrappers +
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
Nothing here imports a DB or a server. Durable authority is limited to the small
runtime `RunLog` seam (memory or locked JSONL); remote host coordination lives outside.
"""

from agent_composer.expr.expressions import ExpressionError, evaluate_when
from agent_composer.compile.model import CompiledFlow
from agent_composer.compile.validation import FlowValidationError
from agent_composer.compose import LoadError, ComposedFlow, load_flow
from agent_composer.run import run_flow
from agent_composer.typesys.pool import VariablePool

__all__ = [
    "ExpressionError",
    "FlowValidationError",
    "LoadError",
    "ComposedFlow",
    "CompiledFlow",
    "VariablePool",
    "evaluate_when",
    "load_flow",
    "run_flow",
]
