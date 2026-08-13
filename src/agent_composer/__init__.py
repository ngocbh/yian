"""Agent Composer — a generic runner for a *workflow of nodes*.

Owns its execution runtime (a single-writer dispatcher over a fixed worker pool),
a typed serializable variable pool, and a durable suspend/resume protocol (the
basis for HUMAN_INPUT and WATCH nodes). See the project README for status.

Layers, leaf-to-root (a package imports only DOWN/leftward — never up, never in a cycle):

    events  <-  typesys  <-  nodes  <-  compile  <-  compose
                            ^  ^
                expr  ──────┘  └──────  llm_clients      (both leaves, imported by nodes upward)

    compile.model  <-  runtime  ->  suspension
    compose + compile.expand + runtime  <-  run.py

Leaves import nothing else in the layer: `events`, `typesys` (the typed value pool), `expr`
(`${...}` evaluation), and `llm_clients` (provider client wrappers + `LLMConfig` — the AGENT
node's model seam). `nodes` are the node kinds — spawners return canonical growth seeds and
never import compile; the synthesized boundary kinds own their
reserved ids (`StartNode.ID == "__start__"`, `EndNode.ID == "__end__"`). `compile` is the
compiled `CompiledFlow` IR + representation-neutral validation checkers, and owns the
kind-aware `build_grow` constructor (and re-exports the boundary ids as
`START_ID`/`END_ID`); `compose` reads Compose-shaped YAML into a
runnable `CompiledFlow`; runtime executes `compile.model` topology but never imports
`compile.expand`; `suspension` is the checkpoint/resume protocol.
`run` (the package-root `run.py`) is the top-level run boundary: it consumes `compose`'s
`ComposedFlow`, injects `compile.expand.build_grow`, and drives the runtime engine, so it
sits above all three and is imported by none.
Nothing here imports a DB or a server; the durable (cross-process) suspend/resume
seam lives outside and is injected.
"""

from agent_composer.expr.expressions import ExpressionError, evaluate_when
from agent_composer.compile.model import CompiledFlow
from agent_composer.compile.validation import FlowValidationError
from agent_composer.compose import LoadError, ComposedFlow, load_flow
from agent_composer.run import run_flow
from agent_composer.runtime.engine import FlowEngine
from agent_composer.typesys.pool import VariablePool

__all__ = [
    "ExpressionError",
    "FlowValidationError",
    "LoadError",
    "ComposedFlow",
    "CompiledFlow",
    "FlowEngine",
    "VariablePool",
    "evaluate_when",
    "load_flow",
    "run_flow",
]
