"""CODE node — run deterministic Python, either a `module:function` reference or inline source.

- `node.py` — the `CodeNode` + source classification; both modes run **in-process**.
- `watchdog.py` — CODE's soft-budget warning for an in-process body that may be stuck.

A killable subprocess runtime is a deferred later phase. This package knows only
the node contract and typed-value validation; it never imports runtime/compose.

Knows about: nodes.base, typesys, peer CODE modules, and the Python standard library.
Never imports: compile, compose, runtime, the run boundary, CLI, or provider SDKs.
"""

from agent_composer.nodes.code.node import CodeNode, classify_code_source

__all__ = ["CodeNode", "classify_code_source"]
