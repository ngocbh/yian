"""CODE node — run deterministic Python, either a `module:function` reference or inline source.

- `node.py` — the `CodeNode`, source classification, and wrapped-line origin selector;
  both modes run **in-process**. It carries optional whole-code and ordered exact-line
  `OriginRef` provenance, owns identity-safe traceback attribution, and extends the
  node-local origin traversal/rewrite hooks for those declared slots.
- `watchdog.py` — CODE's soft-budget warning for an in-process body that may be stuck.

A killable subprocess runtime is a deferred later phase. This package knows only
the node contract and typed-value validation; it never imports runtime/compose.

Knows about: diagnostics, nodes.base, typesys, peer CODE modules, and the Python
             standard library.
Never imports: compile, compose, runtime, the run boundary, CLI, or provider SDKs.
"""

from agent_composer.nodes.code.node import (
    CodeNode,
    classify_code_source,
    select_inline_code_origin,
)

__all__ = ["CodeNode", "classify_code_source", "select_inline_code_origin"]
