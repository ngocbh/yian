"""The graph-stage axis of the Compose front end: one module per cross-node assembly pass.

Issue #70 splits `compose/` along two axes. The *node-kind* axis lives in
`compose/kinds/<kind>.py` (one handle per author-written `kind:`); the cross-node
*graph-stage* axis lives HERE — the passes that run over the WHOLE parsed flow to
assemble it: parse the `nodes:` mapping, desugar inline calls / adaptive questions /
case-call targets, expand case outputs, infer ordering edges, reject cycles, validate
references, and the cross-flow type/wiring checks.

Each stage consumes and produces `.range`/`field_ranges` located metadata on
descriptors/nodes (from the span tree, see `compose/reader.py` + `compose/spanloader.py`)
instead of the legacy `*_lines` maps threaded through `loader.py`. The stages are wired
together by `loader._assemble`.

Layer: sits alongside `compose.kinds` / `compose.bindings`, above the engine core; it is
imported by `compose.loader`, never the other way around. Populated during the #70 Step 7
rollout, one pass at a time.
"""

from __future__ import annotations
