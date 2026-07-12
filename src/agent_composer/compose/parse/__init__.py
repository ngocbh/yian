"""The parse phase: raw Compose text -> a strict `ComposeFile` -> typed node descriptors.

Two steps live here. `file.py` turns YAML text into a strict `ComposeFile` (the
`parse_file` entry point plus section normalization) and re-exports the `*Descriptor`
types for annotation. `nodes.py` turns a located `nodes:` mapping into typed per-kind
descriptors (`parse_nodes`) by dispatching each node body through the `KINDS` table.

Ladder position: `parse` is the third phase in the ordered chain
`helpers <- nodes <- parse <- desugar <- edges <- validate <- build <- loader`.
It imports only `helpers`, `nodes`, and the engine core (`compile.model`); nothing
imports it back except the later phases downstream of it.
"""
