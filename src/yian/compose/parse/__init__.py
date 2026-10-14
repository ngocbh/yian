"""The parse phase: one source document -> a strict file and typed descriptors.

Two steps live here. `file.py` turns a `SourceDocument` into a strict `ComposeFile`
(`parse_file` also accepts text as a convenience boundary) and re-exports the
`*Descriptor` types for annotation. `nodes.py` turns a located `nodes:` mapping into
typed per-kind descriptors (`parse_nodes`) by dispatching each node body through the
`KINDS` table and handing each tagged entry (a value) to the reader its caller supplies.

Ladder position: `parse` follows `source <- helpers <- nodes` in the ordered chain
`source <- helpers <- nodes <- parse <- desugar <- edges <- validate <- build <- loader`.
It imports only `source`, `helpers`, `nodes`, and the engine core (`compile.model`,
`typesys`, `expr`); nothing imports it back except the later phases downstream of it.
"""
