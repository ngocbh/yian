/**
 * Immutable YAML source ownership for the Compose front end.
 *
 * This package owns one eemeli/yaml composition pass per physical document, the
 * source-navigation contracts built over its retained definition graph, and the immutable
 * document-local catalogs that resolve Compose-owned opaque origins by replaying physical
 * occurrences. Lone CRs read as LF, and mapping keys are text: a key YAML reads as non-text
 * is kept as written and reported. Unchanged source-backed text cursors expose whole-scalar
 * provenance and exact constructed-line ranges for literal blocks; catalog sites keep
 * semantic line indexes rather than coordinates. The catalog module also owns collision-free
 * Compose key encoding: document-local semantic paths, and lexical `uses:` scoping that
 * treats a child key as one opaque segment. Exact catalog membership reads only the frozen
 * key table. Later Compose phases populate the catalog builder, which this index doesn't
 * export; a builder lookup may resolve an already-registered site for a load failure without
 * copying or freezing the table. The loader alone consumes the final frozen snapshot, and
 * parser objects stay private. Isolated location-free literals may be parsed separately
 * without creating a document, and one plain scalar's text may be read on its own without
 * parsing YAML structure. Documents and isolated literals parse under eemeli/yaml's failsafe
 * schema and read plain scalars with yian's one shared table: YAML 1.2 core-schema rules plus
 * unquoted timestamps and `<<` merge keys. A parse given the text of each `!include` path
 * reads the value as that text, marked so a type position can refuse it; a path it wasn't given
 * is listed for the loader to read. An `!include(T)` reads the text exactly for `T` `str`, else
 * the value a plain YAML document holds (`SourceDocument.parseData`), tagged `T`, its strings
 * marked as never templates.
 *
 * Knows about:   yaml, defaults, diagnostics, typesys.
 * Never imports: compose helpers or phases, nodes, compile, runtime, the run boundary,
 *                CLI modules.
 */

export { SourceCatalog } from "./catalog.ts";
export type { OccurrencePath, ScalarSource, TaggedNode, TypedInclude } from "./common.ts";
export { SourceAccessError, SourceSyntaxError } from "./common.ts";
export { SourceCursor } from "./cursor.ts";
export type { IncludedValues, NonTextKey, RepeatedKey } from "./document.ts";
export { includeExample, includeTag, SourceDocument } from "./document.ts";
export { parseYamlLiteral, readPlainScalar } from "./literals.ts";
