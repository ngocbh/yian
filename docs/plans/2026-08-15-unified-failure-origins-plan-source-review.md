VERDICT: NEEDS_REVISION

## Summary Assessment

The plan matches issue #130 and the final ownership design, but its current order cannot produce the independent green commits it promises. The source contracts also leave immutability and error translation underspecified at two load boundaries.

## Critical Issues (must fix)

1. **Cycle 1 depends on Cycle 2.** Cycle 1 tests ranges through `document.root`, but `SourceCursor` and its navigation/range behavior are not implemented until Cycle 2. The sketched slotted `SourceDocument` also declares only `text` and `label`, leaving no slots for its retained value, YAML tree, or occurrence index. Either implement a minimal cursor with the document, or move every root/range assertion to the cursor cycle and specify the private document fields.

2. **Cycles 3–6 break the ordinary `load_flow` path between commits.** Cycle 3 changes descriptor parsing to call cursor-only normalization such as `renamed`, while `loader._nodes_reader` still returns `NodeReader` until Cycle 5. Cycle 4 then changes inline lowering to projections while the loader still supplies `NodeReader`. Conversely, Cycle 5's one-parse test includes two defs although `_def_nodes_reader` reparsing is not removed until Cycle 6. Define an explicit temporary adapter/dual-path bridge, or reorder/merge these into vertical slices that each pass at least one normal and one inline `load_flow` test; move the multiple-def parse-count assertion to the def cutover.

3. **The promised immutability has no enforcement mechanism.** A frozen dataclass and read-only index do not make the dict/list returned by `SourceDocument.value` or `SourceCursor.value` immutable, and projected display values are likewise mutable. This is especially nontrivial for recursive aliases. Specify whether values are recursively frozen, copied on exposure, or never exposed directly; then test attempted mutation of root, nested, alias, and projected values without breaking recursive-alias traversal.

4. **Malformed YAML can escape as the wrong exception after the loader cutover.** Cycle 3 translates `SourceSyntaxError` only when `parse_file` receives text. Cycle 5 has `load_flow` call `SourceDocument.parse` directly and then pass a document to `parse_file`; Cycle 7 does the same in `_document_for_path`. Both bypass that translation and can expose `SourceSyntaxError` instead of the current public `LoadError`. Add one compose-boundary document-construction helper used by root and external loads, with focused malformed-root and malformed-child tests preserving message/range behavior.

5. **Several actions are not executable 2–5 minute steps.** Implementing the loader plus recursive occurrence index (1.4), the entire cursor/overlay surface (2.3), migrating all parser tests (3.5), and building compact plus single-node-def projections (6.4) are each multi-hour units. Split them into test-first substeps with exact test names and a runnable check after each. Also replace broad staging commands such as `git add ... tests/compose tests/engine` with exact paths; this shared worktree makes those commands unsafe and they contradict the plan's own staging warning.

6. **The target API is internally incomplete.** The contract block omits `str`, `int`, `mapping`, and `sequence`, although Cycle 2 requires them and current descriptor parsing depends on those methods. It also does not define mapping-versus-sequence behavior for `children()`/`mapping_occurrences()` or how projected children retain a document/path. Finalize those signatures and semantics before implementation so the cursor and migration tests target one contract.

## Suggestions (nice to have)

None; resolve the blockers above before expanding the plan.

## Verified Claims

- The repeated parse sites cited in `parse/file.py`, `loader.py`, `helpers/source.py`, and `parse/nodes.py` exist at the stated lines.
- `compose/source/` is the correct lower-level owner, and the proposed split between document and compiled-flow caches is consistent with the current file resolver.
- The listed production, test, documentation, and skill paths exist; the proposed final PyYAML and layer ratchets fit issue #130.
