VERDICT: NEEDS_REVISION

1. **The slice-1 prerequisite is false.** The source plan explicitly leaves
   `OriginRef`, `SourceSite`, and `SourceCatalog` out of scope and deletes
   `compose/helpers/reader.py`; the rest plan says slice 1 creates
   `compose/source/catalog.py` and later modifies `NodeReader` in Steps 2G and 4B.
   Fix the prerequisite to name only `SourceDocument`, `SourceCursor`, and the
   literal parser. Create `compose/source/catalog.py` in Step 2G, add its tests
   there, and replace every post-slice-1 `NodeReader`/`helpers/reader.py` action
   with direct `SourceCursor` operations. No reader compatibility layer may be
   reintroduced.

2. **Local catalog construction is not specified.** A cursor currently has no
   `site()`/origin API, and the plan never says where cursor occurrences become
   unique `SourceSite` entries before the immutable `SourceCatalog` is frozen.
   Step 2G must define one compose-owned construction boundary (for example,
   `SourceCatalog.from_cursors(...)` or a private build accumulator), the exact
   `_MapStep(key, pair_ordinal)`/`_SeqStep(index)` to typed-key encoding, and how
   virtual overlays reuse a source cursor. Include pair ordinals in the private
   encoding so duplicate keys cannot collide, while aliases retain distinct
   occurrence keys and definition-site ranges.

3. **Resolver composition cannot prove “exactly once” as written.** The
   `OriginResolver.resolve()` protocol exposes neither keys nor multiplicity,
   and assembly currently discards a resolved child's `ComposedFlow` resolver
   when it bakes only `child.compiled` into CALL/MAP/LOOP. Define the rebased and
   composite resolver as an explicit immutable table
   `new OriginRef -> (delegate, old OriginRef)`, reject duplicate new keys at
   construction, and make the build/scope context retain every baked child's
   rebased resolver for the parent. Finalization must walk reachable origins,
   require table membership and successful resolution, and reject a resolver on
   a source-free flow. Add tests for two nodes using the same alias and shared
   compiled child, not just two different aliases.

4. **Generic origin rewriting will double-apply to shared children.** Python
   `deepcopy` preserves shared references, while two CALL nodes can point at the
   same cached child. A naive recursive walk visits that object twice and rebases
   an already-rebased key. Require identity-based visited sets in both
   `iter_flow_origins` and `rewrite_flow_origins` (with a recursion guard), and
   test repeated calls through one alias plus a diamond import. The cached child
   and each rebased copy must remain unchanged after another rewrite.

5. **The canonical `Parameter` migration omits live authorities.** Step 2F does
   not name `typesys/seeding.py` (reads `type_str`), `nodes/end/node.py`,
   `compose/nodes/{call,case}.py`, `compose/__init__.py`, or all child-required
   checks that currently use `decl.default is None`. List and migrate these
   explicitly. All omission/default checks must use `has_default`; string
   coercion must use `coerce_strings` plus `contract.type`; an untyped parameter
   may carry an arbitrary default, while a typed default is validated without
   replacing its stored Python value. Add end-to-end tests for omitted input,
   explicit `null`, nullable default `null`, and invalid non-nullable default.
   Run an `InputDecl|ParamDecl|type_str` deletion census before the Group C
   commit.

6. **#131 and #134 own the same precision work in two places.** Step 2G promises
   nested `ContractOrigins`, and Step 2F consumes `TypeCheckError.path`, but Step
   5A later introduces both recursive contract-site compilation and structured
   paths under #134. Choose one authority. To match the issue scopes, keep
   `ValueContract`/`ContractOrigins` and top-level defaults in #131, and move
   `TypeCheckError.path`, recursive `read_value_contract`, nested-origin
   selection, and their tests to Step 5A/#134. Remove the contradictory nested
   acceptance cases from Steps 2E, 2F, and 2G.

7. **The growth-failure step assigns checks to the wrong layer.** Node budget,
   reference depth, fragment topology, and prune validation currently live in
   `runtime/state.py`; `compile.expand.build_grow()` cannot translate them as
   Step 3B claims. Specify that `GrowthRejected` is the shared source-neutral
   exception contract, that `build_grow` translates only its closed seed-shape
   rejections, and that `runtime/state.py` translates only its known
   node-produced fragment/budget/depth/prune rejections. Remove the catch-all
   `_GrowBuilderError`; an arbitrary injected builder exception must escape by
   identity. Also remove the duplicated second pytest command in Step 3B.

8. **The public cutover does not include all consumers and is not tied to clear
   PR boundaries.** `cli/chat/command.py` and `cli/chat/tools.py` currently read
   `RunResult.error`, and many tests outside the listed files do too. Add a
   repository-wide consumer migration to the coordinated cutover and require a
   full suite before that commit. Define the merge sequence explicitly:
   #131 PR (Groups A-D), one coordinated #132/#133 PR (Groups E-F, fixing both
   issues), #134 PR (G), and #135 PR (H), each branched from freshly merged main.
   The #133 load-path commit may precede F only inside the unmerged coordinated
   branch. Run the full suite at every stated green commit boundary; focused
   subsets alone cannot establish that these cross-cutting IR changes are green.

9. **Several verification commands are incorrect.** Step 5D names
   `tests/engine/test_source_document.py`, but slice 1 creates
   `tests/compose/test_source_document.py`. Step 7A's broad `\.error` grep has
   many legitimate unrelated matches (for example terminal-controller error
   events) and cannot have the stated zero result. Replace it with class/field
   specific AST/import tests or explicit legacy symbols, and make each `rg`
   audit state its allowed matches. Keep `Edge.from_` out of the `.from_`
   deletion check by testing `FlowOutput` specifically.

10. **CODE precision lacks a source API.** `SourceCursor` from slice 1 exposes
    ranges but not scalar style or physical block-content line ranges, so Step
    5C would have to reach into private `_Occurrence.yaml_node`. Add a
    compose/source-owned cursor helper for scalar style and content-line ranges,
    with literal `|`, chomping, indentation, folded/quoted fallback, and virtual
    cursor tests. `CodeNode` should receive only the resulting immutable
    executable-line origin table.

11. **The Mermaid tooling does not match the final design.** Step 6A pins only
    `@mermaid-js/mermaid-cli`, while design-v2 requires exact pins for both
    `mermaid@11.12.1` and `@mermaid-js/mermaid-cli@11.12.0`. It also fails to pin
    the exact documented browser URL. Require and test
    `https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.min.js`, both exact
    package versions (no ranges), the SuperFences custom formatter, `npm ci`,
    `mmdc` validation, and `mkdocs build --strict` in CI.

12. **Foundational contracts are charged to the wrong issues.** Group A/#131
    creates and exports `Failure` (owned by #132) and `Diagnostic` (owned by
    #133), and Group F has an ambiguous `#132/#133` commit without a closing PR
    definition. In #131, add only the leaf contracts needed by origins
    (`SourceRange`, `OriginRef`, `DiagnosticFrame`, `OriginResolver`) plus the
    semantic carriers. Add `Failure` and its code vocabulary in the coordinated
    #132/#133 branch before runtime conversion, and add `Diagnostic`/`LoadError`
    there before renderer conversion. Export the final public vocabulary only at
    that branch's atomic cutover.
