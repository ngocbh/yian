# Implementation Plan: Single-Parse Compose Source Foundation

**Goal**: Parse each physical YAML file exactly once into an immutable
`SourceDocument`, navigate it through occurrence-aware `SourceCursor`s, and make
root flows, defs, external files, compact/single-node forms, and inline composite
bodies reuse that document without changing compiled or run behavior.

**Architecture**: A new chartered `compose/source/` package owns PyYAML,
`SourceDocument`, and `SourceCursor`. The constructed YAML graph is private.
Public value access returns recursive-alias-preserving defensive copies; compose
passes use cursors and an explicit package-private read-only view, except for one
documented graph-wide inline-lowering snapshot that preserves alias sharing. A
temporary `as_cursor(NodeReader)` bridge keeps every migration commit green and
is deleted with `NodeReader`/`SpanLoader`. `_LoadCtx` caches each successfully
compiled external flow by resolved physical path.

**Tech stack**: Python 3.10+, PyYAML, Pydantic, pytest, MkDocs Material.

**Tracking**: [issue #130](https://github.com/ngocbh/agent-composer/issues/130),
delivery slice 1 of
`docs/plans/2026-08-15-unified-failure-origins-design-v2.md`.

## Changes made after plan review

This revision resolves every critical item in
`docs/plans/2026-08-15-unified-failure-origins-plan-source-review.md`:

- `SourceDocument` starts cycle 1 with explicit private slots `_value`,
  `_root_node`, and an empty `_index`; cycle 2 replaces that placeholder with a
  linear definition DAG. Cursor navigation and ranges begin in cycle 2.
- Immutability is enforceable: the internal graph is never public,
  `SourceCursor.value`/`get`/`mapping`/`sequence` return defensive snapshots,
  recursive aliases remain recursive, `_value_view()` is read-only, and virtual
  overlays snapshot/freeze inputs. Root, nested, alias, and overlay mutation tests
  are required.
- The cursor contract now includes all typed accessors and precise mapping,
  sequence, occurrence, and projected-child semantics.
- The production migration is vertical. One temporary
  `compose.helpers.reader.as_cursor` adapter converts remaining legacy readers
  without reparsing. Parser, inline lowering, pipeline, and root loader changes
  land in one green commit; defs then cut over in the next green commit.
- Multiple-def parse counting moved to the def cutover.
- `parse_source_document(text, label)` is the one compose-boundary translator
  from `SourceSyntaxError` to current `LoadError`; root and external loading both
  use it.
- Former multi-hour actions are split into named 2-5 minute test/implementation
  steps with a runnable checkpoint after each small cluster.
- Every staging command names exact files. There is no broad `git add` over test
  or source directories.

## Correction found during cycle-1 adversarial review

The original cycle-2 plan eagerly materialized every occurrence path. That is
unsafe for a compact binary alias DAG: PyYAML retains shared definitions, but an
eager occurrence tree can grow exponentially. The implementation instead interns
each unique YAML node once in a read-only definition DAG. A `SourceCursor`
creates a persistent path link only for a branch compose actually visits, and
materializes its `OccurrencePath` tuple only when an origin is registered.

This correction also closes three boundary ambiguities before implementation:

- `mapping_occurrences()` uses a private no-filter sentinel, so
  `mapping_occurrences(None)` can select an authored YAML null key;
- package-private views cover SafeLoader's mutable `set` values as well as
  mapping/list/tuple containers; virtual/projected values accept only the exact
  recursive SafeLoader value algebra and reject custom objects or container
  subclasses;
- physical occurrence-key encoding distinguishes `None`, exact `bool`, exact
  `int`, and `str`. In particular, YAML 1.1's bare `on:` key constructs as
  `True` and must not collide with integer key `1`.

## Correction found during cycle-3 adversarial review

The first definition DAG still inherited an exponential representation from
`SafeLoader.flatten_mapping`: a short chain that merged the same alias twice per
level materialized every logical inherited pair before the DAG was built. The
root-flow cutover therefore stays uncommitted until the source loader and cursor
use a compact merge representation:

- the private safe loader simulates PyYAML's delete-before-recursion merge
  semantics as a persistent pair rope, without mutating `MappingNode.value` or
  reading unfilled alias placeholders; this also preserves recursive merge
  aliases and their insertion order;
- each mapping definition references an identity-memoized arc rope whose leaves
  retain authored pairs and whose concatenations store child offsets, a logical
  flattened count, and one effective provenance index;
- effective lookup uses that index, while raw flattened occurrences are
  generated only when a caller explicitly asks for them; logical pair ordinals
  still make occurrence paths replayable;
- projected mappings have their own private key index, and inline lowering takes
  one graph-wide value snapshot so unchanged aliases retain sharing.

Retained source metadata is therefore linear in the unique YAML node/merge graph
(plus the ordinary constructed value graph). Explicit occurrence enumeration is
output-sensitive: asking for `F` logical flattened pairs may return `F` cursors,
but loading and ordinary effective navigation do not materialize them.

## Correction found during cycle-5 verification

The proposed separate external-document cache had no consumer. The existing
resolved-path compiled-flow cache already guards the read, parse, and compile as
one successful operation, so repeated aliases and diamond imports read and parse
each external path once. Retaining a second document cache would duplicate state
without changing behavior. Cycle 5 therefore types/renames the single flow cache
and adds exact per-path read/parse ratchets instead.

## Correction found during final adversarial review

Two successful-load stress cases exposed work hidden by the compact source
representation:

- recursive validation plus `copy.deepcopy` failed on valid value graphs near
  Python's recursion limit. One iterative safe-value copier now validates exact
  SafeLoader types, allocates mutable cycle placeholders, finalizes immutable
  containers bottom-up, and preserves shared and recursive aliases;
- constructing and indexing a binary merge DAG rescanned every already-completed
  descendant rope. Both folds now stop at cached mapping-root summaries while
  leaving transient recursive-merge ropes uncached, preserving linear metadata
  for wide self-merges.

The final ratchets use 2,000-level value graphs and structural rope-operation
counts rather than wall-clock thresholds. Inline-body extraction remains the one
intentional internal use of the public value snapshot: it copies the nodes graph
once so unchanged aliases remain shared while projections are built.

## Scope

In scope:

- one full-document parse per physical source;
- occurrence paths that distinguish alias appearances while ranges point at the
  anchor definition;
- last-effective duplicate/merge lookup with every raw pair still navigable;
- immutable virtual views for singular/plural keys, CASE's YAML-1.1 `on`, inline
  bodies, compact flows, and single-node defs;
- one successful compiled-flow cache per resolved external path;
- deletion of `NodeReader`, `SpanLoader`, and full-document reparse helpers;
- preservation of current descriptors, `SourceFrame`, `SourceRange`, compiled
  graphs, fingerprints, runtime results, and current error presentation;
- package charters, docs, and engine/structure skills.

Out of scope:

- `OriginRef`, `SourceSite`, `SourceCatalog`, and origin-bearing IR (slice 2);
- canonical `Failure` or run-record changes (slice 3);
- final `LoadError(Diagnostic)` and CLI diagnostic cutover (slice 4);
- inline CODE executable-line origins (slice 5);
- final Mermaid diagrams and CI configuration (slice 6).

The current `SourceFrame`/concrete-range path remains the temporary compatibility
authority. This slice changes how source is parsed and navigated, not what the
engine receives.

## Verified current evidence

- `compose/parse/file.py:238-251` calls `yaml.safe_load` and then a separate
  `yaml.compose` through `_top_level_lines`.
- `compose/loader.py:60-99` reparses for root nodes and every multi-node def.
- `compose/loader.py:225-247` reparses again to locate a reserved callable name.
- `compose/helpers/source.py:34-95` reparses independently for boundary inputs,
  boundary asserts, and section lines.
- `compose/parse/nodes.py:142-162` mutates constructed mappings and YAML key nodes
  for `input`/`output` normalization.
- `compose/desugar/inline_bodies.py:191-248` copies and mutates YAML wrappers.
- This pre-change baseline passed 52 tests:

  ```bash
  conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
    tests/compose/test_spanloader.py \
    tests/compose/test_reader.py \
    tests/compose/test_parse.py \
    tests/engine/test_defs.py \
    tests/engine/test_uses_files.py
  ```

## Final source contracts

### `SourceDocument`: private retained graph

`src/agent_composer/compose/source/document.py` defines:

```python
@dataclass(frozen=True, slots=True)
class SourceDocument:
    text: str
    label: str
    _value: Any = field(repr=False, compare=False)
    _root_node: Any = field(repr=False, compare=False)
    _definitions: tuple[_Definition, ...] = field(repr=False, compare=False)
    _root_definition: int = field(repr=False, compare=False)

    @classmethod
    def parse(cls, text: str, label: str = "<flow>") -> "SourceDocument": ...

    @property
    def root(self) -> "SourceCursor": ...
```

There is deliberately no public `value`, `root_node`, `definitions`, or spans property.
Only `text`, `label`, and the safe `root` cursor are public. The temporary
no-reparse legacy constructor was deleted with its adapter.

`_DocumentLoader` composes once and constructs against the same node tree. A
temporary `id(yaml_node) -> constructed value` table then interns each unique
node into one frozen `_ScalarDef | _MappingDef | _SequenceDef`. Mapping
definitions retain an identity-shared persistent arc rope: leaves hold authored
non-merge pairs, while concatenations hold child offsets and a logical flattened
count. A private effective-key index retains last-winner provenance without
expanding raw occurrences. Definition IDs are assigned before descent, so
recursive aliases terminate and alias DAG storage stays linear. Temporary
construction and node-to-definition maps are discarded before the loader is
disposed.

`SourceCursor` retains its current definition ID and a persistent private path
link. Each descent allocates one constant-size link; cycle detection scans only
the selected path's finite ancestry, avoiding a copied per-cursor ancestry set.
`cursor.path` materializes a tuple only on demand.
`SourceDocument._resolve_occurrence(path)` replays a finite path through the DAG
for later catalog resolution. Definition IDs never appear in origin keys.

### Enforced value immutability

`SourceCursor` has two deliberately different access paths:

1. Public/cross-boundary access (`value`, `get`, `mapping`, `sequence`) returns a
   fresh iterative safe-value snapshot. Mutable placeholders preserve cycles;
   immutable containers are finalized bottom-up; shared aliases retain identity.
   The copier is stack-safe for deeply nested accepted graphs. Mutating a result
   cannot change the document or another read.
2. Package-private `_value_view()` returns `_ReadOnlyMapping`,
   `_ReadOnlySequence`, a memoized immutable set view, or an immutable SafeLoader
   scalar. The wrappers are memoized by underlying object identity, so alias
   identity and recursive cycles are preserved, but no mutation API exists.
   Compose parse/build code uses this view explicitly and tests use
   `collections.abc.Mapping`/`Sequence`/`Set`, not concrete container assumptions.

Virtual cursors and overlays snapshot incoming values once with the same
iterative copier, then store the same internal read-only representation. Every
later public read is another snapshot. Mutating either the caller's original
object or a returned overlay value cannot change the overlay.

The accepted virtual algebra is exact `dict`, `list`, `tuple`, `set`,
`frozenset`, and SafeLoader scalar values (`None`, `bool`, `int`, `float`, `str`,
`bytes`, `date`, and `datetime`), recursively. Custom objects and container
subclasses raise `TypeError`; a future Python graph front end does not pass
through this YAML-specific cursor layer.

### Complete `SourceCursor` API

`src/agent_composer/compose/source/cursor.py` defines:

```python
@dataclass(frozen=True, slots=True)
class SourceCursor:
    @classmethod
    def virtual(cls, value: Any) -> "SourceCursor": ...

    @property
    def document(self) -> SourceDocument | None: ...

    @property
    def path(self) -> OccurrencePath: ...

    @property
    def value(self) -> Any: ...                 # iterative defensive snapshot

    def _value_view(self) -> Any: ...           # read-only package-private view

    @property
    def range(self) -> SourceRange | None: ...

    @property
    def key_range(self) -> SourceRange | None: ...

    def has(self, key: Any) -> bool: ...
    def get(self, key: Any, default: Any = None) -> Any: ...
    def child(self, key: Any) -> "SourceCursor": ...
    def children(self) -> tuple["SourceCursor", ...]: ...
    def mapping_occurrences(
        self, key: Any = _NO_FILTER
    ) -> tuple["SourceCursor", ...]: ...

    def str(self, key: Any) -> str: ...
    def int(self, key: Any) -> int: ...
    def mapping(self, key: Any) -> dict[Any, Any]: ...
    def sequence(self, key: Any) -> list[Any]: ...

    def range_of(self, key: Any) -> SourceRange | None: ...
    def key_range_of(self, key: Any) -> SourceRange | None: ...
    def item_ranges(self, key: Any) -> dict[str, SourceRange]: ...
    def item_range_list(self, key: Any) -> list[SourceRange | None]: ...
    def mapping_value_ranges(self, key: Any) -> dict[str, SourceRange]: ...
    def sequence_field_ranges(
        self, sequence_key: Any, field: Any
    ) -> list[SourceRange | None]: ...

    def renamed(self, names: Mapping[Any, Any]) -> "SourceCursor": ...
    def with_value(self, value: Any) -> "SourceCursor": ...
    def project_mapping(
        self,
        entries: Mapping[Any, tuple[Any, "SourceCursor | None"]],
    ) -> "SourceCursor": ...
```

Semantics are fixed before implementation:

- `child(key)` accepts a mapping key or a sequence integer index. It chooses the
  last effective mapping pair, matching constructed-dict behavior. Missing keys,
  wrong container shape, wrong typed values, and `bool` passed to `int` raise
  `SourceAccessError(message, range)`.
- `children()` returns effective mapping children in constructed mapping order,
  sequence children in index order, and `()` for a scalar. It never returns
  ignored duplicate pairs.
- `mapping_occurrences()` is mapping-only raw occurrence enumeration. It returns
  flattened pairs in pair order, including ignored duplicates; an optional key
  filters using the same constructed-key equality as the mapping. It returns `()`
  for sequences/scalars. A private sentinel represents no filter, so passing
  `None` filters a real null key.
- `get`, `mapping`, and `sequence` return defensive copies. `str` and `int`
  return immutable scalars. `mapping` never accepts a sequence and `sequence`
  never accepts a mapping.
- `_MapStep(constructed_key, pair_ordinal)` and `_SeqStep(index)` form paths.
  Aliases at two authored locations have distinct paths even when they share a
  YAML node and definition range.
- A persistent-link ancestry guard records a recursive alias occurrence and
  marks it cycle-cut. Its value/ranges/path remain readable and replayable, but
  further child/occurrence traversal raises a located `SourceAccessError`.
- A projected parent retains its own document, path, and range. A projected child
  with a source cursor retains that source cursor's document/path/key/value
  ranges while exposing the replacement value. A child with `source=None` is
  virtual, with `document=None`, empty path, and no ranges.
- `project_mapping` preserves entry insertion order, snapshots every displayed
  value, and exposes one effective occurrence per projected entry.
- `renamed` is a pure projection preserving source cursors; it never changes the
  base mapping or YAML key nodes.

`SourceAccessError` and `SourceSyntaxError` live in
`compose/source/common.py`. The source package imports no `helpers`, nodes,
compile, runtime, run boundary, or CLI module.

### One load-boundary error translator

`compose/parse/file.py` owns exactly one helper:

```python
def parse_source_document(text: str, label: str) -> SourceDocument:
    try:
        return SourceDocument.parse(text, label)
    except SourceSyntaxError as exc:
        raise LoadError(f"invalid YAML: {exc}", range=exc.range) from exc
```

`parse_file(text)`, root `load_flow`, and the external file resolver all call
this helper. `parse_file(document)` reuses the supplied document. Malformed root
and malformed external files therefore expose the same current public
`LoadError`, message prefix, and precise range.

### Temporary migration bridge

Until legacy test helpers and def readers move, add this to
`compose/helpers/reader.py`:

```python
def as_cursor(value: SourceCursor | NodeReader, *, label: str = "<legacy>") -> SourceCursor:
    if isinstance(value, SourceCursor):
        return value
    if value.yaml_node is None:
        return SourceCursor.virtual(value.value)
    return SourceDocument._from_constructed(
        copy.deepcopy(value.value), value.yaml_node, value.spans, label
    ).root
```

The adapter never invokes PyYAML. Parser/desugar/pipeline entry points call it
once, so remaining `NodeReader` callers stay green. New production root loading
passes a `SourceCursor` directly. The adapter and `_from_constructed` disappear
in the legacy-deletion cycle.

## File map

Add:

- `src/agent_composer/compose/source/{__init__,common,document,cursor,literals}.py`
- `tests/compose/test_{source_document,source_cursor,source_ownership}.py`

Delete at final cutover:

- `src/agent_composer/compose/helpers/{reader,spanloader}.py`
- `tests/compose/test_{reader,spanloader}.py`

Principal edits:

- `src/agent_composer/compose/parse/{file,nodes}.py`
- `src/agent_composer/compose/nodes/{base,case}.py`
- `src/agent_composer/compose/desugar/inline_bodies.py`
- `src/agent_composer/compose/build/pipeline.py`
- `src/agent_composer/compose/helpers/{source,types}.py`
- `src/agent_composer/compose/loader.py`
- `src/agent_composer/cli/run.py`
- compose package charters, tests, docs, and engine/structure skills listed below.

## Dependency and green-commit order

| Group | Commit | Depends on | Ordinary `load_flow` after commit |
| --- | --- | --- | --- |
| 0 | Baseline only | issue/branch | unchanged, green |
| 1 | Private `SourceDocument` | 0 | unchanged, green |
| 2 | Immutable `SourceCursor` | 1 | unchanged, green |
| 3 | Vertical local/root cutover + temporary bridge | 2 | normal, inline, compact, defs, external all green |
| 4 | Def/compact cutover | 3 | all local forms green; one parse for multiple defs |
| 5 | External flow-cache ratchets | 4 | all imports green; one read/parse per path |
| 6 | Legacy/test migration and deletion | 5 | all paths green; bridge gone |
| 7 | Docs and skills | 6 | unchanged, green |
| 8 | Full verification/review/PR | 7 | fully verified |

Groups 1 and 2 add unused lower-layer contracts, so they cannot break loading.
Group 3 is one vertical commit: parser/desugar/pipeline changes are not committed
until root, inline, def-through-adapter, and external smoke tests pass. Group 4
then removes the actual def reparse. No commit contains a cursor-only parser fed
by an incompatible loader.

For every microstep below that says “run” a named test or file without repeating
the shell prefix, the exact command is:

```bash
conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q <listed-targets>
```

Use `path/to/test.py::test_name` for a named test. A step never advances after an
unexpected failure.

## Green cycle 0: workflow and baseline

1. **(2 min)** Confirm issue #130 is assigned to `@me`, marked in progress, and
   contains the approved short plan.
2. **(2 min)** Confirm branch `dev/compose/single-parse-source`, based on the
   intended parent branch, and inspect `git status --short` for shared-worktree
   files.
3. **(3 min)** Run the 52-test baseline command in “Verified current evidence.”
4. **(3 min)** Run invariants:

   ```bash
   conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
     tests/engine/test_layer_direction.py tests/engine/test_kind_census.py
   ```

No commit is made.

## Green cycle 1: private `SourceDocument` and syntax boundary

Cycle 1 deliberately does not use `document.root`, source ranges, or cursor
navigation.

1. **(3 min, test first)** Create
   `tests/compose/test_source_document.py::test_document_has_private_retained_slots`.
   Parse `count: 30\nflag: yes\n`; assert `doc._value` is the constructed value,
   `_root_node` exists, `_index` exists, and public attributes `value`,
   `root_node`, and `index` do not.
2. **(2 min)** Run that one test; expect import failure for
   `agent_composer.compose.source`.
3. **(4 min, implementation)** Add the `compose/source/__init__.py` charter,
   `common.py` private path/occurrence/error records, and the slotted
   `SourceDocument` shell in `document.py`.
4. **(3 min, implementation)** Move the SafeLoader construction logic into private
   `_DocumentLoader`; call `get_single_node()` then `construct_document()` exactly
   once and retain `_value`/`_root_node`. Initialize `_index` as an immutable empty
   mapping for this first checkpoint.
5. **(2 min)** Run only `test_document_has_private_retained_slots`; expect green.
6. **(3 min, test first)** Add
   `test_document_uses_one_loader_instance` by spying on
   `_DocumentLoader.__init__`, `get_single_node`, `construct_document`, and
   `dispose`; assert one call to each for one non-empty `SourceDocument.parse`.
7. **(3 min)** Run the loader-instance test; fix only construction/disposal defects.
8. **(3 min, test first)** Add
   `test_document_raises_source_syntax_error_with_problem_range` and
   `test_document_rejects_multiple_documents`. Also pin empty-document behavior
   and ReaderError positions after LF, CR, CRLF, NEL, LS, and PS line breaks.
9. **(4 min, implementation)** Translate PyYAML scanner/parser/composer failures to
   `SourceSyntaxError(message, range)` inside `document.py`, including failures
   raised while the loader is initialized; do not import `LoadError`.
10. **(3 min)** Run all source-document tests.
11. **(2 min)** Run `git diff --check`, inspect status, and stage exactly:

    ```bash
    git add \
      src/agent_composer/compose/source/__init__.py \
      src/agent_composer/compose/source/common.py \
      src/agent_composer/compose/source/document.py \
      tests/compose/test_source_document.py
    git commit -m "feat(compose): add private source documents"
    ```

## Green cycle 2: complete immutable cursor contract

Each substep adds one behavior and runs only its named tests before proceeding.

1. **(3 min, test first)** Create
   `tests/compose/test_source_cursor.py::test_root_value_is_a_defensive_copy`.
   Mutate the returned root mapping and a nested list/dict; assert a second read
   is unchanged.
2. **(4 min, implementation)** Add `cursor.py` with basic `SourceCursor`,
   `SourceDocument.root`, and `value` backed by memoized `copy.deepcopy`.
3. **(2 min)** Run the defensive-copy test.
4. **(4 min, test first)** Add
   `test_value_copy_preserves_alias_identity_and_recursive_alias` and
   `test_value_view_is_recursive_and_read_only`. Add `!!set`, `!!omap`, and
   `!!pairs` cases plus virtual rejection of custom objects/container subclasses.
5. **(5 min, implementation)** Add memoized `_ReadOnlyMapping`,
   `_ReadOnlySequence`, and immutable set handling in `common.py`; implement
   `_value_view()` without mutation methods and with identity memoization. Validate
   virtual/projected snapshots against the exact recursive SafeLoader value
   algebra.
6. **(3 min)** Run the two alias/view tests; explicitly assert
   `copy["root"]["self"] is copy["root"]` and the same relationship for views.
7. **(3 min, test first)** Add
   `test_definition_dag_is_linear_for_binary_alias_graph` using at least 24
   levels, `test_deep_alias_path_is_lazy`, and
   `test_mapping_child_and_children_use_effective_order` with 1-based value/key
   range assertions.
8. **(5 min, implementation)** Override `_DocumentLoader.construct_object()` only
   to retain a temporary YAML-node-to-constructed-value table. After construction,
   intern every unique YAML node once into frozen scalar/mapping/sequence
   definitions. Mapping definitions retain every flattened pair arc and their
   effective first-key-order/last-value ordinals. Assign a definition ID before
   descending, discard both temporary maps, and add persistent cursor path links.
   Do not materialize an occurrence tree or `OccurrencePath -> _Occurrence` map.
9. **(3 min)** Run the DAG/lazy mapping/range tests and assert definition count is
   linear in the unique composed graph.
10. **(3 min, test first)** Add
    `test_sequence_child_and_children_follow_indices`.
11. **(4 min, implementation)** Add `_SeqStep`, sequence definition arcs,
    non-negative non-bool integer `child`, and sequence `children`.
12. **(2 min)** Run the sequence test.
13. **(3 min, test first)** Add
    `test_duplicate_lookup_is_last_but_occurrences_keep_both` and
    `test_merge_local_value_wins_but_inherited_occurrence_remains`. Add
    `test_null_key_filter_is_distinct_from_no_filter`.
14. **(5 min, implementation)** Complete `mapping_occurrences` and last-effective
    lookup using pair ordinals; use a private no-filter sentinel and do not
    collapse raw definition arcs.
15. **(3 min)** Run duplicate/merge tests.
16. **(3 min, test first)** Add
    `test_alias_occurrences_have_distinct_paths_and_anchor_ranges` and
    `test_recursive_alias_path_stops_at_ancestor`. Materialize one cursor path,
    discard the cursor, and assert the document replays it to the same range.
17. **(5 min, implementation)** Add occurrence-local persistent-link descent plus
    a definition-ID ancestry guard. Record the recursive child occurrence before
    marking it cycle-cut; preserve its value/range/path and reject further
    traversal.
18. **(3 min)** Run alias and recursion tests.
19. **(3 min, test first)** Port typed/range cases from `test_reader.py` under these
    exact names:
    `test_typed_accessors`, `test_int_rejects_bool`,
    `test_item_range_list_keeps_duplicate_positions`, and
    `test_sequence_field_ranges_follow_item_order`.
20. **(5 min, implementation)** Implement `has`, `get`, `str`, `int`, `mapping`,
    `sequence`, `range_of`, `key_range_of`, `item_ranges`, `item_range_list`,
    `mapping_value_ranges`, and `sequence_field_ranges` with the contract above.
21. **(3 min)** Run only those four typed/range tests.
22. **(3 min, test first)** Add
    `test_virtual_value_is_snapshotted_and_copy_on_read`,
    `test_renamed_cursor_does_not_mutate_base`, and
    `test_projected_child_retains_source_path_and_range`.
23. **(5 min, implementation)** Implement `virtual`, `with_value`, `renamed`, and
    `project_mapping`; snapshot/freeze all incoming values and preserve source
    cursor identity metadata.
24. **(3 min, test first)** Add
    `test_projected_mutation_cannot_change_overlay`: mutate the caller's original
    object and two separately returned overlay values; assert the overlay and base
    remain unchanged.
25. **(4 min, implementation)** Fix overlay copy-on-read or memoization gaps found
    by that test.
26. **(4 min)** Run the full source foundation:

    ```bash
    conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
      tests/compose/test_source_document.py \
      tests/compose/test_source_cursor.py
    ```

27. **(2 min)** Run `git diff --check`, inspect status, and stage exactly:

    ```bash
    git add \
      src/agent_composer/compose/source/common.py \
      src/agent_composer/compose/source/document.py \
      src/agent_composer/compose/source/cursor.py \
      src/agent_composer/compose/source/__init__.py \
      tests/compose/test_source_document.py \
      tests/compose/test_source_cursor.py
    git commit -m "feat(compose): add immutable source cursors"
    ```

## Green cycle 3: vertical parser, inline, and root-loader cutover

Do not commit after an intermediate substep. The only commit in this cycle is
made after ordinary, inline, compact, def-through-adapter, and external load
smokes all pass.

### Bridge and parser

1. **(3 min, test first)** Add
   `test_as_cursor_preserves_legacy_ranges_without_parsing` to
   `tests/compose/test_source_cursor.py`. Spy on `_DocumentLoader.__init__`, adapt
   one located `NodeReader`, and assert zero new loader calls plus equal ranges.
2. **(5 min, implementation)** Add `_from_constructed` to `SourceDocument` and
   `as_cursor` to `compose/helpers/reader.py`; deep-copy the value, reuse the
   already-composed tree/range lookup only to build the private definition DAG,
   and never expose it. The temporary adapter needs only effective values; it is
   deleted before this issue closes and must not become a second constructor.
3. **(2 min)** Run the adapter test.
4. **(3 min, test first)** In `tests/compose/test_parse.py`, add
   `test_parse_nodes_normalizes_without_mutating_cursor` and
   `test_case_on_normalization_does_not_mutate_cursor`.
5. **(5 min, implementation)** Update `compose/parse/nodes.py` so its entry calls
   `as_cursor`; replace mutating `_back_map_input_output` with a returned renamed
   cursor and translate `SourceAccessError` to current `LoadError` once.
6. **(5 min, implementation)** Update `compose/nodes/base.py` and `case.py` to use
   `_value_view()`, `Mapping`/`Sequence`, returned `_normalize_cursor`, and cursor
   range methods. Do not call public `.value` internally.
7. **(3 min)** Run:

   ```bash
   conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
     tests/compose/test_parse.py tests/compose/test_kind_case_parse.py
   ```

### Inline lowering and pipeline

8. **(3 min, test first)** In `test_inline_composite_bodies.py`, add
   `test_inline_projection_leaves_original_reader_and_document_unchanged` and keep
   existing alias-sibling/merge tests.
9. **(5 min, implementation)** Change `extract_inline_bodies` to call `as_cursor`,
   rename `nodes_reader` to `nodes_cursor`, take one graph-wide defensive value
   snapshot so unchanged aliases retain sharing, and implement pure projected
   masking. Remove `_SpanOverlay`, `_copy_mapping`, wrapper mutation, `copy`, and
   PyYAML usage from `inline_bodies.py`. This is the sole intentional internal
   use of public `cursor.value`.
10. **(3 min)** Run inline tests filtered by
    `"projection or alias or merge or mutat"`.
11. **(4 min, implementation)** Rename `_assemble(nodes_reader=...)` to
    `_assemble(nodes_cursor=...)`, call `as_cursor` at entry, and update
    `InlineBody.nodes_cursor` plus direct assembly call sites.
12. **(3 min)** Run `test_assemble.py` and `test_inline_composite_types.py`.

### File parse and one error boundary

13. **(3 min, test first)** Add to `tests/compose/test_source_ownership.py`:
    `test_parse_file_reuses_supplied_document` and
    `test_parse_source_document_translates_syntax_error`.
14. **(5 min, implementation)** In `parse/file.py`, add
    `parse_source_document`; make `parse_file(str | SourceDocument)` reuse a
    supplied document, derive section lines from the root cursor, and remove
    `yaml.safe_load`, `yaml.compose`, and `_top_level_lines`.
15. **(3 min)** Run the two new tests plus `tests/engine/test_parser.py` and
    `test_compact.py`.
16. **(3 min, test first)** Add
    `test_plain_and_inline_root_use_one_document_loader` and
    `test_root_load_does_not_mutate_captured_document`. Do not include defs in this
    count test.
17. **(5 min, implementation)** Change `load_flow` to call
    `parse_source_document(text, "<flow>")`; make `_load_flow` accept a
    `SourceDocument`; replace `_nodes_reader` with `_nodes_cursor`; use the root
    cursor for private-name lookup and source text.
18. **(4 min, implementation)** Change `_boundary_field_ranges`,
    `_boundary_assert_ranges`, and `_section_lines` in `helpers/source.py` to take
    one cursor. No helper may parse text.
19. **(3 min)** Run the two root ownership tests and `test_load.py`.
20. **(3 min, test first)** Add malformed root and malformed external child cases:
    `test_load_flow_malformed_root_is_located_load_error` in
    `test_source_ownership.py` and
    `test_malformed_external_child_is_located_load_error` in
    `test_uses_files.py`.
21. **(4 min, implementation)** Make the existing file resolver call
    `parse_source_document(cand.read_text(), cand.name)` before `_load_flow`.
    Both malformed paths must now share the same translation helper.
22. **(3 min)** Run those two malformed tests.
23. **(3 min, test first)** Add
    `test_input_default_uses_literal_parser_not_document_loader`.
24. **(4 min, implementation)** Add `compose/source/literals.py`, route
    `helpers/types.py` default parsing through `parse_yaml_literal`, and remove its
    PyYAML import.
25. **(3 min)** Run the literal test plus `test_flow_inputs.py` and
    `test_typedefs.py`.
26. **(3 min, implementation)** Change `top_frame_ranges` to accept an existing
    document/cursor. In `cli/run.py::_top_ranges`, use `loaded.source` normally;
    only the defensive `loaded is None` branch may construct one document through
    `parse_source_document`. Update
    `test_inline_sibling_preserves_outer_and_top_frame_ranges` to construct one
    document and reuse it for both helper calls.
27. **(3 min)** Run `test_call_source_frame.py` and
    `test_inline_sibling_preserves_outer_and_top_frame_ranges`.

### Vertical green gate and commit

28. **(5 min)** Run the required vertical gate:

    ```bash
    conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
      tests/compose/test_parse.py \
      tests/compose/test_source_ownership.py \
      tests/engine/test_load.py \
      tests/engine/test_compact.py \
      tests/engine/test_inline_composite_bodies.py \
      tests/engine/test_defs.py \
      tests/engine/test_uses_files.py \
      tests/engine/test_errors.py \
      tests/engine/test_call_source_frame.py
    ```

29. **(2 min)** Run `git diff --check` and verify field navigation uses cursors or
    `_value_view()`. The graph-wide snapshot in `inline_bodies.py` is the only
    intentional internal `cursor.value` use.
30. **(3 min)** Stage exactly these files and commit:

    ```bash
    git add \
      src/agent_composer/compose/source/__init__.py \
      src/agent_composer/compose/source/document.py \
      src/agent_composer/compose/source/literals.py \
      src/agent_composer/compose/helpers/reader.py \
      src/agent_composer/compose/helpers/source.py \
      src/agent_composer/compose/helpers/types.py \
      src/agent_composer/compose/parse/file.py \
      src/agent_composer/compose/parse/nodes.py \
      src/agent_composer/compose/nodes/base.py \
      src/agent_composer/compose/nodes/case.py \
      src/agent_composer/compose/desugar/inline_bodies.py \
      src/agent_composer/compose/build/pipeline.py \
      src/agent_composer/compose/loader.py \
      src/agent_composer/cli/run.py \
      tests/compose/test_parse.py \
      tests/compose/test_source_cursor.py \
      tests/compose/test_source_ownership.py \
      tests/engine/test_assemble.py \
      tests/engine/test_inline_composite_bodies.py \
      tests/engine/test_uses_files.py
    git commit -m "refactor(compose): load root flows through source cursors"
    ```

## Green cycle 4: defs, compact overlays, and the final local reparse removal

1. **(3 min, test first)** Add
   `test_multiple_defs_share_one_parent_document_parse` to `test_defs.py`; use two
   eager multi-node defs and spy on both `_DocumentLoader.__init__` and the
   temporary legacy `SpanLoader.__init__`. Assert the combined full-document
   loader count is one for the physical parent. Counting `_DocumentLoader` alone
   would miss the exact legacy def reparse this test is meant to expose.
2. **(2 min)** Run it; expect more than one construction because
   `_def_nodes_reader` still reparses.
3. **(3 min, test first)** Add
   `test_multi_node_def_cursor_keeps_absolute_parent_ranges` using existing
   absolute line assertions.
4. **(5 min, implementation)** Replace `_CallableScope(parent_text=...)` with
   `_CallableScope(source: SourceDocument | None)`; derive
   `root/defs/<name>/nodes` directly and pass that cursor to `_assemble`.
5. **(4 min, implementation)** Delete `_def_nodes_reader` and
   `_def_assembly_reader`; keep `_restore_node_surface` only for source-free scope
   unit tests.
6. **(3 min)** Run the multiple-def and absolute-range tests.
7. **(3 min, test first)** Add
   `test_compact_virtual_projection_does_not_mutate_root` and assert the synthetic
   node retains authored top-level field ranges where a source exists.
8. **(5 min, implementation)** Build the compact node cursor as a projection from
   the root cursor. Add auto-wiring/output values as snapshotted virtual entries;
   source authored logic fields at their original children.
9. **(3 min)** Run the compact projection test and all `test_compact.py`.
10. **(3 min, test first)** Add
    `test_single_node_def_virtual_header_and_logic_ranges`.
11. **(4 min, implementation)** Project the synthetic single node's header from
    the `defs:<name>` key and its untouched logic fields from the flat def cursor.
12. **(3 min)** Run that one test.
13. **(3 min, test first)** Add
    `test_single_node_def_generated_inputs_keep_declaration_ranges`.
14. **(4 min, implementation)** Project generated auto-wiring entries from each
    authored input-declaration child, without mutating the normalized body.
15. **(3 min)** Run the generated-input test.
16. **(3 min, test first)** Add
    `test_single_node_def_generated_output_keeps_authored_output_range`.
17. **(4 min, implementation)** Source the synthetic node output type and flow
    output binding from the authored flat `output:` cursor.
18. **(3 min)** Run the output-range test.
19. **(5 min)** Run the local-form gate:

    ```bash
    conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
      tests/engine/test_defs.py \
      tests/engine/test_compact.py \
      tests/engine/test_inline_composite_bodies.py \
      tests/engine/test_child_asserts.py \
      tests/engine/test_call_source_frame.py
    ```

20. **(2 min)** Run `git diff --check`, inspect status, and stage exactly:

    ```bash
    git add \
      src/agent_composer/compose/parse/file.py \
      src/agent_composer/compose/loader.py \
      tests/engine/test_defs.py \
      tests/engine/test_compact.py \
      tests/engine/test_inline_composite_bodies.py
    git commit -m "refactor(compose): reuse source cursors for defs"
    ```

## Green cycle 5: ratchet the existing external compiled-flow cache

1. **(3 min, test first)** Add `test_load_context_has_one_typed_flow_cache` to
   `test_uses_files.py`; require typed `loading` and `flows` fields and reject
   ambiguous `cache`/unused `documents` state.
2. **(3 min, implementation)** Rename `_LoadCtx.cache` to
   `flows: dict[Path, ComposedFlow]` and type the loading set.
3. **(3 min, test first)** Add
   `test_two_aliases_of_one_file_read_and_parse_once`, spying on `Path.read_text`
   and `parse_source_document`.
4. **(3 min, test first)** Extend diamond reuse with exact per-path read and parse
   counts for `mid1`, `mid2`, and `leaf`.
5. **(3 min)** Run both count ratchets plus version-pin and cycle cases. Keep the
   per-alias version check on flow-cache hits and cache only successful flows.
6. **(4 min)** Run the external gate:

    ```bash
    conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
      tests/engine/test_uses.py \
      tests/engine/test_uses_files.py \
      tests/engine/test_call_source_frame.py \
      tests/engine/test_templates_load.py
    ```

7. **(2 min)** Run `git diff --check`, inspect status, and stage exactly:

    ```bash
    git add \
      src/agent_composer/compose/loader.py \
      tests/engine/test_uses_files.py
    git commit -m "test(compose): ratchet external source reuse"
    ```

## Green cycle 6: migrate tests, delete the bridge, and ratchet ownership

### Ownership tests first

1. **(3 min, test first)** Create/extend
   `tests/compose/test_source_ownership.py` with:
   `test_only_source_package_imports_pyyaml`,
   `test_legacy_source_modules_are_absent`, and
   `test_source_package_has_no_upward_imports`. Add
   `test_document_graph_slots_are_private_to_source_package`, which scans
   production AST/text and rejects `_value`, `_root_node`, `_definitions`, or
   `_root_definition` access outside `compose/source/`; `_index` must be absent
   after the cycle-2 placeholder is removed.
2. **(2 min)** Run those tests; expect the first two to fail while legacy modules
   remain.

### Migrate compose tests in small batches

3. **(4 min)** Replace located helpers with `SourceDocument.parse(...).root` and
   spanless helpers with `SourceCursor.virtual(...)` in
   `test_descriptor_base.py` and `test_parse.py`.
4. **(3 min)** Run those two files.
5. **(4 min)** Migrate `test_desugar.py` and `test_edges.py`.
6. **(3 min)** Run those two files.
7. **(4 min)** Migrate `test_kind_agent_parse.py` and
   `test_kind_call_parse.py`.
8. **(3 min)** Run those two files.
9. **(4 min)** Migrate `test_kind_case_parse.py` and
   `test_kind_code_parse.py`.
10. **(3 min)** Run those two files.
11. **(4 min)** Migrate `test_kind_human_input_parse.py` and
    `test_kind_loop_parse.py`.
12. **(3 min)** Run those two files.
13. **(4 min)** Migrate `test_kind_map_parse.py` and
    `test_kind_model_parse.py`.
14. **(3 min)** Run those two files.
15. **(4 min)** Migrate `test_kind_tool_parse.py` and
    `test_kind_wait_parse.py`.
16. **(3 min)** Run those two files.
17. **(4 min)** Migrate `test_validate.py`, then run it.

### Migrate engine tests in small batches

18. **(4 min)** Migrate `test_build_nodes.py` and `test_case.py`.
19. **(3 min)** Run those two files.
20. **(4 min)** Migrate `test_graph.py` and `test_infer.py`.
21. **(3 min)** Run those two files.
22. **(4 min)** Migrate `test_loop_parser.py` and `test_node_parse.py`.
23. **(3 min)** Run those two files.
24. **(4 min)** Migrate `test_parser_effects.py` and
    `test_parser_human_questions.py`.
25. **(3 min)** Run those two files.
26. **(4 min)** Migrate `test_validate_graph.py` and `test_validate_refs.py`.
27. **(3 min)** Run those two files.

### Delete compatibility code

28. **(3 min)** Run:

    ```bash
    rg -n "NodeReader|SpanLoader|load_with_spans|helpers\.(reader|spanloader)|as_cursor" \
      src tests
    ```

    Every remaining hit must be one of the legacy modules/tests about to be
    deleted or a migration bug. Fix migration bugs before proceeding.
29. **(4 min, implementation)** Remove `as_cursor` calls from production entry
    points now that all callers pass `SourceCursor`; remove
    `SourceDocument._from_constructed`.
30. **(3 min, implementation)** Delete with `apply_patch`:
    `compose/helpers/reader.py`, `compose/helpers/spanloader.py`,
    `tests/compose/test_reader.py`, and `tests/compose/test_spanloader.py`.
31. **(4 min, implementation)** Update
    `compose/{__init__,helpers/__init__,parse/__init__}.py` and
    `compose/source/__init__.py` charters/re-exports. Only source modules may import
    PyYAML.
32. **(4 min)** Run:

    ```bash
    conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q tests/compose
    conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
      tests/engine/test_layer_direction.py \
      tests/engine/test_kind_census.py \
      tests/engine/test_flow_identity.py
    ```

33. **(2 min)** Re-run the `rg` command from step 28; expect no hits.
34. **(3 min)** Run `git diff --check`, inspect status, and stage only this exact
    cleanup set:

    ```bash
    git add \
      src/agent_composer/compose/__init__.py \
      src/agent_composer/compose/helpers/__init__.py \
      src/agent_composer/compose/helpers/reader.py \
      src/agent_composer/compose/helpers/spanloader.py \
      src/agent_composer/compose/parse/__init__.py \
      src/agent_composer/compose/parse/nodes.py \
      src/agent_composer/compose/desugar/inline_bodies.py \
      src/agent_composer/compose/build/pipeline.py \
      src/agent_composer/compose/source/__init__.py \
      src/agent_composer/compose/source/document.py \
      tests/compose/test_descriptor_base.py \
      tests/compose/test_desugar.py \
      tests/compose/test_edges.py \
      tests/compose/test_kind_agent_parse.py \
      tests/compose/test_kind_call_parse.py \
      tests/compose/test_kind_case_parse.py \
      tests/compose/test_kind_code_parse.py \
      tests/compose/test_kind_human_input_parse.py \
      tests/compose/test_kind_loop_parse.py \
      tests/compose/test_kind_map_parse.py \
      tests/compose/test_kind_model_parse.py \
      tests/compose/test_kind_tool_parse.py \
      tests/compose/test_kind_wait_parse.py \
      tests/compose/test_parse.py \
      tests/compose/test_reader.py \
      tests/compose/test_source_ownership.py \
      tests/compose/test_spanloader.py \
      tests/compose/test_validate.py \
      tests/engine/test_build_nodes.py \
      tests/engine/test_case.py \
      tests/engine/test_graph.py \
      tests/engine/test_infer.py \
      tests/engine/test_loop_parser.py \
      tests/engine/test_node_parse.py \
      tests/engine/test_parser_effects.py \
      tests/engine/test_parser_human_questions.py \
      tests/engine/test_validate_graph.py \
      tests/engine/test_validate_refs.py
    git commit -m "refactor(compose): remove legacy span readers"
    ```

Deleted paths are intentionally named in `git add`; Git stages their deletion.

## Green cycle 7: docs, charters, and skills

1. **(4 min)** Update `docs/compose.md` to show
   `text -> SourceDocument -> SourceCursor -> descriptors`, one document per
   physical file, occurrence identity versus definition range, defensive-copy
   public reads, and immutable virtual projections.
2. **(4 min)** Update `docs/architecture.md` with the resolved-path compiled-flow
   cache and one-parse load phase. Do not add future origin/failure classes or
   Mermaid configuration in this slice.
3. **(3 min)** Update `src/agent_composer/README.md` layout and inline/def source
   ownership.
4. **(4 min)** Update `.claude/skills/engine/SKILL.md` and `reference.md` to replace
   the `NodeReader`/span-table account with `SourceDocument`/`SourceCursor`, while
   marking `SourceFrame` as the temporary diagnostic bridge.
5. **(4 min)** Update `.claude/skills/structure/SKILL.md` and `reference.md`: source
   owns YAML; parse/desugar/build consume cursors; loader owns the flow cache;
   compile/runtime stay YAML-blind.
6. **(4 min)** Run:

   ```bash
   conda run -n agentcomposer mkdocs build --strict
   git diff --check
   ```

7. **(2 min)** Inspect status and stage exactly:

   ```bash
   git add \
     src/agent_composer/README.md \
     docs/compose.md \
     docs/architecture.md \
     .claude/skills/engine/SKILL.md \
     .claude/skills/engine/reference.md \
     .claude/skills/structure/SKILL.md \
     .claude/skills/structure/reference.md
   git commit -m "docs(compose): document single-parse source ownership"
   ```

`docs/syntax.md` and composing-agents do not change because authoring behavior is
unchanged.

## Green cycle 8: end-to-end verification, review, PR, and merge

1. **(5 min)** Run compose and engine suites:

   ```bash
   conda run -n agentcomposer env PYTHONPATH=src python -m pytest tests/compose
   conda run -n agentcomposer env PYTHONPATH=src python -m pytest tests/engine
   ```

2. **(5 min)** Run the full suite and docs:

   ```bash
   conda run -n agentcomposer env PYTHONPATH=src python -m pytest
   conda run -n agentcomposer mkdocs build --strict
   git diff --check
   ```

3. **(4 min)** Re-run focused acceptance tests:

   ```bash
   conda run -n agentcomposer env PYTHONPATH=src python -m pytest -q \
     tests/compose/test_source_document.py \
     tests/compose/test_source_cursor.py \
     tests/compose/test_source_ownership.py \
     tests/engine/test_defs.py \
     tests/engine/test_compact.py \
     tests/engine/test_inline_composite_bodies.py \
     tests/engine/test_uses_files.py \
     tests/engine/test_errors.py \
     tests/engine/test_call_source_frame.py
   ```

4. **(5 min)** Adversarially verify:

   - no full constructed graph escapes `SourceDocument`;
   - root, nested, alias, recursive-alias, and projected values resist mutation;
   - only inline-body extraction uses public `cursor.value` internally, once per
     nodes graph to preserve alias sharing during projection;
   - plain/inline roots parse once;
   - multiple defs do not reparse their parent;
   - aliases/diamonds read and parse each external path once;
   - malformed root and external YAML both become located `LoadError` through the
     same helper;
   - duplicate/merge/alias occurrence semantics match the contract;
   - compact and single-node-def projections retain sources and do not mutate;
   - `parse_file(text)` remains available;
   - range/error fixtures and fingerprints remain unchanged;
   - runtime kind census remains zero and no import back-edge exists.

5. **(3 min)** Fix every finding test-first. Add a focused green commit rather
   than hiding a fix in the docs commit.
6. **(3 min)** Run separate spec-compliance and code-quality reviews. The quality
   review must focus on alias recursion, iterative snapshot memoization, read-only
   wrappers, overlay snapshotting, and cache/cycle identity.
7. **(3 min)** Open the PR with `Fixes #130`, link its plan comment and parent
   #110, and list exact checks/results.
8. **(2 min)** Merge only after CI/review is green and verify the merge commit is
   on `main`. Do not call the larger origin/failure objective complete; slice 2
   starts from this merged source contract.

## Definition of done

- A plain/inline root and a parent with multiple defs use one `_DocumentLoader`.
- Every external physical file is read and parsed once per load graph, including
  repeated aliases and diamond imports.
- `SourceDocument` exposes no constructed value/tree/definition DAG.
- No production module outside `compose/source/` reads its private graph slots.
- Every public cursor value read is a defensive recursive-alias-preserving copy;
  the internal view is recursively read-only; overlay inputs are snapshotted.
- Duplicate, merge, alias, and recursive-alias occurrence tests pass.
- Root, defs, compact, single-node defs, external uses, and inline children all
  consume cursors and preserve current behavior/ranges.
- Root and external malformed YAML use one `LoadError` translation helper.
- `NodeReader`, `SpanLoader`, `load_with_spans`, and the temporary adapter are
  gone.
- Only `compose/source/` imports PyYAML.
- Docs, charters, engine skill, and structure skill match the code.
- Focused suites, full suite, strict docs, reviews, PR checks, merge, and `main`
  verification are complete.
