"""Shared, kind-agnostic helpers for the Compose front end.

The shared lower layer above `compose.source`: primitives later Compose phases build on,
none of which knows about a specific node kind or a specific graph-stage pass. Collected
here so the layering is legible: `helpers` imports the engine core (`typesys` / `expr` /
`nodes` / `compile`) and the lower `compose.source` boundary; later Compose phases import
down into it.

Modules:
  - `directives`  `CALL_HEAD` and `match_call_directive`: recognizes a whole-value
                  `call(...)` directive in binding text, rejecting malformed text that
                  starts with `call(`.
  - `errors`      the document-owned diagnostic factory plus `LoadError`, whose one
                  canonical `Diagnostic` is ready for direct boundary rendering.
  - `isinstance_types` the type names `isinstance(x, T)` calls name in a binding or
                  condition, their resolution against a flow's `typedefs:`, and which
                  union members an `isinstance` check narrows to.
  - `memory`      strict named-policy parsing plus opaque occurrence attachment.
  - `output_contracts` the template-aware output-binding contract resolver: exact or
                  partial direct-reference projection, source-free proven expression
                  results, closed-record assembly, and closed-path validation.
  - `source`      parse-free boundary-input range projection over an existing
                  `SourceCursor`; compatibility metadata, never a YAML parser. Assertion
                  origin/range pairs come from one retained item cursor in the build
                  conductor, not a helper-side map. `is_extension_key` is the one
                  rule for the top-level `x-*` keys a load ignores.
  - `tags`        a slot's type and value through its YAML tags: `read_tag_type` (tag
                  text -> `Type`, with the strict YAML 1.2 character rule and a
                  close-match hint), `read_tagged_value` (a slot cursor -> `TaggedSlot`:
                  a binding source or literal, with nested tags as typed `Binding`s in a
                  `ValueTemplate`, consumed tags recorded in a ledger), `suggest_tag`
                  (the one place that spells a tagged slot for messages),
                  `reject_unread_tags` / `reject_tagged_output` (a tag outside a typed
                  slot is a located load error),
                  `old_typed_text` / `resolve_type_text` / `spell_type_text` /
                  `old_form_suggestion` (recognize the old `Type = value` and bare
                  type text, now load errors, and spell the tag to write),
                  `QUOTED_STYLES`, and `yaml_quoted`.
                  A tag only checks; it never fills in a field.
  - `types`       one recursive type-shape reader: `read_type`, `read_flow_inputs`,
                  and `read_variables` are source-free (type text only, no defaults);
                  `read_value_contract` and `read_source_parameters` compile retained
                  cursors into `Parameter` / `ValueContract` carriers containing only
                  opaque origins; a declaration (`qty: !int`, `n: !int(1..12) 4`) is
                  read through `tags.read_tagged_value`, and an untagged one is an
                  error that shows the tag. Local variables need a default.
  - `uses`        `UsesRef` / `parse_uses_ref`: the `uses:` child-flow reference parser.
  - `value_nesting` `check_written_value`: rejects a value written out in a flow file
                  that nests deeper than a yian value may, as a located load error,
                  counted as the run-time check counts it and before any recursive
                  walker reads it. `check_env_values`: the same check for each value
                  of an `env:` mapping, located at its key, which also rejects a value
                  that holds itself through a recursive alias.
  - `yaml_words`  `former_boolean_fix`: the boolean a YAML 1.1 boolean spelling (`yes`,
                  `off`) meant, and the fix a load error shows when one reaches a `bool`
                  that the boolean fits, at any depth of a list or record. `yaml_read_as`:
                  what YAML read a non-text value as, for the non-text key and text-field
                  errors.

The sibling `compose.source` package exclusively owns YAML parsing and its private
source graph. Layer: `source <- helpers <- nodes <- parse <- desugar <- edges <-
validate <- build <- loader`. The package `__init__` only re-exports the public surface.
"""

from __future__ import annotations
