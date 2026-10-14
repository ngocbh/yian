"""Shared, kind-agnostic helpers for the Compose front end.

The shared lower layer above `compose.source`: primitives later Compose phases build on,
none of which knows about a specific node kind or a specific graph-stage pass. Collected
here so the layering is legible: `helpers` imports the engine core (`typesys` / `expr` /
`nodes` / `compile`) and the lower `compose.source` boundary; later Compose phases import
down into it.

Modules:
  - `errors`      the document-owned diagnostic factory plus `LoadError`, whose one
                  canonical `Diagnostic` is ready for direct boundary rendering.
  - `isinstance_types` the type names `isinstance(x, T)` calls name in a binding or
                  condition, and their resolution against a flow's `typedefs:`.
  - `memory`      strict named-policy parsing plus opaque occurrence attachment.
  - `output_contracts` the template-aware output-binding contract resolver: exact or
                  partial direct-reference projection, source-free proven expression
                  results, closed-record assembly, and closed-path validation.
  - `secret_literals` `secret_literal_path`: the first literal value under a secret
                  name, walking nested containers; plus the shared message wording
                  and `check_env_secrets` for an `env:` mapping.
  - `source`      parse-free boundary-input range projection over an existing
                  `SourceCursor`; compatibility metadata, never a YAML parser. Assertion
                  origin/range pairs come from one retained item cursor in the build
                  conductor, not a helper-side map.
  - `typed_bindings` `split_typed_binding`: recognizes plain-style `Type = value`
                  binding text, resolving the type and reading the value (a literal
                  value is checked against the type); quoted and block scalars stay
                  plain text. `typed_binding_parts` and `read_typed_value` are its
                  purely syntactic steps.
  - `types`       one recursive type-shape reader: `read_type`, `read_flow_inputs`,
                  and `read_variables` are source-free; `read_value_contract` and
                  `read_source_parameters` compile retained input/variable cursors into
                  `Parameter` / `ValueContract` carriers containing only opaque origins.
                  Local variables reuse the default reader with a mandatory-default policy.
  - `uses`        `UsesRef` / `parse_uses_ref`: the `uses:` child-flow reference parser.
  - `yaml_words`  `former_boolean_fix`: the boolean a YAML 1.1 boolean spelling (`yes`,
                  `off`) meant, and the fix a load error shows when one reaches a `bool`
                  that the boolean fits, at any depth of a list or record. `yaml_read_as`:
                  what YAML read a non-text value as, for the non-text key, prompt, and
                  question-text errors.

The sibling `compose.source` package exclusively owns YAML parsing and its private
source graph. Layer: `source <- helpers <- nodes <- parse <- desugar <- edges <-
validate <- build <- loader`. The package `__init__` only re-exports the public surface.
"""

from __future__ import annotations
