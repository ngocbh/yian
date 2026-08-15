VERDICT: NEEDS_REVISION

## Blocking findings

### 1. Flow-input declarations still have two authorities

The revision defines `Binding`, `Assertion`, and `ValueContract`, but it does not
define the final formal-parameter contract. It says both typed `ParamDecl`s and
“flow input declarations” carry `ValueContract` (`design-v2.md:370-372`) while
the ownership table names neither declaration type. Today those are overlapping
semantic objects: compose's `InputDecl` owns name/type/default/required
(`compose/helpers/types.py:54-71`), nodes' `ParamDecl` owns the same facts
(`nodes/binding.py:46-71`), and `StartNode` retains the former while deriving the
latter (`nodes/start/node.py:57-109`). Adding `ValueContract` to both preserves the
overlap the design is meant to remove.

Define one final source-neutral formal-parameter value owned below `compose/`, and
make any YAML parse descriptor lower into it without escaping the front end. State
where the default, required/declaration origin, coercion policy, and nested
`ValueContract` live, and which origin is selected for an omitted required input,
a bad default, and a boundary type failure. This is necessary for the claimed
single authority and for Python-authored graphs to use START without compose types.

### 2. Opaque custom origins cannot yet be rebased or resolved by the stated contracts

`OriginRef.key` is correctly declared opaque outside its producer, but the custom
`child_resolver` contract then asks compose to apply its structural alias rebasing
to arbitrary keys (`design-v2.md:293-299`) without defining whether the old opaque
key is parsed, replaced, or encoded as one atomic segment. Only the last approach
is generically collision-safe. Specify the exact pure transform, including native
compose keys and arbitrary custom keys.

There is also no contract matching the goal that another front end may provide its
own resolver (`design-v2.md:56-57`): `SourceCatalog` resolves only YAML-backed
`SourceSite`s (`design-v2.md:263-268`) and `ComposedFlow.diagnose` is specified only
in terms of that catalog. Either add a source-neutral diagnostic-resolver protocol
that a catalog implements, or narrow the claim and require non-compose graphs to be
source-free in this change.

### 3. Mermaid is selected, but the reproducible runtime is not yet selected

The diagram types, location, and validation are now concrete. The prior review also
required an exact pinned/vendored runtime; the revision only says to “load one exact”
version (`design-v2.md:546-552`). That leaves the material choice unresolved: a
vendored browser asset, or an exact CDN URL (and integrity policy). This matters
because the current Material integration otherwise fetches floating `@11`, while
`mkdocs.yml:59-70` has no custom fence or pinned script and CI has no Node/docs step
(`.github/workflows/publish.yml:15-42`). Name the exact version and delivery mechanism
so the implementation plan has one reproducible contract.

## Resolved from review 1

The revision otherwise resolves the previous critical items: occurrence-based
immutable parsing, canonical semantic carriers, deterministic derived terminal
failure, `LoadError(Diagnostic)`, a closed failure vocabulary with host boundaries,
the inline-CODE origin protocol, and the required package/class plus
YAML-to-success/failure diagram set. Runtime remains kind-blind and the record-first
transition rule is preserved.
