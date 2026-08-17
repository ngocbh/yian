# Seed gallery — NEGATIVE examples (expected failures)

These flows are **supposed to fail** — most at load/compile, some loudly at runtime. They pin
the Compose loader and runtime contract that failures are loud and source-located, and double as
the loader's negative test fixtures
(`tests/engine/test_errors.py`). Each file's header states the expected diagnostic **and
whether it fails at compile or runtime**, and isolates exactly **one** failure (everything
else in the file is valid).

## The rule — every behavior gets negative fixtures

**For each seed or engine behavior in the gallery, add error fixtures here that capture its
expected failure modes — both compile-time and runtime — that the engine can actually
detect.** Concretely:

- **Compile-time** failures load via `load_flow` and must raise `LoadError`; when a source position
  is available, assert the diagnostic frame's `SourceRange` against the offending
  `.yaml` site, not the IR.
- **Runtime** failures run via `run_flow` and must come back as
  `RunResult.status == "failed"` with the expected substring in
  `RunResult.failure.message` — never an uncaught crash.
- Each fixture isolates **one** failure, states compile-vs-runtime + the expected diagnostic in
  its header, and gets a matching assertion in `tests/engine/test_errors.py`.
- A failure mode whose check isn't built yet goes under `_future/errors/` until that check exists —
  never assert an error the engine can't yet capture.
- **A failure the engine DOES catch but LOCATES WRONG is a third case** — it belongs here, marked ⚠
  in the tables below, and its test asserts the **current, incorrect** caret. It is a *reproduction*,
  not a specification: the header states the expected caret beside the current one, and **the test
  should FAIL when the behavior is corrected** — that failure is the signal to update it to the
  expected line. Rationale: a diagnostics bug argued in prose is a bug nobody can confirm; the point
  of the fixture is that a reader can run it. (This is distinct from `_future/`, which is for checks
  that don't exist at all — here the check fires, it just points at the wrong line.)

A negative gallery is half the value of a type system: it pins what *can't* be written, not
just what can.

## Compile failures (fail at load, regardless of inputs)

| File | Should fail with | Pins |
|---|---|---|
| `e01-undeclared-ref.yaml` | unknown reference `${<typo>.output}` — no such node | reference validation and source location |
| `e02-cycle.yaml` | the inferred graph has a cycle (not a DAG) | data-edge cycle detection |
| `e03-unknown-field.yaml` | dotted access into a record field that doesn't exist | record-field validation |
| `e06-type-mismatch-ref.yaml` | a `float` bound to a REF child input declared `str` | cross-flow signature compatibility |
| `e09-unknown-top-level-key.yaml` | an unknown top-level section key (`extra='forbid'`) | strict top-level schema |
| `e10-unknown-node-kind.yaml` | a node `kind:` outside the closed set | closed `NodeKind` |
| `e11-field-not-allowed.yaml` | a flat field illegal for the kind (`prompt` on a `code`) | per-kind body shape |
| `e12-missing-required-field.yaml` | a kind-required field omitted (`code` w/o `code:`) | per-kind body shape |
| `e13-bad-type-expression.yaml` | a malformed type expression (`outputs: list[`) | the `ast` type parser |
| `e14-unknown-system-ambient.yaml` | `${system.bogus}` — only `today`/`now`/`run_id` are valid | strict `system` ambients |
| `e15-prompt-undeclared-input.yaml` | an AGENT prompt references a non-declared input | prompt input validation |
| `e16-bad-typedef-name.yaml` | a `typedefs:` name shadows a typing constructor | `read_typedefs` rules |
| `e17-case-nonexhaustive.yaml` | a `case … on` enum leaves a tag uncovered, no `else:` | exhaustiveness check |
| `e33-input-named-like-field.yaml` | an input named exactly like a flat field (`env`) | `(kind,key)` keeps the field's range distinct from the like-named input's |
| `e37-synth-node-inherits-host-ranges.yaml` | a bad ref in an inline `call(...)` | directive-owned provenance locates the synthesized binding |
| `e38-case-when-ref-locates-node-header.yaml` | a bad ref inside a `case`'s `when:` — desugared to a synthetic `__r1` input | predicate-owned provenance locates and names the authored case |
| `e39-llm-config-no-range.yaml` | a bad `llm_config:` entry (`provider: bogus_provider`) | retained `llm_config` value range, beginning at the invalid `provider:` entry |
| `e40-agent-controls-no-range.yaml` | retired AGENT `controls:` field | field value (L14) |
| `e30-inline-syntax-error.yaml` | a malformed inline `code:` body (`return (`) | inline syntax gate → exact literal-line `LoadError` via the CODE origin selector |
| `e31-inline-no-return.yaml` | an inline `code:` body with no `return` | has-`return` load gate → located `LoadError` |
| `e32-malformed-code-ref.yaml` | a `code:` value that is neither a valid ref nor inline source (dotted, no colon) | reject bucket → `LoadError` with a "did you mean" hint |
| `e35-bad-map-over-ref.yaml` | a dangling `${…}` in a MAP's reserved `over:` key | the caret boxes the `over:` line, not the `input:` block — `("field", "over")` is tuple-keyed (#101) |
| `e36-bad-map-input-ref.yaml` | a dangling `${…}` in ONE of a MAP's per-element `input:` entries | the caret boxes that entry, not the whole block — the MAP per-element scan path (#87) |

## Runtime failures (fail during the run; the header states the triggering inputs)

| File | Should fail with | Pins |
|---|---|---|
| `e18-false-boundary-assert.yaml` | a false `${inputs}`-only assert (run `window: -5`) | flow-boundary assertion evaluation |
| `e19-false-post-assert.yaml` | a false `${X.output}` assert flips success → failed | post-terminal asserts |
| `e34-dup-assert.yaml` | two IDENTICAL post-asserts on one node, both false (run with no inputs) | asserts located BY POSITION — the caret points at the FIRST occurrence, never collapsing to the last (#88) |
| `e20-code-raises.yaml` | a CODE callable that raises → the run fails | node-failure → `RunFailed` |
| `e21-code-wrong-type.yaml` | a CODE node returns the wrong type for its `outputs:` | typed write boundary |
| `e22-required-missing.yaml` | `${x:?msg}` on an omitted value (run with it omitted) | `:?` use-site required |
| `e23-unknown-tool.yaml` | a TOOL node naming an unregistered `tool_id` | TOOL registry lookup |
| `e08-input-type-mismatch.yaml` | a non-coercible input value for a declared type (run `window: "soon"`) | typed read boundary (mirror of e21) |
| `e24-nested-code-raise.yaml` | a CODE callable that raises INSIDE a called child (namespaced node id `run/boom`) | canonical outer CALL + inner CODE origins and `--engine-trace` |
| `e25-external-raise.yaml` (+ `lib_boom.yaml`) | a CODE callable that raises inside an EXTERNAL `uses:` flow (`go/kaboom`) | ordered origin frames across a file boundary |
| `e26-three-level-raise.yaml` | a CODE callable that raises three levels deep (call → def → def, `outer/via/boom`) | ordered canonical diagnostics with at least three frames |
| `e27-inline-code-raises.yaml` | an inline `code:` body that raises (run `topic: X`) | inline analogue of e20 (in-process raise → `RunFailed`) |
| `e28-inline-wrong-type.yaml` | an inline `code:` body returns the wrong type for its `output:` | typed write boundary (reused, not reinvented) |
| `e29-inline-nonserializable.yaml` | an inline `code:` body returns a nested non-serializable value | node's deep serialize-once check → fails AT the node |
| `e41-structured-output-no-range.yaml` | a structured `output:` whose ONE bad field fails validation | canonical contract-path origin resolves the exact bad field |

## Unsupported checks (`_future/errors/`)

These **load and run clean today** — the engine has no check to fire, so they assert nothing. They
exist because #88's container table reads as if `env:`/`tools:` are one absent line away from a
precise caret; they are not. Both are the clean proof that the **stamp was never the bottleneck**: a
per-entry range for `("env","API_KEY")` would be correct, free under a path model, and asked for by
nothing, ever.

| File | Does today | Blocked on |
|---|---|---|
| `f01-env-ref-not-interpolated.yaml` | `env: {API_KEY: ${nope.output}}` loads AND runs — the ref is passed through as a literal string; `env: {API_KEY: 123}` also passes (only env KEYS are checked) | **no issue filed.** The silent `${...}` pass-through looks like a bug in its own right; out of scope for #88 |
| `f02-agent-tools-no-load-check.yaml` | an unknown id in an AGENT `tools:` list survives compile; surfaces as an unclassified host error with no canonical diagnostic origin | **no issue filed.** Not a dup of `e23`, which pins a TOOL node's `tool_id:` (that one IS checked) |

(`e08` covers the typed input boundary; `e04`/`e05` were dropped with the
`kind: match` + payload-union design — tagged data is a discriminant record routed by
`case … on <field>`.)

(`e07-required-missing.yaml` covers `:?` at an AGENT use-site: binding runs
before the model is built, so the omitted-required ref fails the run with no LLM creds. The CODE
form ships as `e22`.)
