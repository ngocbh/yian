# agent_composer — the engine

The implementation of **Agent Composer**: a from-scratch flow engine. A flow is authored 
as **Docker-Compose-shaped YAML**, read by the `compose/` loader into a compiled IR,
and executed by the engine's own scheduler — there is no compilation down to a third-party graph runtime.

The engine is, at heart, a **small functional programming language** — one whose
leaf computations may be LLM agents. If you know OCaml, you already know the
shape of it:

- A **flow is a function**: typed inputs in, typed outputs out, and nothing
  hidden. An **agent is just a flow** whose leaf computation happens to be an LLM
  loop — same contract, same composability.
- Flows **compose**. A node can *be* another flow, so you build a big
  transformation out of small ones, nested to any depth — function composition,
  not a monolith.
- Computation is **pure at the boundary**: a node *returns* its outputs and the
  engine *binds* them; a node never reaches in and mutates shared state. Outputs
  are immutable, typed, losslessly serializable values. Each kind implements
  `_run_impl`; the base `Node.run` is a template method the engine drives that fires
  the node's asserts (record-only, no pool) around the body — so assert firing is
  **node-owned** and uniform. A flow's own `asserts:` are
  lowered onto its synthesized `start`/`end` boundary nodes (boundary checks ride
  `start`'s `post_asserts` — `${input.X}` rewritten to `${output.X}` — and fire against
  START's coerced output; post checks ride `end`), both on the ordinary node path,
  so every assert — top-level or in a called child — runs through one node path.
- The **program's structure is fixed by the author**, the way a function's call
  graph is fixed by its source. The LLM fills leaf boxes; it does **not** rewrite
  the graph at runtime. That referential transparency is what makes runs
  reproducible, checkpointable, and resumable.

## Layout

```
typesys/      typed value system (values) + variable pool                 [leaf, no deps]
events.py     run + node event vocabulary
_state.py     shared scheduler-state vocabulary                           [leaf, no deps]
expr/         one ${...} expression grammar: parse -> AST -> evaluate / ref-walk on the pool
nodes/        pure Node contract + per-kind implementations; spawners return growth seeds
              (start, end, agent, code, model, tool, case, call, map,
               loop, human_input, wait)
compile/      compiled IR + validation; build_grow; stable manifests/fingerprints;
              effective static-path LLM configuration maps
compose/      the Compose-YAML loader: text -> CompiledFlow
              (helpers/nodes/parse/desugar/edges/validate/build/loader)
run.py        the run boundary: ComposedFlow + inputs -> RunResult (run/resume)
runtime/      kind-blind state manager, engine drains, worker pool
suspension/   dormant artificial-pause reasons, commands, checkpoint
llm_clients/  provider clients (lazy-imported) + LLMConfig + model_from_config
tools/        tool registry + process-owned ask_user interaction seam
cli/          the `ac` entry point + main-thread terminal broker
_settings.py  env-based default provider/model
```

Import direction (acyclic): `events <- typesys <- nodes <- compile <- compose`,
with `compile`/`typesys`/`nodes`/`expr` feeding `compose` (the front-end that reads
YAML into the IR), `expr` feeding nodes, `_state.py` feeding compile and
suspension as a shared leaf, and `suspension` feeding runtime. The
package-root `run.py` is the run boundary: it consumes `compose`'s `ComposedFlow` and
drives the `runtime` engine, sitting above both and imported by neither.

Authorable blocking effects stay inside their node calls. An AGENT may invoke the
explicitly granted `ask_user` tool, HUMAN_INPUT always invokes the same host-owned
interaction seam, and WAIT blocks on its absolute UTC deadline. AGENT checks the
run-owned stop signal between model/tool effects; WAIT blocks on that signal. Each returns
`Output`; none creates checkpoint topology. The generic `Pause`/checkpoint path is
dormant custom/artificial compatibility until issue #117 removes it. CODE's
soft-budget watchdog is package-local under `nodes/code/` and times CODE only.
Pooled drives own their executors through a full join: an abnormal dispatch exit signals
cooperative stop before joining, but closing after a terminal pause does not poison resume.

Dynamic growth preserves that direction through dependency inversion. A spawner
node returns `Grow(seed, prune)` without importing `compile`; the kind-aware
`compile.expand.build_grow(spawner, seed)` turns that seed into a deterministic
`Flow`. The package-root run boundary injects `build_grow` into `FlowEngine`, whose
splice path remains kind-blind and never imports `compile.expand` (it may import
`compile.model`, the topology contract it executes). The seed is the durable
identity of the expansion, and compile-owned expansion digests cover the resulting
entry/end ids and topology.

Compile also owns stable identity for the static program. `flow_manifest`
canonicalizes execution semantics, `flow_fingerprint` hashes that manifest, and
`resolve_llm_cascade` returns a complete map keyed by slash-separated static agent
paths. The run boundary snapshots process provider/model defaults once at run
initialization and supplies them explicitly; `apply_effective_llm_map` can replay
the complete map without consulting ambient settings.
