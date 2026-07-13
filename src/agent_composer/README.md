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
  are immutable, typed, losslessly serializable values. A flow's own `asserts:` are
  lowered onto its synthesized `start`/`end` boundary nodes (boundary checks fire in
  `StartNode.run` after coercion; post checks ride `end` and fire on the ordinary node
  path), so every assert — top-level or in a called child — runs through one node path.
- The **program's structure is fixed by the author**, the way a function's call
  graph is fixed by its source. The LLM fills leaf boxes; it does **not** rewrite
  the graph at runtime. That referential transparency is what makes runs
  reproducible, checkpointable, and resumable.

## Layout

```
typesys/      typed value system (values) + variable pool                 [leaf, no deps]
events.py     run + node event vocabulary
expr/         one ${...} expression grammar: parse -> AST -> evaluate / ref-walk on the pool
nodes/        Node contract (base) + per-kind implementations
              (start, end, agent, code, model, tool, case, call, map,
               loop, human_input, wait)
compile/      compiled IR (CompiledFlow) + representation-neutral validation
compose/      the Compose-YAML loader: text -> CompiledFlow
              (helpers/nodes/parse/desugar/edges/validate/build/loader)
run.py        the run boundary: LoadedFlow + inputs -> RunResult (run/resume)
runtime/      state_manager, single-threaded engine, parallel worker pool, node watchdog
suspension/   pause reasons, commands, checkpoint
llm_clients/  provider clients (lazy-imported) + LLMConfig + model_from_config
tools/        the tool registry (TOOL_REGISTRY / register_tool / resolve_tools)
cli/          the `ac` command-line entry point
_settings.py  env-based default provider/model
```

Import direction (acyclic): `events <- typesys <- nodes <- compile <- compose`,
with `compile`/`typesys`/`nodes`/`expr` feeding `compose` (the front-end that reads
YAML into the IR), `expr` feeding nodes, and `suspension` feeding runtime. The
package-root `run.py` is the run boundary: it consumes `compose`'s `LoadedFlow` and
drives the `runtime` engine, sitting above both and imported by neither.
