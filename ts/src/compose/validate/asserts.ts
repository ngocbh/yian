/**
 * `asserts:` compile-validate and lower, flow-level and node-local.
 *
 * Two halves of the `asserts:` contract, both in the validate phase:
 *
 * FLOW-LEVEL (`LoweredAsserts` and `lowerFlowAsserts`). A flow's `asserts:` is a list of boolean
 * checks in the `when:`/`asserts:` grammar (comparisons, `and`/`or`/`not`, arithmetic and `in`).
 * This is the compile-time half: it parses each assert, validates every `${...}` reference it
 * reads (reusing the leaf checker `classifyPath` of `compile/validation.ts`: a dangling node or
 * input or a bad record field is a loud `LoadError`), and lowers each assert onto the synthesized
 * START/END boundary nodes so it fires node-locally through `evalNode`:
 *
 * - a `${input.X}`-only assert is **pre (boundary)**: it depends only on the run arguments, so it
 *   rides START's ordinary `postAsserts` and fires against START's bound output record (fail-fast,
 *   before any body node runs), with each `${input.X}` rewritten to `${output.X}` (START's own
 *   bound output).
 * - an assert that reads any `${<id>.output[.X]}` is **post-terminal**: it depends on a produced
 *   value, so it rides the synthesized `EndNode.postAsserts` and fires after the run reaches a
 *   terminal, with each producer ref rewritten to a synthesized END input edge (so the dependency
 *   is structural and survives child cloning and namespacing under `kind: call`).
 *
 * `${system.X}` is forbidden in a flow assert (ambient: no producer to draw an edge from). This
 * module does not run the asserts; the boundary nodes fire them. Lowering replaces only each
 * carrier's executable expression; START/END receive the same display- and origin-bearing
 * carriers.
 *
 * An exact one-segment flow-input name is shorthand for `input.<name>` in this flow-level scope. It
 * therefore wins over a same-named node; node results remain explicit as `node.output`. Dotted
 * paths are not shorthand.
 *
 * NODE-LOCAL (`validateNodeAsserts`). A node assert is scoped to one node: a bare `${name}` is a
 * declared input, `${output}`/`${output.field}` is the node's own output. It validates and
 * classifies each PRE/POST and stamps `node.preAsserts` / `node.postAsserts`. The build conductor
 * optionally supplies the origin and validation range together from one retained item cursor,
 * keyed by node id and authored list index; only the origin crosses into the semantic assertion.
 *
 * Both halves check an assert's field reads narrowed by the assert's own checks: `and`, `or` and
 * `not` short-circuit, so an `isinstance`, union tag comparison or null check before a read decides
 * which union members it sees and whether it may be null (`isinstance(v, Buy) and v.qty > 0`,
 * `v != null and v.qty > 0`), and a check that can never be true is a load error
 * (`helpers/condition_guards.ts`, `helpers/narrowing_facts.ts`).
 *
 * Ported so far: `LoweredAsserts`; the assert lowering and checks are not ported yet (T3-20).
 *
 * Knows about: `compile/validation`, `expr`, `typesys`, the `helpers` peers (`condition_guards`,
 * `isinstance_types`, `narrowing_facts`, `output_contracts`, `errors`), `nodes`,
 * `desugar/inline_calls` (inline-call rejection) and `LocalInput` from `typed_bindings`.
 * Never imports: build, the loader, or the runtime; nothing in helpers, nodes, parse, desugar or
 * edges imports this back.
 */

import type { Assertion, Parameter } from "../../nodes/contracts.ts";

/**
 * A flow's `asserts:` lowered into node-local asserts on the synthesized START/END; frozen.
 *
 * Flow-level assertions become node-local assertions on the synthesized boundary nodes, evaluated
 * uniformly by the `Node.run` wrapper. That makes a cross-node post-assert's dependency structural
 * (each `${producer.output}` ref becomes a real END input edge), so it survives child cloning and
 * namespacing under `kind: call` instead of silently failing. Each ordered `Assertion` owns both
 * the lowered executable expression and stable author-facing display text. Rewrites preserve its
 * opaque origin.
 */
export class LoweredAsserts {
  /**
   * Pre (boundary) asserts, `${input.X}` rewritten to `${output.X}` (they ride START's ordinary
   * `postAsserts`, firing against START's bound output record).
   */
  readonly startPre: readonly Assertion[];
  /** Post asserts, each producer/input ref rewritten to its synth-param ref. */
  readonly endPost: readonly Assertion[];
  /** One untyped `Parameter(<synth>)` per synthesized END input. */
  readonly synthParams: readonly Parameter[];
  /**
   * Each synth name's `${...}` source wired into END. A producer ref uses `${pid.output:-null}`
   * (the `:-null` keeps the edge optional so a skipped producer does not co-skip END); an input ref
   * uses `${input.X}` (always present).
   */
  readonly synthWiring: ReadonlyMap<string, unknown>;

  /** Build the lowered asserts; every list and map left out is empty. */
  constructor(
    options: {
      startPre?: readonly Assertion[];
      endPost?: readonly Assertion[];
      synthParams?: readonly Parameter[];
      synthWiring?: ReadonlyMap<string, unknown>;
    } = {},
  ) {
    this.startPre = options.startPre ?? [];
    this.endPost = options.endPost ?? [];
    this.synthParams = options.synthParams ?? [];
    this.synthWiring = options.synthWiring ?? new Map();
    Object.freeze(this);
  }
}
