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
 * Knows about: `compile/validation`, `compile/output_contracts`, `expr`, `typesys`, the `helpers`
 * peers (`errors`, `isinstance_types`, `value_nesting`), `nodes`, `compose/nodes/base`,
 * `desugar/inline_calls` (inline-call rejection), the field-read walk of `field_reads`, `SkipModel`
 * from `skips` and `LocalInput` from `typed_bindings`.
 * Never imports: build, the loader, or the runtime; nothing in helpers, nodes, parse, desugar or
 * edges imports this back.
 */

import { classifyPath } from "../../compile/validation.ts";
import { type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import {
  ExpressionError,
  exprRefs,
  optionalInsideRoot,
  parseCondition,
  type RefRenamed,
  rewriteConditionRefs,
} from "../../expr/index.ts";
import { flowCallCalleesInSpans } from "../../expr/template.ts";
import { type Node, NodeKind } from "../../nodes/base.ts";
import {
  Assertion,
  isExactly,
  type OutputContract,
  Parameter,
  ValueContract,
} from "../../nodes/contracts.ts";
import { messagesKey, NEW_MESSAGES, poolRootLength } from "../../typesys/pool.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isInboxType, ValueKind } from "../../typesys/values.ts";
import { desugarCallDirectives } from "../desugar/inline_calls.ts";
import { LoadDiagnostics, LoadError } from "../helpers/errors.ts";
import { collectConstructorNames, collectIsinstanceNames } from "../helpers/isinstance_types.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { assertExactContracts, contractType } from "../../compile/output_contracts.ts";
import { fieldReadErrors, isUntyped, ReadSite } from "./field_reads.ts";
import type { SkipModel } from "./skips.ts";
import type { LocalInput } from "./typed_bindings.ts";

// Tests import these from here.
export { narrows, rootBaseType } from "./field_reads.ts";

// Reserved prefix for the END params synthesized from a lowered post-assert. Authors cannot write
// `__`-prefixed ids, so a synth name never collides with an author id.
const SYNTH_PREFIX = "__assert_";

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

/**
 * Lower a flow's `asserts:` into node-local asserts for the START/END boundary nodes.
 *
 * Validation parses each assert and walks its `${...}` refs via `classifyPath` (a dangling ref or
 * bad field is a loud `LoadError`). An exact unqualified name in `flowInputs` is first treated as
 * `input.<name>` in this scope. The lowering then rewrites each assert so it can fire
 * node-locally, replacing only each assertion carrier's executable expression:
 *
 * - a PRE (boundary) assert rewrites each `${input.X}` ref to `${output.X}`, since START fires
 *   these from its ordinary `postAsserts` against its own bound output record; and
 * - a POST assert turns every non-`output` ref into a synthesized END param (`${producer.output}`
 *   to `${__assert_<pid>}`, an AGENT's `${pid.new_messages}` to `${__assert_<pid>_new_messages}`,
 *   `${input.X}` to `${__assert_input_<X>}`), with a matching
 *   `synthWiring` edge, so the producer value is present in END's record when the assert fires.
 *   Several refs to the same producer or input share one synth param.
 *
 * `${system.X}` is forbidden anywhere in an assert: it is ambient (no producer to draw an edge
 * from), so a `system` ref raises a located `LoadError` telling the author to move the check into a
 * body node.
 *
 * A `${<id>.output.f}` read is field-checked against the producer's contract, and an
 * `${input.X.f}` read against `inputContracts[X]` when the flow input has one. The assert's own
 * checks narrow its reads: `and`, `or` and `not` short-circuit, so an `isinstance` or union tag
 * comparison before a read (`isinstance(input.o, Buy) and input.o.qty > 0`,
 * `not isinstance(input.o, Buy) or input.o.qty > 0`) decides which members it sees. A shorthand
 * check base (`isinstance(o, Buy)`) is qualified like a read, and reported as written. A check
 * that can never be true is a load error, and the `isinstance` types resolve in `registry` (the
 * flow's typedefs). A producer that may be skipped reads null, since END binds it as
 * `${<id>.output :- null}`, so checks and reads take its type as nullable (their errors name the
 * declared type), and a `.` step right after it is a load error (`expr.nullable_field_read`): write
 * `?.` or check it against null first. That holds for an opaque producer too.
 *
 * @param assertList The flow's authored assertion carriers, in order.
 * @param flowInputs The flow's input names.
 * @param validTargets The flow's node ids, and `<id>.new_messages` for each AGENT.
 * @param producers Each typed producer's output contract, and each AGENT's new messages under
 *   `<id>.new_messages`; an opaque producer is absent.
 * @param assertRanges Each assert's range, aligned 1:1 with `assertList` (`null` where none).
 * @param options.inputContracts Each typed flow input's contract (default none).
 * @param options.registry The flow's typedefs (default none).
 * @param options.skips Which producers may be skipped while END runs (default none may).
 * @param options.diagnostics Factory for located load errors (default an unlocated one).
 * @returns The lowered START/END asserts with the synth END params and their wiring.
 * @throws LoadError on the first assert holding a lone surrogate, malformed assert, bad ref,
 *   impossible check, bad field read, `${system}` ref, or read of an inbox input.
 * @throws TypeError when a producer contract is neither an exact `ValueContract` nor a
 *   `PartialRecordContract`.
 */
export function lowerFlowAsserts(
  assertList: readonly Assertion[],
  flowInputs: ReadonlySet<string>,
  validTargets: ReadonlySet<string>,
  producers: ReadonlyMap<string, OutputContract>,
  assertRanges: readonly (SourceRange | null)[] | null = null,
  options: {
    inputContracts?: ReadonlyMap<string, ValueContract> | null;
    registry?: TypeRegistry | null;
    skips?: SkipModel | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): LoweredAsserts {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const skips = options.skips ?? null;
  const inputContracts = options.inputContracts ?? new Map<string, ValueContract>();
  const registry = options.registry ?? new Map();
  // Retain exact and partial carriers so a closed partial root can reject an unknown field without
  // inventing a semantic Type for opaque siblings.
  assertExactContracts(producers);
  // Indexed by loop position, never keyed by assert text, so two identical flow asserts get
  // distinct carets.
  const ranges = assertRanges ?? [];

  const startPre: Assertion[] = [];
  const endPost: Assertion[] = [];
  const synthParams: Parameter[] = [];
  const synthWiring = new Map<string, string>();
  // The producer id or input name to its synth name, shared across all asserts so repeated refs to
  // the same source reuse the one END edge.
  const synthForProducer = new Map<string, string>();
  const synthForInput = new Map<string, string>();
  // Names already handed out, to guard the (near-impossible) collision case.
  const allocated = new Set<string>();

  /** Allocate a collision-proof synth name from `base`. */
  const alloc = (base: string): string => {
    let name = base;
    let i = 1;
    while (flowInputs.has(name) || validTargets.has(name) || allocated.has(name)) {
      name = `${base}_${i}`;
      i += 1;
    }
    allocated.add(name);
    return name;
  };

  /** Resolve one unqualified flow-input name into the explicit input namespace. */
  const qualifyFlowInput = (path: string): string =>
    !path.includes(".") && flowInputs.has(path) ? `input.${path}` : path;

  /** The qualified head when `qualifyFlowInput` changes the path, else `null`. */
  const qualifiedOrNull = (path: string): RefRenamed | null =>
    qualifyFlowInput(path) !== path ? { text: `input.${path}`, used: 1 } : null;

  // The values whose field reads are checked, with the one contract each has. An AGENT's new
  // messages sit in `producers` under their own pool name, `<id>.new_messages`, which is the root.
  const valueRoot = (key: string): string => (validTargets.has(key) && key.endsWith(`.${NEW_MESSAGES}`) ? key : `${key}.output`);
  const roots = new Map<string, readonly OutputContract[]>([
    ...[...inputContracts].map(([name, contract]): [string, OutputContract[]] => [`input.${name}`, [contract]]),
    ...[...producers].map(([key, contract]): [string, OutputContract[]] => [valueRoot(key), [contract]]),
  ]);
  // An opaque producer (no contract) gets no name check, but its reads are still checked for null.
  const targetRoots = [...validTargets].map(valueRoot);
  const opaqueRoots = new Set(targetRoots.filter((root) => !roots.has(root)));
  // The producers that may be skipped while END runs, which then read null. Only an assert reads
  // them, and each costs a skip query.
  const producerRoots = new Set(
    assertList.length === 0 || skips === null
      ? []
      : targetRoots.filter((root) => skips.maySkipWhileRunning([root.split(".")[0]!], null)),
  );

  assertList.forEach((assertion, i) => {
    const expr = assertion.expression;
    const rng = ranges[i] ?? null;
    checkWrittenValue(expr, `flow field 'asserts[${i}]'`, rng, diagnostics);
    // Parse-check and collect refs on the one unified condition walk (any brace spelling).
    let authored: ExprNode;
    try {
      authored = parseCondition(expr);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      throw diagnostics.error(
        `assert ${reprText(assertion.display)} is not a valid boolean expression: ${error.message}`,
        { range: rng, cause: error },
      );
    }
    // An unqualified input `X` reads as `input.X`, a two-name root.
    const qualified = parseCondition(rewriteConditionRefs(expr, qualifiedOrNull));
    const insideRoot = optionalInsideRoot(qualified, poolRootLength);
    if (insideRoot !== null) {
      throw diagnostics.error(`assert ${reprText(assertion.display)}: ${insideRoot}`, { range: rng });
    }
    const refs = exprRefs(authored).map(qualifyFlowInput);

    // Validate each ref, forbid `${system}`, and detect any producer ref (which makes it POST).
    let isPost = false;
    for (const path of new Set(refs)) {
      const err = classifyPath(path, validTargets, flowInputs);
      if (err !== null) throw diagnostics.error(`assert ${reprText(assertion.display)}: ${err}`, { range: rng });
      const head = path.split(".")[0]!;
      if (head === "system") {
        throw diagnostics.error(
          `assert ${reprText(assertion.display)}: the \`system\` namespace is not allowed in asserts ` +
            "(it is ambient — there is no producer to draw an edge from; move the " +
            "check into a body node)",
          { range: rng },
        );
      }
      const input = head === "input" ? path.split(".")[1] : undefined;
      if (input !== undefined && isInboxType(inputContracts.get(input)?.type)) {
        throw diagnostics.error(
          `assert ${reprText(assertion.display)}: \${input.${input}} is an inbox; an assert can't read it`,
          { range: rng },
        );
      }
      if (head !== "input") isPost = true;
    }

    // Impossible checks and field reads, narrowed by the assert's own checks. END binds each
    // producer it reads as `${<id>.output :- null}`, so a producer the flow's routing skips reads
    // null.
    const errors = fieldReadErrors(qualified, roots, registry, {
      nullableRoots: producerRoots,
      authored,
      opaqueRoots,
      rootLength: poolRootLength,
      why: (root) => `node ${reprText(root.split(".")[0]!)} may be skipped`,
    });
    if (errors.length > 0) {
      const { message, code } = errors[0]!;
      throw diagnostics.error(`assert ${reprText(assertion.display)}: ${message}`, { range: rng, code });
    }

    if (!isPost) {
      // PRE (boundary): START fires these from its ordinary `postAsserts` against its bound
      // output record, so `${output.X}` sees the validated, defaulted value START binds.
      // Unqualified flow-input names first acquire the `input.` prefix, which then maps to
      // `output.`.
      const preRename = (ref: string): RefRenamed | null => {
        const path = qualifyFlowInput(ref);
        // A bare input name is one name of the read; `input.<k>` is two.
        const used = path === ref ? 2 : 1;
        if (path.startsWith("input.")) return { text: `output.${path.split(".")[1]}`, used };
        return null;
      };
      const lowered = rewriteConditionRefs(expr, preRename);
      startPre.push(new Assertion(lowered, assertion.display, { origin: assertion.origin }));
      return;
    }

    // POST: turn each non-`output` ref into a synth END param, wired as an edge.
    const postRename = (ref: string): RefRenamed | null => {
      const segs = qualifyFlowInput(ref).split(".");
      const head = segs[0]!;
      if (head === "input" && segs.length >= 2) {
        const x = segs[1]!;
        let synth = synthForInput.get(x);
        if (synth === undefined) {
          synth = alloc(`${SYNTH_PREFIX}input_${x}`);
          synthForInput.set(x, synth);
          synthParams.push(new Parameter(synth));
          synthWiring.set(synth, `\${input.${x}}`);
        }
        return { text: synth, used: qualifyFlowInput(ref) === ref ? 2 : 1 };
      }
      // An AGENT's new messages: `pid.new_messages[.f...]`, bound like an output read.
      if (segs.length >= 2 && segs[1] === NEW_MESSAGES && validTargets.has(messagesKey(head))) {
        const key = messagesKey(head);
        let synth = synthForProducer.get(key);
        if (synth === undefined) {
          synth = alloc(`${SYNTH_PREFIX}${head}_${NEW_MESSAGES}`);
          synthForProducer.set(key, synth);
          synthParams.push(new Parameter(synth));
          synthWiring.set(synth, `\${${key}:-null}`);
        }
        return { text: synth, used: 2 };
      }
      // A producer-output ref: `pid.output[.f.g...]`, pid a node id.
      if (validTargets.has(head) && segs.length >= 2 && segs[1] === "output") {
        let synth = synthForProducer.get(head);
        if (synth === undefined) {
          synth = alloc(`${SYNTH_PREFIX}${head}`);
          synthForProducer.set(head, synth);
          synthParams.push(new Parameter(synth));
          // `:-null` makes the END edge optional: a conditionally skipped producer binds null
          // instead of co-skipping END to dead.
          synthWiring.set(synth, `\${${head}.output:-null}`);
        }
        return { text: synth, used: 2 };
      }
      return null; // anything else (e.g. an `input` head without a field), kept
    };
    const lowered = rewriteConditionRefs(expr, postRename);
    endPost.push(new Assertion(lowered, assertion.display, { origin: assertion.origin }));
  });

  return new LoweredAsserts({ startPre, endPost, synthParams, synthWiring });
}

// Kinds whose node asserts move onto the spawned child flow's END. MAP is absent because every MAP
// node assert is already rejected.
const SPAWNER_KINDS: ReadonlySet<NodeKind> = new Set([NodeKind.CALL, NodeKind.LOOP]);

/**
 * Pool heads forbidden in a node-local assert (use bare `${name}` / `${output}` instead); only the
 * singular heads exist. Exported for tests.
 */
export const POOL_HEADS: ReadonlySet<string> = new Set(["input", "system", "item"]);

/**
 * Validate, classify and stamp each node's `asserts:`, the per-node contract.
 *
 * A node assert is node-local: a bare `${name}` is one of the node's declared inputs;
 * `${output}`/`${output.field}` is the node's own output. Exact contracts walk their semantic type;
 * partial records enforce their closed root field set and walk any independently proven field
 * contract.
 *
 * Rejected (located `LoadError`): text holding a lone surrogate, a pool head
 * (`${input/system/item.X}`), an inline `${call}`, `${output.field}` on an output that is neither a
 * record nor a `file` (whose `uri`, `mime` and `name` it can read), a `${name.field}` read of a
 * declared input when every alternative of `localContracts[nid][name]` (a `LocalInput` from
 * `localInputContracts`) rejects it (a field only some union members have is readable on a node in
 * `partialUnionNodes`; a name with no entry is not field-checked), a declared input named
 * `output`, `isinstance` or a constructor on a CALL/MAP/LOOP node (its asserts run in the child
 * flow), a read of a local bound to an inbox, and any node assert on a mapped call (a `MapNode`,
 * `kind: map`: the per-node hook has no `${item}` scope). A call inside `${...}` to a name in
 * `typeNames` (the flow's `typedefs:` names) is a constructor, not an inline call.
 *
 * Classifies each PRE (no `${output}`) or POST (reads `${output}`) and stamps `node.preAsserts` /
 * `node.postAsserts`. An assert's own checks narrow its reads: `and`, `or` and `not`
 * short-circuit, so an `isinstance` or union tag comparison before a read
 * (`isinstance(v, Buy) and v.qty > 0`, `v.action != 'buy' or v.qty > 0`) decides which members it
 * sees, with the `isinstance` types resolved in `registry` (the flow's typedefs). A local name's
 * alternatives carry the branch that routes the node, so a check is a load error when it can never
 * be true under every alternative. Checks on `${output}` are tested too, whether or not the assert
 * reads a field of it. A CALL/LOOP node's member-only hint compares the tag instead of naming
 * `isinstance`. Runs in the shared `assemble`, so a `defs:` child flow's internal nodes get this
 * too.
 *
 * @param nodes The built nodes by id; each one with asserts is stamped.
 * @param descriptors The node descriptors by id, carrying the authored `asserts` and their ranges.
 * @param options.assertionSiteAt Optionally supplies each carrier's opaque origin and validation
 *   range from the same authored occurrence, keyed by node id and original authored index.
 * @param options.typeNames The flow's `typedefs:` names (default none).
 * @param options.localContracts Each node's local input contracts (default none).
 * @param options.partialUnionNodes The nodes whose locals may read member-only fields (default none).
 * @param options.skips Which producers may be skipped while a node runs: a local bound straight
 *   from one reads null (default none may).
 * @param options.baseLocals Each node's local input contracts without the CASE routing's narrowing,
 *   to tell an untyped local the routing narrowed (default `localContracts`).
 * @param options.registry The flow's typedefs (default none).
 * @param options.diagnostics Factory for located load errors (default an unlocated one).
 * @throws LoadError holding every error, one per line, located at the first available exact
 *   assertion, offending field, or node range; code `expr.nullable_field_read` when the first is a
 *   nullable field read.
 * @throws TypeError when a node's output contract is neither an exact `ValueContract` nor a
 *   `PartialRecordContract`.
 */
export function validateNodeAsserts(
  nodes: ReadonlyMap<string, Node>,
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  options: {
    assertionSiteAt?:
      | ((nodeId: string, index: number) => readonly [OriginRef | null, SourceRange | null] | null)
      | null;
    typeNames?: ReadonlySet<string>;
    localContracts?: ReadonlyMap<string, ReadonlyMap<string, LocalInput>> | null;
    partialUnionNodes?: ReadonlySet<string>;
    skips?: SkipModel | null;
    baseLocals?: ReadonlyMap<string, ReadonlyMap<string, LocalInput>> | null;
    registry?: TypeRegistry | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const skips = options.skips ?? null;
  const assertionSiteAt = options.assertionSiteAt ?? null;
  const typeNames = options.typeNames ?? new Set<string>();
  const localContracts = options.localContracts ?? new Map<string, ReadonlyMap<string, LocalInput>>();
  const partialUnionNodes = options.partialUnionNodes ?? new Set<string>();
  const registry = options.registry ?? new Map();
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const baseLocals = options.baseLocals ?? localContracts;
  const errors: [message: string, range: SourceRange | null, code?: string][] = [];
  for (const [nid, node] of nodes) {
    const desc = descriptors.get(nid);
    // Not every descriptor has `asserts` (WAIT doesn't).
    const asserts = (desc !== undefined && "asserts" in desc ? (desc.asserts as readonly string[] | null) : null) ?? [];
    if (desc === undefined || asserts.length === 0) continue;
    const aranges = desc.assertRanges;
    const fieldRange = (kind: "field" | "input", key: string) => desc.fieldRanges.get(siteKey(kind, key)) ?? null;
    const assertsRange = fieldRange("field", "asserts") ?? desc.range;
    if (node.bindsPerItem) {
      errors.push([
        `node ${reprText(nid)}: \`asserts:\` is not allowed on a mapped call ` +
          "(`kind: map`) — assert inside the def, or on a downstream node",
        assertsRange,
      ]);
      continue;
    }
    const declared = new Set((node.params ?? []).map((p) => p.name));
    const nodeLocals = localContracts.get(nid) ?? new Map<string, LocalInput>();
    if (declared.has("output")) {
      errors.push([
        `node ${reprText(nid)}: a declared input named 'output' collides with the ` +
          "node-assert `${output}` keyword — rename the input",
        fieldRange("input", "output") ?? assertsRange,
      ]);
      continue;
    }
    const pre: Assertion[] = [];
    const post: Assertion[] = [];
    asserts.forEach((a, i) => {
      const site = assertionSiteAt !== null ? assertionSiteAt(nid, i) : null;
      const [origin, rng] = site ?? [null, aranges[i] ?? null];
      const errorRange = rng ?? assertsRange;
      try {
        checkWrittenValue(a, `node ${reprText(nid)} field 'asserts[${i}]'`, errorRange, diagnostics);
      } catch (error) {
        if (!(error instanceof LoadError)) throw error;
        errors.push([error.message, errorRange]);
        return;
      }
      // Node asserts are node-local: an inline call (a whole-value `call(...)` directive or the
      // retired `${flow(args)}` span) belongs in a flow-level assert, not here.
      let directiveCalls: readonly unknown[];
      try {
        directiveCalls = desugarCallDirectives(a, () => "__a")[1];
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        directiveCalls = ["(malformed call)"];
      }
      // A span that doesn't parse is no call; `parseCondition` below reports its real error.
      let spanCallees: readonly string[];
      try {
        spanCallees = flowCallCalleesInSpans(a, typeNames);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        spanCallees = [];
      }
      if (directiveCalls.length > 0 || spanCallees.length > 0) {
        errors.push([
          `node ${reprText(nid)} assert ${reprText(a)}: an inline call is not allowed in a ` +
            "node-local assert (use a flow-level assert)",
          errorRange,
        ]);
        return;
      }
      // A spawner's asserts are lowered onto the child's END, which resolves `isinstance` and
      // constructor types in the child flow, not here.
      const kind = node.kind;
      if (SPAWNER_KINDS.has(kind)) {
        let used: string | null;
        try {
          used = collectIsinstanceNames(a, { condition: true }).length > 0 ? "isinstance" : null;
          const constructors = collectConstructorNames(a, { condition: true });
          if (used === null && constructors.length > 0) used = `${constructors[0]}(...)`;
        } catch (error) {
          if (!(error instanceof ExpressionError)) throw error;
          used = null;
        }
        if (used !== null) {
          errors.push([
            `node ${reprText(nid)} assert ${reprText(a)}: ${used} in a ${kind} ` +
              "node's asserts is not supported; assert inside the child flow",
            errorRange,
          ]);
          return;
        }
      }
      // Parse-check and collect refs on the one unified condition walk (any spelling).
      let tree: ExprNode;
      try {
        tree = parseCondition(a);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        errors.push([
          `node ${reprText(nid)} assert ${reprText(a)}: not a valid boolean expression (${error.message})`,
          errorRange,
        ]);
        return;
      }
      let readsOutput = false;
      // The values whose checks and field reads are checked, with the contracts each may have, and
      // where each local is bound from.
      const roots = new Map<string, readonly OutputContract[]>();
      const unionRoots = new Map<string, string>();
      const unwalked = new Set<string>();
      for (const ref of new Set(exprRefs(tree))) {
        const segs = ref.split(".");
        const head = segs[0] ?? "";
        if (head === "output") {
          readsOutput = true;
          const contract = node.outputContract;
          if (contract !== null) {
            assertExactContracts(new Map([[nid, contract]]));
            roots.set("output", [contract]);
          }
          // A file output reaches the shared walk, which types its `uri`, `mime` and `name` and
          // rejects any other field.
          if (
            segs.length > 1 &&
            isExactly(contract, ValueContract) &&
            contract.type.kind !== ValueKind.OBJECT &&
            contract.type.kind !== ValueKind.ANY &&
            contract.type.kind !== ValueKind.FILE
          ) {
            errors.push([
              `node ${reprText(nid)} assert: \${${ref}} — the node output is not a record, so it has ` +
                `no field ${reprText(segs.slice(1).join("."))}`,
              errorRange,
            ]);
            unwalked.add("output");
          }
        } else if (declared.has(head)) {
          const local = nodeLocals.get(head);
          const inbox = local?.alternatives.some((alt) => isExactly(alt, ValueContract) && isInboxType(alt.type));
          if (inbox) {
            errors.push([`node ${reprText(nid)} assert: \${${head}} is an inbox; an assert can't read it`, errorRange]);
          } else if (local !== undefined && local.alternatives.length > 0) {
            roots.set(head, local.alternatives);
            if (local.source !== null) unionRoots.set(head, local.source);
          }
        } else if (POOL_HEADS.has(head)) {
          errors.push([
            `node ${reprText(nid)} assert: \${${ref}} — node asserts are node-local; ` +
              "use `${name}` (a declared input) or `${output}`",
            errorRange,
          ]);
        } else {
          errors.push([
            `node ${reprText(nid)} assert: \${${ref}} is not a declared input or \`\${output}\``,
            errorRange,
          ]);
        }
      }
      // A spawner assert with `isinstance` was rejected above, so its impossible checks are never
      // reported on top of that error.
      /** The producer a local is bound straight from, when it may be skipped while the node runs. */
      const skippedProducer = (local: string): string | null => {
        const source = nodeLocals.get(local)?.source ?? null;
        if (skips === null || source === null) return null;
        const names = source.split(".");
        if (poolRootLength(names) !== 2 || (names[1] !== "output" && names[1] !== NEW_MESSAGES)) return null;
        return skips.maySkipWhileRunning([names[0]!], nid) ? names[0]! : null;
      };
      const baseNodeLocals = baseLocals.get(nid) ?? new Map<string, LocalInput>();
      const messages = fieldReadErrors(tree, roots, registry, {
        partialUnionRoots: partialUnionNodes.has(nid)
          ? new Set([...roots.keys()].filter((root) => root !== "output"))
          : new Set(),
        unwalkedRoots: unwalked,
        site: new ReadSite({ spawner: SPAWNER_KINDS.has(kind), unionRoots }),
        nullableRoots: new Set([...roots.keys()].filter((root) => skippedProducer(root) !== null)),
        why: (root) => {
          const producer = skippedProducer(root);
          return producer === null
            ? "it is Optional"
            : `node ${reprText(producer)} may be skipped while ${reprText(nid)} runs`;
        },
        // `output` keeps the node's own contract; a local is untyped by its unnarrowed binding.
        untyped: (root) =>
          ((root === "output" ? roots.get(root) : baseNodeLocals.get(root)?.alternatives) ?? []).every((contract) =>
            isUntyped(contractType(contract)),
          ),
      });
      for (const { message, code } of messages) {
        errors.push([`node ${reprText(nid)} assert: ${message}`, errorRange, code]);
      }
      (readsOutput ? post : pre).push(new Assertion(a, a, { origin }));
    });
    node.preAsserts = pre;
    node.postAsserts = post;
  }
  if (errors.length > 0) {
    const firstRange = errors.find(([, range]) => range !== null)?.[1] ?? null;
    throw diagnostics.error(
      "flow has invalid node asserts:\n  " + errors.map(([message]) => message).join("\n  "),
      { range: firstRange, code: errors[0]![2] },
    );
  }
}
