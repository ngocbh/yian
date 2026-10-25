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
 * Knows about: `compile/validation`, `expr`, `typesys`, the `helpers` peers (`condition_guards`,
 * `isinstance_types`, `narrowing_facts`, `output_contracts`, `errors`, `tags`, `value_nesting`),
 * `nodes`, `compose/nodes/base`,
 * `desugar/inline_calls` (inline-call rejection) and `LocalInput` from `typed_bindings`.
 * Never imports: build, the loader, or the runtime; nothing in helpers, nodes, parse, desugar or
 * edges imports this back.
 */

import { classifyPath } from "../../compile/validation.ts";
import { type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import {
  conditionRefs,
  ExpressionError,
  exprRefs,
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
  PartialRecordContract,
  sameOutputContract,
  ValueContract,
} from "../../nodes/contracts.ts";
import { messagesKey, NEW_MESSAGES } from "../../typesys/pool.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isInboxType, jsTypeName, Type, unionTagFields, ValueKind } from "../../typesys/values.ts";
import { desugarCallDirectives } from "../desugar/inline_calls.ts";
import {
  type Check,
  type CheckLiteral,
  checkText,
  type Guard,
  guardedRefs,
  IsinstanceCheck,
  literalsInContext,
  literalText,
  NotNullCheck,
  sameGuard,
  TagCheck,
  unknownCheck,
} from "../helpers/condition_guards.ts";
import { LoadDiagnostics, LoadError } from "../helpers/errors.ts";
import { collectConstructorNames, collectIsinstanceNames } from "../helpers/isinstance_types.ts";
import {
  type BaseTypeOf,
  checkedRefs,
  guardedReadTypes,
  ImpossibleCheck,
  impossibleCheckMessage,
  impossibleChecks,
  ImpossibleReason,
  NOT_NARROWABLE,
  type NarrowedRead,
  narrowedReadError,
  type NotNarrowable,
  unresolvedType,
} from "../helpers/narrowing_facts.ts";
import { outputContractPathError, projectOutputContract } from "../helpers/output_contracts.ts";
import { namedTypeLabel } from "../helpers/tags.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import type { LocalInput } from "./typed_bindings.ts";

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
 * flow's typedefs). A producer read may be null, since END binds it as `${<id>.output :- null}` and
 * a skipped producer reads null, so checks and reads take each producer's type as nullable; their
 * errors name the declared type.
 *
 * @param assertList The flow's authored assertion carriers, in order.
 * @param flowInputs The flow's input names.
 * @param validTargets The flow's node ids, and `<id>.new_messages` for each AGENT.
 * @param producers Each typed producer's output contract, and each AGENT's new messages under
 *   `<id>.new_messages`; an opaque producer is absent.
 * @param assertRanges Each assert's range, aligned 1:1 with `assertList` (`null` where none).
 * @param options.inputContracts Each typed flow input's contract (default none).
 * @param options.registry The flow's typedefs (default none).
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
    diagnostics?: LoadDiagnostics | null;
  } = {},
): LoweredAsserts {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const inputContracts = options.inputContracts ?? new Map<string, ValueContract>();
  const registry = options.registry ?? new Map();
  // Retain exact and partial carriers so a closed partial root can reject an unknown field without
  // inventing a semantic Type for opaque siblings.
  for (const [nodeId, contract] of producers) {
    if (!isExactly(contract, ValueContract) && !isExactly(contract, PartialRecordContract)) {
      throw new TypeError(
        "producer contracts must be ValueContract or PartialRecordContract values, " +
          `got ${jsTypeName(contract)} for ${reprText(nodeId)}`,
      );
    }
  }
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

  // A synth name must be a legal ref path (defensive; the prefix and author id components are
  // always parseable in practice).
  const parseable = (name: string): boolean => {
    try {
      conditionRefs("${" + name + "}");
      return true;
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      return false;
    }
  };

  /** Allocate a collision-proof synth name from `base`, verified parseable as a ref. */
  const alloc = (base: string): string => {
    let name = base;
    let i = 1;
    while (flowInputs.has(name) || validTargets.has(name) || allocated.has(name) || !parseable(name)) {
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
  const producerRoots = new Set([...producers.keys()].map(valueRoot));

  assertList.forEach((assertion, i) => {
    const expr = assertion.expression;
    const rng = ranges[i] ?? null;
    checkWrittenValue(expr, `flow field 'asserts[${i}]'`, rng, diagnostics);
    // 1. Parse-check and collect refs on the one unified condition walk (any brace spelling).
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
    const refs = exprRefs(authored).map(qualifyFlowInput);

    // 2. Validate each ref, forbid `${system}`, and detect any producer ref (which makes it POST).
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

    // 3. Impossible checks and field reads, narrowed by the assert's own checks. END binds each
    // producer it reads as `${<id>.output :- null}`, so a producer the flow's routing skips reads
    // null.
    const qualified = parseCondition(rewriteConditionRefs(expr, qualifiedOrNull));
    const errors = conditionErrors(qualified, roots, registry, { nullableRoots: producerRoots, authored });
    if (errors.length > 0) {
      throw diagnostics.error(`assert ${reprText(assertion.display)}: ${errors[0]}`, { range: rng });
    }

    if (!isPost) {
      // 4. PRE (boundary): START fires these from its ordinary `postAsserts` against its bound
      // output record, so `${output.X}` sees the validated, defaulted value START binds.
      // Unqualified flow-input names first acquire the `input.` prefix, which then maps to
      // `output.`.
      const preRename = (ref: string): RefRenamed | null => {
        const path = qualifyFlowInput(ref);
        // A bare input name is one name of the read; `input.<k>` is two.
        const used = path === ref ? 2 : 1;
        if (path === "input") return { text: "output", used: 1 }; // defensive; a bare `${input}` head shouldn't appear
        if (path.startsWith("input.")) return { text: `output.${path.split(".")[1]}`, used };
        return null;
      };
      const lowered = rewriteConditionRefs(expr, preRename);
      startPre.push(new Assertion(lowered, assertion.display, { origin: assertion.origin }));
      return;
    }

    // 5. POST: turn each non-`output` ref into a synth END param, wired as an edge.
    const postRename = (ref: string): RefRenamed | null => {
      const segs = qualifyFlowInput(ref).split(".");
      const head = segs[0]!;
      if (head === "output") return null; // the node's own output, kept (defensive)
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

/** Return the root `path` is at or below, else `null`. */
function rootOf(path: string, roots: Iterable<string>): string | null {
  for (const root of roots) {
    if (path === root || path.startsWith(root + ".")) return root;
  }
  return null;
}

/** Return the field names of `path` below `root`, which it is at or below. */
function below(path: string, root: string): string[] {
  return path === root ? [] : path.slice(root.length + 1).split(".");
}

/**
 * Return the declared type of a base path below one of `choice`'s roots, under that root's chosen
 * contract, made nullable below a root in `nullableRoots`; other bases are not narrowed. Exported
 * for tests.
 *
 * @param choice The contract chosen for each root.
 * @param nullableRoots The roots that may be null whatever their contracts say (default none).
 * @returns The site's `BaseTypeOf`: `null` for a base the contract doesn't prove, `NOT_NARROWABLE`
 *   for a base below no root.
 */
export function rootBaseType(
  choice: ReadonlyMap<string, OutputContract>,
  nullableRoots: ReadonlySet<string> = new Set(),
): BaseTypeOf {
  return (base: string): Type | null | NotNarrowable => {
    const root = rootOf(base, choice.keys());
    if (root === null) return NOT_NARROWABLE;
    const projected = projectOutputContract(choice.get(root)!, below(base, root));
    if (!isExactly(projected, ValueContract)) return null;
    if (nullableRoots.has(root)) return projected.type.replace({ nullable: true });
    return projected.type;
  };
}

/** Return `contracts` without repeats, in order. */
function distinct(contracts: readonly OutputContract[]): OutputContract[] {
  const kept: OutputContract[] = [];
  for (const contract of contracts) {
    if (!kept.some((seen) => sameOutputContract(contract, seen))) kept.push(contract);
  }
  return kept;
}

/**
 * Return whether `check` can prove or rule out anything about `root` under some contract. Exported
 * for tests.
 *
 * A tag comparison on a value that is no union with that tag field (`v.a == 'x'` on a record) is an
 * ordinary comparison, so it narrows nothing; nor does a null check on a value that is never null.
 *
 * @param check A check from a literal of the condition.
 * @param root The root path whose values the check may narrow.
 * @param contracts The contracts the root's value may have.
 * @param nullable Whether the root may be null whatever its contracts say, which makes every value
 *   below it nullable too (default `false`).
 * @returns Whether the check narrows the root.
 * @throws TypeError for a check of a kind this function doesn't know.
 */
export function narrows(
  check: Check,
  root: string,
  contracts: readonly OutputContract[],
  nullable = false,
): boolean {
  if (rootOf(check.base, [root]) === null) return false;
  if (check instanceof IsinstanceCheck) return true;
  if (!(check instanceof TagCheck || check instanceof NotNullCheck)) throw unknownCheck(check);
  // `rootBaseType` makes every base below such a root nullable, so a null check there narrows
  // even when no contract lets the value be null.
  if (check instanceof NotNullCheck && nullable) return true;
  const rest = below(check.base, root);
  for (const contract of contracts) {
    const projected = projectOutputContract(contract, rest);
    if (!isExactly(projected, ValueContract)) {
      // An unproven base takes its type from the checks on the value around it.
      return true;
    }
    const typ = projected.type;
    if (check instanceof NotNullCheck) {
      if (typ.nullable) return true;
    } else if (typ.variants !== undefined && unionTagFields(typ).includes(check.field)) {
      return true;
    }
  }
  return false;
}

/** What a condition proves under one choice of root contracts. */
interface ChoiceView {
  readonly choice: ReadonlyMap<string, OutputContract>;
  readonly baseTypeOf: BaseTypeOf;
  readonly impossible: readonly ImpossibleCheck[];
  readonly checked: readonly [string, Guard | null][];
}

/** How a site words a field-read error; frozen. */
class ReadSite {
  /**
   * Whether the condition runs in a spawned child flow, where `isinstance` is not allowed, so a
   * member-only hint compares the tag instead.
   */
  readonly spawner: boolean;
  /** The pool path each root is bound from, named by a spawner's case hint. */
  readonly unionRoots: ReadonlyMap<string, string>;

  constructor(options: { spawner?: boolean; unionRoots?: ReadonlyMap<string, string> } = {}) {
    this.spawner = options.spawner ?? false;
    this.unionRoots = options.unionRoots ?? new Map();
    Object.freeze(this);
  }

  /** Return the read's error on `contract`, else `null`. */
  error(
    contract: OutputContract,
    fields: readonly string[],
    ref: string,
    options: { partialUnionFields: boolean; unionRoot?: string | null },
  ): string | null {
    return outputContractPathError(contract, fields, ref, {
      partialUnionFields: options.partialUnionFields,
      unionRoot: this.spawner ? (options.unionRoot ?? null) : null,
      isinstanceHint: !this.spawner,
      tagHint: this.spawner,
    });
  }
}

/** Return `check` rebuilt with `changes`, as its frozen class can't be updated in place. */
function replaceCheck(
  check: ImpossibleCheck,
  changes: { text?: string; base?: string; alwaysTrue?: string; baseLabel?: string },
): ImpossibleCheck {
  return new ImpossibleCheck(changes.text ?? check.text, changes.base ?? check.base, check.reason, {
    baseLabel: changes.baseLabel ?? check.baseLabel,
    declared: check.declared,
    typeLabel: check.typeLabel,
    alwaysTrue: changes.alwaysTrue ?? check.alwaysTrue,
    value: check.value,
    field: check.field,
    tags: check.tags,
  });
}

/**
 * Return a map from a report on `tree` back to how `authored`, the same condition before its
 * references were rewritten, spells the literal.
 */
function authoredSpelling(
  tree: ExprNode,
  authored: ExprNode | null,
): (check: ImpossibleCheck) => ImpossibleCheck {
  const spelled = new Map<string, CheckLiteral>();
  if (authored !== null) {
    const rewritten = literalsInContext(tree);
    const written = literalsInContext(authored);
    if (rewritten.length === written.length) {
      rewritten.forEach((literal, index) => {
        const text = literalText(literal);
        if (!spelled.has(text)) spelled.set(text, written[index]!);
      });
    }
  }
  return (check) => {
    const original = spelled.get(check.text);
    if (original === undefined) return check;
    const alwaysTrue = check.alwaysTrue ? checkText(original.check, true) : "";
    return replaceCheck(check, { text: literalText(original), base: original.check.base, alwaysTrue });
  };
}

/**
 * Return an assert's impossible checks and field-read errors, each message once.
 *
 * `roots` maps each path whose field reads are checked (a node-local name, `output`, `input.X`,
 * `<id>.output`) to the contracts its value may have; reads below a root in `unwalkedRoots` are
 * left to the site, and a root in `nullableRoots` may also be null whatever its contract says.
 * `and`, `or` and `not` short-circuit, so the checks before a read (`isinstance`, union tag
 * comparisons and null checks) narrow the values it can see (`guardedReadTypes`): a read is fine
 * when any narrowed type allows it, and otherwise the first one's error is reported, naming the
 * narrowed type when it lacks the field. A read nothing narrows is walked on the root's contract.
 * A read that runs only when a reported impossible check is true (`checkedRefs`), or guarded by an
 * `isinstance` whose type name doesn't resolve, is not checked. `authored` is the condition as
 * written when `tree` has its references rewritten; reports spell checks as written.
 *
 * A root with several distinct contracts (a node-local name's alternatives, which carry the branch
 * that routes the node) is checked under each: a check is impossible only when it is impossible
 * under every contract of its base's root, a read is skipped only when it is dead under every
 * contract of some root, and a read is an error only when every contract of its root where it runs
 * rejects it. Facts never cross roots, so each root's contracts are tried with every other such
 * root left unnarrowed, which costs the sum of the alternatives rather than their product. Only
 * roots some check can narrow are tried; reads below the others are walked on each contract. Each
 * distinct (read, guard) pair is checked, since one path can be read both narrowed and not.
 *
 * @param tree The parsed condition.
 * @param roots The checked roots and the contracts each may have.
 * @param registry The flow's typedefs.
 * @param options.partialUnionRoots Roots below which a field only some union members have can be
 *   read (default none).
 * @param options.unwalkedRoots Roots whose reads the site checks itself (default none).
 * @param options.nullableRoots Roots that may be null whatever their contracts say (default none).
 * @param options.site How the site words a field-read error (default a non-spawner site).
 * @param options.authored The condition as written, when `tree` is rewritten (default `null`).
 * @returns The messages without a site prefix, in order, without repeats.
 */
function conditionErrors(
  tree: ExprNode,
  roots: ReadonlyMap<string, readonly OutputContract[]>,
  registry: TypeRegistry,
  options: {
    partialUnionRoots?: ReadonlySet<string>;
    unwalkedRoots?: ReadonlySet<string>;
    nullableRoots?: ReadonlySet<string>;
    site?: ReadSite;
    authored?: ExprNode | null;
  } = {},
): string[] {
  const {
    partialUnionRoots = new Set<string>(),
    unwalkedRoots = new Set<string>(),
    nullableRoots = new Set<string>(),
    site = new ReadSite(),
    authored = null,
  } = options;
  const refs = guardedRefs(tree);
  const used = [...roots.keys()].filter((root) => refs.some(([ref]) => rootOf(ref, [root]) !== null));
  if (used.length === 0) return [];
  const contracts = new Map(used.map((root) => [root, distinct(roots.get(root)!)]));
  const literals = literalsInContext(tree);
  const narrowed = used.filter((root) =>
    literals.some((literal) => narrows(literal.check, root, contracts.get(root)!, nullableRoots.has(root))),
  );
  const fixed = new Map<string, OutputContract>();
  for (const root of narrowed) {
    if (contracts.get(root)!.length === 1) fixed.set(root, contracts.get(root)![0]!);
  }
  const varied = narrowed.filter((root) => contracts.get(root)!.length > 1);
  let choices = varied.flatMap((root) =>
    contracts.get(root)!.map((contract) => new Map([...fixed, [root, contract]])),
  );
  if (narrowed.length > 0 && varied.length === 0) choices = [fixed];
  const views: ChoiceView[] = choices.map((choice) => {
    const baseTypeOf = rootBaseType(choice, nullableRoots);
    return {
      choice,
      baseTypeOf,
      impossible: impossibleChecks(tree, baseTypeOf, registry),
      checked: checkedRefs(tree, baseTypeOf, registry),
    };
  });

  /** The views that try each of `root`'s contracts; all views for any other root. */
  const viewsOf = (root: string | null): ChoiceView[] =>
    root === null || !varied.includes(root) ? views : views.filter((view) => view.choice.has(root));

  const spell = authoredSpelling(tree, authored);
  const order = new Map<string, number>();
  literals.forEach((literal, index) => {
    const text = literalText(literal);
    if (!order.has(text)) order.set(text, index);
  });
  const reported = new Map<string, ImpossibleCheck>();
  for (const view of views) {
    for (const check of view.impossible) {
      if (!reported.has(check.text)) reported.set(check.text, check);
    }
  }
  const messages: string[] = [];
  const rank = (text: string) => order.get(text) ?? order.size;
  for (const text of [...reported.keys()].sort((a, b) => rank(a) - rank(b))) {
    let check = reported.get(text)!;
    const group = viewsOf(rootOf(check.base, narrowed));
    if (group.every((view) => view.impossible.some((seen) => seen.text === text))) {
      check = declaredLabel(check, group[0]!.choice, nullableRoots, registry);
      messages.push(impossibleCheckMessage(spell(check)));
    }
  }
  // A read is dead under every combination of contracts when every contract of one root kills it,
  // since what kills it under one root's contract holds whatever the others are.
  const groups = varied.length > 0 ? varied.map((root) => viewsOf(root)) : [views];
  const seenReads = new Map<string, Guard[]>();
  refs.forEach(([ref, guard], index) => {
    const root = rootOf(ref, used);
    if (root === null || ref === root || unwalkedRoots.has(root)) return;
    const seenGuards = seenReads.get(ref) ?? [];
    if (seenGuards.some((seen) => sameGuard(seen, guard))) return;
    seenReads.set(ref, [...seenGuards, guard]);
    if (unresolvedType(guard, registry)) {
      // The type-table pass reports the unknown name.
      return;
    }
    if (views.length > 0 && groups.some((group) => group.every((view) => view.checked[index]![1] === null))) {
      // Reached by no value under any choice: a reported check explains it.
      return;
    }
    const message = readError(
      ref,
      root,
      index,
      viewsOf(root).filter((view) => view.checked[index]![1] !== null),
      narrowed.includes(root) ? null : contracts.get(root)!,
      registry,
      site,
      { partialUnionFields: partialUnionRoots.has(root), nullable: nullableRoots.has(root) },
    );
    if (message !== null) messages.push(message);
  });
  return [...new Set(messages)];
}

/**
 * Return `check` with its declared base type labelled as declared, for a base below a root that is
 * nullable only because its producer may be skipped.
 */
function declaredLabel(
  check: ImpossibleCheck,
  choice: ReadonlyMap<string, OutputContract>,
  nullableRoots: ReadonlySet<string>,
  registry: TypeRegistry,
): ImpossibleCheck {
  if (!check.declared || (check.reason !== ImpossibleReason.NO_OVERLAP && check.reason !== ImpossibleReason.ALWAYS)) {
    return check;
  }
  if (rootOf(check.base, nullableRoots) === null) return check;
  const declared = rootBaseType(choice)(check.base);
  if (!(declared instanceof Type)) return check;
  return replaceCheck(check, { baseLabel: namedTypeLabel(declared, registry) });
}

/**
 * Return why the `index`-th read `ref` below `root` can't work, else `null`.
 *
 * The read is an error when every view in `live` (those trying `root`'s contracts where it runs)
 * rejects it: on each of the `unnarrowed` contracts when no check narrows `root`, else on the
 * view's narrowed types or chosen contract. A narrowed error is preferred over an unnarrowed one.
 * When `root` is `nullable` only because its producer may be skipped, the error names the declared
 * type, not the nullable one the checks narrowed.
 */
function readError(
  ref: string,
  root: string,
  index: number,
  live: readonly ChoiceView[],
  unnarrowed: readonly OutputContract[] | null,
  registry: TypeRegistry,
  site: ReadSite,
  options: { partialUnionFields: boolean; nullable?: boolean },
): string | null {
  const { partialUnionFields, nullable = false } = options;
  const fields = below(ref, root);
  const unionRoot = site.unionRoots.get(root) ?? null;
  if (unnarrowed !== null) {
    let first: string | null = null;
    for (const contract of unnarrowed) {
      const error = site.error(contract, fields, ref, { partialUnionFields, unionRoot });
      if (error === null) return null;
      first = first ?? error;
    }
    return first;
  }
  const errors: [narrowed: boolean, error: string][] = [];
  for (const { choice, baseTypeOf, checked } of live) {
    const guard = checked[index]![1]!;
    const candidates = guardedReadTypes(ref, guard, baseTypeOf, registry);
    if (candidates.length === 0) {
      const error = site.error(choice.get(root)!, fields, ref, { partialUnionFields, unionRoot });
      if (error === null) return null;
      errors.push([false, error]);
      continue;
    }
    let narrowedError: string | null = null;
    for (const candidate of candidates) {
      const [typ, rest] = candidate;
      const error = site.error(new ValueContract(typ), rest, ref, { partialUnionFields });
      if (error === null) return null;
      if (narrowedError !== null) continue;
      if (nullable) {
        narrowedError = declaredReadError(
          ref,
          candidate,
          error,
          choice.get(root)!,
          fields,
          rootBaseType(choice)(candidate[2]),
          registry,
          site,
          { partialUnionFields, unionRoot },
        );
      } else {
        narrowedError = narrowedReadError(ref, candidate, error, registry);
      }
    }
    errors.push([true, narrowedError!]);
  }
  if (errors.length === 0) return null;
  return (errors.find(([narrowed]) => narrowed) ?? errors[0]!)[1];
}

/**
 * Return a narrowed read's error below a root that is nullable only because its producer may be
 * skipped, worded on the `declared` type of the candidate's base.
 *
 * The narrowed type loses that null, and when the checks leave only null the read (`fields` below
 * the root) is walked on the root's chosen `contract` instead.
 */
function declaredReadError(
  ref: string,
  candidate: NarrowedRead,
  error: string,
  contract: OutputContract,
  fields: readonly string[],
  declared: Type | null | NotNarrowable,
  registry: TypeRegistry,
  site: ReadSite,
  options: { partialUnionFields: boolean; unionRoot: string | null },
): string {
  const { partialUnionFields, unionRoot } = options;
  const [typ, rest, base] = candidate;
  if (!(declared instanceof Type) || declared.nullable) return narrowedReadError(ref, candidate, error, registry);
  if (typ.kind === ValueKind.NONE) {
    return site.error(contract, fields, ref, { partialUnionFields, unionRoot }) ?? error;
  }
  const stripped = typ.replace({ nullable: false });
  const strippedError = site.error(new ValueContract(stripped), rest, ref, { partialUnionFields }) ?? error;
  return narrowedReadError(ref, [stripped, rest, base], strippedError, registry);
}

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
 * @param options.registry The flow's typedefs (default none).
 * @param options.diagnostics Factory for located load errors (default an unlocated one).
 * @throws LoadError holding every error, one per line, located at the first available exact
 *   assertion, offending field, or node range.
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
    registry?: TypeRegistry | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const assertionSiteAt = options.assertionSiteAt ?? null;
  const typeNames = options.typeNames ?? new Set<string>();
  const localContracts = options.localContracts ?? new Map<string, ReadonlyMap<string, LocalInput>>();
  const partialUnionNodes = options.partialUnionNodes ?? new Set<string>();
  const registry = options.registry ?? new Map();
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const errors: [message: string, range: SourceRange | null][] = [];
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
          if (contract !== null && !isExactly(contract, ValueContract) && !isExactly(contract, PartialRecordContract)) {
            throw new TypeError(
              `node ${reprText(nid)} output_contract must be an exact ValueContract, ` +
                `PartialRecordContract, or null, got ${jsTypeName(contract)}`,
            );
          }
          if (contract !== null) roots.set("output", [contract]);
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
      const messages = conditionErrors(tree, roots, registry, {
        partialUnionRoots: partialUnionNodes.has(nid)
          ? new Set([...roots.keys()].filter((root) => root !== "output"))
          : new Set(),
        unwalkedRoots: unwalked,
        site: new ReadSite({ spawner: SPAWNER_KINDS.has(kind), unionRoots }),
      });
      for (const message of messages) errors.push([`node ${reprText(nid)} assert: ${message}`, errorRange]);
      (readsOutput ? post : pre).push(new Assertion(a, a, { origin }));
    });
    node.preAsserts = pre;
    node.postAsserts = post;
  }
  if (errors.length > 0) {
    const firstRange = errors.find(([, range]) => range !== null)?.[1] ?? null;
    throw diagnostics.error(
      "flow has invalid node asserts:\n  " + errors.map(([message]) => message).join("\n  "),
      { range: firstRange },
    );
  }
}
