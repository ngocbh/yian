/**
 * Child resolution and signature: the load-time callable seam shared by `call`, `map` and `loop`.
 *
 * A `call`/`map`/`loop` node names a callable (a `defs:` entry or an external flow). The callable
 * resolver is the load-time seam, `flowId -> ResolvedChild`; the loader composes the defs-first
 * resolution. From the resolved child the kind builder derives its `ChildSignature`
 * (canonical parameters and the exact-or-partial codomain contract) to stamp a CALL's
 * `outputContract` (a LOOP's, or the returned fields of its merged carried contract) and to check
 * the bindings' names, arity and types. Each kind's own `build` bakes the resolved child's
 * compiled flow onto its node; this module supplies the shared resolution, the call nesting limit
 * it enforces, and the signature derivation. Output-binding contract inference lives below
 * compose, in `compile/output_contracts.ts`.
 *
 * Like `base.ts` this is a leaf of the nodes package: it imports `compile/output_contracts.ts`,
 * `helpers/errors.ts`, the node contracts (`nodes/contracts.ts`), the END node class
 * (`nodes/end/node.ts`), `BUILTIN_TYPE_NAMES` (`typesys/types.ts`), `typesys/values.ts`,
 * `defaults.ts`, `diagnostics.ts` and types from the compile layer, but no other kind module and
 * nothing that imports it, so `call`, `loop`, `base.ts` and the loader import it without cycling.
 */

import type { CompiledFlow } from "../../compile/model.ts";
import { current } from "../../defaults.ts";
import { reprText, type SourceRange } from "../../diagnostics.ts";
import {
  isExactly,
  type OutputContract,
  type Parameter,
  PartialRecordContract,
  ValueContract,
} from "../../nodes/contracts.ts";
import { EndNode } from "../../nodes/end/node.ts";
import { YianTypeError } from "../../typesys/errors.ts";
import { BUILTIN_TYPE_NAMES } from "../../typesys/types.ts";
import { jsTypeName, type Type } from "../../typesys/values.ts";
import { LoadDiagnostics, LoadError } from "../helpers/errors.ts";
import { combineOutputContracts, resolveOutputBindingContract } from "../../compile/output_contracts.ts";

/** Whether `contract` is an exact `ValueContract` or `PartialRecordContract`. */
function isOutputContract(contract: unknown): contract is OutputContract {
  return isExactly(contract, ValueContract) || isExactly(contract, PartialRecordContract);
}

/**
 * A child flow's compile-time signature; frozen.
 *
 * `inputs` are the child's canonical `Parameter`s (the name and arity check reads their names
 * and required/default policy). `output` is the child's single codomain: a record child's is an
 * exact or partial closed record keyed by every declared name (even one or none); a value child's
 * re-exports its one declared output's exact or partial contract; `null` means no output or no
 * statically known shape. `outputNames` are the declared terminal names, in order.
 */
export interface ChildSignature {
  /** The child's canonical parameters. */
  readonly inputs: readonly Parameter[];
  /** The child's codomain contract; `null` when it has no output or no known shape. */
  readonly output: OutputContract | null;
  /** The declared terminal output names, in order. */
  readonly outputNames: readonly string[];
}

/** A resolved child flow; structural, so this module need not import the pipeline. */
export interface ResolvedChild {
  /** The child's compiled flow. */
  readonly compiled: CompiledFlow;
  /** The child's canonical parameters (its START params). */
  readonly input: readonly Parameter[];
}

/**
 * Resolves a callable by id for call, map and loop: a `defs:` entry, a compiler-generated body,
 * or a `uses:` alias already loaded with its own ref's version.
 */
export type CallableResolver = (flowId: string) => ResolvedChild;

/** The shape `callNesting` walks: a flow's nodes and each node's static child flows. */
interface NestingFlow {
  readonly nodes: ReadonlyMap<string, { originChildren(): readonly unknown[] }>;
}

/**
 * Derive a resolved child's compile-time `ChildSignature`.
 *
 * The codomain contract follows the child's END: a record END gives an exact or partial closed
 * record keyed by every declared name; a value END re-exports its one output's exact or partial
 * contract, or `null` without one. A `T(...)` constructor in an output is typed through END's
 * type table, which holds the child's own typedefs the outputs use.
 *
 * @param loaded The resolved child, read through `compiled` and `input`.
 * @returns The child's parameters, codomain contract and declared output names.
 * @throws TypeError when the child's END isn't an `EndNode`, or a producer's contract isn't an
 *   exact `ValueContract`, `PartialRecordContract` or `null`.
 */
export function childSignature(loaded: ResolvedChild): ChildSignature {
  const end = loaded.compiled.nodes.get(EndNode.ID);
  if (!(end instanceof EndNode)) {
    throw new YianTypeError(`child END must be an EndNode, got ${end === undefined ? "null" : jsTypeName(end)}`);
  }
  const producers = new Map<string, OutputContract>();
  for (const [nodeId, node] of loaded.compiled.nodes) {
    const contract = node.outputContract;
    if (contract === null) continue;
    if (!isOutputContract(contract)) {
      throw new YianTypeError(
        "child output producer contracts must be ValueContract, PartialRecordContract, or null values, " +
          `got ${jsTypeName(contract)} for ${reprText(nodeId)}`,
      );
    }
    producers.set(nodeId, contract);
  }
  const flowInputs = new Map<string, ValueContract>();
  for (const parameter of loaded.input) {
    if (parameter.contract !== null) flowInputs.set(parameter.name, parameter.contract);
  }
  // `isinstance` may also put a builtin scalar name in the table; no constructor uses one.
  const constructorTypes = new Map<string, Type>();
  for (const [name, typ] of end.types) if (!BUILTIN_TYPE_NAMES.has(name)) constructorTypes.set(name, typ);
  const typeFn = (name: string): Type | null => constructorTypes.get(name) ?? null;
  const declaredOutputs = loaded.compiled.outputs.map(
    (declared) =>
      [declared.name, resolveOutputBindingContract(declared.binding, producers, flowInputs, { typeFn })] as const,
  );
  let output: OutputContract | null;
  if (end.returnsRecord) output = combineOutputContracts(new Map(declaredOutputs));
  else output = declaredOutputs.length > 0 ? declaredOutputs[0]![1] : null;
  return Object.freeze({
    inputs: Object.freeze([...loaded.input]),
    output,
    outputNames: Object.freeze(declaredOutputs.map(([name]) => name)),
  });
}

/** Return how messages name a callable: `childLabel` when given, else `callable '<flowId>'`. */
export function callableDisplay(flowId: string, childLabel: string | null = null): string {
  return childLabel || `callable ${reprText(flowId)}`;
}

// Calls being resolved around the current one in the file being loaded. A module-level number
// is enough because assembly is synchronous.
let activeCalls = 0;

/**
 * Run `fn` with the active call count starting from zero while one flow file loads, restoring
 * the outer count afterwards, even when `fn` throws.
 *
 * Each file counts its own calls; a call into another file is checked against the built
 * callee's depth instead. `fn` must be synchronous: an `await` inside it would let concurrent
 * loads share the count.
 *
 * @param fn The file's load.
 * @returns What `fn` returns.
 */
export function callNestingScope<T>(fn: () => T): T {
  const saved = activeCalls;
  activeCalls = 0;
  try {
    return fn();
  } finally {
    activeCalls = saved;
  }
}

/**
 * Return the longest chain of static child flows under a compiled flow.
 *
 * Kind-blind: child flows come from each node's `originChildren()`. The walk is iterative and
 * visits each flow object once, so a deep or shared child neither overflows the stack nor costs
 * one visit per path.
 *
 * @param flow The flow to measure; read through `nodes` and each node's `originChildren()`.
 * @returns The number of nested bodies on the longest chain; 0 for a flow with no child flow.
 */
export function callNesting(flow: NestingFlow): number {
  // Keyed by identity: two equal-looking flows are still two bodies.
  const depths = new Map<object, number>();
  const children = new Map<object, NestingFlow[]>();
  const stack: NestingFlow[] = [flow];
  while (stack.length > 0) {
    const top = stack[stack.length - 1]!;
    if (depths.has(top)) {
      stack.pop();
      continue;
    }
    let kids = children.get(top);
    if (kids === undefined) {
      kids = [];
      for (const node of top.nodes.values()) {
        for (const child of node.originChildren()) kids.push(child as NestingFlow);
      }
      children.set(top, kids);
      for (const kid of kids) if (!depths.has(kid)) stack.push(kid);
      continue;
    }
    stack.pop();
    let depth = 0;
    for (const kid of kids) depth = Math.max(depth, 1 + depths.get(kid)!);
    depths.set(top, depth);
  }
  return depths.get(flow)!;
}

/**
 * Resolve a callable to its child flow through the resolver, loud on failure.
 *
 * A `LoadError` from the resolver passes through unchanged; any other error is an engine bug and
 * propagates as is.
 *
 * @param nodeId The calling node's id, named in messages.
 * @param flowId The callable the node names.
 * @param resolver The load-time callable resolver; `null` when none was given, which fails as an
 *   unresolvable callable.
 * @param childLabel Author-facing name of the callable; `null` names it by `flowId`.
 * @param options.diagnostics The context errors are raised in; a fresh one when not given.
 * @param options.range Location of the call in the owning document.
 * @param options.kind The calling node's kind (`call`, `map` or `loop`), named in messages;
 *   `call` when not given.
 * @returns The resolved child flow.
 * @throws {LoadError} When the callable can't be resolved, or the call nests bodies deeper than
 *   `current().maxCallNesting` levels.
 */
export function resolveChild(
  nodeId: string,
  flowId: string,
  resolver: CallableResolver | null,
  childLabel: string | null = null,
  options: { diagnostics?: LoadDiagnostics | null; range?: SourceRange | null; kind?: string } = {},
): ResolvedChild {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const range = options.range ?? null;
  const subject = `${options.kind ?? "call"} node ${reprText(nodeId)}`;
  const display = callableDisplay(flowId, childLabel);
  const limit = current().maxCallNesting;

  const tooDeep = (levels: string): LoadError =>
    diagnostics.error(
      `${subject}: calling ${display} nests calls ${levels} deep, past the ` +
        `call nesting limit of ${limit} (each call, map, or loop body is a ` +
        "level); flatten the chain",
      { range },
    );

  const depth = activeCalls;
  // Checked before resolving: building the callee would recurse another level.
  if (depth >= limit) throw tooDeep(`more than ${limit} levels`);
  if (resolver === null) {
    throw diagnostics.error(`${subject}: could not resolve ${display}: no child resolver was given`, { range });
  }
  activeCalls = depth + 1;
  let child: ResolvedChild;
  try {
    child = resolver(flowId);
  } finally {
    activeCalls = depth;
  }
  // A callee built earlier (a cached def, an imported file) never ran on this count.
  const levels = depth + 1 + callNesting(child.compiled);
  if (levels > limit) throw tooDeep(`${levels} levels`);
  return child;
}
