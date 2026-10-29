/**
 * Graph stage: reject a MAP `over:` source whose type can never fit.
 *
 * A MAP node maps over the list its `over:` source resolves to; the node rejects any other value
 * at run time. A wrong literal is already rejected where the node is parsed, with the authored text
 * at hand. This load pass rejects a reference or expression whose inferred type proves the run-time
 * check would fail: `over:` must infer a list kind (or `Any`), nullability aside. The test is on the
 * kind rather than the type relation, so any list loads whatever its element type.
 *
 * A literal source is left to the parse stage, and an unproven (`null`) inferred type loads and is
 * left to the run-time check. A tag on this key is never consumed, so the source never carries a
 * declared contract and is always inferred from its text.
 *
 * It runs after the type tables are filled, so an `isinstance` call in `over:` keeps its more
 * specific error, and before the cycle check, so a wrong-typed source in a cyclic flow reports the
 * type error first.
 *
 * Knows about: `compose/helpers` (literal detection, tag labels, load diagnostics), build results,
 * the narrowing view, typed-binding inference, the MAP node, and `typesys`.
 * Never imports: the loader or the compose package facade; they import this back.
 */

import { reprText } from "../../diagnostics.ts";
import { asBinding, bindingSource, type OutputContract, type ValueContract } from "../../nodes/contracts.ts";
import { MapNode } from "../../nodes/map/node.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isListKind, ValueKind } from "../../typesys/values.ts";
import { LoadDiagnostics, locatedError } from "../helpers/errors.ts";
import { literalValue, NOT_LITERAL, type TypeFn } from "../../compile/output_contracts.ts";
import { namedTypeLabel } from "../helpers/tags.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import type { Narrowing } from "./narrowing.ts";
import { inferBindingType, type ResolveOrigin } from "./typed_bindings.ts";

/**
 * Reject a MAP `over:` source whose inferred type can never fit.
 *
 * @param buildResults Built nodes with their transient value ranges, the fallback location.
 * @param flowWiring Final binding maps keyed by node id and wiring key; a MAP node's `over` holds
 *   the source.
 * @param producers Exact or partial node output contracts keyed by producer id.
 * @param flowInputs Exact flow input contracts keyed by input name.
 * @param options.narrowing The union narrowing each node reads against, from the CASE branches
 *   that route it; `null` (the default) reads every source against the unchanged maps.
 * @param options.typeFn Resolves a constructor name to its type for inference.
 * @param options.registry The flow's typedefs, so a message names a declared record by its
 *   typedef.
 * @param options.resolveOrigin Resolves a binding's registered origin to its source frame, the
 *   preferred location.
 * @param options.diagnostics Factory for located load errors.
 * @throws LoadError when a MAP `over:` source infers a type that is not a list.
 */
export function checkReservedSources(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: {
    narrowing?: Narrowing | null;
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: ResolveOrigin | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const { narrowing = null, typeFn = null, registry = null, resolveOrigin = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  for (const [nodeId, result] of buildResults) {
    if (!(result.node instanceof MapNode)) continue;
    const source = flowWiring.get(nodeId)?.get("over");
    // Absent, or a literal the parse stage already checked.
    if (source === undefined || literalValue(bindingSource(source)) !== NOT_LITERAL) continue;
    const [nodeProducers, nodeInputs] = narrowing !== null ? narrowing.forNode(nodeId) : [producers, flowInputs];
    const inferred = inferBindingType(source, nodeProducers, nodeInputs, { typeFn });
    if (inferred === null) continue; // unproven: the run-time check covers it
    if (isListKind(inferred.kind) || inferred.kind === ValueKind.ANY) continue;
    const label = namedTypeLabel(inferred, registry ?? new Map());
    throw locatedError(
      `node ${reprText(nodeId)} (kind=map): \`over:\` must be a list, but the value is ${label}; ` +
        "write a reference to a list",
      asBinding(source),
      result.fieldRanges.get(siteKey("field", "over")) ?? result.nodeRange,
      resolveOrigin,
      diagnostics,
    );
  }
}
