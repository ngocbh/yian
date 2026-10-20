/**
 * Graph stage: reject a MAP `over:` or WAIT `until:` source whose type can never fit.
 *
 * A MAP node maps over the list its `over:` source resolves to, and a WAIT node waits until the
 * date or datetime its `until:` source resolves to; the node rejects any other value at run time. A
 * wrong literal is already rejected where the node is parsed, with the authored text at hand. This
 * load pass rejects a reference or expression whose inferred type proves the run-time check would
 * fail:
 *
 * - `over:` must infer a list kind (or `Any`), nullability aside. The test is on the kind rather
 *   than the type relation, so any list loads whatever its element type;
 * - `until:` must infer a type that overlaps `date` or `datetime`. A `str` overlaps both (the node
 *   parses text at run time), so a text reference loads.
 *
 * A literal source is left to the parse stage, and an unproven (`null`) inferred type loads and is
 * left to the run-time check. A tag on these keys is never consumed, so the source never carries a
 * declared contract and is always inferred from its text.
 *
 * It runs after the type tables are filled, so an `isinstance` call in `over:` or `until:` keeps
 * its more specific error, and before the cycle check, so a wrong-typed source in a cyclic flow
 * reports the type error first.
 *
 * Knows about: `compile/validation` (the type relation), `compose/helpers` (literal detection, tag
 * labels, load diagnostics), build results, the narrowing view, typed-binding inference, the MAP
 * and WAIT nodes, and `typesys`.
 * Never imports: the loader or the compose package facade; they import this back.
 */

import { typeRelation } from "../../compile/validation.ts";
import { reprText, type SourceRange } from "../../diagnostics.ts";
import { Binding, bindingSource, type OutputContract, type ValueContract } from "../../nodes/contracts.ts";
import { MapNode } from "../../nodes/map/node.ts";
import { WaitNode } from "../../nodes/wait/node.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isListKind, Type, ValueKind } from "../../typesys/values.ts";
import { LoadDiagnostics, type LoadError } from "../helpers/errors.ts";
import { literalValue, NOT_LITERAL, type TypeFn } from "../helpers/output_contracts.ts";
import { namedTypeLabel } from "../helpers/tags.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import type { Narrowing } from "./narrowing.ts";
import { inferBindingType, type ResolveOrigin } from "./typed_bindings.ts";

const DATE = Type.scalar(ValueKind.DATE);
const DATETIME = Type.scalar(ValueKind.DATETIME);

/**
 * Reject a MAP `over:` or WAIT `until:` source whose inferred type can never fit.
 *
 * @param buildResults Built nodes with their transient value ranges, the fallback location.
 * @param flowWiring Final binding maps keyed by node id and wiring key; a MAP node's `over` and a
 *   WAIT node's `until` hold the source.
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
 * @throws LoadError when a MAP `over:` source infers a type that is not a list, or a WAIT `until:`
 *   source infers a type disjoint from both `date` and `datetime`.
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
    const node = result.node;
    let key: string;
    let kind: string;
    if (node instanceof MapNode) [key, kind] = ["over", "map"];
    else if (node instanceof WaitNode) [key, kind] = ["until", "wait"];
    else continue;
    const source = flowWiring.get(nodeId)?.get(key);
    // Absent, or a literal the parse stage already checked.
    if (source === undefined || literalValue(bindingSource(source)) !== NOT_LITERAL) continue;
    const [nodeProducers, nodeInputs] = narrowing !== null ? narrowing.forNode(nodeId) : [producers, flowInputs];
    const inferred = inferBindingType(source, nodeProducers, nodeInputs, { typeFn });
    if (inferred === null) continue; // unproven: the run-time check covers it
    const label = namedTypeLabel(inferred, registry ?? new Map());
    let message: string;
    if (key === "over") {
      if (isListKind(inferred.kind) || inferred.kind === ValueKind.ANY) continue;
      message = `\`over:\` must be a list, but the value is ${label}; write a reference to a list`;
    } else {
      if (typeRelation(inferred, DATE) !== "disjoint" || typeRelation(inferred, DATETIME) !== "disjoint") continue;
      message =
        `\`until:\` must be a date or datetime, but the value is ${label}; ` +
        "write a reference to a date, datetime, or text";
    }
    throw locatedError(
      `node ${reprText(nodeId)} (kind=${kind}): ${message}`,
      source,
      result.fieldRanges.get(siteKey("field", key)) ?? result.nodeRange,
      resolveOrigin,
      diagnostics,
    );
  }
}

/**
 * Build the load error, located at the source's registered origin when it resolves.
 *
 * @param message The error message.
 * @param source The rejected source, a `Binding` when it carries an origin.
 * @param fallback The location used when no origin resolves.
 * @param resolveOrigin Resolves a registered origin to its source frame.
 * @param diagnostics Factory for located load errors.
 * @returns The located error.
 */
function locatedError(
  message: string,
  source: unknown,
  fallback: SourceRange | null,
  resolveOrigin: ResolveOrigin | null,
  diagnostics: LoadDiagnostics,
): LoadError {
  const origin = source instanceof Binding ? source.origin : null;
  const frame = resolveOrigin !== null && origin !== null ? resolveOrigin(origin) : null;
  if (frame !== null) return diagnostics.error(message, { frames: [frame] });
  return diagnostics.error(message, { range: fallback });
}
