/**
 * Graph stage: stamp each TOOL node with the inferred types of its input bindings.
 *
 * A TOOL node hands its bound input record to a host tool whose declared input types are known
 * only when the run supplies its tools. This load pass records, on the node, the type each input
 * binding is proven to produce, so that the run-start tool check can compare them with the tool's
 * declared input types without the authoring source.
 *
 * Each binding is inferred over the producer and flow-input contracts as narrowed by the CASE
 * branches that route the node; an unproven binding is left out. A TOOL has no item scope
 * (`${item}` is legal only in a MAP's own `input:`), so none is used.
 *
 * Knows about: the TOOL node, build results, the narrowing view, and typed-binding inference.
 * Never imports: the loader or the compose package facade; they import this back.
 */

import type { OutputContract, ValueContract } from "../../nodes/contracts.ts";
import { ToolNode } from "../../nodes/tool/node.ts";
import type { Type } from "../../typesys/values.ts";
import type { TypeFn } from "../../compile/output_contracts.ts";
import type { BuildResult } from "../nodes/base.ts";
import type { Narrowing } from "./narrowing.ts";
import { inferBindingType } from "./typed_bindings.ts";

/**
 * Set each TOOL node's `inputTypes` to its bindings' proven inferred types.
 *
 * @param flowWiring Final binding maps keyed by node id and parameter name.
 * @param producers Exact or partial node output contracts keyed by producer id.
 * @param flowInputs Exact flow input contracts keyed by input name.
 * @param buildResults Built nodes keyed by node id.
 * @param options.narrowing The union narrowing each param reads against, from the CASE branches
 *   that route its node; `null` (the default) reads every param against the unchanged maps.
 * @param options.typeFn Resolves a constructor name to its type.
 */
export function stampToolInputTypes(
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  buildResults: ReadonlyMap<string, BuildResult>,
  options: { narrowing?: Narrowing | null; typeFn?: TypeFn | null } = {},
): void {
  const { narrowing = null, typeFn = null } = options;
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    if (!(node instanceof ToolNode)) continue;
    const inputTypes = new Map<string, Type>();
    for (const [name, binding] of flowWiring.get(nodeId) ?? []) {
      const [paramProducers, paramInputs] =
        narrowing !== null ? narrowing.forParam(nodeId, name) : [producers, flowInputs];
      const inferred = inferBindingType(binding, paramProducers, paramInputs, { typeFn });
      if (inferred !== null) inputTypes.set(name, inferred);
    }
    node.inputTypes = inputTypes;
  }
}
