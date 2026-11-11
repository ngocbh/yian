/**
 * Validate: load-time `flow.wiring` / `params` key parity for every node.
 *
 * A loader post-build pass. Every node's `flow.wiring` keys must equal its declared `params`
 * names plus the kind's reserved wiring keys (`over` for a mapped `call`). The node builders keep
 * the two in step, so a mismatch is a defect in yian, never in the flow.
 *
 * Knows about: compose nodes (`BuildResult`) and value text.
 */

import { compareCodeUnits, reprText } from "../../diagnostics.ts";
import { writeAsText } from "../../typesys/values.ts";
import type { BuildResult } from "../nodes/base.ts";

/**
 * Check that every node's `flow.wiring` keys equal its declared `params` names plus the kind's
 * reserved keys (`over` for a mapped `call`).
 *
 * An orphan source (a wiring key with no param), a missing source (a param with no wiring), or a
 * node absent from `flow.wiring` entirely (a silent empty bind, so a lost input) is an internal
 * error. Engine bind-time reserved keys are never author wiring and do not appear here.
 *
 * @throws TypeError on the first node whose keys differ.
 */
export function checkWiringParity(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
): void {
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    const expected = new Set([...(node.params ?? []).map((p) => p.name), ...node.reservedWiringKeys()]);
    const actual = new Set(flowWiring.get(nodeId)?.keys() ?? []);
    if (actual.size !== expected.size || [...actual].some((key) => !expected.has(key))) {
      const sortedActual = [...actual].sort(compareCodeUnits);
      const sortedExpected = [...expected].sort(compareCodeUnits);
      throw new TypeError(
        `yian bug: node ${reprText(nodeId)} has wiring keys ${writeAsText(sortedActual)}, but its params and ` +
          `reserved keys are ${writeAsText(sortedExpected)}`,
      );
    }
  }
}
