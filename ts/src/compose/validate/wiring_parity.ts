/**
 * Validate: load-time `flow.wiring` / `params` key parity for every node.
 *
 * A loader post-build pass. Every node's `flow.wiring` keys must equal its declared `params`
 * names plus the kind's reserved wiring keys (`over` for a mapped `call`). Its
 * range anchor comes from the node's transient Compose `BuildResult.fieldRanges` entry for the
 * `inputs` field (the coarse `input:` mapping span), else the authored node range.
 *
 * Knows about: compose nodes (`BuildResult`, `siteKey`), load diagnostics, and value text.
 * Never imports: build, the loader, or the runtime.
 */

import { compareCodeUnits, reprText } from "../../diagnostics.ts";
import { writeAsText } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";

/**
 * Check that every node's `flow.wiring` keys equal its declared `params` names plus the kind's
 * reserved keys (`over` for a mapped `call`).
 *
 * Closes the duplicate-param hole by construction (a nested-map wiring cannot hold a duplicate
 * param) and turns an orphan source (a wiring key with no param), a missing source (a param with
 * no wiring), or a node absent from `flow.wiring` entirely (a silent empty bind, so a lost input)
 * into a located `LoadError`. Engine bind-time reserved keys are never author wiring and do not
 * appear here.
 *
 * @param options.diagnostics Builds the error (default: an unlocated one).
 * @throws LoadError at the node's `input:` mapping range, else its node range, on the first node
 *   whose keys differ.
 */
export function checkWiringParity(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): void {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    const expected = new Set([...(node.params ?? []).map((p) => p.name), ...node.reservedWiringKeys()]);
    const actual = new Set(flowWiring.get(nodeId)?.keys() ?? []);
    if (actual.size !== expected.size || [...actual].some((key) => !expected.has(key))) {
      const sortedActual = [...actual].sort(compareCodeUnits);
      const sortedExpected = [...expected].sort(compareCodeUnits);
      throw diagnostics.error(
        `node ${reprText(nodeId)}: flow.wiring keys ${writeAsText(sortedActual)} != declared params + reserved ` +
          `${writeAsText(sortedExpected)} (wiring/params key parity)`,
        { range: result.fieldRanges.get(siteKey("field", "inputs")) ?? result.nodeRange },
      );
    }
  }
}
