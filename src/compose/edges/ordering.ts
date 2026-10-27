/**
 * Graph stage: `depends_on:` / `runs_after:` to run-ordering edges.
 *
 * A cross-node assembly pass. Each node may declare ordering-only dependencies on other nodes:
 * `depends_on:` (co-skip: if the source is skipped the dependent dies) and `runs_after:` (pure
 * order: the dependent waits for the source to settle but survives its skip). Both lower to
 * zero-data `Edge`s with `ordering: true`; the `optional` flag is the only difference
 * (`depends_on` false, `runs_after` true). An unknown target is a `LoadError` located at the
 * declaring node descriptor's `range`. Cycle rejection is the caller's job: these edges join the
 * `rejectCycles` set.
 *
 * Knows about: the compile model (`Edge`), descriptors, and load diagnostics.
 * Never imports: validate, build, the loader, or the runtime.
 */

import { Edge } from "../../compile/model.ts";
import { reprText } from "../../diagnostics.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import type { NodeDescriptor } from "../nodes/base.ts";

/**
 * Return the run-ordering edges from `depends_on:` (co-skip) and `runs_after:` (pure order).
 *
 * An ordering edge carries no data (`ordering: true`); it gates the dependent on the source
 * settling. `depends_on` co-skips (`optional: false`); `runs_after` does not (`optional: true`).
 * Ids are `<source>~><node>#<i>`, distinct from the `->` data edges on the same pair.
 *
 * @param descriptors The flow level's descriptors by node id.
 * @param nodeIds The ids an ordering source may name.
 * @param options.diagnostics Locates the unknown-target error (default: an empty one).
 * @returns The ordering edges, in descriptor order, `depends_on` before `runs_after`.
 * @throws LoadError at the declaring node's `range` for a source not in `nodeIds`.
 */
export function inferOrderingEdges(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  nodeIds: ReadonlySet<string>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Edge[] {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const edges: Edge[] = [];
  const counts = new Map<string, number>();
  for (const [nodeId, desc] of descriptors) {
    for (const [key, sources, optional] of [
      ["depends_on", desc.dependsOn, false],
      ["runs_after", desc.runsAfter, true],
    ] as const) {
      for (const src of sources) {
        if (!nodeIds.has(src)) {
          throw diagnostics.error(`node ${reprText(nodeId)}: \`${key}\` references unknown node ${reprText(src)}`, {
            range: desc.range,
          });
        }
        const pair = `${src}~>${nodeId}`;
        const i = counts.get(pair) ?? 0;
        counts.set(pair, i + 1);
        edges.push(new Edge(`${pair}#${i}`, src, nodeId, { ordering: true, optional }));
      }
    }
  }
  return edges;
}
