/**
 * Graph-stage: `case` then:/else: inline-call branch targets to synth call nodes.
 *
 * A `case` branch may route to a FRESH owned call instead of a placed node id:
 * `then: call(take, stance="pro")` (and `else:` symmetric). It desugars to a synth `__call_<n>`
 * node (the same machinery as an inline-binding call); the then:/else: target is rewritten to that
 * synth id (which is then both the CaseNode handle and the control-edge target, exactly like a
 * plain `then: id`). Accepted grammar: a whole-value `call(...)` directive — a route target
 * resolves to exactly one node id, so any other text starting with `call(` is a located
 * `LoadError` naming the problem. A branch that holds a value (quoted text, a `${...}` template, a
 * non-string value) was already rejected by `checkCaseBranches`, so this pass sees node ids and
 * unquoted calls. The outer synth call is the route target, so the case-route veto skip-floods it
 * on a branch not taken. A nested directive's inner calls are lifted too; each records the outer
 * call as its `host`, so the host-edge step gives it the outer call's route and gates it on the
 * outer call's other arguments; it is skipped with it.
 *
 * Runs in `assemble` AFTER `desugarInlineCalls` (sharing its `nextId` minter — else the ids
 * collide) and BEFORE `expandCaseOutputs`, so the synth call's args (which may read
 * `${<case>.output}`) are expanded by that pass and `${<case>.output}` over THIS case sees the
 * rewritten (synth-id) targets.
 *
 * Located parse metadata: a synth branch-target call node locates at the `then:`/`else:` it was
 * lifted from — the case descriptor's positional `thenRanges[N]` for a `then:`, the
 * `siteKey("field", "else")` range for the `else:` — with an empty `fieldRanges`, falling back to
 * the case node's own range when a span is unavailable.
 *
 * Knows about: diagnostics, `expr` (the expression error), the compose helpers (errors, positional
 * metadata), the CASE descriptor and the inline-call desugar (the shared `call(...)` recognizer
 * and synth-node factory).
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import type { InlineCall } from "../helpers/directives.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { nth } from "../helpers/source.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { CaseDescriptor } from "../nodes/case.ts";
import { desugarCallDirectives, toCallDescriptor } from "./inline_calls.ts";

/**
 * Rewrite each `case` then:/else: that is an inline call to a synth call node id.
 *
 * An inline call is a whole-value `call(<flow-id>, kw=...)` directive; its `CallDescriptor`
 * (`over = null`) is minted and the descriptor map returned, augmented with the synth nodes. A
 * plain node id (no `${`, not a `call(...)` directive) passes through unchanged.
 *
 * Each synth branch-target node locates at the `then:`/`else:` it was lifted from (its own source
 * line), falling back to the case node's range when that span is unavailable.
 *
 * @param descriptors Node descriptors keyed by node id.
 * @param mint The synth-id minter `desugarInlineCalls` used.
 * @param options.diagnostics Factory for located load errors.
 * @returns The new descriptor map, the synth nodes after the authored ones.
 * @throws LoadError for a branch target that is neither a node id nor one `call(...)` directive,
 *   or a call the synth-node factory rejects.
 */
export function desugarCaseCallTargets(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  mint: () => string,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Map<string, NodeDescriptor> {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const synth = new Map<string, NodeDescriptor>();
  const newDescriptors = new Map<string, NodeDescriptor>();
  for (const [nodeId, desc] of descriptors) {
    if (!(desc instanceof CaseDescriptor)) {
      newDescriptors.set(nodeId, desc);
      continue;
    }
    const id = reprText(nodeId);
    const newCases = desc.cases.map((spec, index) =>
      Object.freeze({
        ...spec,
        then: liftCaseTarget(spec["then"] ?? null, mint, synth, {
          host: `case node ${id} then:`,
          site: `node ${id} cases[${index}].then`,
          callSite: ["then", nodeId, index],
          range: nth(desc.thenRanges, index) ?? desc.range,
          diagnostics,
        }),
      }),
    );
    const newElse =
      desc.else_ !== null
        ? liftCaseTarget(desc.else_, mint, synth, {
            host: `case node ${id} else:`,
            site: `node ${id} else`,
            callSite: ["else", nodeId],
            range: desc.fieldRanges.get(siteKey("field", "else")) ?? desc.range,
            diagnostics,
          })
        : null;
    newDescriptors.set(nodeId, desc.replace({ cases: newCases, else_: newElse }));
  }
  for (const [nodeId, desc] of synth) newDescriptors.set(nodeId, desc);
  return newDescriptors;
}

/**
 * Return a then:/else: target rewritten to a single node id.
 *
 * Accepts TWO forms and rejects everything else with a located `LoadError`:
 * - a whole-value `call(<flow-id>, kw=...)` directive (the compile-time form): its synth call
 *   node(s) are lifted into `synth` and the OUTER call's synth id returned;
 * - a plain node id (no `${`, not a `call(...)` directive): passed through unchanged.
 *
 * Anything else — text starting with `call(` that is not exactly one directive, or any `${...}`
 * span — is a `LoadError`: a route target must resolve to exactly one node id.
 *
 * `options.site` (`node 'g' cases[0].then`, `node 'g' else`) and `options.callSite` (its address)
 * are passed to every lifted call, nested ones included, so their errors name the branch they were
 * written in.
 *
 * @throws LoadError for a malformed directive, a `${...}` target, or a call `toCallDescriptor`
 *   rejects.
 */
function liftCaseTarget(
  target: unknown,
  mint: () => string,
  synth: Map<string, NodeDescriptor>,
  options: {
    host: string;
    site: string;
    callSite: readonly (string | number | null)[];
    range: SourceRange | null;
    diagnostics: LoadDiagnostics;
  },
): unknown {
  const { host, site, callSite, range, diagnostics } = options;
  if (typeof target !== "string") return target; // null (handled by the caller) or a native literal
  // A whole-value `call(...)` directive: recognized BEFORE the plain-id passthrough, since a
  // directive carries no `${` and would otherwise look like a bare node id. The loader's
  // `checkCaseBranches` rejects a malformed branch call first; this keeps a direct caller's error
  // located too.
  let calls: InlineCall[];
  try {
    [, calls] = desugarCallDirectives(target, mint);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    throw diagnostics.error(`${host}: ${error.message}`, { range, cause: error });
  }
  // With calls, the new value is always the outer call's `${<id>.output}` (a non-directive
  // raises above), so the outer id is the branch target with no further check.
  if (calls.length > 0) {
    // Inner-first: the OUTER call is the last one and names the branch target.
    const outer = calls.at(-1)!;
    for (const call of calls) {
      // The outer call is the route target and gets the route edge; each inner call only feeds
      // it, so the outer call is its host and passes that route on.
      synth.set(
        call.id,
        toCallDescriptor(call, {
          host,
          site,
          callSite,
          hostId: call === outer ? null : outer.id,
          range,
          diagnostics,
        }),
      );
    }
    return outer.id;
  }
  // Not a directive and no span: a plain node id. Text starting with `call(` never gets here; the
  // recognizer above lifts it or raises.
  if (!target.includes("${")) return target;
  // A `${...}` branch target is the retired inline `${flow(args)}` form: a route target is a node
  // id or a whole-value call(...) directive (handled above) — nothing else.
  throw diagnostics.error(
    `${host}: a branch target must be a node id or a call(...) directive ` +
      `(got ${reprText(target)}) — an inline \${flow(args)} target is no longer supported`,
    { range },
  );
}
