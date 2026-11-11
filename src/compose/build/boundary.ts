/**
 * DAG boundary synthesis: the `START_ID`/`END_ID` nodes and their edges over all edges.
 *
 * Every node kind builds through its own `descriptor.build(ctx)` (see `compose/nodes/<kind>.ts`),
 * and the conductor loop drives them kind-blind. This module owns boundary synthesis,
 * `synthesizeBoundaryGraph`, which is representation-neutral graph plumbing, not per-kind build.
 *
 * It runs after both the data edges and the CASE control edges are reconciled, so a CASE branch
 * target carrying only a `gate -> target` control edge is correctly demoted from root. A root is a
 * node with no incoming edge of any kind (data or control); the engine seeds every root
 * unconditionally, so a wrongly marked root makes both branches of a CASE run. The flow `output:`
 * bindings give the terminals: each `${P.output[...]}` producer gets a `P -> END_ID` edge (a
 * coalesce flow output gets one terminal edge per producer; only the taken branch's value survives
 * and is assembled by END).
 *
 * Knows about: the compile model, `expr` co-skip scanning, the START/END nodes, inbox types, binding
 * input groups, `edges/producers`, `compile/output_contracts` and `validate/asserts`
 * (`LoweredAsserts`; a forward edge only, since `validate/asserts.ts` never imports this module).
 */

import { compareCodeUnits, reprText } from "../../diagnostics.ts";
import { Edge, END_ID, type FlowOutput, START_ID } from "../../compile/model.ts";
import { bindingCoSkips } from "../../expr/index.ts";
import type { Node } from "../../nodes/base.ts";
import { bindingInputGroups, type OutputContract, type Parameter } from "../../nodes/contracts.ts";
import { EndNode } from "../../nodes/end/node.ts";
import { StartNode } from "../../nodes/start/node.ts";
import { isInboxType } from "../../typesys/values.ts";
import { bindingProducers } from "../edges/producers.ts";
import { combineOutputContracts } from "../../compile/output_contracts.ts";
import type { LoweredAsserts } from "../validate/asserts.ts";

/**
 * Synthesize the `START_ID`/`END_ID` boundary nodes, their edges and their wiring.
 *
 * Every flow becomes `START_ID -> body -> END_ID` with ordinary edges (the strings persist only as
 * the reserved node ids). START's `${input.X}`-producer data edges come from the edges phase; the
 * root rule here demotes every input reader from root, since an input read resolves through the
 * `START_ID` namespace.
 *
 * END's `outputContract` comes from the outputs' declared types (`name: !Type value`), so a commit
 * and a replay of END's value both re-check them; see `declaredOutputContract`.
 *
 * `lowered` is the flow's `asserts:` lowered into node-local `LoweredAsserts`. When given, its PRE
 * (boundary) asserts are stamped on START's ordinary `postAsserts` (they read `${output.X}`,
 * START's bound output), its POST asserts on END (`end.postAsserts`), and each synthesized END
 * input becomes an extra `Parameter` on END plus a wiring entry and a producer edge, so a
 * cross-node post assert's `${producer.output}` dependency is structural (a real END input) and
 * survives child cloning. `null` leaves the boundary nodes assert-free (a def or in-memory flow
 * with no flow-level asserts).
 *
 * @param inputParameters The flow's input signature, START's params.
 * @param outputs The flow's outputs.
 * @param nodeIds The body node ids.
 * @param bodyEdges Every reconciled body edge (data, control and ordering).
 * @param lowered The lowered flow asserts, or `null` for none.
 * @param options.record Whether END returns a record keyed by output name; else a value-mode END
 *   (at most one output).
 * @returns The boundary nodes (`START_ID` and `END_ID`); the edges: the END_ID producer edges (one
 *   per producer per declared output, `inputGroup` the output name, or `<name>/<pointer>` per
 *   string nested in a value template, `optional` when the binding doesn't co-skip, so a
 *   coalesce/`:-`/`:?` output never co-skips), then the lowered-assert synth END edges (same
 *   numbering, so ids never collide), then the START_ID root edges (a bare `START_ID->nid` for
 *   every body node with no incoming edge of any kind over the full set, END edges included, plus
 *   a bare `START_ID->END_ID` when END has none, so it runs: a value END with no output commits
 *   `null`, an `output: {}` record END commits `{}`); and the boundary wiring: `START_ID` keyed by
 *   input name (a provisional `${input.X}` source, so wiring parity passes; the engine seeds
 *   START's record directly at runtime; an inbox input is wired to its handle `{inbox: <name>}`,
 *   which a child's call seed replaces), `END_ID` keyed by output name to the output's binding,
 *   merged with the lowered synth wiring.
 * @throws RangeError when `record` is false and there is more than one output.
 */
export function synthesizeBoundaryGraph(
  inputParameters: readonly Parameter[],
  outputs: readonly FlowOutput[],
  nodeIds: ReadonlySet<string>,
  bodyEdges: readonly Edge[],
  lowered: LoweredAsserts | null,
  options: { record: boolean },
): [nodes: Map<string, Node>, edges: Edge[], wiring: Map<string, Map<string, unknown>>] {
  const { record } = options;
  const start = new StartNode(START_ID, { params: inputParameters });
  const end = new EndNode(END_ID, { outputNames: outputs.map((output) => output.name), value: !record });
  end.outputContract = declaredOutputContract(outputs, { record });
  // A single output's param name (`result`) is generated, so its errors say just `flow output`; a
  // record field keeps its authored name.
  end.bindingLabels = new Map(
    outputs.map((output) => [output.name, record ? `flow output ${reprText(output.name)}` : "flow output"]),
  );
  const boundaryNodes = new Map<string, Node>([
    [START_ID, start],
    [END_ID, end],
  ]);

  // END_ID producer edges: one per producer per output, keyed by the output name, with the
  // coalesce/optional stance (the `inferDataEdges` convention).
  const endEdges: Edge[] = [];
  const counts = new Map<string, number>();
  const addEndEdges = (name: string, source: unknown): void => {
    // Each string nested in a value template is its own group (`<name>/<pointer>`).
    for (const [group, text] of bindingInputGroups(name, source)) {
      const optional = !bindingCoSkips(text);
      for (const producer of bindingProducers(text)) {
        const i = counts.get(producer) ?? 0;
        counts.set(producer, i + 1);
        endEdges.push(new Edge(`${producer}->${END_ID}#${i}`, producer, END_ID, { inputGroup: group, optional }));
      }
    }
  };
  for (const output of outputs) addEndEdges(output.name, output.binding);
  const endWiring = new Map<string, unknown>(outputs.map((output) => [output.name, output.binding]));

  // Stamp the lowered flow-level asserts onto the boundary nodes (a node's assert slots are
  // mutable; `validateNodeAsserts` sets them the same way).
  if (lowered !== null) {
    // START fires its boundary asserts through the ordinary `postAsserts` seam: the lowered PRE
    // asserts read `${output.X}`, so the node's `run` wrapper fires them against START's bound
    // (defaulted) record. END fires its POST asserts through the same seam; any
    // `${producer.output}` head is lowered to a synth END param that binds the producer's value
    // into END's record, so the assert fires record-only.
    start.postAsserts = [...lowered.startPre];
    end.postAsserts = [...lowered.endPost];
    // Each synthesized END input is an extra param, a wiring source and a producer edge, so the
    // value the post-assert reads is present when END's `run` wrapper evaluates it. END's
    // `forward` reads only its declared output names, so extra params are inert there.
    end.params = [...(end.params ?? []), ...lowered.synthParams];
    // The same per-producer counter as the output edges, so a producer feeding both an output and
    // a synth input never mints a colliding edge id. An `${input.X}` synth co-skips (a required
    // START->END edge; START always runs); a `${pid.output:-null}` synth doesn't, so END waits for
    // the producer to settle, binding null if it was skipped. Each synth source is also an END
    // wiring entry, so binding finds a source per synth param. A synth name never collides with an
    // output name (it is `__assert_`-prefixed; authors can't write `__` ids).
    for (const [synthName, source] of lowered.synthWiring) {
      endWiring.set(synthName, source);
      addEndEdges(synthName, source);
    }
  }

  // START_ID root edges over the full set (body and END_ID producer edges): a body node with no
  // incoming edge of any kind is a bare root. A 0-output END_ID (no producer edge) is itself one.
  const hasIncoming = new Set([...bodyEdges, ...endEdges].map((edge) => edge.to));
  const rootEdges = [...nodeIds]
    .sort(compareCodeUnits)
    .filter((nodeId) => !hasIncoming.has(nodeId))
    .map((nodeId) => new Edge(`${START_ID}->${nodeId}`, START_ID, nodeId));
  if (!hasIncoming.has(END_ID)) rootEdges.push(new Edge(`${START_ID}->${END_ID}`, START_ID, END_ID));

  const boundaryWiring = new Map<string, Map<string, unknown>>([
    // START_ID wiring: input names to `${input.X}` sources. At runtime the engine primes START's
    // record with the run arguments and narrows this compiled wiring to the passed keys (an
    // omitted input then binds absent, so START's wrapper fills its declared default). This
    // compiled entry is what a direct driver or hand-seeded test reads. An inbox input takes no
    // run argument: it is bound to its handle, a constant.
    [
      START_ID,
      new Map(
        inputParameters.map((parameter) => [
          parameter.name,
          isInboxType(parameter.contract?.type) ? { inbox: parameter.name } : `\${input.${parameter.name}}`,
        ]),
      ),
    ],
    [END_ID, endWiring],
  ]);
  return [boundaryNodes, [...endEdges, ...rootEdges], boundaryWiring];
}

/**
 * Return END's output contract from the outputs' declared types.
 *
 * A value END takes its one output's declared contract. A record END with any typed output takes an
 * exact record when every output is typed, else a partial record that leaves each untyped field
 * opaque. END commits through it, so an int in any typed `float` slot (the whole value, a list
 * item, a record field) widens to a float, while an untyped field's value stays as returned.
 *
 * @param outputs The flow's outputs, typed ones carrying a `Binding.contract`.
 * @param options.record Whether END returns a record keyed by output name.
 * @returns The declared contract, or `null` when no output is typed.
 */
function declaredOutputContract(outputs: readonly FlowOutput[], options: { record: boolean }): OutputContract | null {
  if (!options.record) return outputs[0]?.binding.contract ?? null;
  const declared = new Map(outputs.map((output) => [output.name, output.binding.contract]));
  if ([...declared.values()].every((contract) => contract === null)) return null;
  return combineOutputContracts(declared);
}
