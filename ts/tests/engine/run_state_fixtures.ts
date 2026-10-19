/**
 * Builders shared by the run-state tests (Python's `_fakes.py`, `_graph_builder.py` and the
 * helpers `test_run_state_initialization.py` lends to other files).
 *
 * A `*.test.ts` imported by another test file registers its tests twice, so a helper that two or
 * more run-state test files use lives here.
 */

import { CompiledFlow, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { bindingCoSkips, exprRefsOf } from "../../src/expr/template.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { encodePool } from "../../src/runtime/log.ts";
import type { RunState } from "../../src/runtime/state.ts";

/** The run id the run-state tests record. */
export const RUN_ID = "run-123";
/** The flow fingerprint the run-state tests record. */
export const FINGERPRINT = "a".repeat(64);

/**
 * A deterministic node: `fn(inputs)` is its single value. `reads` (`{param: source}`) declares its
 * inputs: it stamps `params`, and `graph` collects it into the flow's wiring.
 */
export class FuncNode extends Node {
  readonly kind = NodeKind.CODE;
  /** The body, from the bound input record to the value. */
  readonly fn: (inputs: Record<string, unknown>) => unknown;
  /** The declared input sources by parameter name. */
  readonly reads: Readonly<Record<string, unknown>>;

  constructor(
    nodeId: string,
    fn: (inputs: Record<string, unknown>) => unknown,
    options: { reads?: Readonly<Record<string, unknown>> } & ConstructorParameters<typeof Node>[1] = {},
  ) {
    const { reads = {}, ...rest } = options;
    super(nodeId, rest);
    this.fn = fn;
    this.reads = { ...reads };
    this.params = Object.keys(reads).map((name) => new Parameter(name));
  }

  override forward(inputs: Record<string, unknown>): NodeResult {
    return new Output(this.fn(inputs));
  }
}

/** Return `START -> work -> END`, with a real START and an output-less END. */
export function flow(): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["work", new FuncNode("work", (inputs) => inputs)],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  return CompiledFlow.fromParts(nodes, [
    new Edge("start-work", START_ID, "work"),
    new Edge("work-end", "work", END_ID),
  ]);
}

/** The flow wiring the nodes declare through `FuncNode.reads`; another node declares none. */
function deriveWiring(nodes: ReadonlyMap<string, Node>): Record<string, Readonly<Record<string, unknown>>> {
  return Object.fromEntries([...nodes].map(([id, node]) => [id, node instanceof FuncNode ? node.reads : {}]));
}

/** The text references of `source`, or none for a literal. */
function refsOf(source: unknown): string[] {
  return typeof source === "string" ? exprRefsOf(source) : [];
}

/** The producer node id of one reference: START for `input.<key>`, `<id>` for `<id>.output...`. */
function refProducer(path: string): string | null {
  const parts = path.split(".");
  if (parts.length >= 2 && parts[0] === "input") return START_ID;
  if (parts.length >= 2 && parts[1] === "output" && parts[0] !== "system" && parts[0] !== "item") return parts[0]!;
  return null;
}

/**
 * Build a compiled flow from `nodes` and `[from, to, handle?]` edges, adding a real START (with
 * one parameter per `${input.X}` the wiring reads) and END. END reads the declared `outputs` (one
 * as a value, more as a record) through edges minted from their bindings, and the manual edges
 * into END are dropped; with no outputs END returns `null`. An END with no incoming edge gets one
 * from START. `wiring` defaults to the `FuncNode.reads` of the nodes.
 */
export function graph(
  nodes: readonly Node[],
  rawEdges: readonly (readonly [string, string, string?])[],
  options: {
    outputs?: readonly FlowOutput[];
    wiring?: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  } = {},
): CompiledFlow {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const sources = options.wiring ?? deriveWiring(nodeMap);
  if (!nodeMap.has(START_ID)) {
    const names = new Set<string>();
    for (const nodeWiring of Object.values(sources)) {
      for (const source of Object.values(nodeWiring)) {
        for (const ref of refsOf(source)) {
          const name = ref.startsWith("input.") ? ref.slice("input.".length).split(".")[0]! : "";
          if (name) names.add(name);
        }
      }
    }
    nodeMap.set(START_ID, new StartNode(START_ID, { params: [...names].map((name) => new Parameter(name)) }));
  }
  const outputs = options.outputs ?? [];
  if (!nodeMap.has(END_ID)) {
    nodeMap.set(
      END_ID,
      outputs.length <= 1
        ? EndNode.value(END_ID, { outputName: outputs[0]?.name ?? null })
        : EndNode.record(END_ID, { outputNames: outputs.map((output) => output.name) }),
    );
  }
  let manual = rawEdges;
  const endEdges: Edge[] = [];
  if (outputs.length > 0) {
    manual = rawEdges.filter(([, to]) => to !== END_ID);
    const counts = new Map<string, number>();
    for (const output of outputs) {
      const optional = !bindingCoSkips(output.binding.source);
      for (const ref of refsOf(output.binding.source)) {
        const producer = refProducer(ref);
        if (producer === null || !nodeMap.has(producer)) continue;
        const index = counts.get(producer) ?? 0;
        counts.set(producer, index + 1);
        const id = `${producer}->${END_ID}#${index}`;
        endEdges.push(new Edge(id, producer, END_ID, { inputGroup: output.name, optional }));
      }
    }
  }
  const edges = manual.map(([from, to, handle], i) => new Edge(`e${i}`, from, to, { sourceHandle: handle ?? null }));
  edges.push(...endEdges);
  if (!edges.some((edge) => edge.to === END_ID)) edges.push(new Edge(`${START_ID}->${END_ID}`, START_ID, END_ID));
  const finalSources = options.wiring ?? deriveWiring(nodeMap);
  const wiring = new Map(
    Object.entries(finalSources).map(([id, nodeWiring]) => [id, new Map(Object.entries(nodeWiring))]),
  );
  if (outputs.length > 0) {
    wiring.set(END_ID, new Map(outputs.map((output) => [output.name, output.binding])));
  }
  return CompiledFlow.fromParts(nodeMap, edges, { outputs, wiring });
}

/**
 * Return every reducer-owned value of `state` in a form `toEqual` compares: the pool as its wire
 * object, maps and sets as entry arrays, JSON values as copies, and the flow and result by
 * reference.
 */
export function stateSnapshot(state: RunState): Record<string, unknown> {
  return {
    flow: state.flow,
    pool: encodePool(state.pool),
    nodeState: [...state.nodeState],
    edgeState: [...state.edgeState],
    ready: [...state.ready],
    inFlight: [...state.inFlight],
    acceptedStarts: [...state.acceptedStarts],
    result: state.result,
    depth: [...state.depth],
    growthOwner: [...state.growthOwner],
    deferredPrune: [...state.deferredPrune],
    wiringOverrides: structuredClone([...state.wiringOverrides]),
    runId: state.runId,
    flowFingerprint: state.flowFingerprint,
    nextSequence: state.nextSequence,
    numWorkers: state.numWorkers,
    input: structuredClone(state.input),
    effectiveLlmByStaticPath: structuredClone(state.effectiveLlmByStaticPath),
  };
}
