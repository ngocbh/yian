/**
 * Runtime graph-expansion machinery, the pure half.
 *
 * When a spawner (CALL / MAP / LOOP) runs, it doesn't run a child engine; it returns a seed-first
 * `Grow`. The injected `buildGrow` interpreter constructs the fragment and the dispatcher prepares
 * a canonical expansion record whose reducer plan splices the fragment.
 *
 * Every cloned node and edge id is a pure function of the callsite, the child's static id and
 * the element index (`ns` / `mapCallsite` from `nodes/ids`), with no emission counter, so prefix
 * recovery reconstructs identical ids.
 *
 * The pure cloner (`cloneChild`) splices the child's own `START_ID..END_ID` (every flow is
 * `START_ID -> body -> END_ID`) into a `Flow`: the child `START_ID` is the alias-seed point, seeded
 * with the call-args as constants (resolved values are data, never re-evaluated as templates),
 * and the child `END_ID` is the alias filler. A child node reading `${input.X}` is re-pointed to
 * the namespaced child START's output object (`${<callsite>/<start>.output.X}`). Transition
 * preparation validates the fragment and records its canonical seed and digest; the reducer then
 * splices it, initializes its dispositions, and queues its START node. Builders preserve
 * seed-route selectors and actual child boundary pairs independently of the materialized values,
 * and instantiate execution scopes from authored locations.
 *
 * Templates are copied with `copyGraph`, which shares arrays, records and carriers with the
 * original, so this module assigns fresh values and never changes a copied node's field in place.
 *
 * Knows about:   compile/model, compile/memory_layout, compile/execution_layout, expr, nodes,
 *                memory contracts, diagnostics and typesys.
 * Never imports: compose, runtime, or the package-root run boundary.
 */

import { type OriginRef, reprText } from "../diagnostics.ts";
import { rewriteConditionRefs, rewriteTemplateRefs } from "../expr/index.ts";
import {
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
  MemorySetupError,
} from "../memory/contracts.ts";
import { type Node, NodeKind } from "../nodes/base.ts";
import type { CallNode } from "../nodes/call/node.ts";
import {
  Assertion,
  Binding,
  ChildFlowBoundary,
  ExecutionPosition,
  ExecutionScope,
  isExactly,
  mapSourceStrings,
  Parameter,
  sameExecutionPosition,
  sameItems,
  SeedConnection,
} from "../nodes/contracts.ts";
import { EndNode } from "../nodes/end/node.ts";
import { mapCallsite, ns } from "../nodes/ids.ts";
import type { LoopNode } from "../nodes/loop/node.ts";
import type { MapNode } from "../nodes/map/node.ts";
import { StartNode } from "../nodes/start/node.ts";
import type { JsonObject, JsonValue } from "../typesys/json.ts";
import { NEW_MESSAGES } from "../typesys/pool.ts";
import { isList, isRecord, jsTypeName, type Value } from "../typesys/values.ts";
import { revalidateScope } from "./execution_layout.ts";
import { revalidateMemoryOccurrencePath, revalidateMemoryScope } from "./memory_layout.ts";
import { copyGraph, Edge, END_ID, Flow, GrowthRejected, START_ID, validateGrowthMetadata } from "./model.ts";

// Reserved prefix for END params synthesized from a lowered spawner post-assert. It mirrors the
// build-time prefix in `compose/validate/asserts.ts`, kept a separate literal here because compile
// never imports compose. Authors can't write `__`-prefixed ids, so a synth name never collides.
const SYNTH_PREFIX = "__assert_";

/**
 * @internal Re-namespace one canonical source under `callsite`, keeping its authored origin.
 *
 * Exactly one `Binding` layer is semantic IR: its source is rewritten and a new carrier keeps the
 * same origin and declared contract. A nested carrier is an opaque payload, never unwrapped.
 *
 * - `${input.<k>...}` becomes `${<callsite>/<start>.output.<k>...}` (the namespaced child START's
 *   output object, read through the node-first head);
 * - `${<X>.output...}` becomes `${<callsite>/<X>.output...}`, and an AGENT's
 *   `${<X>.new_messages...}` `${<callsite>/<X>.new_messages...}`;
 * - `${system.X}`, `${item...}` (even `${item.output}`) and other heads are untouched.
 *
 * Every `${...}` span goes through `rewriteTemplateRefs`, so embedded, coalesced and
 * whole-expression spans are re-namespaced leaf by leaf; each leaf string of a `ValueTemplate`
 * source is rewritten the same way.
 *
 * @throws TypeError when `binding` isn't a `Binding`.
 */
export function rensInternal(binding: Binding, callsite: string): Binding {
  if (!(binding instanceof Binding)) {
    throw new TypeError(`compiled flow wiring must contain Binding values, got ${jsTypeName(binding)}`);
  }
  const rename = (path: string): string | null => {
    const parts = path.split(".");
    if (parts.length >= 2 && parts[0] === "input") {
      const rest = parts.slice(2).join(".");
      const renamed = `${ns(callsite, START_ID)}.output.${parts[1]}`;
      return rest ? `${renamed}.${rest}` : renamed;
    }
    // `system` and `item` are reserved node ids, so `${item.output}` is the item's `output` field.
    // An AGENT's `${x.new_messages}` renames like its `${x.output}`.
    if (parts.length >= 2 && (parts[1] === "output" || parts[1] === NEW_MESSAGES) && parts[0] !== "system" && parts[0] !== "item") {
      return [`${ns(callsite, parts[0]!)}.${parts[1]}`, ...parts.slice(2)].join(".");
    }
    return null;
  };
  return mapSourceStrings(binding, (text) => rewriteTemplateRefs(text, rename)) as Binding;
}

/**
 * Lower a spawner's node-level post `asserts:` onto the cloned child END.
 *
 * A CALL's `asserts:` read `${output}` (its committed output) and its declared inputs (bare
 * `${X}`). The spawner's output is known only when the cloned child END commits under the
 * callsite, so its post asserts ride that END and fire through the ordinary node `run` wrapper:
 *
 * - `${output}` / `${output.field}` is kept: the END's own value is the spawner's output;
 * - each declared input `${X}` becomes a synth END param (`__assert_input_<X>`) wired off the
 *   cloned child START's bound output (`${<callsite>/__start__.output.X:-null}`), with an optional
 *   `<callsite>/__start__ -> <callsite>/__end__` edge, the shape build-time flow-assert lowering
 *   uses. Repeated reads of one input share one synth param.
 *
 * A synth name avoids every name already on the cloned END (the child def's own lowered asserts
 * may hold `__assert_input_<X>`): a taken base becomes `<base>_1`, `<base>_2`, ...
 *
 * @param end The cloned child END; its `params` and `postAsserts` are reassigned.
 * @param endWiring The cloned END's fresh wiring map; synth sources are added.
 * @param endEdges The fragment's edges; synth producer edges are appended.
 */
function lowerSpawnerPostAsserts(
  end: Node,
  endWiring: Map<string, unknown>,
  endEdges: Edge[],
  callsite: string,
  postAsserts: readonly Assertion[],
  inputNames: ReadonlySet<string>,
): void {
  if (postAsserts.length === 0) return;
  const startNs = ns(callsite, START_ID);
  const synthForInput = new Map<string, string>();
  const taken = new Set([...(end.params ?? []).map((param) => param.name), ...endWiring.keys()]);

  const alloc = (base: string): string => {
    let name = base;
    for (let i = 1; taken.has(name); i++) name = `${base}_${i}`;
    taken.add(name);
    return name;
  };

  const rename = (path: string): string | null => {
    const [head, ...rest] = path.split(".") as [string, ...string[]];
    if (head === "output" || !inputNames.has(head)) return null;
    let synth = synthForInput.get(head);
    if (synth === undefined) {
      synth = alloc(`${SYNTH_PREFIX}input_${head}`);
      synthForInput.set(head, synth);
      end.params = [...(end.params ?? []), new Parameter(synth)];
      // `:-null` keeps the END input edge optional, so END waits for START to settle and binds
      // null for an omitted value instead of co-skipping.
      endWiring.set(synth, `\${${startNs}.output.${head}:-null}`);
      endEdges.push(new Edge(`${startNs}->${end.id}#${synth}`, startNs, end.id, { inputGroup: synth, optional: true }));
    }
    return [synth, ...rest].join(".");
  };

  end.postAsserts = [
    ...end.postAsserts,
    ...postAsserts.map(
      (assertion) =>
        new Assertion(rewriteConditionRefs(assertion.expression, rename), assertion.display, {
          origin: assertion.origin,
        }),
    ),
  ];
}

/**
 * Splice a child flow's `START_ID..END_ID` at `callsite`.
 *
 * Every child node (its START and END included) is copied and namespaced; the child START is
 * seeded with `record` as constants and the child END is the alias filler. Pure: the growth
 * reducer applies the returned fragment.
 *
 * @param child The child flow template.
 * @param callsite The namespace, the spawner id or a MAP element's callsite.
 * @param record The resolved call-args seeding the child START.
 * @param options.originContext The enclosing spawner's ordered failure trace, prefixed to each
 *   clone's existing ancestry without deduplication.
 * @param options.occurrencePath The complete dynamic memory occurrence every clone inherits;
 *   static templates must carry an empty one.
 * @param options.executionPath Prefixes each direct template node's initial execution position;
 *   nested child templates keep their static scopes until instantiated.
 * @param options.seedPrefix An optional MAP item index for materialized seed selectors.
 * @param options.postAsserts A CALL spawner's node-level post asserts, lowered onto the cloned END.
 * @param options.inputNames The CALL spawner's declared input names.
 * @returns A `Flow` whose `startId` is the namespaced child START and whose `endId` is the
 *   namespaced child END.
 * @throws TypeError or RangeError for a noncanonical occurrence path, execution path, seed prefix,
 *   execution scope or growth metadata.
 * @throws MemorySetupError for a noncanonical or dynamic static memory scope on a child node.
 */
export function cloneChild(
  child: Flow,
  callsite: string,
  record: Readonly<Record<string, unknown>>,
  options: {
    originContext?: readonly OriginRef[];
    occurrencePath?: readonly MemoryOccurrenceSegment[];
    executionPath?: readonly ExecutionPosition[];
    seedPrefix?: readonly number[];
    postAsserts?: readonly Assertion[] | null;
    inputNames?: ReadonlySet<string> | null;
  } = {},
): Flow {
  const {
    originContext = [],
    occurrencePath = [],
    executionPath = [],
    seedPrefix = [],
    postAsserts = null,
    inputNames = null,
  } = options;
  revalidateMemoryOccurrencePath(occurrencePath, "occurrence_path");
  validateGrowthMetadata(child);
  new ExecutionScope([], { occurrencePath: executionPath });
  if (
    !isList(seedPrefix) ||
    seedPrefix.length > 1 ||
    seedPrefix.some((index) => !Number.isSafeInteger(index) || index < 0)
  ) {
    throw new RangeError("seed_prefix must be empty or one non-negative item index");
  }

  const nodes = new Map<string, Node>();
  for (const [nid, node] of child.nodes) {
    const clone = copyGraph(node);
    clone.id = ns(callsite, nid);
    clone.originContext = [...originContext, ...clone.originContext];
    if (clone.executionScope !== null) {
      const scope: unknown = clone.executionScope;
      if (!isExactly(scope, ExecutionScope)) throw new TypeError("child execution scope must be an ExecutionScope");
      const { staticPath, flowPath, boundary } = scope;
      new ExecutionScope(staticPath, { flowPath, occurrencePath: scope.occurrencePath, boundary });
      const local = clone.isLoop ? [new ExecutionPosition("loop_driver", staticPath, 0)] : [];
      if (!sameItems(scope.occurrencePath, local, sameExecutionPosition)) {
        throw new RangeError("child execution template contains a dynamic occurrence");
      }
      clone.executionScope = new ExecutionScope(staticPath, {
        flowPath,
        occurrencePath: [...executionPath, ...local],
        boundary,
      });
    }
    if (clone.memoryScope !== null) {
      try {
        revalidateMemoryScope(clone.memoryScope, `node ${reprText(nid)} memory_scope`);
      } catch (error) {
        if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
        throw new MemorySetupError(`node ${reprText(nid)} has a noncanonical static memory scope`, {
          origin: clone.origin,
          cause: error,
        });
      }
      if (clone.memoryScope.occurrencePath.length > 0) {
        throw new MemorySetupError(`node ${reprText(nid)} static memory template has a dynamic occurrence`, {
          origin: clone.origin,
        });
      }
      clone.memoryScope = new MemoryScope(clone.memoryScope.staticPath, {
        memories: clone.memoryScope.memories,
        occurrencePath,
      });
    }
    // A compiled loop inside the child attributes itself (`originId` is its own id). The cloned
    // loop is a new compiled origin at the namespaced callsite, so its origin follows to the clone's
    // id; otherwise its commit and body callsite would key the un-namespaced id.
    if (node.originId === nid) clone.originId = clone.id;
    nodes.set(clone.id, clone);
  }

  // Re-namespace every node's wiring into fresh maps (internal reads re-pointed; nothing baked).
  const wiring = new Map<string, Map<string, unknown>>();
  for (const [nid, nodeWiring] of child.wiring) {
    wiring.set(
      ns(callsite, nid),
      new Map([...nodeWiring].map(([param, source]) => [param, rensInternal(source, callsite)])),
    );
  }

  const edges = child.edges.map(
    (e) =>
      new Edge(ns(callsite, e.id), ns(callsite, e.from), ns(callsite, e.to), {
        sourceHandle: e.sourceHandle,
        inputGroup: e.inputGroup,
        optional: e.optional,
        ordering: e.ordering,
      }),
  );

  // Seed the child START with the call-args as constants, replacing the provisional
  // `${input.X}` wiring. The spawner bound them before growing, so each value is data: a text's
  // `$` is escaped to `$$` so START's binder hands it over verbatim (a function replacer, since
  // a `"$$"` replacement string inserts a single `$`).
  const startNs = ns(callsite, child.startId);
  wiring.set(
    startNs,
    new Map(
      Object.entries(record).map(([param, value]) => [
        param,
        typeof value === "string" ? value.replaceAll("$", () => "$$") : value,
      ]),
    ),
  );

  const outId = ns(callsite, child.endId);
  // The child END's own lowered post asserts ride the copied node unchanged, with their synth
  // params, re-namespaced wiring and re-keyed producer edges. A spawner's own post asserts are
  // net-new and lowered here.
  let endWiring = wiring.get(outId);
  if (endWiring === undefined) {
    endWiring = new Map();
    wiring.set(outId, endWiring);
  }
  lowerSpawnerPostAsserts(nodes.get(outId)!, endWiring, edges, callsite, postAsserts ?? [], inputNames ?? new Set());

  const declared = new Set((child.nodes.get(child.startId)!.params ?? []).map((param) => param.name));
  const connections = [
    ...Object.keys(record)
      .filter((name) => declared.has(name))
      .map((name) => new SeedConnection(startNs, name, [...seedPrefix, name])),
    ...child.inputConnections
      .filter((item) => item.childStartId !== child.startId)
      .map(
        (item) =>
          new SeedConnection(ns(callsite, item.childStartId), item.inputName, [...seedPrefix, ...item.seedPath]),
      ),
  ];
  const boundaries = [
    new ChildFlowBoundary(startNs, outId),
    ...child.childBoundaries
      .filter((item) => item.startId !== child.startId || item.endId !== child.endId)
      .map((item) => new ChildFlowBoundary(ns(callsite, item.startId), ns(callsite, item.endId))),
  ];
  return new Flow(nodes, edges, {
    wiring,
    startId: startNs,
    endId: outId,
    inputConnections: connections,
    childBoundaries: boundaries,
  });
}

/**
 * The pure CALL builder: the fragment a CALL spawner grows into.
 *
 * Wraps `cloneChild` and bakes `commitAs = callsite` on the cloned child END, so its output
 * commits under the spawner id on the ordinary success path. The child's own asserts ride the
 * cloned boundary nodes; the spawner's `postAsserts` (reading `${output}` and the declared
 * `inputNames`) are lowered onto the cloned END. `originContext` is the spawner's failure trace,
 * prefixed to every clone. `memoryScope` gives its occurrence path unchanged to each clone;
 * `executionScope` appends the authored CALL position. The seed recipes select parent seed
 * parameter names directly.
 *
 * @throws TypeError or RangeError for a noncanonical memory or execution scope, or as `cloneChild`.
 */
export function callSubgraph(
  child: Flow,
  callsite: string,
  record: Readonly<Record<string, unknown>>,
  options: {
    originContext?: readonly OriginRef[];
    memoryScope?: MemoryScope | null;
    executionScope?: ExecutionScope | null;
    postAsserts?: readonly Assertion[] | null;
    inputNames?: ReadonlySet<string> | null;
  } = {},
): Flow {
  const { originContext = [], memoryScope = null, executionScope = null, postAsserts = null, inputNames = null } =
    options;
  if (memoryScope !== null) revalidateMemoryScope(memoryScope);
  if (executionScope !== null) revalidateScope(executionScope);
  const cloned = cloneChild(child, callsite, record, {
    originContext,
    occurrencePath: memoryScope === null ? [] : memoryScope.occurrencePath,
    executionPath:
      executionScope === null
        ? []
        : [...executionScope.occurrencePath, new ExecutionPosition("call", executionScope.staticPath)],
    postAsserts,
    inputNames,
  });
  cloned.nodes.get(cloned.endId)!.commitAs = callsite;
  return cloned;
}

/**
 * The pure MAP builder: the fragment a MAP spawner grows into.
 *
 * MAP is CALL × N plus a synthesized `<spawner>/__start__` fan-out and a list-END fan-in. Element
 * `i` is `cloneChild` at `mapCallsite(spawnerId, i)`, seeded with `records[i]`. The synthetic
 * START (no params; it runs once and emits `{}`) reaches every element START through an ordering
 * edge, their sole gate since MAP records are resolved values. The N child ENDs fan into one
 * `EndNode.list` (`<spawner>/__end__`) through one `e<i>` input group each; it carries
 * `commitAs = spawnerId`. With no records, an ordering edge joins the synthetic START to the
 * list-END so it still runs and emits `[]`.
 *
 * `originContext` is the spawner's failure trace, given to every clone and both generated nodes.
 * With `memoryScope`, element `i` appends a MAP-item occurrence and the generated nodes reuse the
 * spawner's scope. Execution scopes also append the item index; seed recipes select that same
 * index and parameter name. Only actual item START/END pairs become child boundaries.
 *
 * @returns A `Flow` from the synthetic START to the list-END.
 * @throws TypeError or RangeError for a noncanonical memory or execution scope, or as `cloneChild`.
 */
export function mapSubgraph(
  child: Flow,
  spawnerId: string,
  records: readonly Readonly<Record<string, unknown>>[],
  options: {
    originContext?: readonly OriginRef[];
    memoryScope?: MemoryScope | null;
    executionScope?: ExecutionScope | null;
  } = {},
): Flow {
  const { originContext = [], memoryScope = null, executionScope = null } = options;
  const mapStartId = ns(spawnerId, START_ID);
  const mapEndId = ns(spawnerId, END_ID);
  const nodes = new Map<string, Node>();
  const edges: Edge[] = [];
  const wiring = new Map<string, ReadonlyMap<string, unknown>>();
  const endWiring = new Map<string, unknown>();
  const endEdges: Edge[] = [];
  const startEdges: Edge[] = [];
  const connections: SeedConnection[] = [];
  const boundaries: ChildFlowBoundary[] = [];

  if (memoryScope !== null) revalidateMemoryScope(memoryScope);
  if (executionScope !== null) revalidateScope(executionScope);

  for (const [i, record] of records.entries()) {
    const occurrencePath =
      memoryScope === null
        ? []
        : [
            ...memoryScope.occurrencePath,
            new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, memoryScope.staticPath, i),
          ];
    const cloned = cloneChild(child, mapCallsite(spawnerId, i), record, {
      originContext,
      occurrencePath,
      executionPath:
        executionScope === null
          ? []
          : [...executionScope.occurrencePath, new ExecutionPosition("map_item", executionScope.staticPath, i)],
      seedPrefix: [i],
    });
    for (const [id, node] of cloned.nodes) nodes.set(id, node);
    edges.push(...cloned.edges);
    for (const [id, nodeWiring] of cloned.wiring) wiring.set(id, nodeWiring);
    connections.push(...cloned.inputConnections);
    boundaries.push(...cloned.childBoundaries);
    startEdges.push(new Edge(`${mapStartId}->${cloned.startId}#0`, mapStartId, cloned.startId, { ordering: true }));
    endWiring.set(`e${i}`, `\${${cloned.endId}.output}`);
    endEdges.push(new Edge(`${cloned.endId}->${mapEndId}#${i}`, cloned.endId, mapEndId, { inputGroup: `e${i}` }));
  }

  const unbounded =
    executionScope === null
      ? null
      : new ExecutionScope(executionScope.staticPath, {
          flowPath: executionScope.flowPath,
          occurrencePath: executionScope.occurrencePath,
        });
  const mapStart = new StartNode(mapStartId, { params: [] });
  mapStart.originContext = originContext;
  mapStart.memoryScope = memoryScope;
  mapStart.executionScope = unbounded;
  nodes.set(mapStartId, mapStart);
  wiring.set(mapStartId, new Map());

  const mapEnd = EndNode.list(mapEndId, { n: records.length });
  mapEnd.commitAs = spawnerId;
  mapEnd.originContext = originContext;
  mapEnd.memoryScope = memoryScope;
  mapEnd.executionScope = unbounded;
  nodes.set(mapEndId, mapEnd);
  edges.push(...startEdges, ...endEdges);
  wiring.set(mapEndId, endWiring);
  if (records.length === 0) {
    edges.push(new Edge(`${mapStartId}->${mapEndId}#0`, mapStartId, mapEndId, { ordering: true }));
  }

  return new Flow(nodes, edges, {
    wiring,
    startId: mapStartId,
    endId: mapEndId,
    inputConnections: connections,
    childBoundaries: boundaries,
  });
}

/**
 * The pure LOOP continue builder: body `k` and the fresh next-iteration driver.
 *
 * `origin` is the origin loop id `L`, never the running driver's id, so live construction (from
 * `L~k`) and durable construction (from `L`) build the same `L#k/…` namespace. `carried` seeds
 * body `k`; `driver` is the fresh `L~(k+1)` from `respawn(k + 1, carried)`.
 *
 * Splices body `k` (`cloneChild` at `mapCallsite(origin, k)`, with no `commitAs`: its output
 * commits under its own END id and feeds the next driver by plain wiring), the driver, one
 * producer edge `body_k.END -> L~(k+1)` per returned field (or one ordering edge when none is
 * wired), and the driver's wiring `{field: "${L#k/END.output.field}"}`. Body `k` seeds only the
 * carried fields its START declares. A partial driver wires only its returned fields; each
 * omitted field keeps the default `respawn` gave it and gets a passthrough `SeedConnection`.
 *
 * `originContext` is the running driver's failure trace and prefixes only the body clones. With
 * `memoryScope`, the body appends this LOOP-iteration occurrence while the driver keeps its own
 * scope. The execution path replaces the current driver position with body `k`.
 *
 * @returns A `Flow` from body `k`'s START to its END.
 * @throws TypeError or RangeError for a noncanonical memory or execution scope, or as `cloneChild`.
 * @throws RangeError when the execution scope doesn't end in driver `k`'s position.
 */
export function loopContinueSubgraph(
  child: Flow,
  origin: string,
  carried: Readonly<Record<string, unknown>>,
  k: number,
  driver: LoopNode,
  options: {
    originContext?: readonly OriginRef[];
    memoryScope?: MemoryScope | null;
    executionScope?: ExecutionScope | null;
  } = {},
): Flow {
  const { originContext = [], memoryScope = null, executionScope = null } = options;
  if (memoryScope !== null) revalidateMemoryScope(memoryScope);
  if (driver.memoryScope !== null) revalidateMemoryScope(driver.memoryScope, "loop driver memory_scope");
  let executionPath: ExecutionPosition[] = [];
  if (executionScope !== null) {
    revalidateScope(executionScope);
    const current = new ExecutionPosition("loop_driver", executionScope.staticPath, k);
    const last = executionScope.occurrencePath.at(-1);
    if (last === undefined || !sameExecutionPosition(last, current)) {
      throw new RangeError("loop scope must end in its current driver position");
    }
    executionPath = [
      ...executionScope.occurrencePath.slice(0, -1),
      new ExecutionPosition("loop_body", executionScope.staticPath, k),
    ];
  }

  // The carried record may be wider than the body's inputs; only the declared inputs seed its
  // START, while the whole record continues through the next driver.
  const bodyInputNames = new Set((child.nodes.get(child.startId)!.params ?? []).map((param) => param.name));
  const bodySeed = Object.fromEntries(Object.entries(carried).filter(([name]) => bodyInputNames.has(name)));
  const occurrencePath =
    memoryScope === null
      ? []
      : [
          ...memoryScope.occurrencePath,
          new MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, memoryScope.staticPath, k),
        ];
  const cloned = cloneChild(child, mapCallsite(origin, k), bodySeed, { originContext, occurrencePath, executionPath });
  const bodyEnd = cloned.endId;
  const nodes = new Map(cloned.nodes);
  const edges = [...cloned.edges];
  const wiring = new Map<string, ReadonlyMap<string, unknown>>(cloned.wiring);
  nodes.set(driver.id, driver);
  const returned = driver.returnedFields;
  const updated = Object.keys(carried).filter((name) => returned === null || returned.includes(name));
  const driverWiring = new Map<string, unknown>();
  for (const [i, field] of updated.entries()) {
    driverWiring.set(field, `\${${bodyEnd}.output.${field}}`);
    edges.push(new Edge(`${bodyEnd}->${driver.id}#${i}`, bodyEnd, driver.id, { inputGroup: field }));
  }
  // With no returned field to wire, an ordering edge still runs the driver after the body.
  if (updated.length === 0) {
    edges.push(new Edge(`${bodyEnd}->${driver.id}#0`, bodyEnd, driver.id, { ordering: true }));
  }
  wiring.set(driver.id, driverWiring);
  const passthrough = Object.keys(carried)
    .filter((name) => !updated.includes(name))
    .map((name) => new SeedConnection(driver.id, name, [name]));
  return new Flow(nodes, edges, {
    wiring,
    startId: cloned.startId,
    endId: bodyEnd,
    inputConnections: [...cloned.inputConnections, ...passthrough],
    childBoundaries: cloned.childBoundaries,
  });
}

/** Throw `error` wrapped as a growth rejection. */
function reject(error: Error): never {
  throw new GrowthRejected(error);
}

/** Require one record with non-empty keys from a legal spawner seed. */
function growthRecord(seed: JsonValue, label: string): JsonObject {
  if (!isRecord(seed)) reject(new TypeError(`${label} growth seed must be a record`));
  if (Object.keys(seed).some((key) => !key)) {
    reject(new RangeError(`${label} growth seed record keys must be non-empty`));
  }
  return seed as JsonObject;
}

/** Require MAP's ordered list of per-item records. */
function mapGrowthRecords(seed: JsonValue): JsonObject[] {
  if (!isList(seed)) reject(new TypeError("MAP growth seed must be a list of records"));
  return (seed as JsonValue[]).map((record, index) => growthRecord(record, `MAP item ${index}`));
}

/**
 * Require LOOP's carried record and non-negative iteration index.
 *
 * The canonical seed holds the index as a `bigint`; it becomes a safe-integer `number`.
 */
function loopGrowthSeed(seed: JsonValue): [JsonObject, number] {
  if (!isList(seed) || seed.length !== 2) {
    reject(new RangeError("LOOP growth seed must be a two-item (record, iteration) pair"));
  }
  const [record, iteration] = seed as [JsonValue, JsonValue];
  const carried = growthRecord(record, "LOOP carried");
  if (typeof iteration !== "bigint") reject(new TypeError("LOOP growth seed iteration must be an integer"));
  if (iteration < 0n) reject(new RangeError("LOOP growth seed iteration must be non-negative"));
  if (iteration > BigInt(Number.MAX_SAFE_INTEGER)) {
    reject(
      new RangeError(`LOOP growth seed iteration must be at most ${Number.MAX_SAFE_INTEGER}, got ${iteration}`),
    );
  }
  return [carried, Number(iteration)];
}

/**
 * Build the deterministic fragment one spawner's canonical seed describes.
 *
 * The sole kind-aware growth-construction path: nodes calculate only their pure seed, and the
 * package-root run boundary injects this builder into the kind-blind runtime for both live growth
 * and durable replay.
 *
 * @param spawner The CALL, MAP or LOOP node whose seed is interpreted.
 * @param seed The persisted builder input that node produced: a CALL record, a MAP list of
 *   records, or a LOOP `[record, iteration]` pair with a `bigint` iteration.
 * @returns The self-contained fragment the runtime splices into its private live graph.
 * @throws GrowthRejected when a legal spawner kind produced a malformed seed.
 * @throws RangeError when `spawner` isn't a CALL, MAP or LOOP.
 */
export function buildGrow(spawner: Node, seed: JsonValue): Flow {
  switch (spawner.kind) {
    case NodeKind.CALL: {
      const record = growthRecord(seed, "CALL");
      const call = spawner as CallNode;
      return callSubgraph(call.child as Flow, call.id, { ...record }, {
        originContext: call.failureTrace(),
        memoryScope: call.memoryScope,
        executionScope: call.executionScope,
        postAsserts: call.postAsserts,
        inputNames: new Set((call.params ?? []).map((param) => param.name)),
      });
    }
    case NodeKind.MAP: {
      const records = mapGrowthRecords(seed);
      const map = spawner as MapNode;
      return mapSubgraph(
        map.child as Flow,
        map.id,
        records.map((record) => ({ ...record })),
        { originContext: map.failureTrace(), memoryScope: map.memoryScope, executionScope: map.executionScope },
      );
    }
    case NodeKind.LOOP: {
      const [carried, iteration] = loopGrowthSeed(seed);
      const loop = spawner as LoopNode;
      return loopContinueSubgraph(
        loop.child as Flow,
        loop.originId!,
        { ...carried },
        iteration,
        loop.respawn(iteration + 1, carried as Readonly<Record<string, Value>>),
        { originContext: loop.failureTrace(), memoryScope: loop.memoryScope, executionScope: loop.executionScope },
      );
    }
    default:
      throw new RangeError(`node ${reprText(spawner.id)} (${spawner.kind}) cannot build graph growth`);
  }
}
