/**
 * The descriptor build cache and the assembly helpers of the post-parse pipeline: a port of
 * `tests/engine/test_assemble.py` (the descriptor build cache tests). A cycle's error is a
 * `LoadError`; Python also asserts it isn't a `RecursionError`, which in TS would be a stack
 * overflow `RangeError` before any `LoadError` is made.
 *
 * TS-only: direct rows for `assertionCursors`, `registerAssertionOrigin`, `stampBindingLabels`,
 * `stampNodeWiringOrigins`, `splitTypedInputs`, `attachInputContracts`, `flowOutputs`,
 * `stampFlowOutputOrigins`, `ComposedFlow.input` and `ComposedFlow.diagnose`, which Python tests
 * only through `load_flow`; nodes parse with stand-in kinds. The pinned cases run
 * `parseFile`, `parseNodes` and `splitTypedInputs` as `assemble` calls them.
 *
 * The conductor: the input representation guards
 * (`test_resolved_parameters_reject_both_input_representations`,
 * `test_resolved_parameters_reject_missing_input_representation`) and the resolved parameter
 * boundary (`test_resolved_parameters_preserve_structural_contract_identity`,
 * `test_resolved_parameters_accept_zero_args_and_wire_empty_start`). TS-only conductor rows: a
 * cursor with resolved parameters, the memory declaration check, the errors `parseNodes`, the
 * `uses:` alias check and the value-entry reader raise through `assemble`, and one happy path over
 * stand-in nodes (its data and boundary edges). Later (T3-17): the value-entry rows of
 * `stampNodeWiringOrigins` (a value entry's input owns its parameter and binding at the entry
 * itself) and `splitTypedInputs`
 * (a value entry is returned unchanged), once `isValueEntry` is real.
 * Later (T3-18): the wiring parity rows (`test_wiring_parity_*`).
 *
 * The seed edge and boundary rows: `test_seed01_*`, `test_seed02_case_targets_not_bare_roots`,
 * `test_seed02_branch_target_has_control_edge`, `test_seed02_coalesce_both_producers_terminal`,
 * `test_seed05_*`, `test_seed06_*`, `test_seed18_*`, `test_terminal_edge_ids_unique`,
 * `test_assembled_edges_have_start_and_end` and `test_boundary_end_record_flag_picks_the_mode`.
 * Python's `_all_edges`, `_end_edges` and `_bare_roots` are local helpers over the seed cases
 * (`load-agent/seed_01_structured_agent`, `load-case/seed_02_case`, `load-map-call/seed_05_call_map`,
 * `load-case/seed_06_case_on`, `load-core/seed_18_research_pipeline`); they call
 * `synthesizeBoundaryGraph` over the parsed nodes, so seed 05's MAP child needn't resolve.
 * `test_seed02_branch_target_has_control_edge` reads the edges of seed 02's built `gate`, where
 * Python's `_all_edges` also infers the data edges, which hold no control edge. Python's
 * `ValueError` for two outputs of a value-mode END is a `RangeError`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import {
  type AssembleOptions,
  assemble,
  assertionCursors,
  attachInputContracts,
  type CallableScope,
  ComposedFlow,
  DescriptorBuildCache,
  flowOutputs,
  registerAssertionOrigin,
  splitTypedInputs,
  stampBindingLabels,
  stampFlowOutputOrigins,
  stampNodeWiringOrigins,
} from "../../src/compose/build/pipeline.ts";
import { synthesizeBoundaryGraph } from "../../src/compose/build/boundary.ts";
import { inferDataEdges } from "../../src/compose/edges/data.ts";
import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { BuildContext, BuildResult, type NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { flowNodesCursor, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { CompiledFlow, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { forget, reload } from "../../src/defaults.ts";
import {
  DiagnosticFrame,
  Failure,
  OriginRef,
  type OriginResolver,
  SourceRange,
} from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";
import { Binding, Parameter, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import type { TypeRegistry } from "../../src/typesys/types.ts";
import { isRecord, Type, ValueKind } from "../../src/typesys/values.ts";
import {
  caseDocument,
  caseRegistry,
  caseText,
  cursorAt,
  loadError,
  nodesOf,
  runTextCase,
} from "../compose/declarations.ts";
import { StandInCode, withKinds } from "../compose/stand_ins.ts";

afterEach(() => forget());

const EMPTY: TypeRegistry = new Map();

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Return diagnostics over source text large enough for synthetic descriptor ranges. */
function rangeDiagnostics(): LoadDiagnostics {
  return new LoadDiagnostics("flow.yaml", ("x".repeat(80) + "\n").repeat(60));
}

/** A kind-neutral descriptor whose callback controls one build attempt. */
class FakeBuildDescriptor {
  readonly nodeId: string;
  readonly calls: [string, BuildContext][];
  readonly callback: (() => unknown) | null;
  readonly range: SourceRange | null;

  constructor(
    nodeId: string,
    calls: [string, BuildContext][],
    callback: (() => unknown) | null = null,
    range: SourceRange | null = null,
  ) {
    this.nodeId = nodeId;
    this.calls = calls;
    this.callback = callback;
    this.range = range;
  }

  build(ctx: BuildContext): BuildResult {
    this.calls.push([this.nodeId, ctx]);
    this.callback?.();
    return new BuildResult({ node: new StartNode(this.nodeId), wiring: new Map([["built", this.nodeId]]) });
  }
}

/** Return the ids of the recorded build calls, in order. */
const calledIds = (calls: [string, BuildContext][]) => calls.map(([nodeId]) => nodeId);

describe("the descriptor build cache", () => {
  test("reuses forward builds in source order", () => {
    const calls: [string, BuildContext][] = [];
    let cache: DescriptorBuildCache;
    const descriptors = new Map([
      ["a", new FakeBuildDescriptor("a", calls, () => cache.buildOne("b", "a.input.x"))],
      ["b", new FakeBuildDescriptor("b", calls)],
    ]);
    const ctx = new BuildContext({ registry: EMPTY });
    cache = new DescriptorBuildCache(descriptors, ctx);

    const built = cache.buildInSourceOrder();

    expect([...built.keys()]).toEqual(["a", "b"]);
    expect(calledIds(calls)).toEqual(["a", "b"]);
    expect(calls.every(([, actual]) => actual === ctx)).toBe(true);
    expect(built.get("b")).toBe(cache.buildOne("b"));
    expect(calledIds(calls)).toEqual(["a", "b"]);
  });

  test("reports only the mutual cycle, at the closing range", () => {
    const calls: [string, BuildContext][] = [];
    const closingRange = new SourceRange(31, 9, 31, 27);
    let cache: DescriptorBuildCache;
    const descriptors = new Map([
      ["outside", new FakeBuildDescriptor("outside", calls, () => cache.buildOne("a", "outside.input.seed"))],
      ["a", new FakeBuildDescriptor("a", calls, () => cache.buildOne("b", "a.input.x"))],
      ["b", new FakeBuildDescriptor("b", calls, () => cache.buildOne("a", "b.input.y", closingRange))],
    ]);
    cache = new DescriptorBuildCache(
      descriptors,
      new BuildContext({ registry: EMPTY, diagnostics: rangeDiagnostics() }),
    );

    const error = loadError(() => cache.buildInSourceOrder());

    expect(error.message).toBe("node build dependency cycle: a.input.x -> b.input.y -> a");
    expect(error.message).not.toContain("outside.input.seed");
    expect(error.diagnostic.frames[0]!.range).toEqual(closingRange);
  });

  test("a cycle falls back to the first cycle edge's range", () => {
    const calls: [string, BuildContext][] = [];
    const firstEdgeRange = new SourceRange(17, 7, 17, 21);
    let cache: DescriptorBuildCache;
    const descriptors = new Map([
      ["a", new FakeBuildDescriptor("a", calls, () => cache.buildOne("b", "a.input.x", firstEdgeRange))],
      ["b", new FakeBuildDescriptor("b", calls, () => cache.buildOne("a", "b.input.y"))],
    ]);
    cache = new DescriptorBuildCache(
      descriptors,
      new BuildContext({ registry: EMPTY, diagnostics: rangeDiagnostics() }),
    );

    const error = loadError(() => cache.buildInSourceOrder());

    expect(error.message.endsWith("a.input.x -> b.input.y -> a")).toBe(true);
    expect(error.diagnostic.frames[0]!.range).toEqual(firstEdgeRange);
  });

  test("a cycle names node ids when an edge has no label", () => {
    const calls: [string, BuildContext][] = [];
    let cache: DescriptorBuildCache;
    const descriptors = new Map([
      ["a", new FakeBuildDescriptor("a", calls, () => cache.buildOne("b", "a.input.x"))],
      ["b", new FakeBuildDescriptor("b", calls, () => cache.buildOne("a"))],
    ]);
    cache = new DescriptorBuildCache(descriptors, new BuildContext({ registry: EMPTY }));

    expect(loadError(() => cache.buildInSourceOrder()).message.endsWith("a -> b -> a")).toBe(true);
  });

  test("an unlocated failure is not cached", () => {
    const calls: [string, BuildContext][] = [];
    let cache: DescriptorBuildCache;
    const fail = () => {
      throw new LoadDiagnostics().error("broken descriptor");
    };
    const descriptors = new Map([
      ["outer", new FakeBuildDescriptor("outer", calls, () => cache.buildOne("broken"))],
      ["broken", new FakeBuildDescriptor("broken", calls, fail)],
    ]);
    cache = new DescriptorBuildCache(descriptors, new BuildContext({ registry: EMPTY }));

    for (let attempt = 0; attempt < 2; attempt++) {
      const error = loadError(() => cache.buildOne("outer"));
      expect(error.message).toBe("broken descriptor");
      expect(error.diagnostic.frames).toEqual([]);
    }
    expect(calledIds(calls)).toEqual(["outer", "broken", "outer", "broken"]);
  });

  test("an unlocated failure gets the descriptor's range", () => {
    const calls: [string, BuildContext][] = [];
    const sourceRange = new SourceRange(27, 7, 27, 31);
    const fail = () => {
      throw new LoadDiagnostics().error("broken descriptor");
    };
    const cache = new DescriptorBuildCache(
      new Map([["broken", new FakeBuildDescriptor("broken", calls, fail, sourceRange)]]),
      new BuildContext({ registry: EMPTY, diagnostics: rangeDiagnostics() }),
    );

    const error = loadError(() => cache.buildOne("broken"));

    expect(error.message).toBe("broken descriptor");
    expect(error.diagnostic.frames[0]!.range).toEqual(sourceRange);
  });
});

/** Parse flow `text`'s `nodes:` as `assemble` reads them, with `StandInCode` as `code`. */
function parsed(
  text: string,
  label = "flow.yaml",
): { document: SourceDocument; nodesCursor: SourceCursor; descriptors: Map<string, NodeDescriptor> } {
  const document = parseSourceDocument(text, label);
  const nodesCursor = flowNodesCursor(document, parseFile(document).nodes);
  const descriptors = withKinds([StandInCode], () => parseNodes(nodesCursor));
  return { document, nodesCursor, descriptors };
}

/** Split `descriptors`' typed inputs as `assemble` does, under owner `["root"]`. */
function split(
  document: SourceDocument,
  nodesCursor: SourceCursor,
  descriptors: Map<string, NodeDescriptor>,
  catalogBuilder = new SourceCatalogBuilder(),
): Map<string, NodeDescriptor> {
  return splitTypedInputs(descriptors, nodesCursor, {
    registry: caseRegistry(document),
    owner: ["root"],
    catalogBuilder,
    diagnostics: LoadDiagnostics.fromDocument(document),
  });
}

/** Return the range `catalogBuilder` resolves `origin` to. */
function resolvedRange(catalogBuilder: SourceCatalogBuilder, origin: OriginRef | null): SourceRange {
  return catalogBuilder.resolveRegistered(origin!)!.range;
}

describe("assertionCursors", () => {
  const diagnostics = new LoadDiagnostics();

  test("a source-free list gets one null per assertion", () => {
    expect(assertionCursors(["a", "b"], null, diagnostics)).toEqual([null, null]);
  });

  test("an aligned sequence gives its item cursors", () => {
    const cursor = SourceDocument.parse("asserts: [a, b]\n").root.child("asserts");
    const cursors = assertionCursors(["a", "b"], cursor, diagnostics);
    expect(cursors.map((item) => item!.value)).toEqual(["a", "b"]);
  });

  test("a sequence of another length doesn't align", () => {
    const document = SourceDocument.parse("asserts: [a, b]\n", "flow.yaml");
    const cursor = document.root.child("asserts");
    const error = loadError(() =>
      assertionCursors(["a"], cursor, LoadDiagnostics.fromDocument(document)),
    );
    expect(error.message).toBe("retained assertion cursor does not align with semantic assertions");
    expect(error.diagnostic.frames[0]!.range).toEqual(cursor.range!);
  });

  test("a non-sequence has no items to align", () => {
    const cursor = SourceDocument.parse("asserts: a\n").root.child("asserts");
    expect(loadError(() => assertionCursors(["a"], cursor, diagnostics)).message).toBe(
      "retained assertion cursor does not align with semantic assertions",
    );
  });
});

describe("registerAssertionOrigin", () => {
  const diagnostics = new LoadDiagnostics();

  test("a source-free assertion has no origin", () => {
    const builder = new SourceCatalogBuilder();
    expect(registerAssertionOrigin(null, ["root", "assert", 0], builder, diagnostics)).toBe(null);
  });

  test("an authored assertion registers its item", () => {
    const cursor = SourceDocument.parse("asserts: [a > 1]\n").root.child("asserts").child(0);
    const builder = new SourceCatalogBuilder();
    const origin = registerAssertionOrigin(cursor, ["root", "assert", 0], builder, diagnostics);
    expect(origin).toBeInstanceOf(OriginRef);
    expect(resolvedRange(builder, origin)).toEqual(cursor.range!);
  });

  test("a second registration of the site is a load error", () => {
    const cursor = SourceDocument.parse("asserts: [a > 1]\n").root.child("asserts").child(0);
    const builder = new SourceCatalogBuilder();
    registerAssertionOrigin(cursor, ["root", "assert", 0], builder, diagnostics);
    expect(
      loadError(() => registerAssertionOrigin(cursor, ["root", "assert", 0], builder, diagnostics)),
    ).toBeInstanceOf(LoadError);
  });
});

describe("stampBindingLabels", () => {
  test("a wiring label names its site and a field label its phrase", () => {
    const node = new StartNode("g", {
      params: [new Parameter("on"), new Parameter("x"), new Parameter("y")],
    });
    node.wiringLabels = new Map([["on", "on:"]]);
    const before = node.bindingLabels;
    const result = new BuildResult({
      node,
      wiring: new Map(),
      fieldLabels: new Map([["input 'x'", "inline call 'f' argument 'x'"]]),
    });

    stampBindingLabels(new Map([["g", result]]));

    expect([...node.bindingLabels]).toEqual([
      ["on", "node 'g' on"],
      ["x", "inline call 'f' argument 'x'"],
    ]);
    expect(before.size).toBe(0);
  });
});

describe("stampNodeWiringOrigins", () => {
  const FLOW = `\
id: f
name: f
nodes:
  a:
    kind: code
    code: x
    input:
      qty: \${input.n}
output: \${a.output}
`;

  test("a node's input key owns its parameter and its value the binding", () => {
    const { document, nodesCursor, descriptors } = parsed(FLOW);
    const node = new StartNode("a", { params: [new Parameter("qty")] });
    const wiring = new Map<string, unknown>([["qty", "${input.n}"]]);
    const builder = new SourceCatalogBuilder();

    stampNodeWiringOrigins(descriptors, nodesCursor, new Map([["a", node]]), new Map([["a", wiring]]), {
      owner: ["root"],
      catalogBuilder: builder,
      diagnostics: LoadDiagnostics.fromDocument(document),
    });

    const entry = cursorAt(document, ["nodes", "a", "input", "qty"]);
    const binding = wiring.get("qty") as Binding;
    expect(binding).toBeInstanceOf(Binding);
    expect(binding.source).toBe("${input.n}");
    expect(resolvedRange(builder, binding.origin)).toEqual(entry.range!);
    expect(resolvedRange(builder, node.params![0]!.origin)).toEqual(entry.keyRange!);
  });

  test("a node without wiring or without an authored entry is skipped", () => {
    const { document, nodesCursor, descriptors } = parsed(FLOW);
    const generated = new StandInCode({ id: "gen", code: "x", inputs: { qty: "${a.output}" } });
    const nodes = new Map([
      ["a", new StartNode("a", { params: [new Parameter("qty")] })],
      ["gen", new StartNode("gen", { params: [new Parameter("qty")] })],
    ]);
    const wiring = new Map<string, unknown>([["qty", "${a.output}"]]);

    const withGenerated = new Map([...descriptors, ["gen", generated]]);
    stampNodeWiringOrigins(withGenerated, nodesCursor, nodes, new Map([["gen", wiring]]), {
      owner: ["root"],
      catalogBuilder: new SourceCatalogBuilder(),
      diagnostics: LoadDiagnostics.fromDocument(document),
    });

    expect(wiring.get("qty")).toBe("${a.output}");
    expect([...nodes.values()].map((node) => node.params![0]!.origin)).toEqual([null, null]);
  });
});

describe("splitTypedInputs", () => {
  test("a tagged binding splits into its value and an int contract", () => {
    const { document, nodesCursor, descriptors } = parsed(`\
id: f
name: f
nodes:
  a:
    kind: code
    code: x
  use:
    kind: code
    code: x
    input:
      qty: !int \${a.output}
      note: \${a.output}
output: \${use.output}
`);
    const builder = new SourceCatalogBuilder();

    const result = split(document, nodesCursor, descriptors, builder);

    expect(result.get("a")).toBe(descriptors.get("a")!);
    const use = result.get("use")!;
    expect(use.inputs).toEqual({ qty: "${a.output}", note: "${a.output}" });
    const contract = use.inputTypes.get("qty")!;
    expect(contract.type.kind).toBe(ValueKind.INTEGER);
    expect([...use.inputTypes.keys()]).toEqual(["qty"]);
    expect(builder.consumedTags.size).toBe(1);
    expect(descriptors.get("use")!.inputTypes.size).toBe(0);
  });

  test("a quoted old form is text, and the descriptor is kept", () => {
    const { document, nodesCursor, descriptors } = parsed(`\
id: f
name: f
nodes:
  use:
    kind: code
    code: x
    input:
      n: "int = 5"
output: \${use.output}
`);
    expect(split(document, nodesCursor, descriptors).get("use")).toBe(descriptors.get("use")!);
  });

  test("an untagged mapping holding a span is read as a template, untyped", () => {
    const { document, nodesCursor, descriptors } = parsed(`\
id: f
name: f
nodes:
  use:
    kind: code
    code: x
    input:
      cfg:
        a: \${x.output}
output: \${use.output}
`);
    const use = split(document, nodesCursor, descriptors).get("use")!;
    expect(use.inputs["cfg"]).toBeInstanceOf(ValueTemplate);
    expect(use.inputTypes.size).toBe(0);
  });

  /** Parse a case's flow and split its typed inputs as `assemble` does. */
  const splitCase = (text: string, label: string) => {
    const { document, nodesCursor, descriptors } = parsed(text, label);
    return split(document, nodesCursor, descriptors);
  };

  test.each([
    "load-core/compose_old_forms__typed_binding_text_shows_the_tag",
    "load-types/compose_tagged_bindings__tag_without_a_value_asks_for_one",
  ])("pinned %s", (id) => {
    runTextCase({ id, call: splitCase });
  });
});

describe("attachInputContracts", () => {
  const contract = new ValueContract(Type.scalar(ValueKind.INTEGER));
  const descriptors = new Map<string, NodeDescriptor>([
    ["use", new StandInCode({ id: "use", code: "x", inputTypes: new Map([["qty", contract]]) })],
    ["other", new StandInCode({ id: "other", code: "x", inputTypes: new Map([["qty", contract]]) })],
  ]);

  test("a declared type lands on the binding of its name", () => {
    const wiring = new Map<string, unknown>([["qty", "${a.output}"]]);
    attachInputContracts(descriptors, new Map([["use", wiring]]));
    const binding = wiring.get("qty") as Binding;
    expect(binding.source).toBe("${a.output}");
    expect(binding.contract).toBe(contract);
  });

  test("a declared type with no binding is an internal error", () => {
    const run = () => attachInputContracts(descriptors, new Map([["use", new Map()]]));
    const error = thrown(run);
    expect(error.message).toBe("node 'use': typed input 'qty' has no wiring binding");
    expect(error.constructor).toBe(Error);
  });
});

describe("flowOutputs", () => {
  test("no section declares no outputs", () => {
    expect(flowOutputs(null)).toEqual([]);
  });

  test("a mapping gives one output per name", () => {
    const outputs = flowOutputs({ a: "${x.output}", b: 1n });
    expect(outputs.map((output) => [output.name, output.binding.source])).toEqual([
      ["a", "${x.output}"],
      ["b", 1n],
    ]);
  });

  test("a bare binding is one output named result", () => {
    const binding = new Binding("${x.output}");
    const outputs = flowOutputs(binding);
    expect(outputs.map((output) => output.name)).toEqual(["result"]);
    expect(outputs[0]!.binding).toBe(binding);
  });
});

describe("stampFlowOutputOrigins", () => {
  /** Stamp the `output:` section of `text`, as `assemble` does, returning the result and ranges. */
  function stamp(text: string, outputs: FlowOutput[] | null = null) {
    const document = SourceDocument.parse(text, "flow.yaml");
    const cursor = document.root.child("output");
    const section = cursor.value;
    const builder = new SourceCatalogBuilder();
    const stamped = stampFlowOutputOrigins(outputs ?? flowOutputs(section), section, cursor, {
      owner: ["root"],
      catalogBuilder: builder,
      diagnostics: LoadDiagnostics.fromDocument(document),
    });
    return { stamped, builder, cursor };
  }

  test("each record value owns its output's binding", () => {
    const { stamped, builder, cursor } = stamp("output:\n  a: ${x.output}\n  b: ${y.output}\n");
    expect(stamped.map((output) => resolvedRange(builder, output.binding.origin))).toEqual([
      cursor.child("a").range!,
      cursor.child("b").range!,
    ]);
    expect(stamped[1]!.binding.source).toBe("${y.output}");
  });

  test("a bare value owns the one output's binding", () => {
    const { stamped, builder, cursor } = stamp("output: ${x.output}\n");
    expect(stamped.map((output) => output.name)).toEqual(["result"]);
    expect(resolvedRange(builder, stamped[0]!.binding.origin)).toEqual(cursor.range!);
  });

  test("an already originated binding is kept", () => {
    const output = new FlowOutput("result", new Binding("${x.output}", { origin: new OriginRef("mine") }));
    expect(stamp("output: ${x.output}\n", [output]).stamped[0]).toBe(output);
  });

  test("carriers that don't align with the section are an internal error", () => {
    const carrier = (name: string) => new FlowOutput(name, "${x.output}");
    const record = () => stamp("output:\n  a: ${x.output}\n  b: ${y.output}\n", [carrier("a")]);
    const bare = () => stamp("output: ${x.output}\n", [carrier("a"), carrier("b")]);
    for (const [run, message] of [
      [record, "flow output carriers do not align with their section"],
      [bare, "a bare flow output requires exactly one carrier"],
    ] as const) {
      const error = thrown(run);
      expect(error.message).toBe(message);
      expect(error.constructor).toBe(RangeError);
    }
  });

  test("a source-free section is returned as is", () => {
    const outputs = flowOutputs("${x.output}");
    const stamped = stampFlowOutputOrigins(outputs, "${x.output}", null, {
      owner: ["root"],
      catalogBuilder: new SourceCatalogBuilder(),
      diagnostics: new LoadDiagnostics(),
    });
    expect(stamped).toBe(outputs);
  });
});

describe("ComposedFlow", () => {
  const parameter = new Parameter("n", { required: true });
  const start = () => new StartNode(START_ID, { params: [parameter] });
  const flowOf = (node: StartNode, originResolver: OriginResolver | null = null) =>
    new ComposedFlow(new CompiledFlow(new Map([[START_ID, node]]), []), { originResolver });

  test("input is the START node's signature", () => {
    const node = start();
    const flow = flowOf(node);
    expect(flow.input).toBe(node.params!);
    expect(flow.input[0]).toBe(parameter);
    expect(Object.isFrozen(flow)).toBe(true);
    expect([flow.version, flow.name, flow.description, flow.importDepth]).toEqual([null, null, null, 0]);
  });

  test("input refuses START params that aren't exact Parameters", () => {
    const node = start();
    node.params = [{ ...parameter } as Parameter];
    expect(() => flowOf(node).input).toThrow(
      new TypeError("compiled START params must contain exact Parameter values"),
    );
  });

  const origin = new OriginRef("root/node/a");
  const failure = (trace: OriginRef[] = [origin]) =>
    new Failure({ code: "condition.evaluate", message: "boom", nodeId: "a", trace, causeType: "Error" });
  const range = new SourceRange(1, 1, 1, 5);
  const frame = new DiagnosticFrame({ label: "flow.yaml", text: "a: 1\n", range });
  const resolving = (resolve: (origin: OriginRef) => unknown) =>
    flowOf(start(), { resolve } as OriginResolver);
  const UNAVAILABLE = ["Source presentation was unavailable."];

  test("diagnose keeps a traceless failure plain", () => {
    const diagnostic = resolving(() => frame).diagnose(failure([]));
    expect([diagnostic.code, diagnostic.message, diagnostic.nodeId, diagnostic.causeType]).toEqual([
      "condition.evaluate",
      "boom",
      "a",
      "Error",
    ]);
    expect([diagnostic.frames, diagnostic.notes]).toEqual([[], []]);
  });

  test("diagnose without a resolver notes that presentation is unavailable", () => {
    const diagnostic = flowOf(start()).diagnose(failure());
    expect([diagnostic.frames, diagnostic.notes]).toEqual([[], UNAVAILABLE]);
  });

  test("diagnose copies each resolved frame", () => {
    const diagnostic = resolving(() => frame).diagnose(failure([origin, origin]));
    expect(diagnostic.frames).toEqual([frame, frame]);
    expect(diagnostic.frames[0]).not.toBe(frame);
    expect(diagnostic.notes).toEqual([]);
  });

  test.each([
    ["no frame", () => null],
    ["a zero coordinate", () => new DiagnosticFrame({ ...frame, range: new SourceRange(0, 1, 1, 5) })],
    ["an end before the start", () => new DiagnosticFrame({ ...frame, range: new SourceRange(2, 1, 1, 5) })],
    ["a non-text note", () => new DiagnosticFrame({ ...frame, note: 5 as unknown as string })],
    [
      "a throwing resolver",
      () => {
        throw new Error("down");
      },
    ],
  ])("diagnose with %s notes that presentation is unavailable", (_name, resolve) => {
    const diagnostic = resolving(resolve).diagnose(failure());
    expect([diagnostic.frames, diagnostic.notes]).toEqual([[], UNAVAILABLE]);
  });

  test("diagnose requires a Failure", () => {
    expect(() => flowOf(start()).diagnose({} as Failure)).toThrow(
      new TypeError("diagnose requires a Failure"),
    );
  });
});

/** A `code` stand-in whose node shell declares one parameter per `input:` name. */
class ParamCode extends StandInCode {
  override build(_ctx: BuildContext): BuildResult {
    const params = Object.keys(this.inputs).map((name) => new Parameter(name));
    return this._finish(new StartNode(this.id, { params }), this.sinkWiring());
  }
}

/** A callable scope over `catalogBuilder` that records each resolved id and returns an empty flow. */
function fakeScope(catalogBuilder = new SourceCatalogBuilder()): CallableScope & { resolved: string[] } {
  const resolved: string[] = [];
  return {
    catalogBuilder,
    resolved,
    resolve(flowId: string): ComposedFlow {
      resolved.push(flowId);
      return new ComposedFlow(new CompiledFlow(new Map(), []));
    },
    registerPrivate(): void {},
  };
}

/** Assemble flow `text` as the loader does, with `ParamCode` as `code`; `options` override. */
function assembleText(text: string, options: Partial<AssembleOptions> = {}): ComposedFlow {
  const document = parseSourceDocument(text, "flow.yaml");
  const file = parseFile(document);
  const root = document.root;
  return withKinds([ParamCode], () =>
    assemble({
      inputsSection: file.inputs,
      inputsCursor: root.has("input") ? root.child("input") : null,
      nodesCursor: flowNodesCursor(document, file.nodes),
      outputsSection: file.outputs,
      outputsCursor: root.has("output") ? root.child("output") : null,
      assertsSection: file.asserts,
      registry: caseRegistry(document),
      scope: fakeScope(),
      owner: ["root"],
      version: file.version,
      name: file.name,
      description: file.description,
      flowEnv: file.env,
      flowId: file.id,
      ...options,
    }),
  );
}

describe("assemble", () => {
  /** The required options of a flow with no nodes, and no input representation. */
  const bare = (): AssembleOptions => ({
    nodesCursor: SourceCursor.virtual({}),
    outputsSection: null,
    assertsSection: [],
    registry: EMPTY,
    scope: fakeScope(),
    owner: ["test"],
  });

  test.each([
    ["both input representations", { inputsSection: {}, inputParameters: [] }],
    ["no input representation", {}],
  ])("%s are refused", (_name, inputs) => {
    const error = thrown(() => assemble({ ...bare(), ...inputs }));
    expect(error).toBeInstanceOf(RangeError);
    expect(error).not.toBeInstanceOf(LoadError);
    expect(error.message).toBe("assemble requires exactly one of inputsSection or inputParameters");
  });

  test("resolved parameters keep their structural contract identity", () => {
    const elementType = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["label", new Type({ kind: ValueKind.STRING, nullable: true })]]),
      required: new Set(),
    });
    const structuralType = new Type({ kind: ValueKind.LIST_OBJECT, element: elementType });
    const parameter = new Parameter("items", {
      contract: new ValueContract(structuralType),
      required: true,
      coerceStrings: true,
    });

    const flow = assemble({ ...bare(), inputParameters: [parameter] });
    const start = flow.compiled.nodes.get(START_ID)!;

    expect(flow.input).toBe(start.params!);
    expect(flow.input[0]).toBe(parameter);
    expect(flow.input[0]!.contract).not.toBe(null);
    expect(flow.input[0]!.contract!.type).toBe(structuralType);
  });

  test("zero resolved parameters wire an empty START", () => {
    const flow = assemble({ ...bare(), inputParameters: [] });
    const start = flow.compiled.nodes.get(START_ID)!;

    expect(flow.input).toBe(start.params!);
    expect(start.params).toEqual([]);
    expect(flow.compiled.wiring.get(START_ID)).toEqual(new Map());

    const rawFlow = assemble({ ...bare(), inputsSection: {} });
    expect(rawFlow.input).toEqual([]);
    expect(rawFlow.compiled.wiring.get(START_ID)).toEqual(new Map());
  });

  test("resolved parameters can't carry an inputs cursor", () => {
    const error = thrown(() =>
      assemble({ ...bare(), inputParameters: [], inputsCursor: SourceCursor.virtual({}) }),
    );
    expect(error).toBeInstanceOf(RangeError);
    expect(error).not.toBeInstanceOf(LoadError);
    expect(error.message).toBe("pre-resolved input parameters cannot carry an inputs cursor");
  });

  test("memories must be exact MemoryDecl values", () => {
    class LooseDecl extends MemoryDecl {}
    const decl = new LooseDecl("notes", "reflective", { max_items: 4n }, new DeclSite([], ["test"]));
    const error = thrown(() => assemble({ ...bare(), inputParameters: [], memories: [decl] }));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe("memories must be an array of exact MemoryDecl values");
  });

  test("memories are restamped with this flow's site and one origin each", () => {
    const document = parseSourceDocument("memories:\n  notes: reflective\nlog: reflective\n", "flow.yaml");
    const builder = new SourceCatalogBuilder();
    const decl = (declName: string) =>
      new MemoryDecl(declName, "reflective", { max_items: 4n }, new DeclSite(["elsewhere"], ["other"]));

    const flow = assemble({
      ...bare(),
      inputParameters: [],
      scope: fakeScope(builder),
      memories: [decl("notes"), decl("log")],
      memoriesCursor: document.root.child("memories"),
    });
    const sugar = assemble({
      ...bare(),
      inputParameters: [],
      scope: fakeScope(builder),
      owner: ["sugar"],
      memories: [decl("log")],
      memoriesCursor: document.root.child("log"),
    });

    const [notes, log] = flow.compiled.memories;
    expect([notes!.site.scope, notes!.site.owner, notes!.site.node]).toEqual([[], ["test"], null]);
    expect(resolvedRange(builder, notes!.origin)).toEqual(cursorAt(document, ["memories", "notes"]).keyRange!);
    expect(resolvedRange(builder, log!.origin)).toEqual(cursorAt(document, ["memories"]).range!);
    expect(resolvedRange(builder, sugar.compiled.memories[0]!.origin)).toEqual(cursorAt(document, ["log"]).range!);
  });

  test("a reserved node id surfaces from parseNodes", () => {
    const error = loadError(() =>
      assembleText("id: f\nname: f\nnodes:\n  __start__:\n    kind: code\n    code: x\noutput: null\n"),
    );
    expect(error.message).toStartWith("node id '__start__' is reserved");
  });

  test("a uses: alias colliding with a node id is refused before any alias resolves", () => {
    const scope = fakeScope();
    const error = loadError(() =>
      assembleText("id: f\nname: f\nnodes:\n  a:\n    kind: code\n    code: x\noutput: null\n", {
        scope,
        usesAliases: ["helper", "a"],
      }),
    );
    expect(error.message).toBe('uses: alias ["a"] collides with a node id');
    expect(scope.resolved).toEqual([]);
  });

  test("a value entry reaches the value-entry reader", () => {
    const error = thrown(() => assembleText("id: f\nname: f\nnodes:\n  v: !int 5\noutput: null\n"));
    expect(error.constructor).toBe(Error);
    expect(error.message).toBe("read_value_entry is not ported yet (T3-17)");
  });

  test("a value entry's nesting is checked before the value-entry reader runs", () => {
    reload((name) => (name === "YIAN_MAX_VALUE_NESTING" ? "1" : undefined));
    const error = loadError(() => assembleText("id: f\nname: f\nnodes:\n  v: !list [[1]]\noutput: null\n"));
    expect(error.message).toStartWith("value entry 'v' at [0]: value nests deeper than 1 levels");
  });

  test("two nodes assemble in source order, with env, origins, the typed input and the outputs", () => {
    const text = `\
id: f
name: f
version: "1"
description: two steps
input:
  n: !int
env:
  region: eu
  tier: base
nodes:
  a:
    kind: code
    code: x
    env:
      tier: gold
    input:
      qty: !int \${input.n}
  b:
    kind: code
    code: y
    input:
      note: \${a.output}
output:
  total: \${b.output}
  first: \${a.output}
`;
    const document = parseSourceDocument(text, "flow.yaml");
    const builder = new SourceCatalogBuilder();
    const scope = fakeScope(builder);

    const flow = assembleText(text, { scope, usesAliases: ["helper"] });

    const compiled = flow.compiled;
    expect([...compiled.nodes.keys()]).toEqual(["a", "b", START_ID, END_ID]);
    // The data edges of the wiring (`${input.n}` reads START), then END's producer edges; every
    // body node has an incoming edge, so there is no root edge.
    expect(compiled.edges).toStrictEqual([
      new Edge("__start__->a#0", START_ID, "a", { inputGroup: "qty" }),
      new Edge("a->b#0", "a", "b", { inputGroup: "note" }),
      new Edge("b->__end__#0", "b", END_ID, { inputGroup: "total" }),
      new Edge("a->__end__#0", "a", END_ID, { inputGroup: "first" }),
    ]);
    expect(scope.resolved).toEqual(["helper"]);
    expect([flow.version, flow.name, flow.description, compiled.flowId]).toEqual(["1", "f", "two steps", "f"]);

    const a = compiled.nodes.get("a")!;
    const b = compiled.nodes.get("b")!;
    expect(a.env).toEqual({ region: "eu", tier: "gold" });
    expect(b.env).toEqual({ region: "eu", tier: "base" });
    expect(resolvedRange(builder, a.origin)).toEqual(cursorAt(document, ["nodes", "a"]).keyRange!);
    expect(resolvedRange(builder, b.origin)).toEqual(cursorAt(document, ["nodes", "b"]).keyRange!);

    const qtyEntry = cursorAt(document, ["nodes", "a", "input", "qty"]);
    const qty = compiled.wiring.get("a")!.get("qty")!;
    expect(qty.source).toBe("${input.n}");
    expect(qty.contract!.type.kind).toBe(ValueKind.INTEGER);
    expect(resolvedRange(builder, qty.origin)).toEqual(qtyEntry.range!);
    expect(resolvedRange(builder, a.params![0]!.origin)).toEqual(qtyEntry.keyRange!);
    const note = compiled.wiring.get("b")!.get("note")!;
    expect(note.contract).toBe(null);
    expect(resolvedRange(builder, note.origin)).toEqual(cursorAt(document, ["nodes", "b", "input", "note"]).range!);

    expect(compiled.outputs.map((output) => [output.name, output.binding.source])).toEqual([
      ["total", "${b.output}"],
      ["first", "${a.output}"],
    ]);
    expect(compiled.outputs.map((output) => resolvedRange(builder, output.binding.origin))).toEqual([
      cursorAt(document, ["output", "total"]).range!,
      cursorAt(document, ["output", "first"]).range!,
    ]);
  });
});

/** Return END_ID's producer edges, synthesized from the flow `output:` bindings. */
function endEdges(outputs: unknown, nodeIds: ReadonlySet<string>, edges: readonly Edge[]): Edge[] {
  const [, allEdges] = synthesizeBoundaryGraph([], flowOutputs(outputs), nodeIds, edges, null, {
    record: isRecord(outputs),
  });
  return allEdges.filter((edge) => edge.to === END_ID);
}

/**
 * Return seed case `id`'s node ids, its data edges reconciled with the case builds' control edges,
 * and its `output:` section. Leaf wiring is built for the data-edge inference; a CASE's own build
 * emits its reconciled data and control edges, appended after the leaf data edges, as the loader
 * does. A CASE builds to a `CaseNode` under the same id, so the node ids are the descriptors'.
 */
function allEdges(id: string): { nodeIds: Set<string>; edges: Edge[]; outputs: unknown } {
  const { document } = caseDocument(id);
  const file = parseFile(document);
  const registry = caseRegistry(document);
  const descriptors = parseNodes(flowNodesCursor(document, file.nodes));
  const wiring = new Map(
    [...descriptors]
      .filter(([, descriptor]) => descriptor instanceof AgentDescriptor || descriptor instanceof CodeDescriptor)
      .map(([nodeId, descriptor]) => [nodeId, descriptor.build(new BuildContext({ registry })).wiring]),
  );
  const caseEdges = [...descriptors.values()]
    .filter((descriptor) => descriptor instanceof CaseDescriptor)
    .flatMap((descriptor) => descriptor.build(new BuildContext({ registry })).edges);
  return {
    nodeIds: new Set(descriptors.keys()),
    edges: [...inferDataEdges(descriptors, wiring), ...caseEdges],
    outputs: file.outputs,
  };
}

/**
 * Return the body nodes `synthesizeBoundaryGraph` marks as bare roots: its `START_ID -> nodeId`
 * edges with no `inputGroup` (a data-producer edge carries one; a control edge a `sourceHandle`).
 */
function bareRoots(nodeIds: ReadonlySet<string>, edges: readonly Edge[]): Set<string> {
  const [, allEdges] = synthesizeBoundaryGraph([], [], nodeIds, edges, null, { record: false });
  return new Set(
    allEdges
      .filter((edge) => edge.from === START_ID && edge.to !== END_ID && edge.inputGroup === null)
      .map((edge) => edge.to),
  );
}

const SEED_01 = "load-agent/seed_01_structured_agent";
const SEED_02 = "load-case/seed_02_case";
const SEED_05 = "load-map-call/seed_05_call_map";
const SEED_06 = "load-case/seed_06_case_on";
const SEED_18 = "load-core/seed_18_research_pipeline";

// Roots and input-producer edges: an `${input.X}` reader gets a `START_ID -> reader` data edge,
// not a bare root; a case branch target stays control-gated.

test("seed 01's score is an input producer, not a bare root", () => {
  const { nodeIds, edges } = allEdges(SEED_01);
  // `score` reads `${input.topic}`: a START_ID->score data edge (inputGroup set), no bare root edge.
  expect(edges.some((edge) => edge.from === START_ID && edge.to === "score" && edge.inputGroup !== null)).toBe(true);
  expect(bareRoots(nodeIds, edges).has("score")).toBe(false);
});

test("seed 02's case targets are not bare roots", () => {
  // positive/cautious are control-gated branch targets (gate->positive/cautious); neither is a bare
  // root, though both also read `${input.topic}` (a START_ID data edge).
  const { nodeIds, edges } = allEdges(SEED_02);
  const roots = bareRoots(nodeIds, edges);
  expect(roots.has("positive") || roots.has("cautious")).toBe(false);
  expect(roots.has("gate")).toBe(false); // gate has the score->gate data edge
});

test("seed 02's branch target has a control edge", () => {
  const { text, label } = caseText("load-case/seed_02_case");
  const edges = nodesOf(text, [], label).get("gate")!.build(new BuildContext({ registry: EMPTY })).edges;

  const control = edges.filter((edge) => edge.to === "positive" && edge.sourceHandle !== null);
  expect(control).toHaveLength(1);
  expect(control[0]!.from).toBe("gate");
  expect(control[0]!.sourceHandle).toBe("positive");
});

test("seed 06's notes are not bare roots", () => {
  const { nodeIds, edges } = allEdges(SEED_06);
  const roots = bareRoots(nodeIds, edges);
  for (const note of ["pro_note", "con_note", "choppy_note"]) expect(roots.has(note)).toBe(false);
});

// Terminals from the flow `output:` bindings.

/** Return the producers of seed `id`'s END edges, checking each edge ends at END. */
function terminalProducers(id: string): Set<string> {
  const { nodeIds, edges, outputs } = allEdges(id);
  const ends = endEdges(outputs, nodeIds, edges);
  for (const edge of ends) expect(edge.to).toBe(END_ID);
  return new Set(ends.map((edge) => edge.from));
}

test("seed 01's multi-output terminals", () => {
  // output: {rating: ${score.output.rating}, verdict: ${verdict.output}}
  expect(terminalProducers(SEED_01)).toEqual(new Set(["score", "verdict"]));
});

test("seed 02's coalesce gives both producers a terminal edge", () => {
  // output: ${positive.output | cautious.output}: both get an END edge.
  expect(terminalProducers(SEED_02)).toEqual(new Set(["positive", "cautious"]));
});

test("seed 18's multi-output with a coalesce", () => {
  // stance/confidence from synth; note from pro_note | con_note | neutral_note.
  expect(terminalProducers(SEED_18)).toEqual(new Set(["synth", "pro_note", "con_note", "neutral_note"]));
});

test("seed 05's multi-output terminals", () => {
  // report from compare; briefs from research_each.
  expect(terminalProducers(SEED_05)).toEqual(new Set(["compare", "research_each"]));
});

test("terminal edge ids are unique", () => {
  const { nodeIds, edges, outputs } = allEdges(SEED_18);
  const ids = endEdges(outputs, nodeIds, edges).map((edge) => edge.id);
  expect(new Set(ids).size).toBe(ids.length);
});

// The assembled edge set starts at START and ends at END.

test("the assembled edges have START and END", () => {
  const { nodeIds, edges, outputs } = allEdges(SEED_02);
  const [, boundaryEdges] = synthesizeBoundaryGraph([], flowOutputs(outputs), nodeIds, edges, null, {
    record: isRecord(outputs),
  });
  const full = [...edges, ...boundaryEdges];
  expect(full.some((edge) => edge.from === START_ID)).toBe(true);
  expect(full.some((edge) => edge.to === END_ID)).toBe(true);
});

test("the boundary's record flag picks END's mode", () => {
  const one = [new FlowOutput("order", "${x.output}")];
  const endOf = (outputs: FlowOutput[], record: boolean) =>
    synthesizeBoundaryGraph([], outputs, new Set(["x"]), [], null, { record })[0].get(END_ID) as EndNode;

  expect(endOf(one, true).returnsRecord).toBe(true);
  expect(endOf(one, false).returnsRecord).toBe(false);
  expect(endOf([], false).returnsRecord).toBe(false);
  const two = [new FlowOutput("a", "${x.output}"), new FlowOutput("b", "${x.output}")];
  expect(() => endOf(two, false)).toThrow(new RangeError("EndNode value mode takes at most one output name"));
});
