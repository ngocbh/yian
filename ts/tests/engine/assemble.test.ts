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
 * Later (the conductor, in this item): the input representation guards
 * (`test_resolved_parameters_reject_both_input_representations`,
 * `test_resolved_parameters_reject_missing_input_representation`). Later (T3-17): the resolved
 * parameter boundary (`test_resolved_parameters_preserve_structural_contract_identity`,
 * `test_resolved_parameters_accept_zero_args_and_wire_empty_start`), the seed edge and boundary
 * rows (`test_seed01_*`, `test_seed02_*`, `test_seed05_*`, `test_seed06_*`, `test_seed18_*`,
 * `test_terminal_edge_ids_unique`, `test_assembled_edges_have_start_and_end`,
 * `test_boundary_end_record_flag_picks_the_mode`). Later (T3-18): the wiring parity rows
 * (`test_wiring_parity_*`).
 */

import { afterEach, describe, expect, test } from "bun:test";

import {
  assertionCursors,
  attachInputContracts,
  ComposedFlow,
  DescriptorBuildCache,
  flowOutputs,
  registerAssertionOrigin,
  splitTypedInputs,
  stampBindingLabels,
  stampFlowOutputOrigins,
  stampNodeWiringOrigins,
} from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildContext, BuildResult, type NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { flowNodesCursor, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { CompiledFlow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { forget } from "../../src/defaults.ts";
import {
  DiagnosticFrame,
  Failure,
  OriginRef,
  type OriginResolver,
  SourceRange,
} from "../../src/diagnostics.ts";
import { Binding, Parameter, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import type { TypeRegistry } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { caseRegistry, cursorAt, loadError, runTextCase } from "../compose/declarations.ts";
import { StandInCode, withKinds } from "../compose/stand_ins.ts";

afterEach(() => forget());

const EMPTY: TypeRegistry = new Map();

/** Return source text large enough for synthetic descriptor ranges. */
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
    expect(run).toThrow("node 'use': typed input 'qty' has no wiring binding");
    try {
      run();
    } catch (error) {
      expect(error).not.toBeInstanceOf(LoadError);
    }
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
      expect(run).toThrow(message);
      try {
        run();
      } catch (error) {
        expect(error).toBeInstanceOf(RangeError);
        expect(error).not.toBeInstanceOf(LoadError);
      }
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
