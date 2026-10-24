/**
 * Reference-wiring validation over every binding site of a built flow: a port of
 * `tests/engine/test_validate_refs.py`.
 *
 * `validateReferences` runs the reused leaf checker (`classifyPath`) and the shared field-read walk
 * over node input bindings, flow `output:` bindings and (through the desugared CASE wiring) each
 * `case`'s `on:`/`when:` data refs, and throws one `LoadError` carrying every located error. Two
 * loud mechanisms are pinned here: e01 (a dangling flow-output ref, `${scor.output}`) and e03 (a
 * field miss on an anonymous producer record). The rewritten node-local `__rN`/`__on` of a CASE are
 * never classified.
 *
 * Ported: `test_clean_seed01_passes`, `test_clean_case_seed02_passes`,
 * `test_clean_case_on_seed06_passes`, `test_wait_until_rejects_invalid_references_at_load` (4),
 * `test_wait_until_accepts_valid_reference_sources` (4),
 * `test_system_ambient_field_read_is_a_load_error` (3),
 * `test_system_ambient_field_read_in_case_when_is_a_load_error`,
 * `test_e01_dangling_flow_output_is_loud`, `test_dangling_flow_output_uses_supplied_range`,
 * `test_aggregate_location_uses_first_available_range`, `test_e03_unknown_field_on_anon_record_is_loud`,
 * `test_reference_walk_projects_type_without_replacing_contract_origins`,
 * `test_reference_walk_rejects_non_contract_producer_map_values`,
 * `test_prompt_l1_pool_ref_is_loud`, `test_prompt_l1_declared_input_passes`,
 * `test_case_dangling_data_ref_is_loud` and `test_case_node_local_ref_does_not_error`.
 *
 * Changed: the seeds are conformance cases (`load-agent/seed_01_structured_agent`,
 * `load-case/seed_02_case`, `load-case/seed_06_case_on`, `load-core/seed_e01_undeclared_ref`,
 * `load-types/seed_e03_unknown_field`), parsed with `nodesOf` and `fileOf` and built leaves first,
 * then cases, as Python's `_load` does, with an empty registry where Python's `_load` passes the
 * producer map in the `BuildContext` registry slot. The `load_flow` rows assemble with
 * `assembleText`. Python's `tests/engine/_fakes` `stamp_reads`/`derive_wiring` are the local
 * `withReads` and `wiringOf`. TS yian has no `wait`, so the two wait-until rows bind a `sleep`'s
 * `until` input. The inline CODE bodies are JavaScript (`String(inputs["t"])`). The
 * bad-producer row pins the whole `TypeError` message, and the pool-ref prompt row the whole
 * `LoadError` message.
 *
 * TS-only: the conformance cases of this file
 * (`load-core/engine_validate_refs__{system_run_id_field_read,wait_until_accepts_a_node_output,
 * wait_until_reads_an_unknown_flow_input,wait_until_reads_item_outside_a_map}`), each error with `full`.
 */

import { afterEach, expect, test } from "bun:test";

import { readSourceFlowInputs } from "../../src/compose/helpers/types.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { BuildContext, BuildResult } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { validateReferences } from "../../src/compose/validate/references.ts";
import { forget } from "../../src/defaults.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CaseNode } from "../../src/nodes/case/node.ts";
import { ContractOrigins, type OutputContract, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  caseText,
  fileOf,
  loadError,
  nodesOf,
  runCase,
} from "../compose/declarations.ts";

afterEach(() => forget());

/** A build context with no typedefs. */
const CTX = new BuildContext({ registry: new Map() });

/** The inputs of one direct `validateReferences` call. */
type Loaded = [
  buildResults: Map<string, BuildResult>,
  flowInputs: Set<string>,
  producers: Map<string, OutputContract>,
  outputs: unknown,
  flowWiring: Map<string, ReadonlyMap<string, unknown>>,
];

/**
 * Return the build products and validation inputs of conformance case `id`'s flow: every
 * descriptor built kind-blind, leaves first (to learn the producer output contracts), then the
 * cases, whose `__rN`/`__on` split is part of their own build.
 */
function load(id: string): Loaded {
  const { text, label } = caseText(id);
  const descriptors = nodesOf(text, [], label);
  const buildResults = new Map<string, BuildResult>();
  const flowWiring = new Map<string, ReadonlyMap<string, unknown>>();
  for (const [nodeId, desc] of descriptors) {
    if (desc instanceof CaseDescriptor) continue;
    const result = desc.build(CTX);
    buildResults.set(nodeId, result);
    flowWiring.set(nodeId, result.wiring);
  }
  const producers = new Map<string, OutputContract>();
  for (const [nodeId, result] of buildResults) {
    if (result.node.outputContract !== null) producers.set(nodeId, result.node.outputContract);
  }
  for (const [nodeId, desc] of descriptors) {
    if (!(desc instanceof CaseDescriptor)) continue;
    const result = desc.build(CTX);
    buildResults.set(nodeId, result);
    flowWiring.set(nodeId, result.wiring);
  }
  const flowInputs = new Set(
    readSourceFlowInputs(SourceDocument.parse(text, label).root.child("input"), new Map(), {
      sitePath: ["flow"],
      catalogBuilder: new SourceCatalogBuilder(),
    }).map((decl) => decl.name),
  );
  return [buildResults, flowInputs, producers, fileOf(text, label).outputs, flowWiring];
}

/** Return the load diagnostics of conformance case `id`'s flow. */
function caseDiagnostics(id: string): LoadDiagnostics {
  const { text, label } = caseText(id);
  return new LoadDiagnostics(label, text);
}

/** The sources `withReads` declared, by node. */
const wiringSources = new WeakMap<Node, Map<string, unknown>>();

/** Declare `node`'s inputs from `reads` (param to source), as the flow does; returns the node. */
function withReads(node: Node, reads: Readonly<Record<string, string>>): Node {
  node.params = Object.keys(reads).map((name) => new Parameter(name));
  wiringSources.set(node, new Map(Object.entries(reads)));
  return node;
}

/** Collect each node's declared reads into a flow wiring map; a node with none has `{}`. */
function wiringOf(nodes: ReadonlyMap<string, Node>): Map<string, ReadonlyMap<string, unknown>> {
  return new Map([...nodes].map(([nodeId, node]) => [nodeId, wiringSources.get(node) ?? new Map()]));
}

/** Wrap hand-built semantic nodes for a direct validator call. */
function sourceFree(nodes: ReadonlyMap<string, Node>): Map<string, BuildResult> {
  return new Map([...nodes].map(([nodeId, node]) => [nodeId, new BuildResult({ node, wiring: new Map() })]));
}

const STR = () => new ValueContract(Type.scalar(ValueKind.STRING));

// ---------- a clean flow passes ----------

test.each(["load-agent/seed_01_structured_agent", "load-case/seed_02_case", "load-case/seed_06_case_on"])(
  "clean seed %s passes",
  (id) => {
    validateReferences(...load(id));
  },
);

const SLEEP_REFERENCE_FLOW = `\
id: wait_refs
name: wait_refs
input:
  deadline: !datetime
nodes:
  clock:
    kind: code
    output: datetime
    code: |
      return "2026-01-01T00:00:00Z"
  settle:
    kind: tool
    tool_id: sleep
    input:
      until: SOURCE
output: \${settle.output}
`;

test.each([
  ["${missing.output}", "unknown namespace 'missing'"],
  ["${input.missing}", "unknown flow input 'missing'"],
  ["${system.tomorrow}", "unknown system ambient 'tomorrow'"],
  ["${item}", "unknown namespace 'item'"],
])("sleep until rejects the invalid reference %s at load", (source, message) => {
  const error = loadError(() => assembleText(SLEEP_REFERENCE_FLOW.replace("SOURCE", source)));

  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(15);
});

test.each(["2026-01-01T00:00:00Z", "${input.deadline}", "${clock.output}", "${system.now}"])(
  "sleep until accepts the valid source %s",
  (source) => {
    assembleText(SLEEP_REFERENCE_FLOW.replace("SOURCE", source));
  },
);

const SYSTEM_FIELD_FLOW = `\
id: system_fields
name: system_fields
nodes:
  read:
    kind: code
    input:
      t: SOURCE
    output: str
    code: |
      return String(inputs["t"])
output: \${read.output}
`;

test.each([
  ["${system.today.year}", "reads field 'year' on type date, which has no fields"],
  ["${system.now.hour}", "reads field 'hour' on type datetime, which has no fields"],
  ["${system.run_id.x.y}", "reads field 'x' on type str, which has no fields"],
])("a system ambient field read %s is a load error", (source, message) => {
  const error = loadError(() => assembleText(SYSTEM_FIELD_FLOW.replace("SOURCE", source)));

  expect(error.message).toContain(message);
});

test("a system ambient field read in a case when is a load error", () => {
  const flow = `\
id: system_case
name: system_case
nodes:
  route:
    kind: case
    cases:
      - when: system.today.year == "2026"
        then: done
    else: done
  done:
    kind: code
    output: int
    code: |
      return 1
output: \${done.output}
`;
  const error = loadError(() => assembleText(flow));

  expect(error.message).toContain("reads field 'year' on type date, which has no fields");
});

// ---------- e01: a dangling flow-output reference ----------

const E01 = "load-core/seed_e01_undeclared_ref";

test("e01: a dangling flow output is loud", () => {
  const error = loadError(() => validateReferences(...load(E01)));

  expect(error.message).toContain("scor"); // the typo'd target is named
  expect(error.message).toContain("flow output");
});

test("a dangling flow output uses the supplied range", () => {
  const [nodes, flowInputs, producers, outputs, flowWiring] = load(E01);
  const expected = new SourceRange(31, 4, 31, 24);

  const error = loadError(() =>
    validateReferences(nodes, flowInputs, producers, outputs, flowWiring, {
      outputsRange: expected,
      diagnostics: caseDiagnostics(E01),
    }),
  );

  expect(error.diagnostic.frames[0]!.range).toEqual(expected);
});

test("the aggregate location uses the first available range", () => {
  const node = withReads(new AgentNode("broken", { prompt: "fixed" }), { value: "${missing_node.output}" });
  node.outputContract = STR();
  const nodes = new Map([["broken", node]]);
  const laterOutputRange = new SourceRange(20, 9, 20, 33);

  const error = loadError(() =>
    validateReferences(sourceFree(nodes), new Set(), new Map(), "${missing_output.output}", wiringOf(nodes), {
      outputsRange: laterOutputRange,
      diagnostics: new LoadDiagnostics("flow.yaml", ("x".repeat(80) + "\n").repeat(30)),
    }),
  );

  expect(error.diagnostic.frames[0]!.range).toEqual(laterOutputRange);
});

// ---------- e03: a field miss on an anonymous producer record ----------

test("e03: an unknown field on an anonymous record is loud", () => {
  const loaded = load("load-types/seed_e03_unknown_field");
  // The anonymous `{rating, rationale}` producer is a checked record.
  const score = loaded[2].get("score") as ValueContract;
  expect([...score.type.fields!.keys()].sort()).toEqual(["rating", "rationale"]);

  const error = loadError(() => validateReferences(...loaded));

  expect(error.message).toContain("confidence"); // the unknown field is named
});

test("the reference walk projects the type without replacing the contract's origins", () => {
  // Reference checking reads the semantic type without consuming provenance.
  const outputType = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["rating", Type.scalar(ValueKind.NUMBER)]]),
    required: new Set(["rating"]),
  });
  const contract = new ValueContract(outputType, new ContractOrigins({ default: new OriginRef("producer-output") }));
  const score = new AgentNode("score", { prompt: "fixed" });
  const sink = withReads(new AgentNode("sink", { prompt: "fixed" }), { value: "${score.output.rating}" });
  const nodes = new Map<string, Node>([
    ["score", score],
    ["sink", sink],
  ]);

  validateReferences(sourceFree(nodes), new Set(), new Map([["score", contract]]), null, wiringOf(nodes));

  expect(contract.type).toBe(outputType);
  expect(contract.origins.default).toEqual(new OriginRef("producer-output"));
});

test("the reference walk rejects a producer map value that is not a contract", () => {
  // The validator's boundary takes the canonical carriers, not bare Types.
  expect(() =>
    validateReferences(
      new Map([["score", new BuildResult({ node: new AgentNode("score"), wiring: new Map() })]]),
      new Set(),
      new Map([["score", Type.scalar(ValueKind.STRING) as unknown as OutputContract]]),
      null,
      new Map([["score", new Map()]]),
    ),
  ).toThrow(
    new TypeError("producer contracts must be ValueContract or PartialRecordContract values, got Type for 'score'"),
  );
});

// ---------- a prompt may only interpolate bare declared inputs ----------

test("a prompt's pool ref is loud", () => {
  // An AGENT prompt that interpolates `${upstream.output}` (a pool ref) with no declared inputs.
  const node = withReads(new AgentNode("bad", { prompt: "Summarize ${upstream.output} for the user." }), {});
  node.outputContract = STR();
  const nodes = new Map([["bad", node]]);

  const error = loadError(() => validateReferences(sourceFree(nodes), new Set(), new Map(), null, wiringOf(nodes)));

  expect(error.message).toBe(
    "flow has unresolved references:\n  node 'bad' prompt: ${upstream.output} is not a declared input or variable",
  );
});

test("a prompt's declared input passes", () => {
  const node = withReads(new AgentNode("ok", { prompt: "Summarize ${topic} for the user." }), {
    topic: "${input.topic}",
  });
  node.outputContract = STR();
  const nodes = new Map([["ok", node]]);

  validateReferences(sourceFree(nodes), new Set(["topic"]), new Map(), null, wiringOf(nodes));
});

// ---------- case data refs, validated from the desugared CASE wiring ----------

test("a case's dangling data ref is loud", () => {
  // A searched case whose `when:` reads `${typo.output}` (no such node): the desugared `__r0`
  // source is the dangling ref, found on the flow wiring.
  const result = new CaseDescriptor({
    id: "gate",
    cases: [{ when: "typo.output >= 0.5", then: "yes" }],
    else_: "no",
  }).build(CTX);

  const error = loadError(() =>
    validateReferences(new Map([["gate", result]]), new Set(), new Map(), null, new Map([["gate", result.wiring]])),
  );

  expect(error.message).toContain("typo"); // the dangling producer is named
});

test("a case's node-local ref does not error", () => {
  // The rewritten node-local `__r0` is never classified: a case whose source is a real node passes.
  const loaded = load("load-case/seed_02_case");
  // Seed 02's gate: param `__r0` wired to `${score.output}` (a real node); its `when:` reads `__r0`.
  const gate = loaded[0].get("gate")!.node as CaseNode;
  expect(gate.params!.map((param) => param.name)).toEqual(["__r0"]);
  expect(loaded[4].get("gate")).toEqual(new Map([["__r0", "${score.output}"]]));
  expect(gate.cases[0]!.when).toBe("__r0 >= 0.5"); // node-local: must not be classified

  validateReferences(...loaded);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-core/engine_validate_refs__system_run_id_field_read", call: assembled, full: true },
  { id: "load-core/engine_validate_refs__wait_until_accepts_a_node_output", call: assembled },
  { id: "load-core/engine_validate_refs__wait_until_reads_an_unknown_flow_input", call: assembled, full: true },
  { id: "load-core/engine_validate_refs__wait_until_reads_item_outside_a_map", call: assembled, full: true },
])("assemble: $id", runCase);
