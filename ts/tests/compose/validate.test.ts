/**
 * The validate phase's cross-node checks over the assembled flow: a port of
 * `tests/compose/test_validate.py`.
 *
 * Ported: the `rejectCycles` rows on hand-built edges (`test_reject_cycles_diag_has_range`,
 * `test_reject_cycles_no_ranges_leaves_range_none`,
 * `test_reject_cycles_excludes_nodes_only_blocked_downstream`,
 * `test_reject_cycles_omits_edges_between_separate_cycles`,
 * `test_reject_cycles_reports_a_self_dependency`,
 * `test_reject_cycles_names_a_gate_edge_by_its_host_input`,
 * `test_reject_cycles_keeps_a_dotted_group_that_is_not_a_gate`,
 * `test_reject_cycles_reports_an_alias_as_its_host`,
 * `test_reject_cycles_frames_an_unlocated_host_at_its_synth_node`,
 * `test_reject_cycles_alias_keeps_other_members`,
 * `test_reject_cycles_keeps_a_synth_node_read_inside_the_cycle`,
 * `test_reject_cycles_ignores_synth_out_edges_outside_the_component`);
 * `test_check_wiring_parity_moved` and `test_validate_human_questions_diag_has_range`;
 * `test_check_call_map_types_is_owned_by_call_validation`, `test_check_loop_type_contract_moved`,
 * `test_check_loop_type_contract_tolerates_a_loop_without_child_inputs` and
 * `test_type_validators_reject_noncanonical_contract_maps_before_walking` (direct calls over
 * hand-built `BuildResult`s); and the parse rows
 * `test_node_local_asserts_carry_ranges`, `test_per_input_param_carries_ranges`,
 * `test_input_named_like_field_no_collision` (through `nodesOf`),
 * `test_unknown_top_level_key_lists_authored_names` (through `fileOf`) and
 * `test_field_not_allowed_lists_authored_names` (through `nodesOf`), where Python calls
 * `load_flow`.
 *
 * Changed: `test_reject_cycles_range_map_is_keyword_only`: TS takes one options object, so there
 * is no positional form to refuse. The cycle message writes its node list in the text form
 * (`["a"]`, rule `messages`). The CODE references are TS's
 * (`./fixtures/compose_codefns.ts:pick_pro`).
 *
 * TS-only: `test_a_case_output_over_a_1000_link_alias_chain_is_rejected_as_a_case_output` from
 * `tests/compose/test_record_type_nesting.py`, which has no TS mirror yet (its message pinned; a
 * stack overflow would give another error, so there is no cause-chain check); and the harvested
 * cases from files with no TS mirror, pinned at the end with `full` (every
 * frame and the notes): `load-map-call/compose_call_argument_labels__a_cycle_through_an_inline_call_names_its_host`
 * (its `defs:` entry built by hand),
 * `load-core/engine_limit_overrides__written_value_over_the_set_value_limit` (its `env:` sets the
 * value limit, restored by the `afterEach`), and the type-table pass's
 * `load-core/engine_binding_labels_flows__unknown_type_in_a_single_flow_output` and
 * `load-core/compose_field_read_sites__loop_while_reads_a_field_on_a_builtin_result` (its `body`
 * def built by hand).
 *
 * Later (T3-18 PR 3): the reference validator's rows (`test_validate_references_diag_has_range`,
 * `test_validate_references_diag_locates_case_when_via_synth_param`,
 * `test_validate_references_diag_case_on_form_locates_at_on_line`,
 * `test_validate_references_diag_case_when_multi_ref_locates_at_that_when`,
 * `test_validate_references_diag_case_then_call_locates_at_then_line`).
 */

import { afterEach, describe, expect, test } from "bun:test";

import { Edge } from "../../src/compile/model.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { BuildContext, BuildResult, siteKey } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { checkCallMapTypes } from "../../src/compose/validate/call_types.ts";
import { rejectCycles } from "../../src/compose/validate/cycles.ts";
import { validateHumanQuestions } from "../../src/compose/validate/human_questions.ts";
import { checkLoopTypeContract } from "../../src/compose/validate/loop_types.ts";
import { checkWiringParity } from "../../src/compose/validate/wiring_parity.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { HumanInputNode } from "../../src/nodes/human_input/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  fileOf,
  handChild,
  loadError,
  nodesOf,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

afterEach(() => forget());

/** Return source text large enough for the synthetic ranges of the direct checks. */
const rangeDiagnostics = () => new LoadDiagnostics("flow.yaml", ("x".repeat(80) + "\n").repeat(100));

/** Return a data edge `from -> to` feeding input group `group`, with id `from->to`. */
const dataEdge = (from: string, to: string, group: string | null = null) =>
  new Edge(`${from}->${to}`, from, to, { inputGroup: group });

describe("rejectCycles", () => {
  test("a cycle's diagnostic has a frame per node in authored order", () => {
    // Every stuck node gets one frame in authored source order, independent of node-id order.
    const ra = new SourceRange(9, 3, 9, 8);
    const rb = new SourceRange(3, 3, 3, 8);
    const error = loadError(() =>
      rejectCycles([dataEdge("a", "b", "x"), dataEdge("b", "a", "y")], new Set(["a", "b"]), {
        nodeRanges: new Map([
          ["a", ra],
          ["b", rb],
        ]),
        diagnostics: rangeDiagnostics(),
      }),
    );
    expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([rb, ra]);
    expect(error.diagnostic.notes.length).toBeGreaterThan(0);
  });

  test("with no range map the cycle is still rejected, with no frames", () => {
    const error = loadError(() => rejectCycles([dataEdge("a", "b"), dataEdge("b", "a")], new Set(["a", "b"])));
    expect(error.message).toBe('flow has a cycle involving ["a","b"]; flows must be acyclic');
    expect(error.diagnostic.frames).toEqual([]);
  });

  test("nodes only blocked downstream are not named", () => {
    const ranges = new Map([
      ["tail", new SourceRange(1, 1, 1, 5)],
      ["b", new SourceRange(3, 1, 3, 2)],
      ["a", new SourceRange(5, 1, 5, 2)],
      ["end", new SourceRange(7, 1, 7, 4)],
    ]);
    const edges = [
      dataEdge("a", "b", "from_a"),
      dataEdge("b", "a", "from_b"),
      dataEdge("b", "tail", "from_b"),
      dataEdge("tail", "end", "from_tail"),
    ];
    const error = loadError(() =>
      rejectCycles(edges, new Set(ranges.keys()), { nodeRanges: ranges, diagnostics: rangeDiagnostics() }),
    );
    expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([ranges.get("b")!, ranges.get("a")!]);
    expect(error.message).not.toContain("tail");
    expect(error.diagnostic.notes.every((note) => !note.includes("tail"))).toBe(true);
  });

  test("an edge between two separate cycles is left out of the notes", () => {
    const edges = [
      dataEdge("a", "b", "from_a"),
      dataEdge("b", "a", "from_b"),
      dataEdge("b", "c", "bridge"),
      dataEdge("c", "d", "from_c"),
      dataEdge("d", "c", "from_d"),
    ];
    const error = loadError(() => rejectCycles(edges, new Set(["a", "b", "c", "d"])));
    expect(error.diagnostic.notes).toEqual([
      "a depends on b (a.input.from_b)",
      "b depends on a (b.input.from_a)",
      "c depends on d (c.input.from_d)",
      "d depends on c (d.input.from_c)",
    ]);
  });

  test("a self-dependency is a cycle", () => {
    const error = loadError(() => rejectCycles([dataEdge("a", "a", "self")], new Set(["a"])));
    expect(error.message).toBe('flow has a cycle involving ["a"]; flows must be acyclic');
    expect(error.diagnostic.notes).toEqual(["a depends on a (a.input.self)"]);
  });

  test("a gate edge is named by its host input", () => {
    const edges = [
      new Edge("spy->c#0", "spy", "c", { inputGroup: "publish.notes" }),
      new Edge("c->spy#0", "c", "spy", { inputGroup: "x" }),
    ];
    const error = loadError(() => rejectCycles(edges, new Set(["c", "spy"]), { gateIds: new Set(["spy->c#0"]) }));
    expect(error.diagnostic.notes).toEqual([
      "c depends on spy (gate: publish.input.notes)",
      "spy depends on c (spy.input.x)",
    ]);
  });

  test("a dotted group that is not a gate is kept whole", () => {
    const edges = [
      new Edge("spy->c#0", "spy", "c", { inputGroup: "k.v" }),
      new Edge("c->spy#0", "c", "spy", { inputGroup: "x" }),
    ];
    const error = loadError(() => rejectCycles(edges, new Set(["c", "spy"])));
    expect(error.diagnostic.notes).toContain("c depends on spy (c.input.k.v)");
  });

  /** A gate `g` reading its own output, plus its synth node `c` wired into it. */
  const aliasCycleEdges = () => [dataEdge("c", "g", "__questions"), dataEdge("g", "g", "proj"), dataEdge("g", "c", "proj")];
  const aliasOptions = { synthHosts: new Map([["c", "g"]]), synthWires: new Set(["c->g"]) };

  test("an alias is reported as its host", () => {
    const same = new SourceRange(5, 3, 5, 6);
    const error = loadError(() =>
      rejectCycles(aliasCycleEdges(), new Set(["g", "c"]), {
        nodeRanges: new Map([
          ["g", same],
          ["c", same],
        ]),
        diagnostics: rangeDiagnostics(),
        ...aliasOptions,
      }),
    );
    expect(error.message).toBe('flow has a cycle involving ["g"]; flows must be acyclic');
    expect(error.diagnostic.frames.length).toBe(1);
    expect(error.diagnostic.notes).toEqual(["g depends on g (g.input.proj)"]);
  });

  test("an unlocated host is framed at its synth node", () => {
    const synth = new SourceRange(7, 3, 7, 6);
    const error = loadError(() =>
      rejectCycles(aliasCycleEdges(), new Set(["g", "c"]), {
        nodeRanges: new Map([["c", synth]]),
        diagnostics: rangeDiagnostics(),
        ...aliasOptions,
      }),
    );
    expect(error.message).toBe('flow has a cycle involving ["g"]; flows must be acyclic');
    expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([synth]);
  });

  test("an alias keeps the other members", () => {
    const edges = [
      dataEdge("b", "g", "proj"),
      dataEdge("b", "c", "proj"),
      dataEdge("c", "g", "__questions"),
      dataEdge("g", "b", "a"),
    ];
    const ranges = new Map([
      ["b", new SourceRange(5, 3, 5, 4)],
      ["g", new SourceRange(9, 3, 9, 4)],
      ["c", new SourceRange(9, 3, 9, 4)],
    ]);
    const error = loadError(() =>
      rejectCycles(edges, new Set(ranges.keys()), {
        nodeRanges: ranges,
        diagnostics: rangeDiagnostics(),
        ...aliasOptions,
      }),
    );
    expect(error.message).toBe('flow has a cycle involving ["b","g"]; flows must be acyclic');
    expect(error.diagnostic.frames.map((frame) => frame.range.startLine)).toEqual([5, 9]);
    expect(error.diagnostic.notes).toEqual(["b depends on g (b.input.a)", "g depends on b (g.input.proj)"]);
  });

  test("a synth node read inside the cycle keeps its id", () => {
    const edges = [dataEdge("c", "b", "a"), dataEdge("b", "c", "proj"), dataEdge("c", "g", "__questions")];
    const error = loadError(() =>
      rejectCycles(edges, new Set(["b", "c", "g"]), {
        synthHosts: new Map([
          ["c", "g"],
          ["z", "y"],
        ]),
        synthWires: new Set(["c->g"]),
      }),
    );
    expect(error.message).toBe('flow has a cycle involving ["b","c"]; flows must be acyclic');
    expect(error.diagnostic.notes).toContain("c is generated from node 'g'");
    expect(error.diagnostic.notes.every((note) => !note.includes("z"))).toBe(true);
  });

  test("a synth node's out-edges outside the component are ignored", () => {
    const edges = [...aliasCycleEdges(), dataEdge("c", "log", "q")];
    const error = loadError(() => rejectCycles(edges, new Set(["g", "c", "log"]), aliasOptions));
    expect(error.message).toBe('flow has a cycle involving ["g"]; flows must be acyclic');
  });
});

describe("located pass errors", () => {
  test("checkWiringParity anchors a parity failure at the input: mapping", () => {
    // An orphan wiring key with no matching param fails parity; the error anchors on the node's
    // `input:` mapping span (else its node range), and the frame carries that range.
    const { node, wiring } = new CodeDescriptor({ id: "n", code: "m:f", inputs: { x: "${input.x}" } }).build(
      new BuildContext({ registry: new Map() }),
    );
    const range = new SourceRange(3, 5, 4, 20);
    const result = new BuildResult({ node, wiring, fieldRanges: new Map([[siteKey("field", "inputs"), range]]) });
    const error = loadError(() =>
      checkWiringParity(new Map([["n", result]]), new Map([["n", new Map([...wiring, ["orphan", "${input.y}"]])]]), {
        diagnostics: rangeDiagnostics(),
      }),
    );
    expect(error.message).toContain("key parity");
    expect(error.diagnostic.frames[0]!.range).toEqual(range);
  });

  test("validateHumanQuestions anchors an invalid literal list at the questions: field", () => {
    // An empty list is refused ("expected 1..4 questions, got 0"); the frame carries the gate's
    // `questions:` field span.
    const range = new SourceRange(5, 5, 6, 20);
    const result = new BuildResult({
      node: new HumanInputNode("ask", { questions: [] }),
      wiring: new Map(),
      fieldRanges: new Map([[siteKey("field", "questions"), range]]),
    });
    const error = loadError(() =>
      validateHumanQuestions(new Map([["ask", result]]), { diagnostics: rangeDiagnostics() }),
    );
    expect(error.message).toContain("questions");
    expect(error.diagnostic.frames[0]!.range).toEqual(range);
  });
});

describe("checkCallMapTypes", () => {
  // Python's test_check_call_map_types_is_owned_by_call_validation.
  test("the call type check is owned by call validation", () => {
    // A `call` binding whose source type mismatches the child input's declared type is loud. The
    // error anchors on the call node's `input:` mapping span (the coarsest binding-site span),
    // and the frame carries that range.
    const node = new CallNode("c", {
      flowId: "child",
      childInputs: [
        new Parameter("n", {
          contract: new ValueContract(Type.scalar(ValueKind.INTEGER)),
          required: true,
          coerceStrings: true,
        }),
      ],
    });
    const range = new SourceRange(7, 5, 8, 20);
    const result = new BuildResult({
      node,
      wiring: new Map([["n", "${src.output}"]]),
      fieldRanges: new Map([[siteKey("field", "inputs"), range]]),
    });
    const producers = new Map([["src", new ValueContract(Type.scalar(ValueKind.STRING))]]); // a string source
    const wiring = new Map([["c", new Map([["n", "${src.output}"]])]]); // bound to the int input

    const error = loadError(() =>
      checkCallMapTypes(new Map([["c", result]]), producers, new Map(), wiring, { diagnostics: rangeDiagnostics() }),
    );

    expect(error.message).toContain("child expects");
    expect(error.diagnostic.frames[0]!.range).toEqual(range);
  });
});

describe("checkLoopTypeContract", () => {
  /** A LOOP node `lp` over body `body` whose carried record is `{n: int}`. */
  function intLoop(): LoopNode {
    const node = new LoopNode("lp", { flowId: "body" });
    node.outputContract = new ValueContract(
      new Type({ kind: ValueKind.OBJECT, fields: new Map([["n", Type.scalar(ValueKind.INTEGER)]]) }),
    );
    return node;
  }

  test("the loop type check moved", () => {
    // test_check_loop_type_contract_moved: a carried field seeded from a source type that
    // mismatches the body output's same-named field type is loud, anchored on the loop node's
    // `input:` mapping span, and the frame carries that range.
    const range = new SourceRange(4, 5, 5, 20);
    const result = new BuildResult({
      node: intLoop(),
      wiring: new Map([["n", "${src.output}"]]),
      fieldRanges: new Map([[siteKey("field", "inputs"), range]]),
    });
    const producers = new Map([["src", new ValueContract(Type.scalar(ValueKind.STRING))]]); // a string seed source
    const wiring = new Map([["lp", new Map([["n", "${src.output}"]])]]); // carried `n` seeded from it

    const error = loadError(() =>
      checkLoopTypeContract(new Map([["lp", result]]), producers, new Map(), wiring, {
        diagnostics: rangeDiagnostics(),
      }),
    );

    expect(error.message).toContain("carried field");
    expect(error.diagnostic.frames[0]!.range).toEqual(range);
  });

  test("the loop type check tolerates a loop without child inputs", () => {
    // test_check_loop_type_contract_tolerates_a_loop_without_child_inputs: a bare LoopNode has no
    // body signature; a seed that matches the body output leaves nothing else to compare.
    const result = new BuildResult({ node: intLoop(), wiring: new Map([["n", "${src.output}"]]) });
    const producers = new Map([["src", new ValueContract(Type.scalar(ValueKind.INTEGER))]]);

    checkLoopTypeContract(
      new Map([["lp", result]]),
      producers,
      new Map(),
      new Map([["lp", new Map([["n", "${src.output}"]])]]),
      { diagnostics: rangeDiagnostics() },
    );
  });
});

test("the type validators reject noncanonical contract maps before walking", () => {
  // test_type_validators_reject_noncanonical_contract_maps_before_walking: direct validator calls
  // enforce the same exact-carrier boundary as the pipeline.
  const bad = new Map([["unused", Type.scalar(ValueKind.STRING)]]) as unknown as Map<string, ValueContract>;

  expect(() => checkCallMapTypes(new Map(), bad, new Map(), new Map())).toThrow(
    "output producer contracts must be ValueContract or PartialRecordContract values, got Type for 'unused'",
  );
  expect(() => checkLoopTypeContract(new Map(), new Map(), bad, new Map())).toThrow(
    "flow input contracts must be exact ValueContract values, got Type for 'unused'",
  );
});

describe("parse ranges", () => {
  test("node-local asserts carry ranges", () => {
    // Each authored assert maps to its own range by index, never by its (non-unique) text; the
    // coarse `asserts:` list range stays in `fieldRanges`.
    const flow = `\
id: f
name: f
nodes:
  score:
    kind: agent
    prompt: "Rate \${topic}"
    input:
      topic: \${input.topic}
    output: int
    asserts:
      - output >= 0
      - output <= 10
output: \${score.output}
`;
    const desc = nodesOf(flow).get("score")!;
    expect(desc.assertRanges.map((range) => range!.startLine)).toEqual([11, 12]);
    expect(desc.fieldRanges.has(siteKey("field", "asserts"))).toBe(true);
  });

  test("each input param carries its range", () => {
    const flow = `\
id: f
name: f
nodes:
  report:
    kind: agent
    prompt: "About \${topic} at \${as_of}"
    input:
      topic: \${input.topic}
      as_of: \${input.as_of:?"as_of is required"}
    output: str
output: \${report.output}
`;
    const fr = nodesOf(flow).get("report")!.fieldRanges;
    expect(fr.get(siteKey("input", "topic"))!.startLine).toBe(8);
    expect(fr.get(siteKey("input", "as_of"))!.startLine).toBe(9);
    expect(fr.has(siteKey("field", "inputs"))).toBe(true);
  });

  test("an input named like a field does not collide", () => {
    // An input named `env` and the node's `env:` field keep separate ranges.
    const flow = `\
id: f
name: f
nodes:
  n:
    kind: code
    env:
      retries: 2
    input:
      env: \${input.data}
    output: int
    code: ./fixtures/compose_codefns.ts:pick_pro
input:
  data: !str
output: \${n.output}
`;
    const fr = nodesOf(flow).get("n")!.fieldRanges;
    expect(fr.get(siteKey("input", "env"))!.startLine).toBe(9);
    expect(fr.get(siteKey("field", "env"))!.startLine).toBe(7);
    expect(fr.get(siteKey("input", "env"))).not.toEqual(fr.get(siteKey("field", "env")));
  });

  /** Return the names in a message's `(allowed: a, b, ...)` list. */
  const allowedList = (message: string) => message.split("(allowed: ", 2)[1]!.split(")", 1)[0]!.split(", ");

  test("an unknown top-level key lists the authored names", () => {
    // The allowed list names the authored `input`/`output`, not the retired plural spellings.
    const flow = `\
id: f
name: f
bogus: x
nodes:
  n:
    kind: code
    output: int
    code: ./fixtures/compose_codefns.ts:pick_pro
output: \${n.output}
`;
    const allowed = allowedList(loadError(() => fileOf(flow, "flow.yaml")).message);
    expect(allowed).toContain("input");
    expect(allowed).toContain("output");
    expect(allowed).not.toContain("inputs");
    expect(allowed).not.toContain("outputs");
  });

  test("a field not allowed lists the authored names", () => {
    const flow = `\
id: f
name: f
nodes:
  calc:
    kind: code
    prompt: hi
    output: int
    code: ./fixtures/compose_codefns.ts:pick_pro
output: \${calc.output}
`;
    const message = loadError(() => nodesOf(flow)).message;
    expect(message).toContain("field 'prompt' is not allowed");
    const allowed = allowedList(message);
    expect(allowed).toContain("input");
    expect(allowed).toContain("output");
    expect(allowed).not.toContain("inputs");
    expect(allowed).not.toContain("outputs");
  });
});

// ---------- rows from files with no TS mirror ----------

test("a case output over a 1000-link alias chain is rejected as a case output", () => {
  // From tests/compose/test_record_type_nesting.py. A listed chain's anchors cost quadratically
  // many nodes, past the default node limit, so the limit is raised to reach the case check.
  install({ ...BUILTIN, maxYamlNodes: 10_000_000 });
  const items = ['&m0 {k: "str"}', ...Array.from({ length: 998 }, (_, i) => `&m${i + 1} {k: *m${i}}`)];
  const text =
    `x-anchors: [${items.join(", ")}]\n` +
    "id: f\nname: f\ninput:\n  cmd: !str\nnodes:\n  route:\n    kind: case\n" +
    "    on: ${input.cmd}\n    output: {k: *m998}\n    cases:\n      - when: a\n" +
    "        then: reply\n    else: reply\n  reply:\n" +
    "    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n" +
    "    input:\n      topic: replied\n    output: str\n" +
    "output: ${route.output}\n";

  expect(loadError(() => assembleText(text)).message).toBe(
    "case node 'route': a `case` routes and has no `output:`; tag its result where " +
      "you read it, e.g. `x: !<type> ${route.output}`",
  );
});

// ---------- harvested cases from files with no TS mirror ----------

/** The `child` def of the inline-call cycle case: `y: !str` into `x`, whose output it returns. */
const child = () =>
  handChild({
    inputs: parametersFromYaml("input:\n  y: !str\n", new Map()),
    producers: { x: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: "${x.output}",
  });

/** The `body` def of the field-read case: `st: !Rec3` into `bump`, whose `{st: Rec3}` it returns. */
const body = () => {
  const registry = readTypedefs({ Rec3: { a: "str", b: "Optional(str)" }, Out: { st: "Rec3" } });
  return handChild({
    inputs: parametersFromYaml("input:\n  st: !Rec3\n", registry),
    producers: { bump: new ValueContract(typeFor("Out", registry)) },
    output: "${bump.output}",
  });
};

test.each<CaseRow>([
  {
    id: "load-map-call/compose_call_argument_labels__a_cycle_through_an_inline_call_names_its_host",
    call: assembledWith({ child: child() }),
    full: true,
  },
  {
    id: "load-core/engine_limit_overrides__written_value_over_the_set_value_limit",
    call: assembled,
    full: true,
  },
  {
    id: "load-core/engine_binding_labels_flows__unknown_type_in_a_single_flow_output",
    call: assembled,
    full: true,
  },
  {
    id: "load-core/compose_field_read_sites__loop_while_reads_a_field_on_a_builtin_result",
    call: assembledWith({ body: body() }),
    full: true,
  },
])("assemble: $id", runCase);
