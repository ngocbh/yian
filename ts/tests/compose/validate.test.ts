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
 * `test_reject_cycles_ignores_synth_out_edges_outside_the_component`); and the parse rows
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
 * TS-only: the harvested cases from files with no TS mirror, pinned at the end, with their notes
 * and every frame: `load-map-call/compose_call_argument_labels__a_cycle_through_an_inline_call_names_its_host`
 * (its `defs:` entry built by hand).
 *
 * Later (T3-18): the reference validator's rows (`test_validate_references_diag_has_range`,
 * `test_validate_references_diag_locates_case_when_via_synth_param`,
 * `test_validate_references_diag_case_on_form_locates_at_on_line`,
 * `test_validate_references_diag_case_when_multi_ref_locates_at_that_when`,
 * `test_validate_references_diag_case_then_call_locates_at_then_line`),
 * `test_check_wiring_parity_moved` and `test_validate_human_questions_diag_has_range`.
 * Later (T3-19): `test_check_call_map_types_is_owned_by_call_validation`,
 * `test_check_loop_type_contract_moved`,
 * `test_check_loop_type_contract_tolerates_a_loop_without_child_inputs` and
 * `test_type_validators_reject_noncanonical_contract_maps_before_walking`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { Edge } from "../../src/compile/model.ts";
import { LoadDiagnostics, type LoadError } from "../../src/compose/helpers/errors.ts";
import { siteKey } from "../../src/compose/nodes/base.ts";
import { rejectCycles } from "../../src/compose/validate/cycles.ts";
import { forget } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembledWith,
  assembleText,
  caseText,
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

/** Return the frame ranges of `run`'s `LoadError`. */
const frameRanges = (run: () => unknown) => loadError(run).diagnostic.frames.map((frame) => frame.range);

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
    const run = () =>
      rejectCycles(aliasCycleEdges(), new Set(["g", "c"]), {
        nodeRanges: new Map([["c", synth]]),
        diagnostics: rangeDiagnostics(),
        ...aliasOptions,
      });
    expect(loadError(run).message).toBe('flow has a cycle involving ["g"]; flows must be acyclic');
    expect(frameRanges(run)).toEqual([synth]);
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

// ---------- harvested cases from files with no TS mirror ----------

/** The `child` def of the inline-call cycle case: `y: !str` into `x`, whose output it returns. */
const child = () =>
  handChild({
    inputs: parametersFromYaml("input:\n  y: !str\n", new Map()),
    producers: { x: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: "${x.output}",
  });

const INLINE_CALL_CYCLE = "load-map-call/compose_call_argument_labels__a_cycle_through_an_inline_call_names_its_host";

/** Return each frame's range as `[startLine, startCol, endLine, endCol]`. */
const frameSpans = (error: LoadError) =>
  error.diagnostic.frames.map(({ range }) => [range.startLine, range.startCol, range.endLine, range.endCol]);

test.each([{ id: INLINE_CALL_CYCLE, call: assembledWith({ child: child() }) }])("assemble: $id", runCase);

test("the inline-call cycle names its host in a note and frames both nodes", () => {
  const text = caseText(INLINE_CALL_CYCLE).text;
  const error = loadError(() => assembleText(text, "flow.yaml", { children: { child: child() } }));
  expect(frameSpans(error)).toEqual([
    [19, 3, 19, 4],
    [23, 10, 23, 36],
  ]);
  expect(error.diagnostic.notes).toEqual([
    "__call_0 depends on n (__call_0.input.y)",
    "__call_0 is generated from node 'n'",
    "n depends on __call_0 (n.input.a)",
  ]);
});
