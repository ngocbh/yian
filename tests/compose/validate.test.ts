/** The validate phase's cross-node checks over the assembled flow. */

import { afterEach, describe, expect, test } from "bun:test";

import { Edge } from "../../src/compile/model.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { BuildContext, BuildResult, siteKey } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { checkCallMapTypes } from "../../src/compose/validate/call_types.ts";
import { rejectCycles } from "../../src/compose/validate/cycles.ts";
import { checkLoopTypeContract } from "../../src/compose/validate/loop_types.ts";
import { validateReferences } from "../../src/compose/validate/references.ts";
import { checkWiringParity } from "../../src/compose/validate/wiring_parity.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { type OutputContract, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  fileOf,
  loadError,
  nodesOf,
  runCase,
} from "../helpers/declarations.ts";

afterEach(() => forget());

/** Return source text large enough for the synthetic ranges of the direct checks. */
const rangeDiagnostics = () => new LoadDiagnostics("flow.yaml", ("x".repeat(80) + "\n").repeat(100));

/** Return a data edge `from -> to` feeding input group `group`, with id `from->to`. */
const dataEdge = (from: string, to: string, group: string | null = null) =>
  new Edge(`${from}->${to}`, from, to, { inputGroup: group });

/**
 * Build the agent and code leaves of flow `text`'s `nodes:` with their wiring and output contracts,
 * enough to drive `validateReferences` alone.
 */
function builtFlow(text: string): {
  buildResults: Map<string, BuildResult>;
  wiring: Map<string, ReadonlyMap<string, unknown>>;
  producers: Map<string, OutputContract>;
} {
  const ctx = new BuildContext({ registry: new Map() });
  const buildResults = new Map(
    [...nodesOf(text)]
      .filter(([, desc]) => desc instanceof AgentDescriptor || desc instanceof CodeDescriptor)
      .map(([nodeId, desc]) => [nodeId, desc.build(ctx)]),
  );
  const wiring = new Map([...buildResults].map(([nodeId, result]) => [nodeId, result.wiring]));
  const producers = new Map<string, OutputContract>();
  for (const [nodeId, result] of buildResults) {
    if (result.node.outputContract !== null) producers.set(nodeId, result.node.outputContract);
  }
  return { buildResults, wiring, producers };
}

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
});

describe("checkCallMapTypes", () => {
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

describe("validateReferences", () => {
  test("a dangling node input is located at the offending entry", () => {
    // A node input binding that references an unknown node (`${ghost.output}`) is a dangling ref.
    // The aggregate error anchors on the OFFENDING ENTRY, the per-binding `input` span the parse
    // loop stamps for every kind, not the enclosing `input:` mapping, which would also mark every
    // innocent sibling entry.
    const text =
      "id: f\nname: f\nnodes:\n" +
      "  a:\n    kind: agent\n    prompt: hi\n" +
      "    input:\n      x: ${ghost.output}\n    output: str\n" +
      "output: ${a.output}\n";
    const { buildResults, wiring, producers } = builtFlow(text);

    const error = loadError(() =>
      validateReferences(buildResults, new Set(), producers, "${a.output}", wiring, {
        diagnostics: new LoadDiagnostics("flow.yaml", text),
      }),
    );

    expect(error.message).toContain("ghost");
    const range = error.diagnostic.frames[0]!.range;
    const fieldRanges = buildResults.get("a")!.fieldRanges;
    expect(range).toEqual(fieldRanges.get(siteKey("input", "x"))!);
    // The exact `x:` entry is narrower than the `input:` mapping, which runs through line 9.
    expect(range).not.toEqual(fieldRanges.get(siteKey("field", "inputs"))!);
    expect([range.startLine, range.endLine]).toEqual([8, 8]);
  });

  test("a case when is located through its synthetic param", () => {
    // A desugared CASE binds synthetic params (`__rN`, the hoisted `when:` refs) the author never
    // wrote. The desugar stamps each with its own `cases[N].when` range and label, so a bad ref
    // there locates at the `when:` line and names the author's construct, not `__rN`.
    const text =
      "id: f\nname: f\nnodes:\n" +
      "  a:\n    kind: case\n" +
      '    cases:\n      - when: "ghost.output >= 1"\n        then: b\n' + // `when:` on line 7
      "    else: b\n" +
      "  b:\n    kind: code\n    output: int\n    code: \"return {'output': 1}\"\n" +
      "output: ${b.output}\n";

    const error = loadError(() => assembleText(text));

    expect(error.message).toContain("ghost");
    expect(error.message).not.toContain("'__r0'"); // the synthetic param does not leak...
    expect(error.message).toContain("cases[0].when"); // ...it is named by its author-facing site
    expect(error.diagnostic.frames[0]!.range.startLine).toBe(7);
  });

  test("a case on: is located at the on: line", () => {
    // The `on:` form binds one synthetic `__on` param, named `on:` and located at the `on:` line.
    const text =
      "id: f\nname: f\nnodes:\n" +
      "  gate:\n    kind: case\n" +
      "    on: ${nope.output}\n" + // `on:` on line 6
      "    cases:\n      - when: hi\n        then: p\n" +
      "    else: p\n" +
      "  p:\n    kind: code\n    output: int\n    code: \"return {'output': 1}\"\n" +
      "output: ${p.output}\n";

    const error = loadError(() => assembleText(text));

    expect(error.message).toContain("nope");
    expect(error.message).not.toContain("__on");
    expect(error.message).toContain("on:");
    expect(error.diagnostic.frames[0]!.range.startLine).toBe(6);
  });

  test("a when with two refs is located at that when", () => {
    // A `when:` reading two refs (one valid, one bad) allocates a `__rN` per distinct ref; the bad
    // ref carets this `when:` line and names `cases[0].when`.
    const text =
      "id: f\nname: f\nnodes:\n" +
      "  s:\n    kind: code\n    output: int\n    code: \"return {'output': 1}\"\n" +
      "  gate:\n    kind: case\n" +
      '    cases:\n      - when: "s.output >= 1 and nope.output >= 1"\n        then: p\n' + // line 11
      "    else: p\n" +
      "  p:\n    kind: code\n    output: int\n    code: \"return {'output': 1}\"\n" +
      "output: ${p.output}\n";

    const error = loadError(() => assembleText(text));

    expect(error.message).toContain("nope");
    expect(error.message).toContain("cases[0].when");
    expect(error.message).not.toContain("__r");
    expect(error.diagnostic.frames[0]!.range.startLine).toBe(11);
  });

  test("a then call is located at the then: line", () => {
    // A `then: call(...)` lifts a synth call node; a bad ref inside it carets the `then:` line (the
    // synth node's own provenance), not the case header.
    const text =
      "id: f\nname: f\nnodes:\n" +
      "  s:\n    kind: code\n    output: int\n    code: \"return {'output': 1}\"\n" +
      "  gate:\n    kind: case\n" +
      '    cases:\n      - when: "s.output >= 1"\n' +
      "        then: call(helper, x=${nope.output})\n" + // `then:` on line 12
      "    else: s\n" +
      "output: ${gate.output}\n" +
      "defs:\n" +
      "  helper:\n    input:\n      x: !int\n" +
      "    nodes:\n      h:\n        kind: code\n        code: \"return {'output': 1}\"\n" +
      "        input:\n          x: ${input.x}\n        output: int\n" +
      "    output: ${h.output}\n";

    const error = loadError(() => assembleText(text));

    expect(error.message).toContain(
      "node 'gate' cases[0].then call(helper) argument 'x' from: reference ${nope.output}",
    );
    expect(error.message).not.toContain("__call_");
    expect(error.diagnostic.frames[0]!.range.startLine).toBe(12);
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

// ---------- other rows ----------

test("a case output over a 1000-link alias chain is rejected as a case output", () => {
  // A listed chain's anchors cost quadratically
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

// ---------- other conformance cases ----------

test.each<CaseRow>([
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
  ...[
    "load-core/compose_expression_nesting__too_deep_case_on",
    "load-core/compose_expression_nesting__too_deep_human_question",
    "load-core/compose_expression_nesting__too_deep_prompt_span",
    "yaml/compose_load_error_ranges__mapped_root_output_keeps_the_retained_range",
    "load-case/engine_case_when_escape_e2e__an_on_case_with_no_cases_still_checks_its_on_value",
    "load-case/engine_case_when_escape_e2e__an_unknown_field_inside_an_escape",
    "load-case/engine_case_when_escape_e2e__an_unknown_namespace_in_an_escape_is_reported_once",
    "load-core/unknown_namespace",
    "load-case/engine_value_entries_run__a_routed_entry_reading_an_excluded_field",
  ].map((id) => ({ id, call: assembled, full: true })),
  {
    id: "load-core/compose_expression_nesting__too_deep_call_argument",
    call: assembled,
    full: true,
  },
  {
    id: "load-core/compose_expression_nesting__too_deep_node_assert",
    call: assembled,
    full: true,
  },
  {
    id: "load-core/compose_expression_nesting__too_deep_flow_assert",
    call: assembled,
    full: true,
  },
  { id: "load-leaves/seed_23_asserts_scopes", call: assembled, full: true },
  { id: "load-core/seed_19_binding_stances", call: assembled, full: true },
  { id: "load-core/seed_27_expr_ops", call: assembled, full: true },
])("assemble: $id", runCase);

test.each<CaseRow>([
  { id: "load-core/compose_validate__unknown_namespace_in_a_case_when", call: assembled, full: true },
  { id: "load-core/compose_validate__unknown_namespace_in_a_case_on", call: assembled, full: true },
  {
    id: "load-core/compose_validate__unknown_namespace_in_a_case_then_call",
    call: assembled,
    full: true,
  },
])("assemble: $id", runCase);
