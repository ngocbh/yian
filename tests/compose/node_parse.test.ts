/** The keyed-map flat-body node descriptors `parseNodes` reads, parsed with the registered kinds. */

import { afterEach, describe, expect, test } from "bun:test";

import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import type { DescriptorClass } from "../../src/compose/nodes/index.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { reprText } from "../../src/diagnostics.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  caseText,
  loadError,
  loadErrorAsync,
  nodesOf,
  runCase,
  runTextCase,
} from "../helpers/declarations.ts";
import { withKinds } from "../helpers/stand_ins.ts";

afterEach(() => forget());

/** Parse a bare `nodes:`-body mapping with `kinds` registered. */
function nodes(yamlText: string, kinds: readonly DescriptorClass[] = []) {
  return withKinds(kinds, () => parseNodes(SourceDocument.parse(yamlText).root));
}

/** An `agent` node body under `id`, as written. */
function agentAt(id: string): string {
  return `\n${id}:\n  kind: agent\n  prompt: hi\n  output: str\n`;
}

describe("reserved boundary ids", () => {
  test("reserved reference path separator rejected", () => {
    // A dot cannot be both part of a node id and the reference path separator.
    const error = loadError(() => nodes(agentAt("my.node")));

    expect(error.message).toMatch(/node id 'my\.node'.*'\.' is reserved for reference paths/);
    expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(3);
  });

  test.each(["__start__", "__end__"])("reserved boundary id rejected: %s", (reserved) => {
    // Authors write `input:`/`output:`, never a boundary node.
    expect(loadError(() => nodes(agentAt(reserved))).message).toContain(
      "reserved for the synthesized START_ID/END_ID boundary",
    );
  });
});

describe("reserved resolver head literals", () => {
  test.each(["input", "output", "system", "item"])(
    "reserved singular head rejected: %s",
    (reserved) => {
      // Required so `${<node>.output.k}` and `${input.k}` are unambiguous.
      expect(loadError(() => nodes(agentAt(reserved))).message).toContain(
        `node id ${reprText(reserved)} is reserved`,
      );
    },
  );

  test.each(["inputs", "outputs"])("reserved plural head rejected: %s", (reserved) => {
    // The plural forms are typo-catchers naming the retirement of the plural form.
    expect(loadError(() => nodes(agentAt(reserved))).message).toMatch(
      new RegExp(`node id '${reserved}' is reserved.*plural head is retired`),
    );
  });
});

describe("node ids must be identifiers", () => {
  test.each([
    ["fetch-data", "fetch_data"],
    ["'2x'", "_2x"],
    ["'a b'", "a_b"],
    ["café", "caf_"],
  ])("non-identifier node id rejected: %s", (nodeId, suggestion) => {
    // A node id `${...}` can't reference is rejected where the node is defined.
    const error = loadError(() => nodes(agentAt(nodeId)));

    expect(error.message).toContain("must be an identifier");
    expect(error.message).toContain(`try ${reprText(suggestion)}`);
    expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(3);
  });

  test("empty node id rejected without reference hint", () => {
    const error = loadError(() => nodes(agentAt("''")));

    expect(error.message).toContain("node id '' must be an identifier");
    expect(error.message).not.toContain("${");
    expect(error.message).not.toContain("; try");
  });

  test.each(["and", "or", "not", "in", "'true'", "'false'", "'null'", "none", "inf"])(
    "expression keyword node id rejected: %s",
    (keyword) => {
      // An identifier that is an expression keyword can't be a reference root.
      const name = keyword.replaceAll("'", "");
      expect(loadError(() => nodes(agentAt(keyword))).message).toContain(
        `node id ${reprText(name)} is an expression keyword`,
      );
    },
  );

  test("identifier node ids accepted", () => {
    const text = agentAt("fetch_data") + agentAt("_private2") + agentAt("notify");
    const descs = nodes(text);

    expect([...descs.keys()]).toEqual(["fetch_data", "_private2", "notify"]);
  });

  test("referenced hyphenated node names the id, not split refs", () => {
    const flow = `
id: f
name: f
input: {}
nodes:
  fetch-data:
    kind: code
    code: "return 1"
output: \${fetch-data.output}
`;
    const error = loadError(() => nodesOf(flow));

    expect(error.message).toMatch(/node id 'fetch-data' must be an identifier.*try 'fetch_data'/);
    expect(error.message).not.toContain("unknown namespace");
  });

  test.each(["call", "map"])("hyphenated node in an inline %s body is rejected", (kind) => {
    const over = kind === "map" ? "    over: [1, 2]\n" : "";
    const flow = `
id: f
name: f
input: {}
nodes:
  outer:
    kind: ${kind}
${over}    input: {}
    nodes:
      inner-step:
        kind: code
        code: "return 1"
    output: \${inner-step.output}
output: \${outer.output}
`;

    const error = loadError(() => assembleText(flow));

    expect(error.message).toBe(
      "node id 'inner-step' must be an identifier (ASCII letters, digits, and _, not starting with a " +
        "digit) so ${inner-step.output} can reference it; try 'inner_step'",
    );
  });

  test("a hyphenated single-node def is rejected but a multi-node def name loads", async () => {
    const single = `
id: f
name: f
input: {}
defs:
  my-def:
    kind: code
    code: "return 1"
    output: int
nodes:
  a:
    kind: call
    call: my-def
output: \${a.output}
`;
    const error = await loadErrorAsync(loadFlow(single));
    expect(error.message).toBe(
      "defs entry 'my-def': node id 'my-def' must be an identifier (ASCII letters, digits, and _, not " +
        "starting with a digit) so ${my-def.output} can reference it; try 'my_def'",
    );

    const multi = `
id: f
name: f
input: {}
defs:
  my-def:
    input: {}
    nodes:
      b:
        kind: code
        code: "return 1"
    output: \${b.output}
nodes:
  a:
    kind: call
    call: my-def
output: \${a.output}
`;
    await loadFlow(multi);
  });
});

describe("agent", () => {
  test("agent minimal parses", () => {
    const node = nodes(
      "\nnote:\n  kind: agent\n  input:\n    topic: ${input.topic}\n  output: str\n  prompt: Write a note on ${topic}.\n",
    ).get("note") as AgentDescriptor;

    expect(node).toBeInstanceOf(AgentDescriptor);
    expect(node.id).toBe("note");
    expect(node.prompt).toBe("Write a note on ${topic}.");
    expect(node.inputs).toEqual({ topic: "${input.topic}" });
    expect(node.outputs).toBe("str");
    expect(node.tools).toEqual([]);
    expect(node.llmConfig).toEqual({});
  });

  test("agent knobs parse", () => {
    // seed 14: tools, llm_config and a record output
    const { text, label } = caseText("load-agent/seed_14_agent_tools");
    const node = nodesOf(text, [], label).get("reviewer") as AgentDescriptor;

    expect(node).toBeInstanceOf(AgentDescriptor);
    expect(node.tools).toEqual(["get_web_data", "web_search", "ask_user"]);
    expect(node.llmConfig).toEqual({
      provider: "anthropic",
      model: "claude-opus-4-8",
      anthropic: { temperature: 0.3 },
    });
    expect(node.outputs).toEqual({ claim: "str", confidence: "float" });
  });
});

describe("code", () => {
  test("code parses", () => {
    // seed 01 verdict node
    const { text, label } = caseText("load-agent/seed_01_structured_agent");
    const node = nodesOf(text, [], label).get("verdict") as CodeDescriptor;

    expect(node).toBeInstanceOf(CodeDescriptor);
    expect(node.code).toBe("./fixtures/seeds.ts:one_line_summary");
    expect(node.inputs).toEqual({
      rating: "${score.output.rating}",
      rationale: "${score.output.rationale}",
    });
    expect(node.outputs).toBe("str");
  });
});

describe("tool", () => {
  test("model kind is removed and points at tool", () => {
    // ML inference is a host capability: `kind: model` names `kind: tool` as the replacement,
    // not the generic "unknown kind".
    const text = "\npredict:\n  kind: model\n  model_id: x\n  output: float\n";
    const error = loadError(() => nodes(text));

    expect(error.message).toContain("kind 'model' was removed");
    expect(error.message).toContain("kind: tool");
    expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(3);
  });

  test("tool parses", () => {
    const { text, label } = caseText("load-leaves/seed_08_tool_news");
    const node = nodesOf(text, [], label).get("news") as ToolDescriptor;

    expect(node).toBeInstanceOf(ToolDescriptor);
    expect(node.toolId).toBe("get_facts");
    expect(node.inputs).toEqual({ symbol: "${input.topic}", limit: 10n });
  });

  test("tool legacy args key is loud", () => {
    // `args:` was retired in favor of `input:`: the old key is an unknown field, no silent alias.
    const text =
      "\nnews:\n  kind: tool\n  tool_id: get_facts\n  args:\n    symbol: ${input.topic}\n";
    const error = loadError(() => nodes(text));

    expect(error.message).toContain("args");
    expect(error.message).toContain("not allowed");
    expect(error.diagnostic.frames.at(-1)!.range).not.toBeNull();
  });
});

describe("case", () => {
  test("case searched parses", () => {
    // seed 02 gate: the searched form (no on:)
    const { text, label } = caseText("load-case/seed_02_case");
    const node = nodesOf(text, [], label).get("gate") as CaseDescriptor;

    expect(node).toBeInstanceOf(CaseDescriptor);
    expect(node.on).toBeNull();
    expect(node.cases).toEqual([{ when: "score.output >= 0.5", then: "positive" }]);
    expect(node.else_).toBe("cautious");
  });

  test("case on parses", () => {
    // seed 06 route: the on: form
    const { text, label } = caseText("load-case/seed_06_case_on");
    const node = nodesOf(text, [], label).get("route") as CaseDescriptor;

    expect(node).toBeInstanceOf(CaseDescriptor);
    expect(node.on).toBe("${classify.output}");
    expect(node.cases).toEqual([
      { when: "pro", then: "pro_note" },
      { when: "con", then: "con_note" },
      { when: "mixed", then: "choppy_note" },
    ]);
    expect(node.else_).toBe("choppy_note");
  });

  test("case carries no inputs", () => {
    // A CASE's locals are generated, so it has no `input:` name (and no `inf` local) to author.
    const text = `
gate:
  kind: case
  input:
    x: \${score.output}
  cases:
    - when: "score.output >= 0.5"
      then: a
  else: b
`;

    expect(loadError(() => nodes(text)).message).toBe(
      "node 'gate' (kind=case): field 'input' is not allowed " +
        "(allowed: cases, depends_on, else, env, kind, node_name, on, output, runs_after)",
    );
  });
});

describe("call", () => {
  test("call ref form parses", () => {
    // seed 04: a plain `call` is a single application
    const { text, label } = caseText("load-map-call/seed_04_call");
    const node = nodesOf(text, [], label).get("research") as CallDescriptor;

    expect(node).toBeInstanceOf(CallDescriptor);
    expect(node.kind).toBe("call");
    expect(node.call).toBe("research-one");
    expect(node.inputs).toEqual({ topic: "${input.topic}" });
    expect(node.over).toBeNull();
    expect(node.parallel).toBe(false);
  });

  test("call map form parses", () => {
    // seed 05: a `map` (kind: map and over:) is List.map
    const { text, label } = caseText("load-map-call/seed_05_call_map");
    const node = nodesOf(text, [], label).get("research_each") as CallDescriptor;

    expect(node).toBeInstanceOf(CallDescriptor);
    expect(node.kind).toBe("map");
    expect(node.call).toBe("research-one");
    expect(node.over).toBe("${input.topics}");
    expect(node.inputs).toEqual({ topic: "${item}", as_of: '${input.as_of:-"2026-01-01"}' });
    expect(node.parallel).toBe(true);
    expect(node.nodeName).toBe("Research each topic");
  });

  test("map without over is loud", () => {
    // `over:` is the iteration source: a `kind: map` without it is incomplete.
    expect(loadError(() => nodes("\nn:\n  kind: map\n  call: child\n")).message).toBe(
      "node 'n' (kind=map): missing required field 'over'",
    );
  });

  test("over on kind call is loud", () => {
    // `over:` and `parallel:` are MAP-only; `kind: call` rejects them (use `map`).
    const allowed = "(allowed: asserts, call, depends_on, env, input, kind, node_name, runs_after)";

    expect(loadError(() => nodes("\nn:\n  kind: call\n  call: child\n  over: ${input.xs}\n")).message).toBe(
      `node 'n' (kind=call): field 'over' is not allowed ${allowed}`,
    );
    expect(loadError(() => nodes("\nn:\n  kind: call\n  call: child\n  parallel: true\n")).message).toBe(
      `node 'n' (kind=call): field 'parallel' is not allowed ${allowed}`,
    );
  });
});

describe("node-level asserts: surface", () => {
  test("node asserts parse on code", () => {
    const text = `
x:
  kind: code
  code: m:f
  input:
    n: \${input.v}
  output: int
  asserts:
    - "output >= 0"
`;
    const node = nodes(text).get("x") as CodeDescriptor;

    expect(node.asserts).toEqual(["output >= 0"]);
  });

  test("node asserts must be list of str", () => {
    const text = "\nx:\n  kind: code\n  code: m:f\n  output: int\n  asserts: not-a-list\n";
    const error = loadError(() => nodes(text));

    expect(error.message).toBe("node 'x' (kind=code): `asserts:` must be a list of strings");
  });

  test("node asserts rejected on case", () => {
    const text = `
gate:
  kind: case
  cases:
    - when: "s.output > 0"
      then: a
  asserts:
    - "output >= 0"
`;

    expect(loadError(() => nodes(text)).message).toBe(
      "node 'gate' (kind=case): field 'asserts' is not allowed " +
        "(allowed: cases, depends_on, else, env, kind, node_name, on, output, runs_after)",
    );
  });

  test("legacy ref kind is unknown", () => {
    // The collapse removed `kind: ref`: it is an unknown kind.
    const error = loadError(() => nodes("n:\n  kind: ref\n  call: child\n"));

    expect(error.message).toContain("ref");
    expect(error.message).toContain("unknown kind");
  });
});

test("seed 11 anchored nodes parse", () => {
  // merged-anchor agent nodes all parse to AgentDescriptors
  const { text, label } = caseText("load-core/seed_11_reuse_anchors");
  const descs = nodesOf(text, [], label);

  for (const nodeId of ["pro", "con", "judge"]) {
    const node = descs.get(nodeId) as AgentDescriptor;
    expect(node).toBeInstanceOf(AgentDescriptor);
    expect(node.outputs).toBe("str");
    expect((node.llmConfig as Record<string, unknown>)["provider"]).toBe("anthropic");
  }
});

describe("common fields and loud rules", () => {
  test("node name, depends on and runs after captured", () => {
    // depends_on + runs_after declare run-ordering edges; node_name is the display label.
    const node = nodes(
      `
fetch:
  kind: code
  node_name: Fetch values
  depends_on: [warm_cache]
  runs_after: [log_start]
  input:
    topic: \${input.topic}
  output:
    values: list(float)
  code: tests.seeds.fns:fetch_facts
`,
    ).get("fetch")!;

    expect(node).toBeInstanceOf(CodeDescriptor);
    expect(node.nodeName).toBe("Fetch values");
    expect(node.dependsOn).toEqual(["warm_cache"]);
    expect(node.runsAfter).toEqual(["log_start"]);
  });

  test("depends on and runs after default empty", () => {
    const node = nodes(
      "\nnote:\n  kind: agent\n  input:\n    topic: ${input.topic}\n  output: str\n  prompt: hi\n",
    ).get("note")!;

    expect(node.dependsOn).toEqual([]);
    expect(node.runsAfter).toEqual([]);
    expect(node.nodeName).toBeNull();
  });

  test("field illegal for kind is loud", () => {
    // `prompt` is an agent field, illegal on a code node.
    const text = "\nn:\n  kind: code\n  code: tests.seeds.fns:f\n  prompt: not allowed here\n";
    const error = loadError(() => nodes(text));

    expect(error.message).toContain("prompt");
    expect(error.diagnostic.frames.at(-1)!.range).not.toBeNull();
  });

  test("tool id illegal on agent is loud", () => {
    const text = "\nn:\n  kind: agent\n  tool_id: get_facts\n  output: str\n  prompt: hi\n";
    const error = loadError(() => nodes(text));

    expect(error.message).toContain("tool_id");
  });

  test("unknown kind is loud", () => {
    const error = loadError(() => nodes("\nn:\n  kind: wizard\n  output: str\n"));

    expect(error.message).toContain("wizard");
    expect(error.diagnostic.frames.at(-1)!.range).not.toBeNull();
  });

  test("missing kind is loud", () => {
    const text = "\nn:\n  input:\n    topic: ${input.topic}\n  output: str\n";
    const error = loadError(() => nodes(text));

    expect(error.message).toBe(
      "node 'n': an entry in `nodes:` is a node or a value; add `kind:` for a node, or a tag " +
        "(`!object`, `!Buy`, ...) for a value",
    );
  });

  describe.each(["depends_on", "runs_after"])("%s", (key) => {
    test.each(["5", "null", "true", "{fetch: 1}", "[[fetch]]", "[5]", "''"])(
      "ordering field must be a list of node ids: %s",
      (value) => {
        // A non-list, or a list of non-strings, is a located load error at parse time, not a
        // silent iteration over a string's characters or a mapping's keys.
        const text = `use:\n  kind: code\n  ${key}: ${value}\n  output: str\n  code: m:f\n`;
        const error = loadError(() => nodes(text));

        expect(error.message).toContain(`\`${key}:\` must be a list of node ids`);
        expect(error.diagnostic.frames[0]!.range).not.toBeNull();
      },
    );

    test("ordering field bare string suggests list", () => {
      // `depends_on: fetch` is the natural slip; the error names the list spelling to write.
      const text = `use:\n  kind: code\n  ${key}: fetch\n  output: str\n  code: m:f\n`;

      expect(loadError(() => nodes(text)).message).toContain(
        `write \`${key}: [fetch]\``,
      );
    });
  });
});

test.each([
  "load-core/engine_author_id_ban__author_id_with_hash_is_loud",
  "load-core/engine_node_parse__hyphenated_node_id",
  "load-core/seed_e10_unknown_node_kind",
  "load-core/seed_e42_removed_model_kind",
])("parseNodes: %s", (id) => runTextCase({ id, call: (text, label) => nodesOf(text, [], label) }));

test.each<CaseRow>([{ id: "load-core/engine_node_parse__hyphenated_node_id_in_an_inline_body", call: assembled }])(
  "assemble: $id",
  runCase,
);
