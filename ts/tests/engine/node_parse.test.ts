/**
 * The keyed-map flat-body node descriptors `parseNodes` reads: a port of
 * `tests/engine/test_node_parse.py`. The rows read through `parseNodes` parse with the registered
 * kinds.
 *
 * TS-only: `inf` is an expression keyword, so it is a rejected node id too.
 *
 * Changed: `test_agent_knobs_parse`, `test_code_parses`, `test_tool_parses` and
 * `test_seed_11_anchored_nodes_parse` read seeds 14, 01, 08 and 11 through the conformance cases
 * `load-agent/seed_14_agent_tools`, `load-agent/seed_01_structured_agent`,
 * `load-leaves/seed_08_tool_news` and `load-core/seed_11_reuse_anchors`; seed 01's CODE reference
 * is the TS form (`./fixtures/seeds.ts:one_line_summary`).
 *
 * Later (T3-16): `test_case_searched_parses`, `test_case_on_parses`,
 * `test_case_carries_no_inputs`, `test_call_ref_form_parses`, `test_node_asserts_rejected_on_case`,
 * `test_call_map_form_parses`, `test_map_without_over_is_loud`, `test_over_on_kind_call_is_loud`.
 * Later (T3-17): `test_hyphenated_node_in_inline_body_rejected`.
 * Later (T3-22): `test_hyphenated_single_node_def_rejected_but_multi_node_def_name_loads`.
 *
 * Changed: `test_referenced_hyphenated_node_names_the_id_not_split_refs` reads its flow through
 * `parseFile`, `flowNodesCursor` and `parseNodes`, where Python loads it.
 *
 * The harvested cases whose error `parseNodes` raises are pinned at the end, with
 * `load-core/engine_author_id_ban__author_id_with_hash_is_loud` (from
 * `tests/engine/test_author_id_ban.py`) and the e10 and e42 seeds; the e10 row registers
 * stand-ins for the kinds not registered yet, since its message lists all nine.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import type { DescriptorClass } from "../../src/compose/nodes/index.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { reprText } from "../../src/diagnostics.ts";
import { caseText, loadError, nodesOf, runTextCase } from "../compose/declarations.ts";
import { standIn, UNREGISTERED_KIND_NAMES, withKinds } from "../compose/stand_ins.ts";

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
  // tests/engine/test_author_id_ban.py
  ["load-core/engine_author_id_ban__author_id_with_hash_is_loud", []],
  ["load-core/engine_node_parse__hyphenated_node_id", []],
  ["load-core/seed_e10_unknown_node_kind", UNREGISTERED_KIND_NAMES.map((kind) => standIn(kind))],
  ["load-core/seed_e42_removed_model_kind", []],
] as const)("parseNodes: %s", (id, kinds) =>
  runTextCase({ id, call: (text, label) => nodesOf(text, kinds, label) }),
);
