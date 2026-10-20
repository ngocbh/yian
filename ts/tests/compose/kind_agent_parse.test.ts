/**
 * The `agent` kind's parse side: a port of `tests/compose/test_kind_agent_parse.py`.
 *
 * Changed: Python compares the descriptor to an `AgentDescriptor` with `field_ranges` excluded;
 * TS compares its class and fields. The variable values keep YAML's ints as `bigint` (`4n`).
 *
 * TS-only: a `retries:` past `Number.MAX_SAFE_INTEGER` is refused, and a count is stored as a
 * `number`; the default follows `YIAN_AGENT_RETRIES`; a variable named `inf` is refused at its key,
 * before a like-named input; a lone surrogate in `prompt:` or `system_prompt:` is refused at the
 * field; the constructor refuses a non-exact `MemoryDecl`; `withSourceOrigins` registers one
 * origin per `long_term_memory:` entry; `build` on a hand-built descriptor and context (no
 * declarations build no variables, memory sites stamped with `ctx.owner`, the `llm_config` label
 * and the `__llm_config` range; the variable reader's fallback and return check are in
 * `agent_variables.test.ts`); and the literal `llm_config:` check, which reads `LLMConfig`'s
 * first issue where Python reads pydantic's: a text field read as non-text, at the top level or
 * in a block, is named as `notTextError` names it, and a field taking more than text (`azure.stop`)
 * keeps the validation error.
 *
 * Held here until T3-22: the harvested cases at the end, whose error the AGENT parse or build
 * raises, each under a comment naming the Python file it comes from, and the flows that assemble.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { AgentDescriptor, retriesError } from "../../src/compose/nodes/agent.ts";
import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, forget, reload } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { assembled, type CaseRow, loadError, parsedNodes, runCase } from "./declarations.ts";

afterEach(() => forget());

const FLOW = `nodes:
  writer:
    kind: agent
    prompt: "Write about \${topic}"
    inputs:
      topic: \${input.topic}
    outputs: string
`;

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Parse node `a` of `text` as an AGENT. */
function parseA(text: string): AgentDescriptor {
  return AgentDescriptor.parse("a", readerFor(text, "a"));
}

/** Return a descriptor's fields, leaving out the parse metadata (Python's `compare=False`). */
function fieldsOf(desc: AgentDescriptor): Record<string, unknown> {
  const { range: _range, fieldRanges: _fieldRanges, assertRanges: _assertRanges, ...fields } = desc;
  return fields;
}

test("agent parse matches today's descriptor and stamps ranges", () => {
  const desc = AgentDescriptor.parse("writer", readerFor(FLOW, "writer"));

  expect(desc).toBeInstanceOf(AgentDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(
      new AgentDescriptor({
        id: "writer",
        prompt: "Write about ${topic}",
        inputs: { topic: "${input.topic}" },
        outputs: "string",
      }),
    ),
  );
  // Ranges are recorded for the node body and its fields.
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "prompt"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
  // The prompt value sits on line 4 of the flow (1-based).
  expect(desc.fieldRanges.get(siteKey("field", "prompt"))!.startLine).toBe(4);
});

test("agent descriptor rejects controls field", () => {
  const text = "nodes:\n  writer:\n    kind: agent\n    prompt: hi\n    controls: [ask_user]\n";

  expect(loadError(() => AgentDescriptor.parse("writer", readerFor(text, "writer"))).message).toMatch(
    /controls.*not allowed/,
  );
});

test("agent variable declarations remain separate from binding fields", () => {
  const text = `typedefs:
  Tone: Literal(concise, detailed, technical)
nodes:
  writer:
    kind: agent
    variables:
      num_sentences: !int(1..12) 4
      temperature: !float(0..1) 0.4
      tone: !Tone concise
      instruction: !str "Summarize clearly"
    prompt: "\${instruction} in \${num_sentences} sentences."
`;
  const desc = AgentDescriptor.parse("writer", readerFor(text, "writer"));

  expect(desc.variables).toEqual({
    num_sentences: 4n,
    temperature: 0.4,
    tone: "concise",
    instruction: "Summarize clearly",
  });
  expect(desc.inputs).toEqual({});
  expect(AgentDescriptor.bindingFields).toEqual(["inputs"]);
});

describe("retries", () => {
  test("a count past the largest safe integer is refused", () => {
    const error = loadError(() => parseA("nodes:\n  a:\n    kind: agent\n    retries: 9007199254740992\n"));

    expect(error.message).toBe(
      "node 'a' (kind=agent): `retries:` must be at most 9007199254740991, got 9007199254740992",
    );
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
  });

  test("a count is stored as a number", () => {
    expect(parseA("nodes:\n  a:\n    kind: agent\n    retries: 3\n").retries).toBe(3);
    expect(parseA("nodes:\n  a:\n    kind: agent\n    retries: 9007199254740991\n").retries).toBe(
      Number.MAX_SAFE_INTEGER,
    );
  });

  test("the default follows YIAN_AGENT_RETRIES", () => {
    const text = "nodes:\n  a:\n    kind: agent\n    prompt: hi\n";
    reload((name) => (name === "YIAN_AGENT_RETRIES" ? "5" : undefined));
    expect(parseA(text).retries).toBe(5);
    reload(() => undefined);
    expect(parseA(text).retries).toBe(BUILTIN.agentRetries);
    expect(new AgentDescriptor({ id: "a" }).retries).toBe(BUILTIN.agentRetries);
  });

  test.each<[unknown, string | null]>([
    [0n, null],
    [3n, null],
    [9007199254740991n, null],
    [9007199254740992n, "must be at most 9007199254740991, got 9007199254740992"],
    [-1n, "must be a non-negative integer, got -1"],
    [true, "must be a non-negative integer, got true"],
    [1.5, "must be a non-negative integer, got 1.5"],
    ["3", 'must be a non-negative integer, got "3"'],
    [null, "must be a non-negative integer, got null"],
  ])("retriesError(%p)", (value, problem) => {
    expect(retriesError(value)).toBe(problem);
  });
});

describe("inf names", () => {
  test("a variable named inf is refused at its key", () => {
    const text = "nodes:\n  a:\n    kind: agent\n    prompt: hi\n    variables:\n      inf: !int 1\n";
    const error = loadError(() => parseA(text));

    expect(error.message).toBe(
      "node 'a' variable 'inf': `inf` is infinity in an expression, so ${inf} can't read it; rename it",
    );
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(
      readerFor(text, "a").child("variables").keyRangeOf("inf")!,
    );
  });

  test("a variable named inf is refused before an input named inf", () => {
    const text = `nodes:
  a:
    kind: agent
    prompt: hi
    input:
      inf: \${input.x}
    variables:
      inf: !int 1
`;
    const error = loadError(() => parseNodes(SourceDocument.parse(text).root.child("nodes")));

    expect(error.message).toStartWith("node 'a' variable 'inf': ");
  });
});

describe("lone surrogates", () => {
  test.each(["prompt", "system_prompt"])("a lone surrogate in %s is refused at the field", (key) => {
    const text = `nodes:\n  a:\n    kind: agent\n    ${key}: "\\ud800"\n`;
    const error = loadError(() => parseA(text));

    expect(error.message).toBe(`node 'a' field '${key}': text holding a lone surrogate is not a yian value`);
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(readerFor(text, "a").rangeOf(key)!);
  });
});

test("the constructor refuses memories that aren't exact MemoryDecl values", () => {
  class SubDecl extends MemoryDecl {}
  const site = new DeclSite([], ["root"], "a");
  const message = "agent descriptor memories must be an array of exact MemoryDecl values";

  expect(
    () => new AgentDescriptor({ id: "a", memories: [new SubDecl("notes", "episodic", {}, site)] }),
  ).toThrow(new TypeError(message));
  expect(
    () => new AgentDescriptor({ id: "a", memories: "notes" as unknown as readonly MemoryDecl[] }),
  ).toThrow(new TypeError(message));
  expect(new AgentDescriptor({ id: "a", memories: [new MemoryDecl("notes", "episodic", {}, site)] }).memories)
    .toHaveLength(1);
});

describe("withSourceOrigins", () => {
  test("one origin per long_term_memory entry, at its key", () => {
    const text = "nodes:\n  a:\n    kind: agent\n    long_term_memory:\n      notes: episodic\n      lessons: reflective\n";
    const cursor = readerFor(text, "a");
    const builder = new SourceCatalogBuilder();
    const desc = AgentDescriptor.parse("a", cursor);
    expect(desc.memories.map((decl) => decl.site)).toEqual([
      new DeclSite([], ["root"], "a"),
      new DeclSite([], ["root"], "a"),
    ]);

    const stamped = desc.withSourceOrigins(cursor, { sitePath: ["node", "a"], catalogBuilder: builder });

    expect(stamped).toBeInstanceOf(AgentDescriptor);
    expect(stamped.memories.map((decl) => decl.name)).toEqual(["notes", "lessons"]);
    const memories = cursor.child("long_term_memory");
    for (const decl of stamped.memories) {
      expect(builder.resolveRegistered(decl.origin!)!.range).toEqual(memories.keyRangeOf(decl.name)!);
    }
    expect(desc.memories.every((decl) => decl.origin === null)).toBe(true);
  });

  test("the sugar form's one origin is the whole value", () => {
    const text = "nodes:\n  a:\n    kind: agent\n    long_term_memory: episodic\n";
    const cursor = readerFor(text, "a");
    const builder = new SourceCatalogBuilder();

    const stamped = AgentDescriptor.parse("a", cursor).withSourceOrigins(cursor, {
      sitePath: ["node", "a"],
      catalogBuilder: builder,
    });

    expect(stamped.memories).toHaveLength(1);
    expect(builder.resolveRegistered(stamped.memories[0]!.origin!)!.range).toEqual(
      cursor.rangeOf("long_term_memory")!,
    );
  });

  test("no memories: the descriptor itself", () => {
    const cursor = readerFor("nodes:\n  a:\n    kind: agent\n    long_term_memory: null\n", "a");
    const desc = AgentDescriptor.parse("a", cursor);

    expect(desc.memories).toEqual([]);
    expect(desc.withSourceOrigins(cursor, { sitePath: ["node", "a"], catalogBuilder: new SourceCatalogBuilder() }))
      .toBe(desc);
  });
});

describe("build", () => {
  /** A context over an empty registry, with `owner` as the owner path of the node being built. */
  const reading = (owner?: readonly string[]) => new BuildContext({ registry: new Map(), owner });

  test("a node with no declarations builds no variables", () => {
    expect((new AgentDescriptor({ id: "a" }).build(reading()).node as AgentNode).variables).toEqual([]);
  });

  test("memory sites are stamped with the context's owner", () => {
    const origin = new SourceCatalogBuilder().register(["node", "a"], readerFor("nodes:\n  a: {}\n", "a"));
    const memories = [new MemoryDecl("notes", "episodic", {}, new DeclSite([], ["root"], "a"), origin)];
    const node = new AgentDescriptor({ id: "a", memories }).build(reading(["root", "node", "outer"]))
      .node as AgentNode;

    expect(node.memories).toEqual([
      new MemoryDecl("notes", "episodic", {}, new DeclSite([], ["root", "node", "outer"], "a"), origin),
    ]);
  });

  test("the llm_config label comes from the field labels", () => {
    const label = "node 'ask' adaptive_questions.llm_config";
    const labelled = new AgentDescriptor({ id: "a", fieldLabels: new Map([["llm_config", label]]) });

    expect((labelled.build(reading()).node as AgentNode).llmConfigLabel).toBe(label);
    expect((new AgentDescriptor({ id: "a" }).build(reading()).node as AgentNode).llmConfigLabel).toBeNull();
  });

  test("errors about llm_config values point at the authored llm_config", () => {
    const own = new SourceRange(1, 1, 5, 1);
    const field = new SourceRange(3, 17, 3, 30);
    const fieldRanges = new Map([[siteKey("field", "llm_config"), field]]);

    expect(
      new AgentDescriptor({ id: "a", range: own, fieldRanges }).build(reading()).fieldRanges.get(
        siteKey("field", "__llm_config"),
      ),
    ).toBe(field);
    expect(
      new AgentDescriptor({ id: "a", range: own }).build(reading()).fieldRanges.get(siteKey("field", "__llm_config")),
    ).toBe(own);
  });
});

describe("literal llm_config fields", () => {
  /** Return the message of the `LoadError` building node `a` with `llmConfig` raises. */
  const llmError = (llmConfig: unknown) =>
    loadError(() => new AgentDescriptor({ id: "a", llmConfig }).build(new BuildContext({ registry: new Map() })))
      .message;

  test("a text field read as non-text names the field", () => {
    expect(llmError({ model: 5n })).toBe(
      "node 'a': llm_config: `model:` must be text, but YAML read it as an integer; put it in quotes",
    );
  });

  test("a block's text field read as non-text names its path", () => {
    expect(llmError({ anthropic: { inference_geo: 5n } })).toBe(
      "node 'a': llm_config: `anthropic.inference_geo:` must be text, but YAML read it as an integer; " +
        "put it in quotes",
    );
  });

  // `stop` takes text or a list of text, so its value is not a text field read as non-text.
  test.each([[5n], [[5n]]])("a field taking more than text keeps the validation error: stop %p", (stop) => {
    expect(llmError({ provider: "azure", azure: { stop } })).toStartWith(
      "node 'a': llm_config: 2 validation errors for LLMConfig\nazure.stop.str\n",
    );
  });
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  // tests/cli/test_memory.py
  { id: "load-agent/cli_memory__session_recent_turns_below_zero", call: parsedNodes },
  { id: "load-agent/cli_memory__storage_key_is_host_owned", call: parsedNodes },
  // tests/compose/test_agent_field_shapes.py
  { id: "load-agent/compose_agent_field_shapes__bool_retries", call: parsedNodes },
  { id: "load-agent/compose_agent_field_shapes__negative_retries", call: parsedNodes },
  { id: "load-agent/compose_agent_field_shapes__tools_as_text", call: parsedNodes },
  // tests/compose/test_agent_mode_removed.py
  { id: "load-agent/compose_agent_mode_removed__mode_plain_without_tools", call: parsedNodes },
  // tests/compose/test_agent_skills_yaml.py
  { id: "load-agent/compose_agent_skills_yaml__skills_as_text", call: parsedNodes },
  // tests/engine/test_node_env.py
  { id: "load-agent/engine_node_env__env_must_be_string_keyed_mapping", call: parsedNodes },
  // tests/seeds/errors/
  { id: "load-agent/seed_e40_agent_controls_no_range", call: parsedNodes },
  { id: "load-agent/seed_e43_agent_mode_removed", call: parsedNodes },
  // tests/compose/test_non_text_prompt.py
  { id: "load-core/compose_non_text_prompt__non_text_agent_prompt", call: parsedNodes },
  // TS-only (no Python test): the AGENT half of the `inf` name rule
  { id: "load-agent/compose_infinity_names__an_agent_variable_named_inf", call: parsedNodes },
])("parseNodes: $id", runCase);

test.each<CaseRow>([
  // tests/compose/test_agent_field_shapes.py
  { id: "load-agent/compose_agent_field_shapes__inherit_on_is_text", call: assembled },
  { id: "load-agent/compose_agent_field_shapes__int_inherit", call: assembled },
  // tests/compose/test_llm_config_blocks_load.py
  { id: "load-agent/compose_llm_config_blocks_load__node_level_reserved_field_fails_at_load", call: assembled },
  // tests/engine/test_llm_inherit.py
  { id: "load-agent/engine_llm_inherit__unknown_node_llm_config_key_fails", call: assembled },
  // tests/engine/test_run_api_key.py
  { id: "load-agent/engine_run_api_key__api_key_on_a_node", call: assembled },
  // tests/engine/test_run_base_url.py
  { id: "load-agent/engine_run_base_url__base_url_on_a_node", call: assembled },
  // tests/engine/test_run_connection_settings.py
  { id: "load-agent/engine_run_connection_settings__organization_on_a_node", call: assembled },
  // tests/seeds/errors/
  { id: "load-agent/seed_e39_llm_config_no_range", call: assembled },
  // tests/compose/test_agent_load_checks.py
  { id: "load-agent/compose_agent_load_checks__agent_output_of_file", call: assembled },
  { id: "load-agent/compose_agent_load_checks__node_env_max_tool_iterations_zero", call: assembled },
  // tests/compose/test_agent_mode_removed.py
  { id: "load-agent/compose_agent_mode_removed__builtin_tools_is_rejected_at_the_tools_line", call: assembled },
  // tests/compose/test_agent_output_nesting.py
  { id: "load-agent/compose_agent_output_nesting__a_written_out_record_past_the_limit", call: assembled },
  { id: "load-agent/compose_agent_output_nesting__an_agent_in_an_inline_map_body", call: assembled },
  { id: "load-agent/compose_agent_output_nesting__the_limit_follows_the_variable", call: assembled },
  // tests/compose/test_agent_skills_yaml.py
  { id: "load-agent/compose_agent_skills_yaml__invalid_skill_name", call: assembled },
  // tests/compose/test_llm_config_values_wiring.py
  { id: "load-agent/compose_llm_config_values_wiring__reserved_input_name", call: assembled },
])("assemble: $id", runCase);

test.each<CaseRow>([
  // tests/compose/test_agent_field_shapes.py
  { id: "load-agent/compose_agent_field_shapes__inherit_true_with_a_model_loads", call: assembled },
  // tests/compose/test_agent_load_checks.py
  { id: "load-agent/compose_agent_load_checks__agent_union_output_loads", call: assembled },
  { id: "load-agent/compose_agent_load_checks__max_tool_iterations_minus_one_loads", call: assembled },
  // tests/compose/test_agent_mode_removed.py
  { id: "load-agent/compose_agent_mode_removed__steerable_agent_without_tools_loads", call: assembled },
  // tests/compose/test_agent_output_nesting.py
  { id: "load-agent/compose_agent_output_nesting__a_written_out_record_at_the_limit_loads", call: assembled },
  // tests/compose/test_llm_config_values_load.py
  { id: "load-agent/compose_llm_config_values_load__a_field_value_loads_as_a_template", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__a_whole_record_value_loads_as_a_binding", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__literal_base_url_beside_literal_azure_loads", call: assembled },
  // tests/compose/test_llm_config_values_types.py
  { id: "load-agent/compose_llm_config_values_types__provider_with_a_known_member_loads", call: assembled },
  // examples/
  { id: "load-agent/example_ask_user", call: assembled },
  { id: "load-agent/example_classify", call: assembled },
  { id: "load-agent/example_debate", call: assembled },
  { id: "load-agent/example_hello", call: assembled },
  { id: "load-agent/example_structured_test", call: assembled },
  { id: "load-agent/example_summarize", call: assembled },
  { id: "load-agent/example_triage_ticket", call: assembled },
  // tests/seeds/
  { id: "load-agent/seed_14_agent_tools", call: assembled },
])("assembles: $id", runCase);
