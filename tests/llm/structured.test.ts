/**
 * Tests for `src/llm/structured.ts`, choosing how a structured call is sent and reading the JSON
 * object from the reply: `structuredTarget`, `planStructured` (one test per row of the selection
 * table, first match wins), `forcesTool` and `sendsJsonSchemaAsTool`; a plan's `apply` to a
 * request, its `instruction` and `correction` texts, and `parse` of tool and text replies; and
 * `StructuredOutputError` and `MissingToolCallError`; and that `typeToSchema` emits the schema
 * copies used here.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";

import { NoAuth } from "../../src/llm/auth.ts";
import { encodeBody } from "../../src/llm/body.ts";
import { RouteModel, type WireProtocol } from "../../src/llm/executor.ts";
import {
  anthropicOutputSchema,
  constAsEnum,
  functionTool,
  hasOpenSchema,
  moveAzureKeywords,
  strictJsonSchema,
} from "../../src/llm/json_schema.ts";
import {
  copyJson,
  Message,
  NamedTool,
  OutputFormat,
  type Part,
  ReasoningPart,
  RefusalPart,
  Request,
  Response,
  TextPart,
  ToolCallPart,
} from "../../src/llm/messages.ts";
import { AnthropicMessages } from "../../src/llm/protocols/anthropic_messages.ts";
import { Gemini } from "../../src/llm/protocols/gemini.ts";
import { OllamaChat } from "../../src/llm/protocols/ollama_chat.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { OpenAIResponses } from "../../src/llm/protocols/openai_responses.ts";
import { Route } from "../../src/llm/route.ts";
import {
  FORCED_TOOL_REFUSING_PREFIXES,
  forcesTool,
  MissingToolCallError,
  planStructured,
  sendsJsonSchemaAsTool,
  type Strategy,
  StructuredOutputError,
  type StructuredPlan,
  type StructuredTarget,
  structuredTarget,
} from "../../src/llm/structured.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { typeToSchema } from "../../src/nodes/agent/structured.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";

/** The schemas `typeToSchema` emits for the cases below: ints are `bigint`s. */
const SCHEMAS: Record<string, JsonObject> = {
  "corpus:Rec": {
    properties: { answer: { title: "Answer", type: "string" }, score: { title: "Score", type: "integer" } },
    required: ["answer", "score"],
    title: "Record",
    type: "object",
  },
  "corpus:Bounded": {
    properties: {
      answer: { title: "Answer", type: "string" },
      score: { maximum: 10n, minimum: 0n, title: "Score", type: "integer" },
    },
    required: ["answer", "score"],
    title: "Record",
    type: "object",
  },
  "corpus:Open": {
    properties: {
      answer: { title: "Answer", type: "string" },
      data: { additionalProperties: true, title: "Data", type: "object" },
    },
    required: ["answer", "data"],
    title: "Record",
    type: "object",
  },
  "corpus:Union": {
    $defs: {
      buy: {
        additionalProperties: false,
        properties: { kind: { const: "buy", title: "Kind", type: "string" }, x: { title: "X", type: "integer" } },
        required: ["kind", "x"],
        title: "buy",
        type: "object",
      },
      sell: {
        additionalProperties: false,
        properties: { kind: { const: "sell", title: "Kind", type: "string" }, n: { title: "N", type: "integer" } },
        required: ["kind", "n"],
        title: "sell",
        type: "object",
      },
    },
    properties: { value: { anyOf: [{ $ref: "#/$defs/buy" }, { $ref: "#/$defs/sell" }], title: "Value" } },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
  },
  "open_schema_test:object": {
    properties: { value: { additionalProperties: true, title: "Value", type: "object" } },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
  },
  "open_schema_test:Any": {
    properties: {
      value: {
        anyOf: [
          { type: "boolean" },
          { type: "integer" },
          { type: "number" },
          { type: "string" },
          { additionalProperties: true, type: "object" },
          { items: {}, type: "array" },
        ],
        title: "Value",
      },
    },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
  },
  "open_schema_test:R": {
    properties: {
      a: { title: "A", type: "integer" },
      m: { additionalProperties: true, title: "M", type: "object" },
    },
    required: ["a", "m"],
    title: "Record",
    type: "object",
  },
  // The record of the `run-agent` cases that script structured replies.
  "run-agent:Record": {
    properties: { name: { title: "Name", type: "string" }, score: { title: "Score", type: "integer" } },
    required: ["name", "score"],
    title: "Record",
    type: "object",
  },
};

const REC = SCHEMAS["corpus:Rec"]!;
const BOUNDED = SCHEMAS["corpus:Bounded"]!;
const OPEN = SCHEMAS["corpus:Open"]!;
const UNION = SCHEMAS["corpus:Union"]!;

/** Return a target on OpenAI's Responses with `fields` over it. */
function target(fields: Partial<StructuredTarget> = {}): StructuredTarget {
  return {
    provider: "openai",
    model: "gpt-5",
    protocol: "openai_responses",
    block: null,
    subscription: false,
    ...fields,
  };
}

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Return a reply whose assistant message holds `parts`. */
function reply(parts: Part[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }) });
}

/** Return a plan's fields as plain JSON-like values, the tool choice's name under `named`. */
function fields(plan: StructuredPlan): unknown {
  const choice = plan.toolChoice;
  return {
    strategy: plan.strategy,
    name: plan.name,
    tools: plan.tools.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.parameters,
      strict: t.strict,
    })),
    toolChoice: choice instanceof NamedTool ? { named: choice.name } : choice,
    parallelToolCalls: plan.parallelToolCalls,
    outputFormat: plan.outputFormat && {
      name: plan.outputFormat.name,
      schema: plan.outputFormat.schema,
      strict: plan.outputFormat.strict,
    },
  };
}

/** Return the expected `fields` of a plan; omitted fields are empty. */
function want(options: {
  strategy: Strategy;
  name?: string;
  tools?: { schema: JsonObject; strict: boolean | null }[];
  named?: string;
  parallel?: boolean | null;
  format?: { schema: JsonObject; strict?: boolean | null };
}): unknown {
  return {
    strategy: options.strategy,
    name: options.name ?? "Record",
    tools: (options.tools ?? []).map(({ schema, strict }) => {
      const t = functionTool(schema, strict);
      return { name: t.name, description: t.description, parameters: t.parameters, strict: t.strict };
    }),
    toolChoice: options.named === undefined ? null : { named: options.named },
    parallelToolCalls: options.parallel ?? null,
    outputFormat: options.format
      ? { name: options.name ?? "Record", schema: options.format.schema, strict: options.format.strict ?? null }
      : null,
  };
}

/** Assert `got` equals `want` as JSON values, ints and floats apart, with the same key order. */
function expectSame(got: unknown, wanted: unknown): void {
  expect(got).toEqual(wanted);
  expect(encodeBody(got)).toBe(encodeBody(wanted));
}

/** One plan per strategy, labelled with it; the apply and parse tests filter it. */
const PLANS: [Strategy, StructuredPlan][] = [
  ["forced_tool", planStructured(target({ provider: "xai", model: "grok-4", protocol: "openai_chat" }), REC)],
  [
    "unforced_tool",
    planStructured(target({ provider: "deepseek", model: "deepseek-reasoner", protocol: "openai_chat" }), REC),
  ],
  ["json_schema", planStructured(target({ provider: "azure", model: "gpt-4o" }), BOUNDED)],
  [
    "prompt",
    planStructured(target({ provider: "xai", model: "no-structured-sentinel", protocol: "openai_chat" }), REC),
  ],
];

describe("the test plans", () => {
  test.each(PLANS)("the %s plan uses its strategy", (strategy, plan) => {
    expect(plan.strategy).toBe(strategy);
  });
});

describe("planStructured: row 1, no native structured output", () => {
  test.each([
    ["xai", "openai_chat"],
    ["openai", "openai_responses"],
    ["openai", null],
    ["xai", null],
  ] as const)("the sentinel on %s over %p is prompted", (provider, protocol) => {
    const plan = planStructured(target({ provider, model: "no-structured-sentinel", protocol }), REC);
    expectSame(fields(plan), want({ strategy: "prompt" }));
  });

  test("the subscription is native, so the sentinel gets row 11's forced tool", () => {
    const plan = planStructured(target({ model: "no-structured-sentinel", subscription: true }), REC);
    expectSame(
      fields(plan),
      want({ strategy: "forced_tool", tools: [{ schema: REC, strict: null }], named: "Record", parallel: false }),
    );
  });

  test("azure with the sentinel stays native (row 10)", () => {
    const azure = target({ provider: "azure", model: "no-structured-sentinel", protocol: "openai_chat" });
    const format = { schema: strictJsonSchema(REC), strict: true };
    expectSame(fields(planStructured(azure, REC)), want({ strategy: "json_schema", format }));
  });

  test("anthropic with the sentinel stays native (row 4)", () => {
    const plan = planStructured(
      target({ provider: "anthropic", model: "no-structured-sentinel", protocol: "anthropic_messages" }),
      REC,
    );
    const tools = [{ schema: REC, strict: null }];
    expectSame(fields(plan), want({ strategy: "forced_tool", tools, named: "Record" }));
  });
});

describe("planStructured: row 1a, a model without a wire protocol", () => {
  test("a ScriptedModel gets the tool parameters as its output format", () => {
    const plan = planStructured(structuredTarget(new ScriptedModel([])), SCHEMAS["run-agent:Record"]!);
    const recorded: JsonObject = {
      properties: { name: { type: "string" }, score: { type: "integer" } },
      required: ["name", "score"],
      type: "object",
    };
    expectSame(fields(plan), want({ strategy: "json_schema", format: { schema: recorded } }));
  });

  test.each([
    ["azure", null],
    ["anthropic", { thinking: { type: "enabled", budget_tokens: 1024n } }],
    ["google", null],
  ] as const)("row 1a comes before rows 2-10 for provider %s", (provider, block) => {
    for (const schema of [REC, OPEN, UNION]) {
      const plan = planStructured(target({ provider, model: "gpt-4", protocol: null, block }), schema);
      const name = schema["title"] as string;
      expectSame(
        fields(plan),
        want({ strategy: "json_schema", name, format: { schema: functionTool(schema, null).parameters } }),
      );
    }
  });
});

describe("planStructured: Anthropic Messages (rows 2-5)", () => {
  const anthropic = (model: string, block: JsonObject | null) =>
    target({ provider: "anthropic", model, protocol: "anthropic_messages", block });

  // An open schema under thinking can go neither as an output format (its grammar rejects an
  // open schema) nor as a forced tool (thinking forbids forcing one).
  for (const type of ["adaptive", "enabled", "between_tools"]) {
    for (const key of ["open_schema_test:object", "open_schema_test:Any", "open_schema_test:R"]) {
      test(`row 2: thinking ${type} with ${key} sends an unforced tool that keeps its open part`, () => {
        const schema = SCHEMAS[key]!;
        const block: JsonObject = { thinking: { type, budget_tokens: 1024n }, max_tokens: 4096n };
        const plan = planStructured(anthropic("claude-sonnet-4-5", block), schema);
        const name = schema["title"] as string;
        expectSame(fields(plan), want({ strategy: "unforced_tool", name, tools: [{ schema, strict: null }] }));
        expect(hasOpenSchema(plan.tools[0]!.parameters)).toBe(true);
      });
    }
  }

  test("row 2: the open record's tool keeps additionalProperties: true", () => {
    const block: JsonObject = { thinking: { type: "enabled", budget_tokens: 1024n } };
    const plan = planStructured(anthropic("claude-sonnet-4-5", block), SCHEMAS["open_schema_test:R"]!);
    const m = (plan.tools[0]!.parameters["properties"] as JsonObject)["m"] as JsonObject;
    expect(m["additionalProperties"]).toBe(true);
  });

  test.each([
    ["adaptive", "corpus:Rec"],
    ["adaptive", "corpus:Union"],
    ["enabled", "corpus:Rec"],
    ["enabled", "corpus:Union"],
  ])("row 3: thinking %s with the closed %s sends the Anthropic output schema", (type, key) => {
    const schema = SCHEMAS[key]!;
    const plan = planStructured(anthropic("claude-opus-5-5", { thinking: { type } }), schema);
    const name = schema["title"] as string;
    const format = { schema: anthropicOutputSchema(schema) };
    expectSame(fields(plan), want({ strategy: "json_schema", name, format }));
  });

  test("row 4: a model that accepts a forced tool gets one", () => {
    const plan = planStructured(anthropic("claude-sonnet-4-5", null), OPEN);
    const tools = [{ schema: OPEN, strict: null }];
    expectSame(fields(plan), want({ strategy: "forced_tool", tools, named: "Record" }));
  });

  test("row 5: a model that refuses a forced tool gets an unforced one", () => {
    const plan = planStructured(anthropic("claude-opus-5-5", null), REC);
    expectSame(fields(plan), want({ strategy: "unforced_tool", tools: [{ schema: REC, strict: null }] }));
  });

  test("thinking disabled falls to rows 4 and 5", () => {
    const block: JsonObject = { thinking: { type: "disabled" } };
    expect(planStructured(anthropic("claude-opus-4-8", block), OPEN).strategy).toBe("forced_tool");
    expect(planStructured(anthropic("claude-opus-5-5", block), OPEN).strategy).toBe("unforced_tool");
  });

  test("a thinking value that is not an object is not thinking", () => {
    expect(planStructured(anthropic("claude-opus-4-8", { thinking: true }), OPEN).strategy).toBe("forced_tool");
  });
});

describe("planStructured: Ollama and Gemini (rows 6-7)", () => {
  test("row 6: Ollama gets the schema as given, titles kept", () => {
    const plan = planStructured(target({ provider: "ollama", model: "llama3", protocol: "ollama_chat" }), UNION);
    expectSame(fields(plan), want({ strategy: "json_schema", name: "ScalarWrapper", format: { schema: UNION } }));
  });

  test("row 7: Gemini gets each const as a one-value enum", () => {
    const plan = planStructured(target({ provider: "google", model: "gemini-2.5-pro", protocol: "gemini" }), UNION);
    expectSame(
      fields(plan),
      want({ strategy: "json_schema", name: "ScalarWrapper", format: { schema: constAsEnum(UNION) } }),
    );
    const buy = (plan.outputFormat!.schema["$defs"] as JsonObject)["buy"] as JsonObject;
    expectSame((buy["properties"] as JsonObject)["kind"], { title: "Kind", type: "string", enum: ["buy"] });
  });
});

describe("planStructured: Azure (rows 8-10)", () => {
  const azure = (model: string, protocol: string) => target({ provider: "azure", model, protocol });

  test("row 8: an open schema on Chat is a non-strict forced tool without parallel calls", () => {
    const plan = planStructured(azure("gpt-4o", "openai_chat"), OPEN);
    expectSame(
      fields(plan),
      want({ strategy: "forced_tool", tools: [{ schema: OPEN, strict: null }], named: "Record", parallel: false }),
    );
  });

  test("row 8: an open schema on Responses sends strict: false and no parallel calls flag", () => {
    const plan = planStructured(azure("gpt-5-codex", "openai_responses"), OPEN);
    expectSame(
      fields(plan),
      want({ strategy: "forced_tool", tools: [{ schema: OPEN, strict: false }], named: "Record", parallel: null }),
    );
  });

  test.each(["gpt-4", "gpt-4-turbo", "gpt-35-turbo"])("row 9: legacy %s keeps the keywords in a tool", (model) => {
    const plan = planStructured(azure(model, "openai_chat"), BOUNDED);
    expectSame(
      fields(plan),
      want({ strategy: "forced_tool", tools: [{ schema: BOUNDED, strict: null }], named: "Record", parallel: false }),
    );
    const score = (plan.tools[0]!.parameters["properties"] as JsonObject)["score"] as JsonObject;
    expectSame(score, { maximum: 10n, minimum: 0n, type: "integer" });
  });

  test("row 9 on Responses has no parallel calls flag", () => {
    const plan = planStructured(azure("gpt-4", "openai_responses"), BOUNDED);
    expect(plan.strategy).toBe("forced_tool");
    expect(plan.parallelToolCalls).toBeNull();
  });

  test("row 10: a closed schema is strict JSON schema with the rejected keywords moved", () => {
    const plan = planStructured(azure("gpt-4o", "openai_chat"), BOUNDED);
    const moved = moveAzureKeywords(strictJsonSchema(BOUNDED))!;
    expectSame(fields(plan), want({ strategy: "json_schema", format: { schema: moved, strict: true } }));
    const score = (plan.outputFormat!.schema["properties"] as JsonObject)["score"] as JsonObject;
    expect(score["description"]).toBe("{minimum: 0, maximum: 10}");
  });

  test("row 10: a schema with nothing to move is the strict schema", () => {
    const plan = planStructured(azure("gpt-5", "openai_responses"), REC);
    const format = { schema: strictJsonSchema(REC), strict: true };
    expectSame(fields(plan), want({ strategy: "json_schema", format }));
  });
});

describe("planStructured: row 11, every other route", () => {
  test.each([
    ["deepseek", "deepseek-reasoner", "unforced_tool"],
    ["vllm", "deepseek-reasoner", "unforced_tool"],
    ["openrouter", "deepseek/deepseek-r1", "forced_tool"],
    ["qwen", "MiniMax-M2.5", "unforced_tool"],
    ["xai", "grok-4", "forced_tool"],
  ] as const)("%s %s is a %s", (provider, model, strategy) => {
    const plan = planStructured(target({ provider, model, protocol: "openai_chat" }), REC);
    const named = strategy === "forced_tool" ? "Record" : undefined;
    expectSame(fields(plan), want({ strategy, tools: [{ schema: REC, strict: null }], named, parallel: false }));
  });

  test("an open schema on Responses sends strict: false", () => {
    const plan = planStructured(target(), OPEN);
    expectSame(
      fields(plan),
      want({ strategy: "forced_tool", tools: [{ schema: OPEN, strict: false }], named: "Record", parallel: false }),
    );
  });

  test("a closed schema on Responses and an open one on Chat leave strict unset", () => {
    expect(planStructured(target(), REC).tools[0]!.strict).toBeNull();
    const chat = target({ provider: "openrouter", model: "openai/gpt-4o", protocol: "openai_chat" });
    expect(planStructured(chat, OPEN).tools[0]!.strict).toBeNull();
  });

  test("the subscription sends strict: false for an open schema and forces the tool", () => {
    const plan = planStructured(target({ model: "gpt-5.5", subscription: true }), SCHEMAS["open_schema_test:object"]!);
    expect(plan.strategy).toBe("forced_tool");
    expect(plan.tools[0]!.strict).toBe(false);
  });

  test("the subscription forces the tool even for a model without tool choice", () => {
    const plan = planStructured(target({ model: "deepseek-reasoner", subscription: true }), REC);
    expect(plan.strategy).toBe("forced_tool");
  });
});

describe("planStructured: the schema", () => {
  test.each<JsonObject>([{}, { title: "" }, { title: 1n }])("a schema without a title throws (%p)", (extra) => {
    const schema: JsonObject = { type: "object", properties: {}, ...extra };
    const error = thrown(() => planStructured(target(), schema));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe("a structured output schema needs a title, its name");
  });

  test("two plans of one schema are equal and the input schema is unchanged", () => {
    for (const schema of Object.values(SCHEMAS)) {
      const before = copyJson(schema);
      const first = planStructured(target(), schema);
      const second = planStructured(target(), schema);
      expectSame(fields(first), fields(second));
      expectSame(schema, before);
      expect(first.schema).not.toBe(schema);
      expectSame(first.schema, schema);
    }
  });

  // The copies above stand in for what an AGENT's `output:` sends; this pins them to the real
  // `typeToSchema`, so a change on either side shows here.
  test.each([
    ["corpus:Rec", "Rec"],
    ["corpus:Bounded", "Bounded"],
    ["corpus:Open", "Open"],
    ["corpus:Union", "Union(Buy, Sell)"],
  ])("typeToSchema emits %s, and its title names the plan", (key, text) => {
    const typedefs = readTypedefs({
      Rec: { answer: "str", score: "int" },
      Bounded: { answer: "str", score: "int(0..10)" },
      Open: { answer: "str", data: "object" },
      Buy: { kind: "Literal(buy)", x: "int" },
      Sell: { kind: "Literal(sell)", n: "int" },
    });
    const schema = typeToSchema(typeFor(text, typedefs))!;
    expectSame(schema, SCHEMAS[key]);
    expect(planStructured(target(), schema).name).toBe(schema.title as string);
  });
});

describe("planStructured: a date or datetime field's format hint reaches every route", () => {
  const DATE_HINT = "A date: YYYY-MM-DD, such as 2026-05-21.";
  const DATETIME_HINT =
    "A date and time: YYYY-MM-DDTHH:MM[:SS[.ffffff]][Z|+HH:MM|-HH:MM], such as 2026-05-21T14:30:00Z. " +
    "Seconds, a fraction of one to six digits and the offset are optional.";
  const schema = typeToSchema(typeFor("Due", readTypedefs({ Due: { on: "date", at: "datetime" } })))!;
  const thinking: JsonObject = { thinking: { type: "enabled", budget_tokens: 1024n } };
  const anthropic = (model: string, block: JsonObject | null) =>
    target({ provider: "anthropic", model, protocol: "anthropic_messages", block });

  /** Return the schema `plan` sends: its tool's parameters, its output format's, or the prompted one. */
  function sent(plan: StructuredPlan): JsonObject {
    return plan.tools[0]?.parameters ?? plan.outputFormat?.schema ?? plan.schema;
  }

  // Each route with the `format` the date field keeps; Azure's strict mode rejects `format`, so it
  // moves into the description after the hint.
  test.each([
    ["row 1, prompted", "prompt", target({ model: "no-structured-sentinel" }), "date"],
    ["row 3, anthropic thinking", "json_schema", anthropic("claude-opus-5-5", thinking), "date"],
    ["row 4, anthropic tool", "forced_tool", anthropic("claude-sonnet-4-5", null), "date"],
    ["row 6, ollama", "json_schema", target({ provider: "ollama", model: "llama3", protocol: "ollama_chat" }), "date"],
    ["row 7, gemini", "json_schema", target({ provider: "google", model: "gemini-2.5-pro", protocol: "gemini" }), "date"],
    ["row 9, azure legacy tool", "forced_tool", target({ provider: "azure", model: "gpt-4-turbo" }), "date"],
    ["row 10, azure strict", "json_schema", target({ provider: "azure", model: "gpt-4o" }), null],
    ["row 11, openai tool", "forced_tool", target(), "date"],
  ] as const)("%s", (_row, strategy, route, format) => {
    const plan = planStructured(route, schema);
    expect(plan.strategy).toBe(strategy);
    const properties = sent(plan)["properties"] as JsonObject;
    const on = properties["on"] as JsonObject;
    const at = properties["at"] as JsonObject;
    expect(on["format"]).toBe(format ?? undefined);
    expect(on["description"]).toBe(format === null ? `${DATE_HINT}\n\n{format: "date"}` : DATE_HINT);
    expect(at["format"]).toBeUndefined();
    expect(at["description"]).toBe(DATETIME_HINT);
  });
});

describe("forcesTool and sendsJsonSchemaAsTool", () => {
  test("the refusing prefixes", () => {
    expect([...FORCED_TOOL_REFUSING_PREFIXES]).toEqual([
      "claude-fable-5-1",
      "claude-opus-5-5",
      "claude-sonnet-5-5",
      "claude-mythos-5-1",
    ]);
  });

  test.each([...FORCED_TOOL_REFUSING_PREFIXES.flatMap((p) => [p, `${p}-20260101`]), "claude-opus-5-50"])(
    "%s refuses a forced tool",
    (model) => {
      expect(forcesTool(model)).toBe(false);
    },
  );

  test.each([
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-8",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
  ])("%s accepts a forced tool", (model) => {
    expect(forcesTool(model)).toBe(true);
  });

  test.each(["gpt-3.5-turbo", "gpt-35-turbo", "gpt-4", "gpt-4-turbo", "gpt-4-0613"])("%s is legacy", (model) => {
    expect(sendsJsonSchemaAsTool(model)).toBe(true);
  });

  test.each(["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-40", "gpt-5", "gpt-5.5", "o3", "my-deploy", ""])(
    "%p is not legacy",
    (model) => {
      expect(sendsJsonSchemaAsTool(model)).toBe(false);
    },
  );
});

describe("structuredTarget", () => {
  const routeModel = (provider: string, protocol: WireProtocol) =>
    new RouteModel({
      route: new Route({ provider, baseUrl: "https://api.example", path: "x", auth: new NoAuth() }),
      protocol,
    });

  test("Anthropic Messages: the block is read", () => {
    const block: JsonObject = { thinking: { type: "enabled", budget_tokens: 1024n } };
    const protocol = new AnthropicMessages({ model: "claude-opus-5-5", provider: "anthropic", block });
    const model = routeModel("anthropic", protocol);
    expectSame(structuredTarget(model), {
      provider: "anthropic",
      model: "claude-opus-5-5",
      protocol: "anthropic_messages",
      block,
      subscription: false,
    });
  });

  test("Anthropic Messages without a block", () => {
    const model = routeModel("anthropic", new AnthropicMessages({ model: "claude-opus-5-5", provider: "anthropic" }));
    expect(structuredTarget(model).block).toBeNull();
  });

  test.each([false, true])("Responses: subscription %p", (subscription) => {
    const block: JsonObject = { store: true };
    const protocol = new OpenAIResponses({ model: "gpt-5.5", provider: "openai", subscription, block });
    expectSame(structuredTarget(routeModel("openai", protocol)), {
      provider: "openai",
      model: "gpt-5.5",
      protocol: "openai_responses",
      block: null,
      subscription,
    });
  });

  // Only Anthropic's block is read: each of these carries one that must come out as `null`.
  test.each([
    [
      "deepseek",
      "openai_chat",
      new OpenAIChat({ model: "deepseek-chat", provider: "deepseek", block: { temperature: 0n } }),
    ],
    ["ollama", "ollama_chat", new OllamaChat({ model: "llama3", provider: "ollama", block: { options: {} } })],
    ["google", "gemini", new Gemini({ model: "gemini-2.5-pro", provider: "google", block: { generationConfig: {} } })],
  ] as const)("%s over %s: the block is not read", (provider, name, protocol) => {
    expectSame(structuredTarget(routeModel(provider, protocol)), {
      provider,
      model: protocol.model,
      protocol: name,
      block: null,
      subscription: false,
    });
  });

  test("a model without a wire protocol", () => {
    const model = new ScriptedModel([], { provider: "azure", model: "gpt-4o" });
    expectSame(structuredTarget(model), {
      provider: "azure",
      model: "gpt-4o",
      protocol: null,
      block: null,
      subscription: false,
    });
  });
});

describe("StructuredPlan: apply, instruction and correction", () => {
  const messages = [
    new Message({ role: "system", parts: [new TextPart("be brief")] }),
    new Message({ role: "user", parts: [new TextPart("rate it")] }),
  ];
  const native = PLANS.filter(([strategy]) => strategy !== "prompt");
  const [, prompted] = PLANS.find(([strategy]) => strategy === "prompt")!;

  test.each(native)("apply sets the %s plan's fields and keeps the messages", (_strategy, plan) => {
    const request = new Request({ messages, toolChoice: "auto", parallelToolCalls: true });
    const applied = plan.apply(request);
    expect(applied).not.toBe(request);
    expect(applied.messages).toHaveLength(messages.length);
    applied.messages.forEach((message, i) => expect(message).toBe(messages[i]!));
    expect([...applied.tools]).toEqual([...plan.tools]);
    expect(applied.toolChoice).toBe(plan.toolChoice);
    expect(applied.parallelToolCalls).toBe(plan.parallelToolCalls);
    expect(applied.outputFormat).toBe(plan.outputFormat);
    expect(request.toolChoice).toBe("auto");
    expect(request.parallelToolCalls).toBe(true);
  });

  test("apply returns the prompt request unchanged", () => {
    const request = new Request({ messages, toolChoice: "auto" });
    expect(prompted.apply(request)).toBe(request);
  });

  test.each(PLANS)("apply refuses a request with tools or an output format (%s)", (_strategy, plan) => {
    const tooled = new Request({ messages, tools: [functionTool(OPEN, null)] });
    const formatted = new Request({ messages, outputFormat: new OutputFormat({ name: "Other", schema: OPEN }) });
    for (const request of [tooled, formatted]) {
      const error = thrown(() => plan.apply(request));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe("a structured request carries only the schema's tool");
    }
  });

  test("the prompt instruction writes the schema as compact JSON", () => {
    expect(prompted.instruction()).toBe(
      "Respond with ONLY a JSON object matching this JSON schema (no prose, no code fences):\n" +
        '{"properties":{"answer":{"title":"Answer","type":"string"},"score":{"title":"Score","type":"integer"}},' +
        '"required":["answer","score"],"title":"Record","type":"object"}',
    );
  });

  test.each(native)("the %s plan has no instruction", (_strategy, plan) => {
    const error = thrown(() => plan.instruction());
    expect(error.constructor).toBe(Error);
    expect(error.message).toBe("only the prompt strategy has an instruction");
  });

  test("the prompt correction asks for a JSON object", () => {
    expect(prompted.correction("not json")).toBe(
      "Your previous output was invalid: not json. Respond with ONLY a JSON object matching the schema.",
    );
  });

  test.each(native)("the %s correction asks for valid data", (_strategy, plan) => {
    // The retry text sent after a reply that did not call the tool.
    expect(plan.correction("the reply did not call the Record tool; call it with the answer")).toBe(
      "Your previous output was invalid: the reply did not call the Record tool; call it with the answer. " +
        "Respond with valid data matching the schema.",
    );
  });
});

describe("StructuredPlan: parse a tool reply", () => {
  const toolPlans = PLANS.filter(([strategy]) => strategy === "forced_tool" || strategy === "unforced_tool");
  const call = (name: string, args: JsonObject | null, id = "c1") =>
    new ToolCallPart({ id, name, arguments: args, rawArguments: args === null ? "{bad" : null });

  test.each(toolPlans)("%s: one call gives its arguments", (_strategy, plan) => {
    expect(plan.parse(reply([call("Record", { answer: "x", score: 1n })]))).toEqual({ answer: "x", score: 1n });
  });

  test.each(toolPlans)("%s: of two calls of the tool the first wins", (_strategy, plan) => {
    const parts = [call("Record", { answer: "a", score: 1n }), call("Record", { answer: "b", score: 2n }, "c2")];
    expect(plan.parse(reply(parts))).toEqual({ answer: "a", score: 1n });
  });

  test.each(toolPlans)("%s: an invalid call counts as none, so the valid one after it wins", (_strategy, plan) => {
    const parts = [call("Record", null), call("Record", { answer: "b", score: 2n }, "c2")];
    expect(plan.parse(reply(parts))).toEqual({ answer: "b", score: 2n });
  });

  test.each(toolPlans)("%s: another tool's call first is skipped", (_strategy, plan) => {
    const parts = [call("search", { q: "x" }), call("Record", { answer: "b", score: 2n }, "c2")];
    expect(plan.parse(reply(parts))).toEqual({ answer: "b", score: 2n });
  });

  test.each(toolPlans)("%s: the arguments are returned unvalidated", (_strategy, plan) => {
    expect(plan.parse(reply([call("Record", { answer: "x", score: "notint" })]))).toEqual({
      answer: "x",
      score: "notint",
    });
  });

  test.each(toolPlans)("%s: arguments nesting past maxValueNesting parse from their raw text", (_strategy, plan) => {
    install({ ...BUILTIN, maxValueNesting: 3 });
    // The adapter's parse refused them, so the call holds only the raw text.
    const raw = '{"answer": {"b": [[1]]}}';
    const deep = new ToolCallPart({ id: "c1", name: "Record", arguments: null, rawArguments: raw });
    try {
      expect(plan.parse(reply([deep]))).toEqual({ answer: { b: [[1n]] } });
    } finally {
      install(BUILTIN);
    }
  });

  const missing: [string, Part[]][] = [
    ["text only", [new TextPart('{"answer": "x", "score": 1}')]],
    ["only another tool's call", [call("search", { q: "x" })]],
    ["only an invalid call", [call("Record", null)]],
  ];
  for (const [strategy, plan] of toolPlans) {
    test.each(missing)(`${strategy}: %s is a missing tool call`, (_label, parts) => {
      const error = thrown(() => plan.parse(reply(parts)));
      expect(error).toBeInstanceOf(MissingToolCallError);
      expect(error.message).toBe("the reply did not call the Record tool; call it with the answer");
    });
  }
});

describe("StructuredPlan: parse a text reply", () => {
  afterEach(() => install(BUILTIN));

  const textPlans = PLANS.filter(([strategy]) => strategy === "json_schema" || strategy === "prompt");
  const fence = "```";

  const valid: [string, string, JsonObject][] = [
    ["plain JSON", '{"answer": "x", "score": 1}', { answer: "x", score: 1n }],
    ["a json fence", `${fence}json\n{"answer": "x"}\n${fence}`, { answer: "x" }],
    ["a bare fence", `${fence}\n{"answer": "x"}\n${fence}`, { answer: "x" }],
    ["a JSON info string", `${fence}JSON\n{"answer": "x"}\n${fence}`, { answer: "x" }],
    ["a jsonc info string", `${fence}jsonc\n{"answer": "x"}\n${fence}`, { answer: "x" }],
    ["whitespace and backticks around", ' \n\t`{"answer": "x"}`\r\n ', { answer: "x" }],
    [
      "prose before and after a fence",
      `Here it is:\n${fence}json\n{"answer": "x"}\n${fence}\nHope that helps.`,
      { answer: "x" },
    ],
    // Prose first, so the whole text fails and the fenced text must run to the very end.
    ["an unclosed fence after prose", `Here it is:\n${fence}json\n{"answer": "x"}`, { answer: "x" }],
    [
      "a fenced JSON whose string holds a fence",
      `${fence}json\n{"code": "${fence}py\\nx\\n${fence}"}\n${fence}`,
      { code: `${fence}py\nx\n${fence}` },
    ],
    ["a raw tab inside a string", '{"answer": "a\tb"}', { answer: "a\tb" }],
    [
      "a float and a nested value",
      '{"a": 1.5, "b": [true, null, {"c": "d"}]}',
      { a: 1.5, b: [true, null, { c: "d" }] },
    ],
  ];
  for (const [strategy, plan] of textPlans) {
    test.each(valid)(`${strategy}: %s parses`, (_label, text, value) => {
      expectSame(plan.parse(reply([new TextPart(text)])), value);
    });
  }

  test.each(textPlans)("%s: a 5000-digit int is read whole as a bigint", (_strategy, plan) => {
    const digits = "7".repeat(5000);
    expect(plan.parse(reply([new TextPart(`{"score": ${digits}}`)]))).toEqual({ score: BigInt(digits) });
  });

  test.each(textPlans)("%s: two text parts are joined with a newline", (_strategy, plan) => {
    // The parts split inside a string, so the separator shows in the value ("" would give "xy").
    const parts = [new TextPart('{"answer": "x'), new TextPart('y"}')];
    expect(plan.parse(reply(parts))).toEqual({ answer: "x\ny" });
  });

  test.each(textPlans)("%s: reasoning parts are left out", (_strategy, plan) => {
    const parts = [new ReasoningPart("thinking {not json"), new TextPart('{"answer": "x"}')];
    expect(plan.parse(reply(parts))).toEqual({ answer: "x" });
  });

  const invalid: [string, string][] = [
    ["not json", "not json"],
    ["truncated JSON", '{"answer": "x", "sco'],
    ["two objects", '{"answer": "x"} {"answer": "y"}'],
    ["a JSON array", '[{"answer": "x"}]'],
    ["empty text", ""],
    ["only backticks and whitespace", " ``` \n"],
    ["an empty fence", `${fence}json\n${fence}`],
  ];
  for (const [strategy, plan] of textPlans) {
    test.each(invalid)(`${strategy}: %s is invalid`, (_label, text) => {
      const error = thrown(() => plan.parse(reply([new TextPart(text)])));
      expect(error.constructor).toBe(StructuredOutputError);
      expect(error.message).toBe(`Invalid json output: ${text.trim()}`);
    });
  }

  test.each(textPlans)("%s: the error trims only whitespace from the text", (_strategy, plan) => {
    const error = thrown(() => plan.parse(reply([new TextPart("\n  `not json`  \n")])));
    expect(error.message).toBe("Invalid json output: `not json`");
  });

  test.each(textPlans)("%s: a reply nesting past maxValueNesting parses; the caller checks its depth", (_strategy, plan) => {
    install({ ...BUILTIN, maxValueNesting: 3 });
    expect(plan.parse(reply([new TextPart('{"a": {"b": [[1]]}}')]))).toEqual({ a: { b: [[1n]] } });
  });

  test.each(textPlans)("%s: a refusal with no text names the refusal", (_strategy, plan) => {
    const error = thrown(() => plan.parse(reply([new RefusalPart("I can't help with that.")])));
    expect(error.constructor).toBe(StructuredOutputError);
    expect(error.message).toBe("the model refused: I can't help with that.");
  });

  test.each(textPlans)("%s: refusals beside invalid text are joined with a newline", (_strategy, plan) => {
    const parts = [new RefusalPart("No."), new TextPart("not json"), new RefusalPart("Still no.")];
    expect(thrown(() => plan.parse(reply(parts))).message).toBe("the model refused: No.\nStill no.");
  });

  test.each(textPlans)("%s: a refusal beside valid JSON gives the value", (_strategy, plan) => {
    const parts = [new RefusalPart("I can't help with that."), new TextPart('{"answer": "x"}')];
    expect(plan.parse(reply(parts))).toEqual({ answer: "x" });
  });

  test.each(textPlans)("%s: tool calls are ignored", (_strategy, plan) => {
    const parts = [new ToolCallPart({ id: "c1", name: "Record", arguments: { answer: "t" } }), new TextPart("nope")];
    expect(thrown(() => plan.parse(reply(parts))).message).toBe("Invalid json output: nope");
  });
});

describe("structured-output errors", () => {
  test("MissingToolCallError is a StructuredOutputError is an Error", () => {
    const missing = new MissingToolCallError("m");
    expect(missing).toBeInstanceOf(StructuredOutputError);
    expect(missing).toBeInstanceOf(Error);
    expect(missing.name).toBe("MissingToolCallError");
    expect(missing.message).toBe("m");
    const invalid = new StructuredOutputError("i");
    expect(invalid).toBeInstanceOf(Error);
    expect(invalid).not.toBeInstanceOf(MissingToolCallError);
    expect(invalid.name).toBe("StructuredOutputError");
    expect(String(invalid)).toBe("StructuredOutputError: i");
  });
});
