/**
 * Tests for the structured-output strategy choice in `src/llm/structured.ts`: `structuredTarget`,
 * `planStructured` (one test per row of the selection table, first match wins), `forcesTool` and
 * `sendsJsonSchemaAsTool`, ported from Python yian's Anthropic, Azure and open-schema tests.
 */

import { describe, expect, test } from "bun:test";

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
import { copyJson, NamedTool } from "../../src/llm/messages.ts";
import { AnthropicMessages } from "../../src/llm/protocols/anthropic_messages.ts";
import { Gemini } from "../../src/llm/protocols/gemini.ts";
import { OllamaChat } from "../../src/llm/protocols/ollama_chat.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { OpenAIResponses } from "../../src/llm/protocols/openai_responses.ts";
import { Route } from "../../src/llm/route.ts";
import {
  FORCED_TOOL_REFUSING_PREFIXES,
  forcesTool,
  planStructured,
  sendsJsonSchemaAsTool,
  type StructuredPlan,
  type StructuredTarget,
  structuredTarget,
} from "../../src/llm/structured.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

/** Python's `type_to_schema(...).model_json_schema()` for the cases below: ints are `bigint`s. */
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
  strategy: string;
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

describe("planStructured: row 1, no native structured output", () => {
  test.each([
    ["xai", "openai_chat"],
    ["openai", "openai_responses"],
    ["openai", null],
    ["xai", null],
  ] as const)("the sentinel on %s over %s is prompted", (provider, protocol) => {
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
    const plan = planStructured(azure, REC);
    expect(plan.strategy).toBe("json_schema");
    expect(plan.outputFormat?.strict).toBe(true);
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

  test.each(["corpus:Rec", "corpus:Union"])("row 3: thinking with the closed %s sends the Anthropic output schema", (key) => {
    const schema = SCHEMAS[key]!;
    const plan = planStructured(anthropic("claude-opus-5-5", { thinking: { type: "adaptive" } }), schema);
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

  test.each([
    ["deepseek", new OpenAIChat({ model: "deepseek-chat", provider: "deepseek" }), "openai_chat"],
    ["ollama", new OllamaChat({ model: "llama3", provider: "ollama", block: { options: {} } }), "ollama_chat"],
    ["google", new Gemini({ model: "gemini-2.5-pro", provider: "google", block: { generationConfig: {} } }), "gemini"],
  ] as const)("%s over %s", (provider, protocol, name) => {
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
