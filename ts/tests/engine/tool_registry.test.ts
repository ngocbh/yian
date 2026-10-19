/**
 * The tool registry and `ToolSet`, a port of `tests/engine/test_tool_registry.py` with rows for
 * each construction and group error, `expand`, `forRun`, `toolIds`, `groups`, `validate`,
 * `isConcurrencySafe` and the `ask_user` options cleaning, plus `test_injected_arguments_are_not_inputs`
 * and `test_function_tool_types` from `test_tool_types.py`. Not ported:
 * `test_invocation_supports_pydantic_v1_schema` and `test_invocation_rejects_unvalidated_json_schema_tools`
 * (TS has only `FunctionTool`s), `test_install_ask_user_is_removed` (it never existed in TS),
 * `test_tool_node_uses_public_invocation_path` (needs the TOOL node), and the LangChain binding
 * half of `test_ask_user_openai_schema_offers_optional_string_options`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import {
  type AskUserRequest,
  invokeTool,
  invokeToolTrusted,
  registeredTools,
  registerTool,
  resolveTools,
  TOOL_REGISTRY,
  ToolSet,
  type ToolEntry,
} from "../../src/tools/registry.ts";
import { toolTypes } from "../../src/tools/types.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";

let saved: [string, FunctionTool][] = [];

// Keep process-global registrations isolated between tests.
beforeEach(() => {
  saved = [...TOOL_REGISTRY];
});
afterEach(() => {
  TOOL_REGISTRY.clear();
  for (const [id, tool] of saved) TOOL_REGISTRY.set(id, tool);
});

/** Build a `FunctionTool` named `name` that takes one string `text`. */
function tool(name: string, result: unknown = "done"): FunctionTool {
  return new FunctionTool({
    name,
    description: "Echo.",
    parametersJsonSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: () => result,
  });
}

/** Return the message of the error `fn` throws, after checking its class. */
function thrown(fn: () => unknown, kind: typeof TypeError | typeof RangeError): string {
  try {
    fn();
  } catch (error) {
    expect((error as Error).constructor).toBe(kind);
    return (error as Error).message;
  }
  throw new Error("expected an error");
}

/** Return the message of the error `promise` rejects with, after checking its class. */
async function rejected(promise: Promise<unknown>, kind: typeof TypeError | typeof RangeError): Promise<string> {
  try {
    await promise;
  } catch (error) {
    expect((error as Error).constructor).toBe(kind);
    return (error as Error).message;
  }
  throw new Error("expected a rejection");
}

/** Build an `ask_user` handler that records each request and returns `answer`. */
function recordingHandler(seen: AskUserRequest[], answer: unknown = "answer") {
  return (request: AskUserRequest) => {
    seen.push(request);
    return answer;
  };
}

/** Build a tool set from a mapping the types would refuse. */
function loose(tools: unknown): ToolSet {
  return new ToolSet(tools as Record<string, ToolEntry>);
}

describe("registerTool", () => {
  test("test_generic_registration_rejects_reserved_ask_user_id", () => {
    expect(thrown(() => registerTool("ask_user", tool("ask_user")), RangeError)).toBe(
      "tool id 'ask_user' is reserved; pass tools: {...registeredTools(), ask_user: handler} to runFlow or resumeFlow",
    );
    expect(TOOL_REGISTRY.has("ask_user")).toBe(false);
  });

  test("test_register_tool_rejects_what_it_cannot_register", () => {
    const scale = (value: number) => value * 2;
    for (const target of [scale, { name: "register_scale" }]) {
      const message = thrown(() => registerTool("register_scale", target as unknown as FunctionTool), TypeError);
      expect(message).toContain("'register_scale'");
      expect(message).toContain("FunctionTool");
      expect(message).toContain("tools");
      expect(TOOL_REGISTRY.has("register_scale")).toBe(false);
    }
    expect(thrown(() => registerTool("register_scale", scale as unknown as FunctionTool), TypeError)).toBe(
      "registerTool('register_scale') got a function; it accepts a FunctionTool. Pass anything else in the run's tools mapping instead",
    );
    expect(
      thrown(() => registerTool("register_scale", { name: "register_scale" } as unknown as FunctionTool), TypeError),
    ).toBe(
      "registerTool('register_scale') got a Object; it accepts a FunctionTool. Pass anything else in the run's tools mapping instead",
    );
  });

  test("rejects a duplicate id", () => {
    registerTool("echo", tool("echo"));
    expect(thrown(() => registerTool("echo", tool("echo")), RangeError)).toBe("Duplicate tool id 'echo'");
  });

  test("checks the reserved id before a duplicate, and a duplicate before the target", () => {
    TOOL_REGISTRY.set("ask_user", tool("ask_user"));
    expect(thrown(() => registerTool("ask_user", tool("ask_user")), RangeError)).toBe(
      "tool id 'ask_user' is reserved; pass tools: {...registeredTools(), ask_user: handler} to runFlow or resumeFlow",
    );
    registerTool("echo", tool("echo"));
    expect(thrown(() => registerTool("echo", null as unknown as FunctionTool), RangeError)).toBe(
      "Duplicate tool id 'echo'",
    );
  });

  test("rejects a FunctionTool named differently from its id", () => {
    expect(thrown(() => registerTool("other", tool("echo")), RangeError)).toBe(
      "tool id 'other' names a FunctionTool named 'echo'; its name must match its id",
    );
    expect(TOOL_REGISTRY.has("other")).toBe(false);
  });

  test("an empty, null or undefined descriptionOverride keeps the tool's description", () => {
    for (const [id, override] of [["a", ""], ["b", null], ["c", undefined]] as const) {
      const ft = tool(id);
      expect(registerTool(id, ft, { descriptionOverride: override as string | undefined })).toBe(ft);
      expect(TOOL_REGISTRY.get(id)!.description).toBe("Echo.");
    }
  });

  test("a non-string descriptionOverride is a TypeError", () => {
    expect(
      thrown(() => registerTool("echo", tool("echo"), { descriptionOverride: 3 as unknown as string }), TypeError),
    ).toBe("registerTool('echo') descriptionOverride must be a string");
    expect(TOOL_REGISTRY.has("echo")).toBe(false);
  });
});

describe("registeredTools and resolveTools", () => {
  test("registeredTools is a copy", () => {
    const ft = tool("echo");
    registerTool("echo", ft);
    const copy = registeredTools();
    expect(copy).toEqual({ echo: ft });
    expect(Object.getPrototypeOf(copy)).toBe(Object.prototype);
    delete copy.echo;
    expect(TOOL_REGISTRY.get("echo")).toBe(ft);
  });

  test("resolveTools returns registered tools in order", () => {
    const a = tool("a");
    const b = tool("b");
    registerTool("a", a);
    registerTool("b", b);
    expect(resolveTools(["b", "a"])).toEqual([b, a]);
  });

  test("resolveTools rejects an unknown id", () => {
    registerTool("b", tool("b"));
    registerTool("a", tool("a"));
    expect(thrown(() => resolveTools(["a", "zzz"]), RangeError)).toBe(
      'Unknown tool id \'zzz\'. Known tools: ["a","b"]',
    );
  });
});

describe("ToolSet construction", () => {
  const SERVERLESS_ROWS: [string, unknown, typeof TypeError | typeof RangeError, string][] = [
    ["an array", [tool("echo")], TypeError, "tools must be a record of tool id to tool, or null"],
    ["a Map", new Map(), TypeError, "tools must be a record of tool id to tool, or null"],
    ["an empty key", { "": tool("echo") }, TypeError, "tool ids in tools must be non-empty strings"],
    [
      "a FunctionTool as ask_user",
      { ask_user: tool("ask_user") },
      TypeError,
      "tools['ask_user'] must be a plain function handler; the engine supplies its schema",
    ],
    [
      "a string as ask_user",
      { ask_user: "ask" },
      TypeError,
      "tools['ask_user'] must be a plain function handler; the engine supplies its schema",
    ],
    [
      "a FunctionTool under another key",
      { other: tool("echo") },
      RangeError,
      "tool 'other' is a FunctionTool named 'echo'; its name must match its key",
    ],
    [
      "a non-FunctionTool group item",
      { g: [tool("a"), 3] },
      TypeError,
      "tools['g'][1] must be a FunctionTool, got number",
    ],
    [
      "a function",
      { echo: () => 1 },
      TypeError,
      "tools['echo'] must be a FunctionTool, an MCP server, or an array of FunctionTools, got function",
    ],
    [
      "a plain object",
      { echo: {} },
      TypeError,
      "tools['echo'] must be a FunctionTool, an MCP server, or an array of FunctionTools, got Object",
    ],
    [
      "an ask_user group tool",
      { g: [tool("ask_user")] },
      RangeError,
      "group 'g' has a tool named 'ask_user', a reserved name; pass the handler as the top-level tools['ask_user'] entry",
    ],
    [
      "a load_skill group tool",
      { g: [tool("load_skill")] },
      RangeError,
      "group 'g' has a tool named 'load_skill', a reserved name; granting skills binds it",
    ],
    [
      "a group tool named like a top-level key",
      { echo: tool("echo"), g: [tool("echo")] },
      RangeError,
      "tool 'echo' in group 'g' clashes with tools['echo']",
    ],
    [
      "a group tool named like another group",
      { g: [tool("h")], h: [] },
      RangeError,
      "tool 'h' in group 'g' clashes with tools['h']",
    ],
    [
      "two tools under one name in a group",
      { g: [tool("echo"), tool("echo")] },
      RangeError,
      "tool 'echo' appears twice in group 'g' as different tools",
    ],
    [
      "two tools under one name in two groups",
      { a: [tool("echo")], b: [tool("echo")] },
      RangeError,
      "tool 'echo' is in both groups 'a' and 'b' as different tools",
    ],
  ];

  for (const [label, tools, kind, message] of SERVERLESS_ROWS) {
    test(`rejects ${label}`, () => {
      expect(thrown(() => loose(tools), kind)).toBe(message);
    });
  }

  test("null and undefined mean the registry", () => {
    registerTool("echo", tool("echo"));
    expect(new ToolSet(null).toolIds).toEqual(["echo"]);
    expect(new ToolSet().toolIds).toEqual(["echo"]);
  });

  test("the same tool may sit in several groups and twice in one", () => {
    const shared = tool("shared");
    const tools = new ToolSet({ a: [shared, shared, tool("x")], b: [shared] });
    expect(tools.groups).toEqual(["a", "b"]);
    expect(tools.expand(["a"])).toEqual(["shared", "x"]);
    expect(tools.expand(["b"])).toEqual(["shared"]);
    expect(tools.resolve(["shared"])).toEqual([shared]);
  });

  test("an empty group grants nothing", () => {
    const tools = new ToolSet({ company: [] });
    expect(tools.groups).toEqual(["company"]);
    expect(tools.expand(["company"])).toEqual([]);
  });

  test("the ask_user handler becomes the fixed ask_user tool", async () => {
    const seen: unknown[][] = [];
    const handler = (request: unknown, ctx: unknown) => {
      seen.push([request, ctx]);
      return "answer";
    };
    const tools = new ToolSet({ ask_user: handler });
    const [askUser] = tools.resolve(["ask_user"]);
    expect(askUser!.name).toBe("ask_user");
    expect(askUser!.description).toBe(
      "Ask the user one short question and wait for the answer. Use only when you need a decision or information that only the user can provide. Put the context they need in one or two sentences. When there are clear choices, pass 2-4 short `options`; the user can always type their own answer.",
    );
    expect(JSON.stringify(askUser!.parametersJsonSchema)).toBe(
      '{"properties":{"question":{"type":"string"},"options":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"default":null,"description":"Optional 2-4 short answer choices. The user can always type a different answer."}},"required":["question"],"type":"object"}',
    );
    expect(await tools.invoke("ask_user", { question: "Approve?" }, { callId: "c" })).toBe("answer");
    expect(seen).toHaveLength(1);
    const [request, ctx] = seen[0]!;
    expect(request).toEqual({
      question: "Approve?",
      options: null,
      questions: null,
      answerSchema: null,
      title: null,
      nodeId: null,
    });
    expect(ctx).toBeInstanceOf(ToolContext);
    expect((ctx as ToolContext).callId).toBe("c");
    expect((ctx as ToolContext).toolName).toBe("ask_user");
    expect(TOOL_REGISTRY.has("ask_user")).toBe(false);
  });
});

describe("ToolSet lookup", () => {
  test("test_ask_user_never_falls_back_to_the_global_registry", () => {
    TOOL_REGISTRY.set("ask_user", tool("ask_user", "global answer"));
    for (const tools of [new ToolSet(), new ToolSet({})]) {
      expect(thrown(() => tools.resolve(["ask_user"]), RangeError)).toBe(
        "no ask_user handler for this run; pass tools: {...registeredTools(), ask_user: handler} to runFlow or resumeFlow",
      );
    }
  });

  test("a group id is not a tool", () => {
    expect(thrown(() => new ToolSet({ toolbox: [tool("a")] }).resolve(["toolbox"]), RangeError)).toBe(
      "'toolbox' is a tool group, not a tool; grant it in an AGENT node's tools: list",
    );
  });

  test("an unknown id lists the mapping's tools", () => {
    const tools = new ToolSet({ b: tool("b"), g: [tool("a")] });
    expect(thrown(() => tools.resolve(["zzz"]), RangeError)).toBe(
      'unknown tool id \'zzz\'; known tools: ["a","b"]',
    );
  });

  test("an unknown id lists the registry's tools when there is no mapping", () => {
    registerTool("echo", tool("echo"));
    expect(thrown(() => new ToolSet().resolve(["zzz"]), RangeError)).toBe(
      'unknown tool id \'zzz\'; known tools: ["echo"]',
    );
  });

  test("an inherited property name is an unknown id", () => {
    expect(thrown(() => new ToolSet({ echo: tool("echo") }).resolve(["constructor"]), RangeError)).toBe(
      'unknown tool id \'constructor\'; known tools: ["echo"]',
    );
    expect(thrown(() => new ToolSet().resolve(["constructor"]), RangeError)).toContain(
      "unknown tool id 'constructor'",
    );
  });

  test("the registry is read at lookup time", () => {
    const tools = new ToolSet();
    const ft = tool("late");
    registerTool("late", ft);
    expect(tools.resolve(["late"])).toEqual([ft]);
    expect(tools.toolIds).toEqual(["late"]);
  });

  test("resolve keeps the given order", () => {
    const a = tool("a");
    const b = tool("b");
    expect(new ToolSet({ a, g: [b] }).resolve(["b", "a"])).toEqual([b, a]);
  });

  test("expand deduplicates overlapping and repeated grants", () => {
    const shared = tool("shared");
    const tools = new ToolSet({ a: [tool("x"), shared], b: [shared, tool("y")], solo: tool("solo") });
    expect(tools.expand(["a", "b", "a", "solo", "solo", "unknown"])).toEqual([
      "x",
      "shared",
      "y",
      "solo",
      "unknown",
    ]);
  });

  test("toolIds is sorted, from the mapping and its groups", () => {
    const tools = new ToolSet({ b: tool("b"), g: [tool("a"), tool("C")], ask_user: () => null });
    expect(tools.toolIds).toEqual(["C", "a", "ask_user", "b"]);
    expect(tools.groups).toEqual(["g"]);
  });

  test("toolIds is the registry's when there is no mapping", () => {
    registerTool("b", tool("b"));
    registerTool("a", tool("a"));
    expect(new ToolSet().toolIds).toEqual(["a", "b"]);
    expect(new ToolSet().groups).toEqual([]);
  });
});

describe("ToolSet.forRun", () => {
  test("returns a bound copy and leaves the original unbound", () => {
    const db = { name: "db" };
    const ft = tool("place");
    const original = new ToolSet({ place: ft, g: [tool("a")] }, { context: db });
    const controller = new AbortController();

    const bound = original.forRun("run-1", { signal: controller.signal });

    expect(bound).not.toBe(original);
    expect(original.runId).toBe("");
    expect(original.signal).toBeUndefined();
    expect(bound.runId).toBe("run-1");
    expect(bound.signal).toBe(controller.signal);
    expect(original.context).toBe(db);
    expect(bound.context).toBe(db);
    expect(bound.groups).toEqual(["g"]);
    expect(bound.resolve(["place"])).toEqual([ft]);
    expect(bound.forRun("run-2").signal).toBeUndefined();
  });

  test("a bound copy of the registry still reads it", () => {
    const bound = new ToolSet().forRun("run-9");
    const ft = tool("late");
    registerTool("late", ft);
    expect(bound.resolve(["late"])).toEqual([ft]);
  });

  test("test_for_run_keeps_the_subclass", () => {
    class Sub extends ToolSet {}
    const bound = new Sub({ place: tool("place") }).forRun("r");
    expect(bound).toBeInstanceOf(Sub);
    expect(Object.getPrototypeOf(bound)).toBe(Sub.prototype);
    expect(bound.runId).toBe("r");
  });
});

describe("ask_user invocation", () => {
  const NO_FIELDS = { options: null, questions: null, answerSchema: null, title: null, nodeId: null };

  test("test_ask_user_public_schema_contains_question_and_options", async () => {
    const seen: AskUserRequest[] = [];
    const tools = new ToolSet({ ask_user: recordingHandler(seen) });

    const [askUser] = tools.resolve(["ask_user"]);
    expect(Object.keys(askUser!.parametersJsonSchema.properties as object)).toEqual(["question", "options"]);
    expect(await tools.invoke("ask_user", { question: "Approve?" })).toBe("answer");

    expect(seen).toEqual([{ question: "Approve?", ...NO_FIELDS }]);
    expect(TOOL_REGISTRY.has("ask_user")).toBe(false);
  });

  test("test_ask_user_openai_schema_offers_optional_string_options", () => {
    const [askUser] = new ToolSet({ ask_user: recordingHandler([]) }).resolve(["ask_user"]);
    const schema = askUser!.parametersJsonSchema as any;

    expect(schema.required).toEqual(["question"]);
    expect(schema.properties.options.anyOf).toEqual([{ type: "array", items: { type: "string" } }, { type: "null" }]);
    expect(schema.properties.options.description).toContain("user can always type a different answer");
  });

  const INJECTED: [string, unknown][] = [
    ["questions", [{ id: "reason", prompt: "Why?" }]],
    ["answerSchema", [{ name: "reason", type: "str" }]],
    ["title", "Approval"],
    ["nodeId", "gate"],
    ["unexpected", true],
  ];
  for (const [key, value] of INJECTED) {
    test(`test_public_invocation_rejects_injected_and_unknown_keys[${key}]`, async () => {
      const seen: AskUserRequest[] = [];
      const tools = new ToolSet({ ask_user: recordingHandler(seen) });

      expect(await rejected(tools.invoke("ask_user", { question: "Approve?", [key]: value }), RangeError)).toBe(
        `tool 'ask_user' received unsupported argument(s) ["${key}"] on its public invocation path; ` +
          'allowed: ["options","question"]',
      );
      expect(seen).toEqual([]);
    });
  }

  for (const [label, invoke] of [
    ["invokeTool", invokeTool],
    ["invokeToolTrusted", invokeToolTrusted],
  ] as const) {
    test(`test_invocation_rejects_unknown_tool[${label}]`, async () => {
      expect(await rejected(invoke("not_registered", {}), RangeError)).toContain("unknown tool id 'not_registered'");
    });
  }

  test("test_trusted_invocation_passes_structured_metadata", async () => {
    const seen: AskUserRequest[] = [];
    const metadata = {
      question: "Review the proposal",
      questions: [{ id: "decision", prompt: "Approve?" }],
      answerSchema: [{ name: "decision", type: "bool" }],
      title: "Human review",
      nodeId: "approval",
    };
    const tools = new ToolSet({ ask_user: recordingHandler(seen, { decision: true }) });

    expect(await tools.invokeTrusted("ask_user", metadata)).toEqual({ decision: true });

    expect(seen).toEqual([{ ...metadata, options: null }]);
  });

  test("test_trusted_invocation_rejects_unknown_metadata", async () => {
    const tools = new ToolSet({ ask_user: recordingHandler([]) });

    const inputs = { question: "Approve?", not_metadata: "model supplied" };
    expect(await rejected(tools.invokeTrusted("ask_user", inputs), RangeError)).toBe(
      "tool 'ask_user' received unsupported argument(s) [\"not_metadata\"] on its trusted invocation path; " +
        'allowed: ["answerSchema","nodeId","options","question","questions","title"]',
    );
  });

  test("the trusted path checks the metadata's types", async () => {
    const tools = new ToolSet({ ask_user: recordingHandler([]) });

    await expect(tools.invokeTrusted("ask_user", { question: "Approve?", title: 3 })).rejects.toThrow(
      /^tool 'ask_user' argument 'title': /,
    );
  });

  test("test_ask_user_never_falls_back_to_the_global_registry (invoke)", async () => {
    TOOL_REGISTRY.set("ask_user", tool("ask_user", "global answer"));
    for (const tools of [new ToolSet(), new ToolSet({})]) {
      expect(await rejected(tools.invoke("ask_user", { question: "Approve?" }), RangeError)).toBe(
        "no ask_user handler for this run; pass tools: {...registeredTools(), ask_user: handler} to runFlow or resumeFlow",
      );
    }
  });

  const OPTION_ROWS: [string, unknown, string[] | null][] = [
    ["a JSON list string", '["A","B"]', ["A", "B"]],
    ["a plain string", "A, B", ["A, B"]],
    ["mixed items", [1, "", " a ", "A", null], ["a"]],
    ["labels equal under case folding", ["Straße", "STRASSE"], ["Straße"]],
    ["bad JSON", '["a","a"', ['["a","a"']],
    ["a record", { x: 1 }, null],
  ];
  for (const [label, options, expected] of OPTION_ROWS) {
    test(`cleans options: ${label}`, async () => {
      const seen: AskUserRequest[] = [];
      await new ToolSet({ ask_user: recordingHandler(seen) }).invoke("ask_user", { question: "Q?", options });
      expect(seen[0]!.options).toEqual(expected);
    });
  }

  test("test_injected_arguments_are_not_inputs", () => {
    const [askUser] = new ToolSet({ ask_user: () => "yes" }).resolve(["ask_user"]);
    const types = toolTypes(askUser!);
    const labels = Object.fromEntries(
      [...types.inputs!].map(([name, typ]) => [name, typ === null ? null : canonicalTypeLabel(typ)]),
    );

    expect(labels).toEqual({ question: "str", options: null });
    expect(types.required).toEqual(new Set(["question"]));
  });
});

describe("validate", () => {
  test("checks the arguments without calling the tool", async () => {
    let calls = 0;
    const counted = new FunctionTool({
      name: "echo",
      description: "Echo.",
      parametersJsonSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
      execute: () => ++calls,
    });
    const tools = new ToolSet({ echo: counted });

    expect(await tools.validate("echo", { text: "hi" })).toBeUndefined();
    expect(await rejected(tools.validate("echo", { town: "Paris" }), RangeError)).toBe(
      'tool \'echo\' received unsupported argument(s) ["town"]; allowed: ["text"]',
    );
    expect(await rejected(tools.validate("nope", {}), RangeError)).toContain("unknown tool id 'nope'");
    expect(calls).toBe(0);
  });

  test("refuses arguments that aren't a record", async () => {
    const tools = new ToolSet({ echo: tool("echo") });
    expect(await rejected(tools.validate("echo", [] as never), TypeError)).toBe(
      "tool 'echo' arguments must be a record, got Array",
    );
    expect(await rejected(tools.invoke("echo", "hi" as never), TypeError)).toBe(
      "tool 'echo' arguments must be a record, got string",
    );
  });
});

describe("isConcurrencySafe", () => {
  /** Build a tool whose `isConcurrencySafe` is `check`. */
  function guarded(check: ((args: Record<string, unknown>) => boolean) | null): ToolSet {
    return new ToolSet({
      echo: new FunctionTool({
        name: "echo",
        description: "Echo.",
        parametersJsonSchema: { type: "object", properties: { n: { type: "integer" } } },
        execute: () => null,
        isConcurrencySafe: check,
      }),
    });
  }

  const ROWS: [string, ((args: Record<string, unknown>) => boolean) | null, boolean][] = [
    ["absent", null, false],
    ["true", () => true, true],
    ["false", () => false, false],
    ["a truthy non-true result", () => 1 as unknown as boolean, false],
    [
      "a throw",
      () => {
        throw new Error("boom");
      },
      false,
    ],
  ];
  for (const [label, check, expected] of ROWS) {
    test(`${label} gives ${expected}`, () => {
      expect(guarded(check).isConcurrencySafe("echo", { n: 1 })).toBe(expected);
    });
  }

  test("an unknown tool, non-record arguments or arguments holding undefined give false", () => {
    const tools = guarded(() => true);
    expect(tools.isConcurrencySafe("nope", {})).toBe(false);
    expect(tools.isConcurrencySafe("echo", [] as never)).toBe(false);
    expect(tools.isConcurrencySafe("echo", { n: undefined })).toBe(false);
  });

  test("receives the host form of the arguments", () => {
    const seen: Record<string, unknown>[] = [];
    const args = { n: 2n, big: 2n ** 60n, list: [1] };
    const tools = guarded((received) => {
      seen.push(received);
      return true;
    });

    expect(tools.isConcurrencySafe("echo", args)).toBe(true);
    expect(seen).toEqual([{ n: 2, big: 2n ** 60n, list: [1] }]);
    expect(seen[0]).not.toBe(args);
    expect(seen[0]!.list).not.toBe(args.list);
  });

  test("the ask_user tool is unsafe", () => {
    expect(new ToolSet({ ask_user: () => "yes" }).isConcurrencySafe("ask_user", { question: "Q?" })).toBe(false);
  });
});

describe("test_function_tool_types", () => {
  const place = new FunctionTool({
    name: "place",
    description: "Place an order.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        sym: { type: "string" },
        qty: { type: "integer", minimum: 1 },
        mode: { enum: ["a", "b"], type: "string" },
      },
      required: ["sym", "qty"],
    },
    execute: () => null,
  });
  const BUILDS: [string, () => FunctionTool][] = [
    [
      "registerTool",
      () => {
        registerTool("place", place);
        return new ToolSet().resolve(["place"])[0]!;
      },
    ],
    ["mapping entry", () => new ToolSet({ place }).resolve(["place"])[0]!],
    ["group item", () => new ToolSet({ g: [place] }).resolve(["place"])[0]!],
  ];
  for (const [label, build] of BUILDS) {
    test(label, () => {
      const types = toolTypes(build());
      const labels = Object.fromEntries(
        [...types.inputs!].map(([name, typ]) => [name, typ === null ? null : canonicalTypeLabel(typ)]),
      );
      expect(labels).toEqual({ sym: "str", qty: "int(1..)", mode: "Literal(a, b)" });
      expect(types.required).toEqual(new Set(["sym", "qty"]));
    });
  }
});
