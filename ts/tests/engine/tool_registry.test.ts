/**
 * The tool registry and `ToolSet` construction and lookup, a port of
 * `tests/engine/test_tool_registry.py` (registration and lookup rows) with rows for each
 * construction and group error, `expand`, `forRun`, `toolIds` and `groups`. The invocation rows
 * land with `ToolSet.invoke`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import {
  registeredTools,
  registerTool,
  resolveTools,
  TOOL_REGISTRY,
  ToolSet,
  type ToolEntry,
} from "../../src/tools/registry.ts";

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
    expect(error).toBeInstanceOf(kind);
    expect((error as Error).constructor).toBe(kind);
    return (error as Error).message;
  }
  throw new Error("expected an error");
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
    const [askUser] = new ToolSet({ ask_user: handler }).resolve(["ask_user"]);
    expect(askUser!.name).toBe("ask_user");
    expect(askUser!.description).toBe(
      "Ask the user one short question and wait for the answer. Use only when you need a decision or information that only the user can provide. Put the context they need in one or two sentences. When there are clear choices, pass 2-4 short `options`; the user can always type their own answer.",
    );
    expect(JSON.stringify(askUser!.parametersJsonSchema)).toBe(
      '{"properties":{"question":{"type":"string"},"options":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"default":null,"description":"Optional 2-4 short answer choices. The user can always type a different answer."}},"required":["question"],"type":"object"}',
    );
    const ctx = new ToolContext({
      callId: "c",
      attempt: 1,
      toolName: "ask_user",
      runId: "",
      node: "",
      context: null,
      signal: new AbortController().signal,
    });
    expect(await askUser!.execute({ question: "Approve?" }, ctx)).toBe("answer");
    expect(seen).toEqual([
      [{ question: "Approve?", options: null, questions: null, answerSchema: null, title: null, nodeId: null }, ctx],
    ]);
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
