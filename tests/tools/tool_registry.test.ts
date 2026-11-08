/**
 * The tool registry and `ToolSet`: each construction and group error, namespaced groups and their
 * scope, `forRun`, `toolIds`, `groups`, `validate` and the `CheckedCall` it returns, `invoke` of a
 * call (a thrown non-`Error` is wrapped), `isConcurrencySafe` and the `ask_user` options cleaning.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { ToolNode } from "../../src/nodes/tool/node.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { TOOL_ERROR } from "../../src/tools/errors.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import {
  type AskRequest,
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
import { asToolGrant, expandGrants } from "../../src/tools/grants.ts";
import { toolTypes } from "../../src/tools/types.ts";
import { canonicalTypeLabel, TypeCheckError } from "../../src/typesys/values.ts";

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

/** Build a `FunctionTool` named `name` taking one string `topic`; it returns `<name>:<topic>`. */
function topic(name: string): FunctionTool {
  return new FunctionTool({
    name,
    description: `The ${name} tool.`,
    parametersJsonSchema: {
      type: "object",
      properties: { topic: { type: "string" } },
      required: ["topic"],
    },
    execute: (args) => `${name}:${String(args.topic)}`,
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
  test("generic registration rejects reserved ask user id", () => {
    expect(thrown(() => registerTool("ask_user", tool("ask_user")), RangeError)).toBe(
      "tool name 'ask_user' is reserved by yian: pass tools: {...registeredTools(), ask: handler} to runFlow or advance",
    );
    expect(TOOL_REGISTRY.has("ask_user")).toBe(false);
  });

  test("register tool rejects what it cannot register", () => {
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
      "tool name 'ask_user' is reserved by yian: pass tools: {...registeredTools(), ask: handler} to runFlow or advance",
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
      'tools["ask_user"] must be a handler function or {handler, inputs, required?}; yian supplies the base schema',
    ],
    [
      "a string as ask_user",
      { ask_user: "ask" },
      TypeError,
      'tools["ask_user"] must be a handler function or {handler, inputs, required?}; yian supplies the base schema',
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
      'tools["g"][1] must be a FunctionTool, got number',
    ],
    [
      "a function",
      { echo: () => 1 },
      TypeError,
      'tools["echo"] must be a FunctionTool, an MCP server, or an array of FunctionTools, got function',
    ],
    [
      "a plain object",
      { echo: {} },
      TypeError,
      'tools["echo"] must be a FunctionTool, an MCP server, or an array of FunctionTools, got Object',
    ],
    [
      "an ask_user group tool",
      { g: [tool("ask_user")] },
      RangeError,
      "group 'g' has a tool named 'ask_user', a reserved name; pass tools: {...registeredTools(), ask: handler} to runFlow or advance",
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
      `tool 'echo' in group 'g' clashes with tools["echo"]`,
    ],
    [
      "a group tool named like another group",
      { g: [tool("h")], h: [] },
      RangeError,
      `tool 'h' in group 'g' clashes with tools["h"]`,
    ],
    [
      "two tools under one name in a group",
      { g: [tool("echo"), tool("echo")] },
      RangeError,
      "tool 'echo' appears twice in group 'g' as different tools",
    ],
  ];

  for (const [label, tools, kind, message] of SERVERLESS_ROWS) {
    test(`rejects ${label}`, () => {
      expect(thrown(() => loose(tools), kind)).toBe(message);
    });
  }

  test("two tools under one name in two groups are both kept", () => {
    const a = tool("echo");
    const b = tool("echo");
    const tools = new ToolSet({ a: [a], b: [b] });
    const refs = [
      { group: "a", name: "echo" },
      { group: "b", name: "echo" },
    ];
    expect(tools.resolve(refs)).toEqual([a, b]);
  });

  test("null and undefined mean the registry", () => {
    registerTool("echo", tool("echo"));
    expect(new ToolSet(null).toolIds).toEqual(["echo"]);
    expect(new ToolSet().toolIds).toEqual(["echo"]);
  });

  test("the same tool may sit in several groups and twice in one", () => {
    const shared = tool("shared");
    const tools = new ToolSet({ a: [shared, shared, tool("x")], b: [shared] });
    expect(tools.groups).toEqual(["a", "b"]);
    expect(tools.scope().group("a")).toEqual(["shared", "x"]);
    expect(tools.scope().group("b")).toEqual(["shared"]);
    expect(tools.resolve(["shared"])).toEqual([shared]);
  });

  test("an empty group grants nothing", () => {
    const tools = new ToolSet({ company: [] });
    expect(tools.groups).toEqual(["company"]);
    expect(tools.scope().group("company")).toEqual([]);
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
      "Ask the user a question and wait for their answer: when you need a decision or information only they can give, or when they ask you to ask them something. Put the context they need in one or two sentences. When there are clear choices, pass 2-4 short `options`; the user can always type a different answer.",
    );
    expect(JSON.stringify(askUser!.parametersJsonSchema)).toBe(
      '{"properties":{"question":{"type":"string"},"options":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"default":null,"description":"Optional 2-4 short answer choices. The user can always type a different answer."},"timeout":{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"description":"Optional: how long to wait for an answer, such as 30m, 24h or 2d. After it the call returns \'timed out: no answer\'."}},"required":["question"],"type":"object"}',
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
      callId: "c",
      deadline: null,
      extra: {},
    });
    expect(ctx).toBeInstanceOf(ToolContext);
    expect((ctx as ToolContext).callId).toBe("c");
    expect((ctx as ToolContext).toolName).toBe("ask_user");
    expect(TOOL_REGISTRY.has("ask_user")).toBe(false);
  });
});

describe("namespaced tool groups", () => {
  test("two groups may hold different tools of one name, each reached by its group", async () => {
    const a = topic("search"),
      b = topic("search");
    const set = new ToolSet({ github: [a], docs: [b] });
    expect(set.scope().groupsHolding("search")).toEqual(["github", "docs"]);
    expect(set.scope().group("docs")).toEqual(["search"]);
    expect(set.scope().identity("docs", "search")).toBe("docs_search");
    const call = await set.validate({ group: "docs", name: "search" }, { topic: "x" });
    expect(await set.invoke(call)).toBe("search:x");
  });

  test("a bare name finds its one group's tool", async () => {
    const set = new ToolSet({ box: [topic("place_order")] });
    await expect(set.validate("place_order", { topic: "x" })).resolves.toBeDefined();
  });

  test("one tool object in two groups has one record name, its first group's", () => {
    const web = topic("web");
    const set = new ToolSet({ toolbox: [web, topic("a")], company: [web, topic("b")] });
    expect(set.scope().identity("company", "web")).toBe("toolbox_web");
    expect(set.scope().groupsHolding("web")).toEqual(["toolbox"]);
  });

  test.each([
    [
      { github_search: topic("github_search"), github: [topic("search")] },
      "tool 'github_search' and github's 'search' could both show as 'github_search'; rename one",
    ],
    [
      { a: [topic("b_c")], a_b: [topic("c")] },
      "a's 'b_c' and a_b's 'c' could both show as 'a_b_c'; rename one",
    ],
  ])("a tool set in which two tools could show one name is refused (%#)", (spec, message) => {
    expect(() => new ToolSet(spec as never)).toThrow(message);
  });
});

describe("ToolSet lookup", () => {
  test("ask user never falls back to the global registry", () => {
    TOOL_REGISTRY.set("ask_user", tool("ask_user", "global answer"));
    for (const tools of [new ToolSet(), new ToolSet({})]) {
      expect(thrown(() => tools.resolve(["ask_user"]), RangeError)).toBe(
        "no ask handler for this run; pass tools: {...registeredTools(), ask: handler} to runFlow or advance",
      );
    }
  });

  test("tool node uses public invocation path", async () => {
    registerTool(
      "tool_node_public_probe",
      new FunctionTool({
        name: "tool_node_public_probe",
        description: "Return one public question.",
        parametersJsonSchema: { type: "object", properties: { question: { type: "string" } }, required: ["question"] },
        execute: (args) => args.question,
      }),
    );
    const node = new ToolNode("probe", { toolId: "tool_node_public_probe" });

    const error = await node.run({ question: "Approve?", node_id: "model-injected" }).then(
      () => null,
      (caught: unknown) => caught,
    );

    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain("unsupported argument");
  });

  test("a group id is not a tool, and the error names one of its tools as {group: tool}", () => {
    expect(thrown(() => new ToolSet({ toolbox: [tool("a")] }).resolve(["toolbox"]), RangeError)).toBe(
      "'toolbox' is a tool group, not a tool; name one of its tools, as `{toolbox: a}`",
    );
    expect(thrown(() => new ToolSet({ toolbox: [] }).resolve(["toolbox"]), RangeError)).toBe(
      "'toolbox' is a tool group, not a tool, and it has no tools",
    );
  });

  test("an unknown id lists the mapping's tools, a group's as {group: tool}", () => {
    const tools = new ToolSet({ b: tool("b"), g: [tool("a")] });
    expect(thrown(() => tools.resolve(["zzz"]), RangeError)).toBe(
      'unknown tool id \'zzz\'; known tools: ["b","{g: a}"]',
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

  test("overlapping and repeated grants bind each tool once, under its first grant's name", () => {
    const shared = tool("shared");
    const tools = new ToolSet({ a: [tool("x"), shared], b: [shared, tool("y")], solo: tool("solo") });
    const grants = ["a", "b", "a", "solo", "solo"].map(asToolGrant);
    expect(expandGrants(grants, tools.scopes()).map((granted) => granted.visible)).toEqual([
      "a_x",
      "a_shared",
      "b_y",
      "solo",
    ]);
  });

  test("toolIds is sorted, from the mapping and its groups", () => {
    const tools = new ToolSet({ b: tool("b"), g: [tool("a"), tool("C")], ask_user: () => null });
    expect(tools.toolIds).toEqual(["ask", "ask_user", "b", "{g: C}", "{g: a}"]);
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

  test("for run keeps the subclass", () => {
    class Sub extends ToolSet {}
    const bound = new Sub({ place: tool("place") }).forRun("r");
    expect(bound).toBeInstanceOf(Sub);
    expect(Object.getPrototypeOf(bound)).toBe(Sub.prototype);
    expect(bound.runId).toBe("r");
  });

  test("for run keeps the subclass's own fields", () => {
    class Audited extends ToolSet {
      readonly log: string[] = [];
      note(entry: string): void {
        this.log.push(entry);
      }
    }
    const set = new Audited({ place: tool("place") });
    const bound = set.forRun("r");
    bound.note("x");
    expect(bound.log).toEqual(["x"]);
    expect(bound.log).toBe(set.log);
    expect(bound.runId).toBe("r");
    expect(set.runId).toBe("");
  });
});

describe("ask_user invocation", () => {
  // An unbound set gives each call a fresh random id.
  const CALL_FIELDS = { callId: expect.any(String), deadline: null, extra: {} };
  const NO_FIELDS = { options: null, questions: null, answerSchema: null, title: null, nodeId: null, ...CALL_FIELDS };

  test("ask user public schema contains question and options", async () => {
    const seen: AskUserRequest[] = [];
    const tools = new ToolSet({ ask_user: recordingHandler(seen) });

    const [askUser] = tools.resolve(["ask_user"]);
    expect(Object.keys(askUser!.parametersJsonSchema.properties as object)).toEqual(["question", "options", "timeout"]);
    expect(await tools.invoke("ask_user", { question: "Approve?" })).toBe("answer");

    expect(seen).toEqual([{ question: "Approve?", ...NO_FIELDS }]);
    expect(TOOL_REGISTRY.has("ask_user")).toBe(false);
  });

  test("ask user openai schema offers optional string options", () => {
    const [askUser] = new ToolSet({ ask_user: recordingHandler([]) }).resolve(["ask_user"]);
    const schema = askUser!.parametersJsonSchema as any;

    expect(schema.required).toEqual(["question"]);
    expect(schema.properties.options.anyOf).toEqual([{ type: "array", items: { type: "string" } }, { type: "null" }]);
    expect(schema.properties.options.description).toContain("user can always type a different answer");
  });

  const INJECTED: [string, unknown][] = [
    ["deadline", "2026-10-24T10:30:00.000Z"],
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
          'allowed: ["options","question","timeout"]',
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

  // TS-only: the HUMAN_INPUT node that sent `questions`, `answerSchema`, `title` and `nodeId` on
  // the trusted path is gone, so that path takes only the public `question` and `options`.
  test("the trusted ask_user path passes the question and options", async () => {
    const seen: AskUserRequest[] = [];
    const tools = new ToolSet({ ask_user: recordingHandler(seen, "B") });

    expect(await tools.invokeTrusted("ask_user", { question: "Which?", options: ["A", "B"] })).toBe("B");

    expect(seen).toEqual([{ ...NO_FIELDS, question: "Which?", options: ["A", "B"] }]);
  });

  test("the trusted ask_user path refuses the old HUMAN_INPUT metadata", async () => {
    const tools = new ToolSet({ ask_user: recordingHandler([]) });

    const inputs = { question: "Approve?", questions: [], answerSchema: [], title: "T", nodeId: "n" };
    expect(await rejected(tools.invokeTrusted("ask_user", inputs), RangeError)).toBe(
      'tool \'ask_user\' received unsupported argument(s) ["answerSchema","nodeId","questions","title"] on its ' +
        'trusted invocation path; allowed: ["deadline","options","question","timeout"]',
    );
  });

  // TS-only: the agent turns the model's `timeout` into `deadline` and checks the call on the
  // trusted path; the handler gets the deadline, never the timeout text.
  test("the trusted ask_user path hands the handler the deadline the agent computed", async () => {
    const seen: AskRequest[] = [];
    const tools = new ToolSet({ ask: (request: AskRequest) => (seen.push(request), "answer") });

    const inputs = { question: "Approve?", timeout: "30m", deadline: "2026-10-24T10:30:00.000Z" };
    const checked = await tools.validateTrusted("ask_user", inputs);
    expect(await tools.invoke(checked, { node: "a" })).toBe("answer");

    expect(seen).toEqual([{ ...NO_FIELDS, question: "Approve?", deadline: "2026-10-24T10:30:00.000Z" }]);
  });

  test("the ask and ask_user tools aren't resumable", () => {
    const fromAsk = new ToolSet({ ask: () => "answer" });
    expect(fromAsk.resolve(["ask", "ask_user"]).map((each) => each.resumable)).toEqual([false, false]);
    const fromAskUser = new ToolSet({ ask_user: () => "answer" });
    expect(fromAskUser.resolve(["ask_user"])[0]!.resumable).toBe(false);
  });

  test("test_ask_user_never_falls_back_to_the_global_registry (invoke)", async () => {
    TOOL_REGISTRY.set("ask_user", tool("ask_user", "global answer"));
    for (const tools of [new ToolSet(), new ToolSet({})]) {
      expect(await rejected(tools.invoke("ask_user", { question: "Approve?" }), RangeError)).toBe(
        "no ask handler for this run; pass tools: {...registeredTools(), ask: handler} to runFlow or advance",
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

  test("injected arguments are not inputs", () => {
    const [askUser] = new ToolSet({ ask_user: () => "yes" }).resolve(["ask_user"]);
    const types = toolTypes(askUser!);
    const labels = Object.fromEntries(
      [...types.inputs!].map(([name, typ]) => [name, typ === null ? null : canonicalTypeLabel(typ)]),
    );

    expect(labels).toEqual({ question: "str", options: null, timeout: "Optional(str)" });
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

    const call = await tools.validate("echo", { text: "hi" });
    expect(call.toolId).toBe("echo");
    expect(Object.isFrozen(call)).toBe(true);
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

describe("invoke of a CheckedCall", () => {
  /** Build a tool set whose `echo` tool records the arguments and context of each call. */
  function recording(seen: [Record<string, unknown>, ToolContext][]): ToolSet {
    return new ToolSet({
      echo: new FunctionTool({
        name: "echo",
        description: "Echo.",
        parametersJsonSchema: { type: "object", properties: { list: { type: "array" } } },
        execute: (args, ctx) => {
          seen.push([args, ctx]);
          return "done";
        },
      }),
    });
  }

  test("each invoke gives execute a fresh copy of the checked arguments", async () => {
    const seen: [Record<string, unknown>, ToolContext][] = [];
    const inputs = { list: [1] };
    const tools = recording(seen);
    const call = await tools.validate("echo", inputs);

    expect(await tools.invoke(call)).toBe("done");
    expect(await tools.invoke(call, { attempt: 2 })).toBe("done");

    expect(seen.map(([args]) => args)).toEqual([{ list: [1] }, { list: [1] }]);
    expect(seen[0]![0]).not.toBe(seen[1]![0]);
    expect(seen[0]![0].list).not.toBe(inputs.list);
    expect(seen.map(([, ctx]) => ctx.attempt)).toEqual([1, 2]);
  });

  test("a call validated on one set runs with the context of the set that invokes it", async () => {
    const seen: [Record<string, unknown>, ToolContext][] = [];
    const tools = recording(seen);
    const call = await tools.validate("echo", {});

    await tools.forRun("r").invoke(call, { node: "n" });

    expect(seen[0]![1].runId).toBe("r");
    expect(seen[0]![1].callId).toBe("r/n");
  });

  test("a thrown non-Error rejects as an Error with the value as its cause; an Error passes as it is", async () => {
    const thrownBy = (value: unknown): ToolSet =>
      new ToolSet({
        fail: new FunctionTool({
          name: "fail",
          description: "Fail.",
          parametersJsonSchema: { type: "object", properties: {} },
          execute: () => {
            throw value;
          },
        }),
      });
    const object = { code: 7 };
    const rows: [unknown, string][] = [
      ["boom", "boom"],
      [object, "[object Object]"],
      [undefined, "undefined"],
    ];
    for (const [value, message] of rows) {
      const error = await thrownBy(value).invoke("fail", {}).then(
        () => undefined,
        (reason: unknown) => reason,
      );
      expect((error as Error).constructor).toBe(Error);
      expect((error as Error).message).toBe(message);
      expect((error as Error).cause).toBe(value);
    }
    const own = new RangeError("own");
    expect(await thrownBy(own).invoke("fail", {}).catch((reason: unknown) => reason)).toBe(own);
    // A branded tool error from another realm fails `instanceof Error` but keeps its brand.
    const branded = { [TOOL_ERROR]: ["permanent"], message: "foreign" };
    expect(await thrownBy(branded).invoke("fail", {}).catch((reason: unknown) => reason)).toBe(branded);
  });

  test("anything but a tool id or a CheckedCall is a TypeError", async () => {
    const tools = new ToolSet({ echo: tool("echo") });
    expect(await rejected(tools.invoke({} as never), TypeError)).toBe(
      "invoke takes a tool id or a CheckedCall from validate",
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
        parametersJsonSchema: {
          type: "object",
          properties: { n: { type: "integer" }, big: { type: "integer" }, list: { type: "array" } },
        },
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
    test(`${label} gives ${expected}`, async () => {
      const tools = guarded(check);
      expect(tools.isConcurrencySafe(await tools.validate("echo", { n: 1 }))).toBe(expected);
    });
  }

  test("validate refuses an unknown tool, non-record arguments or undefined; a non-call gives false", async () => {
    const tools = guarded(() => true);
    expect(await rejected(tools.validate("nope", {}), RangeError)).toContain("unknown tool id 'nope'");
    await rejected(tools.validate("echo", [] as never), TypeError);
    await expect(tools.validate("echo", { n: undefined })).rejects.toBeInstanceOf(TypeCheckError);
    expect(tools.isConcurrencySafe({} as never)).toBe(false);
    expect(tools.isConcurrencySafe("echo" as never)).toBe(false);
  });

  test("receives the host form of the arguments", async () => {
    const seen: Record<string, unknown>[] = [];
    const args = { n: 2n, big: 2n ** 60n, list: [1] };
    const tools = guarded((received) => {
      seen.push(received);
      return true;
    });

    expect(tools.isConcurrencySafe(await tools.validate("echo", args))).toBe(true);
    expect(seen).toEqual([{ n: 2, big: 2n ** 60n, list: [1] }]);
    expect(seen[0]).not.toBe(args);
    expect(seen[0]!.list).not.toBe(args.list);
  });

  test("an async check that rejects gives false and leaves no unhandled rejection", async () => {
    const check = (async () => {
      throw new Error("x");
    }) as unknown as (args: Record<string, unknown>) => boolean;
    const tools = guarded(check);
    expect(tools.isConcurrencySafe(await tools.validate("echo", { n: 1 }))).toBe(false);
    // Let the rejection settle, so an unhandled one would fail the run here.
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  test("the ask_user tool is unsafe", async () => {
    const tools = new ToolSet({ ask_user: () => "yes" });
    expect(tools.isConcurrencySafe(await tools.validate("ask_user", { question: "Q?" }))).toBe(false);
  });
});

describe("function tool types", () => {
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

describe("withLocalGroups", () => {
  const tool = (name: string) =>
    new FunctionTool({ name, description: name, parametersJsonSchema: { type: "object" }, execute: () => `local ${name}` });

  test("a local group's tools resolve by group, before a run group of that name, and its scope comes first", async () => {
    const runRead = tool("read");
    const run = new ToolSet({ lessons: [runRead], lookup: tool("lookup") }).forRun("r");
    const read = tool("read");
    const local = run.withLocalGroups(new Map([["lessons", [read, tool("append")]]]), "this node's memories");
    expect(local.resolve([{ group: "lessons", name: "read" }])).toEqual([read]);
    expect(run.resolve([{ group: "lessons", name: "read" }])).toEqual([runRead]);
    expect(local.runId).toBe("r");
    const [scope, runScope] = local.scopes();
    expect(scope!.label).toBe("this node's memories");
    expect(scope!.group("lessons")).toEqual(["read", "append"]);
    expect(scope!.bareMembers).toBe(false);
    expect(runScope!.label).toBe("this run");
    const granted = expandGrants([asToolGrant("lookup"), { kind: "group", group: "lessons", tools: ["append"] }], local.scopes());
    expect(granted.map((each) => [each.visible, each.scope])).toEqual([
      ["lookup", 1],
      ["lessons_append", 0],
    ]);
    expect(await local.invoke({ group: "lessons", name: "append" }, {})).toBe("local append");
    expect(() => local.resolve([{ group: "lessons", name: "edit" }])).toThrow("`lessons` has no tool `edit`; it has read, append");
  });

  test("a bare name never reaches a local group's tool", () => {
    const local = new ToolSet({}).withLocalGroups(new Map([["lessons", [tool("read")]]]), "this node's memories");
    expect(() => expandGrants([asToolGrant("read")], local.scopes())).toThrow("grants unknown tool or group 'read'");
  });
});
