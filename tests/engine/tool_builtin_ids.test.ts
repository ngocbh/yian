import { describe, expect, test } from "bun:test";
import {
  ASK,
  BUILTIN_TOOL_IDS,
  isPending,
  pending,
  PENDING_BRAND,
  SLEEP,
  TAKE,
  toolCallId,
} from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { type AskRequest, registerTool, ToolSet } from "../../src/tools/registry.ts";

describe("built-in ids and the pending marker", () => {
  test("ids", () => {
    expect([TAKE, SLEEP, ASK]).toEqual(["take", "sleep", "ask"]);
    expect([...BUILTIN_TOOL_IDS]).toEqual(["take", "sleep", "ask"]);
  });
  test("a TOOL call id is the run id and the expanded node path", () => {
    expect(toolCallId("run-7f3a", "loop#3/place")).toBe("run-7f3a/loop#3/place");
  });
  test("pending() is a frozen branded marker; isPending reads the brand only", () => {
    const marker = pending();
    expect(isPending(marker)).toBe(true);
    expect(Object.isFrozen(marker)).toBe(true);
    // A marker from another copy of yian carries the same registry symbol.
    expect(isPending({ [Symbol.for("yian.pending")]: true })).toBe(true);
    for (const other of [null, undefined, "pending", {}, { pending: true }, () => {}]) {
      expect(isPending(other)).toBe(false);
    }
    const hostile = new Proxy({}, {
      get() {
        throw new Error("trap");
      },
    });
    expect(isPending(hostile)).toBe(false);
    expect(PENDING_BRAND as symbol).toBe(Symbol.for("yian.pending"));
  });
});

const echo = (name: string) =>
  new FunctionTool({ name, description: "", parametersJsonSchema: { type: "object" }, execute: () => "x" });

describe("reserved names", () => {
  test.each(["take", "sleep", "ask", "ask_user", "load_skill"])("registerTool(%p) is refused", (name) => {
    expect(() => registerTool(name, echo(name))).toThrow(RangeError);
  });
  test("take is reserved by yian", () => {
    expect(() => registerTool("take", echo("take"))).toThrow(
      new RangeError(
        "tool name 'take' is reserved by yian: an engine built-in a TOOL node reaches with tool_id: take",
      ),
    );
  });
  test.each(["take", "sleep", "load_skill"])("a top-level %p key is refused", (name) => {
    expect(() => new ToolSet({ [name]: echo(name) })).toThrow(RangeError);
  });
  test("a top-level take key says why", () => {
    expect(() => new ToolSet({ take: echo("take") })).toThrow(
      new RangeError(
        'tools["take"]: the name is reserved by yian: an engine built-in a TOOL node reaches with tool_id: take',
      ),
    );
  });
  test.each(["take", "sleep", "ask", "ask_user", "load_skill"])("a group member named %p is refused", (name) => {
    expect(() => new ToolSet({ g: [echo(name)] })).toThrow(RangeError);
  });
});

describe("the ask slot", () => {
  test("tools.ask serves both the agent's ask_user and the TOOL built-in ask", async () => {
    const seen: AskRequest[] = [];
    const tools = new ToolSet({
      ask: (request) => {
        seen.push(request);
        return "yes";
      },
    }).forRun("r1");
    expect(tools.toolIds).toEqual(["ask", "ask_user"]);
    expect(await tools.invoke("ask_user", { question: "Ship?" }, { node: "agent" })).toBe("yes");
    expect(
      await tools.invokeTrusted(
        "ask",
        { prompt: "Ship?", deadline: "2026-10-24T00:00:00.000Z", nodeId: "gate" },
        { node: "gate" },
      ),
    ).toBe("yes");
    expect(seen).toEqual([
      {
        question: "Ship?",
        options: null,
        questions: null,
        answerSchema: null,
        title: null,
        nodeId: null,
        callId: "r1/agent",
        deadline: null,
        extra: {},
      },
      {
        question: "Ship?",
        options: null,
        questions: null,
        answerSchema: null,
        title: null,
        nodeId: "gate",
        callId: "r1/gate",
        deadline: "2026-10-24T00:00:00.000Z",
        extra: {},
      },
    ]);
  });
  test("tools.ask_user is the old name of the same slot; both is a TypeError", () => {
    expect(new ToolSet({ ask_user: () => "a" }).toolIds).toEqual(["ask", "ask_user"]);
    expect(() => new ToolSet({ ask: () => "a", ask_user: () => "b" })).toThrow(
      new TypeError('pass the ask handler once, as tools["ask"] (tools["ask_user"] is its old name)'),
    );
  });
  test("the slot takes a handler or {handler, inputs, required?}", () => {
    for (const entry of [echo("ask"), { handler: () => null }, { inputs: {} }, "x"]) {
      expect(() => new ToolSet({ ask: entry as never })).toThrow(
        new TypeError('tools["ask"] must be a handler function or {handler, inputs, required?}; yian supplies the base schema'),
      );
    }
  });
  test("an AskTool declares host extras, checked and passed in extra", async () => {
    const seen: AskRequest[] = [];
    const tools = new ToolSet({
      ask: {
        handler: (request) => {
          seen.push(request);
          return null;
        },
        inputs: { to: { type: "string" } },
        required: ["to"],
      },
    });
    await tools.invokeTrusted("ask", { prompt: "?", to: "lead" });
    expect(seen[0]!.extra).toEqual({ to: "lead" });
    await expect(tools.invokeTrusted("ask", { prompt: "?" })).rejects.toThrow('missing required argument(s) ["to"]');
    await expect(tools.invokeTrusted("ask", { prompt: "?", to: 5 })).rejects.toThrow("argument 'to'");
  });
  test("an extra may not shadow the base surface", () => {
    expect(() => new ToolSet({ ask: { handler: () => null, inputs: { prompt: { type: "string" } } } })).toThrow(
      new RangeError('tools["ask"] input \'prompt\' is part of yian\'s ask surface; name the extra input differently'),
    );
  });
  test("a required extra must be declared", () => {
    expect(() => new ToolSet({ ask: { handler: () => null, inputs: {}, required: ["to"] } })).toThrow(
      new RangeError('tools["ask"] requires \'to\', which isn\'t one of its inputs'),
    );
  });
  test("no handler: ask and ask_user both say how to pass one", async () => {
    const message = "no ask handler for this run; pass tools: {...registeredTools(), ask: handler} to runFlow or advance";
    await expect(new ToolSet({}).invoke("ask_user", { question: "?" })).rejects.toThrow(message);
    await expect(new ToolSet({}).invokeTrusted("ask", { prompt: "?" })).rejects.toThrow(message);
  });
  test.each(["take", "sleep"])("%p is not a tool a set resolves", (name) => {
    expect(() => new ToolSet({}).resolve([name])).toThrow(
      new RangeError(`'${name}' is an engine built-in, not a tool; reach it with a TOOL node's tool_id: ${name}`),
    );
  });
});

describe("pending() passes through", () => {
  test("a handler's marker comes back unchanged, with no output check", async () => {
    const typed = new FunctionTool({
      name: "codex_send",
      description: "",
      parametersJsonSchema: { type: "object" },
      outputJsonSchema: { type: "string" },
      execute: () => pending(),
    });
    expect(await new ToolSet({ codex_send: typed }).invoke("codex_send", {})).toBe(pending());
    expect(await new ToolSet({ ask: () => pending() }).invokeTrusted("ask", { prompt: "?" })).toBe(pending());
  });
});
