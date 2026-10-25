/**
 * TOOL retry defaults, the errors a host tool throws to steer retries, their brands, and
 * `ToolContext`, a port of `tests/engine/test_tool_errors.py`, and the exports of `tools/index.ts`
 * (`test_tool_errors_are_public`, and `test_tool_names_are_exported_from_yian_and_yian_tools` and
 * `test_yian_tools_exports_the_mcp_api` for `tools/index.ts`; the package root doesn't exist yet).
 * Not ported: `test_function_tool_awaitable_result_raises_a_tool_check_error` (a `FunctionTool`'s
 * execute may return a Promise, so an async result is not an error in TS).
 */

import { describe, expect, test } from "bun:test";

import { current, DefaultsError, type Env, read } from "../../src/defaults.ts";
import { ASK, isPending, pending, SLEEP, TAKE, ToolContext } from "../../src/tools/context.ts";
import {
  isPermanentToolError,
  isReportedToolError,
  isToolCheckError,
  isTransientToolError,
  PermanentToolError,
  ReportedToolError,
  TOOL_ERROR,
  ToolCheckError,
  toolErrorKinds,
  type ToolErrorKind,
  TransientToolError,
} from "../../src/tools/errors.ts";
import { FunctionTool, tool } from "../../src/tools/function_tools.ts";
import * as tools from "../../src/tools/index.ts";
import type {
  AskHandler,
  AskRequest,
  AskTool,
  AskUserHandler,
  AskUserRequest,
  CheckedCall,
  InvokeOptions,
  McpServerOptions,
  StandardIssue,
  StandardResult,
  StandardSchemaV1,
  PendingMarker,
  ToolEntry,
  ToolErrorKind as IndexToolErrorKind,
} from "../../src/tools/index.ts";
import * as mcp from "../../src/tools/mcp.ts";
import * as registry from "../../src/tools/registry.ts";

/** Return an environment holding only `vars`. */
function envOf(vars: Readonly<Record<string, string>>): Env {
  return (name) => (Object.hasOwn(vars, name) ? vars[name] : undefined);
}

describe("TOOL retry defaults", () => {
  test("have their built-in values", () => {
    const values = current();
    expect(values.toolRetries).toBe(0);
    expect(values.toolRetryDelaySeconds).toBe(1);
    expect(values.toolRetryMaxDelaySeconds).toBe(30);
  });

  test("retries can be set", () => {
    expect(read(envOf({ YIAN_TOOL_RETRIES: "2" })).toolRetries).toBe(2);
  });

  for (const name of ["TOOL_RETRIES", "TOOL_RETRY_DELAY_SECONDS", "TOOL_RETRY_MAX_DELAY_SECONDS"]) {
    test(`YIAN_${name} refuses a negative value`, () => {
      const run = () => read(envOf({ [`YIAN_${name}`]: "-1" }));
      expect(run).toThrow(DefaultsError);
      expect(run).toThrow(`YIAN_${name}="-1" is out of range`);
    });
  }
});

describe("TransientToolError", () => {
  test("carries retryAfter", () => {
    expect(new TransientToolError("x", { retryAfter: 2.5 }).retryAfter).toBe(2.5);
    expect(new TransientToolError("x", { retryAfter: 0 }).retryAfter).toBe(0);
    expect(new TransientToolError("x").retryAfter).toBeNull();
    expect(new TransientToolError("rate limited", { retryAfter: 1 }).message).toBe("rate limited");
  });

  for (const value of [-1, -0.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    test(`refuses retryAfter ${value}`, () => {
      const run = () => new TransientToolError("x", { retryAfter: value });
      expect(run).toThrow(RangeError);
      expect(run).toThrow("retryAfter");
    });
  }

  for (const value of [true, "1"]) {
    test(`refuses retryAfter ${JSON.stringify(value)}`, () => {
      const run = () => new TransientToolError("x", { retryAfter: value as unknown as number });
      expect(run).toThrow(TypeError);
      expect(run).toThrow("retryAfter");
    });
  }
});

test("the tool errors are independent classes", () => {
  const classes = [PermanentToolError, TransientToolError, ReportedToolError, ToolCheckError];
  for (const each of classes) {
    expect(each.prototype).toBeInstanceOf(Error);
    for (const other of classes) {
      if (other !== each) expect(each.prototype instanceof other).toBe(false);
    }
  }
  expect(new ToolCheckError("x")).toBeInstanceOf(TypeError);
  expect(new PermanentToolError("x").name).toBe("PermanentToolError");
  expect(new ToolCheckError("x").name).toBe("ToolCheckError");
});

test("ReportedToolError refuses a non-boolean permanent", () => {
  expect(() => new ReportedToolError("x", { permanent: 1 as unknown as boolean })).toThrow(TypeError);
  expect(() => new ReportedToolError("x", { permanent: 1 as unknown as boolean })).toThrow("permanent");
});

describe("kinds and guards", () => {
  const guards = {
    permanent: isPermanentToolError,
    transient: isTransientToolError,
    reported: isReportedToolError,
    check: isToolCheckError,
  };
  const cases: [string, unknown, ToolErrorKind[]][] = [
    ["PermanentToolError", new PermanentToolError("x"), ["permanent"]],
    ["TransientToolError", new TransientToolError("x"), ["transient"]],
    ["ReportedToolError", new ReportedToolError("x"), ["reported"]],
    ["permanent ReportedToolError", new ReportedToolError("x", { permanent: true }), ["reported", "permanent"]],
    ["ToolCheckError", new ToolCheckError("x"), ["check"]],
    ["an array brand with an unknown kind", { [TOOL_ERROR]: ["fatal", "permanent"] }, ["permanent"]],
  ];
  for (const [label, error, kinds] of cases) {
    test(label, () => {
      expect(toolErrorKinds(error)).toEqual(kinds);
      expect(Object.isFrozen(toolErrorKinds(error))).toBe(true);
      for (const [kind, guard] of Object.entries(guards)) {
        expect(guard(error)).toBe(kinds.some((each) => each === kind));
      }
    });
  }

  const throwing = Object.defineProperty({}, TOOL_ERROR, {
    get() {
      throw new Error("boom");
    },
  });
  const trappedArray = new Proxy([], {
    get() {
      throw new Error("trap");
    },
  });
  const revoked = Proxy.revocable({}, {});
  revoked.revoke();
  // A species constructor makes `filter` return a proxy whose reads throw outside the `try`.
  const speciesArray = Object.assign([], {
    constructor: {
      [Symbol.species]: function () {
        return new Proxy([], {
          get() {
            throw new Error("species");
          },
        });
      },
    },
  });
  const iteratorArray = Object.assign([], {
    [Symbol.iterator]() {
      throw new Error("iterator");
    },
  });
  const unbranded: [string, unknown][] = [
    ["a plain Error", new Error("x")],
    ["null", null],
    ["a string", "permanent"],
    ["a string brand", { [TOOL_ERROR]: "permanent" }],
    ["a throwing getter", throwing],
    ["a brand array whose proxy traps throw", { [TOOL_ERROR]: trappedArray }],
    ["a revoked proxy", revoked.proxy],
    ["a brand array with an own filter", { [TOOL_ERROR]: Object.assign([], { filter: () => 1 }) }],
    ["a brand array with a hostile species constructor", { [TOOL_ERROR]: speciesArray }],
    ["a brand array with a throwing iterator", { [TOOL_ERROR]: iteratorArray }],
  ];
  for (const [label, value] of unbranded) {
    test(`${label} has no kinds`, () => {
      expect(toolErrorKinds(value)).toEqual([]);
      for (const guard of Object.values(guards)) expect(guard(value)).toBe(false);
    });
  }
});

test("a second module copy's errors are classified by the brand", async () => {
  // A computed specifier, so `tsc` doesn't resolve it. Bun shares one instance for a `file:` URL
  // whatever its query, but loads a second instance for a relative specifier with a query.
  const specifier = "../../src/tools/errors.ts?copy=2";
  const copy = (await import(specifier)) as typeof import("../../src/tools/errors.ts");
  expect(copy.PermanentToolError).not.toBe(PermanentToolError);
  expect(new copy.PermanentToolError("x")).not.toBeInstanceOf(PermanentToolError);
  expect(isPermanentToolError(new copy.PermanentToolError("x"))).toBe(true);
  expect(isTransientToolError(new copy.TransientToolError("x"))).toBe(true);
  expect(isReportedToolError(new copy.ReportedToolError("x"))).toBe(true);
  expect(isToolCheckError(new copy.ToolCheckError("x"))).toBe(true);
  expect(isPermanentToolError(new copy.ReportedToolError("x", { permanent: true }))).toBe(true);
});

test("ToolContext is frozen and keeps its fields", () => {
  const signal = new AbortController().signal;
  const app = { broker: "paper" };
  const ctx = new ToolContext({
    callId: "run-7f3a/place",
    attempt: 1,
    toolName: "place",
    runId: "run-7f3a",
    node: "place",
    context: app,
    signal,
  });
  expect(Object.isFrozen(ctx)).toBe(true);
  expect(ctx.callId).toBe("run-7f3a/place");
  expect(ctx.attempt).toBe(1);
  expect(ctx.toolName).toBe("place");
  expect(ctx.runId).toBe("run-7f3a");
  expect(ctx.node).toBe("place");
  expect(ctx.context).toBe(app);
  expect(ctx.signal).toBe(signal);
});

test("tools/index.ts exports the tools API, the same objects as its modules'", () => {
  const expected: Record<string, unknown> = {
    ToolContext,
    TAKE,
    SLEEP,
    ASK,
    pending,
    isPending,
    PermanentToolError,
    ReportedToolError,
    TransientToolError,
    ToolCheckError,
    TOOL_ERROR,
    toolErrorKinds,
    isPermanentToolError,
    isTransientToolError,
    isReportedToolError,
    isToolCheckError,
    FunctionTool,
    tool,
    TOOL_REGISTRY: registry.TOOL_REGISTRY,
    ToolSet: registry.ToolSet,
    registerTool: registry.registerTool,
    registeredTools: registry.registeredTools,
    resolveTools: registry.resolveTools,
    invokeTool: registry.invokeTool,
    invokeToolTrusted: registry.invokeToolTrusted,
    mcpServer: mcp.mcpServer,
    McpServer: mcp.McpServer,
    McpError: mcp.McpError,
    McpConnectError: mcp.McpConnectError,
    McpToolError: mcp.McpToolError,
    McpOutputError: mcp.McpOutputError,
    McpRequestError: mcp.McpRequestError,
    McpInvalidRequestError: mcp.McpInvalidRequestError,
    McpTimeoutError: mcp.McpTimeoutError,
    McpTransportError: mcp.McpTransportError,
    McpServerClosed: mcp.McpServerClosed,
    McpCallCancelled: mcp.McpCallCancelled,
    isMcpError: mcp.isMcpError,
  };
  // Exactly these names: no translator, no `ask_user` id (the built-in ids are), no `isFunctionTool` or
  // `isMcpServer`.
  expect(Object.keys(tools).sort()).toEqual(Object.keys(expected).sort());
  const exported = tools as Record<string, unknown>;
  expect(Object.keys(expected).filter((name) => exported[name] !== expected[name])).toEqual([]);
  // The type exports, checked by `tsc`.
  const kind: IndexToolErrorKind = "permanent" satisfies ToolErrorKind;
  const handler: AskUserHandler = (request: AskUserRequest) => request.question;
  const entry: ToolEntry = handler;
  const ask: AskHandler = (request: AskRequest) => request.callId;
  const askTool: AskTool = { handler: ask, inputs: {} };
  const marker: PendingMarker = pending();
  const entries: ToolEntry[] = [ask, askTool];
  const options: McpServerOptions = { command: "echo" };
  const invoke: InvokeOptions = { callId: "c1" };
  const toolIdOf = (call: CheckedCall): string => call.toolId;
  const issue: StandardIssue = { message: "bad", path: ["a", { key: 0 }] };
  const result: StandardResult<number> = { issues: [issue] };
  const validator: StandardSchemaV1<number> = {
    "~standard": { version: 1, vendor: "hand", validate: (value) => ({ value: Number(value) }) },
  };
  expect([kind, typeof entry, options.command, invoke.callId, typeof toolIdOf]).toEqual([
    "permanent",
    "function",
    "echo",
    "c1",
    "function",
  ]);
  expect([result.issues?.length, validator["~standard"].vendor]).toEqual([1, "hand"]);
  expect([entries.length, isPending(marker)]).toEqual([2, true]);
});
