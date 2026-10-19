/**
 * TOOL retry defaults, the errors a host tool throws to steer retries, their brands, and
 * `ToolContext`, a port of `tests/engine/test_tool_errors.py`. Not ported:
 * `test_tool_errors_are_public` (the package exports land with the package index) and
 * `test_function_tool_awaitable_result_raises_a_tool_check_error` (it needs `FunctionTool`).
 */

import { describe, expect, test } from "bun:test";

import { current, DefaultsError, type Env, read } from "../../src/defaults.ts";
import { ToolContext } from "../../src/tools/context.ts";
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
  TransientToolError,
} from "../../src/tools/errors.ts";

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
});

describe("kinds and guards", () => {
  const guards = {
    permanent: isPermanentToolError,
    transient: isTransientToolError,
    reported: isReportedToolError,
    check: isToolCheckError,
  };
  const cases: [string, unknown, string[]][] = [
    ["PermanentToolError", new PermanentToolError("x"), ["permanent"]],
    ["TransientToolError", new TransientToolError("x"), ["transient"]],
    ["ReportedToolError", new ReportedToolError("x"), ["reported"]],
    ["permanent ReportedToolError", new ReportedToolError("x", { permanent: true }), ["reported", "permanent"]],
    ["ToolCheckError", new ToolCheckError("x"), ["check"]],
  ];
  for (const [label, error, kinds] of cases) {
    test(label, () => {
      expect(toolErrorKinds(error)).toEqual(kinds as never);
      expect(Object.isFrozen(toolErrorKinds(error))).toBe(true);
      for (const [kind, guard] of Object.entries(guards)) expect(guard(error)).toBe(kinds.includes(kind));
    });
  }

  const throwing = Object.defineProperty({}, TOOL_ERROR, {
    get() {
      throw new Error("boom");
    },
  });
  const unbranded: [string, unknown][] = [
    ["a plain Error", new Error("x")],
    ["null", null],
    ["a string", "permanent"],
    ["a string brand", { [TOOL_ERROR]: "permanent" }],
    ["a throwing getter", throwing],
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
