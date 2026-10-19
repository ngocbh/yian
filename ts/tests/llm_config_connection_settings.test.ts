/**
 * Mirrors `tests/test_llm_config_connection_settings.py`. This file ports the module-function
 * tests; the `LLMConfig` tests come with `config.ts`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../src/defaults.ts";

import {
  OPENAI_REFUSED_HEADER_NAMES,
  openaiIdHeaders,
  REFUSED_HEADER_NAMES,
  validateHeaderName,
  validateHeaders,
  validateHeaderValue,
  validateOrganization,
} from "../src/llm_clients/connection_settings.ts";

afterEach(() => install(BUILTIN));

const GATEWAY = "gw-secret-91c4e2";
const CREDENTIAL_NAMES = ["Authorization", "Proxy-Authorization", "x-api-key", "api-key", "x-goog-api-key"];
const TRANSPORT_NAMES = ["Host", "Content-Length", "Transfer-Encoding", "Connection", "Content-Type"];

/** Return the error `run` throws, failing the test if it throws none. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

test("refused name tables are lowercase", () => {
  expect(new Set(Object.keys(REFUSED_HEADER_NAMES))).toEqual(
    new Set([...CREDENTIAL_NAMES, ...TRANSPORT_NAMES].map((name) => name.toLowerCase())),
  );
  expect(new Set(Object.keys(OPENAI_REFUSED_HEADER_NAMES))).toEqual(
    new Set(["openai-organization", "openai-project"]),
  );
});

test("module functions return valid input and throw RangeError", () => {
  expect(validateHeaderName("X-Gateway", "openai")).toBe("X-Gateway");
  expect(validateHeaderName("OpenAI-Project", "deepseek")).toBe("OpenAI-Project");
  expect(validateHeaderValue("X-Gateway", GATEWAY)).toBe(GATEWAY);
  expect(validateHeaders({ "X-A": "1", "X-B": "2" }, "openai")).toEqual({ "X-A": "1", "X-B": "2" });
  expect(validateOrganization("org-1", "project")).toBe("org-1");
  for (const call of [
    () => validateHeaderName("OpenAI-Project", "openai"),
    () => validateHeaderName(3, null),
    () => validateHeaderValue("X-Gateway", 3),
    () => validateHeaders(["X-A"], null),
    () => validateOrganization(null, "organization"),
  ]) {
    expect(call).toThrow(RangeError);
  }
});

test("openaiIdHeaders maps the ids to their headers", () => {
  expect(openaiIdHeaders("org-abc123", "proj_XYZ")).toEqual({
    "OpenAI-Organization": "org-abc123",
    "OpenAI-Project": "proj_XYZ",
  });
  expect(Object.keys(openaiIdHeaders("org-abc123", "proj_XYZ"))).toEqual(["OpenAI-Organization", "OpenAI-Project"]);
  expect(openaiIdHeaders(null, "proj_XYZ")).toEqual({ "OpenAI-Project": "proj_XYZ" });
  expect(openaiIdHeaders(null, null)).toEqual({});
  const error = thrown(() => openaiIdHeaders("org-abc123", "proj secret"));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("project must be non-empty");
  expect(error.message).not.toContain("proj secret");
});

describe("TS additions", () => {
  test("a non-string header value is refused without echo", () => {
    const error = thrown(() => validateHeaders({ "X-V": 2024 }, "openai"));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toBe("header X-V value must be a string");
    expect(error.message).not.toContain("2024");
  });

  test("a __proto__ header name stays an own key of the result", () => {
    const headers: unknown = JSON.parse('{"__proto__": "v1"}');
    const checked = validateHeaders(headers, "openai");
    expect(Object.hasOwn(checked, "__proto__")).toBe(true);
    expect(Object.keys(checked)).toEqual(["__proto__"]);
    expect(Object.getPrototypeOf(checked)).toBe(Object.prototype);
  });

  test("an undefined entry is skipped and not counted for the header cap", () => {
    install({ ...BUILTIN, maxConnectionHeaders: 1 });
    expect(validateHeaders({ "X-A": "1", "X-B": undefined }, null)).toEqual({ "X-A": "1" });
    expect(() => validateHeaders({ "X-A": "1", "X-B": "2" }, null)).toThrow("at most 1 headers");
  });

  test("lengths count code points, as Python's len", () => {
    install({ ...BUILTIN, maxOrganizationChars: 2, maxHeaderValueChars: 2 });
    // Two emoji are four UTF-16 units but two code points: within the cap, refused as non-ASCII.
    expect(() => validateOrganization("😀😀", "organization")).toThrow("non-empty visible ASCII");
    expect(() => validateOrganization("😀😀😀", "organization")).toThrow("organization must be at most 2 characters");
    expect(() => validateHeaderValue("X-A", "😀😀")).toThrow("non-empty visible ASCII");
    expect(() => validateHeaderName("😀😀", null)).toThrow("HTTP tokens");
    expect(() => validateHeaderName("😀😀😀", null)).toThrow("header names must be at most 2 characters");
  });

  test("for a string name the length check runs before the token check", () => {
    install({ ...BUILTIN, maxHeaderValueChars: 3 });
    expect(() => validateHeaderName("X Y-Long", null)).toThrow("header names must be at most 3 characters");
    expect(() => validateHeaderName("X Y", null)).toThrow("HTTP tokens");
  });

  test("constructor and toString are ordinary header names", () => {
    expect(validateHeaderName("constructor", "openai")).toBe("constructor");
    expect(validateHeaderName("toString", null)).toBe("toString");
    expect(validateHeaders({ constructor: "a", toString: "b" }, "openai")).toEqual({ constructor: "a", toString: "b" });
  });
});
