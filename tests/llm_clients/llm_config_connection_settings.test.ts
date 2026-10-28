/** `LLMConfig`'s connection settings: organization, project and headers. */

import { inspect } from "node:util";

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
import { LLMConfig } from "../../src/llm_clients/config.ts";

import {
  OPENAI_REFUSED_HEADER_NAMES,
  openaiIdHeaders,
  REFUSED_HEADER_NAMES,
  validateHeaderName,
  validateHeaders,
  validateHeaderValue,
  validateOrganization,
} from "../../src/llm_clients/connection_settings.ts";
import { ValidationError } from "../../src/llm_clients/provider_blocks.ts";

afterEach(() => install(BUILTIN));

const KEY = "sk-sentinel-7d1f3b20";
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

/** Return an `openai` config with `KEY` and `fields`. */
function config(fields: Record<string, unknown>): LLMConfig {
  return new LLMConfig({ provider: "openai", api_key: KEY, ...fields });
}

/** Assert that `fields` are refused with `part` and that no value is echoed; return the message. */
function refused(part: string, fields: Record<string, unknown>): string {
  const error = thrown(() => new LLMConfig(fields));
  expect(error).toBeInstanceOf(ValidationError);
  expect(error.message).toContain(part);
  expect(error.message).not.toContain(KEY);
  expect(error.message).not.toContain(GATEWAY);
  return error.message;
}

test("valid settings are kept", () => {
  const kept = config({
    organization: "org-abc123",
    project: "proj_XYZ",
    headers: { "X-Gateway": GATEWAY, "X-Trace": "a b\tc" },
  });
  expect(kept.organization).toBe("org-abc123");
  expect(kept.project).toBe("proj_XYZ");
  expect(new Set(Object.keys(kept.headers!))).toEqual(new Set(["X-Gateway", "X-Trace"]));
  expect(kept.headers!["X-Gateway"]).toBe(GATEWAY);
  expect(Object.isFrozen(kept.headers)).toBe(true);
});

test("header values stay out of dumps and inspect", () => {
  const kept = config({ headers: { "X-Gateway": GATEWAY } });
  for (const text of [inspect(kept), String(kept), JSON.stringify(kept)]) {
    expect(text).not.toContain(GATEWAY);
  }
  expect(Object.hasOwn(kept.dump(), "headers")).toBe(false);
});

test("organization and project dump when set and add nothing when unset", () => {
  expect(config({ model: "gpt-5.5", organization: "org-1", project: "proj-1" }).dump()).toEqual({
    provider: "openai",
    model: "gpt-5.5",
    organization: "org-1",
    project: "proj-1",
  });
  // Recorded snapshots compare full dumps, so unset settings must add nothing.
  expect(new LLMConfig({ provider: "openai" }).dump()).toEqual({ provider: "openai", model: null });
});

test("an empty header mapping is no headers", () => {
  expect(new LLMConfig({ provider: "openai", headers: {} }).headers).toBeNull();
});

test("credential header names are refused for every provider", () => {
  const spellings = [
    (name: string) => name,
    (name: string) => name.toLowerCase(),
    (name: string) => name.toUpperCase(),
  ];
  for (const provider of ["openai", "deepseek", "anthropic", "google", "azure"]) {
    for (const name of CREDENTIAL_NAMES) {
      for (const spell of spellings) {
        refused("yian sends the credential itself", { provider, api_key: KEY, headers: { [spell(name)]: GATEWAY } });
      }
    }
  }
});

test.each(TRANSPORT_NAMES)("transport header name %s is refused", (name) => {
  refused("yian's client sets it", { provider: "deepseek", api_key: KEY, headers: { [name.toLowerCase()]: GATEWAY } });
});

test.each(["OpenAI-Organization", "openai-project"])("openai id header %s is refused for openai only", (name) => {
  refused("use llm_config.organization / project", { provider: "openai", api_key: KEY, headers: { [name]: GATEWAY } });
  const kept = new LLMConfig({ provider: "deepseek", api_key: KEY, headers: { [name]: "proj-1" } });
  expect(Object.keys(kept.headers!)).toEqual([name]);
});

test("duplicate names by case are refused", () => {
  refused("more than once", {
    provider: "openai",
    api_key: KEY,
    headers: { "X-Gateway": GATEWAY, "x-gateway": GATEWAY },
  });
});

const NOT_TOKENS = ["", "X Gateway", "X:Gateway", "X-Gätway", "X\nY", "(x)"];

test.each(NOT_TOKENS)("name %p that is not a token is refused", (name) => {
  const text = refused("HTTP token", { provider: "openai", api_key: KEY, headers: { [name]: GATEWAY } });
  if (name) expect(text).not.toContain(name);
});

test.each([
  `${GATEWAY}\r`,
  `${GATEWAY}\nX-Evil: 1`,
  `${GATEWAY}\x00`,
  `${GATEWAY}\x01`,
  `${GATEWAY}\x7f`,
  `${GATEWAY}é`,
  ` ${GATEWAY}`,
  `${GATEWAY} `,
  `${GATEWAY}\t`,
  "",
])("bad header value %p is refused", (value) => {
  refused("visible ASCII", { provider: "openai", api_key: KEY, headers: { "X-Gateway": value } });
});

test("header value and count caps", () => {
  install({ ...BUILTIN, maxHeaderValueChars: 10, maxConnectionHeaders: 2 });
  expect(config({ headers: { A: "x".repeat(10), B: "y" } }).headers!["A"]).toBe("x".repeat(10));
  const over = "secret-val!";
  const text = refused("at most 10 characters", { provider: "openai", api_key: KEY, headers: { A: over } });
  expect(text).not.toContain(over);
  refused("at most 2 headers", { provider: "openai", api_key: KEY, headers: { A: "1", B: "2", C: "3" } });
});

test("header names share the value cap", () => {
  install({ ...BUILTIN, maxHeaderValueChars: 10 });
  expect(Object.keys(config({ headers: { ["X".repeat(10)]: "1" } }).headers!)).toEqual(["X".repeat(10)]);
  const longName = "X-Long-Name";
  const text = refused("header names must be at most 10 characters", {
    provider: "openai",
    api_key: KEY,
    headers: { [longName]: "1" },
  });
  expect(text).not.toContain(longName);
});

test("a non-string header value is refused", () => {
  const error = thrown(() => config({ headers: { "X-V": 2024n } }));
  expect(error).toBeInstanceOf(ValidationError);
  expect(error.message).not.toContain("2024");
});

test.each(["organization", "project"])("%s values", (field) => {
  install({ ...BUILTIN, maxOrganizationChars: 8 });
  expect(config({ [field]: "org-1234" })[field as "organization" | "project"]).toBe("org-1234");
  const secretish = "org-45678";
  const text = refused("at most 8 characters", { provider: "openai", api_key: KEY, [field]: secretish });
  expect(text).not.toContain(secretish);
  for (const bad of ["", "org 1", "org\n1", "orgé", "org\x7f"]) {
    refused("visible ASCII", { provider: "openai", api_key: KEY, [field]: bad });
  }
});

test("organization and project are openai only", () => {
  for (const provider of ["deepseek", "azure", "anthropic", "vllm"]) {
    for (const field of ["organization", "project"]) {
      refused("only to provider openai", { provider, api_key: KEY, [field]: "org-1" });
    }
  }
});

test("headers are refused for ollama", () => {
  refused("provider ollama does not take llm_config.headers", {
    provider: "ollama",
    api_key: KEY,
    headers: { "X-Gateway": GATEWAY },
  });
});

test.each(["organization", "project"])("a wrong provider is reported before a missing key (%s)", (field) => {
  refused("only to provider openai", { provider: "deepseek", [field]: "org-1" });
});

test("ollama headers are reported before a missing key", () => {
  refused("provider ollama does not take llm_config.headers", {
    provider: "ollama",
    headers: { "X-Gateway": GATEWAY },
  });
});

test("an openai id header is reported before a missing key", () => {
  refused("use llm_config.organization / project", { provider: "openai", headers: { "OpenAI-Project": GATEWAY } });
});

test("each setting needs key and provider in the same config", () => {
  for (const fields of [{ organization: "org-1" }, { project: "proj-1" }, { headers: { "X-Gateway": GATEWAY } }]) {
    for (const missing of ["api_key", "provider"]) {
      const given: Record<string, unknown> = { provider: "openai", api_key: KEY, ...fields };
      delete given[missing];
      refused("needs llm_config.api_key and llm_config.provider", given);
    }
  }
});

test("azure settings need the key too", () => {
  refused("needs llm_config.api_key", {
    provider: "azure",
    base_url: "https://res.openai.azure.com/",
    headers: { "X-G": GATEWAY },
  });
});

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

describe("edge cases", () => {
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

  test("lengths count code points", () => {
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
