/** Tests for routes in `src/llm/route.ts`. */

import { describe, expect, test } from "bun:test";

import { BearerAuth } from "../../src/llm/auth.ts";
import { Route } from "../../src/llm/route.ts";

function route(overrides: Partial<ConstructorParameters<typeof Route>[0]> = {}): Route {
  return new Route({
    provider: "openai",
    baseUrl: "https://api.example/v1",
    path: "responses",
    auth: new BearerAuth("sk-test-key-0123456789"),
    ...overrides,
  });
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

describe("Route.url()", () => {
  for (const baseUrl of ["https://api.example/v1", "https://api.example/v1/"]) {
    for (const path of ["chat/completions", "/chat/completions"]) {
      test(`joins ${baseUrl} and ${path} with one slash`, () => {
        expect(route({ baseUrl, path }).url()).toBe("https://api.example/v1/chat/completions");
      });
    }
  }

  test("the query is encoded", () => {
    const url = route({ query: { "api-version": "a b&c=d" } }).url();
    const parsed = new URL(url);
    expect(parsed.searchParams.get("api-version")).toBe("a b&c=d");
    expect(url).not.toContain("a b");
    expect(url).not.toContain("&c=d");
    expect(parsed.pathname).toBe("/v1/responses");
  });

  test("no query adds no question mark", () => {
    expect(route({ query: {} }).url()).toBe("https://api.example/v1/responses");
  });
});

describe("Route", () => {
  test("copies query and headers read-only", () => {
    const query = { "api-version": "1" };
    const headers = { "x-extra": "a" };
    const r = route({ query, headers });
    query["api-version"] = "2";
    headers["x-extra"] = "b";
    expect(r.query).toEqual({ "api-version": "1" });
    expect(r.headers).toEqual({ "x-extra": "a" });
    expect(Object.isFrozen(r.query)).toBe(true);
    expect(Object.isFrozen(r.headers)).toBe(true);
    expect(() => {
      (r.query as Record<string, string>)["api-version"] = "3";
    }).toThrow(TypeError);
  });

  test("defaults to no query and no headers", () => {
    const r = route();
    expect(r.query).toEqual({});
    expect(r.headers).toEqual({});
  });

  for (const name of ["x extra", "x:extra"]) {
    test(`header name ${JSON.stringify(name)} throws`, () => {
      const error = thrown(() => route({ headers: { [name]: "v" } }));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe(`route header name ${JSON.stringify(name)} is not a valid HTTP header name`);
    });
  }

  for (const value of ["secret\nvalue", "sécret"]) {
    test(`header value ${JSON.stringify(value)} throws without echoing it`, () => {
      const error = thrown(() => route({ headers: { "x-extra": value } }));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe("route header x-extra must be non-empty HTTP header text");
      expect(error.stack ?? "").not.toContain(value);
    });
  }
});
