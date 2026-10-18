/** Tests for the auth objects, their headers and credentials, and the scrub in `src/llm/auth.ts`. */

import { describe, expect, test } from "bun:test";

import {
  BearerAuth,
  HeaderKeyAuth,
  NoAuth,
  REDACTED,
  type Secret,
  type SubscriptionAccess,
  SubscriptionAuth,
  scrub,
  secretForms,
} from "../../src/llm/auth.ts";

/** Return a source yielding `values` in turn, and a counter of its calls. */
function countingSource<T>(...values: T[]): [() => T, { count: number }] {
  const calls = { count: 0 };
  const pending = [...values];
  return [
    () => {
      calls.count++;
      return pending.shift()!;
    },
    calls,
  ];
}

function access(accessToken: string, accountId: string, residency: string | null = null): SubscriptionAccess {
  return { accessToken, accountId, residency };
}

/** Return the error `promise` rejects with, failing when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

/** Assert `promise` rejects with a `TypeError` whose message is `message` and never holds `value`. */
async function refuses(promise: Promise<unknown>, message: string, value: unknown): Promise<void> {
  const error = await rejection(promise);
  expect(error).toBeInstanceOf(TypeError);
  expect(error.message).toBe(message);
  if (typeof value === "string" && value) expect(error.stack ?? "").not.toContain(value);
}

describe("headers", () => {
  test("NoAuth", async () => {
    const auth = new NoAuth();
    expect(await auth.headers()).toEqual({});
    expect(auth.credentials()).toEqual([]);
  });

  test("BearerAuth", async () => {
    const auth = new BearerAuth("sk-static");
    expect(await auth.headers()).toEqual({ Authorization: "Bearer sk-static" });
    expect(auth.credentials()).toEqual(["sk-static"]);
  });

  for (const header of ["x-api-key", "api-key", "x-goog-api-key"]) {
    test(`HeaderKeyAuth ${header}`, async () => {
      const auth = new HeaderKeyAuth(header, "k-static");
      expect(auth.header).toBe(header);
      expect(await auth.headers()).toEqual({ [header]: "k-static" });
      expect(auth.credentials()).toEqual(["k-static"]);
    });
  }

  test("SubscriptionAuth without residency", async () => {
    const auth = new SubscriptionAuth(() => access("tok", "acct"));
    expect(await auth.headers()).toEqual({
      Authorization: "Bearer tok",
      "ChatGPT-Account-Id": "acct",
      originator: "yian",
    });
  });

  test("SubscriptionAuth with residency", async () => {
    const auth = new SubscriptionAuth(async () => access("tok", "acct", "us"));
    expect(await auth.headers()).toEqual({
      Authorization: "Bearer tok",
      "ChatGPT-Account-Id": "acct",
      originator: "yian",
      "x-openai-internal-codex-residency": "us",
    });
  });
});

describe("callable keys", () => {
  const makers: [string, (key: Secret) => BearerAuth | HeaderKeyAuth, (key: string) => Record<string, string>][] = [
    ["BearerAuth", (key) => new BearerAuth(key), (key) => ({ Authorization: `Bearer ${key}` })],
    ["HeaderKeyAuth", (key) => new HeaderKeyAuth("x-api-key", key), (key) => ({ "x-api-key": key })],
  ];
  for (const [name, make, render] of makers) {
    test(`${name} resolves on every call`, async () => {
      const [source, calls] = countingSource("k1", "k2", "k1");
      const auth = make(source);
      expect(calls.count).toBe(0);
      expect(auth.credentials()).toEqual([]);
      expect(await auth.headers()).toEqual(render("k1"));
      expect(await auth.headers()).toEqual(render("k2"));
      expect(await auth.headers()).toEqual(render("k1"));
      expect(calls.count).toBe(3);
      // In first-seen order with no duplicates, and listing never calls the source.
      expect(auth.credentials()).toEqual(["k1", "k2"]);
      expect(calls.count).toBe(3);
    });
  }

  test("an async source is awaited", async () => {
    const auth = new BearerAuth(async () => "k-async");
    expect(await auth.headers()).toEqual({ Authorization: "Bearer k-async" });
    expect(auth.credentials()).toEqual(["k-async"]);
  });

  test("SubscriptionAuth resolves on every call", async () => {
    const [source, calls] = countingSource(access("t1", "a1"), access("t2", "a1"));
    const auth = new SubscriptionAuth(source);
    expect(auth.credentials()).toEqual([]);
    expect((await auth.headers()).Authorization).toBe("Bearer t1");
    expect((await auth.headers()).Authorization).toBe("Bearer t2");
    expect(auth.credentials()).toEqual(["t1", "a1", "t2"]);
    expect(calls.count).toBe(2);
  });

  for (const bad of [null, undefined, 42, ""]) {
    test(`a source returning ${JSON.stringify(bad)} throws`, async () => {
      const source = (() => bad) as unknown as () => string;
      await refuses(new BearerAuth(source).headers(), "API key must be non-empty HTTP header text", bad);
      await refuses(new HeaderKeyAuth("x-api-key", source).headers(), "API key must be non-empty HTTP header text", bad);
    });
  }

  test("a value is recorded before it is checked", async () => {
    const auth = new BearerAuth(() => "sk-bad\nvalue");
    await refuses(auth.headers(), "API key must be non-empty HTTP header text", "sk-bad\nvalue");
    expect(auth.credentials()).toEqual(["sk-bad\nvalue"]);
  });
});

const BAD_VALUES: unknown[] = [
  "",
  "sk-abc\r\ninjected: 1",
  "sk-abc\nx",
  "sk-abc\rx",
  "sk-\x00abc",
  "sk-\tabc",
  "sk-\x7fabc",
  "sk-é",
  " sk-abc",
  "sk-abc ",
  42,
  null,
];

describe("value checks", () => {
  for (const value of BAD_VALUES) {
    const label = JSON.stringify(value);

    test(`bad static key ${label}`, async () => {
      const key = value as string;
      for (const auth of [new BearerAuth(key), new HeaderKeyAuth("x-api-key", key)]) {
        await refuses(auth.headers(), "API key must be non-empty HTTP header text", value);
      }
    });

    test(`bad callable key ${label}`, async () => {
      await refuses(new BearerAuth(() => value as string).headers(), "API key must be non-empty HTTP header text", value);
    });

    for (const field of ["accessToken", "accountId", "residency"] as const) {
      // A null residency means none, so it is not a bad value there.
      if (field === "residency" && value === null) continue;
      test(`bad subscription ${field} ${label}`, async () => {
        const values = { accessToken: "tok", accountId: "acct", residency: "us", [field]: value };
        const auth = new SubscriptionAuth(() => values as SubscriptionAccess);
        await refuses(auth.headers(), `Subscription ${field} must be non-empty HTTP header text`, value);
      });
    }
  }

  test("inner space allowed", async () => {
    expect(await new BearerAuth("a b").headers()).toEqual({ Authorization: "Bearer a b" });
  });
});

describe("secretForms", () => {
  test("names each form of a value with a slash, space, plus, quote and non-ASCII character", () => {
    expect(secretForms('a/b c+d"é')).toEqual([
      { form: "plain", text: 'a/b c+d"é' },
      { form: "percent-encoded", text: "a%2Fb%20c%2Bd%22%C3%A9" },
      { form: "plus-encoded", text: "a%2Fb+c%2Bd%22%C3%A9" },
      { form: "JSON-escaped", text: 'a/b c+d\\"\\u00e9' },
      { form: "JSON-escaped", text: 'a\\/b c+d\\"\\u00e9' },
      { form: "JSON-escaped", text: 'a/b c+d\\"é' },
      { form: "JSON-escaped", text: 'a\\/b c+d\\"é' },
      {
        form: "\\u-escaped",
        text: "\\u0061\\u002f\\u0062\\u0020\\u0063\\u002b\\u0064\\u0022\\u00e9",
        ignoreCase: true,
      },
    ]);
  });

  test("percent-encodes every character but letters, digits and _.-~", () => {
    const forms = secretForms("!'()*~-_.");
    expect(forms.find(({ form }) => form === "percent-encoded")?.text).toBe("%21%27%28%29%2A~-_.");
  });

  test("returns each text once, under the first name", () => {
    expect(secretForms("sk1")).toEqual([
      { form: "plain", text: "sk1" },
      { form: "\\u-escaped", text: "\\u0073\\u006b\\u0031", ignoreCase: true },
    ]);
  });

  test("a value with a lone surrogate has no percent or plus form", () => {
    expect(secretForms("a\ud800")).toEqual([
      { form: "plain", text: "a\ud800" },
      { form: "JSON-escaped", text: "a\\ud800" },
      { form: "\\u-escaped", text: "\\u0061\\ud800", ignoreCase: true },
    ]);
  });

  test("a JSON-escaped form that is also the all-\\u form compares case-insensitively", () => {
    expect(secretForms("é").filter(({ text }) => text === "\\u00e9")).toEqual([
      { form: "JSON-escaped", text: "\\u00e9", ignoreCase: true },
    ]);
    expect(scrub('{"k": "\\u00E9"}', ["é"])).toBe(`{"k": "${REDACTED}"}`);
  });
});

describe("scrub", () => {
  test("replaces a credential", () => {
    expect(REDACTED).toBe("[REDACTED]");
    expect(scrub("key sk-1 and sk-1", ["sk-1"])).toBe(`key ${REDACTED} and ${REDACTED}`);
  });

  test("replaces the longest credential first", () => {
    expect(scrub("sk-123456", ["sk-1", "sk-123456"])).toBe(REDACTED);
  });

  test("replaces percent- and plus-encoded forms", () => {
    expect(scrub("/k/a%2Fb%20c?k=a%2Fb+c", ["a/b c"])).toBe(`/k/${REDACTED}?k=${REDACTED}`);
  });

  test("replaces a JSON-escaped form with \\/", () => {
    expect(scrub('{"error": "bad key a\\/b\\"c"}', ['a/b"c'])).toBe(`{"error": "bad key ${REDACTED}"}`);
  });

  test("replaces the all-\\u form in lower and upper hex", () => {
    expect(scrub("x \\u0073\\u006b\\u0031 y", ["sk1"])).toBe(`x ${REDACTED} y`);
    expect(scrub("x \\u0073\\u006B\\u0031 y", ["sk1"])).toBe(`x ${REDACTED} y`);
  });

  test("ignores an empty credential", () => {
    expect(scrub("text", ["", "absent"])).toBe("text");
  });

  test("uses a custom marker, literally", () => {
    expect(scrub("key sk-1", ["sk-1"], "<redacted>")).toBe("key <redacted>");
    expect(scrub("key sk-1", ["sk-1"], "$&")).toBe("key $&");
    expect(scrub("\\u0073\\u006B\\u0031", ["sk1"], "$&")).toBe("$&");
  });
});
