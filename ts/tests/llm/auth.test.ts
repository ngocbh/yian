/** Tests for the auth objects, their headers and credentials, and the scrub in `src/llm/auth.ts`. */

import { inspect } from "node:util";

import { describe, expect, test } from "bun:test";

import {
  type Auth,
  BasicAuth,
  BearerAuth,
  ExtraHeadersAuth,
  HeaderKeyAuth,
  NoAuth,
  REDACTED,
  type Secret,
  type SubscriptionAccess,
  SubscriptionAuth,
  scrub,
  secretForms,
} from "../../src/llm/auth.ts";
import { current } from "../../src/defaults.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { RouteModel } from "../../src/llm/executor.ts";
import { Request } from "../../src/llm/messages.ts";
import { Route } from "../../src/llm/route.ts";
import { validateHeaderValue } from "../../src/llm_clients/connection_settings.ts";

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

/** Return the error a `RouteModel` over `auth` ends with when the server answers 401 echoing `secret`. */
async function errorEchoing(secret: string, auth: Auth): Promise<ProviderError> {
  const send = async () => new Response(JSON.stringify({ error: `bad credential ${secret}` }), { status: 401 });
  const model = new RouteModel({
    route: new Route({ provider: "ollama", baseUrl: "http://gpu-box:11434", path: "api/chat", auth }),
    protocol: { name: "test", model: "m", framing: "sse", buildBody: () => ({}), async *parse() {} },
    send,
  });
  const error = await rejection(
    (async () => {
      for await (const _ of model.stream(new Request({ messages: [] })));
    })(),
  );
  expect(error).toBeInstanceOf(ProviderError);
  return error as ProviderError;
}

describe("BasicAuth", () => {
  test("sends Basic and the base64 of user:password", async () => {
    const auth = new BasicAuth("u", "p");
    expect(await auth.headers()).toEqual({ Authorization: "Basic dTpw" });
    expect(auth.credentials()).toEqual(["p", "dTpw"]);
  });

  test("encodes a non-ASCII user as UTF-8", async () => {
    const auth = new BasicAuth("üser", "pw");
    // base64 of the UTF-8 bytes c3 bc 73 65 72 3a 70 77.
    expect(await auth.headers()).toEqual({ Authorization: "Basic w7xzZXI6cHc=" });
    expect(auth.credentials()).toEqual(["pw", "w7xzZXI6cHc="]);
  });

  test("an empty password leaves only the token in the credentials", async () => {
    const auth = new BasicAuth("u", "");
    expect(await auth.headers()).toEqual({ Authorization: "Basic dTo=" });
    expect(auth.credentials()).toEqual(["dTo="]);
  });

  test("a 401 echoing the password is scrubbed from the error", async () => {
    const password = "hunter2-secret-pw";
    const error = await errorEchoing(password, new BasicAuth("me", password));
    expect(error.message).toContain(REDACTED);
    expect(error.message).not.toContain(password);
    expect(error.stack ?? "").not.toContain(password);
  });
});

describe("ExtraHeadersAuth", () => {
  test("merges the headers, the inner auth's winning a clash", async () => {
    const auth = new ExtraHeadersAuth(new BearerAuth("sk-1"), { "x-extra": "e1", Authorization: "lost" });
    expect(await auth.headers()).toEqual({ "x-extra": "e1", Authorization: "Bearer sk-1" });
  });

  test("lists the header values after the inner auth's credentials, without duplicates", () => {
    const auth = new ExtraHeadersAuth(new BearerAuth("sk-1"), { "x-a": "v-a", "x-b": "sk-1", "x-c": "v-c" });
    expect(auth.credentials()).toEqual(["sk-1", "v-a", "v-c"]);
  });

  test("accepts an inner tab", async () => {
    const auth = new ExtraHeadersAuth(new NoAuth(), { "x-tab": "a\tb" });
    expect(await auth.headers()).toEqual({ "x-tab": "a\tb" });
  });

  for (const value of ["sec\rret", "\tsecret", "secret ", "", "sécret"]) {
    test(`refuses ${JSON.stringify(value)} without echoing it`, () => {
      let error: unknown;
      try {
        new ExtraHeadersAuth(new NoAuth(), { "x-bad": value });
      } catch (caught) {
        error = caught;
      }
      expect(error).toBeInstanceOf(TypeError);
      expect((error as TypeError).message).toBe(
        "header x-bad value must be non-empty visible ASCII, with spaces or tabs only between characters",
      );
      if (value) expect((error as TypeError).stack ?? "").not.toContain(value);
    });
  }

  test("an extra header value is scrubbed from the error", async () => {
    const value = "org-secret-header-value";
    const error = await errorEchoing(value, new ExtraHeadersAuth(new NoAuth(), { "x-org-secret": value }));
    expect(error.message).toContain(REDACTED);
    expect(error.message).not.toContain(value);
    expect(error.stack ?? "").not.toContain(value);
  });

  test("checks values exactly as validateHeaderValue does", () => {
    const limit = current().maxHeaderValueChars;
    const values: unknown[] = [
      ...["a\tb", "a b", " a", "a ", "\ta", "a\t", "a\nb", "é", ""],
      "x".repeat(limit),
      "x".repeat(limit + 1),
      42,
    ];
    /** Return `"ok"` or the message `check` throws, and the class it throws. */
    const outcome = (check: () => unknown): [string, string] => {
      try {
        check();
        return ["ok", ""];
      } catch (error) {
        return [(error as Error).message, (error as Error).constructor.name];
      }
    };
    for (const value of values) {
      const ours = outcome(() => new ExtraHeadersAuth(new NoAuth(), { x: value as string }));
      const theirs = outcome(() => validateHeaderValue("x", value));
      expect(ours[0], JSON.stringify(value)).toBe(theirs[0]);
      if (ours[0] !== "ok") expect([ours[1], theirs[1]]).toEqual(["TypeError", "RangeError"]);
    }
    // Both accept exactly these.
    expect(values.filter((value) => outcome(() => validateHeaderValue("x", value))[0] === "ok")).toEqual([
      "a\tb",
      "a b",
      "x".repeat(limit),
    ]);
  });

  test("copies its input", async () => {
    const headers = { "x-extra": "before" };
    const auth = new ExtraHeadersAuth(new NoAuth(), headers);
    headers["x-extra"] = "after";
    expect(await auth.headers()).toEqual({ "x-extra": "before" });
    expect(auth.credentials()).toEqual(["before"]);
  });
});

describe("inspect", () => {
  const cases: [string, () => Auth][] = [
    ["BearerAuth", () => new BearerAuth("sk-bearer-secret")],
    ["BearerAuth with a callable key", () => new BearerAuth(() => "sk-callable-secret")],
    ["HeaderKeyAuth", () => new HeaderKeyAuth("x-api-key", "k-header-secret")],
    ["SubscriptionAuth", () => new SubscriptionAuth(() => access("tok-secret", "acct-secret", "us"))],
    ["BasicAuth", () => new BasicAuth("user", "pw-secret")],
    ["ExtraHeadersAuth", () => new ExtraHeadersAuth(new BearerAuth("sk-inner-secret"), { "x-gateway": "gw-secret" })],
  ];
  for (const [name, make] of cases) {
    test(`${name} shows no secret, even after sending them`, async () => {
      const auth = make();
      await auth.headers();
      const secrets = auth.credentials();
      expect(secrets.length).toBeGreaterThan(0);
      const shown = [inspect(auth), inspect(auth, { depth: Infinity })];
      for (const text of shown) for (const secret of secrets) expect(text).not.toContain(secret);
    });
  }
});
