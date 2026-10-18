/** Tests for provider errors in `src/llm/errors.ts`: status mapping, retry policy, retry-after parsing, messages. */

import { describe, expect, test } from "bun:test";

import {
  capDetail,
  ERROR_KINDS,
  errorFromResponse,
  kindForStatus,
  parseRetryAfter,
  ProviderError,
  redirectDetail,
  type ErrorKind,
} from "../../src/llm/errors.ts";

/** Return the error for a reply from provider `p` with `status`, header record `headers` and `body`. */
function reply(status: number, headers: Record<string, string> = {}, body = "", maxRetryAfter?: number | null) {
  const options = maxRetryAfter === undefined ? { provider: "p" } : { provider: "p", maxRetryAfter };
  return errorFromResponse(status, new Headers(headers), body, options);
}

/** Return the part of `err`'s message after the first ": ". */
function detailOf(err: ProviderError): string {
  return err.message.slice(err.message.indexOf(": ") + 2);
}

describe("kinds", () => {
  const rows: [number, string, ErrorKind][] = [
    [401, "", "auth"],
    [403, "", "permission"],
    [404, "", "not_found"],
    [408, "", "timeout"],
    [429, "", "rate_limit"],
    [529, "", "overloaded"],
    [503, '{"error": {"type": "Overloaded_error"}}', "overloaded"],
    [503, "busy", "server"],
    [500, "", "server"],
    [502, "", "server"],
    [599, "", "server"],
    [301, "", "redirect"],
    [307, "", "redirect"],
    [400, "", "bad_request"],
    [409, "", "bad_request"],
    [413, "", "bad_request"],
    [422, "", "bad_request"],
    [402, "", "quota"],
    [429, '{"error":{"code":"insufficient_quota"}}', "quota"],
    [429, "INSUFFICIENT_QUOTA", "quota"],
  ];

  test.each(rows)("status maps to kind: %d %s -> %s", (status, body, kind) => {
    const err = errorFromResponse(status, new Headers(), body, { provider: "openai" });
    expect(err.kind).toBe(kind);
    expect(err.status).toBe(status);
    expect(err.provider).toBe("openai");
    expect(kindForStatus(status, body)).toBe(kind);
  });

  const statuses = [400, 413, 422];
  const phrases = ["context_length_exceeded", "Prompt is too long", "MAXIMUM CONTEXT LENGTH"];
  test.each(statuses.flatMap((status) => phrases.map((phrase): [number, string] => [status, phrase])))(
    "context overflow body: %d %s",
    (status, phrase) => {
      const body = JSON.stringify({ error: { message: `oops: ${phrase} here` } });
      expect(reply(status, {}, body).kind).toBe("context_length");
    },
  );

  test("context phrase on other status is not context_length", () => {
    expect(reply(500, {}, "maximum context length").kind).toBe("server");
  });

  test("an unexpected status is a protocol error", () => {
    expect(kindForStatus(200, "")).toBe("protocol");
    expect(kindForStatus(101, "")).toBe("protocol");
  });

  test("ERROR_KINDS lists every kind, frozen", () => {
    expect([...ERROR_KINDS]).toEqual([
      "auth",
      "permission",
      "not_found",
      "rate_limit",
      "quota",
      "overloaded",
      "bad_request",
      "context_length",
      "refused",
      "server",
      "timeout",
      "connection",
      "redirect",
      "incomplete",
      "protocol",
    ]);
    expect(Object.isFrozen(ERROR_KINDS)).toBe(true);
  });
});

describe("retry policy", () => {
  const rows: [number, string, boolean][] = [
    [400, "", false],
    [401, "", false],
    [404, "", false],
    [408, "", true],
    [409, "", true],
    [429, "", true],
    [500, "", true],
    [503, "", true],
    [529, "", true],
    [301, "", false],
    [402, "", false],
    [429, '{"error":{"code":"insufficient_quota"}}', false],
  ];

  test.each(rows)("retryable by status: %d %s -> %p", (status, body, retryable) => {
    expect(reply(status, {}, body).retryable).toBe(retryable);
  });

  test("should-retry header beats status", () => {
    expect(reply(400, { "x-should-retry": "true" }).retryable).toBe(true);
    expect(reply(500, { "x-should-retry": "false" }).retryable).toBe(false);
    expect(reply(429, { "X-Should-Retry": "false" }).retryable).toBe(false);
    expect(reply(429, { "x-should-retry": "true" }, "insufficient_quota").retryable).toBe(true);
    expect(reply(402, { "x-should-retry": "true" }).retryable).toBe(true);
  });

  test("long retry-after disables retry", () => {
    const err = reply(429, { "retry-after": "121" });
    expect(err.retryable).toBe(false);
    expect(err.retryAfter).toBe(121);
    // The retry-after refusal comes before x-should-retry, as in openai.
    expect(reply(429, { "retry-after": "500", "x-should-retry": "true" }).retryable).toBe(false);
    expect(reply(429, { "retry-after": "120" }).retryable).toBe(true);
  });

  test("errorFromResponse passes now to the HTTP-date retry-after", () => {
    const now = Date.UTC(2027, 0, 15, 8, 0, 0) / 1000;
    const soon = errorFromResponse(429, new Headers({ "retry-after": "Fri, 15 Jan 2027 08:00:30 GMT" }), "", {
      provider: "p",
      now,
    });
    expect(soon.retryAfter).toBe(30);
    expect(soon.retryable).toBe(true);
    const late = errorFromResponse(429, new Headers({ "retry-after": "Fri, 15 Jan 2027 08:03:20 GMT" }), "", {
      provider: "p",
      now,
    });
    expect(late.retryAfter).toBe(200);
    expect(late.retryable).toBe(false);
  });

  test("no maxRetryAfter keeps retry", () => {
    const err = reply(429, { "retry-after": "500" }, "", null);
    expect(err.retryable).toBe(true);
    expect(err.retryAfter).toBe(500);
  });

  test.each(["inf", "nan"])("literal infinite retry-after falls through: %s", (value) => {
    expect(reply(429, { "retry-after": value }).retryable).toBe(true);
    expect(reply(400, { "retry-after": value }).retryable).toBe(false);
  });
});

describe("ProviderError", () => {
  test("error text is the message", () => {
    const err = new ProviderError("protocol", "bad frame");
    expect(err).toBeInstanceOf(Error);
    expect(err.message).toBe("bad frame");
    expect(err.name).toBe("ProviderError");
    expect(String(err)).toBe("ProviderError: bad frame");
    expect([err.status, err.retryAfter, err.provider, err.causeName]).toEqual([null, null, null, null]);
    expect(err.retryable).toBe(false);
  });

  test("keeps every option", () => {
    const err = new ProviderError("connection", "reset", {
      status: 502,
      retryAfter: 1.5,
      retryable: false,
      provider: "openai",
      causeName: "TypeError",
    });
    expect(err.kind).toBe("connection");
    expect(err.message).toBe("reset");
    expect(err.status).toBe(502);
    expect(err.retryAfter).toBe(1.5);
    expect(err.retryable).toBe(false);
    expect(err.provider).toBe("openai");
    expect(err.causeName).toBe("TypeError");
    err.provider = "azure";
    expect(err.provider).toBe("azure");
  });

  const retryableKinds = new Set<ErrorKind>(["rate_limit", "overloaded", "server", "timeout", "connection"]);
  test.each(ERROR_KINDS.map((kind): [ErrorKind, boolean] => [kind, retryableKinds.has(kind)]))(
    "retryable defaults by kind: %s -> %p",
    (kind, retryable) => {
      expect(new ProviderError(kind, "x").retryable).toBe(retryable);
    },
  );
});

describe("parseRetryAfter", () => {
  test("milliseconds preferred", () => {
    expect(parseRetryAfter(new Headers({ "retry-after-ms": "1500", "retry-after": "9" }))).toBe(1.5);
  });

  test("seconds", () => {
    expect(parseRetryAfter(new Headers({ "retry-after": "7" }))).toBe(7);
  });

  test("float seconds", () => {
    expect(parseRetryAfter(new Headers({ "Retry-After": "0.25" }))).toBe(0.25);
  });

  test("HTTP date", () => {
    const now = Date.UTC(2027, 0, 15, 8, 0, 0) / 1000;
    expect(parseRetryAfter(new Headers({ "retry-after": "Fri, 15 Jan 2027 08:00:30 GMT" }), now)).toBe(30);
    // The day name is optional and not checked.
    expect(parseRetryAfter(new Headers({ "retry-after": "15 Jan 2027 08:00:30 GMT" }), now)).toBe(30);
  });

  test("HTTP date in the past is negative", () => {
    const now = Date.UTC(2027, 0, 15, 8, 1, 0) / 1000;
    expect(parseRetryAfter(new Headers({ "retry-after": "Fri, 15 Jan 2027 08:00:30 GMT" }), now)).toBe(-30);
  });

  test("HTTP date with a numeric zone", () => {
    const now = Date.UTC(2027, 0, 15, 8, 0, 0) / 1000;
    expect(parseRetryAfter(new Headers({ "retry-after": "Fri, 15 Jan 2027 08:00:30 +0000" }), now)).toBe(30);
    expect(parseRetryAfter(new Headers({ "retry-after": "Fri, 15 Jan 2027 09:30:30 +0130" }), now)).toBe(30);
  });

  test("HTTP date with a one-digit day and no seconds", () => {
    const now = Date.UTC(2027, 0, 5, 8, 0, 0) / 1000;
    expect(parseRetryAfter(new Headers({ "retry-after": "Tue, 5 Jan 2027 08:01 GMT" }), now)).toBe(60);
  });

  test.each(["Fri, 15 Jan 2027 08:00:30", "Fri, 15 Foo 2027 08:00:30 GMT", "Fri, 15 Jan 27 08:00:30 GMT"])(
    "an unsupported or unparsable date is null: %s",
    (value) => {
      expect(parseRetryAfter(new Headers({ "retry-after": value }), 0)).toBeNull();
    },
  );

  test("a four-digit year below 100 follows Python's two-digit rule", () => {
    const now = Date.UTC(2050, 0, 15, 8, 0, 0) / 1000;
    expect(parseRetryAfter(new Headers({ "retry-after": "15 Jan 0050 08:00:30 GMT" }), now)).toBe(30);
    const then = Date.UTC(1999, 0, 15, 8, 0, 0) / 1000;
    expect(parseRetryAfter(new Headers({ "retry-after": "15 Jan 0099 08:00:30 GMT" }), then)).toBe(30);
  });

  test("a long numeric header is refused in linear time", () => {
    const start = performance.now();
    expect(parseRetryAfter(new Headers({ "retry-after": "1".repeat(32768) + "x" }))).toBeNull();
    expect(performance.now() - start).toBeLessThan(100);
  });

  test("garbage ms falls back to seconds", () => {
    expect(parseRetryAfter(new Headers({ "retry-after-ms": "soon", "retry-after": "3" }))).toBe(3);
  });

  test.each([
    ["absent", {}],
    ["garbage", { "retry-after": "soon" }],
    ["empty", { "retry-after": "" }],
  ])("absent or garbage: %s", (_, headers) => {
    expect(parseRetryAfter(new Headers(headers))).toBeNull();
  });

  test.each(["retry-after", "retry-after-ms"])("overflow is the sentinel: %s", (name) => {
    expect(parseRetryAfter(new Headers({ [name]: "1e400" }))).toBe(121);
    expect(parseRetryAfter(new Headers({ [name]: "9".repeat(400) }))).toBe(121);
  });

  test("overflow refuses retry by default", () => {
    const err = reply(503, { "retry-after": "9".repeat(400) });
    expect(err.retryAfter).toBe(121);
    expect(err.retryable).toBe(false);
  });

  test("signed literals and a negative overflow", () => {
    expect(parseRetryAfter(new Headers({ "retry-after": "inf" }))).toBe(Infinity);
    expect(parseRetryAfter(new Headers({ "retry-after": " +Infinity " }))).toBe(Infinity);
    expect(parseRetryAfter(new Headers({ "retry-after": "-inf" }))).toBe(-Infinity);
    expect(parseRetryAfter(new Headers({ "retry-after-ms": "-INF" }))).toBe(-Infinity);
    expect(parseRetryAfter(new Headers({ "retry-after": "-1e400" }))).toBe(-Infinity);
    expect(parseRetryAfter(new Headers({ "retry-after": "+nan" }))).toBeNaN();
  });

  test("float text grammar", () => {
    expect(parseRetryAfter(new Headers({ "retry-after": " .5 " }))).toBe(0.5);
    expect(parseRetryAfter(new Headers({ "retry-after": "2." }))).toBe(2);
    expect(parseRetryAfter(new Headers({ "retry-after": "+1e1" }))).toBe(10);
    expect(parseRetryAfter(new Headers({ "retry-after": "1_0" }))).toBeNull();
    expect(parseRetryAfter(new Headers({ "retry-after": "0x10" }))).toBeNull();
  });
});

describe("message", () => {
  test("JSON error message", () => {
    const body = JSON.stringify({ error: { message: "Invalid model", type: "x" } });
    const err = errorFromResponse(400, new Headers(), body, { provider: "openai" });
    expect(err.message).toBe("openai returned HTTP 400 (bad_request): Invalid model");
    expect(String(err)).toBe(`ProviderError: ${err.message}`);
  });

  test("JSON top-level message and detail", () => {
    expect(reply(404, {}, '{"message": "no such"}').message.endsWith(": no such")).toBe(true);
    expect(reply(422, {}, '{"detail": "bad field"}').message.endsWith(": bad field")).toBe(true);
    // A blank candidate gives way to the next.
    expect(reply(422, {}, '{"error": {"message": " "}, "detail": "bad field"}').message.endsWith(": bad field")).toBe(
      true,
    );
  });

  test("plain text body collapses whitespace", () => {
    const err = errorFromResponse(502, new Headers(), "  Bad\n\n gateway\t here ", { provider: "azure" });
    expect(err.message).toBe("azure returned HTTP 502 (server): Bad gateway here");
  });

  test("JSON without known fields uses the text", () => {
    expect(reply(500, {}, '{"code": 7}').message.endsWith(': {"code": 7}')).toBe(true);
    expect(reply(500, {}, '["a"]').message.endsWith(': ["a"]')).toBe(true);
  });

  test("a duplicate-key JSON body falls back to the text", () => {
    const err = reply(500, {}, '{"message": "a", "message": "b"}');
    expect(detailOf(err)).toBe('{"message": "a", "message": "b"}');
  });

  test("empty body", () => {
    expect(reply(500, {}, "  \n").message).toBe("p returned HTTP 500 (server): no body");
  });

  test("detail cut at 2,000 code points", () => {
    expect(detailOf(reply(500, {}, "x".repeat(5000)))).toBe("x".repeat(2000) + "…");
    const astral = "\u{1F600}".repeat(2001);
    expect(detailOf(reply(500, {}, astral))).toBe("\u{1F600}".repeat(2000) + "…");
    expect(capDetail("\u{1F600}".repeat(2000))).toBe("\u{1F600}".repeat(2000));
    expect(capDetail("short")).toBe("short");
  });

  test("redirect names only scheme, host and port", () => {
    const headers = { Location: "https://user:pw@evil.example.com:8443/steal/path?token=abc#f" };
    const err = reply(302, headers, "see /steal/path?token=abc");
    expect(err.kind).toBe("redirect");
    expect(err.message).toBe("p returned HTTP 302 (redirect): redirect to https://evil.example.com:8443");
    for (const leak of ["steal", "token", "abc", "user", "pw"]) expect(err.message).not.toContain(leak);
  });

  test("redirect without Location", () => {
    expect(reply(301).message).toBe("p returned HTTP 301 (redirect): redirect without a Location header");
  });

  test("redirect to a Location without a scheme", () => {
    const err = reply(302, { Location: "//evil.example.com/steal?token=abc" });
    expect(err.message.endsWith(": redirect to a Location without a scheme")).toBe(true);
    for (const leak of ["evil", "steal", "token"]) expect(err.message).not.toContain(leak);
  });

  test("redirect to a relative Location", () => {
    expect(reply(302, { Location: "/steal?token=abc" }).message.endsWith(": redirect to a relative Location")).toBe(
      true,
    );
  });

  test("redirectDetail", () => {
    expect(redirectDetail(null)).toBe("redirect without a Location header");
    expect(redirectDetail("")).toBe("redirect without a Location header");
    expect(redirectDetail("http://[::1")).toBe("redirect to an unparsable Location");
    expect(redirectDetail("//[::1")).toBe("redirect to an unparsable Location");
    expect(redirectDetail("https://[::1]:8443/x?t=1")).toBe("redirect to https://[::1]:8443");
    expect(redirectDetail("https://h.example:443/x")).toBe("redirect to https://h.example");
    expect(redirectDetail(" HTTP://H.Example/x ")).toBe("redirect to http://h.example");
    expect(redirectDetail("mailto:x")).toBe("redirect to a relative Location");
    expect(redirectDetail("///x")).toBe("redirect to a relative Location");
    expect(redirectDetail("https:///x")).toBe("redirect to a relative Location");
    expect(redirectDetail("//h.example/x")).toBe("redirect to a Location without a scheme");
    expect(redirectDetail("x/y")).toBe("redirect to a relative Location");
  });
});
