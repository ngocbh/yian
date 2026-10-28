/**
 * The subscription route `chatModelFromConfig` builds from a saved login, against a recording
 * `send`: an auth failure never falls back to an API key, and missing credentials fail before
 * any request.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { collect } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { AuthError, isAuthFailure, OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { chatModelFromConfig } from "../../src/llm_clients/factory.ts";
import type { Send } from "../../src/platform.ts";

// Each test's environment: a private store in a fresh directory, holding one subscription login.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(async () => {
  home = mkdtempSync(join(tmpdir(), "yian-openai-subscription-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
  await new AuthStore(null, { env }).save(
    "openai",
    new OAuthCredentials({
      accessToken: "synthetic-access",
      refreshToken: "synthetic-refresh",
      // Far ahead, so no read refreshes.
      expiresAt: 4_102_444_800,
      accountId: "synthetic-account",
    }),
    "gpt-5.5",
  );
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** A `send` that records each request's headers and answers it with `status` and a JSON error. */
function recorder(status: number): { requests: Headers[]; send: Send } {
  const requests: Headers[] = [];
  const send: Send = async (_input, init) => {
    requests.push(new Headers(init.headers));
    return new Response(JSON.stringify({ error: { message: "rejected" } }), {
      status,
      headers: { "content-type": "application/json" },
    });
  };
  return { requests, send };
}

const HI = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("Hi")] })] });

/** Return the error `promise` rejects with, failing the test if it resolves. */
async function rejected(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

test("auth failure does not fall back to api key", async () => {
  vars.set("OPENAI_API_KEY", "must-not-be-used");
  const { requests, send } = recorder(401);
  const model = await chatModelFromConfig({}, { env, send, onWarning: () => {} });
  const error = await rejected(collect(model.stream(HI)));
  expect(isAuthFailure(error)).toBe(true);
  expect(requests.map((headers) => headers.get("authorization"))).toEqual(["Bearer synthetic-access"]);
});

test("missing credentials fail before http", async () => {
  const { requests, send } = recorder(200);
  const model = await chatModelFromConfig({}, { env, send, onWarning: () => {} });
  await new AuthStore(null, { env }).remove("openai");
  const error = await rejected(collect(model.stream(HI)));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("yian auth openai");
  expect(requests).toEqual([]);
});
