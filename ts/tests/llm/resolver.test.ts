/**
 * The resolver case table for `src/llm_clients/factory.ts`'s `chatModelFromConfig`: the planned
 * case table of #366's resolver stages (`tests/llm/resolver_cases.py` and `test_resolver*.py` in
 * Python, never built), so it is a new file rather than a port of one.
 *
 * Each case gives a config, an optional saved login and an environment, and replays one golden
 * cassette from `cassettes/<protocol>/`, deep-copied and edited only where the case differs (the
 * URL, a header, the body). A case that fails names the error's class and exact message, with no
 * request sent. Every case sends the request the matching `*_cassettes.test.ts` sends for its
 * golden (`REQUESTS`). Later providers append cases and requests.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { collect } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { type Cassette, CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { ApiKeyCredentials, AuthError, OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import {
  type AgentConfig,
  type Connection,
  keyIdentity,
  legacyConnectionConfig,
  OPENAI_API_ENDPOINT,
  resolveModelConfig,
} from "../../src/llm_clients/connection.ts";
import { accountHash, chatModelFromConfig } from "../../src/llm_clients/factory.ts";
import { virtualScheduler } from "../../src/platform.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

const CASSETTES = fileURLToPath(new URL("cassettes/", import.meta.url));
// Far enough ahead that the build-time read never refreshes: the default refresh would use
// `OpenAIAuth`'s own transport, not the case's replay.
const FAR_FUTURE = 4_102_444_800;
const HOST_KEY = "test-host-key";
const SAVED_KEY = "test-saved-key";
const ENV_KEY = "test-env-key";
const ACCESS = "test-access-token";
const ACCOUNT = "test-account-1";

const system = (text: string) => new Message({ role: "system", parts: [new TextPart(text)] });
const user = (text: string) => new Message({ role: "user", parts: [new TextPart(text)] });

/** The request each golden's own cassette test sends, by `<protocol>/<cassette>`. */
const REQUESTS: Record<string, Request> = {
  "openai_responses/plain": new Request({ messages: [system("You are terse."), user("Say hello.")] }),
  "openai_responses/connection_settings": new Request({ messages: [system("You are terse."), user("Hi")] }),
  "openai_responses/subscription_plain": new Request({ messages: [system("You are terse."), user("Double 3.")] }),
  "openai_responses/subscription_no_system": new Request({ messages: [user("Double 3.")] }),
};

/** A login saved into the case's store before the config is resolved. */
interface Login {
  /** Defaults to `openai`. */
  provider?: string;
  credentials: ApiKeyCredentials | OAuthCredentials;
  model: string;
  organization?: string;
  project?: string;
}

/** What a config function may use: the case's environment and store. */
interface Setup {
  env: Env;
  store: AuthStore;
}

/** How the golden's one recorded request differs for the case. */
interface Edit {
  url?: string;
  /** Header values to set; `null` removes the header. */
  headers?: Record<string, string | null>;
  /** Top-level body fields to set. */
  json?: JsonObject;
}

/** A model that builds and answers. */
interface Built {
  /** The credential headers sent, by lower-case name. */
  credential: Record<string, string>;
  provider: string;
  model: string;
  account: string | null;
  /** The `onWarning` texts; defaults to none. */
  warnings?: string[];
}

/** A model that fails at build or on its first request, before anything is sent. */
interface Failed {
  error: new (...args: never[]) => Error;
  message: string;
  at: "build" | "request";
}

interface Case {
  name: string;
  /** The config, or a function building it once `saved` is in the store. */
  config: AgentConfig | ((setup: Setup) => AgentConfig | Promise<AgentConfig>);
  saved?: Login;
  env: Record<string, string>;
  /** `<protocol>/<cassette>`, also the key of the case's request in `REQUESTS`. */
  golden: string;
  edit?: Edit;
  /** Environment changes made after the build, before the request. */
  afterBuild?: Record<string, string>;
  expect: Built | Failed;
}

const OPENAI = { provider: "openai", model: "gpt-5.5" };
const SUBSCRIPTION: Login = {
  credentials: new OAuthCredentials({
    accessToken: ACCESS,
    refreshToken: "test-refresh-token",
    expiresAt: FAR_FUTURE,
    accountId: ACCOUNT,
    residency: "us",
  }),
  model: "gpt-5.5",
};
const SUBSCRIPTION_NO_RESIDENCY: Login = {
  ...SUBSCRIPTION,
  credentials: new OAuthCredentials({
    accessToken: ACCESS,
    refreshToken: "test-refresh-token",
    expiresAt: FAR_FUTURE,
    accountId: ACCOUNT,
  }),
};
const SAVED: Login = { credentials: new ApiKeyCredentials(SAVED_KEY), model: "gpt-5.5" };
const SAVED_SETTINGS: Login = {
  credentials: new ApiKeyCredentials(SAVED_KEY, { "X-Gateway": "gateway-1" }),
  model: "gpt-5.5",
  organization: "org-test",
  project: "proj-test",
};
const SETTINGS = { organization: "org-test", project: "proj-test", headers: { "X-Gateway": "gateway-1" } };

/** The built expectation of an OpenAI API-key route sending `key`. */
function openaiKey(key: string): Built {
  return { credential: { authorization: `Bearer ${key}` }, ...OPENAI, account: accountHash(key) };
}

const SUBSCRIBED: Built = {
  credential: { authorization: `Bearer ${ACCESS}`, "chatgpt-account-id": ACCOUNT },
  ...OPENAI,
  account: accountHash(ACCOUNT),
};

/** Resolve the default config against the saved subscription, then give it a host key. */
function hostKeyOnSubscription({ env }: Setup): AgentConfig {
  const config = resolveModelConfig({}, { env });
  expect((config["_connection"] as Connection).method).toBe("subscription");
  return { ...config, api_key: HOST_KEY };
}

const CASES: Case[] = [
  {
    name: "openai host key",
    config: { ...OPENAI, api_key: HOST_KEY },
    env: {},
    golden: "openai_responses/plain",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai saved key",
    config: {},
    saved: SAVED,
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: openaiKey(SAVED_KEY),
  },
  {
    name: "openai OPENAI_API_KEY",
    config: OPENAI,
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: openaiKey(ENV_KEY),
  },
  {
    name: "openai legacy record",
    config: legacyConnectionConfig(OPENAI),
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: openaiKey(ENV_KEY),
  },
  {
    name: "openai host key with base_url",
    config: { ...OPENAI, api_key: HOST_KEY, base_url: "https://gateway.example/v1" },
    env: { OPENAI_BASE_URL: "https://ignored.example/v1" },
    golden: "openai_responses/plain",
    edit: { url: "https://gateway.example/v1/responses" },
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai OPENAI_BASE_URL",
    config: OPENAI,
    env: { OPENAI_API_KEY: ENV_KEY, OPENAI_BASE_URL: "https://proxy.example/v1" },
    golden: "openai_responses/plain",
    edit: { url: "https://proxy.example/v1/responses" },
    expect: openaiKey(ENV_KEY),
  },
  {
    name: "openai host key with connection settings",
    config: { ...OPENAI, api_key: HOST_KEY, ...SETTINGS },
    env: {},
    golden: "openai_responses/connection_settings",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai host key with a header value holding a tab",
    config: { ...OPENAI, api_key: HOST_KEY, ...SETTINGS, headers: { "X-Gateway": "gateway\t1" } },
    env: {},
    golden: "openai_responses/connection_settings",
    edit: { headers: { "x-gateway": "gateway\t1" } },
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai saved login with saved settings",
    config: {},
    saved: SAVED_SETTINGS,
    env: {},
    golden: "openai_responses/connection_settings",
    expect: openaiKey(SAVED_KEY),
  },
  {
    name: "openai host key with a block",
    config: { ...OPENAI, api_key: HOST_KEY, openai: { reasoning: { effort: "high" }, max_output_tokens: 4096n } },
    env: {},
    golden: "openai_responses/plain",
    edit: { json: { reasoning: { effort: "high" }, max_output_tokens: 4096n } },
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "subscription with residency",
    config: {},
    saved: SUBSCRIPTION,
    env: { OPENAI_API_KEY: ENV_KEY, OPENAI_BASE_URL: "https://ignored.example/v1" },
    golden: "openai_responses/subscription_plain",
    expect: SUBSCRIBED,
  },
  {
    name: "subscription without residency",
    config: {},
    saved: SUBSCRIPTION_NO_RESIDENCY,
    env: {},
    golden: "openai_responses/subscription_plain",
    edit: { headers: { "x-openai-internal-codex-residency": null } },
    expect: SUBSCRIBED,
  },
  {
    name: "subscription with no system message",
    config: OPENAI,
    saved: SUBSCRIPTION_NO_RESIDENCY,
    env: {},
    golden: "openai_responses/subscription_no_system",
    expect: SUBSCRIBED,
  },
  {
    name: "host key on a subscription connection, OPENAI_BASE_URL set",
    config: hostKeyOnSubscription,
    saved: SUBSCRIPTION,
    env: { OPENAI_BASE_URL: "https://proxy.example/v1" },
    golden: "openai_responses/plain",
    edit: { url: "https://proxy.example/v1/responses" },
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "host key on a subscription connection, OPENAI_BASE_URL empty",
    config: hostKeyOnSubscription,
    saved: SUBSCRIPTION,
    env: { OPENAI_BASE_URL: "" },
    golden: "openai_responses/plain",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "host key on a subscription connection, OPENAI_BASE_URL unset",
    config: hostKeyOnSubscription,
    saved: SUBSCRIPTION,
    env: {},
    golden: "openai_responses/plain",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai missing key",
    config: OPENAI,
    env: {},
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message: "No OpenAI connection was selected; run 'yian auth openai' or set OPENAI_API_KEY before starting a new run",
      at: "build",
    },
  },
  {
    name: "openai environment key changed before the build",
    config: {
      ...OPENAI,
      _connection: {
        method: "api_key",
        source: "environment",
        credential_id: keyIdentity("test-old-key"),
        endpoint: OPENAI_API_ENDPOINT,
        store_path: null,
      },
    },
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message: "The OpenAI environment API key changed since this run started; start a new run",
      at: "build",
    },
  },
  {
    name: "openai host connection without its key",
    config: {
      ...OPENAI,
      _connection: {
        method: "api_key",
        source: "host",
        credential_id: null,
        endpoint: OPENAI_API_ENDPOINT,
        store_path: null,
      },
    },
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message:
        "This run's openai API key came from the host's llm_config.api_key and is not recorded; " +
        "pass api_key to resume_flow or RunResult.backward",
      at: "build",
    },
  },
  {
    name: "openai environment key changed after the build",
    config: OPENAI,
    env: { OPENAI_API_KEY: ENV_KEY },
    afterBuild: { OPENAI_API_KEY: "test-other-key" },
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message: "The OpenAI environment API key changed since this run started; start a new run",
      at: "request",
    },
  },
  {
    name: "openai saved header names differ from the recorded ones",
    config: ({ env }) => {
      const config = resolveModelConfig({}, { env });
      (config["_connection"] as Connection).header_names = ["X-Other"];
      return config;
    },
    saved: { ...SAVED_SETTINGS, credentials: new ApiKeyCredentials(SAVED_KEY, { "X-Gateway": "gateway-secret" }) },
    env: {},
    golden: "openai_responses/connection_settings",
    expect: {
      error: AuthError,
      message:
        "The saved openai login's connection headers differ from the ones this run recorded; " +
        "run 'yian auth openai' and start a new run",
      at: "build",
    },
  },
  {
    name: "openai saved login saved again after resolving",
    config: async ({ env, store }) => {
      const config = resolveModelConfig({}, { env });
      await store.save("openai", new ApiKeyCredentials("test-new-key"), "gpt-5.5");
      return config;
    },
    saved: SAVED,
    env: {},
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message: "The saved connection changed. Run 'yian auth openai' and start a new run",
      at: "build",
    },
  },
  {
    name: "subscription refuses a temperature",
    config: { openai: { temperature: 0.2 } },
    saved: SUBSCRIPTION,
    env: {},
    golden: "openai_responses/subscription_plain",
    expect: {
      error: RangeError,
      message:
        "`openai.temperature` is not accepted by the ChatGPT subscription endpoint (a subset of the Responses " +
        "API); allowed: access_programs, max_tool_calls, metadata, moderation, prompt_cache_key, " +
        "prompt_cache_options, prompt_cache_retention, reasoning, safety_identifier, service_tier, text, " +
        "top_logprobs, top_p, truncation, user",
      at: "build",
    },
  },
];

let home: string;

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-resolver-"));
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Return the golden cassette, deep-copied, with `edit` applied to its one request. */
function editedGolden(golden: string, edit: Edit = {}): Cassette {
  const cassette = structuredClone(loadCassette(join(CASSETTES, `${golden}.json`)));
  expect(cassette.interactions.length).toBe(1);
  const request = cassette.interactions[0]!.request;
  if (edit.url !== undefined) request.url = edit.url;
  const headers = (request.headers ??= {});
  for (const [name, value] of Object.entries(edit.headers ?? {})) {
    if (value === null) delete headers[name];
    else headers[name] = value;
  }
  if (edit.json !== undefined) Object.assign(request.json as JsonObject, structuredClone(edit.json));
  return cassette;
}

describe("chatModelFromConfig", () => {
  for (const each of CASES) {
    test(each.name, async () => {
      const vars = new Map(Object.entries({ ...each.env, YIAN_AUTH_FILE: join(home, "auth.json") }));
      const env: Env = (name) => vars.get(name);
      const store = new AuthStore(null, { env });
      if (each.saved) {
        const { provider = "openai", credentials, model, organization, project } = each.saved;
        await store.save(provider, credentials, model, { organization, project });
      }
      const config = typeof each.config === "function" ? await each.config({ env, store }) : each.config;
      const cassette = editedGolden(each.golden, each.edit);
      const replay = new CassetteReplay(cassette);
      const warnings: string[] = [];
      const build = () =>
        chatModelFromConfig(config, {
          env,
          send: replay.send,
          scheduler: virtualScheduler(),
          onWarning: (message) => void warnings.push(message),
        });
      const request = REQUESTS[each.golden]!;
      const expected = each.expect;

      if ("error" in expected) {
        let outcome: Promise<unknown>;
        if (expected.at === "build") {
          outcome = build();
        } else {
          // Built outside the catch, so a build failure fails the case.
          const model = await build();
          for (const [name, value] of Object.entries(each.afterBuild ?? {})) vars.set(name, value);
          outcome = collect(model.stream(request));
        }
        const error = await outcome.then(
          () => null,
          (thrown: unknown) => thrown,
        );
        expect(error).toBeInstanceOf(expected.error);
        expect((error as Error).message).toBe(expected.message);
        expect(replay.requests.length).toBe(0);
        return;
      }

      const model = await build();
      for (const [name, value] of Object.entries(each.afterBuild ?? {})) vars.set(name, value);
      await collect(model.stream(request));
      replay.assertDone();
      expect(model.provider).toBe(expected.provider);
      expect(model.model).toBe(expected.model);
      expect((model.protocol as { account?: string | null }).account ?? null).toBe(expected.account);
      expect(warnings).toEqual(expected.warnings ?? []);
      const sent = replay.requests[0]!.headers;
      for (const [name, value] of Object.entries(expected.credential)) expect(sent.get(name)).toBe(value);
      // CassetteReplay checks only the headers the cassette records; this checks nothing else was sent.
      const recorded = Object.keys(cassette.interactions[0]!.request.headers ?? {}).map((name) => name.toLowerCase());
      const names = [...sent.keys()].filter((name) => !IGNORED_HEADERS.has(name));
      expect(names.sort()).toEqual(recorded.filter((name) => !IGNORED_HEADERS.has(name)).sort());
    });
  }
});

describe("accountHash", () => {
  test("is the first 16 hex digits of the SHA-256", () => {
    expect(accountHash("abc")).toBe("ba7816bf8f01cfea");
  });
});
