/**
 * OpenAI connections: saved logins, subscriptions, refresh and the environment key. The model
 * calls build with `chatModelFromConfig` and answer from a recording `send`
 * (`helpers/run_state_fixtures.ts`'s) that replies with one Responses stream. A pinned login is
 * read at build. The cascade tests go through `resolveRunLlm` and the run tests through
 * `runFlow` with the test's environment.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import { resolveRunLlm } from "../../src/compile/llm_cascade.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { collect } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { planStructured, structuredTarget } from "../../src/llm/structured.ts";
import {
  ApiKeyCredentials,
  AuthError,
  HttpResponse,
  OAuthCredentials,
  type ProviderSelection,
} from "../../src/llm_clients/auth/common.ts";
import { type AuthRequest, OpenAIAuth } from "../../src/llm_clients/auth/openai.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { LLMConfig } from "../../src/llm_clients/config.ts";
import {
  checkModelConfig,
  type Connection,
  connectionCredentials,
  connectionSnapshot,
  legacyConnectionConfig,
  resolveModelConfig,
  splitModelConfig,
} from "../../src/llm_clients/connection.ts";
import { chatModelFromConfig } from "../../src/llm_clients/factory.ts";
import type { Send } from "../../src/platform.ts";
import { runFlow } from "../../src/run.ts";
import { recorder, responsesText, toolReply } from "../helpers/run_state_fixtures.ts";

// Each test's environment: a private store in a fresh directory and no default provider, model,
// OpenAI key or endpoint.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-openai-connection-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Return the test's store. */
function authStore(): AuthStore {
  return new AuthStore(null, { env });
}

/** Save an OpenAI subscription for `gpt-selected` and make it the default. */
function subscription(): Promise<ProviderSelection> {
  return authStore().save(
    "openai",
    new OAuthCredentials({
      accessToken: "access-secret",
      refreshToken: "refresh-secret",
      expiresAt: Date.now() / 1000 + 3600,
      accountId: "account",
    }),
    "gpt-selected",
  );
}

/** Return the error `run` throws, failing the test if it throws none. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

/** Return the error `promise` rejects with, failing the test if it resolves. */
async function rejected(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

/** A Responses stream that answers "ok". */
const TEXT_REPLY = responsesText("ok");

const HI = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });

/** Build `config` with this test's environment and `send`. */
function build(config: Record<string, unknown>, send?: Send) {
  return chatModelFromConfig(config, { env, send, onWarning: () => {} });
}

test("no builtin provider or model", () => {
  let error = thrown(() => resolveModelConfig({}, { env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/^No model is set; .*'yian auth <provider>'/);
  error = thrown(() => resolveModelConfig({ provider: "anthropic" }, { env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/^No model is set for provider 'anthropic'; .*'yian auth anthropic'/);
});

test("azure deployment environment is not a model default", () => {
  // The Azure deployment is always llm_config.model; no environment default fills it.
  vars.set("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deployment");
  expect(Object.hasOwn(connectionSnapshot({ env }).providerModels, "azure")).toBe(false);
  expect(() => resolveModelConfig({ provider: "azure" }, { env })).toThrow("No model is set for provider 'azure'");
});

test("saved selection wins incidental key and is nonsecret", async () => {
  const selected = await subscription();
  vars.set("OPENAI_API_KEY", "incidental-secret");
  const config = resolveModelConfig({}, { env });
  expect(config["model"]).toBe("gpt-selected");
  const connection = config["_connection"] as Connection;
  expect(connection.method).toBe("subscription");
  expect(connection.credential_id).toBe(selected.credentialId);
  expect(JSON.stringify(config)).not.toContain("secret");
});

test("authored model wins saved default", async () => {
  await subscription();
  expect(resolveModelConfig({ model: "explicit" }, { env })["model"]).toBe("explicit");
});

test("provider only cascade uses that providers saved model", async () => {
  await authStore().save("anthropic", new ApiKeyCredentials("key"), "saved-anthropic");
  const snap = connectionSnapshot({ env });
  const flow = await loadFlow(
    "id: test\nname: test\n" +
      "nodes:\n  a: {kind: agent, prompt: hi, llm_config: {provider: anthropic}}\noutput: ${a.output}\n",
  );
  const { effective } = resolveRunLlm(flow.compiled, {}, {
    defaults: snap.defaults,
    providerModels: snap.providerModels,
  });
  expect(effective["a"]!["model"]).toBe("saved-anthropic");
});

test("authored model still inherits across provider fields", async () => {
  const snap = connectionSnapshot({ env });
  const flow = await loadFlow(
    "id: test\nname: test\nllm_config: {model: authored}\n" +
      "nodes:\n  a: {kind: agent, prompt: hi, llm_config: {provider: anthropic}}\noutput: ${a.output}\n",
  );
  const { effective } = resolveRunLlm(flow.compiled, {}, {
    defaults: snap.defaults,
    providerModels: snap.providerModels,
  });
  expect(effective["a"]!["model"]).toBe("authored");
});

test("snapshot does not follow reconnect", async () => {
  // The pinned login is read at build, so the build fails.
  await subscription();
  const config = resolveModelConfig({}, { env });
  await authStore().save("openai", new ApiKeyCredentials("new-secret"), "other");
  const { requests, send } = recorder(TEXT_REPLY);
  const error = await rejected(build(config, send));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("changed");
  expect(requests).toEqual([]);
});

test("api environment identity checked on every request", async () => {
  vars.set("OPENAI_API_KEY", "first-secret");
  const config = resolveModelConfig({ provider: "openai", model: "gpt-5.5" }, { env });
  const { requests, send } = recorder(TEXT_REPLY);
  const model = await build(config, send);
  vars.set("OPENAI_API_KEY", "second-secret");
  const error = await rejected(collect(model.stream(HI)));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("changed");
  expect(requests).toEqual([]);
});

test("changed environment key says start a new run", async () => {
  // A changed key is reported after the failure is recorded, so restoring it can't help.
  vars.set("OPENAI_API_KEY", "first-secret");
  const connection = connectionSnapshot({ env }).connections["openai"];
  vars.set("OPENAI_API_KEY", "second-secret");
  const error = await rejected(connectionCredentials(connection, "openai", { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("changed since this run started; start a new run");
  expect(error.message).not.toContain("restore");
});

test("reconnected store credential names one fix", async () => {
  // A pinned saved credential that was replaced names one fix, without 'reconnect'.
  await authStore().save("openai", new ApiKeyCredentials("old-secret"), "gpt-5.5");
  const connection = connectionSnapshot({ env }).connections["openai"];
  await authStore().save("openai", new ApiKeyCredentials("new-secret"), "gpt-5.5");
  const error = await rejected(connectionCredentials(connection, "openai", { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("changed");
  expect(error.message.split("start a new run").length - 1).toBe(1);
  expect(error.message).not.toContain("reconnect");
});

test("missing identity does not adopt later environment key", async () => {
  const config = resolveModelConfig({ provider: "openai", model: "gpt-5.5" }, { env });
  vars.set("OPENAI_API_KEY", "new-secret");
  const error = await rejected(connectionCredentials(config["_connection"], "openai", { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("yian auth openai");
});

test("legacy openai cannot select subscription", async () => {
  await subscription();
  vars.set("OPENAI_API_KEY", "legacy-secret");
  const old = { provider: "openai", model: "old" };
  const config = legacyConnectionConfig(old);
  expect(old).toEqual({ provider: "openai", model: "old" });
  expect((config["_connection"] as Connection).method).toBe("api_key");
  expect((config["_connection"] as Connection).source).toBe("legacy_environment");
  expect(legacyConnectionConfig(config)).toEqual(config);
});

test("internal metadata is not YAML configuration", () => {
  const config = resolveModelConfig({ provider: "openai", model: "gpt-5.5" }, { env });
  expect(() => new LLMConfig(config)).toThrow(/Extra inputs/);
  (config["_connection"] as Record<string, unknown>)["token"] = "bad";
  expect(() => splitModelConfig(config)).toThrow(/connection/);
});

test("factory subscription supports structured output on responses", async () => {
  // The route half; the strategy table is `llm/structured.test.ts`'s.
  await subscription();
  const { requests, send } = recorder(toolReply("Answer", '{"value": 3}'));
  const model = await build(resolveModelConfig({}, { env }), send);
  const target = structuredTarget(model);
  expect(target.subscription).toBe(true);
  const schema = {
    title: "Answer",
    type: "object",
    properties: { value: { type: "integer" } },
    required: ["value"],
  };
  const plan = planStructured(target, schema);
  const request = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("Give an integer")] })] });
  const reply = await collect(model.stream(plan.apply(request)));
  expect(plan.parse(reply)).toEqual({ value: 3n });
  expect((requests[0]!.body["tools"] as { name: string }[])[0]!.name).toBe("Answer");
  expect(requests[0]!.headers.get("authorization")).toBe("Bearer access-secret");
});

test("saved api key ignores incidental environment", async () => {
  // The factory half; the key preflight goes with the CLI.
  await authStore().save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  vars.set("OPENAI_API_KEY", "incidental-key");
  const { requests, send } = recorder(TEXT_REPLY);
  await collect((await build({}, send)).stream(HI));
  expect(new URL(requests[0]!.url).host).toBe("api.openai.com");
  expect(requests[0]!.headers.get("authorization")).toBe("Bearer saved-key");
});

const GATEWAY = "https://gateway.example.com/v1";
const OTHER_ENDPOINT =
  "The saved OpenAI API-key login was made for another endpoint than this run's OPENAI_BASE_URL " +
  "(unset is OpenAI's API); set OPENAI_BASE_URL to the endpoint 'yian auth' lists for it, or run " +
  "'yian auth openai' again, and start a new run";

test("a run after a gateway login, without OPENAI_BASE_URL, sends nothing to api.openai.com", async () => {
  await authStore().save("openai", new ApiKeyCredentials("gateway-key"), "gpt-5.5", { endpoint: GATEWAY });
  const { requests, send } = recorder(TEXT_REPLY);

  const error = await rejected(build({}, send));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toBe(OTHER_ENDPOINT);
  const flow = await loadFlow(SLOW_THEN_AGENT.replace("{extra}", ""));
  const models = { env, createModel: (config: Record<string, unknown>) => build(config, send) };
  const result = await runFlow(flow, { topic: "x" }, { env, models });
  expect(result.status).toBe("failed");
  expect(result.failure?.message).toContain(OTHER_ENDPOINT);
  expect(requests).toEqual([]);
});

test("a gateway login runs at its endpoint when OPENAI_BASE_URL names it, a trailing slash or not", async () => {
  await authStore().save("openai", new ApiKeyCredentials("gateway-key"), "gpt-5.5", { endpoint: `${GATEWAY}/` });
  vars.set("OPENAI_BASE_URL", GATEWAY);
  const { requests, send } = recorder(TEXT_REPLY);
  await collect((await build({}, send)).stream(HI));
  expect(requests[0]!.url).toBe(`${GATEWAY}/responses`);
  expect(requests[0]!.headers.get("authorization")).toBe("Bearer gateway-key");
  // An OpenAI login doesn't run at a gateway either.
  vars.set("OPENAI_BASE_URL", "https://other.example.com/v1");
  expect((await rejected(build({}, send))).message).toBe(OTHER_ENDPOINT);
  expect(requests).toHaveLength(1);
});

test("an OpenAI API-key login saved without its endpoint is refused until it is saved again", async () => {
  // As logins were saved before they recorded their endpoint.
  await authStore().save("openai", new ApiKeyCredentials("old-key"), "gpt-5.5", { endpoint: null });
  const { requests, send } = recorder(TEXT_REPLY);
  const error = await rejected(build({}, send));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toBe(
    "The saved OpenAI API-key login doesn't record its endpoint; run 'yian auth openai' again and start a new run",
  );
  expect(requests).toEqual([]);
});

test("a pinned record whose endpoint isn't its login's is refused, as on a resume", async () => {
  const saved = await authStore().save("openai", new ApiKeyCredentials("gateway-key"), "gpt-5.5", {
    endpoint: GATEWAY,
  });
  // A run started before the login recorded its endpoint pinned OpenAI's API.
  const pinned: Connection = {
    method: "api_key",
    source: "store",
    credential_id: saved.credentialId,
    endpoint: "https://api.openai.com/v1",
    store_path: authStore().path,
  };
  const error = await rejected(connectionCredentials(pinned, "openai", { env }));
  expect(error.message).toBe(OTHER_ENDPOINT);
  const { requests, send } = recorder(TEXT_REPLY);
  const config = { provider: "openai", model: "gpt-5.5", _connection: pinned };
  expect((await rejected(build(config, send))).message).toBe(OTHER_ENDPOINT);
  expect(requests).toEqual([]);
  const credentials = await connectionCredentials({ ...pinned, endpoint: GATEWAY }, "openai", { env });
  expect((credentials as ApiKeyCredentials).apiKey).toBe("gateway-key");
});

describe("OpenAI login and runs accept and refuse the same endpoints", () => {
  const listing: AuthRequest = async () => new HttpResponse(200, { data: [{ id: "gpt-5.5" }] });
  /** Whether a login lists models at `endpoint`. */
  const loginAccepts = (endpoint: string) =>
    new OpenAIAuth({ request: listing })
      .apiModels(new ApiKeyCredentials("k"), { endpoint })
      .then(() => true, (error) => {
        if (!(error instanceof AuthError)) throw error;
        return false;
      });
  /** Whether a run takes `endpoint` from OPENAI_BASE_URL. */
  const runAccepts = (endpoint: string) => {
    vars.set("OPENAI_BASE_URL", endpoint);
    try {
      connectionSnapshot({ env });
      return true;
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      return false;
    }
  };
  test.each([
    ["https://gateway.example.com/v1", true],
    ["HTTPS://gateway.example.com/v1", true],
    ["Https://gateway.example.com:8443/v1/", true],
    ["http://localhost:8000/v1", true],
    ["HTTP://LOCALHOST:8000/v1", true],
    ["http://127.0.0.1:8000/v1", true],
    ["http://[::1]:8000/v1", true],
    ["http://gateway.example.com/v1", false],
    ["HTTP://gateway.example.com/v1", false],
    ["http://192.168.1.5:8000/v1", false],
    ["http://host.docker.internal:8000/v1", false],
    ["http://localhost.evil.example/v1", false],
    ["http://127.0.0.2/v1", false],
    ["https://user:pw@gateway.example.com/v1", false],
    ["https://gateway.example.com/v1?key=x", false],
    ["ftp://gateway.example.com/v1", false],
  ])("%s", async (endpoint, accepted) => {
    expect([await loginAccepts(endpoint), runAccepts(endpoint)]).toEqual([accepted, accepted]);
  });
});

test("subscription needs no api key", async () => {
  // The factory half; the key preflight goes with the CLI.
  await subscription();
  const { requests, send } = recorder(TEXT_REPLY);
  await collect((await build({}, send)).stream(HI));
  expect(new URL(requests[0]!.url).host).toBe("chatgpt.com");
  expect(requests[0]!.headers.get("authorization")).toBe("Bearer access-secret");
});

/** Save an OpenAI subscription for `gpt-5.5` that expires in a second and make it the default. */
function expiringSubscription(): Promise<ProviderSelection> {
  return authStore().save(
    "openai",
    new OAuthCredentials({
      accessToken: "expired-token",
      refreshToken: "refresh-token",
      expiresAt: Date.now() / 1000 + 1,
      accountId: "account",
    }),
    "gpt-5.5",
  );
}

/** Return fresh subscription tokens for the same account. */
function freshTokens(): OAuthCredentials {
  return new OAuthCredentials({
    accessToken: "fresh-token",
    refreshToken: "new-refresh",
    expiresAt: Date.now() / 1000 + 3600,
    accountId: "account",
  });
}

test("refresh retains connection identity on model requests", async () => {
  // Model requests resolve credentials this way.
  const selected = await expiringSubscription();
  const refreshed: OAuthCredentials[] = [];
  const spy = spyOn(OpenAIAuth.prototype, "refresh").mockImplementation(async (credentials) => {
    refreshed.push(credentials);
    return freshTokens();
  });
  try {
    const connection = resolveModelConfig({}, { env })["_connection"] as Connection;
    const tokens: string[] = [];
    for (let call = 0; call < 2; call++) {
      const credentials = await connectionCredentials(connection, "openai", { env });
      tokens.push((credentials as OAuthCredentials).accessToken);
    }
    expect(refreshed.length).toBe(1);
    expect(refreshed[0]!.accessToken).toBe("expired-token");
    expect(tokens).toEqual(["fresh-token", "fresh-token"]);
    expect(connection.credential_id).toBe(selected.credentialId);
    expect(authStore().selection("openai")?.credentialId).toBe(selected.credentialId);
  } finally {
    spy.mockRestore();
  }
});

test("an explicit refresh wins over OpenAI's", async () => {
  await expiringSubscription();
  const spy = spyOn(OpenAIAuth.prototype, "refresh").mockImplementation(async () => {
    throw new Error("OpenAI's refresh ran");
  });
  try {
    const connection = resolveModelConfig({}, { env })["_connection"];
    const credentials = await connectionCredentials(connection, "openai", { env, refresh: () => freshTokens() });
    expect((credentials as OAuthCredentials).accessToken).toBe("fresh-token");
    expect(spy).not.toHaveBeenCalled();
  } finally {
    spy.mockRestore();
  }
});

test("a refused refresh keeps its HTTP status on the rewrapped AuthError", async () => {
  await expiringSubscription();
  const connection = resolveModelConfig({}, { env })["_connection"];
  const refused = new AuthError("OpenAI authentication request failed (HTTP 401)", { status: 401 });
  const error = await rejected(
    connectionCredentials(connection, "openai", {
      env,
      refresh: () => {
        throw refused;
      },
    }),
  );
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toBe(
    "OpenAI authentication request failed (HTTP 401). Run 'yian auth openai' and start a new run",
  );
  expect((error as AuthError).status).toBe(401);
  expect(error.cause).toBe(refused);
});

test("endpoint metadata cannot record embedded secrets", () => {
  vars.set("OPENAI_BASE_URL", "https://api.example/v1?key=hidden-secret");
  const error = thrown(() => connectionSnapshot({ env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("without credentials");
  expect(error.message).not.toContain("hidden-secret");
});

test("unsupported default provider environment is named", () => {
  vars.set("YIAN_DEFAULT_PROVIDER", "claude");
  const error = thrown(() => connectionSnapshot({ env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/YIAN_DEFAULT_PROVIDER='claude'.*anthropic/);
});

const SLOW_THEN_AGENT = `
id: test
name: test
input:
  topic: !str
output: \${w.output}
{extra}nodes:
  first:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: |
      return inputs["topic"]
  w:
    kind: agent
    input:
      text: \${first.output}
    output: str
    prompt: Write about \${text}.
`;

/** Run `SLOW_THEN_AGENT` with `extra` before its nodes; return the rejection and the events seen. */
async function rejectedRun(extra: string): Promise<[Error, unknown[]]> {
  const events: unknown[] = [];
  const flow = await loadFlow(SLOW_THEN_AGENT.replace("{extra}", extra));
  const error = await rejected(runFlow(flow, { topic: "x" }, { env, onEvent: (event) => events.push(event) }));
  return [error, events];
}

test("run rejects unsupported default provider before any node", async () => {
  vars.set("YIAN_DEFAULT_PROVIDER", "claude");
  vars.set("YIAN_DEFAULT_MODEL", "claude-model");
  const [error, events] = await rejectedRun("");
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("YIAN_DEFAULT_PROVIDER");
  expect(events).toEqual([]);
});

describe("check model config checks the openai block against the subscription endpoint", () => {
  test.each([[{ temperature: 0.2 }], [{ max_output_tokens: 100n }]])("%#", async (block) => {
    await subscription();
    const config = resolveModelConfig({ openai: block }, { env });
    const error = thrown(() => checkModelConfig(config));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain(`\`openai.${Object.keys(block)[0]}\``);
    expect(error.message).toContain("ChatGPT subscription endpoint");
    // A host key sends the same agent to the Responses API, which accepts the block.
    checkModelConfig({ ...config, api_key: "host-key" });
  });
});

describe("check model config accepts the openai block on an api key connection", () => {
  test.each([[{ temperature: 0.2 }], [{ max_output_tokens: 100n }]])("%#", async (block) => {
    await authStore().save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
    const config = resolveModelConfig({ openai: block }, { env });
    expect((config["_connection"] as Connection).method).toBe("api_key");
    checkModelConfig(config);
  });
});

test("factory subscription sends the openai block", async () => {
  await subscription();
  const { requests, send } = recorder(TEXT_REPLY);
  const model = await build(resolveModelConfig({ openai: { reasoning: { effort: "low" } } }, { env }), send);
  await collect(model.stream(HI));
  expect((requests[0]!.body["reasoning"] as { effort: string }).effort).toBe("low");
});

test("run rejects a subscription openai block before any node", async () => {
  await subscription();
  const [error, events] = await rejectedRun("llm_config:\n  openai:\n    max_output_tokens: 100\n");
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/Agent 'w'.*`openai\.max_output_tokens`.*ChatGPT subscription/s);
  expect(events).toEqual([]);
});

test("run rejects subscription temperature before any node", async () => {
  await subscription();
  const [error, events] = await rejectedRun("llm_config:\n  openai:\n    temperature: 0.2\n");
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/Agent 'w'.*`openai\.temperature`.*ChatGPT subscription/s);
  expect(events).toEqual([]);
});
