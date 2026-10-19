/**
 * Mirrors `tests/test_llm_config_base_url.py`: the `LLMConfig`, `host_only_field_error`,
 * `pop_host_credentials` and `check_model_config` tests, and the `splitModelConfig` half of the
 * pinned-connection test; the client and redirect tests go with the factory, the
 * `resolve_model_config` tests with `connection.ts`.
 */

import { describe, expect, test } from "bun:test";
import { inspect } from "node:util";

import {
  azureResourceEndpoint,
  HostConnection,
  hostOnlyFieldError,
  LLMConfig,
  popApiKey,
  popHostCredentials,
} from "../src/llm_clients/config.ts";
import { checkModelConfig, splitModelConfig } from "../src/llm_clients/connection.ts";
import { ValidationError } from "../src/llm_clients/provider_blocks.ts";

const SENTINEL = "sk-sentinel-7d1f3b20";
const AZURE_V1 = "https://res.openai.azure.com/openai/v1/";
const AZURE_RESOURCE = "https://res.openai.azure.com/";
const GATEWAY = "https://gateway.example.com/";

/** Return each field of `host`, read through its getters. */
function fields(host: HostConnection): Record<string, unknown> {
  const { apiKey, baseUrl, organization, project, headers } = host;
  return { apiKey, baseUrl, organization, project, headers };
}

const NO_CONNECTION = { apiKey: null, baseUrl: null, organization: null, project: null, headers: null };

/** Return the error `run` throws. */
function caught(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Return the error that building an `LLMConfig` from `data` throws. */
function thrown(data: Record<string, unknown>): ValidationError {
  try {
    new LLMConfig(data);
  } catch (error) {
    expect(error).toBeInstanceOf(ValidationError);
    return error as ValidationError;
  }
  throw new Error("expected a ValidationError");
}

test("a set base_url dumps and an unset one adds nothing", () => {
  const config = new LLMConfig({ provider: "openai", model: "gpt-5.4", base_url: AZURE_V1, api_key: SENTINEL });
  expect(config.dump()).toEqual({ provider: "openai", model: "gpt-5.4", base_url: AZURE_V1 });
  expect(JSON.stringify(config)).not.toContain(SENTINEL);
  // Recorded snapshots compare full dumps, so an unset base_url must add nothing.
  expect(Object.hasOwn(new LLMConfig({ provider: "openai", model: "gpt-5.5" }).dump(), "base_url")).toBe(false);
});

test.each([
  AZURE_V1,
  "https://gateway.example.com",
  "http://localhost:8000/v1",
  "http://127.0.0.1:11434/v1",
  "http://[::1]:8000/v1",
  "http://[::1]/v1",
  "https://[::1]:8443",
  "https://gateway.example.com:443/v1",
  "https://gateway.example.com:65535/v1",
  "https://xn--gtewy-hra.example.com/v1",
  "https://gateway.example.com/openai/v1/deployments/a-b_c~d%20e",
  GATEWAY + "a".repeat(2048 - GATEWAY.length),
])("https and local http url %# is accepted", (url) => {
  expect(new LLMConfig({ provider: "vllm", base_url: url, api_key: SENTINEL }).baseUrl).toBe(url);
});

test.each([
  "http://gateway.example.com/v1",
  "http://localhost.example.com/v1",
  "ftp://localhost/v1",
  "res.openai.azure.com/openai/v1/",
  "https://",
  "https://user:pw@res.openai.azure.com/openai/v1/",
  "https://res.openai.azure.com/openai/v1/?api-version=1",
  "https://res.openai.azure.com/openai/v1/#frag",
  "https://[bad/v1",
  "https://res.openai.azure.com/openai/v1/?",
  "https://res.openai.azure.com/openai/v1/#",
  "https://user@res.openai.azure.com/openai/v1/",
  "//res.openai.azure.com/openai/v1/",
  // Whitespace and control characters.
  " https://res.openai.azure.com/openai/v1/",
  "https://res.openai.azure.com/openai/v1/ ",
  "https://res.openai.azure.com/openai/v1/\n",
  "https://res.openai.azure.com/openai/v1/\r\n",
  "https://res.openai.\tazure.com/openai/v1/",
  "http://local\thost/v1",
  "https://res.openai azure.com/openai/v1/",
  "https://res.openai.azure.com/openai/v1/\x00",
  "https://res.openai.azure.com/openai/v1/\x7f",
  "https://res.openai.azure.com/openai/v1/\u00a0",
  // Ports.
  "https://gateway.example.com:abc/v1",
  "https://gateway.example.com:99999/v1",
  "https://gateway.example.com:65536/v1",
  "https://gateway.example.com:0/v1",
  "https://gateway.example.com:/v1",
  "https://gateway.example.com:+443/v1",
  "https://gateway.example.com:4_43/v1",
  // Hosts: ASCII only, no IDNA normalisation, exact local names.
  "https://gateway\u3002example.com/v1",
  "https://g\u00e4teway.example.com/v1",
  "https://\uff47ateway.example.com/v1",
  "https://%67ateway.example.com/v1",
  "https://gateway.example.com\\@collector.example.com/v1",
  "http://localhost.evil.com/v1",
  "http://127.0.0.1.nip.io/v1",
  "http://127.0.0.1@gateway.example.com/v1",
  "http://[0:0:0:0:0:0:0:1]:8000/v1",
  "https://[::1%25lo0]/v1",
  "https://[gateway.example.com]/v1",
  "http://[::1].example.com/v1",
  // The scheme must be lowercase, as written.
  "HTTPS://res.openai.azure.com/openai/v1/",
  "Https://res.openai.azure.com/openai/v1/",
  "HTTP://localhost:8000/v1",
  // Length.
  "https://" + "a".repeat(100_000) + ".example.com/v1",
  GATEWAY + "a".repeat(2049 - GATEWAY.length),
])("other url %# is rejected without echoing it", (url) => {
  const message = thrown({ provider: "openai", base_url: url, api_key: SENTINEL }).message;
  expect(message).toContain("base_url");
  // The shape message contains "https://", so the bare "https://" value can't be checked for an echo.
  if (url !== "https://") expect(message).not.toContain(url);
  expect(message).not.toContain(SENTINEL);
});

test.each([[{ provider: "openai" }], [{ api_key: SENTINEL }], [{}]])(
  "base_url needs a key and provider in the same config %#",
  (fields) => {
    const message = thrown({ base_url: AZURE_V1, ...fields }).message;
    expect(message).toContain("api_key and llm_config.provider");
    expect(message).not.toContain(SENTINEL);
  },
);

test("ollama rejects base_url", () => {
  expect(thrown({ provider: "ollama", base_url: AZURE_V1, api_key: SENTINEL }).message).toContain(
    "provider ollama does not take",
  );
});

test("azure takes its resource endpoint as base_url with or without a key", () => {
  const config = new LLMConfig({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  expect(config.dump()).toEqual({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  const keyed = new LLMConfig({ provider: "azure", base_url: AZURE_RESOURCE, api_key: SENTINEL });
  expect(keyed.baseUrl).toBe(AZURE_RESOURCE);
  expect(thrown({ provider: "azure", base_url: "http://res.openai.azure.com/" }).message).toContain("https");
});

test("an authored layer may set base_url only beside provider azure", () => {
  expect(hostOnlyFieldError({ provider: "azure", base_url: AZURE_RESOURCE })).toBeNull();
  for (const layer of [{ base_url: AZURE_RESOURCE }, { provider: "openai", base_url: AZURE_V1 }]) {
    expect(hostOnlyFieldError(layer)).toContain("provider: azure");
  }
  expect(hostOnlyFieldError({ provider: "azure", api_key: SENTINEL })).not.toBeNull();
});

test("popHostCredentials takes the url only with the layer's key", () => {
  let layer: Record<string, unknown> = { provider: "openai", model: "gpt-5.4", base_url: AZURE_V1, api_key: SENTINEL };
  let host = popHostCredentials(layer);
  expect([host.apiKey, host.baseUrl]).toEqual([SENTINEL, AZURE_V1]);
  expect(layer).toEqual({ provider: "openai", model: "gpt-5.4" });
  host = popHostCredentials({ provider: "openai", api_key: SENTINEL });
  expect([host.apiKey, host.baseUrl]).toEqual([SENTINEL, null]);
  expect(fields(popHostCredentials({ provider: "openai" }))).toEqual(NO_CONNECTION);

  let error = caught(() => popHostCredentials({ provider: "openai", base_url: AZURE_V1 }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key");

  // Without a host key, an Azure endpoint is an ordinary layer setting.
  layer = { provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE };
  expect(fields(popHostCredentials(layer))).toEqual(NO_CONNECTION);
  expect(layer).toEqual({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  // A host Azure key never goes to an endpoint another layer names.
  error = caught(() => popHostCredentials({ provider: "azure", model: "gpt-4o", api_key: SENTINEL }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("base_url");
  expect(error.message).not.toContain(SENTINEL);
  host = popHostCredentials({ provider: "azure", base_url: AZURE_RESOURCE, api_key: SENTINEL });
  expect([host.apiKey, host.baseUrl]).toEqual([SENTINEL, AZURE_RESOURCE]);

  const unsafe = { provider: "openai", base_url: "http://gateway.example.com", api_key: SENTINEL };
  error = caught(() => popHostCredentials(unsafe));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("https");
  expect(error.message).not.toContain("gateway.example.com");
  expect(error.message).not.toContain(SENTINEL);
});

test("azure config without base_url fails the model check", () => {
  expect(() => checkModelConfig({ provider: "azure", model: "gpt-4o" })).toThrow(/set llm_config\.base_url/);
  expect(() => checkModelConfig({ provider: "azure", model: "gpt-4o", api_key: SENTINEL })).toThrow(
    /set llm_config\.base_url/,
  );
  const [validated] = checkModelConfig({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  expect(validated.baseUrl).toBe(AZURE_RESOURCE);
});

test("a base_url must match its pinned connection", () => {
  // What resolving a host OpenAI key with this base_url records.
  const resolved = {
    provider: "openai",
    model: "gpt-5.4",
    base_url: AZURE_V1,
    api_key: SENTINEL,
    _connection: { method: "api_key", source: "host", credential_id: null, endpoint: AZURE_V1, store_path: null },
  };
  const [validated, connection] = splitModelConfig(resolved);
  expect([validated.baseUrl, connection!.endpoint]).toEqual([AZURE_V1, AZURE_V1]);
  const moved = { ...resolved, _connection: { ...resolved._connection, endpoint: "https://api.openai.com/v1" } };
  const stored = {
    ...resolved,
    _connection: {
      method: "api_key",
      source: "environment",
      credential_id: "a".repeat(64),
      endpoint: AZURE_V1,
      store_path: null,
    },
  };
  for (const config of [moved, stored]) {
    const error = caught(() => splitModelConfig(config));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain("host connection endpoint");
  }
});

describe("TS additions", () => {
  test("azureResourceEndpoint strips /openai/v1, /openai and trailing slashes", () => {
    expect(azureResourceEndpoint(AZURE_V1)).toBe("https://res.openai.azure.com");
    expect(azureResourceEndpoint("https://res.openai.azure.com/openai")).toBe("https://res.openai.azure.com");
    expect(azureResourceEndpoint("https://res.openai.azure.com//openai/v1//")).toBe("https://res.openai.azure.com");
    expect(azureResourceEndpoint(AZURE_RESOURCE)).toBe("https://res.openai.azure.com");
    // Only one suffix is stripped.
    expect(azureResourceEndpoint("https://res.example.com/openai/openai/v1")).toBe("https://res.example.com/openai");
    expect(azureResourceEndpoint("https://res.example.com/v1")).toBe("https://res.example.com/v1");
  });

  test("a HostConnection hides its key and headers and is frozen", () => {
    const host = popHostCredentials({
      provider: "openai",
      api_key: SENTINEL,
      organization: "org-1",
      headers: { "X-Team": "sentinel-header-value" },
    });
    expect(fields(host)).toEqual({
      apiKey: SENTINEL,
      baseUrl: null,
      organization: "org-1",
      project: null,
      headers: { "X-Team": "sentinel-header-value" },
    });
    for (const shown of [JSON.stringify(host), inspect(host, { showHidden: true, depth: null })]) {
      expect(shown).not.toContain(SENTINEL);
      expect(shown).not.toContain("sentinel-header-value");
    }
    expect(JSON.parse(JSON.stringify(host))).toEqual({ baseUrl: null, organization: "org-1", project: null });
    expect(Object.isFrozen(host)).toBe(true);
    expect(Object.isFrozen(host.headers)).toBe(true);
    expect(host.headers).not.toBe(host.headers);
    expect(fields(new HostConnection())).toEqual(NO_CONNECTION);
  });

  test("popHostCredentials removes the fields it takes, null ones too", () => {
    const layer: Record<string, unknown> = {
      provider: "openai",
      api_key: null,
      organization: null,
      project: null,
      headers: null,
      base_url: null,
      model: "m",
    };
    expect(fields(popHostCredentials(layer))).toEqual(NO_CONNECTION);
    expect(layer).toEqual({ provider: "openai", model: "m" });
    expect(Object.keys(layer)).toEqual(["provider", "model"]);
  });

  test("popHostCredentials refuses Azure settings without a key", () => {
    const layer: Record<string, unknown> = { provider: "azure", organization: "org-1", base_url: AZURE_RESOURCE };
    const error = caught(() => popHostCredentials(layer));
    expect(error).toBeInstanceOf(ValidationError);
    expect(error.message).toContain("apply only to provider openai");
    expect(caught(() => popHostCredentials({ provider: "azure", headers: { "X-Team": "t" } })).message).toContain(
      "llm_config.headers needs llm_config.api_key",
    );
    const badUrl = caught(() => popHostCredentials({ provider: "azure", base_url: "http://res.openai.azure.com/" }));
    expect(badUrl).toBeInstanceOf(RangeError);
    expect(badUrl).not.toBeInstanceOf(ValidationError);
    expect(badUrl.message).toContain("https");
  });

  test("hostOnlyFieldError counts a null value and follows the field order", () => {
    expect(hostOnlyFieldError({ api_key: null })).toBe(
      "llm_config.api_key is a host credential; pass it in the run's llm_config, not in a flow file",
    );
    expect(hostOnlyFieldError({ headers: {}, project: "p" })).toBe(
      "llm_config.project goes wherever the host's api_key is sent; pass it with that key in the run's " +
        "llm_config, or save it with the login by `yian auth`, not in a flow file",
    );
    expect(hostOnlyFieldError({ api_key: undefined, model: "m" })).toBeNull();
    expect(hostOnlyFieldError(["api_key"])).toBeNull();
    expect(hostOnlyFieldError(null)).toBeNull();
  });

  test("popApiKey needs a truthy provider", () => {
    for (const provider of [undefined, null, false, "", 0n, 0, [], {}]) {
      const layer: Record<string, unknown> = { provider, api_key: SENTINEL };
      const error = caught(() => popApiKey(layer));
      expect(error).toBeInstanceOf(RangeError);
      expect(error).not.toBeInstanceOf(ValidationError);
      expect(error.message).toBe(
        "llm_config.api_key needs llm_config.provider, so the key goes only to that provider",
      );
      expect(Object.hasOwn(layer, "api_key")).toBe(false);
    }
    const layer: Record<string, unknown> = { provider: "openai", api_key: SENTINEL };
    expect(popApiKey(layer)).toBe(SENTINEL);
    expect(layer).toEqual({ provider: "openai" });
    expect(popApiKey({ api_key: null })).toBeNull();
    expect(caught(() => popApiKey({ provider: "bogus", api_key: SENTINEL }))).toBeInstanceOf(ValidationError);
    expect(caught(() => popApiKey({ provider: "openai", api_key: "" })).message).toContain("must not be empty");
  });
});
