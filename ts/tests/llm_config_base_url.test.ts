/**
 * Mirrors `tests/test_llm_config_base_url.py`: the `LLMConfig` tests. The
 * `host_only_field_error` and `pop_host_credentials` tests come with the next change to
 * `config.ts`; `check_model_config` goes with `connection.ts`; the client and redirect tests go
 * with the factory.
 */

import { expect, test } from "bun:test";

import { LLMConfig } from "../src/llm_clients/config.ts";
import { ValidationError } from "../src/llm_clients/provider_blocks.ts";

const SENTINEL = "sk-sentinel-7d1f3b20";
const AZURE_V1 = "https://res.openai.azure.com/openai/v1/";
const AZURE_RESOURCE = "https://res.openai.azure.com/";
const GATEWAY = "https://gateway.example.com/";

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
