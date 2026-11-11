/**
 * LLM clients and `LLMConfig` for AGENT nodes: yian's model seam.
 *
 * This is the public `@ngocbh/yian/llm_clients` entry, for hosts that set up providers; the README
 * lists its supported names, and its other exports are internal.
 *
 * The model-access leaf: `LLMConfig` (the selection carrier) and `chatModelFromConfig` (resolve it
 * to a ready `RouteModel` on its provider's route), and `embedderFromConfig` (an embedding model's
 * `/embeddings` client, with the same credentials). `provider_blocks.ts` holds the schema of each
 * provider block, the request body of the API that provider's calls reach, checked at load and
 * again when a model is built, then sent as written by the wire protocols in `llm/protocols/`.
 * `auth/` owns staged login, private storage of one saved login per provider, and refresh.
 * `connection.ts` snapshots the environment and saved defaults (there is no built-in provider or
 * model), records nonsecret connection identities, resolves pinned credentials for requests, and
 * lists them so errors can be scrubbed (`callCredentials`, `scrubCallCredentials`).
 * `connection_settings.ts` checks the settings that travel with a key (an OpenAI organization or
 * project id and extra request headers). `layers.ts` resolves an agent's `llm_config` layers into
 * its effective configuration. `listModels` lists the model ids a provider's host offers, using
 * only the key, base URL and connection settings passed in.
 *
 * Knows about: `llm/` (routes, auth objects, wire protocols, errors), `defaults.ts`, `platform.ts`
 * and the `typesys` and `diagnostics.ts` helpers.
 * Never imports: nodes, compose, compile, runtime, memory or the CLI. It is a leaf, so those may
 * import it, never the reverse.
 */

export type { AuthRequest, AuthRequestOptions, AuthState, AuthStoreOptions, Credentials } from "./auth/index.ts";
export {
  ApiKeyCredentials,
  AuthError,
  AuthStore,
  HttpResponse,
  isStorageError,
  OAuthCredentials,
  OpenAIAuth,
  ProviderSelection,
} from "./auth/index.ts";
export {
  azureResourceEndpoint,
  LLM_CONFIG_KEYS,
  LLMConfig,
  type LLMProvider,
  SUPPORTED_LLM_PROVIDERS,
} from "./config.ts";
export { callCredentials, scrubCallCredentials } from "./connection.ts";
export type { Embed, Embedder, EmbeddingModel } from "../llm/embeddings.ts";
export { chatModelFromConfig, defaultEmbed, defaultModelAccess, embedderFromConfig, type ModelAccess } from "./factory.ts";
export { type ListModelsOptions, listModels, ModelListError } from "./model_list.ts";
export { type Issue, ValidationError } from "./provider_blocks.ts";
