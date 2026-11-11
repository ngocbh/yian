/**
 * Provider credentials, private selection storage, and explicit login effects.
 *
 * This model-access leaf stages credentials before selection is committed. The store owns atomic
 * credential/default updates and refresh serialization; OpenAI services own authentication HTTP
 * and browser callback lifetimes. Neither owns a model, agent loop, conversation, CLI prompt, or
 * native Codex process. Authentication HTTP goes through the transport's `shortRequest`, which
 * guards the whole exchange with one deadline and cancellation, caps the reply, and never follows
 * redirects.
 *
 * Also exports the option types of the store (`AuthStoreOptions`) and of `OpenAIAuth`'s injected
 * sender (`AuthRequest`, `AuthRequestOptions`).
 *
 * Knows about: Node's built-in modules (`node:net` for the browser callback listener,
 * `node:crypto`, the file system), `defaults.ts`, `platform.ts`'s scheduler and file lock, the
 * transport's `shortRequest` and `llm/errors.ts`'s `ProviderError`, the `typesys` and
 * `diagnostics.ts` helpers, the connection settings checks in `connection_settings.ts`, and
 * injected HTTP, clock, and host callbacks.
 * Never imports: provider SDKs, settings, nodes, memory, compile, runtime, or CLI.
 */

export type { AuthState, Credentials } from "./common.ts";
export {
  ApiKeyCredentials,
  AuthError,
  HttpResponse,
  isAuthFailure,
  OAuthCredentials,
  ProviderSelection,
} from "./common.ts";
export { type AuthRequest, type AuthRequestOptions, OpenAIAuth } from "./openai.ts";
export { AuthStore, type AuthStoreOptions, isStorageError } from "./store.ts";
