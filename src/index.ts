/**
 * The package root: the names a host needs to load, run and extend a flow, each re-exported
 * from the module that defines it:
 *
 * - loading: `loadFlow` (`compose/loader`), `LoadError` (`compose/helpers/errors`) and
 *   `ComposedFlow` (`compose/build/pipeline`);
 * - running: `runFlow` and `advance` (`run`);
 * - `Failure` and `Diagnostic` (`diagnostics`), `CompiledFlow` (`compile/model`),
 *   `FlowValidationError` (`compile/validation`) and `VariablePool` (`typesys/pool`);
 * - `evaluateWhen` and `ExpressionError` (`expr/expressions`), and `registerTemplateFn`
 *   (`expr/builtins`);
 * - tools: `ToolContext` and `pending` (`tools/context`), the marker a host tool returns when its
 *   answer comes later; `FunctionTool` and `tool` (`tools/function_tools`); `PermanentToolError`,
 *   `TransientToolError` and `ReportedToolError` (`tools/errors`), and `ToolAttemptsFailed`
 *   (`nodes/tool/node`);
 * - the host calls on a run log, `send`, `withdraw`, `complete`, `close`, `interrupt`, `cancel` and
 *   `pendingCalls`, with their errors `RunBusyError` and `DeliveryRefused` (`run`);
 * - storage: `SQLiteStorage` and `checkStoragePath` (`memory/storage/sqlite`), `FileMarkdownStore`,
 *   `InMemoryMarkdownStore`, `MarkdownStore`, `MarkdownVersionMismatch` and `markdownVersion`
 *   (`memory/storage/markdown`), and `ProfileMemory` (`memory/profile`) for hosts that keep the
 *   profile document in their own `MarkdownStore`.
 *
 * `RunResult` is on `yian/run` (`run.ts`). `Message`, `MessagePart`, `checkMessage`,
 * `checkConversation` and `ConversationError` (`nodes/agent/messages`), the built-in message value
 * and the conversation a model route can send, are also on `yian/run`.
 * `InboxRef` (`typesys/values`), the handle an `Inbox(T)` input holds, is on both.
 *
 * Knows about: compile, compose, diagnostics, expr, memory storage, `memory/profile`, nodes,
 * `run`, tools and typesys.
 * Never imports: the CLI; nothing in the package imports this module.
 */

export { CompiledFlow } from "./compile/model.ts";
export { FlowValidationError } from "./compile/validation.ts";
export { ComposedFlow } from "./compose/build/pipeline.ts";
export { LoadError } from "./compose/helpers/errors.ts";
export { loadFlow } from "./compose/loader.ts";
export { Diagnostic, Failure } from "./diagnostics.ts";
export { registerTemplateFn } from "./expr/builtins.ts";
export { ExpressionError, evaluateWhen } from "./expr/expressions.ts";
export { ProfileMemory } from "./memory/profile.ts";
export {
  FileMarkdownStore,
  InMemoryMarkdownStore,
  type MarkdownStore,
  MarkdownVersionMismatch,
  markdownVersion,
} from "./memory/storage/markdown.ts";
export { checkStoragePath, SQLiteStorage } from "./memory/storage/sqlite.ts";
export {
  checkConversation,
  checkMessage,
  ConversationError,
  type Message,
  type MessagePart,
} from "./nodes/agent/messages.ts";
export { ToolAttemptsFailed } from "./nodes/tool/node.ts";
export {
  advance,
  cancel,
  close,
  complete,
  DeliveryRefused,
  interrupt,
  pendingCalls,
  RunBusyError,
  runFlow,
  send,
  withdraw,
} from "./run.ts";
export { pending, ToolContext } from "./tools/context.ts";
export { PermanentToolError, ReportedToolError, TransientToolError } from "./tools/errors.ts";
export { FunctionTool, tool } from "./tools/function_tools.ts";
export { VariablePool } from "./typesys/pool.ts";
export { InboxRef } from "./typesys/values.ts";
