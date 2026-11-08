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
 * - memory: `MemoryBase`, `isMemory`, the session's errors `MemoryBusyError`,
 *   `ChangeIdReusedError` and `MemoryMissingError`, `memoryErrorKind`, and the contract and
 *   session types (`memory/base`); `FileMemory` and `FileMemoryOptions` (`memory/file`);
 *   `VectorDBMemory` and `VectorDBMemoryOptions` (`memory/vectordb`), whose embedder a host builds
 *   with `defaultEmbed` from `yian/llm_clients`; and `MemoryRef` (`typesys/values`), the value a
 *   memory input records when a run starts.
 *
 * `RunResult` is on `yian/run` (`run.ts`), with `MemoryDrift`, a memory a crash resume found
 * changed. `Message`, `MessagePart`, `checkMessage`, `checkConversation` and `ConversationError`
 * (`nodes/agent/messages`), the built-in message value and the conversation a model route can
 * send, are also on `yian/run`. `InboxRef` (`typesys/values`), the handle an `Inbox(T)` input
 * holds, is on both, as are `MemoryRef` and `RecordedMemoryTool`.
 *
 * Knows about: compile, compose, diagnostics, expr, memory, nodes, `run`, tools and typesys.
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
export {
  type Applied,
  type Author,
  type ChangeEntry,
  ChangeIdReusedError,
  type Erased,
  isMemory,
  MemoryBase,
  MemoryBusyError,
  type MemoryContract,
  type MemoryErrorKind,
  memoryErrorKind,
  type MemoryInfo,
  MemoryMissingError,
  type MemorySession,
  type MemoryToolSpec,
  type Recall,
  type RecordedMemoryTool,
} from "./memory/base.ts";
export { FileMemory, type FileMemoryOptions } from "./memory/file.ts";
export { VectorDBMemory, type VectorDBMemoryOptions } from "./memory/vectordb.ts";
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
export { InboxRef, MemoryRef } from "./typesys/values.ts";
