/**
 * Tool registry seam for TOOL and AGENT nodes, the run's ask slot and the pending marker.
 *
 * The core ships no domain tools; a host registers its own `FunctionTool`s with `registerTool`.
 * Model-requested calls use the strict public schema. Host-owned interactions use the trusted
 * path for injected metadata. `ToolSet` holds one run's tools: the global registry, or a host
 * mapping (with its `ask` handler, which serves both `ask_user` and the TOOL built-in `ask`) that
 * replaces it. `tools/types` reads a tool's declared argument types as yian types. `tools/context`
 * holds `ToolContext`, the call's identity, the host's app object and the call's abort signal; the
 * built-in ids `take`, `sleep` and `ask`; and `pending()`, the marker a host handler returns when
 * the answer comes later. `tools/function_tools` holds `FunctionTool` (a
 * tool given as a name, description, JSON Schema and `execute`, with optional Standard Schema
 * validators and an output JSON Schema) and `tool()`, which builds one from a Standard Schema
 * validator or a raw JSON Schema. `tools/errors` holds
 * `ReportedToolError`, which a tool throws for a failure an AGENT's model reads as the call's
 * result, `PermanentToolError` and `TransientToolError`, which a tool throws to stop or steer a
 * TOOL node's retries, and `ToolCheckError`, thrown when a check yian runs on a call fails; yian
 * classifies them by brand, with `toolErrorKinds` and the guards. `tools/mcp` holds `mcpServer`,
 * which describes a Model Context Protocol server, the `McpServer` it returns, whose opened tools
 * a `ToolSet` takes as one group, and the `McpError` classes its calls throw. `tools/grants`
 * parses an AGENT's `tools:` grants, names a group's tools `<group>_<tool>` for a model, and
 * expands grants over `GrantScope`s, a `ToolSet`'s `scope()` among them.
 *
 * Knows about: `defaults`, `diagnostics`, `platform`, typesys, `llm/messages` (the JSON check),
 * `llm/auth`, `llm/body` and `llm/errors` (for MCP servers) and the skills leaf (`skills/tool`).
 */

export { ASK, isPending, pending, type PendingMarker, SLEEP, TAKE, ToolContext } from "./context.ts";
export {
  isPermanentToolError,
  isReportedToolError,
  isToolCheckError,
  isTransientToolError,
  PermanentToolError,
  ReportedToolError,
  TOOL_ERROR,
  ToolCheckError,
  toolErrorKinds,
  type ToolErrorKind,
  TransientToolError,
} from "./errors.ts";
export {
  FunctionTool,
  type StandardIssue,
  type StandardResult,
  type StandardSchemaV1,
  tool,
} from "./function_tools.ts";
export { type GrantScope } from "./grants.ts";
export {
  isMcpError,
  McpCallCancelled,
  McpConnectError,
  McpError,
  McpInvalidRequestError,
  McpOutputError,
  McpRequestError,
  McpSdkMissingError,
  McpServer,
  McpServerClosed,
  type McpServerOptions,
  McpTimeoutError,
  McpToolError,
  McpTransportError,
  mcpServer,
} from "./mcp.ts";
export {
  type AskHandler,
  type AskRequest,
  type AskTool,
  type AskUserHandler,
  type AskUserRequest,
  type CheckedCall,
  type InvokeOptions,
  invokeTool,
  invokeToolTrusted,
  registeredTools,
  registerTool,
  resolveTools,
  TOOL_REGISTRY,
  type ToolEntry,
  type ToolRef,
  ToolSet,
} from "./registry.ts";
