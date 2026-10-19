/**
 * Tool registry seam for TOOL, AGENT, and host-owned human interaction.
 *
 * The core ships no domain tools; a host registers its own `FunctionTool`s with `registerTool`.
 * Model-requested calls use the strict public schema. Host-owned interactions use the trusted
 * path for injected metadata. `ToolSet` holds one run's tools: the global registry, or a host
 * mapping (with its `ask_user` handler) that replaces it. `tools/types` reads a tool's declared
 * argument types as yian types. `tools/context` holds `ToolContext`, the call's identity, the
 * host's app object and the call's abort signal. `tools/function_tools` holds `FunctionTool` (a
 * tool given as a name, description, JSON Schema and `execute`). `tools/errors` holds
 * `ReportedToolError`, which a tool throws for a failure an AGENT's model reads as the call's
 * result, `PermanentToolError` and `TransientToolError`, which a tool throws to stop or steer a
 * TOOL node's retries, and `ToolCheckError`, thrown when a check yian runs on a call fails; yian
 * classifies them by brand, with `toolErrorKinds` and the guards. `tools/mcp` holds `mcpServer`,
 * which describes a Model Context Protocol server, the `McpServer` it returns, whose opened tools
 * a `ToolSet` takes as one group, and the `McpError` classes its calls throw.
 *
 * Knows about: `defaults`, `diagnostics`, typesys, `llm/messages` (the JSON check) and the skills
 * leaf (`skills/tool`).
 * Never imports: nodes, compile, compose, runtime, run, or cli.
 */

export { ToolContext } from "./context.ts";
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
export { FunctionTool } from "./function_tools.ts";
export {
  isMcpError,
  McpCallCancelled,
  McpConnectError,
  McpError,
  McpInvalidRequestError,
  McpRequestError,
  McpServer,
  McpServerClosed,
  type McpServerOptions,
  McpTimeoutError,
  McpToolError,
  McpTransportError,
  mcpServer,
} from "./mcp.ts";
export {
  type AskUserHandler,
  type AskUserRequest,
  type InvokeOptions,
  invokeTool,
  invokeToolTrusted,
  registeredTools,
  registerTool,
  resolveTools,
  TOOL_REGISTRY,
  type ToolEntry,
  ToolSet,
} from "./registry.ts";
