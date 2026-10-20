/**
 * Provider setup commands: `yian auth`, `yian models`, `yian providers` and `yian config check`.
 * Not ported to TypeScript yet: each prints so on stderr and exits 1.
 */

import type { CliIo } from "./utils.ts";

/** Print that `yian <command>` isn't ported yet, on stderr, and return exit code 1. */
async function unported(command: string, io: CliIo): Promise<number> {
  await io.stderr.write(`yian ${command} is not ported to TypeScript yet\n`);
  return 1;
}

/** List, set up, choose the default among, or remove saved provider logins. Not ported yet. */
export function auth(io: CliIo): Promise<number> {
  return unported("auth", io);
}

/** List the models an openai subscription login or a provider's key offers. Not ported yet. */
export function models(io: CliIo): Promise<number> {
  return unported("models", io);
}

/** List the supported providers, their login kinds and API-key variables. Not ported yet. */
export function providers(io: CliIo): Promise<number> {
  return unported("providers", io);
}

/** Check the host llm_config layer given as a JSON object on stdin. Not ported yet. */
export function configCheck(io: CliIo): Promise<number> {
  return unported("config check", io);
}
