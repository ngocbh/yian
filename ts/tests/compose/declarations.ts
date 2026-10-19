/**
 * Read authored `input:` or `variables:` declarations straight from YAML text: a port of
 * `tests/compose/_declarations.py`.
 *
 * TS-only: `caseDocument`, which reads a conformance case's flow as a parsed document.
 *
 * Not yet ported: `parameters_from_yaml`, which comes with the declaration readers.
 */

import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";

import { SourceDocument } from "../../src/compose/source/index.ts";
import { reload } from "../../src/defaults.ts";
import { type Case, prepare, readCase } from "../conformance.ts";

const CASES = fileURLToPath(new URL("../../../conformance/cases/", import.meta.url));

/**
 * Read conformance case `id` (`group/name`) as the TS runner runs it: the prepared case and its
 * flow parsed as `flow.yaml` (another file of the case with `file`).
 *
 * The case's `env:` is applied with `reload` before the flow parses, since parsing and typedef
 * reading use the configured limits; this changes the process's defaults, and the caller
 * restores them with `forget()`, e.g. in `afterEach`.
 *
 * @throws Error when the case has no such flow or file text.
 */
export function caseDocument(
  id: string,
  options: { file?: string } = {},
): { case_: Case; document: SourceDocument } {
  const { file } = options;
  const case_ = prepare(readCase(readFileSync(`${CASES}${id}.yaml`, "utf8")));
  const env = (case_["env"] ?? {}) as Record<string, string>;
  reload((name) => (Object.hasOwn(env, name) ? env[name] : undefined));
  const files = case_["files"];
  const text =
    file === undefined
      ? case_["flow"]
      : typeof files === "object" && files !== null && !Array.isArray(files)
        ? files[file]
        : undefined;
  if (typeof text !== "string") {
    throw new Error(`case ${id} has no ${file === undefined ? "flow" : `file ${file}`}`);
  }
  const document = SourceDocument.parse(text, file === undefined ? "flow.yaml" : basename(file));
  return { case_, document };
}
