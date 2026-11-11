/**
 * Graph-stage: a `case` branch names a node or calls a flow, and a `case` has no `output:`.
 *
 * A `case` only routes. Each `then:`/`else:` is classified by its YAML tag and scalar style:
 *
 * - an unquoted string starting with `call(` calls a flow. A whole-value directive is left for
 *   `desugarCaseCallTargets`; any other text starting with `call(` is a load error naming the
 *   problem. Any other untagged unquoted string without `${` that names a node names it; one that
 *   names no node is a load error that suggests the nearest node. An unquoted date is text, so it
 *   is such a word too;
 * - anything else is a value, a load error that says to put it in an entry in `nodes:`: a tagged
 *   value (even a word naming a node, as in a value entry), quoted text (`"..."`, `'...'`, or a
 *   block `|`/`>`), `null` or an empty `else:`, a number, a bool, a mapping, a list, unquoted text
 *   holding `${`, and an `!include`d file.
 *
 * A case `output:` (`output: str`, or tagged `output: !str`) is a load error at the `output:` that
 * says to tag the result where it is read (`x: !<type> ${route.output}`). A case with both reports
 * its first value branch first.
 *
 * Runs in `assemble` before `desugarInlineCalls`, so the branches are still as authored.
 *
 * Knows about: `diagnostics`, `expr`, and the compose peers `desugar/inline_calls` (the `call(...)`
 * check), `helpers` (errors, the `call(` head, positional metadata, YAML quoting), `nodes/case`
 * (the descriptor) and `source`.
 * Never imports: build, the loader, or the runtime; only the loader pipeline imports this back.
 */

import { getCloseMatches, reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/index.ts";
import { isRecord } from "../../typesys/values.ts";
import { ArgumentValueError, desugarCallDirectives } from "../desugar/inline_calls.ts";
import { CALL_HEAD } from "../helpers/directives.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { nth } from "../helpers/source.ts";
import { CLOSE_MATCH_CUTOFF, QUOTED_STYLES } from "../helpers/tags.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { CaseDescriptor } from "../nodes/case.ts";
import type { ScalarStyle } from "../source/common.ts";
import type { SourceCursor } from "../source/index.ts";

/** Returns the authored `output:` of a case, or `null` when it has none. */
export type OutputReader = (nodeId: string) => SourceCursor | null;

/** One authored `then:`/`else:` of a case. */
interface Branch {
  /** `"then:"` or `"else:"`. */
  readonly label: string;
  /** The authored value. */
  readonly value: unknown;
  /** Its YAML scalar style; `null` (unknown) reads as plain. */
  readonly style: ScalarStyle | null;
  /** Its local YAML tag, such as `"int"`, or `null`. */
  readonly tag: string | null;
  /** Whether it was read with `!include(T) PATH`. */
  readonly included: boolean;
  /** The branch's range, else the case's range. */
  readonly range: SourceRange | null;
}

/**
 * Reject every `case` branch that is not a node or a call, and every case `output:`.
 *
 * @param descriptors The flow's parsed descriptors, keyed by node id; branches are as authored.
 * @param options.outputAt Returns a case's authored `output:` by node id, or `null` when it has
 *   none. It sees a tagged `output: !int`, which leaves the descriptor's `outputs` empty.
 * @param options.diagnostics Source ownership for located errors.
 * @throws LoadError when a branch holds a value, is an unquoted word that names no node, or is
 *   text starting with `call(` that is not one well-formed directive; or a case declares
 *   `output:`. Cases are checked in order, each one's `then:` branches, its `else:`, then its
 *   `output:`.
 *
 * Example: a case `route` with `then: "approve"` fails with ``case node 'route' then: a branch
 * names a node or calls a flow; put the value in an entry in `nodes:` and name that entry``.
 */
export function checkCaseBranches(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  options: { outputAt: OutputReader; diagnostics: LoadDiagnostics },
): void {
  const { outputAt, diagnostics } = options;
  for (const [nid, desc] of descriptors) {
    if (!(desc instanceof CaseDescriptor)) continue;
    for (const branch of branches(desc)) checkBranch(branch, { caseId: nid, descriptors, diagnostics });
    const output = outputAt(desc.id);
    if (output !== null) {
      throw diagnostics.error(
        `case node ${reprText(nid)}: a \`case\` routes and has no \`output:\`; tag its result ` +
          `where you read it, e.g. \`x: !<type> \${${nid}.output}\``,
        // The key line, which a block mapping's value range starts below.
        { range: output.keyRange ?? output.range ?? desc.range },
      );
    }
  }
}

/** Return the case's `then:` branches in order, then its `else:` when written. */
function branches(desc: CaseDescriptor): Branch[] {
  const found: Branch[] = [];
  for (const [index, spec] of desc.cases.entries()) {
    if (!isRecord(spec) || !Object.hasOwn(spec, "then")) continue;
    found.push({
      label: "then:",
      value: spec["then"],
      style: nth(desc.thenStyles, index),
      tag: nth(desc.thenTags, index),
      included: nth(desc.thenIncludes, index) !== null,
      range: nth(desc.thenRanges, index) ?? desc.range,
    });
  }
  if (desc.hasElse) {
    found.push({
      label: "else:",
      value: desc.else_,
      style: desc.elseStyle,
      tag: desc.elseTag,
      included: desc.elseInclude !== null,
      range: desc.fieldRanges.get(siteKey("field", "else")) ?? desc.range,
    });
  }
  return found;
}

/**
 * Accept a node name or a `call(...)` branch; reject anything else.
 *
 * @param branch The branch.
 * @param options.caseId The case that owns it.
 * @param options.descriptors The flow's descriptors; a bare word must name one.
 * @param options.diagnostics Source ownership for located errors.
 * @throws LoadError as `checkCaseBranches` says.
 */
function checkBranch(
  branch: Branch,
  options: { caseId: string; descriptors: ReadonlyMap<string, NodeDescriptor>; diagnostics: LoadDiagnostics },
): void {
  const { caseId, descriptors, diagnostics } = options;
  const host = `case node ${reprText(caseId)} ${branch.label}`;
  const value = branch.value;
  if (!branch.included && typeof value === "string" && !QUOTED_STYLES.has(branch.style)) {
    if (CALL_HEAD.test(value.trim())) {
      checkCall(branch, host, diagnostics);
      return;
    }
    // A tag makes the branch a value, as it makes a `nodes:` entry one.
    if (branch.tag === null && !value.includes("${")) {
      if (descriptors.has(value)) return;
      const matches = getCloseMatches(value, descriptors.keys(), 1, CLOSE_MATCH_CUTOFF);
      const suggestion = matches.length > 0 ? ` Did you mean \`${matches[0]}\`?` : "";
      throw diagnostics.error(
        `${host} \`${value}\` names no node.${suggestion} For a value, put it in an entry in \`nodes:\` ` +
          "and name that entry",
        { range: branch.range },
      );
    }
  } else if (typeof value === "string" && CALL_HEAD.test(value.trim()) && value.includes("${")) {
    throw diagnostics.error(
      `${host} a quoted \`call(...)\` is text, so it can't hold \`\${...}\` here; ` +
        "drop the quotes to call the flow, or reword the text",
      { range: branch.range },
    );
  }
  throw diagnostics.error(
    `${host} a branch names a node or calls a flow; put the value in an entry in \`nodes:\` and name ` +
      "that entry",
    { range: branch.range },
  );
}

/**
 * Reject unquoted text starting with `call(` that is not one well-formed directive.
 *
 * @param branch The branch; its value is unquoted text starting with `call(`.
 * @param host Message prefix naming the branch, such as `"case node 'r' then:"`.
 * @param diagnostics Source ownership for located errors.
 * @throws LoadError when the directive has an unreadable argument or is not exactly one call.
 */
function checkCall(branch: Branch, host: string, diagnostics: LoadDiagnostics): void {
  try {
    desugarCallDirectives(branch.value as string, () => "__check");
  } catch (error) {
    if (error instanceof ArgumentValueError) {
      // The target is a single call; only one of its arguments is wrong.
      throw diagnostics.error(`${host} ${error.message}`, { range: branch.range, cause: error });
    }
    if (!(error instanceof ExpressionError)) throw error;
    throw diagnostics.error(`${host} a branch target must be a node id or a single call: ${error.message}`, {
      range: branch.range,
      cause: error,
    });
  }
}
