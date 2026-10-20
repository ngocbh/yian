/**
 * Reject a binding, case branch or flow output written out deeper than a value may nest.
 *
 * Runs on the authored values, before any pass that walks them recursively (the tagged-binding
 * read, the case-branch check, the desugars), so a value built deep through YAML aliases is a
 * located load error rather than a stack overflow. Each binding is checked as the run-time binding
 * check counts it. MAP's `over:` is checked when the node is parsed, so it is skipped here. A case
 * branch that holds a value is always a load error that suggests a value entry; one past the limit
 * reports the limit instead, since that entry couldn't load either. Flow outputs follow the tagged
 * split: an untagged mapping is a record whose fields sit one level in, and a tagged section or
 * any other value is the single output.
 *
 * Knows about: compose nodes (descriptors, `BINDING_FIELDS`, `siteKey`), value entries, the
 * written-value check, and load diagnostics.
 * Never imports: build, the loader, or the runtime.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { isRecord } from "../../typesys/values.ts";
import { isValueEntry } from "../desugar/value_entries.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { CaseDescriptor } from "../nodes/case.ts";
import { BINDING_FIELDS } from "../nodes/index.ts";

/**
 * Reject a binding, case branch or flow output whose written-out value nests too deep.
 *
 * @param descriptors Authored node descriptors keyed by node id.
 * @param outputsSection The authored `output:` section, tags not yet split.
 * @param options.outputsTagged Whether the whole section carries a tag, which makes it the single
 *   output even when it is a mapping.
 * @param options.outputRange Location of a record output, by name.
 * @param options.outputsRange Location of the whole `output:` section.
 * @param options.diagnostics Builds the located load errors.
 * @throws LoadError at the offending binding, branch or output for the first value past the
 *   limit, nodes in authored order and then outputs.
 */
export function checkWrittenValues(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  outputsSection: unknown,
  options: {
    outputsTagged: boolean;
    outputRange: (name: unknown) => SourceRange | null;
    outputsRange: SourceRange | null;
    diagnostics: LoadDiagnostics;
  },
): void {
  const { outputsTagged, outputRange, outputsRange, diagnostics } = options;
  for (const [nodeId, desc] of descriptors) {
    if (isValueEntry(desc)) continue;
    if (desc instanceof CaseDescriptor) {
      for (const [slot, value, site] of caseValueBranches(desc)) {
        checkWrittenValue(value, `node ${reprText(nodeId)} ${slot}`, site ?? desc.range, diagnostics);
      }
      continue;
    }
    if (!BINDING_FIELDS.get(desc.kind)?.includes("inputs") || !isRecord(desc.inputs)) continue;
    for (const [key, value] of Object.entries(desc.inputs)) {
      const range =
        desc.fieldRanges.get(siteKey("input", key)) ?? desc.fieldRanges.get(siteKey("field", "inputs")) ?? desc.range;
      checkWrittenValue(value, `node ${reprText(nodeId)} input ${reprText(key)}`, range, diagnostics);
    }
  }
  if (outputsTagged || !isRecord(outputsSection)) {
    checkWrittenValue(outputsSection, "flow output", outputsRange, diagnostics);
    return;
  }
  for (const [key, value] of Object.entries(outputsSection)) {
    checkWrittenValue(value, `flow output ${reprText(key)}`, outputRange(key), diagnostics, {
      enclosingLevels: 1,
    });
  }
}

type CaseBranch = [string, unknown, SourceRange | null];

/**
 * List a case's authored `then:` and `else:` branches with their slots and ranges: each branch's
 * slot as messages name it (`"then[0] value"`, `"else value"`), its authored value, and its range,
 * `then:` branches in order and then `else:`.
 */
function caseValueBranches(desc: CaseDescriptor): CaseBranch[] {
  const branches: CaseBranch[] = [];
  desc.cases.forEach((case_, index) => {
    if (isRecord(case_) && Object.hasOwn(case_, "then")) {
      branches.push([`then[${index}] value`, case_["then"], desc.thenRanges[index] ?? null]);
    }
  });
  if (desc.hasElse) {
    branches.push(["else value", desc.else_, desc.fieldRanges.get(siteKey("field", "else")) ?? null]);
  }
  return branches;
}
