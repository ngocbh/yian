/**
 * Desugar pass: `${<case>.output}` binding refs to a branch coalesce over the case's leaves.
 *
 * `expandCaseOutputs` (below) walks every binding field (any node's `input:` / mapped-call
 * `over:`), each AGENT's `llm_config:` values (an adaptive composer's too), the flow `llm_config:`
 * values and the flow `output:` bindings, and rewrites each top-level case-output span
 * `${<case>.output[.path]}`, whole or embedded in text, to a coalesce
 * `${t1.output[.path] | t2.output… }` over the case's LEAF branch targets (an inner case is
 * flattened to its own leaves).
 *
 * `caseOutputReads` lists the cases those same bindings read, before the rewrite, so the validate
 * phase can require every value-producing case to be exhaustive.
 *
 * A case-value ref nested in a coalesce or default operand, or in a `case` condition
 * (`on:`/`when:`) or an `assert`, is a located `LoadError` — those are not bindings.
 *
 * Sound via the case-route veto: the losing branch skip-floods (its target output is null), so
 * only the taken leaf is non-null in the emitted coalesce.
 *
 * Runs AFTER `desugarCaseCallTargets` (so a `then:/else: call(...)` synth id is a leaf target
 * here) and, like the other desugars, BEFORE the build loop.
 *
 * Knows about: diagnostics, `expr` (the ref walk), node contracts, the compose helpers (errors,
 * positional metadata), the kind-blind binding-field walk, the CASE descriptor and the inline-call
 * desugar (the `output:` walk — a within-phase peer).
 * Never imports: compose/parse, edges, validate, build, the loader, or the runtime; only the
 * pipeline imports this back.
 */

import { type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import {
  conditionRefs,
  ExpressionError,
  optionalInsideRoot,
  parseCondition,
  type RefStep,
  splitReference,
  stepsText,
} from "../../expr/expressions.ts";
import { callSuffixOf, type ExprNode, findSpanEnd, parseExpr, Tree } from "../../expr/grammar.ts";
import { exprRefsOf, scanTemplate, Span } from "../../expr/template.ts";
import {
  Assertion,
  Binding,
  iterSourceStrings,
  mapSourceStrings,
  type ValuePath,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import { poolRootLength } from "../../typesys/pool.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { nth } from "../helpers/source.ts";
import { AgentDescriptor } from "../nodes/agent.ts";
import { type NodeDescriptor, type OriginPresenter, siteKey } from "../nodes/base.ts";
import { mapBindingStrings } from "../nodes/bindings.ts";
import { CaseDescriptor } from "../nodes/case.ts";
import { mapOutputsStrings } from "./inline_calls.ts";

/**
 * Expand every `${<case>.output}` binding ref to a coalesce over its branch targets.
 *
 * Walks every binding field (any node's `input:` / a mapped call's `over:`), each AGENT's
 * `llm_config:` values, the flow `llm_config:` values and the flow `output:` bindings. Rejects
 * (located `LoadError`) a case-value ref in a coalesce or default operand, or in a `case` condition
 * (`on:`/`when:`) or an `assert` — those are not bindings.
 *
 * @param descriptors Node descriptors keyed by node id.
 * @param outputsSection The flow `output:` bindings.
 * @param assertsSection The flow assertions, checked for case refs.
 * @param options.outputsRange Locates a bad ref in the flow outputs.
 * @param options.assertionRangeAt Locates a bad ref in the flow assertion at an index.
 * @param options.flowLlmConfig The flow `llm_config:` layer.
 * @param options.flowLlmConfigLabel Names the flow layer in a message; defaults to `flow llm_config`.
 * @param options.resolveOrigin Locates a bad ref in an `llm_config:` value at its registered site.
 * @param options.diagnostics Source ownership for located errors.
 * @param options.spellings Collects each expanded read as written, by expanded span interior.
 * @returns `[newDescriptors, newOutputs, newFlowLlmConfig]`. A flow with no `case` node returns its
 *   descriptors (in a fresh map), its outputs and its flow layer unchanged.
 * @throws LoadError when a case-value ref sits in an unsupported position, or a read case has no
 *   resolvable branch value.
 */
export function expandCaseOutputs(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  outputsSection: unknown,
  assertsSection: readonly Assertion[],
  options: {
    outputsRange?: SourceRange | null;
    assertionRangeAt?: ((index: number) => SourceRange | null) | null;
    flowLlmConfig?: unknown;
    flowLlmConfigLabel?: string;
    resolveOrigin?: OriginPresenter | null;
    diagnostics?: LoadDiagnostics | null;
    spellings?: Map<string, CaseRead> | null;
  } = {},
): [descriptors: Map<string, NodeDescriptor>, outputsSection: unknown, flowLlmConfig: unknown] {
  const { outputsRange = null, assertionRangeAt = null, spellings = null } = options;
  const { flowLlmConfig = null, flowLlmConfigLabel = "flow llm_config", resolveOrigin = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const caseIds = caseIdsOf(descriptors);
  if (caseIds.size === 0) return [new Map(descriptors), outputsSection, flowLlmConfig];
  const targets = caseLeafTargets(descriptors, caseIds);
  /** Expand the strings of one `llm_config` layer, each named by its field and located at its value. */
  const expandLayer = (layer: unknown, label: string, fallback: SourceRange | null): unknown =>
    mapLayerStrings(layer, resolveOrigin, fallback, (text, path, range) =>
      expander(caseIds, targets, label + path.map((part) => `.${part}`).join(""), { range, diagnostics, spellings })(
        text,
      ),
    );

  const newDescriptors = new Map<string, NodeDescriptor>();
  for (const [nodeId, descriptor] of descriptors) {
    let rewritten = mapBindingStrings(descriptor, (field, key, value) =>
      expander(caseIds, targets, `node ${reprText(nodeId)}`, {
        range: bindingRange(descriptor, field, key),
        diagnostics,
        spellings,
      })(value),
    );
    if (rewritten instanceof AgentDescriptor) {
      const host = rewritten.fieldLabels.get("llm_config") ?? `node ${reprText(nodeId)} llm_config`;
      const llmConfig = expandLayer(rewritten.llmConfig, host, fieldRange(rewritten, "llm_config"));
      if (llmConfig !== rewritten.llmConfig) rewritten = rewritten.replace({ llmConfig });
    }
    newDescriptors.set(nodeId, rewritten);
  }

  const newOutputs = mapOutputsStrings(
    outputsSection,
    expander(caseIds, targets, "flow outputs", { range: outputsRange, diagnostics, spellings }),
  );
  const newFlowLlmConfig = expandLayer(flowLlmConfig, flowLlmConfigLabel, null);

  // A case value in a condition / assert (not a binding) is unsupported — loud.
  for (const [nodeId, d] of descriptors) {
    if (!(d instanceof CaseDescriptor)) continue;
    const where = `case node ${reprText(nodeId)} condition`;
    if (d.on !== null) {
      rejectCaseRefs(d.on, caseIds, where, { range: fieldRange(d, "on"), diagnostics });
      // An `on:` case's `when:` values are match values (text), not conditions.
      for (const [index, spec] of d.cases.entries()) {
        rejectCaseRefs(spec["when"], caseIds, where, { range: whenRange(d, index), diagnostics });
      }
      continue;
    }
    for (const [index, spec] of d.cases.entries()) {
      rejectCaseRefs(spec["when"], caseIds, where, { range: whenRange(d, index), diagnostics, condition: true });
    }
  }
  for (const [index, assertion] of assertsSection.entries()) {
    const range = assertionRangeAt !== null ? assertionRangeAt(index) : null;
    rejectCaseRefs(assertion, caseIds, "assert", { range, diagnostics, condition: true });
  }

  return [newDescriptors, newOutputs, newFlowLlmConfig];
}

/** A case value read as written: the case and the leaf targets its expansion reads instead. */
export interface CaseRead {
  readonly caseId: string;
  readonly leaves: readonly string[];
}

/**
 * Return `message`, about a binding whose text is `text`, with each expanded case read spelled as
 * written (`gate.output.k` for `a.output?.k | b.output?.k`); `spellings` is what
 * `expandCaseOutputs` recorded, by expanded span interior.
 */
export function caseReadsAsWritten(message: string, text: string, spellings: ReadonlyMap<string, CaseRead>): string {
  for (const [expanded, { caseId, leaves }] of spellings) {
    if (!text.includes("${" + expanded + "}")) continue;
    for (const leaf of leaves) message = message.replaceAll(`${leaf}.output?.`, `${caseId}.output.`);
  }
  return message;
}

/**
 * Return the ids of the cases read as `${<case>.output[.path]}` in a binding.
 *
 * Walks the same bindings as `expandCaseOutputs` (every node's binding fields, each AGENT's
 * `llm_config:` values, the flow `llm_config:` values and the flow `output:` bindings), so it must
 * run before that pass rewrites the refs. A read counts whole, dotted, or embedded in text; a
 * malformed span is skipped (the binding parser reports it later).
 *
 * @param descriptors Node descriptors keyed by node id.
 * @param outputsSection The flow `output:` bindings.
 * @param flowLlmConfig The flow `llm_config:` layer.
 * @returns The case ids whose `output` some binding reads.
 */
export function caseOutputReads(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  outputsSection: unknown,
  flowLlmConfig: unknown = null,
): ReadonlySet<string> {
  const caseIds = caseIdsOf(descriptors);
  const read = new Set<string>();
  if (caseIds.size === 0) return read;

  const collect = (value: string): string => {
    for (const span of topLevelSpans(value)) {
      let refs: string[];
      try {
        refs = exprRefsOf(span);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        continue;
      }
      for (const ref of refs) {
        const parts = ref.split(".");
        if (parts.length >= 2 && parts[1] === "output" && caseIds.has(parts[0]!)) read.add(parts[0]!);
      }
    }
    return value;
  };

  for (const descriptor of descriptors.values()) {
    mapBindingStrings(descriptor, (_field, _key, value) => collect(value));
    if (!(descriptor instanceof AgentDescriptor)) continue;
    for (const [, text] of iterSourceStrings(descriptor.llmConfig)) collect(text);
  }
  for (const [, text] of iterSourceStrings(flowLlmConfig)) collect(text);
  mapOutputsStrings(outputsSection, collect);
  return read;
}

/** Return the ids of the CASE descriptors. */
function caseIdsOf(descriptors: ReadonlyMap<string, NodeDescriptor>): Set<string> {
  return new Set([...descriptors].filter(([, d]) => d instanceof CaseDescriptor).map(([nodeId]) => nodeId));
}

/** Return each balanced top-level `${...}` span of `value`, `$$` escapes skipped. */
function topLevelSpans(value: string): string[] {
  const spans: string[] = [];
  let i = 0;
  while (i < value.length) {
    if (value[i] === "$" && value[i + 1] === "$") {
      i += 2;
    } else if (value[i] === "$" && value[i + 1] === "{") {
      const j = findSpanEnd(value, i + 2);
      if (j === null) break;
      spans.push(value.slice(i, j + 1));
      i = j + 1;
    } else {
      i += 1;
    }
  }
  return spans;
}

/** Return the narrowest available range for a descriptor binding. */
function bindingRange(descriptor: NodeDescriptor, field: string, key: string): SourceRange | null {
  const ranges = descriptor.fieldRanges;
  if (field === "inputs") {
    return ranges.get(siteKey("input", key)) ?? ranges.get(siteKey("field", "inputs")) ?? descriptor.range;
  }
  return ranges.get(siteKey("field", field)) ?? descriptor.range;
}

/**
 * Rebuild an `llm_config` layer with `fn` applied to each template string, given its field path
 * and the range of its value: the registered site of its key, or of the nearest key above it, else
 * `fallback`.
 */
function mapLayerStrings(
  layer: unknown,
  resolveOrigin: OriginPresenter | null,
  fallback: SourceRange | null,
  fn: (text: string, path: ValuePath, range: SourceRange | null) => string,
): unknown {
  // `mapSourceStrings` visits the strings `iterSourceStrings` yields, in the same order.
  const paths = [...iterSourceStrings(layer)].map(([path]) => path);
  let index = 0;
  return mapSourceStrings(layer, (text) => {
    const path = paths[index++] ?? [];
    return fn(text, path, layerRange(layer, path, resolveOrigin) ?? fallback);
  });
}

/** Return the resolved range of the nearest registered site at or above `path` in a layer. */
function layerRange(layer: unknown, path: ValuePath, resolveOrigin: OriginPresenter | null): SourceRange | null {
  if (resolveOrigin === null) return null;
  let origin: OriginRef | null = layer instanceof Binding ? layer.origin : null;
  if (layer instanceof ValueTemplate) {
    for (let length = path.length; length > 0 && origin === null; length -= 1) {
      origin = layer.originAt(path.slice(0, length));
    }
  }
  return origin !== null ? (resolveOrigin(origin)?.range ?? null) : null;
}

/** Return a scalar field's range, falling back to its descriptor range. */
function fieldRange(descriptor: NodeDescriptor, name: string): SourceRange | null {
  return descriptor.fieldRanges.get(siteKey("field", name)) ?? descriptor.range;
}

/** Return a case item's positional `when:` range with progressively coarser fallbacks. */
function whenRange(descriptor: CaseDescriptor, index: number): SourceRange | null {
  return nth(descriptor.whenRanges, index) ?? fieldRange(descriptor, "cases");
}

/**
 * Return a `value -> value` binding transform.
 *
 * It expands each clean case-output span to the case's branch coalesce and rejects a case ref in
 * any non-clean position. It scans each `${…}` span, so a case ref inside a larger expression (a
 * `:-` default `${x :- gate.output}`, a `:?` message, arithmetic) is caught as a non-clean ref of
 * that span.
 */
function expander(
  caseIds: ReadonlySet<string>,
  targets: (caseId: string) => readonly unknown[],
  host: string,
  options: { range: SourceRange | null; diagnostics: LoadDiagnostics; spellings: Map<string, CaseRead> | null },
): (value: string) => string {
  return (value) => {
    if (!value.includes("${")) return value;
    const out: string[] = [];
    let i = 0;
    while (i < value.length) {
      if (value[i] === "$" && value[i + 1] === "$") {
        out.push("$$"); // a literal `$` — never a span start
        i += 2;
      } else if (value[i] === "$" && value[i + 1] === "{") {
        const j = findSpanEnd(value, i + 2);
        if (j === null) {
          // An unbalanced `${`: leave the rest for the binding parser.
          out.push(value.slice(i));
          break;
        }
        out.push(expandSpan(value.slice(i + 2, j), caseIds, targets, host, options));
        i = j + 1;
      } else {
        out.push(value[i]!);
        i += 1;
      }
    }
    return out.join("");
  };
}

/**
 * Return the replacement `${…}` text for one TOP-LEVEL `${…}` span, given its interior.
 *
 * A clean case-output interior `<case>.output[.path]` becomes the branch coalesce over the case's
 * LEAF (non-case) targets (the path suffix re-applied per branch, its first step `?.` since a
 * branch the route didn't take is skipped; a case-of-case is flattened by
 * `caseLeafTargets`). Any OTHER interior that reads a case value (via a nested ref / coalesce /
 * default operand) is a located `LoadError`. Everything else (a non-case ref, a `:?` message
 * literal) is returned unchanged.
 *
 * @throws LoadError for a read case with no leaf targets, or a non-clean case read.
 */
function expandSpan(
  interior: string,
  caseIds: ReadonlySet<string>,
  targets: (caseId: string) => readonly unknown[],
  host: string,
  options: { range: SourceRange | null; diagnostics: LoadDiagnostics; spellings: Map<string, CaseRead> | null },
): string {
  const { range, diagnostics, spellings } = options;
  const read = caseRead(interior, caseIds);
  if (read !== null) {
    const { caseId, steps } = read;
    const leaves = targets(caseId);
    // No then:/else:, or a case-target cycle (the flatten dropped the arm).
    if (leaves.length === 0) {
      throw diagnostics.error(
        `${host}: \${${caseId}.output} — case ${reprText(caseId)} has no resolvable branch value ` +
          "(a case needs a `then:` or `else:`, and its targets must not form a cycle)",
        { range },
      );
    }
    // The first step after each branch root is ?.: a branch the route didn't take is skipped.
    const suffix = stepsText(steps.map((step, index) => (index === 0 ? { ...step, optional: true } : step)));
    const expanded = leaves.map((target) => `${String(target)}.output${suffix}`).join(" | ");
    spellings?.set(expanded, { caseId, leaves: leaves.map(String) });
    return "${" + expanded + "}";
  }
  rejectCaseRefs("${" + interior + "}", caseIds, host, { range, diagnostics }); // a non-clean read is loud
  return "${" + interior + "}";
}

/**
 * Return the case and the steps after `output` when a span interior is just a case value read,
 * `<case>.output[.path]` (spaces around a `.` allowed), else `null`; a malformed interior is
 * `null`, left for the binding parser.
 */
function caseRead(interior: string, caseIds: ReadonlySet<string>): { caseId: string; steps: RefStep[] } | null {
  let tree: ExprNode;
  try {
    tree = parseExpr(interior);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    return null;
  }
  if (!(tree instanceof Tree) || tree.data !== "refcall" || callSuffixOf(tree) !== undefined) return null;
  const { head, steps } = splitReference(tree);
  // `${g?.output}` puts ?. inside the root; the rejection below names that.
  if (!caseIds.has(head) || steps[0]?.name !== "output" || steps[0].optional) return null;
  return { caseId: head, steps: steps.slice(1) };
}

/**
 * Throw a located `LoadError` when `value` reads a `<case>.output` value.
 *
 * `<case>` is a case node id; this is the floor for the non-clean / condition / assert positions
 * where the value-case is not supported. With `options.condition`, a string `value` is a condition
 * (bare or one whole `${...}`); otherwise it is a binding template.
 *
 * @throws LoadError naming the first case value read.
 */
function rejectCaseRefs(
  value: unknown,
  caseIds: ReadonlySet<string>,
  where: string,
  options: { range: SourceRange | null; diagnostics: LoadDiagnostics; condition?: boolean },
): void {
  const { range, diagnostics, condition = false } = options;
  const source = value instanceof Assertion ? value.expression : value;
  for (const ref of sourceRefs(source, condition)) {
    const parts = ref.split(".");
    // Singular only: a case id in head position of `<id>.output`.
    if (parts.length >= 2 && parts[1] === "output" && caseIds.has(parts[0]!)) {
      // `${g?.output.k}` is misspelled, not a case value read in an unsupported position.
      const misplaced = optionalInsideRootOf(source, condition);
      if (misplaced !== null) throw diagnostics.error(`${where}: ${misplaced}`, { range });
      throw diagnostics.error(
        `${where}: \${${parts[0]}.output} — a case value can't be read in a coalesce operand, a ` +
          "default, an expression, a condition, or an assert; read it as a whole binding or inside text",
        { range },
      );
    }
  }
}

/**
 * Return the `?.`-inside-a-root error of the first reference in `source` that has one, else `null`.
 * With `condition`, a string `source` is a condition (bare or one whole `${...}`).
 */
function optionalInsideRootOf(source: unknown, condition: boolean): string | null {
  let texts: string[];
  if (condition) texts = typeof source === "string" ? [source] : [];
  else texts = [...iterSourceStrings(source)].map(([, text]) => text);
  for (const text of texts) {
    try {
      const trees = condition
        ? [parseCondition(text)]
        : scanTemplate(text).flatMap((segment) => (segment instanceof Span ? [segment.tree] : []));
      for (const tree of trees) {
        const misplaced = optionalInsideRoot(tree, poolRootLength);
        if (misplaced !== null) return misplaced;
      }
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
    }
  }
  return null;
}

/**
 * Return every reference path the template strings of `source` read, malformed ones skipped (the
 * relevant downstream pass reports those, located). With `condition`, a string `source` is a
 * condition (bare or one whole `${...}`).
 */
function sourceRefs(source: unknown, condition: boolean): string[] {
  if (condition) {
    if (typeof source !== "string") return [];
    try {
      return conditionRefs(source);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      return [];
    }
  }
  const refs: string[] = [];
  for (const [, text] of iterSourceStrings(source)) {
    try {
      refs.push(...exprRefsOf(text));
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
    }
  }
  return refs;
}

/**
 * Return a reader of each case id's branch targets, flattened to LEAF (non-case) node ids.
 *
 * Direct targets are each `then:` (case order) then the `else:`; a target that is itself a `case`
 * is flattened to ITS leaf targets (recursive, cycle-guarded — a case-target cycle drops that arm
 * and surfaces as an empty-leaves `LoadError` here, or a graph cycle at DAG validation).
 * Duplicates are removed preserving order. Sound via the case-route veto: the nested gate
 * skip-floods when the outer branch loses, so only the taken leaf is non-null in the coalesce.
 * A case's leaves are computed when first read.
 */
function caseLeafTargets(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  caseIds: ReadonlySet<string>,
): (caseId: string) => readonly unknown[] {
  const raw = new Map<string, unknown[]>();
  for (const [nodeId, d] of descriptors) {
    if (!(d instanceof CaseDescriptor)) continue;
    const direct: unknown[] = d.cases.map((spec) => spec["then"] ?? null).filter((target) => target !== null);
    if (d.else_ !== null) direct.push(d.else_);
    raw.set(nodeId, direct);
  }

  // Shared nested cases are reached by many paths, so each case is flattened once. A walk that
  // cut a cycle depends on its path (which cases were above it), so only a cut-free result is
  // kept; a cut-free walk saw no cycle below it, so every path gives that result.
  const memo = new Map<string, readonly unknown[]>();
  const walk = (caseId: string, path: Set<string>): [leaves: readonly unknown[], cut: boolean] => {
    const known = memo.get(caseId);
    if (known !== undefined) return [known, false];
    const out = new Set<unknown>();
    let cut = false;
    for (const target of raw.get(caseId) ?? []) {
      if (!(typeof target === "string" && caseIds.has(target))) {
        out.add(target);
      } else if (path.has(target)) {
        cut = true;
      } else {
        path.add(target);
        const [inner, innerCut] = walk(target, path);
        path.delete(target);
        for (const leaf of inner) out.add(leaf);
        cut ||= innerCut;
      }
    }
    const leaves = [...out];
    if (!cut) memo.set(caseId, leaves);
    return [leaves, cut];
  };

  return (caseId) => walk(caseId, new Set([caseId]))[0];
}
