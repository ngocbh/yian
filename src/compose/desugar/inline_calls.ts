/**
 * Graph-stage: whole-value `call(<flow-id>, kw=...)` directive desugar.
 *
 * `desugarCallDirectives` (below) is the pure string-to-data half: a field whose WHOLE trimmed
 * value is exactly `call( … )` rewrites to `${<synth>.output}` and emits an `InlineCall`. This
 * module is the assembly pass that runs it over a flow's parsed descriptors and `output:` section
 * and installs each synth `call` node (`over = null`). After it runs, the rest of `assemble`
 * (build, edge inference, validation, DAG) sees the synth nodes as ordinary `call` nodes: inline
 * calls are pure load-time sugar, no new runtime kind, and the synth callee resolves through the
 * SAME composite resolver (defs first) as a named call.
 *
 * An unquoted value whose trimmed text starts with `call(` must be exactly one well-formed
 * directive; anything else (unbalanced parens, trailing text, a missing comma, a positional or
 * repeated keyword) is a located `LoadError` that names the problem. A quoted value (YAML
 * `'...'`, `"..."`, or a block scalar) is text and never a directive, so quoting is the escape for
 * literal `call(` text. An assert is an expression, not text, so it is held to the directive rule
 * however it is quoted.
 *
 * The retired inline `${ f(args) }` form, a FLOW call embedded inside a `${...}` span, is a
 * hard-break `LoadError` pointing the author at the `call(...)` directive; a pure builtin
 * (`upper`, `join`, …) inside `${...}` stays legal.
 *
 * Binding positions and the flow `asserts:` section: every descriptor-declared binding field
 * (ordinary node `input:` mappings plus a mapped call's `over:`), the flow `output:` bindings, AND
 * each flow `asserts:` expression (a whole-value `call(...)` lifts to a synth node so the assert
 * reads its output). NOT `when:`/`on:` (the case condition grammar) and NOT a node-local
 * `asserts:` (those stay node-local, with no inline call). A `case` descriptor carries no walked
 * bindings, so it is untouched. Each leaf string of a value template is checked for the retired
 * span form, but only a whole binding lifts a `call(...)`: an unquoted leaf written as one is a
 * located `LoadError`, and a quoted one is text.
 *
 * This is also the flow-assert carrier boundary. Raw entries become one ordered `Assertion` per
 * source position; assembly injects an origin lookup by authored index. Desugaring replaces only
 * the executable expression, preserving display and any existing or injected origin for later
 * START/END lowering.
 *
 * Each synth node from a node binding records that node as its `host`; the host-edge step
 * (`compose/edges/hosts.ts`) gives it the host's CASE routes, `dependsOn`, and `runsAfter`, and
 * gates it on the host's other inputs, so it runs only when the host can. A synth node from the
 * flow `output:`/`asserts:` has no host.
 *
 * Two guards (loud): an inline call cannot target the compiler-private `__inline_flow_`
 * namespace or capture `${item}` (the synth node is a top-level node with no map-element scope;
 * use a named `call` node with `over:`). A synth id can't collide with a user node id, since
 * parsing refuses every id starting with `__`.
 *
 * Located parse metadata: each parsed host descriptor carries its own `range`/`fieldRanges` (read
 * from its source cursor). A synth `call` node locates at the SITE it desugared from, the host
 * binding's own line (looked up in the host `fieldRanges` by the binding key), not the host node
 * header, and owns an empty `fieldRanges`, so a bad ref inside the directive carets the directive
 * rather than a host input that shares its param name. The flow `output:`/`asserts:` sections
 * have no descriptor, so their inline calls use a supplied section/item range.
 *
 * Knows about: diagnostics, node contracts, `expr` (the ref walk and the span flow-call
 * detector), typesys, the compose helpers (errors, directives, YAML style and quoting), the
 * compose source (a bare argument's plain-scalar reading), the kind-blind binding-field walk and
 * the CALL descriptor.
 * Never imports: compose/parse, edges, validate, build, the loader, or the runtime; the loader,
 * the case desugars and the assert check import this back.
 */

import { type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError, unknownFunctionMessage } from "../../expr/expressions.ts";
import { exprRefsOf, flowCallCalleesInSpans } from "../../expr/template.ts";
import {
  Assertion,
  Binding,
  iterSourceStrings,
  mapSourceStrings,
  type ValuePath,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import { BUILTIN_TYPE_NAMES } from "../../typesys/types.ts";
import { isRecord, jsTypeName, type Value } from "../../typesys/values.ts";
import { CALL_HEAD, InlineCall, matchCallDirective, spanCallFlowId, splitCallsAware } from "../helpers/directives.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { QUOTED_STYLES, yamlQuoted } from "../helpers/tags.ts";
import { type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { mapBindingSources } from "../nodes/bindings.ts";
import { CallDescriptor } from "../nodes/call.ts";
import type { ScalarStyle } from "../source/common.ts";
import { readPlainScalar, SourceSyntaxError } from "../source/index.ts";
import { INLINE_FLOW_PREFIX } from "./inline_bodies.ts";

// The reserved id prefix for synthesized inline-call nodes. A monotonic counter per flow/def keeps
// them unique; the collision guard reserves the namespace from users.
export const SYNTH_PREFIX = "__call_";
// Text before a lone `=` that ends in a separate word: `1 b` in `a=1 b=2`, a missing comma.
const WORD_BEFORE_EQUALS = /\s[A-Za-z_][A-Za-z0-9_-]*\s*$/;

/** The address of the binding a `call(...)` was written in, as `CallDescriptor.callSite`. */
type CallSite = readonly (string | number | null)[];

/**
 * A `call(...)` keyword argument's bare value can't be read as a YAML scalar.
 *
 * Its message never repeats the authored text; quoting the value keeps it as text.
 */
export class ArgumentValueError extends ExpressionError {
  override name = "ArgumentValueError";

  /**
   * @param flowId The called flow's id.
   * @param name The keyword argument's name.
   * @param reason Why the value can't be read.
   */
  constructor(flowId: string, name: string, reason: string) {
    super(
      `call(${flowId}, ...) keyword ${reprText(name)} value can't be read as YAML ` +
        `(${reason}); quote it to keep it as text`,
    );
  }
}

/** A fresh `__call_0`, `__call_1`, … minter (one per flow/def `assemble` run). */
export function synthIdGen(): () => string {
  let counter = 0;
  return () => `${SYNTH_PREFIX}${counter++}`;
}

/**
 * Desugar every inline call in a flow's binding sites into synth `call` nodes.
 *
 * Returns `[descriptors, outputs, asserts]`: the descriptor map augmented with one synth
 * `CallDescriptor` per inline call (the host binding rewritten to `${<synth>.output}`), the flow
 * `output:` section, and ordered semantic `Assertion` carriers. `assertionOriginAt` supplies an
 * optional origin by authored list index exactly when raw text crosses into its carrier. Existing
 * origins take precedence. An assertion's inline calls rewrite only its executable expression;
 * display and origin stay stable. A span-wrapped inline `${ f(...) }` in an assert expression is
 * the retired form and fails as in any binding.
 *
 * Each synth node from a node host locates at the range of the binding it was found in (its own
 * `call(...)` directive line). A synth from flow `output:` uses `outputsRange`; one from
 * `asserts:` uses the authored item returned by `assertionRangeAt`.
 *
 * `bindingStyleAt(nodeId, field, key, path)` and `outputStyleAt(key, path)` (`key` is `null` for
 * a single output) return the YAML scalar style of a binding, or of the leaf at `path` inside it
 * (`[]` for the binding itself); a quoted value is text, so it is never a directive. A missing
 * lookup or a `null` style reads as plain. An assert is an expression, so its quoting
 * never makes it text. `typedOutputs` names the outputs declared with a type (`null` for a single
 * output); like a typed node input, a malformed directive there gets no quoting fix, since quoting
 * would drop the type.
 *
 * A call inside `${...}` to a name in `typeNames` (the flow's `typedefs:` names) is a `T(...)`
 * constructor, not a flow call.
 *
 * @param descriptors The flow's parsed descriptors, keyed by node id.
 * @param outputsSection The flow `output:` section.
 * @returns A fresh map (the authored descriptors in order, then the synth ones), the rewritten
 *   `output:` section, and the assertion carriers (`[]` for no `asserts:`).
 * @throws LoadError on a malformed inline call (any unquoted value starting with `call(` that is
 *   not exactly one directive), a quoted `call(...)` holding a span, a `${item}` capture, or a
 *   call to a reserved `__inline_flow_` target, located at the host node, `output:` or `asserts:`.
 * @throws TypeError when an `asserts:` item is neither text nor an `Assertion`.
 */
export function desugarInlineCalls(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  outputsSection: unknown,
  options: {
    assertsSection?: readonly unknown[] | null;
    outputsRange?: SourceRange | null;
    assertionRangeAt?: ((index: number) => SourceRange | null) | null;
    assertionOriginAt?: ((index: number) => OriginRef | null) | null;
    bindingStyleAt?: ((nodeId: string, field: string, key: string, path: ValuePath) => ScalarStyle | null) | null;
    outputStyleAt?: ((key: string | null, path: ValuePath) => ScalarStyle | null) | null;
    typedOutputs?: ReadonlySet<string | null>;
    nextId?: (() => string) | null;
    typeNames?: ReadonlySet<string>;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): [descriptors: Map<string, NodeDescriptor>, outputsSection: unknown, assertsSection: Assertion[]] {
  const {
    assertsSection = null,
    outputsRange = null,
    assertionRangeAt = null,
    assertionOriginAt = null,
    bindingStyleAt = null,
    outputStyleAt = null,
    typedOutputs = new Set<string | null>(),
    typeNames = new Set<string>(),
  } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const mint = options.nextId ?? synthIdGen();

  const synth = new Map<string, CallDescriptor>();
  const newDescriptors = new Map<string, NodeDescriptor>();
  for (const [nodeId, desc] of descriptors) {
    const [newDesc, calls, origins, sites, callSites] = desugarDescriptor(desc, mint, {
      host: nodeId,
      styleAt: bindingStyleAt === null ? null : (field, key, path) => bindingStyleAt(nodeId, field, key, path),
      typeNames,
      diagnostics,
    });
    newDescriptors.set(nodeId, newDesc);
    for (const call of calls) {
      // The synth node locates at the binding the `call(...)` came from (its own `input:` line),
      // not the host node header, so a bad ref inside the directive carets the directive, not a
      // (valid) host input that shares the param name.
      synth.set(
        call.id,
        toCallDescriptor(call, {
          host: `node ${reprText(nodeId)}`,
          site: sites.get(call.id)!,
          callSite: callSites.get(call.id)!,
          hostId: nodeId,
          range: origins.get(call.id)!,
          diagnostics,
        }),
      );
    }
  }

  const [newOutputs, outCalls, outSites, outCallSites] = desugarOutputs(outputsSection, mint, {
    range: outputsRange,
    styleAt: outputStyleAt,
    typed: typedOutputs,
    typeNames,
    diagnostics,
  });
  for (const call of outCalls) {
    synth.set(
      call.id,
      toCallDescriptor(call, {
        host: "flow outputs",
        site: outSites.get(call.id)!,
        callSite: outCallSites.get(call.id)!,
        range: outputsRange,
        diagnostics,
      }),
    );
  }

  const [newAsserts, assertCalls, assertRanges, assertSites, assertCallSites] = desugarAsserts(
    assertsSection,
    mint,
    { assertionRangeAt, assertionOriginAt, typeNames, diagnostics },
  );
  for (const call of assertCalls) {
    synth.set(
      call.id,
      toCallDescriptor(call, {
        host: "assert",
        site: assertSites.get(call.id)!,
        callSite: assertCallSites.get(call.id)!,
        range: assertRanges.get(call.id) ?? null,
        diagnostics,
      }),
    );
  }

  for (const [callId, desc] of synth) newDescriptors.set(callId, desc);
  return [newDescriptors, newOutputs, newAsserts];
}

// ---------- the flow `output:` binding walk ----------
//
// Used by this pass AND the case-value expansion in `compose/desugar/case_outputs.ts`. Descriptor
// binding fields go through the kind-blind `compose/nodes/bindings.ts` `mapBindingSources`.

/** Whether `value` is a binding source the walk maps: a string or a `ValueTemplate`. */
function isSource(value: unknown): value is string | ValueTemplate {
  return typeof value === "string" || value instanceof ValueTemplate;
}

/**
 * Apply `fn` to each template-bearing binding in the flow `output:` section.
 *
 * The section is a name-to-binding record, a bare binding, a tagged single output's `Binding`
 * carrier, or `null`. A source is a string or a `ValueTemplate`; any other value is left as is,
 * and a carrier keeps its contract around the new source.
 */
export function mapOutputsSources(outputs: unknown, fn: (source: string | ValueTemplate) => unknown): unknown {
  return mapOutputSources(outputs, (_key, source) => fn(source));
}

/**
 * Apply `fn(key, source)` to each template-bearing binding in the flow `output:` section, as
 * `mapOutputsSources` does; `key` is the record field name, or `null` for a single output.
 */
export function mapOutputSources(
  outputs: unknown,
  fn: (key: string | null, source: string | ValueTemplate) => unknown,
): unknown {
  if (isRecord(outputs)) {
    return Object.fromEntries(
      Object.entries(outputs).map(([key, value]) => [key, isSource(value) ? fn(key, value) : value]),
    );
  }
  if (isSource(outputs)) return fn(null, outputs);
  if (outputs instanceof Binding && isSource(outputs.source)) {
    return new Binding(fn(null, outputs.source), { origin: outputs.origin, contract: outputs.contract });
  }
  return outputs;
}

/**
 * Apply `fn` to each template string in the flow `output:` section.
 *
 * Walks the same sources as `mapOutputsSources`; a `ValueTemplate` supplies each of its leaf
 * strings.
 */
export function mapOutputsStrings(outputs: unknown, fn: (text: string) => unknown): unknown {
  return mapOutputsSources(outputs, (source) => mapSourceStrings(source, fn));
}

/**
 * The source range a synth node desugared from binding `key` should locate at.
 *
 * Prefers the binding's OWN entry, `siteKey("input", key)` for the `inputs` mapping and
 * `siteKey("field", field)` for a scalar binding field (a map's `over:`), i.e. the line the
 * `call(...)` directive was written on. Falls back to the host node's range when there is no
 * stamped entry (a spanless flow, or a binding shape with no per-entry range).
 */
export function bindingOrigin(desc: NodeDescriptor, field: string, key: string): SourceRange | null {
  const ranges = desc.fieldRanges;
  if (field === "inputs") {
    return ranges.get(siteKey("input", key)) ?? ranges.get(siteKey("field", "inputs")) ?? desc.range;
  }
  return ranges.get(siteKey("field", field)) ?? desc.range;
}

/**
 * Desugar a descriptor's binding fields.
 *
 * Walks the binding fields via `mapBindingSources`; each inline call is tagged in `origins` with
 * the SOURCE RANGE of the binding it was found in (via `bindingOrigin`), so its synth node
 * locates at its own `call(...)` directive rather than the host header. It is also tagged in
 * `sites` with that binding's author-facing phrase (`node 'x' input 'a'`, `value entry 'v'`,
 * `node 'm' map over`) and in `callSites` with its address (`["input", host, key]`, or
 * `["field", host, field]` for a scalar binding field). Nested inner-first calls share their
 * enclosing binding's tags. `styleAt(field, key, path)` gives the YAML scalar style of each
 * binding, or of the leaf at `path` inside it (plain when absent).
 *
 * @returns `[newDesc, calls, origins, sites, callSites]`.
 */
function desugarDescriptor(
  desc: NodeDescriptor,
  mint: () => string,
  options: {
    host: string;
    styleAt: ((field: string, key: string, path: ValuePath) => ScalarStyle | null) | null;
    typeNames: ReadonlySet<string>;
    diagnostics: LoadDiagnostics;
  },
): [
  NodeDescriptor,
  InlineCall[],
  Map<string, SourceRange | null>,
  Map<string, string>,
  Map<string, CallSite>,
] {
  const { host, styleAt, typeNames, diagnostics } = options;
  const calls: InlineCall[] = [];
  const origins = new Map<string, SourceRange | null>();
  const sites = new Map<string, string>();
  const callSites = new Map<string, CallSite>();

  const mapper = (field: string, key: string, source: string | ValueTemplate): unknown => {
    const origin = bindingOrigin(desc, field, key);
    let site: string;
    let callSite: CallSite;
    if (field === "inputs") {
      // A value entry names its one input `value entry '<id>'`.
      site =
        desc.fieldLabels.get(`input ${reprText(key)}`) || `node ${reprText(host)} input ${reprText(key)}`;
      callSite = ["input", host, key];
    } else {
      site = `node ${reprText(host)} ${desc.kind} ${field}`;
      callSite = ["field", host, field];
    }
    const before = calls.length;
    const newSource = desugarSource(source, mint, calls, host, {
      range: origin,
      styleAt: styleAt === null ? null : (path) => styleAt(field, key, path),
      // Quoting a typed binding would drop its type, so no quoting fix is offered.
      quotable: !(field === "inputs" && desc.inputTypes.has(key)),
      typeNames,
      diagnostics,
    });
    for (const call of calls.slice(before)) {
      origins.set(call.id, origin);
      sites.set(call.id, site);
      callSites.set(call.id, callSite);
    }
    return newSource;
  };

  return [mapBindingSources(desc, mapper), calls, origins, sites, callSites];
}

/**
 * Desugar one binding source: a string as in `desugarValue`, or each leaf string of a
 * `ValueTemplate`. Only a whole binding lifts a `call(...)` directive, so an unquoted leaf string
 * starting with `call(` is a located `LoadError`; a quoted one is text. `styleAt(path)` gives the
 * YAML scalar style at `path` (`[]` for the whole binding).
 */
function desugarSource(
  source: string | ValueTemplate,
  mint: () => string,
  calls: InlineCall[],
  host: string,
  options: {
    range?: SourceRange | null;
    styleAt?: ((path: ValuePath) => ScalarStyle | null) | null;
    quotable?: boolean;
    typeNames: ReadonlySet<string>;
    diagnostics: LoadDiagnostics;
  },
): unknown {
  const { range = null, styleAt = null, quotable = true, typeNames, diagnostics } = options;
  if (!(source instanceof ValueTemplate)) {
    const style = styleAt === null ? null : styleAt([]);
    return desugarValue(source, mint, calls, host, { range, style, quotable, typeNames, diagnostics });
  }
  for (const [path, text] of iterSourceStrings(source)) {
    if (!CALL_HEAD.test(text.trim())) continue;
    if (styleAt === null || !QUOTED_STYLES.has(styleAt(path))) {
      throw diagnostics.error(`${host}: a \`call(...)\` inside a value isn't supported; bind it at the top level`, {
        range,
      });
    }
    rejectQuotedCallSpan(text, host, { range, diagnostics });
  }
  // The leaves starting with `call(` passed above as quoted text.
  return mapSourceStrings(source, (text) =>
    CALL_HEAD.test(text.trim()) ? text : desugarValue(text, mint, calls, host, { range, typeNames, diagnostics }),
  );
}

/**
 * Desugar one binding string. A WHOLE-VALUE `call(<flow-id>, kw=...)` directive lifts to a synth
 * call node (accumulated in `calls`, the value rewritten to `${<synth>.output}`). A flow call
 * embedded INSIDE a `${...}` span is the retired `${flow(args)}` form: a located `LoadError`
 * pointing at the `call(...)` directive. A builtin type name called like a function
 * (`${int(x)}`) is a located `LoadError` explaining that constructors are `typedefs:` names. Any
 * other value (plain refs, pure builtins, constructors named in `typeNames`, literals) is
 * returned unchanged.
 *
 * `style` is the value's YAML scalar style. A quoted value is text: it is never a directive, and
 * a quoted `call(...)` holding a span is a load error (its author most likely meant the call).
 * `expression` is true for an assert, whose quoting does not make it text. A malformed
 * directive's error suggests quoting only when that gives the text: not for an assert, a
 * span-bearing value, a `quotable: false` (typed) binding, or an unreadable argument
 * (`ArgumentValueError` says to quote that argument). Maps an `ExpressionError` (a malformed
 * directive or an unparseable span) to a located `LoadError` at `range`.
 */
function desugarValue(
  value: string,
  mint: () => string,
  calls: InlineCall[],
  host: string,
  options: {
    range?: SourceRange | null;
    style?: ScalarStyle | null;
    quotable?: boolean;
    expression?: boolean;
    typeNames: ReadonlySet<string>;
    diagnostics: LoadDiagnostics;
  },
): string {
  const { range = null, style = null, quotable = true, expression = false, typeNames, diagnostics } = options;
  const quoted = !expression && QUOTED_STYLES.has(style);
  const isCall = CALL_HEAD.test(value.trim());
  if (quoted && isCall) rejectQuotedCallSpan(value, host, { range, diagnostics });
  if (isCall && !quoted) {
    let newValue: string;
    let found: InlineCall[];
    try {
      [newValue, found] = desugarCallDirectives(value, mint);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      const hint =
        quotable && !expression && !(error instanceof ArgumentValueError) && !value.includes("${")
          ? `. If you meant the text, quote the whole value: \`${yamlQuoted(value.trim())}\``
          : "";
      throw diagnostics.error(`${host}: ${error.message}${hint}`, { range, cause: error });
    }
    // A keyword argument is a binding too, so it gets the same retired-form check (nested
    // calls are in `found`, each with its own arguments). A malformed span is left for the
    // binding parser, which names the argument.
    for (const call of found) {
      for (const [name, arg] of Object.entries(call.args)) {
        if (typeof arg === "string" && arg.includes("${")) {
          rejectSpanFlowCalls(arg, `${host}: call(${call.callee}) argument ${reprText(name)}`, {
            range,
            typeNames,
            diagnostics,
            skipMalformed: true,
          });
        }
      }
    }
    calls.push(...found);
    return newValue;
  }
  rejectSpanFlowCalls(value, host, { range, typeNames, diagnostics });
  return value;
}

/**
 * Throw a located `LoadError` when quoted text starting with `call(` holds a `${...}` span: its
 * author most likely meant the call.
 */
function rejectQuotedCallSpan(
  value: string,
  host: string,
  options: { range: SourceRange | null; diagnostics: LoadDiagnostics },
): void {
  if (!value.includes("${")) return;
  throw options.diagnostics.error(
    `${host}: a quoted \`call(...)\` is text, so it can't hold \`\${...}\` here; ` +
      "drop the quotes to call the flow, or reword the text",
    { range: options.range },
  );
}

/**
 * Throw a located `LoadError` (prefixed `where:`) when a `${...}` span of `value` calls a flow,
 * the retired `${flow(args)}` form, or calls a builtin type name like a function; also when a
 * span doesn't parse, unless `skipMalformed`. A call to a name in `typeNames` is a constructor
 * and passes.
 */
function rejectSpanFlowCalls(
  value: string,
  where: string,
  options: {
    range: SourceRange | null;
    typeNames: ReadonlySet<string>;
    diagnostics: LoadDiagnostics;
    skipMalformed?: boolean;
  },
): void {
  const { range, typeNames, diagnostics, skipMalformed = false } = options;
  let flowCallees: string[];
  try {
    flowCallees = flowCallCalleesInSpans(value, typeNames);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    if (skipMalformed) return;
    throw diagnostics.error(`${where}: ${error.message}`, { range, cause: error });
  }
  if (flowCallees.length === 0) return;
  const callee = flowCallees[0]!;
  if (BUILTIN_TYPE_NAMES.has(callee)) {
    throw diagnostics.error(`${where}: ${unknownFunctionMessage(callee)}`, { range });
  }
  if (callee === "call") {
    throw diagnostics.error(
      `${where}: call(...) can't go inside \${...}. Write it unquoted as the whole value, ` +
        `call(${spanCallFlowId(value)}, ...), or call the flow in its own node and read its output.`,
      { range },
    );
  }
  throw diagnostics.error(
    `${where}: a flow call ${reprText(callee)} inside \${...} is not supported — write it ` +
      `as the whole value with the call(...) directive: call(${callee}, ...)`,
    { range },
  );
}

/**
 * Desugar a WHOLE-VALUE `call(<flow-id>, kw=...)` directive, else no-op.
 *
 * A value whose trimmed text starts with `call(` must be exactly one directive: a leading `call`,
 * balanced parens, and nothing non-whitespace after the close paren. Its FIRST arg is the
 * POSITIONAL flow id (a bare identifier, may be hyphenated: flow ids allow `-`); every remaining
 * arg is a `name=value` keyword, each at most once. Each keyword value passes through
 * `argSource`: a bare value reads like a named `input:` value (`window=30` gives `30n`, `1e3`
 * gives `1000`), a quoted value loses only its outer quotes (no escape processing), and a
 * `${…}`-bearing value stays a binding string. An unquoted arg value starting with `call(` is
 * ITSELF a directive, under the same rule, desugared inner-first.
 *
 * @param value The binding text.
 * @param mint Mints each synth node id.
 * @returns `[newValue, calls]`: if `value` is a directive, `newValue` is the synth
 *   `${<id>.output}` ref and `calls` lists the `InlineCall`s (each nested call before its
 *   enclosing one, inner-first). If `value` does not start with `call(`, returns `[value, []]`
 *   unchanged (the caller then checks the value for a retired span flow call).
 * @throws ExpressionError naming the problem when a value starting with `call(` is not one
 *   directive: unbalanced parentheses, text after the close paren, a first arg that is not a bare
 *   flow id, a positional arg after the flow id (the message mentions "keyword"), an unquoted `=`
 *   inside a keyword value (a missing comma), or a keyword passed more than once.
 * @throws ArgumentValueError when a bare keyword value the flow's YAML rules can't build (an
 *   impossible date).
 */
export function desugarCallDirectives(value: string, mint: () => string): [string, InlineCall[]] {
  const match = matchCallDirective(value);
  if (match === null) return [value, []];
  const [flowId, argsStr] = match;
  const calls: InlineCall[] = [];
  const args = new Map<string, Value>();
  for (const pair of splitCallsAware(argsStr, ",")) {
    if (!pair.trim()) continue; // a trailing comma / an empty kwarg slot
    const parts = splitCallsAware(pair, "=");
    if (parts.length < 2) {
      throw new ExpressionError(
        `call(${flowId}, ...) passes a positional argument ${reprText(pair.trim())}; ` +
          "every argument after the flow id must be a keyword (name=value)",
      );
    }
    const name = parts[0]!.trim();
    if (!name || /\s/.test(name)) {
      throw new ExpressionError(
        `call(${flowId}, ...) argument ${reprText(pair.trim())} needs one name before ` +
          `\`=\`; ${name ? "add the missing comma" : "write the keyword name"}`,
      );
    }
    const argValue = parts.slice(1).join("=");
    if (hasInnerKeyword(parts.slice(1))) {
      throw new ExpressionError(
        `call(${flowId}, ...) keyword ${reprText(name)} value ${reprText(argValue.trim())} looks ` +
          "like two keywords; add the missing comma, or quote the value",
      );
    }
    const [newValue, inner] = desugarCallDirectives(argValue, mint); // inner-first recursion
    calls.push(...inner);
    if (args.has(name)) {
      throw new ExpressionError(`call(${flowId}, ...) passes keyword ${reprText(name)} more than once`);
    }
    try {
      args.set(name, argSource(newValue));
    } catch (error) {
      if (!(error instanceof SourceSyntaxError)) throw error;
      throw new ArgumentValueError(flowId, name, error.message);
    }
  }
  const callId = mint();
  calls.push(new InlineCall(callId, flowId, Object.fromEntries(args)));
  return ["${" + callId + ".output}", calls];
}

/**
 * Read one `call(...)` keyword value: a bare value the way the named form reads its `input:`, a
 * quoted value as the text between its outer quotes, kept verbatim (no YAML escape processing,
 * so `'it''s'` stays `it''s`).
 *
 * @param value The keyword's raw value text.
 * @returns For a value that starts and ends with the same quote (`"…"`/`'…'`), the text between
 *   them, a binding template if it interpolates (`"hi ${name}"` gives `hi ${name}`); for an
 *   unquoted `${…}`-bearing value, the binding string; else the trimmed text read as one plain scalar like a named `input:` value
 *   (`30` gives `30n`, `1e3` a number, `~` or an empty value `null`, `None` text). YAML
 *   structure such as `[1]` stays text: a bare argument is one scalar.
 * @throws SourceSyntaxError when the bare text resolves to a type that can't build it, such as an
 *   impossible date; the message does not echo the text.
 */
function argSource(value: string): Value {
  const stripped = value.trim();
  if (stripped.length >= 2 && (stripped[0] === "'" || stripped[0] === '"') && stripped.at(-1) === stripped[0]) {
    return stripped.slice(1, -1); // a quoted scalar -> its inner text (template or literal)
  }
  if (stripped.includes("${")) return stripped;
  return readPlainScalar(stripped);
}

/**
 * Whether a keyword value, split on its top-level `=`, holds a would-be keyword `... name=...`.
 *
 * Only a lone `=` after a separate word counts (`1 b=2`); `==`, `!=`, `<=`, `>=`, and a glued
 * `a=b` stay part of the value's text.
 *
 * @param parts The value split on each top-level `=`.
 */
function hasInnerKeyword(parts: readonly string[]): boolean {
  for (let index = 0; index + 1 < parts.length; index++) {
    const left = parts[index]!;
    const right = parts[index + 1]!;
    if (!left || !right || "!<>".includes(left.at(-1)!)) continue;
    if (WORD_BEFORE_EQUALS.test(left)) return true;
  }
  return false;
}

/**
 * Desugar the flow `output:` section.
 *
 * `range` is the `output:` section source range; `styleAt(key, path)` gives the YAML scalar
 * style of an output, or of the leaf at `path` inside it (`key` is `null` for a single output;
 * plain when absent), and `typed` names the outputs declared with a type.
 *
 * @returns `[newOutputs, inlineCalls, sites, callSites]`: `sites` maps each call's synth id to
 *   its output's phrase (`flow output`, `flow output 'r'`) and `callSites` to its address
 *   `["output", key]` (`key` is `null` for a single output).
 */
function desugarOutputs(
  outputs: unknown,
  mint: () => string,
  options: {
    range: SourceRange | null;
    styleAt: ((key: string | null, path: ValuePath) => ScalarStyle | null) | null;
    typed: ReadonlySet<string | null>;
    typeNames: ReadonlySet<string>;
    diagnostics: LoadDiagnostics;
  },
): [unknown, InlineCall[], Map<string, string>, Map<string, CallSite>] {
  const { range, styleAt, typed, typeNames, diagnostics } = options;
  const calls: InlineCall[] = [];
  const sites = new Map<string, string>();
  const callSites = new Map<string, CallSite>();

  const desugar = (key: string | null, source: string | ValueTemplate): unknown => {
    const before = calls.length;
    const newSource = desugarSource(source, mint, calls, "flow outputs", {
      range,
      styleAt: styleAt === null ? null : (path) => styleAt(key, path),
      quotable: !typed.has(key),
      typeNames,
      diagnostics,
    });
    for (const call of calls.slice(before)) {
      sites.set(call.id, key === null ? "flow output" : `flow output ${reprText(key)}`);
      callSites.set(call.id, ["output", key]);
    }
    return newSource;
  };

  return [mapOutputSources(outputs, desugar), calls, sites, callSites];
}

/**
 * Cross raw flow asserts into semantic carriers and desugar inline calls.
 *
 * Parsed YAML supplies strings. An existing carrier is accepted for source-neutral programmatic
 * composition and retains its display and origin. `assertionOriginAt` supplies an optional
 * source-neutral origin by authored list index; an existing origin wins. `assertionRangeAt`
 * supplies a source range by authored list index, keeping duplicate assertion text distinct.
 * Rewriting changes only the executable expression, so generated node ids never replace authored
 * presentation. `null` or empty gives `[]`.
 *
 * @returns `[newAsserts, inlineCalls, callRanges, sites, callSites]`: each map takes a call's
 *   synth id to its assertion's source range, its phrase (`flow assert '<display>'`), and its
 *   address `["assert", index]`.
 * @throws TypeError when an item is neither text nor an `Assertion`.
 */
function desugarAsserts(
  asserts: readonly unknown[] | null,
  mint: () => string,
  options: {
    assertionRangeAt: ((index: number) => SourceRange | null) | null;
    assertionOriginAt: ((index: number) => OriginRef | null) | null;
    typeNames: ReadonlySet<string>;
    diagnostics: LoadDiagnostics;
  },
): [Assertion[], InlineCall[], Map<string, SourceRange | null>, Map<string, string>, Map<string, CallSite>] {
  const { assertionRangeAt, assertionOriginAt, typeNames, diagnostics } = options;
  const calls: InlineCall[] = [];
  const callRanges = new Map<string, SourceRange | null>();
  const sites = new Map<string, string>();
  const callSites = new Map<string, CallSite>();
  const newAsserts: Assertion[] = [];
  if (asserts === null || asserts.length === 0) return [newAsserts, calls, callRanges, sites, callSites];
  asserts.forEach((value, index) => {
    const assertionRange = assertionRangeAt === null ? null : assertionRangeAt(index);
    let assertion: Assertion;
    if (value instanceof Assertion) {
      assertion = value;
      if (assertion.origin === null && assertionOriginAt !== null) {
        assertion = new Assertion(assertion.expression, assertion.display, { origin: assertionOriginAt(index) });
      }
    } else if (typeof value === "string") {
      assertion = new Assertion(value, value, {
        origin: assertionOriginAt === null ? null : assertionOriginAt(index),
      });
    } else {
      throw new TypeError(`flow assertions must be strings or Assertion carriers, got ${jsTypeName(value)}`);
    }
    const before = calls.length;
    const expression = desugarValue(assertion.expression, mint, calls, "assert", {
      range: assertionRange,
      expression: true,
      typeNames,
      diagnostics,
    });
    for (const call of calls.slice(before)) {
      callRanges.set(call.id, assertionRange);
      sites.set(call.id, `flow assert ${reprText(assertion.display)}`);
      callSites.set(call.id, ["assert", index]);
    }
    newAsserts.push(new Assertion(expression, assertion.display, { origin: assertion.origin }));
  });
  return [newAsserts, calls, callRanges, sites, callSites];
}

/**
 * One `InlineCall` to a synth `CallDescriptor` (`over = null`), after its guards.
 *
 * Authored calls into the compiler-private inline-flow namespace are rejected at this shared
 * boundary, before any synth descriptor can reach child resolution. The synth node carries its
 * OWN provenance: `range` is the site it desugared from (a node input's binding line, a case
 * `then:`/`else:` line, or a section item). It owns no authored fields, so `fieldRanges` is empty
 * and a bad ref inside the directive carets `range`, never a host line.
 *
 * @param call The recognized directive: synth id, callee, and keyword args.
 * @param options.host The site named in error messages (`node 'x'`, `flow outputs`, ...).
 * @param options.site The authored binding the directive was written in (`node 'x' input 'a'`,
 *   `flow output`, ...). When given, the synth node's `fieldLabels` name the call
 *   `<site> call(<callee>)` and each argument `<site> call(<callee>) argument '<name>'`, so its
 *   errors read as the authored text, not the synth id.
 * @param options.callSite The address of that binding, recorded as the synth node's `callSite`.
 * @param options.hostId The node the call was lifted out of, recorded as the synth node's `host`
 *   so it takes that node's CASE routes, `dependsOn`, and `runsAfter`. `null` for a call with no
 *   host node (flow `output:`/`asserts:`, or a branch target).
 * @param options.range The site the directive desugared from.
 * @param options.diagnostics Locates load errors; absent uses a range-only default.
 * @returns The synth call node.
 * @throws LoadError when the callee is in the private inline-flow namespace or an arg captures
 *   `${item}`.
 */
export function toCallDescriptor(
  call: InlineCall,
  options: {
    host: string;
    site?: string | null;
    callSite?: CallSite | null;
    hostId?: string | null;
    range?: SourceRange | null;
    diagnostics?: LoadDiagnostics | null;
  },
): CallDescriptor {
  const { host, site = null, callSite = null, hostId = null, range = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  if (call.callee.startsWith(INLINE_FLOW_PREFIX)) {
    throw diagnostics.error(
      `${host}: call targets beginning with ${reprText(INLINE_FLOW_PREFIX)} are reserved ` +
        "for compiler-generated inline child flows",
      { range },
    );
  }
  for (const [name, value] of Object.entries(call.args)) {
    if (capturesItem(value)) {
      throw diagnostics.error(
        `${host}: inline call ${reprText(call.callee)} cannot capture \${item} ` +
          `(arg ${reprText(name)}) — it desugars to a top-level call node with no ` +
          "map-element scope; use a named `call` node (whose per-element " +
          "`input:` may read `${item}`)",
        { range },
      );
    }
  }
  const fieldLabels = new Map<string, string>();
  if (site !== null) {
    fieldLabels.set("call", `${site} call(${call.callee})`);
    for (const name of Object.keys(call.args)) {
      fieldLabels.set(`input ${reprText(name)}`, `${site} call(${call.callee}) argument ${reprText(name)}`);
    }
  }
  return new CallDescriptor({
    id: call.id,
    call: call.callee,
    inputs: { ...call.args },
    over: null,
    host: hostId,
    range,
    fieldRanges: new Map(),
    fieldLabels,
    callSite,
  });
}

/** True if a binding-string arg value reads `${item}` (a ref with head `item`). */
function capturesItem(value: unknown): boolean {
  if (typeof value !== "string") return false;
  let refs: string[];
  try {
    refs = exprRefsOf(value);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    return false; // malformed -> surfaced (located) by the downstream ref-wiring pass
  }
  return refs.some((ref) => ref.split(".")[0] === "item");
}
