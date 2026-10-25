/**
 * Graph stage: reference-wiring validation over the built flow.
 *
 * A cross-node assembly pass, the analogue of `compile/validation`'s reference collector, walking
 * the BUILT flow: it calls the leaf checker (`classifyPath`) directly,
 * supplying the valid-target and flow-input sets and the producer contracts, and only
 * reimplements the per-node walk glue. Every located problem accumulates; one `LoadError` carries
 * them all. Builtin calls are checked at every binding and prompt site.
 *
 * **Prompt scope**: an AGENT `prompt:` or `system_prompt:` may interpolate this node's inputs and
 * variables as bare `${name}`; a `${<id>.output}`/`${input.X}`/`${system.X}` in either is a located
 * error. Each `${name.f}` read is field-checked
 * against the name's proven contracts (its binding's, under the node's narrowing, or a typed
 * variable's), and is an error only when all of them reject it.
 *
 * **Case node-local refs**: the desugared CASE `when:` strings use `__rN`/`__on`, which resolve
 * against the bound input record; they are excluded here. Only the `__rN`/`__on` SOURCES are
 * classified, against the pool (mirrors the strict CASE check).
 *
 * Located parse metadata: each error anchors on the referencing field's `SourceRange`, read from
 * the transient Compose `BuildResult` (the per-binding `input` entry, else the node's `input:`
 * mapping, else the node header, for an input-binding ref; its `prompt:` or `system_prompt:` for a
 * template ref). A field or input a desugar authored under another node is named by its
 * `BuildResult.fieldLabels` entry: an adaptive ask's brief is reported as the ask's
 * `adaptive_questions.prompt` and anchors on the brief, and an ask input its composer copies is
 * reported as the ask's input. A plain input binding whose site name, source, and range were
 * already scanned is skipped, so such a copied binding is reported once.
 *
 * Knows about: `compile/validation` (the leaf namespace checker), `expr` (the binding and template
 * parsers), `nodes` (node behaviour, the AGENT and CASE nodes, and the binding
 * carriers), `typesys` (`Type`), `compose/helpers` (closed-path
 * projection, binding-contract resolution, narrowed reads, and errors), and `compose/validate`
 * (`constructors` for the constructor and builtin checks, `narrowing` for each node's view, and
 * `typed_bindings` for the node-local contracts and a MAP's item contract).
 * Never imports: the loader or the compose package facade; they import this back.
 */

import { classifyPath } from "../../compile/validation.ts";
import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import {
  checkTemplateCalls,
  conditionRefs,
  ExpressionError,
  exprRefsOf,
  promptRefs,
  scanTemplate,
} from "../../expr/index.ts";
import { Span } from "../../expr/template.ts";
import type { Node } from "../../nodes/base.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../nodes/agent/node.ts";
import { CallNode } from "../../nodes/call/node.ts";
import { CaseNode } from "../../nodes/case/node.ts";
import {
  Binding,
  bindingSource,
  isExactly,
  iterSourceStrings,
  type OutputContract,
  type Parameter,
  outputContractType,
  PartialRecordContract,
  ValueContract,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import { ToolNode } from "../../nodes/tool/node.ts";
import { TAKE } from "../../tools/context.ts";
import { messagesKey, NEW_MESSAGES } from "../../typesys/pool.ts";
import { builtinType, type TypeRegistry } from "../../typesys/types.ts";
import {
  canonicalTypeLabel,
  isInboxType,
  isRecord,
  jsTypeName,
  Type,
  typeEquals,
  ValueKind,
  valueRepr,
} from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { constructorTypeFn } from "../helpers/isinstance_types.ts";
import { type Env, envReadType, narrowedReadError } from "../helpers/narrowing_facts.ts";
import {
  directReferencePathParts,
  localFieldError,
  outputContractPathError,
  projectOutputContract,
  referenceContract,
} from "../helpers/output_contracts.ts";
import { SYNTH_PREFIX } from "../desugar/inline_calls.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import { builtinFieldErrors, constructorErrors } from "./constructors.ts";
import type { Narrowing } from "./narrowing.ts";
import { itemContract, LocalInput, type ResolveOrigin } from "./typed_bindings.ts";

/**
 * Return whether a value of `typ` can be a prompt message (the built-in `Message`, or a record with
 * non-null `role` and `text: str`) or, with `list`, a list of them. `object` and `Any` can; a union
 * can when every member can.
 */
function holdsMessages(typ: Type, list: boolean): boolean {
  if (typ.variants !== undefined) return typ.variants.every((variant) => holdsMessages(variant, list));
  if (typ.kind === ValueKind.ANY) return true;
  if (typ.kind === ValueKind.OBJECT) {
    // Compared without `nullable`, as a nullable `{role, text}` record loads: the render refuses a spliced null.
    if (typeEquals(typ.replace({ nullable: false }), builtinType("Message")!)) return true;
    if (typ.fields === undefined) return true;
    const role = typ.fields.get("role");
    const text = typ.fields.get("text");
    // A nullable `role` or `text` could splice a null, which the render refuses.
    return role?.kind === ValueKind.STRING && !role.nullable && text?.kind === ValueKind.STRING && !text.nullable;
  }
  if (list && (typ.kind === ValueKind.LIST_OBJECT || typ.kind === ValueKind.LIST_ANY)) {
    return typ.element === undefined || holdsMessages(typ.element, false);
  }
  return false;
}

/**
 * Return the type label of a spliced prompt local whose declared type can't hold messages, or
 * `null` when it can or its type isn't known.
 */
function spliceTypeError(local: LocalInput | undefined): string | null {
  if (local === undefined || local.alternatives.length === 0) return null;
  const types = local.alternatives.map((contract) => outputContractType(contract));
  if (types.some((typ) => typ === null || holdsMessages(typ, true))) return null;
  return canonicalTypeLabel(types[0]!);
}

/**
 * Heads forbidden in a strict prompt (always trigger the bespoke hint). `input`/`system`/`item` are
 * the pool-style heads; a node-id head (e.g. `${other.output}`) is detected separately by
 * `validTargets.has(head)`.
 */
const PROMPT_FORBIDDEN_HEADS: ReadonlySet<string> = new Set(["input", "system", "item"]);

/** The per-site options of one `scan`. */
interface ScanOptions {
  /** The producer contracts this site sees (a node's narrowed view, or the unnarrowed map). */
  siteProducers: ReadonlyMap<string, OutputContract>;
  /** The flow-input contracts this site sees. */
  siteInputs: ReadonlyMap<string, ValueContract>;
  /** The element contract of a MAP per-element input's `${item}`, when known. */
  itemContract?: ValueContract | null;
  /** The body-local scopes treated as lenient heads, e.g. `item` in a MAP per-element input. */
  extraHeads?: readonly string[];
  /** Lets a field only some union members have be read. */
  partialUnionFields?: boolean;
  /**
   * Checks the site's `${T(...)}` calls; a reserved key, where they are not supported, leaves that
   * error to the type-table pass.
   */
  constructors?: boolean;
  /** The facts a searched CASE's guarded param is read under. */
  readFacts?: Env | null;
  /** The bases whose narrowing a read owes to the routing. */
  routedBases?: ReadonlySet<string>;
  /**
   * Why this site refuses a whole inbox read, the error's closing words; `null` where one is
   * allowed. Defaults to telling the author to take from the inbox.
   */
  inboxTail?: string | null;
  /** Lets an AGENT's `${x.new_messages}` be read; a CASE's routing reads only outputs. */
  messages?: boolean;
}

/** Why most sites refuse a whole inbox read: only `take` reads its events. */
const TAKE_TAIL = "(take from it with `tool_id: take`)";
/** Why a MAP's sources refuse an inbox read. */
const MAP_TAIL = "(a map body can't read an inbox)";
/** Why text, an expression, or a case's `when:` or `on:` refuses an inbox read. */
const EXPRESSION_TAIL = "(not inside text or an expression)";

/**
 * Return why node `node`'s input `param` can't take an inbox whole, or `null` when it can: a
 * take's `from`, or an input a named call's or a loop's child declares as an Inbox, so a loop never
 * carries an inbox it can't drop.
 */
function inboxSiteTail(nodeId: string, node: Node, param: string): string | null {
  if (node instanceof ToolNode && node.builtin === TAKE) return param === "from" ? null : TAKE_TAIL;
  if (node instanceof CaseNode) return EXPRESSION_TAIL;
  if (!(node instanceof CallNode || node.isLoop)) return TAKE_TAIL;
  if (nodeId.startsWith(SYNTH_PREFIX)) return "(pass it through a named `kind: call` node)";
  const childInputs = (node as { childInputs?: readonly Parameter[] }).childInputs ?? [];
  if (childInputs.some((input) => input.name === param && isInboxType(input.contract?.type))) return null;
  return `(the child's \`${param}\` isn't an Inbox input)`;
}

/**
 * Return the flow `output:` section as a flat list of binding values (the sites to check).
 *
 * A whole-string binding (`${a.output | b.output}`) or a tagged single output's `Binding` carrier
 * is one value; a name to binding map is each mapped value. Mirrors the pipeline's own reading
 * (kept local: this pass doesn't import the build).
 */
function outputBindings(outputs: unknown): unknown[] {
  if (isRecord(outputs)) return Object.values(outputs);
  if (outputs === null) return [];
  return [outputs];
}

/**
 * Validate every `${...}` reference site of a built flow.
 *
 * `buildResults` is the transient Compose build map: its nodes include leaf runtime `Node`s and
 * desugared `CaseNode`s (their wiring carries the `__rN`/`__on` sources), while its range fields
 * retain YAML locations only for this validation phase. `flowInputs` is the flow's input names
 * (the `${input.X}` set); `inputContracts` maps those names to their exact contracts, so
 * `${input.X.f}` is field-checked too (`null` or absent checks names only). `producers` maps node
 * id to its exact or partial output contract; an absent contract stays lenient. The namespace
 * checker validates the reference head, then the shared output-contract helper checks exact fields
 * or the partial arm's closed root name set without inventing a type for opaque fields. A MAP's
 * per-element `${item.f}` is field-checked against the element type of `over:` when that is a
 * proven list. `outputs` is the raw flow `output:` section.
 *
 * Each error anchors on the referencing site's `SourceRange`, resolved from its
 * `BuildResult.fieldRanges`: the OFFENDING ENTRY where one exists (`input` `<name>`, stamped per
 * binding by the parse loop for every kind), else the enclosing block (the `inputs` block, or the
 * `prompt:` field), else the node header. An AGENT prompt site and a plain input site are named by
 * `BuildResult.siteName`: the `fieldLabels` entry when a desugar authored the field or binding
 * under another node (an adaptive ask's composer brief reads `node '<ask>'
 * adaptive_questions.prompt` and anchors on the brief; its copied input reads
 * `node '<ask>' input '<p>'`), else `node '<id>' <site>`. A plain input binding whose site name,
 * source text, and range match one already scanned is skipped, so a binding a desugar copied onto
 * a synth node is reported once. A string nested in a value template is located at its own
 * authored site instead, resolved through `resolveOrigin`. The aggregate carries the first
 * retained range.
 *
 * With `narrowing`, each node's references (a CASE's own sources included) are checked against
 * that node's view of the producer and flow-input contracts, so a branch can read the fields of
 * the union members that reach it. A node a topological order could not reach skips the
 * member-only field rule, so the later cycle check reports the cycle. The flow outputs are always
 * checked unnarrowed.
 *
 * With `localContracts` (`localInputContracts`, keyed by node id then param), each prompt read
 * `${name.f}` of an AGENT input is field-checked against that name's alternatives,
 * and an AGENT read of a typed variable against its declared type; a name with no entry is not
 * field-checked.
 *
 * With `registry`, every `${T(...)}` constructor call in a binding is checked too
 * (`constructorErrors`), with each reference typed by the same contracts.
 *
 * Every builtin call in a binding or prompt must pass arguments the builtin takes.
 *
 * An inbox input (`Inbox(T)` in `inputContracts`) is read only whole, as the lone
 * `${input.<name>}` bound to an input a named call's or a loop's child declares as an Inbox, or
 * the lone `${input.<name>}` of an AGENT's `steer_inbox:`. Any
 * other read of it (a field, text or an expression, a typed binding or value, another node kind,
 * a MAP, an inline `call(...)`, a flow output) is an error at its site. An inbox input with no
 * whole read is an error too, located at its declaration, reported only when no other one is.
 *
 * @param buildResults Built nodes with their transient ranges, keyed by node id.
 * @param flowInputs The flow's input names.
 * @param producers Exact or partial node output contracts keyed by producer id.
 * @param outputs The raw flow `output:` section.
 * @param flowWiring Final binding maps keyed by node id and wiring key.
 * @param options.inputContracts Exact flow input contracts keyed by input name.
 * @param options.outputsRange The authored output value or block range, when retained.
 * @param options.narrowing The union narrowing each node reads against.
 * @param options.localContracts The proven contracts of each node-local name, by node id then
 *   param, for the text sites.
 * @param options.registry The flow's typedefs, naming a narrowed type and resolving constructors.
 * @param options.resolveOrigin Resolves a registered origin to its source frame.
 * @param options.diagnostics Factory for located load errors.
 * @throws TypeError when a producer contract is not a `ValueContract` or `PartialRecordContract`.
 * @throws LoadError listing every unresolved reference, else every constructor call that can never
 *   succeed, else every inbox input nothing reads.
 */
export function validateReferences(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowInputs: ReadonlySet<string>,
  producers: ReadonlyMap<string, OutputContract>,
  outputs: unknown,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  options: {
    inputContracts?: ReadonlyMap<string, ValueContract> | null;
    outputsRange?: SourceRange | null;
    narrowing?: Narrowing | null;
    localContracts?: ReadonlyMap<string, ReadonlyMap<string, LocalInput>> | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: ResolveOrigin | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const { outputsRange = null, narrowing = null, registry = null, resolveOrigin = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const inputContracts = options.inputContracts ?? new Map<string, ValueContract>();
  // An AGENT's new messages are read under their own pool name, `<id>.new_messages`.
  const nodeTargets = new Set(buildResults.keys());
  const validTargets = new Set([
    ...nodeTargets,
    ...[...buildResults].filter(([, result]) => result.node.producesMessages).map(([nodeId]) => messagesKey(nodeId)),
  ]);
  // Keep exact and partial carriers intact; the Compose projection below checks closed record
  // paths without inventing a Type for opaque fields.
  for (const [nodeId, contract] of producers) {
    if (!isExactly(contract, ValueContract) && !isExactly(contract, PartialRecordContract)) {
      throw new TypeError(
        "producer contracts must be ValueContract or PartialRecordContract values, " +
          `got ${jsTypeName(contract)} for ${reprText(nodeId)}`,
      );
    }
  }
  const errors: [message: string, range: SourceRange | null][] = [];
  const constructorProblems: [message: string, range: SourceRange | null][] = [];
  // The `[site name, source, range]` of each plain input binding already scanned, as JSON.
  const scanned = new Set<string>();
  const unordered = narrowing !== null ? narrowing.unordered : new Set<string>();
  const localContracts = options.localContracts ?? new Map<string, ReadonlyMap<string, LocalInput>>();
  const typeFn = registry !== null ? constructorTypeFn(registry) : null;
  // The inbox inputs, and those read whole at a site that may take them.
  const inboxInputs = new Set([...inputContracts].filter(([, contract]) => isInboxType(contract.type)).map(([n]) => n));
  const inboxesRead = new Set<string>();

  /**
   * Name-check every reference a BINDING `from:` reads (coalesce and nested-default split by
   * `exprRefsOf`); a non-string source or literal yields no refs. Each string nested in a
   * `ValueTemplate` is scanned on its own and located at its authored site when that resolves.
   */
  function scan(fromValue: unknown, where: string, rng: SourceRange | null, site: ScanOptions): void {
    const source = bindingSource(fromValue);
    for (const [valuePath, text] of iterSourceStrings(fromValue)) {
      let textRng = rng;
      if (source instanceof ValueTemplate && resolveOrigin !== null) {
        const origin = source.originAt(valuePath);
        const frame = origin !== null ? resolveOrigin(origin) : null;
        if (frame !== null) textRng = frame.range;
      }
      scanText(text, where, textRng, site);
    }
  }

  /**
   * Name-check every reference one template string reads; see `scan`. A read of a field its
   * narrowed type lacks, under `readFacts`, names that type, credited to the routing when its
   * base is in `routedBases`. A call to a builtin with arguments it doesn't take, and each field
   * read on a builtin's result its type can't have, is one more error at `where`.
   */
  function scanText(text: string, where: string, rng: SourceRange | null, site: ScanOptions): void {
    const {
      siteProducers,
      siteInputs,
      itemContract: elementContract = null,
      extraHeads = [],
      partialUnionFields = false,
      constructors = true,
      readFacts = null,
      routedBases = new Set<string>(),
      inboxTail = TAKE_TAIL,
      messages = true,
    } = site;
    let paths: string[];
    try {
      paths = exprRefsOf(text);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      errors.push([`${where}: ${error.message}`, rng]);
      return;
    }
    const trees = spanTrees(text);
    addBuiltinFieldErrors(trees, where, rng);
    if (registry !== null && constructors) {
      /** The proven type of one reference this site reads. */
      const refType = (path: string): Type | null => {
        const [head, ...fields] = path.split(".");
        let contract: OutputContract | null;
        if (head === "item") {
          contract = elementContract !== null ? projectOutputContract(elementContract, fields) : null;
        } else {
          contract = referenceContract(path, siteProducers, siteInputs);
        }
        return outputContractType(contract);
      };
      for (const tree of trees) {
        for (const problem of constructorErrors(tree, registry, refType)) {
          constructorProblems.push([`${where}: ${problem}`, rng]);
        }
      }
    }
    for (const path of paths) {
      const parts = path.split(".");
      if (parts[0] === "input" && inboxInputs.has(parts[1]!)) {
        checkInboxRead(text, parts, where, rng, inboxTail);
        continue;
      }
      let err = classifyPath(path, messages ? validTargets : nodeTargets, flowInputs, extraHeads);
      if (!messages && parts[1] === NEW_MESSAGES && validTargets.has(messagesKey(parts[0]!))) {
        err =
          `reference \${${path}}: a case can't route on an agent's new_messages yet; ` +
          "read it in a node input, and route on that node's output";
      }
      const producer = siteProducers.get(parts[0]!);
      const input = parts.length >= 3 && parts[0] === "input" ? siteInputs.get(parts[1]!) : undefined;
      if (err === null && parts.length >= 2 && parts[1] === "output" && producer !== undefined) {
        err = outputContractPathError(producer, parts.slice(2), path, { partialUnionFields });
      } else if (err === null && parts.length >= 2 && parts[1] === NEW_MESSAGES) {
        const messagesContract = siteProducers.get(messagesKey(parts[0]!));
        if (messagesContract !== undefined) {
          err = outputContractPathError(messagesContract, parts.slice(2), path, { partialUnionFields });
        }
      } else if (err === null && input !== undefined) {
        err = outputContractPathError(input, parts.slice(2), path, { partialUnionFields });
      } else if (err === null && parts.length >= 2 && parts[0] === "item" && elementContract !== null) {
        err = outputContractPathError(elementContract, parts.slice(1), path, { partialUnionFields });
      }
      if (err !== null && readFacts !== null && readFacts.size > 0) {
        const candidate = envReadType(path, readFacts);
        if (candidate !== null) {
          err = narrowedReadError(path, candidate, err, registry ?? new Map(), {
            byRouting: routedBases.has(candidate[2]),
          });
        }
      }
      if (err !== null) errors.push([`${where}: ${err}`, rng]);
    }
    try {
      checkTemplateCalls(text);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      errors.push([`${where}: ${error.message}`, rng]);
    }
  }

  /**
   * Count one inbox read in `text` as allowed, or add its error at `where`: an inbox is read only
   * whole, as the lone `${input.<name>}`, at a site whose `inboxTail` is `null`.
   */
  function checkInboxRead(
    text: string,
    parts: readonly string[],
    where: string,
    rng: SourceRange | null,
    inboxTail: string | null,
  ): void {
    const lone = directReferencePathParts(text);
    let tail = inboxTail;
    if (parts.length > 2) tail = "(an inbox has no fields)";
    else if (lone === null || lone.length !== 2) tail = EXPRESSION_TAIL;
    if (tail === null) {
      inboxesRead.add(parts[1]!);
      return;
    }
    errors.push([
      `${where}: \${input.${parts[1]}} is an inbox; it is read only whole, bound to a child's Inbox input ${tail}`,
      rng,
    ]);
  }

  /**
   * Check an AGENT's `steer_inbox:` source: the lone `${input.<name>}` of an inbox input, counted
   * as a whole read. Any other read of an inbox gets the inbox read errors; a source that reads no
   * inbox isn't an inbox.
   */
  function checkSteerInbox(
    fromValue: unknown,
    where: string,
    rng: SourceRange | null,
    site: Pick<ScanOptions, "siteProducers" | "siteInputs">,
  ): string | null {
    const source = bindingSource(fromValue);
    const text = typeof source === "string" ? source : null;
    const lone = directReferencePathParts(fromValue);
    if (lone !== null && lone.length === 2 && lone[0] === "input" && inboxInputs.has(lone[1]!)) {
      scan(fromValue, where, rng, { ...site, constructors: false, inboxTail: null });
      return lone[1]!;
    }
    let readsInbox = false;
    try {
      readsInbox =
        text !== null &&
        exprRefsOf(text).some((path) => {
          const parts = path.split(".");
          return parts[0] === "input" && inboxInputs.has(parts[1]!);
        });
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
    }
    if (readsInbox) {
      scan(fromValue, where, rng, { ...site, constructors: false });
      return null;
    }
    const written = text ?? valueRepr(source);
    errors.push([`${where}: \`${written}\` isn't an inbox; \`steer_inbox:\` names an Inbox input`, rng]);
    return null;
  }

  /**
   * Check an AGENT's steer `render:` template: it names only `${event}`, whose field reads are
   * checked against `inbox`'s element type. A malformed span is one error at `where`.
   */
  function checkSteerRender(
    where: string,
    template: string,
    inbox: string,
    rng: SourceRange | null,
    partialUnionFields: boolean,
  ): void {
    let refs: string[] = [];
    try {
      refs = promptRefs(template);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
    }
    const others = refs.filter((ref) => ref.split(".", 1)[0]!.trim() !== "event");
    for (const ref of others) {
      const head = ref.split(".", 1)[0]!.trim();
      errors.push([`${where}: \`\${${ref}}\` names \`${head}\`; a steer render reads only \`\${event}\``, rng]);
    }
    if (others.length > 0) return;
    const element = inputContracts.get(inbox)?.type.element ?? null;
    const contracts = new Map<string, LocalInput>(
      element === null ? [] : [["event", new LocalInput([new ValueContract(element)])]],
    );
    checkTemplate(where, template, new Set(["event"]), rng, {
      noun: "steer event",
      hintTail: "a steer render",
      contracts,
      partialUnionFields,
    });
  }

  /** Add one error at `where` per field read on a builtin's result in `trees`. */
  function addBuiltinFieldErrors(trees: readonly ExprNode[], where: string, rng: SourceRange | null): void {
    for (const tree of trees) {
      for (const problem of builtinFieldErrors(tree)) errors.push([`${where}: ${problem}`, rng]);
    }
  }

  /**
   * Parse one rendered template; check each ref against `declared` and `contracts`.
   *
   * A malformed span, unknown builtin, builtin called with arguments it doesn't take, or nested
   * `${...}` is one error at `where`, and so is each field read on a builtin's result its type
   * can't have; each ref whose head is not in `declared` is another, with the pool-ref hint (ending
   * in `hintTail`) when the head is a pool-style head or a node id. Each distinct ref reading a
   * field of a declared name with an entry in `contracts` is another when every alternative
   * rejects the read (`partialUnionFields` allows a field only some union members have).
   */
  function checkTemplate(
    where: string,
    template: string,
    declared: ReadonlySet<string>,
    rng: SourceRange | null,
    site: {
      noun: string;
      hintTail: string;
      contracts: ReadonlyMap<string, LocalInput>;
      partialUnionFields: boolean;
    },
  ): void {
    let refs: string[];
    try {
      refs = promptRefs(template);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      errors.push([`${where}: ${error.message}`, rng]);
      return;
    }
    addBuiltinFieldErrors(spanTrees(template), where, rng);
    const fieldChecked = new Set<string>();
    for (const ref of refs) {
      const head = ref.split(".", 1)[0]!.trim();
      if (!declared.has(head)) {
        const hint =
          PROMPT_FORBIDDEN_HEADS.has(head) || validTargets.has(head)
            ? ` (a pool ref belongs in an input \`from:\`, not ${site.hintTail})`
            : "";
        errors.push([`${where}: \${${ref}} is not a declared ${site.noun}${hint}`, rng]);
        continue;
      }
      const path = ref.trim();
      const parts = path.split(".");
      const local = site.contracts.get(head);
      if (local === undefined || parts.length < 2 || fieldChecked.has(path)) continue;
      fieldChecked.add(path);
      const err = localFieldError(local.alternatives, parts.slice(1), path, {
        source: local.source,
        partialUnionFields: site.partialUnionFields,
      });
      if (err !== null) errors.push([`${where}: ${err}`, rng]);
    }
  }

  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    const franges = result.fieldRanges;
    const nodeRange = result.nodeRange;
    const [nodeProducers, nodeInputs] =
      narrowing !== null ? narrowing.forNode(nodeId) : [producers, inputContracts];
    const partialUnionFields = unordered.has(nodeId);
    // Prompt scope: a strict AGENT prompt or system prompt may interpolate ONLY this node's
    // declared inputs and variables as `${name}` (or a builtin-call argument); a pool ref in either
    // is loud, and an unknown builtin or malformed span is rejected here (via `promptRefs`, the
    // renderer's compile-time companion).
    const nodeLocals = localContracts.get(nodeId) ?? new Map<string, LocalInput>();
    if (node instanceof AgentNode) {
      const declared = new Set((node.params ?? []).map((param) => param.name));
      for (const variable of node.variables) declared.add(variable.name);
      const promptContracts = new Map(nodeLocals);
      for (const variable of node.variables) {
        if (variable.contract !== null) promptContracts.set(variable.name, new LocalInput([variable.contract]));
      }
      const promptSite = {
        noun: "input or variable",
        hintTail: "a strict AGENT prompt",
        contracts: promptContracts,
        partialUnionFields,
      };
      const promptRange = franges.get(siteKey("field", "prompt")) ?? nodeRange;
      if (typeof node.prompt === "string") {
        const where = result.siteName(nodeId, "prompt");
        if (node.prompt) checkTemplate(where, node.prompt, declared, promptRange, promptSite);
      } else {
        // A list prompt: each message's text is a template; a splice names a local holding messages.
        node.prompt.forEach((item, index) => {
          const where = `${result.siteName(nodeId, "prompt")}[${index}]`;
          const rng = franges.get(siteKey("field", `prompt[${index}]`)) ?? promptRange;
          if ("text" in item) {
            if (item.text) checkTemplate(where, item.text, declared, rng, promptSite);
            return;
          }
          const ref = `\${${item.splice}}`;
          checkTemplate(where, ref, declared, rng, promptSite);
          const label = declared.has(item.splice) ? spliceTypeError(promptContracts.get(item.splice)) : null;
          if (label !== null) {
            errors.push([
              `${where}: ${ref} is \`${label}\`, which can't hold messages; splice a \`Message\`, a record ` +
                "with `role` and `text: str`, or a list of them",
              rng,
            ]);
          }
        });
      }
      if (node.systemPrompt) {
        checkTemplate(
          result.siteName(nodeId, "system_prompt"),
          node.systemPrompt,
          declared,
          franges.get(siteKey("field", "system_prompt")) ?? nodeRange,
          promptSite,
        );
      }
    }
    // Node-owned reserved wiring (`over`, `until`, ...) follows the same pool reference grammar as
    // ordinary inputs. It never receives a body-local scope: in particular, MAP's `${item}` is
    // valid only in each per-element input, not in `over:`.
    const nwiring = flowWiring.get(nodeId) ?? new Map<string, unknown>();
    const reserved = node.reservedWiringKeys();
    const inputRng = franges.get(siteKey("field", "inputs")) ?? nodeRange;
    for (const key of [...reserved].sort(compareCodeUnits)) {
      // A reserved wiring key is a KIND FIELD, so its range is keyed `field` — not the bare key,
      // which would silently fall back to the whole `input:` block or node header.
      const reservedRng = franges.get(siteKey("field", key)) ?? inputRng;
      if (node instanceof AgentNode && key === STEER_INBOX_KEY) {
        const where = result.siteName(nodeId, "steer_inbox");
        const inbox = checkSteerInbox(nwiring.get(key), where, reservedRng, {
          siteProducers: nodeProducers,
          siteInputs: nodeInputs,
        });
        if (node.steerRender !== null && inbox !== null) {
          const renderRng = franges.get(siteKey("field", "steer_inbox.render")) ?? reservedRng;
          checkSteerRender(`${where} render`, node.steerRender, inbox, renderRng, partialUnionFields);
        }
        continue;
      }
      scan(nwiring.get(key), result.siteName(nodeId, node.reservedKeyLabel(key)), reservedRng, {
        siteProducers: nodeProducers,
        siteInputs: nodeInputs,
        partialUnionFields,
        constructors: false,
        inboxTail: node.bindsPerItem ? MAP_TAIL : TAKE_TAIL,
      });
    }

    // A per-item binder (MAP) reads its iteration source above, then validates each element
    // input with the body-local `${item}` scope enabled.
    if (node.bindsPerItem) {
      const item = itemContract(result, nwiring, nodeProducers, nodeInputs, typeFn);
      for (const [param, src] of nwiring) {
        if (reserved.has(param)) continue;
        scan(
          src,
          `node ${reprText(nodeId)} map input ${reprText(param)} from`,
          franges.get(siteKey("input", param)) ?? inputRng,
          {
            siteProducers: nodeProducers,
            siteInputs: nodeInputs,
            itemContract: item,
            extraHeads: ["item"],
            partialUnionFields,
            inboxTail: MAP_TAIL,
          },
        );
      }
      continue;
    }

    // Each node input `from:` source, read from the flow-owned wiring. For a desugared CASE these
    // are the `__rN`/`__on` SOURCES (the original `${<id>.output}` refs), validated against the
    // pool. The rewritten node-local `__rN`/`__on` live in `node.cases` `when:` and are excluded
    // here (they resolve against the record).
    // Anchor each error on the OFFENDING ENTRY, not the whole `input:` mapping: the parse loop
    // stamps a per-binding `input` range for every kind, and a desugar stamps one for its
    // synthetic params (a CASE `__rN` -> its `cases[N].when` line); the block range is only the
    // fallback (a spanless flow has no ranges at all). `wiringRefLabel` gives a synthetic param
    // the author-facing name it was hoisted from instead of leaking `__rN`.
    const caseReads =
      node instanceof CaseNode
        ? new Set(
            node.cases.flatMap((branch) =>
              branch.when ? conditionRefs(branch.when).map((ref) => ref.split(".", 1)[0]!) : [],
            ),
          )
        : null;
    for (const [param, src] of nwiring) {
      if (reserved.has(param)) continue;
      if (caseReads !== null && caseReads.size > 0 && !caseReads.has(param)) {
        // In a searched case whose `when:`s read something, a param they don't read is a join
        // input over a co-skipping escape's producers; each of its references is already checked
        // at the read that introduced it, so scanning it would repeat that error. An `on:` case
        // with no cases reads nothing, so its `on:` value is still checked.
        continue;
      }
      let label = node.wiringRefLabel(param);
      let paramRng = franges.get(siteKey("input", param)) ?? inputRng;
      let [paramProducers, paramInputs] = [nodeProducers, nodeInputs];
      let readFacts: Env | null = null;
      let routedBases: ReadonlySet<string> = new Set();
      if (narrowing !== null) {
        if (narrowing.paramUnchecked(nodeId, param)) {
          // Every read follows a check that can never be true, which `checkImpossibleWhens`
          // reports; no value reaches them.
          continue;
        }
        // A searched CASE's param read only after the checks before it in its `when:` has its
        // own view; one read guarded first and unguarded later reports at the later read.
        [paramProducers, paramInputs] = narrowing.forParam(nodeId, param);
        readFacts = narrowing.paramEnv(nodeId, param);
        routedBases = narrowing.paramRoutedBases(nodeId, param);
        const paramSite = narrowing.paramSite(nodeId, param);
        if (paramSite !== null) {
          label = paramSite[0];
          paramRng = paramSite[1] ?? paramRng;
        }
      }
      const where = `${result.siteName(nodeId, label)} from`;
      const source = bindingSource(src);
      if (typeof source === "string") {
        // A desugar may copy an authored binding to a synth node, so check the authored site once.
        const key = JSON.stringify([where, source, paramRng]);
        if (scanned.has(key)) continue;
        scanned.add(key);
      }
      // A typed binding or a value holding the inbox doesn't pass it whole.
      const whole = !(src instanceof Binding && src.contract !== null) && !(source instanceof ValueTemplate);
      scan(src, where, paramRng, {
        siteProducers: paramProducers,
        siteInputs: paramInputs,
        partialUnionFields,
        readFacts,
        routedBases,
        inboxTail: whole ? inboxSiteTail(nodeId, node, param) : TAKE_TAIL,
        // A CASE's routing edges follow only `${x.output}` reads.
        messages: !(node instanceof CaseNode),
      });
    }
  }

  // Flow outputs (the codomain): each binding `from:` is checked like a node input and located at
  // its retained value or block range. No built node owns this source site; source-free callers
  // deliberately remain unlocated.
  for (const value of outputBindings(outputs)) {
    scan(value, "flow output from", outputsRange, { siteProducers: producers, siteInputs: inputContracts });
  }

  for (const [headline, problems] of [
    ["flow has unresolved references", errors],
    ["flow has constructor calls that can never succeed", constructorProblems],
  ] as const) {
    if (problems.length > 0) {
      const firstRange = problems.find(([, range]) => range !== null)?.[1] ?? null;
      throw diagnostics.error(`${headline}:\n  ${problems.map(([message]) => message).join("\n  ")}`, {
        range: firstRange,
      });
    }
  }

  // Reported only when no site error was: a refused read leaves its inbox unread too.
  const unread = [...inboxInputs].filter((name) => !inboxesRead.has(name));
  if (unread.length > 0) {
    const problems = unread.map((name): [string, SourceRange | null] => {
      const origin = inputContracts.get(name)!.origins.default;
      const frame = origin !== null && resolveOrigin !== null ? resolveOrigin(origin) : null;
      const message = `inbox input \`${name}\` is never read; take from it, or pass it to a call`;
      return [message, frame?.range ?? outputsRange];
    });
    throw diagnostics.error(`flow has unread inbox inputs:\n  ${problems.map(([message]) => message).join("\n  ")}`, {
      range: problems.find(([, range]) => range !== null)?.[1] ?? null,
    });
  }
}

/** Return the parsed tree of each `${...}` span of `text`. */
function spanTrees(text: string): ExprNode[] {
  return scanTemplate(text).flatMap((segment) => (segment instanceof Span ? [segment.tree] : []));
}
