/**
 * Infer canonical value contracts for Compose output bindings.
 *
 * It lives below `compose/` so that the run time can derive the same types the loader checks.
 *
 * This module owns the Compose-specific envelope around the generic expression engine: template
 * shape, reference-to-contract lookup, record-field projection, and provenance retention. Direct
 * references re-export their exact or partial contract; computed expressions receive a
 * source-free exact contract only when `expr` proves their result type.
 *
 * Knows about: `expr` AST result inference, semantic binding/contract carriers, and
 * source-neutral `typesys` values.
 * Never imports: YAML/source documents, node kinds, graph-stage passes, the loader, or runtime.
 */

import { type OriginRef, reprText } from "../diagnostics.ts";
import {
  ExpressionError,
  evalBinding,
  exprRefs,
  inferExprType,
  MISSING_ROOT,
  scanTemplate,
  splitReference,
} from "../expr/index.ts";
import { type ExprNode, Token, Tree } from "../expr/grammar.ts";
import { Span, templateRefs } from "../expr/template.ts";
import {
  Binding,
  bindingSource,
  ContractOrigins,
  comparePaths,
  isExactly,
  type OutputContract,
  outputContractFieldError,
  outputContractType,
  PartialRecordContract,
  sameItems,
  ValueContract,
  type ValuePath,
  ValueTemplate,
} from "../nodes/contracts.ts";
import {
  buildValue,
  isList,
  isRecord,
  jsTypeName,
  listKindForElement,
  Type,
  TypeCheckError,
  unionField,
  ValueKind,
} from "../typesys/index.ts";
import type { Node } from "../nodes/base.ts";
import { messagesKey, NEW_MESSAGES } from "../typesys/pool.ts";
import { fileFieldType, isListKind, joinedType, typeEquals } from "../typesys/values.ts";
import { SYSTEM_AMBIENTS, typeRelation } from "./validation.ts";

/** Marks a binding source that is a template, not a literal, in `literalValue`. */
export const NOT_LITERAL: unique symbol = Symbol("NOT_LITERAL");

/** Resolves a `T(...)` constructor name to its type, or `null` for a name that is none. */
export type TypeFn = (name: string) => Type | null;

/**
 * Project a proven path from an exact or partial record output contract.
 *
 * A field read on a union projects to the members' shared field type; a field only some members
 * have, or whose types differ, projects to `null`. A file's `uri`, `mime` and `name` project to
 * their `fileFieldType`. A projected type is `Optional` when anything on the path may be absent.
 *
 * @throws TypeError when `contract` is neither an exact `ValueContract` nor a
 *   `PartialRecordContract`.
 */
export function projectOutputContract(contract: OutputContract, fields: readonly string[]): OutputContract | null {
  if (isExactly(contract, PartialRecordContract)) {
    contract.validateShape();
    if (fields.length === 0) return contract;
    const [fieldName, ...remaining] = fields as [string, ...string[]];
    const entry = contract.fields.find(([name]) => name === fieldName);
    if (entry === undefined || entry[1] === null) return null;
    return projectOutputContract(entry[1], remaining);
  }
  if (!isExactly(contract, ValueContract)) {
    throw new TypeError(
      `output contract must be an exact ValueContract or PartialRecordContract, got ${jsTypeName(contract)}`,
    );
  }
  if (fields.length === 0) return contract;
  let typ = contract.type;
  let pathNullable = typ.nullable;
  for (const field of fields) {
    if (typ.members !== undefined) {
      const read = unionField(typ, field);
      if (read.kind !== "read" || read.type === null) return null;
      // The shared type is already nullable when any member's field may be absent.
      pathNullable = pathNullable || read.type.nullable;
      typ = read.type;
      continue;
    }
    if (typ.fields === undefined) {
      const fieldType = typ.kind === ValueKind.FILE ? fileFieldType(field) : null;
      if (fieldType === null) return null;
      pathNullable = pathNullable || fieldType.nullable;
      typ = fieldType;
      continue;
    }
    const fieldType = typ.fields.get(field);
    if (fieldType === undefined) return null;
    pathNullable = pathNullable || !(typ.required?.has(field) ?? false) || fieldType.nullable;
    typ = fieldType;
  }
  if (pathNullable && !typ.nullable) typ = typ.replace({ nullable: true });
  const entries = contract.origins.entries
    .filter(([path]) => path.length > fields.length && sameItems(path.slice(0, fields.length), fields))
    .map(([path, origin]) => [path.slice(fields.length), origin] as const);
  return new ValueContract(typ, new ContractOrigins({ default: contract.origins.at(fields), entries }));
}

/**
 * Check the contract maps a validation pass reads from.
 *
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name (default none).
 * @throws TypeError naming the first producer contract that is not a `ValueContract` or
 *   `PartialRecordContract`, or the first flow input contract that is not an exact
 *   `ValueContract`.
 */
export function assertExactContracts(
  producers: ReadonlyMap<string, unknown>,
  flowInputs: ReadonlyMap<string, unknown> = new Map(),
): void {
  for (const [name, contract] of producers) checkProducerContract(name, contract);
  for (const [name, contract] of flowInputs) checkFlowInputContract(name, contract);
}

/** @throws TypeError when `contract` is not an exact `ValueContract` or `PartialRecordContract`. */
function checkProducerContract(name: string, contract: unknown): void {
  if (!isExactly(contract, ValueContract) && !isExactly(contract, PartialRecordContract)) {
    throw new TypeError(
      "output producer contracts must be ValueContract or PartialRecordContract values, " +
        `got ${jsTypeName(contract)} for ${reprText(name)}`,
    );
  }
}

/** @throws TypeError when `contract` is not an exact `ValueContract`. */
function checkFlowInputContract(name: string, contract: unknown): void {
  if (!isExactly(contract, ValueContract)) {
    throw new TypeError(
      `flow input contracts must be exact ValueContract values, got ${jsTypeName(contract)} for ${reprText(name)}`,
    );
  }
}

/**
 * Resolve one `input.X...`, `<id>.output...`, `system.X` or an AGENT's `<id>.new_messages...`
 * path to its exact or projected contract. A system ambient has its scalar type.
 *
 * @param path The dotted reference path.
 * @param producers Node output contracts by producer id, and an AGENT's new messages' contract
 *   under its pool name `<id>.new_messages`.
 * @param flowInputs Flow input contracts by name.
 * @returns The projected contract, or `null` for an unknown or unsupported path.
 * @throws TypeError when a referenced producer is not an `OutputContract`, or a flow-input
 *   contract is not an exact `ValueContract`.
 */
export function referenceContract(
  path: string,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
): OutputContract | null {
  const parts = path.split(".");
  const head = parts[0];
  if (head === "input" && parts.length >= 2) {
    const name = parts[1]!;
    const contract = flowInputs.get(name);
    if (contract === undefined) return null;
    checkFlowInputContract(name, contract);
    return projectOutputContract(contract, parts.slice(2));
  }
  if (head === "system") {
    const kind = parts.length === 2 ? SYSTEM_AMBIENTS.get(parts[1]!) : undefined;
    return kind === undefined ? null : new ValueContract(Type.scalar(kind));
  }
  if (parts.length >= 2 && parts[1] === NEW_MESSAGES) {
    const messages = producers.get(messagesKey(parts[0]!));
    return messages === undefined ? null : projectOutputContract(messages, parts.slice(2));
  }
  if (parts.length < 2 || parts[1] !== "output") return null;

  const producerId = parts[0]!;
  const contract = producers.get(producerId);
  if (contract === undefined) return null;
  checkProducerContract(producerId, contract);
  return projectOutputContract(contract, parts.slice(2));
}

/**
 * Return the field-read error for a node-local name's path, else `null`.
 *
 * A local name's value may have any of several proven contracts (a typed binding's declared one
 * and the one its source is inferred to have, or a LOOP's seed and carried field), so the read
 * is an error only when every alternative rejects it.
 *
 * @param alternatives The name's proven contracts; the first one's error is reported.
 * @param fields The field names the reference reads below the name.
 * @param path The reference path without `${}`, for the message.
 * @param options.source The pool path the name is bound from when its binding's one span reads
 *   a single path, so a member-only field's fix names a union a case can route on. Default
 *   `null`.
 * @param options.partialUnionFields Whether a field only some union members have can be read.
 *   Default `false`.
 * @returns The first alternative's error, or `null` when any alternative accepts the read.
 */
export function localFieldError(
  alternatives: readonly OutputContract[],
  fields: readonly string[],
  path: string,
  options: { source?: string | null; partialUnionFields?: boolean } = {},
): string | null {
  const { source = null, partialUnionFields = false } = options;
  let first: string | null = null;
  for (const contract of alternatives) {
    const error = outputContractFieldError(contract, fields, path, { partialUnionFields, unionRoot: source });
    if (error === null) return null;
    first ??= error;
  }
  return first;
}

/**
 * Resolve one Compose output binding to its canonical value contract.
 *
 * A typed binding (a `Binding` whose `contract` is set) resolves to its declared contract, since
 * that type is what the binding guarantees at runtime. Otherwise a direct whole-value reference
 * retains the referenced contract, including provenance. An explicit null default
 * `${ref :- null}` over such a reference retains it as `Optional`; `${(a | b) :- null}` over
 * references of one exact type receives a source-free `Optional` contract. A computed
 * whole-value expression receives a new source-free contract only when the expression engine
 * proves its result type. Embedded or multiple template spans always produce `str`; malformed,
 * literal-text, and unsupported computed bindings remain untyped. A `T(...)` constructor has type
 * `T` (`typeFn` resolves it). A value template's contract is inferred from its structure
 * (`templateContract`), each string by these rules and each literal by its kind.
 *
 * @param binding A raw binding value or exact semantic `Binding` carrier, possibly carrying a
 *   declared contract.
 * @param producers Exact or partial node output contracts keyed by producer id. A proven field
 *   of a partial record remains projectable.
 * @param flowInputs Exact flow input contracts keyed by input name.
 * @param options.typeFn Resolves a constructor name to its type; `null` (the default) leaves
 *   constructors untyped.
 * @returns The proven canonical exact or partial contract, or `null` when the binding's result
 *   shape is not statically proven.
 * @throws TypeError when a referenced producer is not an `OutputContract`, or a flow-input
 *   contract is not an exact `ValueContract`.
 */
export function resolveOutputBindingContract(
  binding: unknown,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: { typeFn?: TypeFn | null } = {},
): OutputContract | null {
  const { typeFn = null } = options;
  if (binding instanceof Binding && binding.contract !== null) return binding.contract;
  const source = bindingSource(binding);
  if (source instanceof ValueTemplate) {
    return templateContract(source, (text) => resolveOutputBindingContract(text, producers, flowInputs, { typeFn }));
  }
  if (typeof source !== "string") return null;
  let segments;
  try {
    segments = scanTemplate(source);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
  if (!segments.some((segment) => segment instanceof Span)) return null;
  const refs = templateRefs(segments);
  const referenceContracts = new Map(refs.map((path) => [path, referenceContract(path, producers, flowInputs)]));
  const only = segments.length === 1 ? segments[0] : undefined;
  if (!(only instanceof Span)) return new ValueContract(Type.scalar(ValueKind.STRING));

  const expression = only.tree;
  if (isDirectReference(expression)) {
    if (refs.length !== 1) return null;
    const contract = referenceContracts.get(refs[0]!) ?? null;
    // A `?.` step reads null when its receiver is null or lacks the key, as `inferExprType` types it.
    if (isExactly(contract, ValueContract) && hasOptionalStep(expression)) {
      return new ValueContract(contract.type.replace({ nullable: true }), contract.origins);
    }
    return contract;
  }
  const nullDefault = nullDefaultContract(expression, referenceContracts);
  if (nullDefault !== null) return nullDefault;

  /** Project one referenced contract to the type requested by `expr`. */
  const referenceType = (path: string): Type | null => {
    const contract = referenceContracts.has(path)
      ? referenceContracts.get(path)!
      : referenceContract(path, producers, flowInputs);
    return outputContractType(contract);
  };

  const inferred = inferExprType(expression, referenceType, typeFn);
  return inferred === null ? null : new ValueContract(inferred);
}

/**
 * Infer a value template's contract from its structure.
 *
 * A mapping is a closed record of its fields' contracts: exact when every field is exact, else
 * partial. A list is a list of its items' joined type when every item is exact and the items are
 * equal up to `Optional` (or become equal once literal refinements are widened); otherwise, and
 * for an empty list, it is unproven. A nested typed binding has its declared contract, a template
 * string the contract `stringContract` gives it, and a literal (including a string with no
 * `${...}` span) its kind, as a top-level literal would leave it open to change.
 *
 * @param template The template to infer.
 * @param stringContract Infers one template string's contract.
 * @param options.exactLiterals Give each literal its exact type (a one-tag `Literal`, one-point
 *   bounds), for checking the value against a declared type, where a wrong literal must not fit.
 *   Default `false`.
 * @returns The inferred exact or partial contract, or `null` when the value's shape is not
 *   proven.
 */
export function templateContract(
  template: ValueTemplate,
  stringContract: (text: string) => OutputContract | null,
  options: { exactLiterals?: boolean } = {},
): OutputContract | null {
  const { exactLiterals = false } = options;
  return treeContract(template.value, stringContract, exactLiterals);
}

/** Infer the contract of one node of a template tree (see `templateContract`). */
function treeContract(
  node: unknown,
  stringContract: (text: string) => OutputContract | null,
  exactLiterals: boolean,
): OutputContract | null {
  if (node instanceof Binding) {
    if (node.contract !== null) return node.contract;
    return treeContract(node.source, stringContract, exactLiterals);
  }
  if (isRecord(node)) {
    return combineOutputContracts(
      new Map(Object.entries(node).map(([key, item]) => [key, treeContract(item, stringContract, exactLiterals)])),
    );
  }
  if (isList(node)) {
    return listContract(node.map((item) => treeContract(item, stringContract, exactLiterals)));
  }
  const literal = literalValue(node);
  if (literal === NOT_LITERAL) return stringContract(node as string);
  const typ = literalType(literal);
  if (typ === null) return null;
  return new ValueContract(exactLiterals ? typ : widened(typ));
}

/**
 * Infer the contract of a span-free plain literal, as written out.
 *
 * A mapping is a closed record of its keys' contracts, a list is a list of its items' joined type
 * (as in `templateContract`), and a scalar has its kind's widened type. A nested mapping with an
 * unproven field, or a list whose items don't join, is opaque as a whole, since a partial record
 * can't nest. Every string is `str`: the value is already read, so `$$` escapes are undone and a
 * string is never scanned for `${...}` spans.
 *
 * @param value The literal, with no template or binding inside it.
 * @param options.exactLiterals Give each scalar its exact type (a one-tag `Literal`, one-point
 *   bounds) instead, so a literal tag selects the union member it names. Default `false`.
 * @returns The exact or partial contract, or `null` when the value's shape is unproven (a
 *   non-finite number, a list whose items don't join).
 */
export function literalContract(value: unknown, options: { exactLiterals?: boolean } = {}): OutputContract | null {
  const { exactLiterals = false } = options;
  // Not `treeContract`: it would undo `$$` escapes a second time and read a `${...}` left by an
  // escape as a span, but this value's strings are final text.
  if (isRecord(value)) {
    return combineOutputContracts(
      new Map(Object.entries(value).map(([key, item]) => [key, literalContract(item, { exactLiterals })])),
    );
  }
  if (isList(value)) {
    return listContract(value.map((item) => literalContract(item, { exactLiterals })));
  }
  const typ = literalType(value);
  if (typ === null) return null;
  return new ValueContract(exactLiterals ? typ : widened(typ));
}

/** Return the list contract of exact items with one joined type, else `null`. */
function listContract(items: readonly (OutputContract | null)[]): ValueContract | null {
  // Unlike an inline body's strict element join, this leaves a list it can't join unproven (the
  // runtime check covers it) instead of failing the load.
  if (items.length === 0 || !items.every((item) => isExactly(item, ValueContract))) return null;
  const types = items.map((item) => item.type);
  const element = joinedType(types) ?? joinedType(types.map(widened));
  if (element === null) return null;
  return new ValueContract(new Type({ kind: listKindForElement(element), element }));
}

/** Return `typ` with literal refinements (tags, bounds) dropped, at every depth. */
function widened(typ: Type): Type {
  return typ.replace({
    tags: undefined,
    bounds: undefined,
    fields:
      typ.fields === undefined
        ? undefined
        : new Map([...typ.fields].map(([name, field]) => [name, widened(field)] as const)),
    element: typ.element === undefined ? undefined : widened(typ.element),
  });
}

/**
 * Return the value a literal binding source binds, or `NOT_LITERAL` for a template.
 *
 * A string with no `${...}` span binds its text after `$$` unescaping, exactly as the binder
 * evaluates it; a value template is a template; any other value is bound as is.
 *
 * @param source The binding's raw source.
 * @returns The bound literal value, or `NOT_LITERAL`.
 */
export function literalValue(source: unknown): unknown {
  if (source instanceof ValueTemplate) return NOT_LITERAL;
  if (typeof source !== "string") return source;
  try {
    const segments = scanTemplate(source);
    if (segments.some((segment) => segment instanceof Span)) return NOT_LITERAL;
    return evalBinding(source, MISSING_ROOT);
  } catch (error) {
    if (error instanceof ExpressionError) return NOT_LITERAL; // malformed text is reported by reference validation
    throw error;
  }
}

/**
 * Return the exact type of a literal value, else `null`.
 *
 * A `bigint` is an `int` and a `number` a `float`.
 *
 * @param value The bound literal.
 * @returns A one-tag `Literal` for non-empty text, one-point bounds for a finite number, the
 *   value's kind otherwise; `null` for a non-finite number or a value with no type.
 */
export function literalType(value: unknown): Type | null {
  if (typeof value === "bigint") return new Type({ kind: ValueKind.INTEGER, bounds: [value, value] });
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return null;
    return new Type({ kind: ValueKind.NUMBER, bounds: [value, value] });
  }
  if (typeof value === "string") {
    // A tag must be non-empty, so empty text is plain str.
    if (!value) return Type.scalar(ValueKind.STRING);
    return new Type({ kind: ValueKind.STRING, tags: new Set([value]) });
  }
  try {
    return Type.scalar(buildValue(value).kind);
  } catch (error) {
    if (error instanceof TypeCheckError) return null;
    throw error;
  }
}

/**
 * Type `head :- null` as Optional when `head` is a reference or a coalesce of them.
 *
 * @param expression The parsed tree of a template's single span.
 * @param referenceContracts The resolved contract of every reference path the span reads.
 * @returns The head reference's exact contract made nullable with its origins, a source-free
 *   nullable contract for a coalesce of references of one exact type, or `null` for any other
 *   expression.
 */
function nullDefaultContract(
  expression: ExprNode,
  referenceContracts: ReadonlyMap<string, OutputContract | null>,
): ValueContract | null {
  if (
    !(
      expression instanceof Tree &&
      expression.data === "default_expr" &&
      expression.children.length === 2 &&
      expression.children[1] instanceof Token &&
      expression.children[1].type === "NULL"
    )
  ) {
    return null;
  }
  const head = expression.children[0]!;
  if (head instanceof Tree && head.data === "coalesce") {
    const operands = head.children.map((operand) => directValueContract(operand, referenceContracts));
    if (!operands.every((operand) => operand !== null)) return null;
    const types = operands.map((operand) => operand.type.replace({ nullable: false }));
    if (types.slice(1).some((typ) => !typeEquals(typ, types[0]!))) return null;
    return new ValueContract(types[0]!.replace({ nullable: true }));
  }
  const contract = directValueContract(head, referenceContracts);
  if (contract === null) return null;
  return new ValueContract(contract.type.replace({ nullable: true }), contract.origins);
}

/**
 * Return the exact contract of an operand that is one direct reference.
 *
 * @param operand One parsed operand of a span's expression tree.
 * @param referenceContracts The resolved contract of every reference path the span reads.
 * @returns The referenced exact contract, or `null` when the operand is not a lone reference or
 *   its contract is partial or unknown.
 */
function directValueContract(
  operand: ExprNode,
  referenceContracts: ReadonlyMap<string, OutputContract | null>,
): ValueContract | null {
  if (!isDirectReference(operand)) return null;
  const contract = referenceContracts.get(exprRefs(operand)[0]!) ?? null;
  return isExactly(contract, ValueContract) ? contract : null;
}

/**
 * Return the parsed tree of a binding that is one whole-value `${...}` span.
 *
 * @param binding A raw binding value or semantic `Binding` carrier.
 * @returns The span's expression tree, or `null` for a non-text value, a literal, embedded text,
 *   or malformed text.
 */
export function loneSpanTree(binding: unknown): ExprNode | null {
  const source = bindingSource(binding);
  if (typeof source !== "string") return null;
  let segments;
  try {
    segments = scanTemplate(source);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
  const only = segments.length === 1 ? segments[0] : undefined;
  return only instanceof Span ? only.tree : null;
}

/**
 * Return the names of the path a binding reads when it is one lone whole-value reference.
 *
 * @param binding A raw binding value or semantic `Binding` carrier.
 * @returns The path's names (`["item", "name"]`), or `null` for a literal, an embedded or
 *   computed template, a call, or malformed text.
 */
export function directReferencePathParts(binding: unknown): string[] | null {
  const tree = loneSpanTree(binding);
  return tree !== null ? directReferenceParts(tree) : null;
}

/**
 * Return the path a binding reads when it is one lone whole-value reference.
 *
 * @param binding A raw binding value or semantic `Binding` carrier.
 * @returns The dotted reference path (`item`, `item.name`, `a.output`), or `null` for a literal,
 *   an embedded or computed template, a call, or malformed text.
 */
export function directReferencePath(binding: unknown): string | null {
  return directReferencePathParts(binding)?.join(".") ?? null;
}

/** Return whether a reference has a `?.` step. */
export function hasOptionalStep(reference: Tree): boolean {
  return splitReference(reference).steps.some((step) => step.optional);
}

/**
 * Return whether a whole-value span is a bare reference, not a call or operator.
 *
 * @param expression The parsed tree of a template's single span.
 * @returns `true` for a bare or dotted reference with no call suffix.
 */
function isDirectReference(expression: ExprNode): expression is Tree {
  return (
    expression instanceof Tree &&
    expression.data === "refcall" &&
    !expression.children.some((child) => child instanceof Tree && child.data === "call_suffix")
  );
}

/**
 * Combine named outputs into one exact or partial closed-record contract.
 *
 * @param fields Output field names mapped to proven contracts or `null` for opaque fields. A
 *   nested partial contract cannot be represented inside one root field and is therefore
 *   retained as opaque at this boundary. Insertion order is retained.
 * @returns One exact closed record when every field is exact; otherwise one partial record that
 *   retains the closed name set and every exact sibling contract.
 * @throws TypeError when a field's contract is not a `ValueContract`, a `PartialRecordContract`
 *   or `null`.
 */
export function combineOutputContracts(fields: ReadonlyMap<string, OutputContract | null>): OutputContract {
  for (const [name, contract] of fields) {
    if (contract !== null && !isExactly(contract, ValueContract) && !isExactly(contract, PartialRecordContract)) {
      throw new TypeError(
        "output fields must contain ValueContract, PartialRecordContract, " +
          `or null values, got ${jsTypeName(contract)} for ${reprText(name)}`,
      );
    }
  }
  const entries = [...fields];
  if (entries.some(([, contract]) => !isExactly(contract, ValueContract))) {
    return new PartialRecordContract(
      entries.map(([name, contract]) => [name, isExactly(contract, ValueContract) ? contract : null] as const),
    );
  }
  const exact = entries as [string, ValueContract][];
  return new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map(exact.map(([name, contract]) => [name, contract.type])),
      required: new Set(fields.keys()),
    }),
    recordFieldOrigins(exact),
  );
}

/**
 * Index each field contract's origins under its field name, as one record's origins.
 *
 * @param fields Record field names paired with exact contracts; `null` (an opaque field)
 *   contributes no origin.
 * @returns No root origin; each field's root origin at `[name]` and its nested origins below it,
 *   in canonical path order.
 */
export function recordFieldOrigins(fields: Iterable<readonly [string, ValueContract | null]>): ContractOrigins {
  const origins: (readonly [ValuePath, OriginRef])[] = [];
  for (const [name, contract] of fields) {
    if (contract === null) continue;
    if (contract.origins.default !== null) origins.push([[name], contract.origins.default]);
    for (const [path, origin] of contract.origins.entries) origins.push([[name, ...path], origin]);
  }
  origins.sort((a, b) => comparePaths(a[0], b[0]));
  return new ContractOrigins({ entries: origins });
}

/**
 * Return the path of a lone plain reference, excluding calls and expressions.
 *
 * @param expression One parsed `${...}` interior, as `Span.tree`.
 * @returns The dotted path's names, such as `["d", "output", "action"]` (a `?.` step gives its
 *   name too), or `null` when the expression is not a plain `refcall` of names (an index, a call,
 *   an escape such as `:-`, or any operator).
 */
export function directReferenceParts(expression: ExprNode): string[] | null {
  if (!(expression instanceof Tree) || expression.data !== "refcall") return null;

  const children = expression.children;
  const first = children[0];
  if (!(first instanceof Token) || first.type !== "NAME") return null;

  const parts = [first.value];
  for (const child of children.slice(1)) {
    // A `?.` trailer is `trailer[NAME, QDOT]`; its NAME is still the step.
    const name = child instanceof Tree && child.data === "trailer" ? child.children[0] : null;
    if (!(name instanceof Token) || name.type !== "NAME") return null;
    parts.push(name.value);
  }
  return parts;
}

/**
 * Return the contracts one binding's value is proven to have, else `null`.
 *
 * Unlike `inferBindingType`, a partial record is kept, since its closed root names can still be
 * checked, and a literal is widened, so a message names `str` rather than its one-tag `Literal`. A
 * typed binding also keeps the type inferred from its raw source when that fits the declared type,
 * as a narrower view of the same value.
 *
 * @param binding A wiring binding, raw or a `Binding`.
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name.
 * @param typeFn Resolves a constructor name to its type.
 * @returns The alternatives, the declared contract first for a typed binding, or `null` when the
 *   value's contract is unproven.
 */
export function bindingAlternatives(
  binding: unknown,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  typeFn: TypeFn | null,
): OutputContract[] | null {
  const source = bindingSource(binding);
  const literal = literalValue(source);
  if (binding instanceof Binding && binding.contract !== null) {
    const declared = binding.contract;
    // Infer from the raw source: the Binding itself resolves to its declared contract. Exact
    // literals, so a literal tag selects its union member.
    let inferred: OutputContract | null;
    if (literal !== NOT_LITERAL) {
      inferred = literalContract(literal, { exactLiterals: true });
    } else if (source instanceof ValueTemplate) {
      inferred = templateContract(
        source,
        (text) => resolveOutputBindingContract(text, producers, flowInputs, { typeFn }),
        { exactLiterals: true },
      );
    } else {
      inferred = resolveOutputBindingContract(source, producers, flowInputs, { typeFn });
    }
    if (isExactly(inferred, ValueContract) && typeRelation(inferred.type, declared.type) === "fits") {
      return [declared, inferred];
    }
    return [declared];
  }
  const contract =
    literal !== NOT_LITERAL
      ? literalContract(literal)
      : resolveOutputBindingContract(binding, producers, flowInputs, { typeFn });
  return contract === null ? null : [contract];
}

/**
 * Return the exact element contract `${item}` reads on a per-item node, else `null`.
 *
 * @param node The node; only a node with an item source key (MAP) has an item scope.
 * @param wiring The node's binding map, holding the item source under that key.
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name.
 * @param typeFn Resolves a constructor name to its type.
 * @returns The element contract when the item source is an exact list with a known element type,
 *   else `null`.
 */
export function itemContract(
  node: Node | null,
  wiring: ReadonlyMap<string, unknown>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  typeFn: TypeFn | null = null,
): ValueContract | null {
  const key = node !== null ? node.itemSourceKey : null;
  if (key === null || !wiring.has(key)) return null;
  const source = resolveOutputBindingContract(wiring.get(key), producers, flowInputs, { typeFn });
  if (!isExactly(source, ValueContract)) return null;
  if (!isListKind(source.type.kind) || source.type.element === undefined) return null;
  return new ValueContract(source.type.element);
}

/**
 * Return the type a value with `contract` has for field reads: a value contract's type; for a
 * partial record, a record of its names, each required, typed by its contract or `Any` when
 * opaque; `null` for no contract.
 */
export function contractType(contract: OutputContract | null): Type | null {
  if (contract === null) return null;
  if (isExactly(contract, ValueContract)) return contract.type;
  const fields = new Map<string, Type>();
  for (const [name, field] of contract.fields) {
    fields.set(name, field !== null ? field.type : Type.scalar(ValueKind.ANY));
  }
  return new Type({ kind: ValueKind.OBJECT, fields, required: new Set(fields.keys()) });
}
