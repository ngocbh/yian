/**
 * The type system's public surface, as Python's `yian.typesys`: typed values, the type
 * language and the variable pool.
 *
 * Python's `AnyValue`, `DateValue` and `DateTimeValue` carriers are all `TypedValue` here;
 * `Value`, `ValueRecord` and `PoolWire` are exported because public signatures return them.
 */

export type { PoolWire } from "./pool.ts";
export { VariablePool } from "./pool.ts";
export type { RecordDef, TypeRegistry, VariantDef } from "./types.ts";
export { BUILTIN_TYPE_NAMES, parseType, resolveType, typeFor } from "./types.ts";
export type {
  FieldAbsent,
  FieldPartial,
  FieldRead,
  StorageKind,
  Value,
  ValueRecord,
} from "./values.ts";
export {
  buildValue,
  buildValueAs,
  canonicalTypeLabel,
  canonicalUnionOrder,
  FileRef,
  listKindForElement,
  normalizeFiles,
  restrictUnion,
  Type,
  TypeCheckError,
  TypedValue,
  typesOverlap,
  unionField,
  unionMemberForTag,
  unionTagFields,
  ValueKind,
} from "./values.ts";
