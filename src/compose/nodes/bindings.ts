/**
 * The kind-blind binding-field walk: one place for "which fields are bindings".
 *
 * A node descriptor carries `${...}` binding text in the fields its kind's `bindingFields`
 * declares (`inputs`, plus `over` on mapped calls). Inline-call desugar and case-output expansion
 * rewrite that text in place through this table-driven walk over `BINDING_FIELDS[kind]`, so
 * changing a kind's binding surface is a local edit in its `compose/nodes/<kind>.ts`.
 *
 * Knows about: the sibling kind tables, nodes contracts and typesys values.
 * Never imports: the desugar passes that call it.
 */

import { ValueTemplate, mapSourceStrings } from "../../nodes/contracts.ts";
import { isRecord } from "../../typesys/values.ts";
import type { NodeDescriptor } from "./base.ts";
import { BINDING_FIELDS } from "./index.ts";

/** Whether `value` is a binding source the walk maps: a string or a `ValueTemplate`. */
function isSource(value: unknown): value is string | ValueTemplate {
  return typeof value === "string" || value instanceof ValueTemplate;
}

/**
 * Apply `mapper(field, key, source)` to each source in `desc`'s binding fields.
 *
 * The fields are `BINDING_FIELDS.get(desc.kind)`. A record field (e.g. `inputs`) supplies
 * `("inputs", bindingName, source)` for each value; a scalar field (e.g. `over`) supplies
 * `("over", "over", source)`, so the field tells an `input: {over: ...}` key from a scalar
 * `over:`. A source is a string or a `ValueTemplate`, passed whole; any other value is left.
 *
 * @param desc The descriptor to walk.
 * @param mapper Maps one source given its owning field and binding key; it may throw.
 * @returns `desc.replace(updates)` when a field was visited, else `desc` itself.
 */
export function mapBindingSources<D extends NodeDescriptor>(
  desc: D,
  mapper: (field: string, key: string, source: string | ValueTemplate) => unknown,
): D {
  const fields = BINDING_FIELDS.get(desc.kind) ?? [];
  const updates: Record<string, unknown> = {};
  for (const name of fields) {
    const current: unknown = Reflect.get(desc, name);
    if (isRecord(current)) {
      updates[name] = Object.fromEntries(
        Object.entries(current).map(([key, value]) => [key, isSource(value) ? mapper(name, key, value) : value]),
      );
    } else if (isSource(current)) {
      updates[name] = mapper(name, name, current);
    }
    // null or a literal scalar: nothing to rewrite.
  }
  return Object.keys(updates).length > 0 ? desc.replace(updates) : desc;
}

/**
 * Apply `mapper(field, key, text)` to each template string in `desc`'s binding fields.
 *
 * Walks the same sources as `mapBindingSources`: a string source is one text, and a
 * `ValueTemplate` supplies each leaf string under its binding's field and key.
 *
 * @param desc The descriptor to walk.
 * @param mapper Maps one template string given its owning field and binding key; it may throw.
 * @returns `desc.replace(updates)` when a field was visited, else `desc` itself.
 */
export function mapBindingStrings<D extends NodeDescriptor>(
  desc: D,
  mapper: (field: string, key: string, text: string) => unknown,
): D {
  return mapBindingSources(desc, (field, key, source) => mapSourceStrings(source, (text) => mapper(field, key, text)));
}
