"""Step 6 — the two-axis `kinds` dispatch tables (`KINDS`/`BINDING_FIELDS`).

6.0 scaffolded the empty tables + `KindSpec`; by 6.12 every author-written kind has a
`compose/kinds/<kind>.py` handle registered in BOTH tables. `start`/`end` are
loader-synthesized boundaries (never author-written), so they are NOT in `KINDS`. Adding a
new author kind without a kinds/ handle fails the registration ratchet below."""

from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

# The author-written surface kinds (mirrors docs/syntax.md; excludes the synthesized
# start/end boundaries). Adding a kind here without registering its handle fails below.
AUTHORED_KINDS = frozenset(
    {"agent", "code", "model", "tool", "case", "human_input", "wait", "call", "map", "loop"}
)


def test_kinds_tables_present():
    assert isinstance(KINDS, dict)
    assert isinstance(BINDING_FIELDS, dict)


def test_kindspec_fields():
    spec = KindSpec(descriptor_cls=int, parse=lambda r: None, build=lambda d, **k: (None, {}))
    assert spec.descriptor_cls is int
    assert callable(spec.parse)
    assert callable(spec.build)


def test_all_authored_kinds_registered():
    assert AUTHORED_KINDS <= set(KINDS), sorted(AUTHORED_KINDS - set(KINDS))
    assert AUTHORED_KINDS <= set(BINDING_FIELDS), sorted(AUTHORED_KINDS - set(BINDING_FIELDS))


def test_kinds_and_binding_fields_agree():
    # Every registered kind carries a binding-fields entry (possibly empty), and vice versa.
    assert set(KINDS) == set(BINDING_FIELDS)
