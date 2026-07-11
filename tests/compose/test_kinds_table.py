"""Step 6.0 scaffold: the `kinds` package exists with the two dispatch tables."""

from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS


def test_kinds_tables_present_and_empty_at_scaffold():
    # The two-axis dispatch tables exist and start empty (kinds register during the rollout).
    assert isinstance(KINDS, dict)
    assert isinstance(BINDING_FIELDS, dict)


def test_kindspec_fields():
    spec = KindSpec(descriptor_cls=int, parse=lambda r: None, build=lambda d, **k: (None, {}))
    assert spec.descriptor_cls is int
    assert callable(spec.parse)
    assert callable(spec.build)
