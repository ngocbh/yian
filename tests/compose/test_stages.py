"""Step 7 — the `compose/stages/` graph-stage axis: cross-node assembly passes.

7.0 scaffold: the package exists with its charter. Later sub-tasks (7.1+) populate it with
one relocated assembly pass each, driven off the span tree's `.range`/`field_ranges`."""

import agent_composer.compose.stages as stages


def test_stages_package_imports():
    assert stages.__doc__ is not None
