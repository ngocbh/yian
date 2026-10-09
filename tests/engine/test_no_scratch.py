"""Structural-purity lock-in: scratch is fully eliminated.

AGENT owns its live loop, while HUMAN_INPUT and WAIT block and return ``Output``.
No node receives an engine-delivered continuation value or reads scratch. This pins
that the field + helpers are gone and that no node source even references the word
`scratch`.
"""

import importlib
import inspect
import pkgutil

from yian.typesys.pool import VariablePool


def test_pool_has_no_scratch():
    p = VariablePool()
    assert not hasattr(p, "scratch")
    assert not hasattr(p, "scratch_get") and not hasattr(p, "scratch_set")


def test_scratch_modules_gone():
    for mod in ("yian.nodes.scratch_cap",
                "yian.nodes.agent.scratch"):
        try:
            importlib.import_module(mod)
            assert False, f"{mod} still exists"
        except ModuleNotFoundError:
            pass


def test_no_node_source_reads_scratch():
    # walk the nodes package source; no node run/source references 'scratch'
    import yian.nodes as nodes_pkg

    seen = []
    for _, name, _ in pkgutil.walk_packages(nodes_pkg.__path__, nodes_pkg.__name__ + "."):
        src = inspect.getsource(importlib.import_module(name))
        if "scratch" in src:
            seen.append(name)
    assert seen == [], f"scratch still referenced in: {seen}"
