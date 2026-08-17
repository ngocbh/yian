"""Seed flow gallery — Compose-YAML example flows + their CODE callables.

Data + `fns.py`: one `.yaml` per shipped example flow (read by
`agent_composer.load_flow`), the negative `errors/` fixtures, future negative fixtures under
`_future/errors/`, and `fns.py` (the `module:function` targets the CODE nodes
reference). `__init__.py` makes `tests.seeds.fns` importable and lets
`importlib.resources` find the YAML files as package resources.
"""
