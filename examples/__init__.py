"""Marks `examples/` as an importable package so flow `code:` refs resolve.

The `trading-volume-spike` flow wires deterministic steps as `code:
examples.trading_fns:<fn>`; that import only resolves when `examples` is a package on
`sys.path`. Run such flows from the repo root with the root importable, e.g.
`PYTHONPATH=$PWD composer run examples/trading-volume-spike.yaml`.
"""
