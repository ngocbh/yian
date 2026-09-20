"""Private supervised access to the pinned official Codex Python SDK.

This leaf owns a finite-JSON process boundary, an isolated SDK worker, and its
parent-side lifecycle. Composer keeps tools and publication callbacks in the
calling process; this package owns neither agent policy nor retained history.

Knows about:   the standard library and the optional official openai_codex SDK.
Never imports: nodes, memory, runtime, compile, compose, run, CLI, or tool registries.
"""
