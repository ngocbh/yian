"""`yian chat` — the interactive REPL, dogfooded as an ordinary Yian flow.

A CLI-host concern, not engine core. This package bundles the pieces of the chat
surface: the turn-taking flow (`chat.yaml`) — which rejoins a flow-owned exit record or
normal transcript through output bindings — the workspace-confined flow-op
tools the Yian assistant calls (`tools.py`), and the Typer subcommand that owns the
broker/controller lifecycle (`command.py`). Typed `/exit` and `/quit` are ordinary values
routed by the flow; only EOF is host-normalized. The controller stop context gives a
nested `run_flow` tool a one-way child event: host cancellation flows inward, while a
child failure remains an ordinary tool result. Each top-level attempt gets a newly
minted, reported durable log; a failed-turn restart never reuses its predecessor's
authority. Nothing here changes engine core.

Knows about: peer CLI modules, compose, the package-root run boundary, tools, and events.
Never imports: runtime scheduling internals or provider SDKs; owns no graph mutation.
"""
