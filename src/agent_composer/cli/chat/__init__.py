"""`ac chat` — the interactive REPL, dogfooded as an ordinary Agent Composer flow.

A CLI-host concern, not engine core. This package bundles the pieces of the chat
surface: the turn-taking flow (`chat.yaml`) — which rejoins a flow-owned exit record or
normal transcript through defaulted output bindings — the workspace-confined flow-op
tools the composer assistant calls (`tools.py`), and the Typer subcommand that owns the
broker/controller lifecycle (`command.py`). Typed `/exit` and `/quit` are ordinary values
routed by the flow; only EOF is host-normalized. Nothing here changes engine core.
"""
