"""The `yian` command-line interface — provider setup and a brokered run host.

`yian auth openai` stages subscription or API-key authentication, presents available
models, and saves the selected method and default model together. It owns prompts;
provider services own the protocol and credential storage.

`yian run FLOW.yaml` loads a Compose flow, gathers fresh inputs from flags and an
available interactive prompt, selects and reports a durable run-log path, and drives
it to a terminal. Without an interactive stdin, optional/defaulted declarations stay
absent for START to resolve and missing required declarations fail before provider or
run-log setup.
``--resume`` instead recovers an existing path without gathering fresh inputs or
model configuration. The main thread owns terminal prompts/renders while a
controller drives the flow;
blocking HUMAN_INPUT/``ask_user`` calls cross that broker. WAIT remains live in
its worker until its deadline. The CLI owns path policy, not engine scheduling
or durable state-transition logic. `memory` turns ``--memory-root`` and
``--memory-storage KEY=PATH`` into a `MemoryRegistry` from
`yian.build_memories`; the controller owns its lifetime, after worker
drain. Storage paths stay host-only. A terminal resume never opens memory.

Knows about: compose; compile metadata and LLM-cascade helpers; the package-root run
boundary; the package-root memory host; memory contracts and built-in backends;
events; node host-stop signals; typed values; tools; LLM API-key metadata;
the default snapshot in ``llm_clients.connection``; and peer CLI modules. Never
imports runtime scheduling internals or provider SDKs; owns no graph mutation.
"""

from __future__ import annotations

import typer

from yian.cli.chat.command import chat as _chat
from yian.cli.auth import auth as _auth
from yian.cli.run import run as _run

app = typer.Typer(
    name="yian",
    help="Yian - The Agent Composer. Run agent flows from the command line.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Keep `run` a named subcommand (a single-command Typer app otherwise collapses
    it, so `yian run flow.yaml` would read `run` as the flow path)."""


app.command("run")(_run)
app.command("chat")(_chat)
app.command("auth")(_auth)


def main() -> None:
    """Console-script entry point (`yian = yian.cli:main`)."""
    app()
