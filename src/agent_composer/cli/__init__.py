"""The `ac` command-line interface — a brokered host over the run boundary.

`ac run FLOW.yaml` loads a Compose flow, gathers fresh inputs (flags first, then
an interactive prompt for every missing declaration), selects and reports a
durable run-log path, and drives it to a terminal. Optional/defaulted prompts can
be omitted, leaving START to apply a declared default or retain optional absence.
``--resume`` instead recovers an existing path without gathering fresh inputs or
model configuration. The main thread owns terminal prompts/renders while a
controller drives the flow;
blocking HUMAN_INPUT/``ask_user`` calls cross that broker. WAIT remains live in
its worker until its deadline. The CLI owns path policy, not engine scheduling
or durable state-transition logic.

Knows about: compose; compile metadata and LLM-cascade helpers; the package-root run
boundary; events; node host-stop signals; typed values; tools; LLM API-key metadata;
process defaults in ``_settings``; and peer CLI modules. Never imports runtime scheduling
internals or provider SDKs; owns no graph mutation.
"""

from __future__ import annotations

import typer

from agent_composer.cli.chat.command import chat as _chat
from agent_composer.cli.run import run as _run

app = typer.Typer(
    name="ac",
    help="Agent Composer — run agent flows from the command line.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Keep `run` a named subcommand (a single-command Typer app otherwise collapses
    it, so `ac run flow.yaml` would read `run` as the flow path)."""


app.command("run")(_run)
app.command("chat")(_chat)


def main() -> None:
    """Console-script entry point (`ac = agent_composer.cli:main`)."""
    app()
