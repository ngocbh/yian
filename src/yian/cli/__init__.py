"""The `yian` command-line interface — provider setup and a brokered run host.

`yian auth` lists the saved logins. `yian auth PROVIDER` signs in to any supported
provider (an API key, OpenAI subscription, Azure endpoint plus key, or no credential),
checks the model, and saves the credential and model together; `--default` and
`--remove` choose the default provider or delete a login. It owns prompts; provider
services own the protocol and credential storage.

`yian config check` checks a host `llm_config` layer read from stdin, and `yian
providers` lists the providers and their login kinds; each prints text or, with
`--json`, one `yian.cli.v1` document.

`yian check FLOW...` loads flows without running them and reports each as text or, with
`--json`, as one `yian.cli.v1` document. `yian skills check DIR...` does the same for
skill directories.

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
`mcp_config` reads the user's ``~/.yian/mcp.json`` and each ``--mcp-config`` file
(``mcpServers`` JSON) into unopened MCP servers; the controller passes the servers
the flow grants to the run unopened, as tool groups, and the run opens them after its
own checks. The controller opens servers itself only to probe for an unresolved
tool name, and closes those when the run ends.

Knows about: compose; compile metadata and LLM-cascade helpers; the package-root run
boundary; the package-root memory host; memory contracts and built-in backends;
events; node host-stop signals; typed values; tools; LLM API-key metadata;
the default snapshot in ``llm_clients.connection``; ``yian.defaults``; and peer CLI
modules. Never imports runtime scheduling internals or provider SDKs; owns no
graph mutation.
"""

from __future__ import annotations

import sys

from rich.text import Text
import typer

from yian.cli.chat.command import chat as _chat
from yian.cli.auth import auth as _auth
from yian.cli.auth import config_check as _config_check
from yian.cli.auth import providers as _providers
from yian.cli.run import check as _check
from yian.cli.run import err_console
from yian.cli.run import run as _run
from yian.cli.run import skills_check as _skills_check
from yian.cli.utils import print_json
from yian.defaults import DefaultsError, current

app = typer.Typer(
    name="yian",
    help="Yian - The Agent Composer. Run agent flows from the command line.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Check the `YIAN_<NAME>` defaults before any command runs.

    The callback also keeps `run` a named subcommand (a single-command Typer app
    otherwise collapses it, so `yian run flow.yaml` would read `run` as the flow
    path).

    A bad variable is reported on stderr and, when the arguments hold `--json`, also as a
    `defaults.invalid` error document on stdout.

    Raises:
        `typer.Exit`: With code 1 when a `YIAN_<NAME>` variable holds a value its
            default does not accept.
    """
    try:
        current()
    except DefaultsError as err:
        err_console.print(Text(str(err), style="red"))
        # Typer runs this callback before it parses the subcommand's options, so `--json`
        # is looked for in the raw arguments. Two limits: `CliRunner` arguments aren't in
        # `sys.argv`, and a `--json` after `--` also matches.
        if "--json" in sys.argv[1:]:
            print_json({"error": {"code": "defaults.invalid", "message": str(err)}})
        raise typer.Exit(code=1) from None


app.command("run")(_run)
app.command("chat")(_chat)
app.command("auth")(_auth)
app.command("check")(_check)
app.command("providers")(_providers)

config_app = typer.Typer(name="config", no_args_is_help=True, help="Check an llm_config layer.")
config_app.command("check")(_config_check)
app.add_typer(config_app, name="config")

skills_app = typer.Typer(name="skills", no_args_is_help=True, help="Check skill directories.")
skills_app.command("check")(_skills_check)
app.add_typer(skills_app, name="skills")


def main() -> None:
    """Console-script entry point (`yian = yian.cli:main`)."""
    app()
