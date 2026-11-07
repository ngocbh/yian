# Yian - The Agent Composer

> Yet Another Agent Composer

yian is a small declarative language, written in YAML, for flows of agents, code and tools that
people and LLMs write, check and improve. The graph you write doesn't change at run time: LLMs
fill in values and choose among the tools you grant.

- **A fixed graph you can read before it runs.** Data references wire the steps together.
- **Typed values, checked at load.** A flow whose parts don't fit fails before it calls a model.
- **An append-only run log.** Each step is logged before it's applied, so a run resumes after a crash.

## Goals and Non-Goals
- **Goals:** work that repeats, must be reliable and should improve across runs, such as standard
operating procedures, routines, reviews, triage and reports. Flows that people and LLMs both write,
read and fix.
- **Non-goals:** interactive coding agents, and flexible or one-off work. Use Codex,
Claude Code or OpenCode for that.

## Install

```console
$ npm install @ngocbh/yian
```

Or `bun add @ngocbh/yian`. yian runs on Node ≥ 22.18 or Bun ≥ 1.4. To put the `yian` command on
your PATH, install it globally with `npm i -g @ngocbh/yian`, or run it with `npx @ngocbh/yian`
or `bunx --bun @ngocbh/yian`. Plain `npx yian` fetches an unrelated package.

From a clone of the repository, `install.sh` puts that checkout's `yian` on your PATH. It installs
Bun if you have none and writes a `yian` wrapper into `~/.local/bin`, so the command runs your
checkout and follows `git pull`. `./uninstall.sh` takes it back out.

```console
$ ./install.sh
```

## Set up a model

yian has no built-in model. Save a login for a provider; [the docs](https://yian.readthedocs.io/models/)
cover the others:

```console
$ yian auth anthropic
```

## Run a flow

<!-- yian-test {"replies": ["Hello, Ada!"]} -->

```yaml title="hello.yaml" check=runnable
id: hello
name: hello
input:
  name: !str
output: str
kind: agent
prompt: Write a one-sentence greeting for ${name}.
```

```console
$ yian run hello.yaml --input name=Ada
Hello, Ada!
```

## Chat with Yian

Yian Chat is the first application built on Yian, serving as a way to dogfood the framework through real-world use. See [CLI](https://github.com/ngocbh/yian/tree/main/src/cli) and [CLI Flows](https://github.com/ngocbh/yian/tree/main/src/cli/flows) for example.

```console
$ yian chat
$ yian chat hello.yaml
```

`yian chat` opens Yian, yian's assistant, in the current folder. Yian runs, writes, edits and
improves flows with you, and asks before it writes a file or starts a flow. With a flow named,
the chat is about that flow. See [the CLI docs](https://yian.readthedocs.io/cli/).

## Documentation and license

Read the docs at [yian.readthedocs.io](https://yian.readthedocs.io/), starting with
[Get started](https://yian.readthedocs.io/get-started/). yian is licensed under Apache-2.0.
