# Yian - The Agent Composer

> Yet Another Agent Composer

yian is a small declarative language, written in YAML, for flows of agents, code and tools that
people and LLMs write, check and improve. The graph you write doesn't change at run time: LLMs
fill in values and choose among the tools you grant.

- **A fixed graph you can read before it runs.** Data references wire the steps together.
- **Typed values, checked at load.** A flow whose parts don't fit fails before it calls a model.
- **An append-only run log.** Each step is logged before it's applied, so a run resumes after a crash.

**Goals:** work that repeats, must be reliable and should improve across runs, such as SOPs and
routines. **Non-goals:** interactive coding agents. Flexible or one-off work belongs in Codex,
Claude Code or OpenCode.

## Install

```console
$ npm install @ngocbh/yian
```

Or `bun add @ngocbh/yian`. yian runs on Node ≥ 22.18 or Bun ≥ 1.4. To put the `yian` command on
your PATH, install it globally with `npm i -g @ngocbh/yian`, or run it with `npx @ngocbh/yian`
or `bunx @ngocbh/yian`. Plain `npx yian` fetches an unrelated package.

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

## Documentation and license

Read the docs at [yian.readthedocs.io](https://yian.readthedocs.io/), starting with
[Get started](https://yian.readthedocs.io/get-started/). yian is licensed under Apache-2.0.
