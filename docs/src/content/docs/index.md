---
title: "yian"
description: "Readable YAML flows of agents, code and tools that people and LLMs write, check and improve."
---

yian runs flows: YAML files that wire agents, code and tools into a fixed graph you can read
before it runs. This page says what yian is for and where to go next.

:::caution[Draft]
This page is being written. It will cover: what yian is, its goals and non-goals, a 15-line
example, and where to go next.
:::

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
