/**
 * Chat flows for the tests of `yian chat` and of `yian run` handing a terminal chat to it: a loop
 * of turns over an `Inbox(str)` input, each turn taking every line sent and replying.
 */

/** A chat turn per line: take everything sent, reply with `reply` (YAML lines, 6 spaces in). */
export function turnFlow(reply: string, { input = "", nodes = "" } = {}): string {
  return `id: echo_chat
name: echo_chat
input:
  messages: !Inbox(str)
${input}defs:
  turn:
    input:
      said: !list(str)
      closed: !bool
      messages: !Inbox(str)
${input.replace(/^ {2}/gm, "      ")}    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.messages}
          take: all
${nodes}${reply}    output:
      said: \${input.said + heard.output.events}
      closed: \${heard.output.closed}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      said: []
      closed: false
      messages: \${input.messages}
${input.replace(/^ {2}(\w+):.*$/gm, "      $1: ${input.$1}")}    while: not closed
output: \${chat.output.said}
`;
}

export const ECHO_REPLY = `      reply:
        kind: code
        input:
          events: \${heard.output.events}
        output: str
        code: |
          return "echo: " + inputs["events"].join(" | ");
`;

/** A chat that echoes the lines each turn took: `echo: a | b`. */
export const ECHO = turnFlow(ECHO_REPLY);
