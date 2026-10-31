import type { AskEvent } from "./types";

/**
 * Incremental parser for Server-Sent Events.
 *
 * Network chunks don't line up with events: one chunk can hold half an event,
 * or three of them. Text is buffered until a blank line ends an event, then
 * every `data:` line in it is joined and parsed as JSON.
 */
export function createSSEParser(onEvent: (event: AskEvent) => void) {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk.replace(/\r\n/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = raw
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) continue;
      try {
        onEvent(JSON.parse(data) as AskEvent);
      } catch {
        // Ignore a malformed event rather than abandoning the whole answer.
      }
    }
  };
}
