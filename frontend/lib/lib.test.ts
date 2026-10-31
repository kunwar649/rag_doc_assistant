import { describe, expect, it } from "vitest";
import { breakParts, citationNumber, isRefusal, linkCitations, NO_ANSWER, normalizeTypography, orderSources } from "./citations";
import { createSSEParser } from "./sse";
import type { AskEvent, Source } from "./types";

describe("createSSEParser", () => {
  it("handles events split across network chunks and several events in one chunk", () => {
    const events: AskEvent[] = [];
    const parse = createSSEParser((e) => events.push(e));
    parse('data: {"type":"token","te');
    parse('xt":"Hel"}\n\ndata: {"type":"token","text":"lo"}\n\ndata: {"type":"do');
    expect(events).toHaveLength(2);
    parse('ne","cited":[1],"provider":"groq:m","latency_ms":5}\n\n');
    expect(events.map((e) => e.type)).toEqual(["token", "token", "done"]);
    expect(events[1]).toEqual({ type: "token", text: "lo" });
  });

  it("accepts CRLF line endings and skips malformed events", () => {
    const events: AskEvent[] = [];
    const parse = createSSEParser((e) => events.push(e));
    parse('data: not json\r\n\r\ndata: {"type":"token","text":"ok"}\r\n\r\n');
    expect(events).toEqual([{ type: "token", text: "ok" }]);
  });
});

describe("citations", () => {
  it("links in-range citations only and leaves real links alone", () => {
    expect(linkCitations("See [1] and [2][3]. Not [9]. A [link](http://x).", 3)).toBe(
      "See [1](#cite-1) and [2](#cite-2)[3](#cite-3). Not [9]. A [link](http://x).",
    );
  });

  it("parses citation hrefs", () => {
    expect(citationNumber("#cite-4")).toBe(4);
    expect(citationNumber("https://example.com")).toBeNull();
    expect(citationNumber(undefined)).toBeNull();
  });

  it("orders cited sources first", () => {
    const s = (n: number): Source => ({ n, source: "a.pdf", page: n, citation: `a.pdf (p. ${n})`, score: 1, text: "" });
    expect(orderSources([s(1), s(2), s(3)], [3, 1]).map((x) => x.n)).toEqual([3, 1, 2]);
  });

  it("detects the refusal answer", () => {
    expect(isRefusal(` ${NO_ANSWER} `)).toBe(true);
    expect(isRefusal("Exit code 2 means [1].")).toBe(false);
  });
});


describe("breakParts", () => {
  it("splits file names after separators without losing characters", () => {
    const name = "Tennessee_Adapter_Detailed_Explanation.pdf";
    expect(breakParts(name)).toEqual(["Tennessee_", "Adapter_", "Detailed_", "Explanation.", "pdf"]);
    expect(breakParts(name).join("")).toBe(name);
  });
});

describe("normalizeTypography", () => {
  it("replaces spaces and hyphens the fonts can't draw", () => {
    expect(normalizeTypography("exit code\u202f2, re\u2011rank")).toBe("exit code 2, re-rank");
  });
});
