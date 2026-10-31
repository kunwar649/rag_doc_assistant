import type { Source } from "./types";

export const NO_ANSWER = "I couldn't find this in the provided documents.";

/**
 * Turn citation markers like [2] into markdown links (#cite-2) so the markdown
 * renderer can swap them for clickable buttons. Real links such as [text](url)
 * are left alone, and so are numbers the answer has no source for.
 */
export function linkCitations(text: string, sourceCount: number): string {
  return text.replace(/\[(\d+)\](?!\()/g, (match, n: string) => {
    const num = Number(n);
    return num >= 1 && num <= sourceCount ? `[${n}](#cite-${n})` : match;
  });
}

export function citationNumber(href: string | undefined): number | null {
  const m = href?.match(/^#cite-(\d+)$/);
  return m ? Number(m[1]) : null;
}

/** Cited sources first, in the order the answer cites them, then the rest by rank. */
export function orderSources(sources: Source[], cited: number[]): Source[] {
  const byN = new Map(sources.map((s) => [s.n, s]));
  const first = cited.map((n) => byN.get(n)).filter((s): s is Source => Boolean(s));
  return [...first, ...sources.filter((s) => !cited.includes(s.n))];
}

export function sourceTitle(s: Source): string {
  return s.source ?? "Unknown file";
}

export function isRefusal(text: string): boolean {
  return text.trim() === NO_ANSWER;
}

/**
 * File names like "Tennessee_Adapter_Detailed_Explanation.pdf" have no spaces,
 * so browsers break them mid-word. Split after _ . - so they wrap at natural points;
 * the component inserts a <wbr> after each part.
 */
export function breakParts(name: string): string[] {
  return name.split(/(?<=[_.\-/])/);
}

/**
 * Narrow/non-breaking spaces and non-breaking hyphens that models emit but these fonts
 * lack. The API already cleans new answers; this also fixes answers saved earlier.
 */
export function normalizeTypography(text: string): string {
  return text.replace(/[    ]/g, " ").replace(/‑/g, "-");
}
