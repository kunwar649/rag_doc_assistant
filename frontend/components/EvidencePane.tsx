"use client";

import { useEffect, useRef } from "react";
import { orderSources, sourceTitle } from "@/lib/citations";
import type { ChatMessage } from "@/lib/types";
import FileName from "./FileName";

interface Props {
  message: ChatMessage | null;
  selectedN: number | null;
  onSelect: (messageId: string, n: number) => void;
  onClose?: () => void;
}

/**
 * The passages behind an answer. Cited ones come first; the selected one is
 * shown in full and highlighted so the reader can check the claim against it.
 */
export default function EvidencePane({ message, selectedN, onSelect, onClose }: Props) {
  const listRef = useRef<HTMLOListElement>(null);

  useEffect(() => {
    if (selectedN === null) return;
    listRef.current?.querySelector(`[data-n="${selectedN}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedN, message?.id]);

  const sources = message?.sources ?? [];
  const cited = message?.cited ?? [];

  return (
    <div className="flex min-h-0 w-full flex-col bg-panel">
      <div className="flex items-start justify-between gap-3 px-5 pt-5">
        <div className="min-w-0">
          <h2 className="font-serif text-xl font-semibold">Evidence</h2>
          {message?.question && <p className="mt-1 line-clamp-2 text-sm text-muted">For: {message.question}</p>}
        </div>
        {onClose && (
          <button type="button" className="rounded-md px-2 py-1 text-sm text-muted" onClick={onClose}>
            Close
          </button>
        )}
      </div>

      {!message || sources.length === 0 ? (
        <p className="px-5 pt-4 text-sm text-muted">
          {message ? "No passages matched this question." : "Passages behind each answer appear here."}
        </p>
      ) : (
        <ol ref={listRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto px-5 py-4">
          {orderSources(sources, cited).map((s) => {
            const isCited = cited.includes(s.n);
            const selected = s.n === selectedN;
            return (
              <li key={s.n} data-n={s.n}>
                <button
                  type="button"
                  aria-expanded={selected}
                  onClick={() => onSelect(message.id, s.n)}
                  className={`w-full rounded-lg border p-3 text-left ${selected ? "border-teal" : "border-line hover:border-muted"}`}
                >
                  <span className="flex items-baseline gap-2">
                    <span
                      className={`shrink-0 rounded px-1.5 text-xs font-semibold ${isCited ? "bg-mark-soft text-ink" : "border border-line text-muted"}`}
                    >
                      {s.n}
                    </span>
                    <span className="min-w-0 text-sm font-medium [overflow-wrap:anywhere]">
                      <FileName name={sourceTitle(s)} />
                    </span>
                  </span>
                  <span className="mt-1 flex flex-wrap gap-x-3 text-xs text-muted">
                    {s.page != null && <span>Page {s.page}</span>}
                    <span>{isCited ? "Cited in the answer" : "Retrieved, not cited"}</span>
                    <span>Relevance {s.score.toFixed(2)}</span>
                  </span>
                  <span
                    className={`mt-2 whitespace-pre-wrap font-serif text-[0.95rem] leading-relaxed [overflow-wrap:anywhere] ${selected ? "block" : "line-clamp-3 text-muted"}`}
                  >
                    {selected ? <span className="passage-marked">{s.text}</span> : s.text}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
