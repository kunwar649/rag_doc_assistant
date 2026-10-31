"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { API_URL } from "@/lib/api";
import { citationNumber, isRefusal, linkCitations, normalizeTypography } from "@/lib/citations";
import type { ChatMessage, Library } from "@/lib/types";
import type { ApiStatus, Selection } from "./Assistant";

interface Props {
  messages: ChatMessage[];
  library: Library | null;
  status: ApiStatus;
  selection: Selection;
  onSelectCitation: (messageId: string, n: number) => void;
  onRetry: () => void;
}

export default function Conversation({ messages, library, status, selection, onSelectCitation, onRetry }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const last = messages[messages.length - 1];

  // Follow the streaming answer, unless the reader has scrolled up to read something.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages.length, last?.content, last?.status]);

  return (
    <div
      ref={scrollRef}
      className="min-h-0 flex-1 overflow-y-auto px-4 py-6 md:px-8"
      onScroll={(e) => {
        const el = e.currentTarget;
        stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}
    >
      {status === "down" && (
        <div className="mx-auto mb-6 max-w-[68ch] rounded-lg border border-danger/40 p-4 text-sm" role="alert">
          <p className="font-medium text-danger">Can&apos;t reach the API at {API_URL}.</p>
          <p className="mt-1 text-muted">Start it from the project folder, then try again:</p>
          <code className="mt-2 block overflow-x-auto rounded bg-paper px-2 py-1.5 text-xs">uvicorn rag.api:app --app-dir src --reload</code>
          <button type="button" className="mt-3 rounded-md bg-teal px-3 py-1.5 font-medium text-teal-ink" onClick={onRetry}>
            Try again
          </button>
        </div>
      )}

      {messages.length === 0 ? (
        <EmptyState library={library} />
      ) : (
        <ol className="mx-auto flex max-w-[68ch] flex-col gap-8" aria-label="Conversation">
          {messages.map((m) =>
            m.role === "user" ? (
              <li key={m.id} className="border-l-2 border-teal pl-4 font-medium leading-snug [overflow-wrap:anywhere]">
                {m.content}
              </li>
            ) : (
              <li key={m.id}>
                <Answer message={m} selection={selection} onSelectCitation={onSelectCitation} />
              </li>
            ),
          )}
        </ol>
      )}
    </div>
  );
}

function EmptyState({ library }: { library: Library | null }) {
  const count = library?.documents.length ?? 0;
  return (
    <div className="mx-auto mt-[12vh] max-w-[46ch]">
      <h2 className="font-serif text-3xl font-semibold leading-tight md:text-4xl">Ask about your documents</h2>
      <p className="mt-4 text-muted">
        {library && count === 0
          ? "Add a PDF, Word or text file to the library to get started."
          : "Every answer cites the passages it used. Select a highlighted number to read the passage it came from."}
      </p>
      {count > 0 && (
        <p className="mt-6 text-sm text-muted">
          Searching {count} {count === 1 ? "document" : "documents"}: {library?.documents.join(", ")}
        </p>
      )}
    </div>
  );
}

function Answer({ message: m, selection, onSelectCitation }: { message: ChatMessage; selection: Selection; onSelectCitation: Props["onSelectCitation"] }) {
  const sourceCount = m.sources?.length ?? 0;
  const streaming = m.status === "streaming";
  const refusal = m.status === "done" && isRefusal(m.content);

  return (
    <article aria-busy={streaming}>
      {streaming && !m.content && (
        <p className="text-sm text-muted" role="status">
          {sourceCount ? `Reading ${sourceCount} passages…` : "Searching your documents…"}
        </p>
      )}

      {m.content && (
        <div className={`answer ${refusal ? "italic text-muted" : ""} ${streaming ? "caret" : ""}`}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a({ href, children }) {
                const n = citationNumber(href);
                if (n === null) {
                  return (
                    <a href={href} target="_blank" rel="noopener noreferrer" className="text-teal underline">
                      {children}
                    </a>
                  );
                }
                const active = selection?.messageId === m.id && selection.n === n;
                return (
                  <button
                    type="button"
                    className="cite"
                    aria-pressed={active}
                    aria-label={`Show source ${n}`}
                    onClick={() => onSelectCitation(m.id, n)}
                  >
                    {n}
                  </button>
                );
              },
            }}
          >
            {linkCitations(normalizeTypography(m.content), sourceCount)}
          </ReactMarkdown>
        </div>
      )}

      {m.status === "stopped" && <p className="mt-2 text-sm text-muted">Stopped before the answer finished.</p>}
      {m.status === "error" && (
        <p className="mt-2 text-sm text-danger" role="alert">
          The answer couldn&apos;t be completed: {m.error}
        </p>
      )}

      {m.status === "done" && (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
          {m.query && m.query !== m.question && <span>Searched for “{normalizeTypography(m.query)}”</span>}
          {m.provider && <span>{m.provider.split(":").slice(1).join(":") || m.provider}</span>}
          {m.latencyMs !== undefined && <span>{(m.latencyMs / 1000).toFixed(1)} s</span>}
          {m.usage && (
            <span>
              {m.usage.input.toLocaleString()} tokens in, {m.usage.output.toLocaleString()} out
            </span>
          )}
          {!m.provider && sourceCount === 0 && <span>No matching passages found</span>}
        </div>
      )}
    </article>
  );
}
