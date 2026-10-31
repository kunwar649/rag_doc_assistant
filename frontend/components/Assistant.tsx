"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getConfig, getLibrary, streamAnswer, toHistory } from "@/lib/api";
import { load, save } from "@/lib/storage";
import type { ApiConfig, ChatMessage, Library, SearchSettings } from "@/lib/types";
import Composer from "./Composer";
import Conversation from "./Conversation";
import EvidencePane from "./EvidencePane";
import LibraryPanel from "./LibraryPanel";

const CHAT_KEY = "docassist.chat.v1";
const SETTINGS_KEY = "docassist.settings.v1";
const DEFAULT_SETTINGS: SearchSettings = { mode: "rerank", topK: 5, source: null };
const WIDE_QUERY = "(min-width: 64rem)"; // Tailwind's lg breakpoint, where the evidence pane is always visible

export type ApiStatus = "loading" | "ok" | "down";
export type Selection = { messageId: string; n: number } | null;

/** Config + library in one go; null means the API couldn't be reached. */
const fetchApiState = () =>
  Promise.all([getConfig(), getLibrary()]).then(
    ([config, library]) => ({ config, library }),
    () => null,
  );

const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

export default function Assistant() {
  const [config, setConfig] = useState<ApiConfig | null>(null);
  const [library, setLibrary] = useState<Library | null>(null);
  const [status, setStatus] = useState<ApiStatus>("loading");
  // An answer that was mid-stream when the page closed can't resume; mark it stopped.
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    load<ChatMessage[]>(CHAT_KEY, []).map((m) => (m.status === "streaming" ? { ...m, status: "stopped" } : m)),
  );
  const [settings, setSettings] = useState<SearchSettings>(() => ({
    ...DEFAULT_SETTINGS,
    ...load<Partial<SearchSettings>>(SETTINGS_KEY, {}),
  }));
  const [selection, setSelection] = useState<Selection>(null);
  const [libraryOpen, setLibraryOpen] = useState(false); // drawer on small screens
  const [evidenceOpen, setEvidenceOpen] = useState(false); // bottom sheet below the large breakpoint
  const abortRef = useRef<AbortController | null>(null);

  const busy = messages.some((m) => m.status === "streaming");
  // A saved "search in" filter may point to a file that has since been removed.
  const sourceFilter = settings.source && library?.documents.includes(settings.source) ? settings.source : null;

  const applyApiState = useCallback((state: { config: ApiConfig; library: Library } | null) => {
    if (state) {
      setConfig(state.config);
      setLibrary(state.library);
      setStatus("ok");
    } else {
      setStatus("down");
    }
  }, []);

  const refresh = useCallback(async () => applyApiState(await fetchApiState()), [applyApiState]);

  useEffect(() => {
    let live = true; // ignore a late response after unmount
    fetchApiState().then((state) => live && applyApiState(state));
    return () => {
      live = false;
    };
  }, [applyApiState]);

  useEffect(() => {
    if (!busy) save(CHAT_KEY, messages);
  }, [messages, busy]);

  useEffect(() => {
    save(SETTINGS_KEY, settings);
  }, [settings]);

  const patchMessage = useCallback(
    (id: string, patch: Partial<ChatMessage> | ((m: ChatMessage) => Partial<ChatMessage>)) =>
      setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m))),
    [],
  );

  async function ask(question: string) {
    const history = toHistory(messages);
    const reply: ChatMessage = { id: newId(), role: "assistant", content: "", status: "streaming", question, sources: [], cited: [] };
    setMessages((ms) => [...ms, { id: newId(), role: "user", content: question }, reply]);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamAnswer(
        question,
        history,
        { ...settings, source: sourceFilter },
        (event) => {
          switch (event.type) {
            case "sources":
              patchMessage(reply.id, { sources: event.sources, query: event.query });
              break;
            case "token":
              patchMessage(reply.id, (m) => ({ content: m.content + event.text }));
              break;
            case "done":
              patchMessage(reply.id, {
                status: "done",
                cited: event.cited,
                provider: event.provider,
                latencyMs: event.latency_ms,
                usage: event.usage ?? null,
              });
              // Open the evidence on the first passage the answer relies on.
              if (event.cited.length) setSelection({ messageId: reply.id, n: event.cited[0] });
              break;
            case "error":
              patchMessage(reply.id, { status: "error", error: event.message });
              break;
          }
        },
        controller.signal,
      );
      patchMessage(reply.id, (m) =>
        m.status === "streaming" ? { status: "error", error: "The connection closed before the answer finished." } : {},
      );
    } catch (err) {
      if (controller.signal.aborted) {
        patchMessage(reply.id, { status: "stopped" });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        patchMessage(reply.id, { status: "error", error: message });
        if (!(err instanceof ApiError)) void refresh(); // network failure: re-check whether the API is up
      }
    } finally {
      abortRef.current = null;
    }
  }

  // Wide screens show the evidence pane permanently; narrow ones open it as a bottom sheet.
  useEffect(() => {
    const wide = window.matchMedia(WIDE_QUERY);
    const onChange = () => wide.matches && setEvidenceOpen(false);
    wide.addEventListener("change", onChange);
    return () => wide.removeEventListener("change", onChange);
  }, []);

  function selectCitation(messageId: string, n: number) {
    setSelection({ messageId, n });
    if (!window.matchMedia(WIDE_QUERY).matches) setEvidenceOpen(true);
  }

  function newConversation() {
    abortRef.current?.abort();
    setMessages([]);
    setSelection(null);
    setLibraryOpen(false);
  }

  // The evidence pane follows the selected citation, or else the latest answer with sources.
  const evidenceMessage = useMemo(() => {
    if (selection) {
      const m = messages.find((x) => x.id === selection.messageId);
      if (m) return m;
    }
    return [...messages].reverse().find((m) => m.role === "assistant" && m.sources?.length) ?? null;
  }, [messages, selection]);
  const selectedN = selection && evidenceMessage?.id === selection.messageId ? selection.n : null;

  const libraryPanel = (
    <LibraryPanel
      library={library}
      config={config}
      status={status}
      settings={{ ...settings, source: sourceFilter }}
      onSettingsChange={setSettings}
      onUploaded={refresh}
      onNewConversation={newConversation}
      onClose={() => setLibraryOpen(false)}
    />
  );

  return (
    <div className="grid h-dvh grid-cols-1 md:grid-cols-[17rem_minmax(0,1fr)] lg:grid-cols-[17rem_minmax(0,1fr)_24rem]">
      <aside className="hidden min-h-0 border-r border-line md:flex">{libraryPanel}</aside>

      <main className="flex min-h-0 min-w-0 flex-col">
        <header className="flex items-center gap-2 border-b border-line px-4 py-3 md:px-8">
          <button
            type="button"
            className="rounded-md border border-line px-3 py-1.5 text-sm md:hidden"
            onClick={() => setLibraryOpen(true)}
          >
            Library
          </button>
          <h1 className="text-[15px] font-semibold tracking-tight">Document Assistant</h1>
          <button
            type="button"
            className="ml-auto rounded-md border border-line px-3 py-1.5 text-sm disabled:opacity-40 lg:hidden"
            onClick={() => setEvidenceOpen(true)}
            disabled={!evidenceMessage}
          >
            Evidence
          </button>
        </header>

        <Conversation
          messages={messages}
          library={library}
          status={status}
          selection={selection}
          onSelectCitation={selectCitation}
          onRetry={refresh}
        />

        <Composer
          busy={busy}
          disabledReason={
            status === "down"
              ? "Start the API to ask questions."
              : library && library.documents.length === 0
                ? "Add a document to the library first."
                : null
          }
          scopeLabel={sourceFilter ? `Searching ${sourceFilter}` : null}
          onAsk={ask}
          onStop={() => abortRef.current?.abort()}
        />
      </main>

      <aside className="hidden min-h-0 border-l border-line lg:flex">
        <EvidencePane message={evidenceMessage} selectedN={selectedN} onSelect={selectCitation} />
      </aside>

      {libraryOpen && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Library">
          <button type="button" aria-label="Close library" className="absolute inset-0 bg-black/40" onClick={() => setLibraryOpen(false)} />
          <div className="absolute inset-y-0 left-0 flex w-[min(20rem,88vw)] bg-panel shadow-xl">{libraryPanel}</div>
        </div>
      )}

      {evidenceOpen && evidenceMessage && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Evidence">
          <button type="button" aria-label="Close evidence" className="absolute inset-0 bg-black/40" onClick={() => setEvidenceOpen(false)} />
          <div className="absolute inset-x-0 bottom-0 flex h-[78dvh] rounded-t-xl bg-panel shadow-xl">
            <EvidencePane
              message={evidenceMessage}
              selectedN={selectedN}
              onSelect={selectCitation}
              onClose={() => setEvidenceOpen(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
