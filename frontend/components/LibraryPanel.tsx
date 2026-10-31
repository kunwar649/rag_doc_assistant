"use client";

import { useRef, useState } from "react";
import { uploadFile } from "@/lib/api";
import type { ApiConfig, Library, Mode, SearchSettings } from "@/lib/types";
import type { ApiStatus } from "./Assistant";
import FileName from "./FileName";

const MODES: { value: Mode; label: string; help: string }[] = [
  { value: "rerank", label: "Best match", help: "Searches by meaning and keywords, then re-reads the top 20 passages to pick the best. Most accurate; about a second slower." },
  { value: "hybrid", label: "Meaning and keywords", help: "Combines both searches without the re-reading step." },
  { value: "vector", label: "Meaning only", help: "Finds passages that say the same thing in other words. Can miss exact codes and names." },
  { value: "bm25", label: "Keywords only", help: "Finds passages containing your exact words. Misses rewording." },
];

type UploadState = { name: string; state: "uploading" | "done" | "failed"; note?: string };

interface Props {
  library: Library | null;
  config: ApiConfig | null;
  status: ApiStatus;
  settings: SearchSettings;
  onSettingsChange: (s: SearchSettings) => void;
  onUploaded: () => Promise<void>;
  onNewConversation: () => void;
  onClose: () => void;
}

export default function LibraryPanel({ library, config, status, settings, onSettingsChange, onUploaded, onNewConversation, onClose }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploads, setUploads] = useState<UploadState[]>([]);
  const [dragging, setDragging] = useState(false);
  const accept = (config?.extensions ?? [".pdf", ".docx", ".txt", ".md"]).join(",");
  const maxMb = config?.max_upload_mb ?? 20;
  const mode = MODES.find((m) => m.value === settings.mode) ?? MODES[0];

  async function handleFiles(files: FileList | File[]) {
    const list = Array.from(files);
    if (!list.length) return;
    setUploads(list.map((f) => ({ name: f.name, state: "uploading" })));
    // One at a time: each file is embedded on the server, and parallel uploads would compete for the CPU.
    for (const file of list) {
      const update = (patch: Partial<UploadState>) =>
        setUploads((us) => us.map((u) => (u.name === file.name ? { ...u, ...patch } : u)));
      if (file.size > maxMb * 1024 * 1024) {
        update({ state: "failed", note: `larger than ${maxMb} MB` });
        continue;
      }
      try {
        const res = await uploadFile(file);
        update({ state: "done", note: `${res.chunks} passages` });
      } catch (err) {
        update({ state: "failed", note: err instanceof Error ? err.message : "upload failed" });
      }
    }
    await onUploaded();
  }

  return (
    <div className="flex min-h-0 w-full flex-col bg-panel">
      <div className="flex items-center justify-between px-5 pt-5">
        <h2 className="font-serif text-xl font-semibold">Library</h2>
        <button type="button" className="rounded-md px-2 py-1 text-sm text-muted md:hidden" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-5">
        {status === "down" ? (
          <p className="mt-2 text-sm text-danger">The API isn&apos;t reachable, so the library can&apos;t load.</p>
        ) : !library ? (
          <p className="mt-2 text-sm text-muted">Loading documents…</p>
        ) : (
          <>
            <p className="mt-1 text-sm text-muted">
              {library.documents.length} {library.documents.length === 1 ? "document" : "documents"}, {library.chunks} passages
            </p>
            <ul className="mt-3 space-y-1">
              {library.documents.map((name) => (
                <li key={name} className="text-sm leading-snug [overflow-wrap:anywhere]" title={name}>
                  <FileName name={name} />
                </li>
              ))}
            </ul>
          </>
        )}

        <div
          className={`mt-4 rounded-lg border border-dashed p-4 text-center text-sm transition-colors ${
            dragging ? "border-teal bg-mark-soft/40" : "border-line"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void handleFiles(e.dataTransfer.files);
          }}
        >
          <p className="text-muted">Drop files here or</p>
          <button
            type="button"
            className="mt-2 rounded-md bg-teal px-3 py-1.5 font-medium text-teal-ink disabled:opacity-40"
            onClick={() => inputRef.current?.click()}
            disabled={status !== "ok"}
          >
            Add documents
          </button>
          <p className="mt-2 text-xs text-muted">PDF, Word, text or Markdown, up to {maxMb} MB each</p>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={accept}
            className="sr-only"
            onChange={(e) => {
              if (e.target.files) void handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        {uploads.length > 0 && (
          <ul className="mt-3 space-y-1 text-sm" aria-live="polite">
            {uploads.map((u) => (
              <li key={u.name} className="[overflow-wrap:anywhere]">
                <span className={u.state === "failed" ? "text-danger" : u.state === "done" ? "text-teal" : "text-muted"}>
                  {u.state === "uploading" ? "Indexing" : u.state === "done" ? "Added" : "Not added"}
                </span>{" "}
                {u.name}
                {u.note && <span className="text-muted"> ({u.note})</span>}
              </li>
            ))}
          </ul>
        )}

        <h2 className="mt-8 font-serif text-xl font-semibold">Search</h2>

        <label className="mt-3 block text-sm font-medium" htmlFor="mode">
          Method
        </label>
        <select
          id="mode"
          className="mt-1 w-full rounded-md border border-line bg-panel px-2 py-1.5 text-sm"
          value={settings.mode}
          onChange={(e) => onSettingsChange({ ...settings, mode: e.target.value as Mode })}
        >
          {MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs leading-relaxed text-muted">{mode.help}</p>

        <label className="mt-4 block text-sm font-medium" htmlFor="topk">
          Passages per answer: {settings.topK}
        </label>
        <input
          id="topk"
          type="range"
          min={1}
          max={10}
          value={settings.topK}
          onChange={(e) => onSettingsChange({ ...settings, topK: Number(e.target.value) })}
          className="mt-2 w-full accent-[var(--teal)]"
        />

        <label className="mt-4 block text-sm font-medium" htmlFor="scope">
          Search in
        </label>
        <select
          id="scope"
          className="mt-1 w-full rounded-md border border-line bg-panel px-2 py-1.5 text-sm"
          value={settings.source ?? ""}
          onChange={(e) => onSettingsChange({ ...settings, source: e.target.value || null })}
        >
          <option value="">All documents</option>
          {library?.documents.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      <div className="border-t border-line px-5 py-3">
        <button type="button" className="w-full rounded-md border border-line px-3 py-1.5 text-sm hover:bg-paper" onClick={onNewConversation}>
          New conversation
        </button>
        {config && <p className="mt-2 text-xs text-muted">Tracing {config.tracing ? "on (Langfuse)" : "off"}</p>}
      </div>
    </div>
  );
}
