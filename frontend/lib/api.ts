import { createSSEParser } from "./sse";
import type { ApiConfig, AskEvent, ChatMessage, Library, SearchSettings } from "./types";

// Inlined at build time (see .env.local.example). Defaults to the local FastAPI server.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    // not JSON; keep the status text
  }
  return new ApiError(detail, res.status);
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw await errorFrom(res);
  return res.json() as Promise<T>;
}

export const getConfig = () => getJSON<ApiConfig>("/config");
export const getLibrary = () => getJSON<Library>("/documents");

export async function uploadFile(file: File): Promise<{ file: string; chunks: number }> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_URL}/upload`, { method: "POST", body });
  if (!res.ok) throw await errorFrom(res);
  return res.json();
}

/** Prior turns in the shape the API expects: finished user/assistant text only. */
export function toHistory(messages: ChatMessage[]) {
  return messages
    .filter((m) => m.role === "user" || m.status === "done")
    .map((m) => ({ role: m.role, content: m.content }));
}

/** POST /ask with streaming on; calls onEvent for each server event until the stream ends. */
export async function streamAnswer(
  question: string,
  history: { role: "user" | "assistant"; content: string }[],
  settings: SearchSettings,
  onEvent: (event: AskEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      history,
      mode: settings.mode,
      top_k: settings.topK,
      source: settings.source,
      stream: true,
    }),
    signal,
  });
  if (!res.ok) throw await errorFrom(res);
  if (!res.body) throw new ApiError("The server sent no response body.");

  const parse = createSSEParser(onEvent);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parse(decoder.decode(value, { stream: true }));
  }
  parse(decoder.decode() + "\n\n"); // flush anything left in the buffer
}
