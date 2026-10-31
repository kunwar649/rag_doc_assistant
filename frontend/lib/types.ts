// Shapes returned by the FastAPI backend (see src/rag/api.py and src/rag/pipeline.py).

export type Mode = "rerank" | "hybrid" | "vector" | "bm25";

export interface Source {
  n: number;
  source: string | null;
  page: number | null;
  citation: string;
  score: number;
  text: string;
}

export interface Usage {
  input: number;
  output: number;
}

export type AskEvent =
  | { type: "sources"; query: string; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; cited: number[]; provider: string | null; latency_ms: number; usage?: Usage | null }
  | { type: "error"; message: string };

export interface ApiConfig {
  top_k: number;
  max_upload_mb: number;
  extensions: string[];
  tracing: boolean;
}

export interface Library {
  documents: string[];
  chunks: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "streaming" | "done" | "stopped" | "error";
  question?: string;
  query?: string;
  sources?: Source[];
  cited?: number[];
  provider?: string | null;
  latencyMs?: number;
  usage?: Usage | null;
  error?: string;
}

export interface SearchSettings {
  mode: Mode;
  topK: number;
  source: string | null;
}
