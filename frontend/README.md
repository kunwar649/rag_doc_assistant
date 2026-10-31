# Frontend (Next.js)

A React chat interface for the RAG Document Assistant. It talks to the FastAPI backend
in the project root and adds an **evidence pane**: select a citation in an answer to read
the passage it came from, highlighted.

## Run it

You need **Node.js 20.9 or newer** (nodejs.org, LTS) and the backend running.

```bash
# Terminal 1, project root: start the API
uvicorn rag.api:app --app-dir src --reload --reload-dir src

# Terminal 2, this folder
npm install
copy .env.local.example .env.local      # macOS/Linux: cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000.

| Command | What it does |
|---|---|
| `npm run dev` | Development server with hot reload |
| `npm run build` then `npm start` | Production build and server |
| `npm test` | Unit tests (Vitest) |
| `npm run lint` | ESLint |

`NEXT_PUBLIC_API_URL` in `.env.local` sets where the API lives (default `http://127.0.0.1:8000`).
It's baked in at build time, so rebuild after changing it. The API must list the frontend's
address in `CORS_ORIGINS` (default allows `http://localhost:3000`).

## How it works

```
app/page.tsx ─► components/ClientApp.tsx (browser-only render)
                  └─ Assistant.tsx        state: messages, settings, selected citation, API status
                       ├─ LibraryPanel    documents, upload, search settings
                       ├─ Conversation    questions + answers (markdown, citation buttons)
                       ├─ Composer        question box, Ask / Stop
                       └─ EvidencePane    passages behind an answer, cited first
lib/api.ts        fetch calls; POST /ask streamed with fetch + ReadableStream
lib/sse.ts        Server-Sent Events parser (handles events split across network chunks)
lib/citations.ts  [n] → clickable citation, source ordering, file-name wrapping
lib/storage.ts    guarded localStorage (chat history and settings survive a reload)
```

- **Streaming:** `/ask` is a POST, which the browser's `EventSource` can't send, so the
  response body is read chunk by chunk and parsed as SSE.
- **Stop:** an `AbortController` cancels the request; the answer is marked as stopped.
- **Browser-only render:** the assistant reads saved history from `localStorage`, so it's
  loaded with `next/dynamic` and `ssr: false` to avoid server/browser mismatches.
- **Responsive:** three columns on wide screens; below that the library is a drawer and the
  evidence pane a bottom sheet. Dark mode follows the system setting.
- **Fonts:** Literata (answers, passages) and Instrument Sans (interface) are bundled from
  npm via Fontsource, so nothing is requested from Google at runtime or build time.
