"use client";

import dynamic from "next/dynamic";

// The assistant reads saved chat history and settings from the browser, so it
// renders only in the browser. That avoids server/browser mismatches entirely.
const Assistant = dynamic(() => import("./Assistant"), {
  ssr: false,
  loading: () => (
    <div className="flex h-dvh items-center justify-center text-muted" role="status">
      Loading…
    </div>
  ),
});

export default function ClientApp() {
  return <Assistant />;
}
