"use client";

import { useRef, useState } from "react";

interface Props {
  busy: boolean;
  disabledReason: string | null;
  scopeLabel: string | null;
  onAsk: (question: string) => void;
  onStop: () => void;
}

export default function Composer({ busy, disabledReason, scopeLabel, onAsk, onStop }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const disabled = Boolean(disabledReason);

  function submit() {
    const q = text.trim();
    if (!q || busy || disabled) return;
    onAsk(q);
    setText("");
    if (ref.current) ref.current.style.height = "";
  }

  return (
    <form
      className="border-t border-line px-4 py-3 md:px-8"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className="mx-auto max-w-[68ch]">
        <div className="flex items-end gap-2 rounded-xl border border-line bg-panel p-2 focus-within:border-teal">
          <label htmlFor="question" className="sr-only">
            Question
          </label>
          <textarea
            id="question"
            ref={ref}
            rows={1}
            value={text}
            disabled={disabled}
            placeholder={disabledReason ?? "Ask a question about your documents"}
            className="max-h-40 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 outline-none placeholder:text-muted disabled:cursor-not-allowed"
            onChange={(e) => {
              setText(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${e.target.scrollHeight}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
          />
          {busy ? (
            <button type="button" className="rounded-lg border border-line px-4 py-2 text-sm font-medium" onClick={onStop}>
              Stop
            </button>
          ) : (
            <button
              type="submit"
              className="rounded-lg bg-teal px-4 py-2 text-sm font-medium text-teal-ink disabled:opacity-40"
              disabled={disabled || !text.trim()}
            >
              Ask
            </button>
          )}
        </div>
        <p className="mt-1.5 px-1 text-xs text-muted">
          {scopeLabel ? `${scopeLabel}. ` : ""}Enter to ask, Shift+Enter for a new line.
        </p>
      </div>
    </form>
  );
}
