import type { Metadata, Viewport } from "next";
// Fonts are bundled from npm (Fontsource), so they work offline and need no requests to Google.
import "@fontsource-variable/instrument-sans";
import "@fontsource-variable/literata";
import "./globals.css";

export const metadata: Metadata = {
  title: "Document Assistant",
  description: "Ask questions about your documents and check every answer against its source.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#eef1ec" },
    { media: "(prefers-color-scheme: dark)", color: "#151b1d" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
