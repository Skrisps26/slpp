import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GCIS — Grounded Clinical Intelligence System",
  description: "Research-grade clinical NLP with AI-generated SOAP notes",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50">
        <header className="bg-white border-b border-slate-200 py-4">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between">
              <h1 className="text-xl font-bold text-slate-900">GCIS</h1>
              <span className="text-sm text-slate-500">
                Grounded Clinical Intelligence System
              </span>
            </div>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
