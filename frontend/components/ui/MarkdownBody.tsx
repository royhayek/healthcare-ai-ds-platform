"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { clsx } from "clsx"

/**
 * Normalizes AI-generated text that compresses a numbered/bulleted list into a
 * single paragraph. Inserts a blank line before patterns like "2. " or "• " so
 * react-markdown parses them as separate list items.
 */
function normalizeListText(text: string): string {
  // Insert a newline before inline numbered items: " 2. ", " 3. " …
  // Handles both ". " and ") " delimiters (e.g. "1) ").
  return text
    .replace(/\s+(\d+)[.)]\s+/g, (_, n) => `\n${n}. `)
    .replace(/\s+[•·–-]\s+/g, "\n- ")
    .trim()
}

interface MarkdownBodyProps {
  children: string
  className?: string
  /** When true, runs the inline-list normalizer before rendering */
  normalize?: boolean
}

export function MarkdownBody({ children, className, normalize = true }: MarkdownBodyProps) {
  const source = normalize ? normalizeListText(children) : children

  return (
    <div className={clsx("text-sm leading-relaxed text-neutral-100 space-y-1.5", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="mb-1.5 last:mb-0">{children}</p>,
          ul: ({ children }) => (
            <ul className="list-disc list-outside ml-4 space-y-1 mb-1.5">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-outside ml-4 space-y-1 mb-1.5">{children}</ol>
          ),
          li: ({ children }) => <li>{children}</li>,
          strong: ({ children }) => (
            <strong className="font-semibold text-neutral-100">{children}</strong>
          ),
          em: ({ children }) => <em className="italic text-neutral-200">{children}</em>,
          code: ({ inline, children }: { inline?: boolean; children?: React.ReactNode }) =>
            inline ? (
              <code className="bg-neutral-800 rounded px-1 py-0.5 font-mono text-xs text-neutral-200">
                {children}
              </code>
            ) : (
              <pre className="bg-neutral-800 rounded p-2 my-1.5 overflow-x-auto font-mono text-xs whitespace-pre text-neutral-200">
                <code>{children}</code>
              </pre>
            ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-neutral-600 pl-3 text-neutral-300 italic">
              {children}
            </blockquote>
          ),
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  )
}
