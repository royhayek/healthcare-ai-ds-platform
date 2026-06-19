"use client"

import { type Message } from "@/lib/types"
import { clsx } from "clsx"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

interface Props {
  message: Message
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === "user"

  return (
    <div className={clsx("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={clsx(
          "min-w-0 max-w-[85%] overflow-hidden rounded-lg px-3 py-2 text-sm leading-relaxed break-words",
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-neutral-800 text-neutral-100",
        )}
      >
        {message.isStreaming && message.content === "" ? (
          <span className="inline-flex gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-bounce [animation-delay:-0.3s]" />
            <span className="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-bounce [animation-delay:-0.15s]" />
            <span className="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-bounce" />
          </span>
        ) : !isUser && message.content.trim() === "" ? (
          // Backstop: a finalized assistant turn with no text (e.g. an empty
          // model completion, or a pre-fix persisted blank) must not render as an
          // invisible bubble. The backend now substitutes a fallback, but older
          // rows may still be empty.
          <span className="italic text-neutral-500">No response - please try again.</span>
        ) : isUser ? (
          <span className="whitespace-pre-wrap">{message.content}</span>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0 break-words">{children}</p>,
              ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
              li: ({ children }) => <li className="ml-2 break-words">{children}</li>,
              code: ({ inline, children }: { inline?: boolean; children?: React.ReactNode }) =>
                inline ? (
                  <code className="bg-neutral-700 rounded px-1 py-0.5 font-mono text-xs break-words">{children}</code>
                ) : (
                  <pre className="bg-neutral-700 rounded p-2 my-2 max-w-full overflow-x-auto font-mono text-xs whitespace-pre">
                    <code>{children}</code>
                  </pre>
                ),
              table: ({ children }) => (
                <div className="my-2 max-w-full overflow-x-auto">
                  <table className="w-full border-collapse text-xs">{children}</table>
                </div>
              ),
              thead: ({ children }) => <thead className="border-b border-neutral-600">{children}</thead>,
              th: ({ children }) => (
                <th className="px-2 py-1 text-left font-semibold text-neutral-200 align-top">{children}</th>
              ),
              td: ({ children }) => (
                <td className="px-2 py-1 align-top border-b border-neutral-700/60 break-words">{children}</td>
              ),
              strong: ({ children }) => <strong className="font-semibold text-neutral-100">{children}</strong>,
              em: ({ children }) => <em className="italic">{children}</em>,
              h1: ({ children }) => <h1 className="font-bold text-base mb-1">{children}</h1>,
              h2: ({ children }) => <h2 className="font-bold text-sm mb-1">{children}</h2>,
              h3: ({ children }) => <h3 className="font-semibold mb-1">{children}</h3>,
              blockquote: ({ children }) => (
                <blockquote className="border-l-2 border-neutral-500 pl-2 my-2 text-neutral-300">{children}</blockquote>
              ),
              a: ({ href, children }) => (
                <a href={href} className="underline text-indigo-400 hover:text-indigo-300" target="_blank" rel="noreferrer">{children}</a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        )}
        {message.isStreaming && message.content !== "" && (
          <span className="inline-block w-0.5 h-3.5 bg-neutral-400 ml-0.5 animate-pulse align-middle" />
        )}
      </div>
    </div>
  )
}
