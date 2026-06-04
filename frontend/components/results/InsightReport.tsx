"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

interface Props {
  report: string
}

export function InsightReport({ report }: Props) {
  if (!report) return null

  return (
    <div className="markdown-insight text-sm text-zinc-300 leading-relaxed space-y-3
      [&_h1]:text-lg [&_h1]:font-semibold [&_h1]:text-zinc-100 [&_h1]:mt-4 [&_h1]:mb-2
      [&_h2]:text-base [&_h2]:font-semibold [&_h2]:text-zinc-100 [&_h2]:mt-4 [&_h2]:mb-2
      [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:text-zinc-200 [&_h3]:mt-3 [&_h3]:mb-1
      [&_p]:text-zinc-300 [&_p]:leading-relaxed
      [&_strong]:text-zinc-100 [&_strong]:font-semibold
      [&_em]:text-zinc-300 [&_em]:italic
      [&_hr]:border-zinc-700 [&_hr]:my-4
      [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-1
      [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:space-y-1
      [&_li]:text-zinc-300
      [&_code]:bg-zinc-800 [&_code]:text-blue-300 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:font-mono
      [&_pre]:bg-zinc-900 [&_pre]:border [&_pre]:border-zinc-800 [&_pre]:rounded-lg [&_pre]:p-3 [&_pre]:overflow-x-auto
      [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-zinc-300
      [&_blockquote]:border-l-2 [&_blockquote]:border-zinc-600 [&_blockquote]:pl-4 [&_blockquote]:text-zinc-400 [&_blockquote]:italic
      [&_table]:w-full [&_table]:text-xs [&_table]:border-collapse
      [&_th]:border [&_th]:border-zinc-700 [&_th]:bg-zinc-800 [&_th]:px-3 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-medium [&_th]:text-zinc-300
      [&_td]:border [&_td]:border-zinc-800 [&_td]:px-3 [&_td]:py-1.5 [&_td]:text-zinc-400
      [&_tr:nth-child(even)_td]:bg-zinc-900/30
      [&_a]:text-blue-400 [&_a]:underline [&_a:hover]:text-blue-300
    ">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {report}
      </ReactMarkdown>
    </div>
  )
}
