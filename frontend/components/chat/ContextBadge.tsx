"use client"

import { useChatStore } from "@/store/chatStore"

interface Props {
  runId: string
}

export default function ContextBadge({ runId }: Props) {
  return (
    <span className="font-mono text-[10px] text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded truncate max-w-[120px]" title={runId}>
      {runId.slice(0, 8)}…
    </span>
  )
}
