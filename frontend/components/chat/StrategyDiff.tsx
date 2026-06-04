"use client"

import { type StrategyDiff } from "@/lib/types"

interface Props {
  diff: StrategyDiff
}

export default function StrategyDiffCard({ diff }: Props) {
  return (
    <div className="rounded-md border border-amber-800/50 bg-amber-950/40 px-3 py-2 text-xs">
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-amber-400 font-medium">⚡ Strategy changed</span>
        <span className="text-neutral-500 font-mono">{diff.field_path}</span>
      </div>
      <p className="text-neutral-300">{diff.summary}</p>
      <div className="mt-1.5 flex gap-3 font-mono">
        <span className="text-red-400 line-through">{String(diff.before ?? "-")}</span>
        <span className="text-emerald-400">{String(diff.after ?? "-")}</span>
      </div>
    </div>
  )
}
