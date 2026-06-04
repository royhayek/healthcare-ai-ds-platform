"use client"

import { useState } from "react"
import type { DeliverableItem } from "@/lib/types"
import { downloadDeliverable } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

const FORMAT_ICON: Record<string, string> = {
  pdf: "📄",
  xlsx: "📊",
  md: "📝",
  csv: "🗒",
  json: "{ }",
  yaml: "⚙",
  zip: "📦",
}

interface Props {
  runId: string
  deliverable: DeliverableItem
}

export function DeliverableCard({ runId, deliverable }: Props) {
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleDownload = async () => {
    setDownloading(true)
    setError(null)
    try {
      const blob = await downloadDeliverable(runId, deliverable.name)
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${deliverable.name}.${deliverable.format}`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed")
    } finally {
      setDownloading(false)
    }
  }

  const icon = FORMAT_ICON[deliverable.format] ?? "📄"

  return (
    <div className="flex items-center gap-3 rounded-lg border border-zinc-700 bg-zinc-800/40 px-4 py-3 hover:bg-zinc-800/70 transition-colors">
      <span className="text-xl leading-none shrink-0" aria-hidden>
        {icon}
      </span>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium text-zinc-100 text-sm">
            {deliverable.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
          </span>
          <Badge variant="outline" className="font-mono text-[10px]">
            {deliverable.format.toUpperCase()}
          </Badge>
        </div>
        {deliverable.audience && (
          <p className="text-xs text-zinc-500 mt-0.5 truncate">{deliverable.audience}</p>
        )}
        {deliverable.generated_at && (
          <p className="text-xs text-zinc-600 mt-0.5">
            {new Date(deliverable.generated_at).toLocaleString()}
          </p>
        )}
        {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
      </div>

      <Button
        variant="outline"
        onClick={handleDownload}
        disabled={downloading}
        className="shrink-0 text-xs h-7 px-3"
      >
        {downloading ? "…" : "Download"}
      </Button>
    </div>
  )
}
