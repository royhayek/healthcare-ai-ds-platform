"use client"

import { useState } from "react"
import type { DeliverableItem } from "@/lib/types"
import { DeliverableCard } from "./DeliverableCard"
import { downloadDeliverable, regenerateDeliverables } from "@/lib/api"
import { Button } from "@/components/ui/button"

const DELIVERABLE_ORDER = [
  "executive_summary",
  "technical_report",
  "model_card",
  "model_card_pdf",
  "data_quality_report",
  "predictions",
  "audit_log_csv",
  "audit_log_json",
  "repro_manifest",
  "risk_register",
  "bundle",
]

interface Props {
  runId: string
  deliverables: DeliverableItem[]
  onRegenerate?: () => void
}

export function DeliverableBundle({ runId, deliverables, onRegenerate }: Props) {
  const [regenerating, setRegenerating] = useState(false)
  const [regenError, setRegenError] = useState<string | null>(null)
  const [downloadingBundle, setDownloadingBundle] = useState(false)

  const sorted = [...deliverables].sort((a, b) => {
    const ai = DELIVERABLE_ORDER.indexOf(a.name)
    const bi = DELIVERABLE_ORDER.indexOf(b.name)
    if (ai === -1 && bi === -1) return a.name.localeCompare(b.name)
    if (ai === -1) return 1
    if (bi === -1) return -1
    return ai - bi
  })

  const bundle = sorted.find((d) => d.name === "bundle")
  const rest = sorted.filter((d) => d.name !== "bundle")

  const handleRegenerate = async () => {
    setRegenerating(true)
    setRegenError(null)
    try {
      await regenerateDeliverables(runId)
      onRegenerate?.()
    } catch (err) {
      setRegenError(err instanceof Error ? err.message : "Regeneration failed")
    } finally {
      setRegenerating(false)
    }
  }

  const handleBundleDownload = async () => {
    if (!bundle) return
    setDownloadingBundle(true)
    try {
      const blob = await downloadDeliverable(runId, "bundle")
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `deliverables_${runId.slice(0, 8)}.zip`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } finally {
      setDownloadingBundle(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-sm font-semibold text-zinc-100">
            Deliverables
            <span className="ml-2 text-xs font-normal text-zinc-500">
              ({rest.length} file{rest.length !== 1 ? "s" : ""})
            </span>
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">All documents generated from this run</p>
        </div>
        <div className="flex items-center gap-2">
          {bundle && (
            <Button
              onClick={handleBundleDownload}
              disabled={downloadingBundle}
              className="text-xs"
            >
              📦 {downloadingBundle ? "Downloading…" : "Download All (ZIP)"}
            </Button>
          )}
          <Button
            variant="outline"
            onClick={handleRegenerate}
            disabled={regenerating}
            className="text-xs"
          >
            {regenerating ? "Queuing…" : "Regenerate"}
          </Button>
        </div>
      </div>

      {regenError && (
        <p className="text-xs text-red-400 rounded bg-red-950/30 border border-red-900 px-3 py-2">
          {regenError}
        </p>
      )}

      {rest.length === 0 ? (
        <div className="rounded-lg border border-dashed border-zinc-700 py-8 text-center">
          <p className="text-sm text-zinc-500">No deliverables yet - run must complete first.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {rest.map((d) => (
            <DeliverableCard key={d.id} runId={runId} deliverable={d} />
          ))}
        </div>
      )}
    </div>
  )
}
