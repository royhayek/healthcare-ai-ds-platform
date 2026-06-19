"use client"

import { useEffect, useRef, useState } from "react"
import { getRunPlot, getRunPlots } from "@/lib/api"
import type { DatasetPlot } from "@/lib/types"

interface Props {
  runId: string
  stage: "eda" | "preprocessing" | "preprocessing_after" | "training" | "drift"
  /** When true, priority=2 plots can be collapsed behind a toggle (expanded by default). */
  priorityOnly?: boolean
}

const STAGE_LABELS: Record<string, string> = {
  eda: "EDA",
  preprocessing: "Preprocessing",
  preprocessing_after: "post-preprocessing",
  training: "Training",
  drift: "Drift",
}

const POLL_INTERVAL_MS = 2000
// Last-resort fallback for stages that are never rendered at all (e.g. a stage
// with no plot specs). Active stages report completion via the backend's
// `complete` flag and finish as soon as rendering is done, so this only fires
// when nothing ever arrives. Kept generous to tolerate a slow stage start.
const MAX_EMPTY_POLLS = 60 // 120 s

export function RunPlotGrid({ runId, stage, priorityOnly = false }: Props) {
  const [plots, setPlots] = useState<DatasetPlot[]>([])
  // map plotId → base64 string ("" = fetching, "__error__" = failed)
  const [images, setImages] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(true)
  const [complete, setComplete] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetched = useRef<Set<string>>(new Set())
  const emptyPolls = useRef(0)

  async function fetchImage(plotId: string) {
    if (fetched.current.has(plotId)) return
    fetched.current.add(plotId)
    setImages((prev) => ({ ...prev, [plotId]: "" })) // "" = loading
    try {
      const { image_b64 } = await getRunPlot(runId, plotId)
      setImages((prev) => ({ ...prev, [plotId]: image_b64 }))
    } catch {
      setImages((prev) => ({ ...prev, [plotId]: "__error__" }))
      fetched.current.delete(plotId) // allow a retry on the next poll
    }
  }

  useEffect(() => {
    let cancelled = false

    async function poll() {
      if (cancelled) return
      try {
        const res = await getRunPlots(runId, stage)
        if (cancelled) return

        setPlots(res.plots)
        setError(res.error ?? null)
        res.plots.forEach((p) => {
          if (p.status === "ready") fetchImage(p.plot_id)
        })

        if (res.plots.length === 0 && !res.complete) {
          emptyPolls.current += 1
          if (emptyPolls.current >= MAX_EMPTY_POLLS) {
            setComplete(true)
            return
          }
        } else {
          emptyPolls.current = 0
        }

        if (res.complete) {
          setComplete(true)
          return // rendering finished — stop polling
        }
      } catch {
        // transient error — keep polling
      }
      setTimeout(poll, POLL_INTERVAL_MS)
    }

    poll()
    return () => { cancelled = true }
  }, [runId, stage]) // eslint-disable-line react-hooks/exhaustive-deps

  const visiblePlots = showAll ? plots : plots.filter((p) => p.priority === 1)
  const hiddenCount = plots.filter((p) => p.priority !== 1).length
  const readyCount = plots.filter((p) => p.status === "ready").length

  return (
    <div className="space-y-3">
      {/* df-load / render error banner */}
      {error && (
        <div className="rounded-md border border-amber-800/50 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
          {error} Showing the {readyCount} plot{readyCount !== 1 ? "s" : ""} that
          could be generated from the dataset profile.
        </div>
      )}

      {/* Progress while plots are still rendering */}
      {!complete && (
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span className="inline-block w-3 h-3 rounded-full border-2 border-zinc-500 border-t-transparent animate-spin" />
          {plots.length === 0
            ? `Generating ${STAGE_LABELS[stage]} plots…`
            : `Rendered ${readyCount}/${plots.length} ${STAGE_LABELS[stage]} plots…`}
        </div>
      )}

      {complete && plots.length === 0 && (
        <p className="text-xs text-zinc-600 py-2">No {stage} plots available.</p>
      )}

      {/* Plot grid */}
      {visiblePlots.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          {visiblePlots.map((plot) => {
            const img = images[plot.plot_id]
            const isFailed = plot.status === "failed" || img === "__error__"
            const hasImage = img && img !== "" && img !== "__error__"
            const isLoading = !isFailed && !hasImage

            return (
              <div
                key={plot.plot_id}
                className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden cursor-pointer hover:border-zinc-600 transition-colors"
                onClick={() => hasImage && setExpanded(plot.plot_id)}
              >
                {/* Card header */}
                <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between gap-2">
                  <span className="text-xs font-medium text-zinc-300 truncate">{plot.title}</span>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {plot.priority === 1 && (
                      <span className="text-[10px] bg-blue-900/40 text-blue-400 border border-blue-800/50 rounded px-1">
                        key
                      </span>
                    )}
                    <span className="text-[10px] text-zinc-600">{plot.plot_type}</span>
                  </div>
                </div>

                {/* Plot body */}
                <div className="p-2">
                  {isLoading && (
                    <div className="flex items-center justify-center min-h-[80px]">
                      <span className="inline-block w-4 h-4 rounded-full border-2 border-zinc-600 border-t-transparent animate-spin" />
                    </div>
                  )}
                  {isFailed && (
                    <div className="flex items-center justify-center min-h-[80px]">
                      <span className="text-xs text-zinc-600">Unavailable</span>
                    </div>
                  )}
                  {hasImage && (
                    <img
                      src={`data:image/png;base64,${img}`}
                      alt={plot.title}
                      className="w-full rounded"
                    />
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Collapse / expand toggle for priority=2 plots */}
      {priorityOnly && hiddenCount > 0 && (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          {showAll ? `↑ Hide ${hiddenCount} secondary plots` : `↓ Show ${hiddenCount} more plots`}
        </button>
      )}

      {/* Lightbox */}
      {expanded && images[expanded] && images[expanded] !== "__error__" && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8"
          onClick={() => setExpanded(null)}
        >
          <div
            className="bg-zinc-900 rounded-xl border border-zinc-700 max-w-3xl w-full shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
              <span className="text-sm font-medium text-zinc-200">
                {plots.find((p) => p.plot_id === expanded)?.title}
              </span>
              <button
                onClick={() => setExpanded(null)}
                className="text-zinc-500 hover:text-zinc-200 text-lg leading-none"
              >
                ✕
              </button>
            </div>
            <div className="p-4">
              <img
                src={`data:image/png;base64,${images[expanded]}`}
                alt="Plot"
                className="w-full rounded"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
