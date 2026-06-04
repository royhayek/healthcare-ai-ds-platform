"use client"

import { useEffect, useRef, useState } from "react"
import { getRunPlot, getRunPlots } from "@/lib/api"
import type { DatasetPlot } from "@/lib/types"

interface Props {
  runId: string
  stage: "eda" | "preprocessing" | "preprocessing_after" | "training" | "drift"
  /** If true, show priority=2 plots in a collapsible section */
  priorityOnly?: boolean
}

const STAGE_LABELS: Record<string, string> = {
  eda: "EDA",
  preprocessing: "Preprocessing",
  preprocessing_after: "post-preprocessing",
  training: "Training",
  drift: "Drift",
}

// Poll every 2 s while plots are still arriving; give up after 90 s of no change
const POLL_INTERVAL_MS = 2000
const MAX_STALE_POLLS = 45 // 90 s with no new plots → stop

export function RunPlotGrid({ runId, stage, priorityOnly = false }: Props) {
  const [plots, setPlots] = useState<DatasetPlot[]>([])
  // map plotId → base64 string (undefined = not yet fetched, "" = fetching, "__error__" = failed)
  const [images, setImages] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(!priorityOnly)
  const [renderingDone, setRenderingDone] = useState(false)

  // Track known IDs so we don't re-fetch
  const knownIds = useRef<Set<string>>(new Set())
  const staleCount = useRef(0)

  // Auto-load a single plot image
  async function fetchImage(plotId: string) {
    setImages((prev) => ({ ...prev, [plotId]: "" })) // "" = loading
    try {
      const { image_b64 } = await getRunPlot(runId, plotId)
      setImages((prev) => ({ ...prev, [plotId]: image_b64 }))
    } catch {
      setImages((prev) => ({ ...prev, [plotId]: "__error__" }))
    }
  }

  useEffect(() => {
    let cancelled = false

    async function poll() {
      if (cancelled) return
      try {
        const manifest = await getRunPlots(runId, stage)
        if (cancelled) return

        const newPlots = manifest.filter((p) => !knownIds.current.has(p.plot_id))
        if (newPlots.length > 0) {
          staleCount.current = 0
          newPlots.forEach((p) => knownIds.current.add(p.plot_id))
          setPlots((prev) => [...prev, ...newPlots])
          // Immediately kick off image fetches for new plots
          newPlots.forEach((p) => fetchImage(p.plot_id))
        } else {
          staleCount.current += 1
        }

        if (staleCount.current >= MAX_STALE_POLLS) {
          setRenderingDone(true)
          return
        }
      } catch {
        staleCount.current += 1
      }

      setTimeout(poll, POLL_INTERVAL_MS)
    }

    poll()
    return () => { cancelled = true }
  }, [runId, stage]) // eslint-disable-line react-hooks/exhaustive-deps

  const visiblePlots = showAll ? plots : plots.filter((p) => p.priority === 1)
  const hiddenCount = plots.filter((p) => p.priority !== 1).length
  const isRendering = !renderingDone || plots.length === 0

  return (
    <div className="space-y-3">
      {/* Status bar while plots are arriving */}
      {isRendering && (
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span className="inline-block w-3 h-3 rounded-full border-2 border-zinc-500 border-t-transparent animate-spin" />
          {plots.length === 0
            ? `Generating ${STAGE_LABELS[stage]} plots…`
            : `Rendered ${plots.length} plot${plots.length !== 1 ? "s" : ""}… more incoming`}
        </div>
      )}

      {!isRendering && plots.length === 0 && (
        <p className="text-xs text-zinc-600 py-2">No {stage} plots available.</p>
      )}

      {/* Plot grid */}
      {visiblePlots.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          {visiblePlots.map((plot) => {
            const img = images[plot.plot_id]
            const isLoading = img === "" || img === undefined
            const isError = img === "__error__"
            const hasImage = img && img !== "" && img !== "__error__"

            return (
              <div
                key={plot.plot_id}
                className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden cursor-pointer hover:border-zinc-600 transition-colors"
                onClick={() => setExpanded(plot.plot_id)}
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

                {/* Plot body - always displayed inline */}
                <div className="p-2">
                  {isLoading && (
                    <div className="flex items-center justify-center min-h-[80px]">
                      <span className="inline-block w-4 h-4 rounded-full border-2 border-zinc-600 border-t-transparent animate-spin" />
                    </div>
                  )}
                  {isError && (
                    <div className="flex items-center justify-center min-h-[80px]">
                      <span className="text-xs text-red-500">Failed to load</span>
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

      {/* Show more / less toggle for priority=2 plots */}
      {priorityOnly && hiddenCount > 0 && (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          {showAll ? `↑ Show fewer plots` : `↓ Show ${hiddenCount} more plots`}
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
