"use client"

import { useEffect, useRef, useState } from "react"
import { getDatasetPlot, getDatasetPlots } from "@/lib/api"
import type { DatasetPlot } from "@/lib/types"

interface Props {
  projectId: string
  datasetId: string
  referenceDatasetId?: string
}

const POLL_INTERVAL_MS = 2000
// Last-resort fallback for the case where rendering never started. Normal
// uploads report completion via the backend's `complete` flag.
const MAX_EMPTY_POLLS = 60 // 120 s

export function DatasetPlotGrid({ projectId, datasetId }: Props) {
  const [plots, setPlots] = useState<DatasetPlot[]>([])
  // "" = fetching, "__error__" = failed, else = b64
  const [images, setImages] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState<string | null>(null)
  const [complete, setComplete] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetched = useRef<Set<string>>(new Set())
  const emptyPolls = useRef(0)

  async function fetchImage(plotId: string) {
    if (fetched.current.has(plotId)) return
    fetched.current.add(plotId)
    setImages((prev) => ({ ...prev, [plotId]: "" }))
    try {
      const { image_b64 } = await getDatasetPlot(projectId, datasetId, plotId)
      setImages((prev) => ({ ...prev, [plotId]: image_b64 }))
    } catch {
      setImages((prev) => ({ ...prev, [plotId]: "__error__" }))
      fetched.current.delete(plotId)
    }
  }

  useEffect(() => {
    let cancelled = false

    async function poll() {
      if (cancelled) return
      try {
        const res = await getDatasetPlots(projectId, datasetId)
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
          return
        }
      } catch {
        // transient — keep polling
      }
      setTimeout(poll, POLL_INTERVAL_MS)
    }

    poll()
    return () => { cancelled = true }
  }, [projectId, datasetId]) // eslint-disable-line react-hooks/exhaustive-deps

  const readyCount = plots.filter((p) => p.status === "ready").length

  return (
    <div className="space-y-3">
      {error && (
        <div className="rounded-md border border-amber-800/50 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
          {error} Showing the {readyCount} plot{readyCount !== 1 ? "s" : ""} that
          could be generated from the dataset profile.
        </div>
      )}

      {!complete && (
        <div className="flex items-center gap-2 py-2 text-xs text-zinc-500">
          <span className="inline-block w-3 h-3 rounded-full border-2 border-zinc-500 border-t-transparent animate-spin" />
          {plots.length === 0
            ? "Generating EDA plots…"
            : `Rendered ${readyCount}/${plots.length} plots…`}
        </div>
      )}

      {complete && plots.length === 0 && (
        <p className="text-xs text-zinc-600 py-2">No plots available yet.</p>
      )}

      {plots.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          {plots.map((plot) => {
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
                <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between">
                  <span className="text-xs font-medium text-zinc-300 truncate">{plot.title}</span>
                  <span className="text-xs text-zinc-600 ml-2 shrink-0">{plot.plot_type}</span>
                </div>
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
