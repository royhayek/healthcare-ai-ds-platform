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
const MAX_STALE_POLLS = 45 // 90 s of no new plots → stop

export function DatasetPlotGrid({ projectId, datasetId, referenceDatasetId }: Props) {
  const [plots, setPlots] = useState<DatasetPlot[]>([])
  // undefined = not fetched, "" = fetching, "__error__" = failed, else = b64
  const [images, setImages] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState<string | null>(null)
  const [renderingDone, setRenderingDone] = useState(false)

  const knownIds = useRef<Set<string>>(new Set())
  const staleCount = useRef(0)

  async function fetchImage(plotId: string) {
    setImages((prev) => ({ ...prev, [plotId]: "" }))
    try {
      const { image_b64 } = await getDatasetPlot(projectId, datasetId, plotId)
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
        const manifest = await getDatasetPlots(projectId, datasetId)
        if (cancelled) return

        const newPlots = manifest.filter((p) => !knownIds.current.has(p.plot_id))
        if (newPlots.length > 0) {
          staleCount.current = 0
          newPlots.forEach((p) => knownIds.current.add(p.plot_id))
          setPlots((prev) => [...prev, ...newPlots])
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
  }, [projectId, datasetId]) // eslint-disable-line react-hooks/exhaustive-deps

  const isRendering = !renderingDone || plots.length === 0

  return (
    <div className="space-y-3">
      {isRendering && (
        <div className="flex items-center gap-2 py-2 text-xs text-zinc-500">
          <span className="inline-block w-3 h-3 rounded-full border-2 border-zinc-500 border-t-transparent animate-spin" />
          {plots.length === 0
            ? "Generating EDA plots…"
            : `Rendered ${plots.length} plot${plots.length !== 1 ? "s" : ""}… more incoming`}
        </div>
      )}

      {!isRendering && plots.length === 0 && (
        <p className="text-xs text-zinc-600 py-2">No plots available yet.</p>
      )}

      {plots.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          {plots.map((plot) => {
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
