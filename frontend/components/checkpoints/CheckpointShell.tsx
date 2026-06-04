"use client"

/**
 * CheckpointShell - shared layout for all 5 checkpoint pages.
 *
 * Renders:
 *   - Breadcrumb back to the analysis progress feed
 *   - Checkpoint number badge + title
 *   - The checkpoint-specific content (children)
 *   - A "Resume pipeline" button that calls POST /runs/{runId}/resume
 *
 * The Resume button is disabled while the request is in-flight.
 * On success it navigates back to the analysis page.
 */

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { ArrowLeft, Play, Loader2, RotateCw, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useStrategyStore } from "@/store/strategyStore"
import type { Run } from "@/lib/types"

interface CheckpointShellProps {
  run: Run
  projectId: string
  checkpointNumber: 1 | 2 | 3 | 4 | 5
  title: string
  subtitle?: string
  children: React.ReactNode
}

const CHECKPOINT_TITLES: Record<number, string> = {
  1: "EDA Review",
  2: "Preprocessing Decisions",
  3: "Model Selection",
  4: "Training Results",
  5: "Final Review",
}

export function CheckpointShell({
  run,
  projectId,
  checkpointNumber,
  title,
  subtitle,
  children,
}: CheckpointShellProps) {
  const router = useRouter()
  const [resuming, setResuming] = useState(false)
  const [rerunning, setRerunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const strategy = useStrategyStore((s) => s.strategy)

  // Surface when this checkpoint's model output fell back to defaults (a JSON
  // parse failure or a refusal). The checkpoint still reads as "successful", so
  // without this the user can't tell the decision wasn't a real model output.
  const fallbackNote: string | null =
    checkpointNumber === 2
      ? run.preprocessing_strategy?.notes ?? null
      : checkpointNumber === 3
        ? run.model_selection?.notes ?? null
        : null
  const usedFallback = !!fallbackNote && /fallback|parse fail|refus/i.test(fallbackNote)

  const handleResume = async () => {
    setResuming(true)
    setError(null)
    try {
      const body: Record<string, unknown> = {}

      // Include any strategy changes that were applied via chat
      if (strategy) {
        const keys = Object.keys(strategy) as string[]
        const overrideKeys = ["model_selection", "preprocessing_strategy"]
        const hasOverride = keys.some((k) => overrideKeys.includes(k))
        if (hasOverride) {
          body.strategy_override = Object.fromEntries(
            keys.filter((k) => overrideKeys.includes(k)).map((k) => [k, strategy[k]])
          )
        }
      }

      const res = await fetch(`/api/proxy/runs/${run.id}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
      }

      router.push(`/project/${projectId}/analysis/${run.id}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Resume failed")
      setResuming(false)
    }
  }

  const handleRerun = async () => {
    setRerunning(true)
    setError(null)
    try {
      const res = await fetch(`/api/proxy/runs/${run.id}/rerun`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
      }

      router.push(`/project/${projectId}/analysis/${run.id}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Re-run failed")
      setRerunning(false)
    }
  }

  const busy = resuming || rerunning
  const analysisPath = `/project/${projectId}/analysis/${run.id}`

  return (
    <div className="p-8 max-w-3xl mx-auto space-y-6">
      {/* Breadcrumb */}
      <Link
        href={analysisPath}
        className="inline-flex items-center gap-1 text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
      >
        <ArrowLeft className="w-3 h-3" />
        Back to progress feed
      </Link>

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px] px-2 font-mono">
              Checkpoint {checkpointNumber}/5
            </Badge>
            <span className="text-xs text-neutral-500">
              {CHECKPOINT_TITLES[checkpointNumber]}
            </span>
          </div>
          <h1 className="text-lg font-semibold text-neutral-100">{title}</h1>
          {subtitle && <p className="text-sm text-neutral-500">{subtitle}</p>}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <Button
            size="sm"
            variant="outline"
            className="border-neutral-700 text-neutral-200 hover:bg-neutral-800"
            onClick={handleRerun}
            disabled={busy}
            title="Re-run this step from scratch (regenerates this checkpoint)"
          >
            {rerunning ? (
              <>
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                Re-running…
              </>
            ) : (
              <>
                <RotateCw className="w-3.5 h-3.5 mr-1.5" />
                Re-run step
              </>
            )}
          </Button>
          <Button
            size="sm"
            variant="default"
            className="bg-emerald-700 hover:bg-emerald-600 text-white"
            onClick={handleResume}
            disabled={busy}
          >
            {resuming ? (
              <>
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                Resuming…
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 mr-1.5" />
                Resume pipeline
              </>
            )}
          </Button>
        </div>
      </div>

      {usedFallback && (
        <div className="rounded-lg border border-amber-900/60 bg-amber-950/30 px-4 py-3 text-sm text-amber-200 flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <div className="space-y-1">
            <p className="font-medium">This step used a rule-based fallback, not the model&apos;s output.</p>
            <p className="text-amber-300/80 text-xs">
              {fallbackNote} - the decisions below are safe defaults. Click{" "}
              <span className="font-medium">Re-run step</span> to regenerate them with the model.
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Checkpoint content */}
      {children}

      {/* Footer resume */}
      <div className="pt-4 border-t border-neutral-800 flex justify-between items-center">
        <p className="text-xs text-neutral-600">
          The co-pilot on the right is active. Ask it to explain or override any decision before resuming.
        </p>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            className="border-neutral-700 text-neutral-200 hover:bg-neutral-800"
            onClick={handleRerun}
            disabled={busy}
            title="Re-run this step from scratch (regenerates this checkpoint)"
          >
            {rerunning ? (
              <><Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />Re-running…</>
            ) : (
              <><RotateCw className="w-3.5 h-3.5 mr-1.5" />Re-run step</>
            )}
          </Button>
          <Button
            size="sm"
            variant="default"
            className="bg-emerald-700 hover:bg-emerald-600 text-white"
            onClick={handleResume}
            disabled={busy}
          >
            {resuming ? (
              <><Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />Resuming…</>
            ) : (
              <><Play className="w-3.5 h-3.5 mr-1.5" />Resume pipeline</>
            )}
          </Button>
        </div>
      </div>
    </div>
  )
}
