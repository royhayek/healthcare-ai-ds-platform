"use client"

import { useParams } from "next/navigation"
import useSWR from "swr"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"
import { fetcher } from "@/lib/api"
import type { Run, DeliverableItem } from "@/lib/types"
import { DeliverableBundle } from "@/components/deliverables/DeliverableBundle"
import { Badge } from "@/components/ui/badge"

export default function DeliverablesPage() {
  const { id } = useParams<{ id: string }>()

  const { data: runs, error: runsError } = useSWR<Run[]>(
    `/api/proxy/projects/${id}/runs`,
    fetcher,
    { refreshInterval: 5000 },
  )

  const latestRun = runs?.[0] ?? null

  const {
    data: deliverables,
    error: delivError,
    mutate: mutateDeliverables,
  } = useSWR<DeliverableItem[]>(
    latestRun ? `/api/proxy/runs/${latestRun.id}/deliverables` : null,
    fetcher,
    { refreshInterval: latestRun?.status === "completed" ? 0 : 5000 },
  )

  if (runsError) {
    return (
      <div className="p-6 text-sm text-red-400">
        Failed to load runs: {runsError.message}
      </div>
    )
  }

  if (!runs) {
    return <div className="p-6 text-sm text-zinc-500">Loading…</div>
  }

  if (!latestRun) {
    return (
      <div className="p-8 max-w-3xl mx-auto">
        <div className="rounded-lg border border-dashed border-zinc-700 py-12 text-center">
          <p className="text-sm font-medium text-zinc-400">No runs yet</p>
          <p className="text-xs text-zinc-600 mt-1">
            Start an analysis to generate deliverables.
          </p>
          <Link
            href={`/project/${id}`}
            className="inline-block mt-3 text-xs text-blue-400 hover:text-blue-300"
          >
            ← Back to project
          </Link>
        </div>
      </div>
    )
  }

  const isRunning = latestRun.status === "running" || latestRun.status === "queued"
  const isFailed = latestRun.status === "failed"
  const isCompleted = latestRun.status === "completed"
  const isAwaiting = latestRun.status === "awaiting_checkpoint"

  const statusVariant = isCompleted
    ? "success"
    : isFailed
    ? "error"
    : isAwaiting
    ? "warning"
    : "info"

  return (
    <div className="p-8 max-w-3xl mx-auto space-y-6">
      <Link
        href={`/project/${id}/results`}
        className="inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
      >
        <ArrowLeft className="w-3 h-3" />
        Back to results
      </Link>

      {/* Run status banner */}
      <div className="rounded-lg border border-zinc-700 bg-zinc-800/50 p-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <p className="text-xs text-zinc-500">Latest run</p>
            <p className="font-mono text-xs text-zinc-400 mt-0.5">{latestRun.id}</p>
          </div>
          <div className="flex items-center gap-3">
            {isRunning && (
              <span className="text-xs text-zinc-500">{latestRun.progress}%</span>
            )}
            <Badge variant={statusVariant}>
              {isRunning && (
                <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
              )}
              {latestRun.status}
            </Badge>
          </div>
        </div>

        {isRunning && (
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-zinc-700">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-500"
              style={{ width: `${latestRun.progress}%` }}
            />
          </div>
        )}

        {isFailed && latestRun.error_message && (
          <p className="mt-2 text-xs text-red-400">{latestRun.error_message}</p>
        )}
      </div>

      {/* Deliverables bundle */}
      {isCompleted ? (
        delivError ? (
          <p className="text-sm text-red-400">
            Failed to load deliverables: {delivError.message}
          </p>
        ) : !deliverables ? (
          <p className="text-sm text-zinc-500">Loading deliverables…</p>
        ) : (
          <DeliverableBundle
            runId={latestRun.id}
            deliverables={deliverables}
            onRegenerate={() => {
              setTimeout(() => mutateDeliverables(), 3000)
            }}
          />
        )
      ) : isRunning || isAwaiting ? (
        <div className="rounded-lg border border-dashed border-zinc-700 py-10 text-center">
          <p className="text-sm text-zinc-500">
            Deliverables will appear when the run completes.
          </p>
          <p className="text-xs text-zinc-600 mt-1">
            Current step: {latestRun.current_step ?? "-"}
          </p>
        </div>
      ) : isFailed ? (
        <div className="rounded-lg border border-dashed border-red-900 py-10 text-center">
          <p className="text-sm text-red-400">Run failed - no deliverables generated.</p>
          <p className="text-xs text-zinc-600 mt-1">
            Check the error message above and start a new run.
          </p>
        </div>
      ) : null}
    </div>
  )
}
