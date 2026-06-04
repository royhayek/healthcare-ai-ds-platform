"use client"

import { useState } from "react"
import { useParams, useSearchParams } from "next/navigation"
import useSWR from "swr"
import Link from "next/link"
import { fetcher, getAuditLog, verifyAuditChain } from "@/lib/api"
import type { AuditEvent, AuditVerifyResult, Run } from "@/lib/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

const ACTOR_COLORS: Record<string, string> = {
  ai: "text-blue-400",
  user: "text-green-400",
  system: "text-zinc-400",
}

const CATEGORY_COLORS: Record<string, string> = {
  eda: "text-purple-400",
  preprocessing: "text-amber-400",
  model_selection: "text-orange-400",
  training: "text-cyan-400",
  calibration: "text-teal-400",
  threshold: "text-lime-400",
  shap: "text-sky-400",
  similarity: "text-indigo-400",
  drift: "text-rose-400",
  fairness: "text-pink-400",
  holdout: "text-red-400",
  prediction: "text-green-400",
  deliverable: "text-emerald-400",
  chat: "text-violet-400",
}

export default function AuditPage() {
  const { id: projectId } = useParams<{ id: string }>()
  const searchParams = useSearchParams()
  const runIdParam = searchParams.get("run_id")

  const { data: runs } = useSWR<Run[]>(
    `/api/proxy/projects/${projectId}/runs`,
    fetcher,
  )

  const completedRuns = runs?.filter((r) => r.status === "completed") ?? []
  const [selectedRunId, setSelectedRunId] = useState<string>(runIdParam ?? "")
  const activeRunId = selectedRunId || completedRuns[completedRuns.length - 1]?.id

  const [page, setPage] = useState(0)
  const pageSize = 50

  const { data: events, isLoading } = useSWR<AuditEvent[]>(
    activeRunId ? `audit-${activeRunId}-${page}` : null,
    () => (activeRunId ? getAuditLog(activeRunId, pageSize, page * pageSize) : Promise.resolve([])),
  )

  const { data: verify } = useSWR<AuditVerifyResult | null>(
    activeRunId ? `audit-verify-${activeRunId}` : null,
    () => (activeRunId ? verifyAuditChain(activeRunId) : Promise.resolve(null)),
  )

  const [expanded, setExpanded] = useState<string | null>(null)

  return (
    <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-zinc-100">Audit log</h1>
        <Button variant="ghost" asChild>
          <Link href={`/project/${projectId}/results${activeRunId ? `?run_id=${activeRunId}` : ""}`}>
            ← Results
          </Link>
        </Button>
      </div>

      {/* Run selector */}
      {completedRuns.length > 1 && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-zinc-500">Run:</span>
          <select
            className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-zinc-200 text-xs font-mono"
            value={activeRunId ?? ""}
            onChange={(e) => {
              setSelectedRunId(e.target.value)
              setPage(0)
            }}
          >
            {completedRuns.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id.slice(0, 8)}… ({new Date(r.created_at).toLocaleDateString()})
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Chain integrity banner */}
      {verify && (
        <div
          className={`flex items-center gap-3 rounded-lg px-4 py-3 border ${
            verify.chain_valid
              ? "bg-green-950/30 border-green-800 text-green-300"
              : "bg-red-950/30 border-red-800 text-red-300"
          }`}
        >
          <span className="text-lg">{verify.chain_valid ? "✓" : "✗"}</span>
          <div>
            <p className="text-sm font-medium">
              {verify.chain_valid ? "Chain integrity verified" : "Chain integrity BROKEN"}
            </p>
            <p className="text-xs opacity-70">
              {verify.total_events} events in chain
              {verify.error && ` - ${verify.error}`}
            </p>
          </div>
          {verify.chain_valid && (
            <Badge variant="success" className="ml-auto">SHA-256 verified</Badge>
          )}
          {!verify.chain_valid && (
            <Badge variant="error" className="ml-auto">TAMPERED</Badge>
          )}
        </div>
      )}

      {/* Events table */}
      <Card>
        <CardHeader>
          <CardTitle>
            Events
            {events && (
              <span className="ml-2 text-sm font-normal text-zinc-500">
                (showing {page * pageSize + 1}-{page * pageSize + events.length})
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading && (
            <div className="text-zinc-500 text-sm py-8 text-center">Loading audit events…</div>
          )}

          {!isLoading && (!events || events.length === 0) && (
            <div className="text-zinc-500 text-sm py-8 text-center">
              {!activeRunId ? "Select a completed run above." : "No audit events found."}
            </div>
          )}

          {events && events.length > 0 && (
            <div className="space-y-0 text-xs font-mono">
              <div className="grid grid-cols-[3rem_8rem_6rem_8rem_1fr] gap-2 pb-1 border-b border-zinc-800 text-zinc-500 text-[11px]">
                <span>#</span>
                <span>Timestamp</span>
                <span>Actor</span>
                <span>Category / Action</span>
                <span>Payload</span>
              </div>
              {events.map((e) => (
                <div key={e.id}>
                  <button
                    className="w-full text-left grid grid-cols-[3rem_8rem_6rem_8rem_1fr] gap-2 py-1.5 hover:bg-zinc-800/40 rounded transition-colors"
                    onClick={() => setExpanded(expanded === e.id ? null : e.id)}
                  >
                    <span className="text-zinc-600">{e.seq}</span>
                    <span className="text-zinc-500 truncate">
                      {new Date(e.timestamp).toLocaleTimeString()}
                    </span>
                    <span className={ACTOR_COLORS[e.actor] ?? "text-zinc-400"}>{e.actor}</span>
                    <span>
                      <span className={CATEGORY_COLORS[e.category] ?? "text-zinc-400"}>
                        {e.category}
                      </span>
                      <span className="text-zinc-500">/{e.action}</span>
                    </span>
                    <span className="text-zinc-500 truncate">
                      {JSON.stringify(e.payload).slice(0, 80)}
                    </span>
                  </button>

                  {expanded === e.id && (
                    <div className="ml-12 mt-1 mb-2 p-3 rounded bg-zinc-900 border border-zinc-800 space-y-2 text-[11px]">
                      <div className="grid grid-cols-[6rem_1fr] gap-1">
                        <span className="text-zinc-600">reason</span>
                        <span className="text-zinc-300">{e.reason ?? "-"}</span>
                        <span className="text-zinc-600">prev_hash</span>
                        <span className="text-zinc-500 break-all">{e.prev_hash}</span>
                        <span className="text-zinc-600">self_hash</span>
                        <span className="text-zinc-500 break-all">{e.self_hash}</span>
                      </div>
                      <pre className="text-zinc-300 whitespace-pre-wrap text-[11px]">
                        {JSON.stringify(e.payload, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Pagination */}
          {events && (
            <div className="flex items-center justify-between pt-4 border-t border-zinc-800 mt-4">
              <Button
                variant="outline"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
              >
                Previous
              </Button>
              <span className="text-xs text-zinc-500">Page {page + 1}</span>
              <Button
                variant="outline"
                onClick={() => setPage((p) => p + 1)}
                disabled={events.length < pageSize}
              >
                Next
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
