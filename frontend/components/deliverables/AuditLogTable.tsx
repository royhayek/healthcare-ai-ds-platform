"use client"

/** Audit log table - renders hash-chained audit events with chain validity indicator (§21, §24). */

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { AuditEvent, AuditVerifyResult } from "@/lib/types"

const ACTOR_VARIANT: Record<string, "success" | "warning" | "default" | "outline"> = {
  ai: "default",
  user: "success",
  system: "outline",
}

const CATEGORY_COLORS: Record<string, string> = {
  profiler: "text-blue-400",
  eda: "text-purple-400",
  preprocessing: "text-indigo-400",
  model_selection: "text-cyan-400",
  training: "text-teal-400",
  tuning: "text-emerald-400",
  calibration: "text-green-400",
  threshold: "text-lime-400",
  shap: "text-yellow-400",
  similarity: "text-amber-400",
  drift: "text-orange-400",
  fairness: "text-red-400",
  holdout: "text-rose-400",
  deliverables: "text-pink-400",
  audit: "text-zinc-400",
  chat: "text-violet-400",
}

function EventRow({ event }: { event: AuditEvent }) {
  const catColor = CATEGORY_COLORS[event.category] ?? "text-zinc-400"
  return (
    <tr className="border-b border-zinc-800 hover:bg-zinc-800/30 transition-colors text-xs">
      <td className="py-1.5 pr-3 font-mono text-zinc-600 tabular-nums">{event.seq}</td>
      <td className="py-1.5 pr-3 font-mono text-zinc-500 whitespace-nowrap">
        {new Date(event.timestamp).toLocaleTimeString()}
      </td>
      <td className="py-1.5 pr-3">
        <Badge variant={ACTOR_VARIANT[event.actor] ?? "outline"} className="text-xs capitalize">
          {event.actor}
        </Badge>
      </td>
      <td className={`py-1.5 pr-3 font-mono ${catColor}`}>{event.category}</td>
      <td className="py-1.5 pr-3 font-mono text-zinc-300">{event.action}</td>
      <td className="py-1.5 pr-3 text-zinc-500 max-w-[240px] truncate">{event.reason ?? "-"}</td>
      <td className="py-1.5 font-mono text-zinc-700 text-[10px] truncate max-w-[100px]">
        {event.self_hash.slice(0, 8)}…
      </td>
    </tr>
  )
}

interface AuditLogTableProps {
  events: AuditEvent[]
  verifyResult?: AuditVerifyResult | null
}

export function AuditLogTable({ events, verifyResult }: AuditLogTableProps) {
  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium">
            Audit Log ({events.length} events)
          </CardTitle>
          {verifyResult && (
            <Badge
              variant={verifyResult.chain_valid ? "success" : "error"}
              className="text-xs"
            >
              {verifyResult.chain_valid ? "✓ Chain valid" : "✗ Chain broken"}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {verifyResult && !verifyResult.chain_valid && verifyResult.error && (
          <div className="mb-3 rounded-md border border-red-800/50 bg-red-900/20 px-3 py-2 text-xs text-red-300">
            Chain integrity error: {verifyResult.error}
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-700">
                <th className="text-left pb-2 font-medium text-zinc-500">#</th>
                <th className="text-left pb-2 font-medium text-zinc-500">Time</th>
                <th className="text-left pb-2 font-medium text-zinc-500">Actor</th>
                <th className="text-left pb-2 font-medium text-zinc-500">Category</th>
                <th className="text-left pb-2 font-medium text-zinc-500">Action</th>
                <th className="text-left pb-2 font-medium text-zinc-500">Reason</th>
                <th className="text-left pb-2 font-medium text-zinc-500">Hash</th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev) => <EventRow key={ev.id} event={ev} />)}
            </tbody>
          </table>
          {events.length === 0 && (
            <p className="py-6 text-center text-sm text-zinc-500">No audit events yet.</p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
