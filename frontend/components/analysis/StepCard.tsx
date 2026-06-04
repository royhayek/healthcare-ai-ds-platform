/** Single step row in the progress feed - icon, label, message, timing. */

import { CheckCircle2, XCircle, Loader2, Clock } from "lucide-react"
import { EventBadge } from "./EventBadge"
import { cn } from "@/lib/cn"

export type StepStatus = "pending" | "running" | "done" | "error" | "checkpoint"

interface StepCardProps {
  eventType: string
  message: string
  status: StepStatus
  pct?: number
  elapsed?: number
  isLatest?: boolean
}

const STATUS_ICON: Record<StepStatus, React.ReactNode> = {
  pending: <Clock className="h-4 w-4 text-zinc-600" />,
  running: <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />,
  done: <CheckCircle2 className="h-4 w-4 text-emerald-400" />,
  error: <XCircle className="h-4 w-4 text-red-400" />,
  checkpoint: <CheckCircle2 className="h-4 w-4 text-amber-400" />,
}

export function StepCard({ eventType, message, status, pct, elapsed, isLatest }: StepCardProps) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-lg px-3 py-2.5 transition-colors",
        isLatest ? "bg-zinc-800/60" : "bg-transparent",
        status === "error" && "border border-red-800/40",
      )}
    >
      <div className="mt-0.5 shrink-0">{STATUS_ICON[status]}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <EventBadge eventType={eventType} />
          {pct != null && (
            <span className="text-xs font-mono text-zinc-500">{pct}%</span>
          )}
          {elapsed != null && (
            <span className="text-xs text-zinc-600 ml-auto">{elapsed.toFixed(1)}s</span>
          )}
        </div>
        <p className="text-sm text-zinc-300 leading-snug truncate">{message}</p>
      </div>
    </div>
  )
}
