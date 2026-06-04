/** Reusable metric display atom used across results, checkpoints, and model card. */

import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/cn"

interface MetricCardProps {
  label: string
  value: string | number | null | undefined
  sub?: string
  highlight?: "success" | "warning" | "error" | "neutral"
  className?: string
}

const HIGHLIGHT_CLASSES: Record<NonNullable<MetricCardProps["highlight"]>, string> = {
  success: "text-emerald-400",
  warning: "text-amber-400",
  error: "text-red-400",
  neutral: "text-zinc-200",
}

export function MetricCard({ label, value, sub, highlight = "neutral", className }: MetricCardProps) {
  const displayValue = value == null ? "-" : typeof value === "number" ? value.toFixed(4) : value

  return (
    <Card className={cn("bg-zinc-900 border-zinc-800", className)}>
      <CardContent className="pt-4 pb-3 px-4">
        <div className="text-xs text-zinc-500 mb-1 truncate">{label}</div>
        <div className={cn("font-mono text-lg font-semibold leading-none", HIGHLIGHT_CLASSES[highlight])}>
          {displayValue}
        </div>
        {sub && <div className="mt-1 text-xs text-zinc-600 truncate">{sub}</div>}
      </CardContent>
    </Card>
  )
}
