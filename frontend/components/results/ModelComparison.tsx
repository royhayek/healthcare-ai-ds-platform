"use client"

import type { StabilityResult } from "@/lib/types"
import { Badge } from "@/components/ui/badge"

interface Props {
  results: StabilityResult[]
  bestModelName: string | null
  primaryMetric?: string
  statTests?: Record<string, unknown> | null
}

export function ModelComparison({ results, bestModelName, primaryMetric = "auc", statTests }: Props) {
  if (!results || results.length === 0) return null

  const sorted = [...results].sort((a, b) => b.mean - a.mean)

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-700 text-zinc-400 text-left">
              <th className="pb-2 pr-4 font-medium">Model</th>
              <th className="pb-2 pr-4 font-medium text-right">Mean {primaryMetric.toUpperCase()}</th>
              <th className="pb-2 pr-4 font-medium text-right">± Std</th>
              <th className="pb-2 pr-4 font-medium text-right">Train Mean</th>
              <th className="pb-2 font-medium text-right">Overfit Gap</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800">
            {sorted.map((r) => {
              const isWinner = r.model_name === bestModelName
              const overfitHigh = r.overfit_gap > 0.15

              return (
                <tr key={r.model_name} className={isWinner ? "bg-blue-950/30" : ""}>
                  <td className="py-2 pr-4 font-mono">
                    <span className={isWinner ? "text-blue-400 font-semibold" : "text-zinc-300"}>
                      {r.model_name}
                    </span>
                    {isWinner && (
                      <Badge variant="outline" className="ml-2 border-blue-500 text-blue-400 text-[10px]">
                        BEST
                      </Badge>
                    )}
                  </td>
                  <td className="py-2 pr-4 text-right font-mono text-zinc-200">
                    {r.mean.toFixed(4)}
                  </td>
                  <td className="py-2 pr-4 text-right font-mono text-zinc-400">
                    {r.std.toFixed(4)}
                  </td>
                  <td className="py-2 pr-4 text-right font-mono text-zinc-400">
                    {r.train_mean.toFixed(4)}
                  </td>
                  <td className="py-2 text-right font-mono">
                    <span className={overfitHigh ? "text-amber-400" : "text-zinc-400"}>
                      {r.overfit_gap.toFixed(4)}
                      {overfitHigh && " ⚠"}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {statTests && (
        <div className="text-xs text-zinc-500 border-t border-zinc-800 pt-2">
          {(statTests as Record<string, unknown>).test_name
            ? `Stat test: ${statTests.test_name} - p=${
                typeof statTests.p_value === "number"
                  ? statTests.p_value.toFixed(4)
                  : "N/A"
              }${
                typeof statTests.significant === "boolean"
                  ? statTests.significant
                    ? " (significant)"
                    : " (not significant)"
                  : ""
              }`
            : null}
        </div>
      )}

      <p className="text-[11px] text-zinc-600">
        Scores are mean ± std across 3 seeds × k-fold CV (project rule 12).
        Overfit gap &gt; 0.15 flagged ⚠.
      </p>
    </div>
  )
}
