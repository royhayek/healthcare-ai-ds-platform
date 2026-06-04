"use client"

import type { SHAPSummary } from "@/lib/types"

interface Props {
  summary: SHAPSummary
  maxFeatures?: number
}

export function ShapPlot({ summary, maxFeatures = 15 }: Props) {
  if (!summary || !summary.feature_names?.length) return null

  // Zip and sort by mean_abs_shap descending, cap at maxFeatures
  const pairs = summary.feature_names
    .map((name, i) => ({ name, value: summary.mean_abs_shap[i] ?? 0 }))
    .sort((a, b) => b.value - a.value)
    .slice(0, maxFeatures)

  const max = pairs[0]?.value ?? 1

  return (
    <div className="space-y-1">
      {pairs.map(({ name, value }) => {
        const pct = max > 0 ? (value / max) * 100 : 0
        return (
          <div key={name} className="flex items-center gap-2 text-xs">
            <span className="w-36 truncate text-right text-zinc-400 font-mono shrink-0" title={name}>
              {name}
            </span>
            <div className="flex-1 bg-zinc-800 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="w-14 text-left text-zinc-500 font-mono shrink-0">
              {value.toFixed(4)}
            </span>
          </div>
        )
      })}
      <p className="text-[11px] text-zinc-600 pt-1">
        Mean |SHAP| across {summary.n_samples} test samples ({summary.explainer_type} explainer).
      </p>
    </div>
  )
}
