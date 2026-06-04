"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import type { ScoreBin } from "@/lib/types"

interface Props {
  data: ScoreBin[]
  threshold?: number
}

export function ScoreDistribution({ data, threshold }: Props) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-zinc-400">Score Distribution</p>
        {threshold != null && (
          <p className="text-xs text-zinc-500">
            Threshold: <span className="font-mono text-amber-400">{threshold.toFixed(3)}</span>
          </p>
        )}
      </div>
      <p className="text-xs text-zinc-500 mb-2">Distribution of predicted probabilities by true class</p>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 20, left: 8 }} barCategoryGap="5%">
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            dataKey="score"
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: "Predicted probability", position: "insideBottom", offset: -12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <YAxis
            label={{ value: "Count", angle: -90, position: "insideLeft", offset: 12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
            labelFormatter={(v) => `Score ≈ ${Number(v).toFixed(2)}`}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }} />
          <Bar dataKey="negative" name="Class 0 (negative)" fill="#3b82f6" opacity={0.75} isAnimationActive={false} />
          <Bar dataKey="positive" name="Class 1 (positive)" fill="#f59e0b" opacity={0.75} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
