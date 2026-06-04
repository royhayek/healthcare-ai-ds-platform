"use client"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import type { PrCurveData } from "@/lib/types"

interface Props {
  data: PrCurveData
}

export function PrCurve({ data }: Props) {
  const chartData = data.recall.map((rec, i) => ({
    recall: parseFloat(rec.toFixed(4)),
    precision: parseFloat(data.precision[i].toFixed(4)),
  }))

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-zinc-400">
          Precision-Recall Curve - AP{" "}
          <span className="font-mono font-semibold text-emerald-400">{data.ap.toFixed(4)}</span>
        </p>
        <p className="text-xs text-zinc-500">Important for imbalanced data</p>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: "Recall", position: "insideBottom", offset: -12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: "Precision", angle: -90, position: "insideLeft", offset: 12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
            labelStyle={{ color: "#a1a1aa", fontSize: 11 }}
            formatter={(v: number) => [v.toFixed(4)]}
            labelFormatter={(rec) => `Recall: ${Number(rec).toFixed(4)}`}
          />
          <Line
            type="monotone"
            dataKey="precision"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
