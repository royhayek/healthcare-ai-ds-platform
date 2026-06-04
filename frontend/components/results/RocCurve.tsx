"use client"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts"
import type { RocCurveData } from "@/lib/types"

interface Props {
  data: RocCurveData
  auc: number
}

export function RocCurve({ data, auc }: Props) {
  const chartData = data.fpr.map((fpr, i) => ({
    fpr: parseFloat(fpr.toFixed(4)),
    tpr: parseFloat(data.tpr[i].toFixed(4)),
  }))

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-zinc-400">
          ROC Curve - AUC{" "}
          <span className="font-mono font-semibold text-blue-400">{auc.toFixed(4)}</span>
        </p>
        <p className="text-xs text-zinc-500">Higher AUC = better discrimination</p>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: "False positive rate", position: "insideBottom", offset: -12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: "True positive rate", angle: -90, position: "insideLeft", offset: 12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
            labelStyle={{ color: "#a1a1aa", fontSize: 11 }}
            formatter={(v: number) => [v.toFixed(4)]}
            labelFormatter={(fpr) => `FPR: ${Number(fpr).toFixed(4)}`}
          />
          {/* Diagonal random-classifier baseline */}
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
            stroke="#52525b"
            strokeDasharray="4 4"
          />
          <Line
            type="monotone"
            dataKey="tpr"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
