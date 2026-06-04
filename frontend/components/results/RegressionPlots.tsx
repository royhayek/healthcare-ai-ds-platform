"use client"

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts"
import type { ScatterPoint, ResidualPoint } from "@/lib/types"

interface PredActualProps {
  data: ScatterPoint[]
}

export function PredictedVsActual({ data }: PredActualProps) {
  const allVals = data.flatMap((d) => [d.actual, d.predicted])
  const minVal = Math.min(...allVals)
  const maxVal = Math.max(...allVals)

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-zinc-400">Predicted vs Actual</p>
        <p className="text-xs text-zinc-500">Diagonal = perfect prediction</p>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            dataKey="actual"
            type="number"
            name="Actual"
            domain={[minVal, maxVal]}
            label={{ value: "Actual", position: "insideBottom", offset: -12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <YAxis
            dataKey="predicted"
            type="number"
            name="Predicted"
            domain={[minVal, maxVal]}
            label={{ value: "Predicted", angle: -90, position: "insideLeft", offset: 12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
            formatter={(v: number) => [v.toFixed(4)]}
          />
          <ReferenceLine
            segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]}
            stroke="#52525b"
            strokeDasharray="4 4"
          />
          <Scatter data={data} fill="#3b82f6" opacity={0.5} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

interface ResidualsProps {
  data: ResidualPoint[]
}

export function ResidualsPlot({ data }: ResidualsProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-zinc-400">Residuals vs Predicted</p>
        <p className="text-xs text-zinc-500">Should be centered around zero with no pattern</p>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            dataKey="predicted"
            type="number"
            name="Predicted"
            label={{ value: "Predicted", position: "insideBottom", offset: -12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <YAxis
            dataKey="residual"
            type="number"
            name="Residual"
            label={{ value: "Residual", angle: -90, position: "insideLeft", offset: 12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
            formatter={(v: number) => [v.toFixed(4)]}
          />
          <ReferenceLine y={0} stroke="#52525b" strokeDasharray="4 4" />
          <Scatter data={data} fill="#f59e0b" opacity={0.5} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
