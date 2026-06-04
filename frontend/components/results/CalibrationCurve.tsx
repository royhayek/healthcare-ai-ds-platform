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
import type { CalibrationCurveData } from "@/lib/types"

interface Props {
  data: CalibrationCurveData
}

export function CalibrationCurve({ data }: Props) {
  const chartData = data.prob_pred.map((pred, i) => ({
    predicted: parseFloat(pred.toFixed(4)),
    actual: parseFloat(data.prob_true[i].toFixed(4)),
  }))

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-zinc-400">Reliability Diagram (Calibration)</p>
        <p className="text-xs text-zinc-500">Diagonal = perfectly calibrated</p>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            dataKey="predicted"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: "Mean predicted probability", position: "insideBottom", offset: -12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: "Fraction of positives", angle: -90, position: "insideLeft", offset: 12, fill: "#71717a", fontSize: 12 }}
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
            labelFormatter={(v) => `Predicted: ${Number(v).toFixed(4)}`}
            formatter={(v: number) => [v.toFixed(4), "Actual fraction"]}
          />
          {/* Perfect calibration diagonal */}
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
            stroke="#52525b"
            strokeDasharray="4 4"
          />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#a78bfa"
            strokeWidth={2}
            dot={{ r: 4, fill: "#a78bfa" }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
