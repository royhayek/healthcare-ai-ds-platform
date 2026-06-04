"use client"

import type { ConfusionMatrixData, ConfusionMatrixMultiData } from "@/lib/types"

interface BinaryProps {
  data: ConfusionMatrixData
}

export function BinaryConfusionMatrix({ data }: BinaryProps) {
  const { tn, fp, fn, tp } = data
  const total = tn + fp + fn + tp

  const cells = [
    { label: "True Negative", value: tn, pct: tn / total, color: "bg-blue-950/60 border-blue-800/40", text: "text-blue-300" },
    { label: "False Positive", value: fp, pct: fp / total, color: "bg-rose-950/60 border-rose-800/40", text: "text-rose-300" },
    { label: "False Negative", value: fn, pct: fn / total, color: "bg-rose-950/60 border-rose-800/40", text: "text-rose-300" },
    { label: "True Positive", value: tp, pct: tp / total, color: "bg-emerald-950/60 border-emerald-800/40", text: "text-emerald-300" },
  ]

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-zinc-400">Confusion Matrix</p>
        <div className="flex gap-4 text-xs text-zinc-500">
          <span>Predicted →</span>
        </div>
      </div>
      <div className="flex gap-6 items-start">
        <div className="flex flex-col justify-around text-xs text-zinc-500 h-[120px]">
          <span className="writing-vertical">Actual ↓</span>
        </div>
        <div className="flex-1">
          <div className="grid grid-cols-3 text-xs text-zinc-500 mb-1 pl-16">
            <div />
            <div className="text-center">Predicted 0</div>
            <div className="text-center">Predicted 1</div>
          </div>
          <div className="grid grid-cols-3 gap-1">
            <div className="flex items-center justify-end pr-2 text-xs text-zinc-500">Actual 0</div>
            {[cells[0], cells[1]].map((c, i) => (
              <div key={i} className={`border rounded-lg p-3 text-center ${c.color}`}>
                <div className={`text-2xl font-mono font-bold ${c.text}`}>{c.value.toLocaleString()}</div>
                <div className="text-xs text-zinc-500 mt-0.5">{(c.pct * 100).toFixed(1)}%</div>
                <div className="text-xs text-zinc-400 mt-1">{c.label}</div>
              </div>
            ))}
            <div className="flex items-center justify-end pr-2 text-xs text-zinc-500">Actual 1</div>
            {[cells[2], cells[3]].map((c, i) => (
              <div key={i} className={`border rounded-lg p-3 text-center ${c.color}`}>
                <div className={`text-2xl font-mono font-bold ${c.text}`}>{c.value.toLocaleString()}</div>
                <div className="text-xs text-zinc-500 mt-0.5">{(c.pct * 100).toFixed(1)}%</div>
                <div className="text-xs text-zinc-400 mt-1">{c.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

interface MultiProps {
  data: ConfusionMatrixMultiData
}

export function MultiConfusionMatrix({ data }: MultiProps) {
  const { matrix, classes } = data
  // Normalize each row for color intensity
  const rowMaxes = matrix.map((row) => Math.max(...row))

  return (
    <div>
      <p className="text-sm text-zinc-400 mb-3">Confusion Matrix (multi-class)</p>
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <th className="p-1 text-zinc-500 text-right pr-3">Actual ↓ / Predicted →</th>
              {classes.map((cls) => (
                <th key={cls} className="p-1 text-center text-zinc-400 font-mono min-w-[56px]">{cls}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                <td className="p-1 text-zinc-400 font-mono text-right pr-3">{classes[ri]}</td>
                {row.map((val, ci) => {
                  const intensity = rowMaxes[ri] > 0 ? val / rowMaxes[ri] : 0
                  const isCorrect = ri === ci
                  return (
                    <td
                      key={ci}
                      className="p-1 text-center font-mono"
                      style={{
                        background: isCorrect
                          ? `rgba(16, 185, 129, ${0.1 + intensity * 0.5})`
                          : `rgba(239, 68, 68, ${intensity * 0.4})`,
                        color: intensity > 0.5 ? "#f4f4f5" : "#a1a1aa",
                      }}
                    >
                      {val}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
