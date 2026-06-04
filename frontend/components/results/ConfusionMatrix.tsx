"use client"

interface Props {
  metrics: Record<string, number>
  taskType?: string
}

export function ConfusionMatrix({ metrics, taskType = "binary_classification" }: Props) {
  if (!metrics || Object.keys(metrics).length === 0) return null

  const isClassification = taskType !== "regression"

  if (!isClassification) {
    // Regression metrics
    const regressionKeys = ["r2", "mae", "rmse", "mse"]
    const displayKeys = regressionKeys.filter((k) => k in metrics)
    if (displayKeys.length === 0) return null
    return (
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {displayKeys.map((k) => (
          <MetricTile key={k} label={k.toUpperCase()} value={metrics[k]} />
        ))}
      </div>
    )
  }

  // Classification metrics
  const classKeys = ["accuracy", "f1", "precision", "recall", "roc_auc", "log_loss", "brier_score"]
  const display = classKeys.filter((k) => k in metrics)
  const extra = Object.keys(metrics).filter((k) => !classKeys.includes(k))

  const tp = metrics.tp ?? metrics.TP
  const tn = metrics.tn ?? metrics.TN
  const fp = metrics.fp ?? metrics.FP
  const fn = metrics.fn ?? metrics.FN
  const hasCM = [tp, tn, fp, fn].every((v) => v !== undefined)

  return (
    <div className="space-y-4">
      {/* Classification metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {display.map((k) => (
          <MetricTile key={k} label={k.replace(/_/g, " ").toUpperCase()} value={metrics[k]} />
        ))}
        {extra.map((k) => (
          <MetricTile key={k} label={k.replace(/_/g, " ").toUpperCase()} value={metrics[k]} />
        ))}
      </div>

      {/* 2×2 confusion matrix */}
      {hasCM && (
        <div className="space-y-1">
          <p className="text-[11px] text-zinc-500">Confusion matrix</p>
          <div className="grid grid-cols-3 w-fit gap-px text-xs font-mono">
            <div />
            <div className="bg-zinc-800 px-3 py-1 text-center text-zinc-400">Pred +</div>
            <div className="bg-zinc-800 px-3 py-1 text-center text-zinc-400">Pred −</div>
            <div className="bg-zinc-800 px-3 py-1 text-zinc-400">Actual +</div>
            <div className="bg-green-900/40 border border-green-800 px-3 py-2 text-center text-green-300">
              TP {tp}
            </div>
            <div className="bg-red-900/30 border border-red-900 px-3 py-2 text-center text-red-400">
              FN {fn}
            </div>
            <div className="bg-zinc-800 px-3 py-1 text-zinc-400">Actual −</div>
            <div className="bg-red-900/30 border border-red-900 px-3 py-2 text-center text-red-400">
              FP {fp}
            </div>
            <div className="bg-green-900/40 border border-green-800 px-3 py-2 text-center text-green-300">
              TN {tn}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MetricTile({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-zinc-800/50 rounded-lg p-3 space-y-1">
      <p className="text-[11px] text-zinc-500">{label}</p>
      <p className="text-base font-mono font-semibold text-zinc-200">
        {value !== undefined && value !== null
          ? typeof value === "number"
            ? Math.abs(value) < 10
              ? value.toFixed(4)
              : value.toFixed(2)
            : String(value)
          : "-"}
      </p>
    </div>
  )
}
