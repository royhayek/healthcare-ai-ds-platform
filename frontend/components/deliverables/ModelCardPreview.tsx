"use client"

/** In-browser model card preview - renders key model card fields without
 * downloading the PDF. Mirrors the structure of backend/deliverables/model_card.py. */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { MetricCard } from "@/components/shared/MetricCard"

interface ModelCardPreviewProps {
  modelName: string | null
  taskType: string | null
  finalMetrics: Record<string, number> | null
  thresholdUsed: number
  topFeatures: string[]
  calibrationMethod: string | null
  stabilityMean: number | null
  driftSeverity: string | null
  fairnessSeverity: string | null
  completedAt: string | null
}

const SEVERITY_BADGE: Record<string, "success" | "warning" | "error" | "outline"> = {
  stable: "success",
  none: "success",
  mild: "warning",
  moderate: "warning",
  significant: "error",
  severe: "error",
}

export function ModelCardPreview({
  modelName,
  taskType,
  finalMetrics,
  thresholdUsed,
  topFeatures,
  calibrationMethod,
  stabilityMean,
  driftSeverity,
  fairnessSeverity,
  completedAt,
}: ModelCardPreviewProps) {
  const metrics = finalMetrics ?? {}

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium">Model Card Preview</CardTitle>
          <div className="flex gap-1.5">
            {taskType && <Badge variant="outline" className="text-xs">{taskType}</Badge>}
            {modelName && <Badge variant="default" className="text-xs font-mono">{modelName}</Badge>}
          </div>
        </div>
        {completedAt && (
          <p className="text-xs text-zinc-600 mt-1">
            Trained {new Date(completedAt).toLocaleDateString()}
          </p>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Metrics grid */}
        {Object.keys(metrics).length > 0 && (
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(metrics).slice(0, 6).map(([k, v]) => (
              <MetricCard
                key={k}
                label={k.replace(/_/g, " ")}
                value={v}
                highlight="neutral"
              />
            ))}
          </div>
        )}

        {/* Threshold + calibration */}
        <div className="rounded-lg bg-zinc-800/40 px-4 py-3 grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-xs text-zinc-500 mb-1">Decision threshold</div>
            <div className="font-mono font-semibold">{thresholdUsed.toFixed(3)}</div>
          </div>
          {calibrationMethod && (
            <div>
              <div className="text-xs text-zinc-500 mb-1">Calibration</div>
              <div className="font-mono font-semibold capitalize">{calibrationMethod}</div>
            </div>
          )}
          {stabilityMean != null && (
            <div>
              <div className="text-xs text-zinc-500 mb-1">CV mean (stability)</div>
              <div className="font-mono font-semibold">{stabilityMean.toFixed(4)}</div>
            </div>
          )}
        </div>

        {/* Top features */}
        {topFeatures.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-2">Top features (SHAP)</div>
            <div className="flex flex-wrap gap-1.5">
              {topFeatures.slice(0, 10).map((f, i) => (
                <span key={i} className="rounded-md bg-zinc-800 px-2 py-0.5 font-mono text-xs text-zinc-300">
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Drift / fairness */}
        {(driftSeverity || fairnessSeverity) && (
          <div className="flex gap-3 text-sm">
            {driftSeverity && (
              <div className="flex items-center gap-1.5">
                <span className="text-zinc-500 text-xs">Drift:</span>
                <Badge variant={SEVERITY_BADGE[driftSeverity] ?? "outline"} className="text-xs capitalize">
                  {driftSeverity}
                </Badge>
              </div>
            )}
            {fairnessSeverity && (
              <div className="flex items-center gap-1.5">
                <span className="text-zinc-500 text-xs">Fairness:</span>
                <Badge variant={SEVERITY_BADGE[fairnessSeverity] ?? "outline"} className="text-xs capitalize">
                  {fairnessSeverity}
                </Badge>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
