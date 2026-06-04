"use client"

/** Similarity score distribution visualization (§18).
 *
 * Shows the distribution of nearest-neighbor similarity scores across the
 * test set, and optionally a per-prediction score from a single inference.
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface SimilarityBin {
  score: number
  count: number
}

interface SimilarityScoreProps {
  bins?: SimilarityBin[]
  singleScore?: number | null
  confidenceBand?: "high" | "medium" | "low" | null
  indexBuilt: boolean
}

const BAND_CLASSES: Record<string, string> = {
  high: "text-emerald-400",
  medium: "text-amber-400",
  low: "text-red-400",
}

const BAND_BADGE: Record<string, "success" | "warning" | "error"> = {
  high: "success",
  medium: "warning",
  low: "error",
}

export function SimilarityScorePanel({
  bins,
  singleScore,
  confidenceBand,
  indexBuilt,
}: SimilarityScoreProps) {
  if (!indexBuilt) {
    return (
      <Card className="bg-zinc-900 border-zinc-800">
        <CardContent className="py-6 text-center text-sm text-zinc-500">
          Similarity index was not built for this run (faiss-cpu not installed).
        </CardContent>
      </Card>
    )
  }

  const maxCount = bins ? Math.max(...bins.map((b) => b.count), 1) : 1

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium">Similarity Score Distribution</CardTitle>
          {confidenceBand && (
            <Badge variant={BAND_BADGE[confidenceBand]} className="text-xs capitalize">
              {confidenceBand} confidence
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {singleScore != null && (
          <div className="rounded-lg bg-zinc-800/50 px-4 py-3 flex items-center gap-4">
            <div>
              <div className="text-xs text-zinc-500 mb-1">Similarity to training data</div>
              <div className={`font-mono text-2xl font-semibold ${BAND_CLASSES[confidenceBand ?? "medium"]}`}>
                {(singleScore * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-xs text-zinc-500 max-w-[180px]">
              Higher = the prediction is based on training examples the model has seen
              many similar cases for.
            </div>
          </div>
        )}

        {bins && bins.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-xs text-zinc-500 mb-2">Score distribution (test set)</div>
            {bins.map((b, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-12 text-right font-mono text-zinc-500">
                  {(b.score * 100).toFixed(0)}%
                </span>
                <div className="flex-1 h-4 bg-zinc-800 rounded-sm overflow-hidden">
                  <div
                    className="h-full bg-blue-500/60 rounded-sm"
                    style={{ width: `${(b.count / maxCount) * 100}%` }}
                  />
                </div>
                <span className="w-10 font-mono text-zinc-500">{b.count}</span>
              </div>
            ))}
          </div>
        )}

        {!bins && !singleScore && (
          <p className="text-sm text-zinc-500">No similarity data available for this run.</p>
        )}
      </CardContent>
    </Card>
  )
}
