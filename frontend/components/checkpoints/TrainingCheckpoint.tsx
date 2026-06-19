"use client"

import type { Run, StabilityResult } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { MarkdownBody } from "@/components/ui/MarkdownBody"
import { RunPlotGrid } from "./RunPlotGrid"

export function TrainingCheckpoint({ run, runId }: { run: Run; runId: string }) {
  const leaderboard = run.model_comparison
  const statTests = run.stat_tests as Record<string, unknown> | null

  if (!leaderboard?.length) return <EmptyState />

  // Two distinct concepts that diverge under a user override:
  //   selectedName  - the model chosen to go forward (run.best_model_name)
  //   topScorer     - the actual highest-scoring model (leader of the sorted board)
  const selectedName = run.best_model_name
  const metric = (run.model_selection?.primary_metric ?? "score").toUpperCase()

  const topScorerResult = leaderboard.reduce(
    (best, r) => (r.mean > best.mean ? r : best),
    leaderboard[0],
  )
  const topScorer = topScorerResult.model_name
  const isOverride = run.model_selection?.primary_source === "user_override"
  const overrideDivergesFromTop = isOverride && selectedName !== topScorer

  return (
    <div className="space-y-6">
      {/* Leaderboard */}
      <Section title={`Stability Leaderboard - ${metric} (3 seeds × 5 folds)`}>
        <div className="rounded-lg border border-neutral-800 overflow-hidden">
          <table className="w-full text-xs">
            <thead className="border-b border-neutral-800 bg-neutral-900/70">
              <tr>
                <th className="px-4 py-2 text-left text-neutral-500 font-medium">Model</th>
                <th className="px-4 py-2 text-right text-neutral-500 font-medium">Mean</th>
                <th className="px-4 py-2 text-right text-neutral-500 font-medium">± Std</th>
                <th className="px-4 py-2 text-right text-neutral-500 font-medium">Train mean</th>
                <th className="px-4 py-2 text-right text-neutral-500 font-medium">Overfit gap</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((result: StabilityResult) => (
                <tr
                  key={result.model_name}
                  className={`border-b border-neutral-800/50 last:border-0 ${result.model_name === selectedName ? "bg-emerald-950/20" : "bg-neutral-900/30"}`}
                >
                  <td className="px-4 py-2 font-mono text-neutral-200">
                    <span className="flex items-center gap-1.5">
                      {result.model_name}
                      {result.model_name === selectedName && (
                        <Badge variant="success" className="text-[9px] px-1 py-0">primary</Badge>
                      )}
                      {result.model_name === topScorer && result.model_name !== selectedName && (
                        <Badge variant="info" className="text-[9px] px-1 py-0">top {metric}</Badge>
                      )}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-neutral-200">
                    {result.mean.toFixed(4)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-neutral-400">
                    {result.std.toFixed(4)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-neutral-400">
                    {result.train_mean > 0 ? result.train_mean.toFixed(4) : "-"}
                  </td>
                  <td className={`px-4 py-2 text-right font-mono ${result.overfit_gap > 0.15 ? "text-red-400" : "text-neutral-400"}`}>
                    {result.overfit_gap > 0 ? result.overfit_gap.toFixed(4) : "-"}
                    {result.overfit_gap > 0.15 && " ⚠"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Stat test result */}
      {statTests && (
        <Section title="Statistical Significance Test">
          <StatTestCard tests={statTests} />
        </Section>
      )}

      {/* Selected primary summary */}
      {selectedName && run.best_model_score != null && (
        <div className="rounded-lg border border-emerald-900/60 bg-emerald-950/20 px-4 py-3 flex items-center justify-between gap-3">
          <div>
            <p className="text-sm text-neutral-200">
              <span className="font-mono text-emerald-300">{selectedName}</span> selected as primary
              {isOverride && <span className="text-neutral-400"> (your override)</span>}
            </p>
            <p className="text-xs text-neutral-500 mt-0.5">
              Mean {metric}: {run.best_model_score.toFixed(4)}
            </p>
            {overrideDivergesFromTop && (
              <p className="text-xs text-amber-400/90 mt-1">
                Not the highest scorer — top {metric} was{" "}
                <span className="font-mono">{topScorer}</span> ({topScorerResult.mean.toFixed(4)}).
                Using your choice for downstream tuning, calibration and SHAP.
              </p>
            )}
          </div>
          <Badge variant="success">Primary</Badge>
        </div>
      )}

      {/* Training-stage plots: ROC, calibration, score distribution, feature importance */}
      <Section title="Training Plots">
        <RunPlotGrid runId={runId} stage="training" priorityOnly />
      </Section>
    </div>
  )
}

function StatTestCard({ tests }: { tests: Record<string, unknown> }) {
  const testName = String(tests.test_name ?? "").toUpperCase()
  const pValue = typeof tests.p_value === "number" ? tests.p_value : null
  const interpretation = typeof tests.interpretation === "string" ? tests.interpretation : null
  const modelA = String(tests.model_a ?? "")
  const modelB = String(tests.model_b ?? "")
  const gap = typeof tests.score_gap === "number" ? tests.score_gap : null

  const significant = pValue != null && pValue < 0.05

  return (
    <div className={`rounded-lg border px-4 py-3 space-y-1.5 ${significant ? "border-yellow-900/50 bg-yellow-950/20" : "border-neutral-800 bg-neutral-900/40"}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-neutral-300">
          {testName} - {modelA} vs {modelB}
        </span>
        {pValue != null && (
          <span className={`text-xs font-mono px-2 py-0.5 rounded ${significant ? "bg-yellow-900/40 text-yellow-300" : "bg-neutral-800 text-neutral-400"}`}>
            p = {pValue.toFixed(4)}
          </span>
        )}
      </div>
      {gap != null && (
        <p className="text-[11px] text-neutral-500">Score gap: {gap.toFixed(4)}</p>
      )}
      {interpretation && <MarkdownBody>{interpretation}</MarkdownBody>}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">{title}</h3>
      {children}
    </div>
  )
}

function EmptyState() {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-6 py-10 text-center">
      <p className="text-sm text-neutral-500">Training results not yet available.</p>
    </div>
  )
}
