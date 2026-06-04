"use client"

import { DecisionCard } from "./DecisionCard"
import { RunPlotGrid } from "./RunPlotGrid"
import { MarkdownBody } from "@/components/ui/MarkdownBody"
import type { Run } from "@/lib/types"

export function PreprocessingCheckpoint({ run, runId }: { run: Run; runId: string }) {
  const strategy = run.preprocessing_strategy
  if (!strategy) return <EmptyState />

  const kept = Object.entries(strategy.columns).filter(([, s]) => s.action === "keep")
  const dropped = Object.entries(strategy.columns).filter(([, s]) => s.action === "drop")

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="grid grid-cols-3 gap-2">
        <StatBox label="Features kept" value={kept.length} />
        <StatBox label="Features dropped" value={dropped.length} />
        <StatBox label="Task type" value={strategy.task_type} />
      </div>

      {/* High-correlation drops */}
      {strategy.drop_high_correlation.length > 0 && (
        <Section title="Dropped (high correlation)">
          <div className="flex flex-wrap gap-1.5">
            {strategy.drop_high_correlation.map((col) => (
              <span
                key={col}
                className="text-xs font-mono bg-yellow-950/40 text-yellow-400 border border-yellow-900/50 rounded px-2 py-0.5"
              >
                {col}
              </span>
            ))}
          </div>
        </Section>
      )}

      {/* Numeric columns */}
      <Section title={`Numeric columns (${kept.filter(([, s]) => s.dtype_hint === "numeric").length})`}>
        <div className="space-y-1.5">
          {kept
            .filter(([, s]) => s.dtype_hint === "numeric")
            .map(([col, colStrategy]) => (
              <DecisionCard
                key={col}
                fieldPath={`preprocessing.columns.${col}.scale_strategy`}
                label={col}
                value={`${colStrategy.impute_strategy ?? "-"} → ${colStrategy.scale_strategy ?? "no scale"}`}
                reason={colStrategy.reason}
                severity="info"
              />
            ))}
        </div>
      </Section>

      {/* Categorical columns */}
      <Section title={`Categorical columns (${kept.filter(([, s]) => s.dtype_hint === "categorical").length})`}>
        <div className="space-y-1.5">
          {kept
            .filter(([, s]) => s.dtype_hint === "categorical")
            .map(([col, colStrategy]) => (
              <DecisionCard
                key={col}
                fieldPath={`preprocessing.columns.${col}.encode_strategy`}
                label={col}
                value={colStrategy.encode_strategy ?? "onehot"}
                reason={colStrategy.reason}
                severity="info"
              />
            ))}
        </div>
      </Section>

      {/* Dropped columns */}
      {dropped.length > 0 && (
        <Section title={`Dropped columns (${dropped.length})`}>
          <div className="space-y-1.5">
            {dropped.map(([col, colStrategy]) => (
              <DecisionCard
                key={col}
                fieldPath={`preprocessing.columns.${col}.action`}
                label={col}
                value="dropped"
                reason={colStrategy.reason}
                severity="warn"
              />
            ))}
          </div>
        </Section>
      )}

      {strategy.notes && (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3 space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 mb-2">
            Notes
          </p>
          <MarkdownBody>{strategy.notes}</MarkdownBody>
        </div>
      )}

      {/* Preprocessing plots - before/after cap, log previews, etc. */}
      <Section title="Preprocessing Effect Plots">
        <p className="text-xs text-zinc-500 mb-3">
          Before/after visualisations for capping and log-transforms applied to numeric columns.
        </p>
        <RunPlotGrid runId={runId} stage="preprocessing" priorityOnly />
      </Section>

      {/* Re-plots on the cleaned data - verifies AI decisions actually improved the data */}
      <Section title="Data After Preprocessing">
        <p className="text-xs text-zinc-500 mb-3">
          Distributions, correlations, and missingness plots re-generated on the cleaned
          dataset - after imputation, scaling, and column drops have been applied.
          Use these to verify the AI's decisions produced the expected improvements.
        </p>
        <RunPlotGrid runId={runId} stage="preprocessing_after" />
      </Section>
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

function StatBox({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/50 px-3 py-2 space-y-0.5">
      <span className="text-[10px] text-neutral-500">{label}</span>
      <p className="text-sm font-mono text-neutral-200">{value}</p>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-6 py-10 text-center">
      <p className="text-sm text-neutral-500">Preprocessing strategy not yet available.</p>
    </div>
  )
}
