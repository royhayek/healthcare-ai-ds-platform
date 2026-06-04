"use client"

import { DecisionCard } from "./DecisionCard"
import { Badge } from "@/components/ui/badge"
import { MarkdownBody } from "@/components/ui/MarkdownBody"
import type { Run } from "@/lib/types"

export function ModelSelectionCheckpoint({ run }: { run: Run }) {
  const sel = run.model_selection
  if (!sel) return <EmptyState />

  return (
    <div className="space-y-6">
      {/* Primary model */}
      <Section title="Primary Candidate">
        <DecisionCard
          fieldPath="model_selection.primary"
          label="Selected model"
          value={sel.primary}
          reason={sel.reasoning}
          severity="info"
        />
      </Section>

      {/* All candidates */}
      <Section title={`Candidate pool (${sel.candidates.length})`}>
        <div className="space-y-1.5">
          {sel.candidates.map((name) => (
            <div
              key={name}
              className={`rounded-lg border px-4 py-2.5 flex items-center justify-between
                ${name === sel.primary
                  ? "border-emerald-900/60 bg-emerald-950/20"
                  : "border-neutral-800 bg-neutral-900/40"}`}
            >
              <span className="text-sm font-mono text-neutral-200">{name}</span>
              {name === sel.primary && (
                <Badge variant="success" className="text-[10px] px-1.5">primary</Badge>
              )}
            </div>
          ))}
        </div>
      </Section>

      {/* Primary metric */}
      <Section title="Evaluation Metric">
        <DecisionCard
          fieldPath="model_selection.primary_metric"
          label="Primary metric"
          value={sel.primary_metric}
          reason="Optimised throughout stability runs, tuning, and threshold optimisation."
          severity="info"
        />
      </Section>

      {/* Excluded */}
      {sel.excluded.length > 0 && (
        <Section title={`Excluded (${sel.excluded.length})`}>
          <div className="space-y-1.5">
            {sel.excluded.map((exc) => (
              <div
                key={exc.name}
                className="rounded-lg border border-neutral-800 bg-neutral-900/30 px-4 py-2.5 flex items-start justify-between gap-2"
              >
                <span className="text-sm font-mono text-neutral-500">{exc.name}</span>
                <span className="text-xs text-neutral-600 text-right">{exc.reason}</span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {sel.notes && (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3 space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500 mb-2">Notes</p>
          <MarkdownBody>{sel.notes}</MarkdownBody>
        </div>
      )}
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
      <p className="text-sm text-neutral-500">Model selection not yet available.</p>
    </div>
  )
}
