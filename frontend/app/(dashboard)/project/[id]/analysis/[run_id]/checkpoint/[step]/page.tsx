"use client"

import { useEffect } from "react"
import { useParams } from "next/navigation"
import useSWR from "swr"
import { Loader2 } from "lucide-react"
import { fetcher } from "@/lib/api"
import { useChatStore } from "@/store/chatStore"
import { useStrategyStore } from "@/store/strategyStore"
import { CheckpointShell } from "@/components/checkpoints/CheckpointShell"
import { EdaCheckpoint } from "@/components/checkpoints/EdaCheckpoint"
import { PreprocessingCheckpoint } from "@/components/checkpoints/PreprocessingCheckpoint"
import { ModelSelectionCheckpoint } from "@/components/checkpoints/ModelSelectionCheckpoint"
import { TrainingCheckpoint } from "@/components/checkpoints/TrainingCheckpoint"
import { FinalCheckpoint } from "@/components/checkpoints/FinalCheckpoint"
import type { Run } from "@/lib/types"

const STEP_META: Record<
  string,
  { number: 1 | 2 | 3 | 4 | 5; title: string; subtitle: string }
> = {
  checkpoint_1_eda: {
    number: 1,
    title: "EDA Complete",
    subtitle: "Review dataset insights before preprocessing decisions are made.",
  },
  checkpoint_2_preprocessing: {
    number: 2,
    title: "Preprocessing Strategy",
    subtitle: "Review per-column decisions. Use the co-pilot to override any strategy.",
  },
  checkpoint_3_model_selection: {
    number: 3,
    title: "Model Selection",
    subtitle: "Review candidate models. Use the co-pilot to change the primary model.",
  },
  checkpoint_4_training: {
    number: 4,
    title: "Training Results",
    subtitle: "Review stability scores across 3 seeds × 5 folds before tuning starts.",
  },
  checkpoint_5_final: {
    number: 5,
    title: "Final Results",
    subtitle: "Review metrics, threshold, SHAP, and insight report before generating deliverables.",
  },
}

function CheckpointContent({ step, run, runId }: { step: string; run: Run; runId: string }) {
  switch (step) {
    case "checkpoint_1_eda":
      return <EdaCheckpoint run={run} runId={runId} />
    case "checkpoint_2_preprocessing":
      return <PreprocessingCheckpoint run={run} runId={runId} />
    case "checkpoint_3_model_selection":
      return <ModelSelectionCheckpoint run={run} />
    case "checkpoint_4_training":
      return <TrainingCheckpoint run={run} runId={runId} />
    case "checkpoint_5_final":
      return <FinalCheckpoint run={run} />
    default:
      return (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-6 py-10 text-center">
          <p className="text-sm text-neutral-500">Unknown checkpoint: {step}</p>
        </div>
      )
  }
}

export default function CheckpointPage() {
  const { id: projectId, run_id: runId, step } = useParams<{
    id: string
    run_id: string
    step: string
  }>()

  const setRunId = useChatStore((s) => s.setRunId)
  const setStrategy = useStrategyStore((s) => s.setStrategy)

  // Activate the co-pilot for this run
  useEffect(() => {
    setRunId(runId)
    return () => setRunId(null)
  }, [runId, setRunId])

  const { data: run, isLoading } = useSWR<Run>(
    runId ? `/api/proxy/runs/${runId}` : null,
    fetcher,
    { revalidateOnFocus: false },
  )

  // Hydrate strategy store with current run strategy
  useEffect(() => {
    if (!run) return
    const strategySnapshot: Record<string, unknown> = {}
    if (run.preprocessing_strategy) {
      strategySnapshot.preprocessing_strategy = run.preprocessing_strategy
    }
    if (run.model_selection) {
      strategySnapshot.model_selection = run.model_selection
    }
    if (run.threshold_config) {
      strategySnapshot.threshold_config = run.threshold_config
    }
    if (Object.keys(strategySnapshot).length > 0) {
      setStrategy(strategySnapshot)
    }
  }, [run, setStrategy])

  const meta = STEP_META[step]

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-5 h-5 text-neutral-500 animate-spin" />
      </div>
    )
  }

  if (!run) {
    return (
      <div className="p-8 text-sm text-neutral-500">Run not found.</div>
    )
  }

  if (!meta) {
    return (
      <div className="p-8 text-sm text-neutral-500">Unknown checkpoint step: {step}</div>
    )
  }

  return (
    <CheckpointShell
      run={run}
      projectId={projectId}
      checkpointNumber={meta.number}
      title={meta.title}
      subtitle={meta.subtitle}
    >
      <CheckpointContent step={step} run={run} runId={runId} />
    </CheckpointShell>
  )
}
