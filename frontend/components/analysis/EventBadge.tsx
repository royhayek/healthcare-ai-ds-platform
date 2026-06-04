/** Badge that colour-codes pipeline event types in the progress feed. */

import { Badge } from "@/components/ui/badge"

const EVENT_BADGE_MAP: Record<string, { label: string; variant: "success" | "warning" | "error" | "outline" | "default" }> = {
  checkpoint: { label: "Checkpoint", variant: "warning" },
  error: { label: "Error", variant: "error" },
  load: { label: "Load", variant: "outline" },
  profile: { label: "Profile", variant: "outline" },
  eda: { label: "EDA", variant: "default" },
  preprocessing: { label: "Preprocess", variant: "default" },
  model_selection: { label: "Models", variant: "default" },
  training: { label: "Training", variant: "default" },
  tuning: { label: "Tuning", variant: "default" },
  calibration: { label: "Calibrate", variant: "default" },
  threshold: { label: "Threshold", variant: "default" },
  shap: { label: "SHAP", variant: "default" },
  similarity: { label: "Similarity", variant: "default" },
  drift: { label: "Drift", variant: "default" },
  fairness: { label: "Fairness", variant: "default" },
  holdout: { label: "Holdout", variant: "default" },
  insight: { label: "Insight", variant: "default" },
  deliverables_queued: { label: "Deliverables", variant: "success" },
  done: { label: "Done", variant: "success" },
}

interface EventBadgeProps {
  eventType: string
}

export function EventBadge({ eventType }: EventBadgeProps) {
  const entry = EVENT_BADGE_MAP[eventType]
  if (!entry) return <Badge variant="outline" className="text-xs">{eventType}</Badge>
  return <Badge variant={entry.variant} className="text-xs">{entry.label}</Badge>
}
