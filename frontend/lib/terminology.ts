/**
 * Central clinical terminology map.
 * All UI strings that differ between "generic DS platform" and
 * "healthcare platform" live here - one place to update.
 *
 * Components import from this module instead of using inline strings
 * so terminology is consistent and refactorable.
 */

// Display labels for pipeline concepts in clinical context
export const TERM = {
  // Entities
  run: "Analysis",
  runs: "Analyses",
  dataset: "Patient cohort",
  datasets: "Patient cohorts",
  model: "Prediction model",
  models: "Prediction models",
  feature: "Clinical variable",
  features: "Clinical variables",
  target: "Clinical outcome",
  prediction: "Risk score",
  predictions: "Risk scores",

  // Metrics (clinical framing)
  shap_value: "Risk contribution",
  threshold: "Decision threshold",
  auc: "Discrimination (AUC-ROC)",
  brier_score: "Probability reliability (Brier)",
  calibration: "Probability calibration",
  sensitivity: "Sensitivity (true positive rate)",
  specificity: "Specificity (true negative rate)",
  fnr: "Miss rate (false negatives / all positives)",
  fpr: "False alarm rate",
  fairness: "Equity analysis",
  drift: "Population shift",

  // Clinical cost matrix labels
  fn_cost_label: "Cost of a missed case (False Negative)",
  fp_cost_label: "Cost of a false alarm (False Positive)",
  fn_cost_clinical:
    "A patient who has the condition is predicted negative - missed opportunity for intervention.",
  fp_cost_clinical:
    "A healthy patient is flagged positive - unnecessary follow-up or treatment.",

  // Risk tiers
  risk_low: "Low risk",
  risk_medium: "Medium risk",
  risk_high: "High risk",

  // Equity
  equity_gap_label: "Equity gap",
  equity_protected_label: "Demographic group",
  equity_tpr_label: "Detection rate (TPR)",
  equity_fpr_label: "False alarm rate (FPR)",
  equity_selection_label: "Positive rate",

  // PHI
  phi_warning_title: "Patient data privacy notice",
  phi_warning_body:
    "The following columns appear to contain Protected Health Information (PHI). " +
    "They will be excluded from AI analysis to protect patient privacy. " +
    "Review and confirm before proceeding.",

  // Clinical disclaimer (appended to all AI outputs)
  clinical_disclaimer:
    "⚕ This analysis is intended to assist clinical decision-making. " +
    "All AI-generated risk scores and recommendations must be reviewed " +
    "by a licensed clinician before clinical action is taken.",
} as const

/** Map raw severity strings to clinical-friendly labels. */
export const SEVERITY_LABEL: Record<string, string> = {
  none: "No disparity",
  mild: "Mild disparity",
  moderate: "Moderate disparity - review recommended",
  severe: "Severe disparity - clinical review required",
}

/** Risk tier thresholds. Defaults match a screening (FN-heavy cost) context. */
export const RISK_TIER = {
  low_max: 0.15,
  medium_max: 0.40,
} as const

export function riskTier(probability: number): "low" | "medium" | "high" {
  if (probability < RISK_TIER.low_max) return "low"
  if (probability < RISK_TIER.medium_max) return "medium"
  return "high"
}

export const RISK_TIER_CLASSES = {
  low: {
    badge: "bg-emerald-900/40 text-emerald-300 border border-emerald-700/50",
    label: TERM.risk_low,
  },
  medium: {
    badge: "bg-amber-900/40 text-amber-300 border border-amber-700/50",
    label: TERM.risk_medium,
  },
  high: {
    badge: "bg-red-900/40 text-red-300 border border-red-700/50",
    label: TERM.risk_high,
  },
} as const
