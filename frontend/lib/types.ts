/** Shared TypeScript types mirroring backend Pydantic models. */

// ── Domain models ──────────────────────────────────────────────────────────────

export interface CostMatrix {
  fp_cost: number
  fn_cost: number
  tp_value: number
  tn_value: number
}

export interface CaseBrief {
  raw_text: string
  source_files: string[]
  objectives: string[]
  cost_matrix: CostMatrix | null
  known_data_issues: string[]
  deliverable_requirements: string[]
  evaluation_criteria: string[]
  stakeholder_name: string | null
  stakeholder_role: string | null
  parsed: boolean
}

export interface BriefFileRef {
  filename: string
  storage_path: string
}

export interface Project {
  id: string
  user_id: string
  name: string
  description: string | null
  case_brief: CaseBrief | null
  brief_files: BriefFileRef[] | null
  created_at: string
  updated_at: string
}

export interface Dataset {
  id: string
  project_id: string
  role: "training" | "inference" | "holdout" | "reference" | "comparison"
  filename: string
  storage_path: string
  file_size_bytes: number | null
  sha256: string
  schema_hash: string
  row_count: number | null
  col_count: number | null
  target_column: string | null
  task_type: string | null
  profile: Record<string, unknown> | null
  created_at: string
}

export interface Run {
  id: string
  project_id: string
  training_dataset_id: string | null
  holdout_dataset_id: string | null
  job_id: string | null
  status: "queued" | "running" | "awaiting_checkpoint" | "completed" | "failed"
  current_step: string | null
  progress: number

  // Pipeline outputs (populated as each step completes)
  eda_report: Record<string, unknown> | null
  preprocessing_strategy: PreprocessingStrategy | null
  model_selection: ModelSelectionStrategy | null
  model_comparison: StabilityResult[] | null
  stat_tests: Record<string, unknown> | null
  best_model_name: string | null
  best_model_score: number | null
  tuning_result: Record<string, unknown> | null
  calibration_report: CalibrationReport | null
  threshold_result: ThresholdResult | null
  threshold_config: Record<string, unknown> | null
  final_metrics: Record<string, number> | null
  eval_plots: EvalPlots | null
  shap_summary: SHAPSummary | null
  similarity_index_built: boolean
  drift_report: DriftReport | null
  fairness_report: FairnessReport | null
  insight_report: string | null

  error_message: string | null
  created_at: string
  completed_at: string | null
}

// ── Pipeline strategy types ────────────────────────────────────────────────────

export interface ColumnStrategy {
  action: "keep" | "drop"
  impute_strategy: string | null
  encode_strategy: string | null
  scale_strategy: string | null
  dtype_hint: "numeric" | "categorical" | null
  reason: string
}

export interface PreprocessingStrategy {
  columns: Record<string, ColumnStrategy>
  target_column: string
  task_type: string
  drop_high_correlation: string[]
  notes: string | null
}

export interface ModelSelectionStrategy {
  candidates: string[]
  primary: string
  primary_metric: string
  excluded: Array<{ name: string; reason: string }>
  reasoning: string
  notes: string | null
}

export interface StabilityResult {
  model_name: string
  scores: number[]
  mean: number
  std: number
  train_scores: number[]
  train_mean: number
  overfit_gap: number
}

export interface CalibrationReport {
  method: string
  brier_before: number
  brier_after: number
  ece_before: number
  ece_after: number
  improvement_pct: number
}

export interface ThresholdResult {
  optimal_threshold: number
  cost_at_default: number
  cost_at_optimal: number
  improvement_pct: number
  metric_at_optimal: Record<string, number>
  note: string
}

export interface SHAPSummary {
  feature_names: string[]
  mean_abs_shap: number[]
  top_k_features: string[]
  explainer_type: string
  n_samples: number
}

// ── Eval plots (§14, §15) ──────────────────────────────────────────────────────

export interface RocCurveData {
  fpr: number[]
  tpr: number[]
}

export interface PrCurveData {
  precision: number[]
  recall: number[]
  ap: number
}

export interface ConfusionMatrixData {
  tn: number
  fp: number
  fn: number
  tp: number
}

export interface ScoreBin {
  score: number
  negative: number
  positive: number
}

export interface CalibrationCurveData {
  prob_true: number[]
  prob_pred: number[]
}

export interface ConfusionMatrixMultiData {
  matrix: number[][]
  classes: string[]
}

export interface ScatterPoint {
  actual: number
  predicted: number
}

export interface ResidualPoint {
  predicted: number
  residual: number
}

export interface EvalPlots {
  roc_curve?: RocCurveData
  pr_curve?: PrCurveData
  confusion_matrix?: ConfusionMatrixData
  score_distribution?: ScoreBin[]
  calibration_curve?: CalibrationCurveData
  confusion_matrix_multi?: ConfusionMatrixMultiData
  predicted_vs_actual?: ScatterPoint[]
  residuals?: ResidualPoint[]
}

// ── Drift & Fairness types (§17, §19) ─────────────────────────────────────────

export interface FeatureDriftResult {
  feature: string
  type: "numeric" | "categorical"
  psi: number | null
  ks_statistic: number | null
  ks_p_value: number | null
  wasserstein: number | null
  wasserstein_relative: number | null
  chi2: number | null
  chi2_p_value: number | null
  js_divergence: number | null
  severity: "stable" | "mild" | "significant"
}

export interface DriftReport {
  overall_severity: "stable" | "mild" | "significant"
  aggregate_psi: number
  features: FeatureDriftResult[]
  n_features_drifted: number
  significant_features: string[]
  n_train_rows: number
  n_new_rows: number
  warning: string | null
}

export interface GroupMetrics {
  group: string
  n_samples: number
  selection_rate: number
  true_positive_rate: number | null
  false_positive_rate: number | null
  precision: number | null
}

export interface AttributeFairnessResult {
  attribute: string
  demographic_parity_diff: number
  equalized_odds_diff: number | null
  equal_opportunity_diff: number | null
  by_group: GroupMetrics[]
  severity: "none" | "mild" | "moderate" | "severe"
  note: string
}

export interface FairnessReport {
  attributes: AttributeFairnessResult[]
  intersectional: AttributeFairnessResult[]
  overall_severity: "none" | "mild" | "moderate" | "severe"
  blocks_deliverables: boolean
  requires_acknowledgment: boolean
  acknowledged: boolean
}

// ── Deliverables (§4, §23) ────────────────────────────────────────────────────

export interface DeliverableItem {
  id: string
  name: string
  format: "pdf" | "xlsx" | "md" | "csv" | "json" | "yaml" | "ipynb" | "zip"
  storage_path: string
  checksum_sha256: string
  generator_version: string
  inputs_used: string[] | null
  audience: string | null
  generated_at: string | null
}

// ── Prediction (§26) ──────────────────────────────────────────────────────────

export interface PredictRequest {
  input_data: Record<string, unknown>
}

export interface PredictResponse {
  prediction: unknown
  probability: number | null
  threshold_used: number
  confidence_band: "high" | "medium" | "low"
  similarity_score: number | null
  shap_drivers: string[]
  shap_dampeners: string[]
  task_type: string
  prediction_id: string
}

export interface PredictionListItem {
  id: string
  prediction: Record<string, unknown>
  probability: number | null
  similarity_score: number | null
  confidence_band: string | null
  threshold_used: number | null
  shap_values: Record<string, unknown> | null
  risk_flag: boolean
  created_at: string
}

// ── Audit (§21, §24) ──────────────────────────────────────────────────────────

export interface AuditEvent {
  id: string
  seq: number
  timestamp: string
  actor: "ai" | "user" | "system"
  category: string
  action: string
  payload: Record<string, unknown>
  reason: string | null
  prev_hash: string
  self_hash: string
}

export interface AuditVerifyResult {
  run_id: string
  chain_valid: boolean
  total_events: number
  error: string | null
}

// ── Clinical / PHI types (§healthcare) ───────────────────────────────────────

export interface PhiColumnFlag {
  column: string
  confidence: "low" | "medium" | "high"
}

export interface ClinicalRangeFlag {
  reference_min: number | null
  reference_max: number | null
  unit: string
  pct_below_reference_min: number
  pct_above_reference_max: number
  pct_critical_range: number
  observed_min: number | null
  observed_max: number | null
  clinical_concern: boolean
}

// ── Dataset preview & plots (§D) ─────────────────────────────────────────────

export interface DatasetPreview {
  columns: string[]
  dtypes: Record<string, string>
  rows: Record<string, unknown>[]
  total_rows: number
}

export interface DatasetPlot {
  plot_id: string
  plot_type: string
  title: string
  column?: string | null
  stage: string
  priority: number
  image_b64?: string
}

// ── Multi-dataset joins (§7) ──────────────────────────────────────────────────

export interface JoinKeyCandidate {
  column: string
  left_unique: number
  right_unique: number
  overlap_pct: number
  recommended: boolean
}

export interface JoinSuggestResponse {
  left_dataset_id: string
  right_dataset_id: string
  candidates: JoinKeyCandidate[]
  recommended_join_type: string
  note: string
}

export interface JoinRecord {
  id: string
  left_dataset_id: string
  right_dataset_id: string
  result_dataset_id: string | null
  join_type: string
  join_keys: string[]
  rows_before_left: number | null
  rows_before_right: number | null
  rows_after: number | null
}

// ── Chat / SSE (existing) ──────────────────────────────────────────────────────

export type IntentType = "question" | "modify" | "abort" | "request_artifact" | "navigate"
export type IntentCategory =
  | "eda"
  | "preprocessing"
  | "model_selection"
  | "threshold"
  | "fairness"
  | "drift"
  | "deliverables"
  | "general"
  | "clinical_query"   // clinical domain question (ranges, terminology, interpretation)
  | "equity_query"     // fairness / demographic disparity question
  | "threshold_query"  // threshold / cost-matrix / sensitivity-specificity question

export interface ChatIntent {
  intent: IntentType
  confidence: number
  category: IntentCategory
  structured_payload: Record<string, unknown>
  needs_confirmation: boolean
  reasoning: string
}

export interface StrategyDiff {
  field_path: string
  before: unknown
  after: unknown
  summary: string
  run_id: string
}

export type SSEEventType = "text_chunk" | "strategy_diff" | "intent" | "artifact_task" | "error" | "done"

export interface SSETextChunk {
  type: "text_chunk"
  content: string
}

export interface SSEStrategyDiff {
  type: "strategy_diff"
  diffs: StrategyDiff[]
}

export interface SSEIntent {
  type: "intent"
  intent: ChatIntent
}

export interface SSEError {
  type: "error"
  error: string
}

export interface SSEArtifactTask {
  type: "artifact_task"
  task_id: string | null
  artifact_type: string
}

export interface SSEDone {
  type: "done"
}

export type SSEEvent = SSETextChunk | SSEStrategyDiff | SSEIntent | SSEArtifactTask | SSEError | SSEDone

export interface Message {
  id: string
  role: "user" | "assistant" | "system"
  content: string
  intent?: ChatIntent | null
  diffs?: StrategyDiff[] | null
  isStreaming?: boolean
}
