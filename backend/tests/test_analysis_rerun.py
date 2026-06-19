"""Tests for override-driven step re-run logic in tasks/analysis_task.py.

A chat override edits a decision field but only takes effect once the consuming
step re-runs. rerun_step_for_override decides which step (if any) must re-run,
firing only when the override invalidates ALREADY-computed state.
"""

from backend.tasks.analysis_task import rerun_step_for_override, resolve_training_primary


# ── resolve_training_primary: a user override is authoritative ─────────────────

_LEADERBOARD = ["xgboost", "logistic_regression", "lightgbm", "random_forest"]


def test_user_override_forces_primary_even_when_not_top():
    # The live bug: lightgbm was the leaderboard winner, but the user overrode to
    # logistic_regression. The override must win, not the leaderboard.
    sel = {"primary": "logistic_regression", "primary_source": "user_override"}
    assert resolve_training_primary(sel, _LEADERBOARD, fallback="xgboost") == "logistic_regression"


def test_user_override_honoured_even_if_not_trained():
    # Defensive: never silently revert a user's explicit pick to the leaderboard top.
    sel = {"primary": "logistic_regression", "primary_source": "user_override"}
    assert resolve_training_primary(sel, ["xgboost", "lightgbm"], fallback="xgboost") == "logistic_regression"


def test_agent_primary_used_when_trained():
    sel = {"primary": "lightgbm"}  # no override marker
    assert resolve_training_primary(sel, _LEADERBOARD, fallback="xgboost") == "lightgbm"


def test_agent_primary_falls_back_when_not_trained():
    # An agent recommendation that names an untrained model falls back to the top.
    sel = {"primary": "gradient_boosting"}  # not in leaderboard, not a user override
    assert resolve_training_primary(sel, _LEADERBOARD, fallback="xgboost") == "xgboost"


def test_missing_primary_uses_fallback():
    assert resolve_training_primary({}, _LEADERBOARD, fallback="xgboost") == "xgboost"


def test_model_selection_override_at_training_checkpoint_reruns_training():
    # The live case: user switches primary at the training-results checkpoint.
    assert rerun_step_for_override("model_selection", "checkpoint_4_training") == "training"


def test_model_selection_override_at_final_checkpoint_reruns_training():
    # Changing the model after everything ran invalidates the whole tail.
    assert rerun_step_for_override("model_selection", "checkpoint_5_final") == "training"


def test_model_selection_override_before_training_does_not_rerun():
    # At the model-selection checkpoint, training hasn't run yet - nothing stale;
    # the normal resume consumes the change.
    assert rerun_step_for_override("model_selection", "checkpoint_3_model_selection") is None


def test_preprocessing_override_reruns_producer_at_its_checkpoint():
    # A preprocessing override at checkpoint 2 must re-run the preprocessing
    # PRODUCER so the agent regenerates the strategy honouring the human's
    # directive. (Previously this returned None, which is why "drop both X and Y"
    # was only half-applied - the override never reached the agent.)
    assert rerun_step_for_override("preprocessing", "checkpoint_2_preprocessing") == "preprocessing"


def test_preprocessing_override_after_training_reruns_producer():
    # Even after training, a preprocessing change re-runs the producer and
    # re-pauses at checkpoint 2; resuming through it retrains (Rule 8).
    assert rerun_step_for_override("preprocessing", "checkpoint_4_training") == "preprocessing"
    assert rerun_step_for_override("preprocessing", "checkpoint_5_final") == "preprocessing"


def test_preprocessing_override_before_producer_does_not_rerun():
    # At the EDA checkpoint the preprocessing step has not run yet, so there is
    # nothing to regenerate; the normal resume will produce it.
    assert rerun_step_for_override("preprocessing", "checkpoint_1_eda") is None


def test_target_override_reruns_from_eda():
    # Target hygiene changes the task type + every downstream decision, so an
    # override re-runs from EDA and re-pauses at checkpoint 1, at any later stage.
    assert rerun_step_for_override("target", "checkpoint_2_preprocessing") == "eda"
    assert rerun_step_for_override("target", "checkpoint_5_final") == "eda"
    assert rerun_step_for_override("target", "checkpoint_1_eda") == "eda"


def test_threshold_override_at_final_checkpoint_reruns_tuning():
    assert rerun_step_for_override("threshold", "checkpoint_5_final") == "tuning"


def test_fairness_override_at_final_checkpoint_reruns_tuning():
    # Adding a protected attribute must re-run the step that does the fairness audit.
    assert rerun_step_for_override("fairness", "checkpoint_5_final") == "tuning"


def test_fairness_override_before_tuning_does_not_rerun():
    assert rerun_step_for_override("fairness", "checkpoint_4_training") is None


def test_threshold_override_before_tuning_does_not_rerun():
    # Threshold feeds step 5 (tuning/calibration/eval); at checkpoint 4 it's future state.
    assert rerun_step_for_override("threshold", "checkpoint_4_training") is None


def test_unknown_category_does_not_rerun():
    # Categories that drive no recompute (drift is informational; eda is upstream).
    assert rerun_step_for_override("drift", "checkpoint_5_final") is None
    assert rerun_step_for_override("eda", "checkpoint_1_eda") is None


def test_missing_or_unknown_step_is_safe():
    assert rerun_step_for_override("model_selection", None) is None
    assert rerun_step_for_override("model_selection", "nonsense_step") is None
