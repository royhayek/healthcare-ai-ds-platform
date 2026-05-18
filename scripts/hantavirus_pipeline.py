"""
Hantavirus Genome Dataset - Full ML Pipeline
=============================================

Project     : Hantavirus Pathogenicity Classifier  (leakage-corrected - see the README)
Description : Predict pathogenicity class (high / moderate / low) of hantavirus
              isolates from the GENOME SEQUENCE ITSELF (k-mer composition spectrum),
              with honest known-vs-novel evaluation and an abstention gate for
              genuinely novel virus families.

Why this differs from the first run
-----------------------------------
The first run reported ~100% accuracy. That was target leakage, not skill:
  * `clinical_syndrome` IS the label (1-to-1) and `clade` is a near-deterministic
    curated copy of it - handing either to the model just memorises a lookup table.
    Both are now EXCLUDED from the features.
  * "unknown" was not a danger tier - it was 39% missing labels. It is now excluded
    from the supervised problem and routed to the novelty gate.
  * The split shared virus families across train/test, so the score measured
    memorisation. We now report BOTH a within-family split (known viruses) and a
    clade-grouped split (genuinely novel families - the real outbreak scenario).

Business brief:
    A wrong HIGH→LOW miss costs ~$2M in under-prepared outbreak response.
    A LOW→HIGH false alarm costs ~$50K in unnecessary BSL-4 lab handling.
    For a genuinely novel family the correct action is to ABSTAIN ("manual BSL
    review"), never to emit a confident risk class.

Target column  : pathogenicity_class  (3 classes: high / moderate / low; "unknown" excluded)
Features       : genome k-mer spectrum + scalar sequence stats + weak metadata
                 (NO clade, NO clinical_syndrome)

Pipeline stages
---------------
1.  Load & describe dataset (drop "unknown" missing-label rows)
2.  Sequence feature engineering (GC content, length, entropy, k-mer spectrum)
3.  Leakage audit & feature selection (drop label-proxy columns)
4.  EDA: profile with backend profiler, print summary
5.  80/20 group split by isolate_id (segments stay together)
6.  Preprocessing (sklearn ColumnTransformer - imputation + scaling + OHE)
7.  Model training with stability runs (3 seeds × 5-fold CV per candidate)
8.  Leaderboard with mean ± std; stat-test when top-2 are within 0.005
9.  Fit final model on full training set
10. Calibration (Platt scaling)
11. SHAP feature importance on test set
12. Final test-set evaluation (known-family) - accuracy, macro-F1, AUC, confusion matrix
13. NOVEL-FAMILY evaluation (clade-grouped CV) - the honest novel-virus number
14. Novelty / out-of-distribution abstention gate
15. Prediction on held-out test rows with realistic interpretation
16. Save all plots to ./pipeline_plots/

Author: AI-DS Platform - Hantavirus Demo Run (corrected)
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── stdlib / third-party ───────────────────────────────────────────────────────
import json
import itertools
import math
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ── backend profiler (aggregate only, no raw rows to the model) ──────────────────
BACKEND = Path(__file__).parent / "backend"
sys.path.insert(0, str(Path(__file__).parent))
from backend.ml.profiler import compress_profile_for_claude, profile_dataset

# ── Output directory ───────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = _REPO_ROOT / "pipeline_plots"
PLOTS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 0, 1]
N_SPLITS = 5
TEST_SIZE = 0.20
TARGET = "pathogenicity_class"
CSV_PATH = _REPO_ROOT / "datasets" / "hantavirus_genome.csv"

# k for the genome k-mer composition spectrum (4^3 = 64 features). This is the
# real, leakage-free biological signal - see the README.
KMER_K = 3

# Rows labelled "unknown" are MISSING LABELS (39% of the raw data), not a danger
# tier. They are excluded from the supervised problem and routed to the novelty
# gate instead. The supervised task is a clean 3-class problem.
EXCLUDED_LABEL = "unknown"

# Ordered class labels (most → least dangerous)
CLASS_ORDER = ["high", "moderate", "low"]

# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 0: Project card
# ──────────────────────────────────────────────────────────────────────────────

def print_section(title: str, width: int = 72) -> None:
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_project_card() -> None:
    print_section("PROJECT CARD")
    card = textwrap.dedent(f"""
    Title       : Hantavirus Pathogenicity Classifier  (corrected, leakage-free)
    Dataset     : hiyata/hantavirus-genome-dataset (HuggingFace)
    Rows        : 2,096 genomic isolates  →  ~1,287 after dropping "unknown" labels
    Task        : 3-class classification  (high / moderate / low)
    Target      : pathogenicity_class    ("unknown" excluded - it is missing labels)
    Features    : genome {KMER_K}-mer spectrum + sequence stats + weak metadata
                  (clade & clinical_syndrome EXCLUDED - they leak the label)
    Business    : BSL-4 triage - mis-classifying a HIGH pathogen as LOW
                  triggers catastrophic outbreak response failure (~$2M cost).
    Split       : 80/20 group split by isolate_id  (+ clade-grouped novel-family eval)
    Candidates  : RandomForest · GradientBoosting · LogisticRegression
    Stability   : 3 seeds × 5-fold CV per candidate (§14)
    """)
    print(card)


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 1: Load
# ──────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print_section("SECTION 1 - LOAD DATASET")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns from {CSV_PATH.name}")

    # De-duplicate exact rows first
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dup = n_before - len(df)
    print(f"  Removed {n_dup} exact duplicate rows → {len(df):,} rows remaining")

    # Drop "unknown" - these are MISSING LABELS (no characterised pathogenicity),
    # not a danger tier. Keeping them inflates accuracy via a trivial
    # clade=="Unknown" → "unknown" lookup and is scientifically meaningless.
    n_unknown = int((df[TARGET] == EXCLUDED_LABEL).sum())
    df = df[df[TARGET] != EXCLUDED_LABEL].reset_index(drop=True)
    print(f"  Dropped {n_unknown} rows with label '{EXCLUDED_LABEL}' (missing labels, "
          f"not a class) → {len(df):,} labelled rows for the 3-class problem")

    print(f"\n  Target distribution ({TARGET}):")
    vc = df[TARGET].value_counts()
    for cls, n in vc.items():
        pct = 100 * n / len(df)
        print(f"    {cls:<12} {n:>5}  ({pct:.1f}%)")

    # Unique isolates (S, M, L segments share an isolate_id)
    n_isolates = df["isolate_id"].nunique()
    print(f"\n  Unique isolates (isolate_id): {n_isolates}")
    print(f"  Average segments per isolate : {len(df)/n_isolates:.2f}")
    print(f"\n  NOTE: S/M/L segments of the same isolate share identical metadata.")
    print(f"  We use GROUP-BASED splitting by isolate_id to prevent segment leakage.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 2: Sequence feature engineering
# ──────────────────────────────────────────────────────────────────────────────

# Canonical ordering of the 4^KMER_K k-mers over the ACGT alphabet. These names
# become the genome-composition feature columns (e.g. kmer_AAA … kmer_TTT).
KMER_VOCAB = ["".join(p) for p in itertools.product("ACGT", repeat=KMER_K)]
KMER_FEATURES = [f"kmer_{km}" for km in KMER_VOCAB]
_KMER_INDEX = {km: i for i, km in enumerate(KMER_VOCAB)}


def _kmer_entropy(seq: str, k: int = 3) -> float:
    """Shannon entropy of k-mer frequency distribution."""
    if len(seq) < k:
        return 0.0
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    probs = [v / total for v in counts.values()]
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def _kmer_spectrum(seq: str, k: int = KMER_K) -> np.ndarray:
    """Normalised k-mer frequency vector (length 4^k). Pure-ACGT k-mers only;
    rows are L1-normalised so genomes of different lengths are comparable."""
    vec = np.zeros(len(KMER_VOCAB), dtype=float)
    if len(seq) < k:
        return vec
    for i in range(len(seq) - k + 1):
        sub = seq[i : i + k]
        idx = _KMER_INDEX.get(sub)
        if idx is not None:  # skips k-mers containing N or other ambiguity codes
            vec[idx] += 1.0
    total = vec.sum()
    return vec / total if total > 0 else vec


def engineer_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("SECTION 2 - SEQUENCE FEATURE ENGINEERING")
    seq = df["sequence"].fillna("").astype(str).str.upper()

    df["seq_length"]    = seq.str.len()
    df["gc_content"]    = (seq.str.count("G") + seq.str.count("C")) / df["seq_length"].clip(lower=1)
    df["at_content"]    = (seq.str.count("A") + seq.str.count("T")) / df["seq_length"].clip(lower=1)
    df["n_fraction"]    = seq.str.count("N") / df["seq_length"].clip(lower=1)
    g = seq.str.count("G").astype(float)
    c = seq.str.count("C").astype(float)
    df["gc_skew"]       = (g - c) / (g + c + 1e-9)
    a = seq.str.count("A").astype(float)
    t = seq.str.count("T").astype(float)
    df["at_skew"]       = (a - t) / (a + t + 1e-9)
    df["kmer3_entropy"] = seq.apply(_kmer_entropy)

    print("  Derived scalar sequence features:")
    for feat in ["seq_length", "gc_content", "at_content", "n_fraction", "gc_skew", "at_skew", "kmer3_entropy"]:
        s = df[feat]
        print(f"    {feat:<18}  mean={s.mean():.4f}  std={s.std():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}")

    # Genome k-mer composition spectrum - the real, leakage-free biological signal.
    spectra = np.vstack([_kmer_spectrum(s) for s in seq])
    kmer_df = pd.DataFrame(spectra, columns=KMER_FEATURES, index=df.index)
    df = pd.concat([df, kmer_df], axis=1)
    print(f"\n  Derived genome {KMER_K}-mer spectrum: {len(KMER_FEATURES)} features "
          f"(kmer_{KMER_VOCAB[0]} … kmer_{KMER_VOCAB[-1]}), L1-normalised per genome")

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 3: Leakage audit & feature selection
# ──────────────────────────────────────────────────────────────────────────────

# Features that directly encode the target (leakage). VERIFIED against the data:
#   * clinical_syndrome maps 1-to-1 with pathogenicity_class
#       (HFRS/HPS→high, mild→moderate, non-human→low, unknown→unknown)
#   * clade is a near-deterministic curated copy of the label: 24 of 25 clades map
#     to a single pathogenicity level, so it lets the model memorise a lookup table.
# Both are EXCLUDED from the features - see the README (target-leakage section).
LEAKY_COLS = [
    "clinical_syndrome",   # direct label synonym
    "clade",               # ~deterministic label proxy (kept only as a CV group key)
    "reservoir_host",      # same info as host_category but row-level
    "standardized_host",   # redundant with host_category
]

# High-cardinality string columns with no predictive signal beyond genus/clade
DROP_COLS = [
    "sequence",            # replaced by engineered features
    "accession",           # identifier
    "strain_name",         # near-unique free-text
    "virus_name",          # 149 values; genus captures this
    "isolate_id",          # identifier
    "matched_accessions",  # identifier list
    "isolation_date",      # would need temporal encoding; skip for now
    "location",            # 116 values; geographic noise
    "host",                # replaced by standardized_host → host_category
    "standardized_location",# redundant with location
    "coding_completeness",  # stringified dict
    "gemini_annotated",    # annotation metadata, not biology
]

# Final feature columns (computed after all drops/engineers).
# NOTE: clade is deliberately NOT here - it leaks the label (see LEAKY_COLS).
CAT_FEATURES = [
    "segment",
    "genus",
    "host_category",
    "reservoir_confidence",
    "collection_method",
]

# Scalar sequence + completeness features (printed individually in EDA).
NUM_FEATURES = [
    "seq_length",
    "gc_content",
    "at_content",
    "n_fraction",
    "gc_skew",
    "at_skew",
    "kmer3_entropy",
    "segment_completeness",
]

BOOL_FEATURES = [
    "zoonotic",
    "outbreak_associated",
    "reassortment_suspected",
    "is_complete_segment",
    "has_all_segments",
    "lab_culture",
    "wastewater_sewage",
]

# All numeric inputs to the model = scalar stats + the k-mer composition spectrum.
NUMERIC_ALL = NUM_FEATURES + KMER_FEATURES

# clade is retained ONLY as a grouping key for the novel-family CV, never as a feature.
GROUP_COLS = ["isolate_id", "clade"]

ALL_FEATURES = NUMERIC_ALL + BOOL_FEATURES + CAT_FEATURES


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("SECTION 3 - LEAKAGE AUDIT & FEATURE SELECTION")
    print(f"\n  Dropping leaky columns : {LEAKY_COLS}")
    print(f"  Dropping noise columns : {DROP_COLS}")
    print(f"  Keeping as CV groups   : {GROUP_COLS}  (NOT used as model features)")
    # Keep isolate_id + clade as group columns for splitting (not features).
    keep = [TARGET] + GROUP_COLS + ALL_FEATURES
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df2 = df[keep].copy()

    # Cast booleans
    for c in BOOL_FEATURES:
        df2[c] = df2[c].astype(float)

    print(f"\n  Retained {len(ALL_FEATURES)} model features:")
    print(f"    Numeric scalar ({len(NUM_FEATURES)}): {NUM_FEATURES}")
    print(f"    Genome k-mers  ({len(KMER_FEATURES)}): kmer_{KMER_VOCAB[0]} … kmer_{KMER_VOCAB[-1]}")
    print(f"    Boolean        ({len(BOOL_FEATURES)}): {BOOL_FEATURES}")
    print(f"    Categorical    ({len(CAT_FEATURES)}): {CAT_FEATURES}")

    print(f"\n  Nulls per retained column (excluding k-mer block):")
    nulls = df2.drop(columns=KMER_FEATURES).isnull().sum()
    for col, n in nulls[nulls > 0].items():
        print(f"    {col}: {n} ({100*n/len(df2):.1f}%)")
    if nulls.sum() == 0:
        print("    (none)")

    return df2


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 4: EDA - backend profiler
# ──────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    print_section("SECTION 4 - EXPLORATORY DATA ANALYSIS (Backend Profiler)")
    profile = profile_dataset(df, target_column=TARGET)
    compressed = compress_profile_for_claude(profile)

    print(f"\n  Dataset shape   : {profile.n_rows} rows × {profile.n_cols} columns")
    print(f"  Duplicates      : {profile.duplicate_count}")
    print(f"  Task type       : {profile.task_type}")
    print(f"  Numeric cols    : {profile.numeric_columns}")
    print(f"  Categorical cols: {profile.categorical_columns}")

    if profile.high_correlation_pairs:
        print(f"\n  High-correlation pairs (|r| ≥ 0.85):")
        for pair in profile.high_correlation_pairs[:5]:
            print(f"    {pair['col_a']} ↔ {pair['col_b']}  r={pair['correlation']:.3f}")
    else:
        print("\n  No high-correlation pairs (|r| ≥ 0.85)")

    if profile.vif:
        high_vif = {k: v for k, v in profile.vif.items() if v > 5}
        if high_vif:
            print(f"\n  High-VIF features (>5) - multicollinearity warning:")
            for col, v in sorted(high_vif.items(), key=lambda x: -x[1]):
                print(f"    {col}: {v:.1f}")

    if profile.isolation_score_summary:
        iso = profile.isolation_score_summary
        print(f"\n  Isolation Forest anomaly score (higher = more normal):")
        print(f"    p5={iso.get('p5')}  p50={iso.get('p50')}  p95={iso.get('p95')}")
        print(f"    Rough outlier fraction: {iso.get('outlier_pct_rough', 0):.1%}")

    # --- Plot: class distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vc = df[TARGET].value_counts().reindex(CLASS_ORDER).dropna()
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#95a5a6"]
    vc.plot(kind="bar", ax=axes[0], color=colors[:len(vc)], edgecolor="black", alpha=0.85)
    axes[0].set_title("Pathogenicity Class Distribution", fontweight="bold")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    for i, (cls, n) in enumerate(vc.items()):
        axes[0].text(i, n + 5, str(n), ha="center", fontsize=9)

    # --- Plot: GC content by class ---
    for i, cls in enumerate([c for c in CLASS_ORDER if c in df[TARGET].unique()]):
        subset = df[df[TARGET] == cls]["gc_content"].dropna()
        axes[1].hist(subset, bins=25, alpha=0.55, label=cls, color=colors[i])
    axes[1].set_title("GC Content Distribution by Pathogenicity Class", fontweight="bold")
    axes[1].set_xlabel("GC Content")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    p = PLOTS_DIR / "01_eda_class_distribution.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"\n  [PLOT] Saved: {p}")

    # --- Plot: sequence length by class ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data_by_class = [df[df[TARGET] == cls]["seq_length"].dropna().values
                     for cls in CLASS_ORDER if cls in df[TARGET].unique()]
    labels_present = [cls for cls in CLASS_ORDER if cls in df[TARGET].unique()]
    axes[0].boxplot(data_by_class, labels=labels_present, patch_artist=True,
                    boxprops=dict(facecolor="#3498db", alpha=0.5))
    axes[0].set_title("Sequence Length by Pathogenicity Class", fontweight="bold")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Sequence Length (bp)")

    # k-mer entropy by class
    data_kmer = [df[df[TARGET] == cls]["kmer3_entropy"].dropna().values
                 for cls in CLASS_ORDER if cls in df[TARGET].unique()]
    axes[1].boxplot(data_kmer, labels=labels_present, patch_artist=True,
                    boxprops=dict(facecolor="#9b59b6", alpha=0.5))
    axes[1].set_title("3-mer Entropy by Pathogenicity Class", fontweight="bold")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Shannon Entropy (bits)")

    plt.tight_layout()
    p = PLOTS_DIR / "02_eda_sequence_features.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  [PLOT] Saved: {p}")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 5: Train/Test Split
# ──────────────────────────────────────────────────────────────────────────────

def make_split(df: pd.DataFrame):
    print_section("SECTION 5 - TRAIN / TEST SPLIT  (80 / 20, group-based by isolate_id)")
    print("  Group split ensures all segments of the same isolate stay in the same fold.")
    print("  This prevents the model from seeing isolate metadata in both train and test.\n")

    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_raw = df[TARGET].copy()
    y = pd.Series(le.transform(y_raw), name=TARGET, index=df.index)
    groups = df["isolate_id"].values
    # Drop target + group keys (isolate_id, clade) - groups are not model features.
    X = df.drop(columns=[TARGET] + GROUP_COLS)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    print(f"  Train  : {len(X_train):>5} rows ({100*len(X_train)/len(X):.1f}%)  "
          f"from {len(set(groups[train_idx]))} unique isolates")
    print(f"  Test   : {len(X_test):>5} rows ({100*len(X_test)/len(X):.1f}%)  "
          f"from {len(set(groups[test_idx]))} unique isolates")
    print(f"\n  Isolate overlap check: "
          f"{len(set(groups[train_idx]) & set(groups[test_idx]))} shared isolates "
          f"(must be 0)")

    print(f"\n  Class label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    print("\n  Train class distribution:")
    for cls_id, cls_name in enumerate(le.classes_):
        n = (y_train == cls_id).sum()
        pct = 100 * n / len(y_train)
        print(f"    [{cls_id}] {cls_name:<12} {n:>4} ({pct:.1f}%)")

    print("\n  Test class distribution:")
    for cls_id, cls_name in enumerate(le.classes_):
        n = (y_test == cls_id).sum()
        pct = 100 * n / len(y_test)
        print(f"    [{cls_id}] {cls_name:<12} {n:>4} ({pct:.1f}%)")

    return X_train, X_test, y_train, y_test, le


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 6: Preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    bool_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num",  numeric_pipe, NUMERIC_ALL),
        ("bool", bool_pipe,    BOOL_FEATURES),
        ("cat",  cat_pipe,     CAT_FEATURES),
    ], remainder="drop")


def describe_preprocessor() -> None:
    print_section("SECTION 6 - PREPROCESSING PIPELINE")
    print("  Numeric  : median imputation → StandardScaler")
    print("  Boolean  : mode imputation   → pass-through (already 0/1)")
    print("  Categori.: 'missing' fill    → OneHotEncoder (handle_unknown=ignore)")
    print("  Reminder : preprocessor is fit on TRAIN ONLY - no leakage into test")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 7: Stability training
# ──────────────────────────────────────────────────────────────────────────────

CANDIDATES = {
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced"
    ),
    "GradientBoosting": lambda seed: GradientBoostingClassifier(
        n_estimators=200, random_state=seed
    ),
    "LogisticRegression": lambda seed: LogisticRegression(
        max_iter=1000, random_state=seed, class_weight="balanced",
        solver="lbfgs"
    ),
}


def _cv_one_seed(
    name: str,
    make_est,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> dict:
    import copy
    prep = copy.deepcopy(preprocessor)
    est = make_est(seed)
    pipe = Pipeline([("prep", prep), ("model", est)])
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    res = cross_validate(
        pipe, X_train, y_train,
        cv=cv,
        scoring={
            "roc_auc_ovr": "roc_auc_ovr_weighted",
            "f1_macro":    "f1_macro",
            "accuracy":    "accuracy",
        },
        return_train_score=True,
        n_jobs=1,
    )
    return {
        "name":        name,
        "seed":        seed,
        "val_auc":     res["test_roc_auc_ovr"].tolist(),
        "train_auc":   res["train_roc_auc_ovr"].tolist(),
        "val_f1":      res["test_f1_macro"].tolist(),
        "val_acc":     res["test_accuracy"].tolist(),
    }


def train_all_candidates(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[dict]:
    print_section("SECTION 7 - STABILITY TRAINING  (3 seeds × 5-fold CV per candidate)")
    all_results = []
    for name, make_est in CANDIDATES.items():
        seed_results = []
        for seed in SEEDS:
            print(f"  {name}  seed={seed} …", end=" ", flush=True)
            r = _cv_one_seed(name, make_est, preprocessor, X_train, y_train, seed)
            seed_results.append(r)
            mean_val = np.mean(r["val_auc"])
            print(f"val_AUC={mean_val:.4f}")

        all_val  = [s for r in seed_results for s in r["val_auc"]]
        all_train= [s for r in seed_results for s in r["train_auc"]]
        all_f1   = [s for r in seed_results for s in r["val_f1"]]
        all_acc  = [s for r in seed_results for s in r["val_acc"]]

        summary = {
            "name":        name,
            "val_auc_mean":  float(np.mean(all_val)),
            "val_auc_std":   float(np.std(all_val)),
            "train_auc_mean":float(np.mean(all_train)),
            "overfit_gap":   max(0.0, float(np.mean(all_train)) - float(np.mean(all_val))),
            "val_f1_mean":   float(np.mean(all_f1)),
            "val_f1_std":    float(np.std(all_f1)),
            "val_acc_mean":  float(np.mean(all_acc)),
            "all_val_auc":   all_val,
            "seed_results":  seed_results,
        }
        all_results.append(summary)

    all_results.sort(key=lambda r: r["val_auc_mean"], reverse=True)
    return all_results


def print_leaderboard(results: list[dict]) -> None:
    print_section("SECTION 7b - LEADERBOARD")
    print(f"\n  {'Model':<22} {'Val AUC':>12} {'±std':>8} {'Train AUC':>12} {'Overfit':>10} {'Val F1':>10}")
    print("  " + "-" * 78)
    for r in results:
        print(
            f"  {r['name']:<22}"
            f"  {r['val_auc_mean']:.4f}      "
            f"±{r['val_auc_std']:.4f}  "
            f"{r['train_auc_mean']:.4f}    "
            f"{r['overfit_gap']:>+.4f}   "
            f"{r['val_f1_mean']:.4f}"
        )

    best = results[0]
    print(f"\n  Winner: {best['name']}")
    print(f"  CV AUC  (OvR weighted): {best['val_auc_mean']:.4f} ± {best['val_auc_std']:.4f}")
    print(f"  CV F1   (macro)       : {best['val_f1_mean']:.4f} ± {best['val_f1_std']:.4f}")
    print(f"  Overfit gap           : {best['overfit_gap']:.4f}")

    # Overfitting interpretation
    gap = best["overfit_gap"]
    if gap < 0.02:
        verdict = "GOOD - minimal train/val gap, model generalizes well"
    elif gap < 0.06:
        verdict = "ACCEPTABLE - moderate gap, within typical tree-model range"
    elif gap < 0.12:
        verdict = "WATCH - notable overfitting; consider regularization or fewer trees"
    else:
        verdict = "OVERFIT - train/val gap is large; reduce complexity"
    print(f"\n  Overfitting verdict: {verdict}")

    # Check top-2 closeness for stat test
    if len(results) >= 2:
        gap_12 = abs(results[0]["val_auc_mean"] - results[1]["val_auc_mean"])
        if gap_12 <= 0.005:
            from scipy.stats import wilcoxon
            try:
                stat, pval = wilcoxon(results[0]["all_val_auc"], results[1]["all_val_auc"])
                print(f"\n  Top-2 within 0.005 AUC - Wilcoxon signed-rank test:")
                print(f"    {results[0]['name']} vs {results[1]['name']}: "
                      f"p={pval:.4f} ({'significant' if pval < 0.05 else 'not significant'})")
            except Exception:
                pass

    # --- Plot: stability leaderboard ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names  = [r["name"] for r in results]
    means  = [r["val_auc_mean"] for r in results]
    stds   = [r["val_auc_std"]  for r in results]
    trains = [r["train_auc_mean"] for r in results]

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, trains, w, label="Train AUC", color="#3498db", alpha=0.8)
    axes[0].bar(x + w/2, means,  w, yerr=stds, label="Val AUC ± std",
                color="#e74c3c", alpha=0.8, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylabel("AUC (OvR Weighted)")
    axes[0].set_title("Stability Run Results: Train vs Val AUC", fontweight="bold")
    axes[0].legend()
    axes[0].set_ylim(0.5, 1.05)

    # Per-seed boxplot for best model
    best_seed_data = [r["val_auc"] for r in results[0]["seed_results"]]
    axes[1].boxplot(best_seed_data, labels=[f"seed={s}" for s in SEEDS], patch_artist=True,
                    boxprops=dict(facecolor="#2ecc71", alpha=0.6))
    axes[1].set_title(f"{results[0]['name']}: Val AUC per Seed (5 folds each)", fontweight="bold")
    axes[1].set_ylabel("Fold AUC")
    axes[1].set_ylim(0.5, 1.05)

    plt.tight_layout()
    p = PLOTS_DIR / "03_stability_leaderboard.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"\n  [PLOT] Saved: {p}")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 8: Fit final model on full training set
# ──────────────────────────────────────────────────────────────────────────────

def fit_final_model(
    best_name: str,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    print_section("SECTION 8 - FIT FINAL MODEL ON FULL TRAINING SET")
    import copy
    prep = copy.deepcopy(preprocessor)
    est = CANDIDATES[best_name](42)
    pipe = Pipeline([("prep", prep), ("model", est)])
    pipe.fit(X_train, y_train)
    print(f"  Fitted {best_name} on {len(X_train)} training samples")
    return pipe


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 9: Calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_model(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CalibratedClassifierCV:
    print_section("SECTION 9 - PROBABILITY CALIBRATION  (Platt scaling, cv=3)")
    # Wrap the already-fitted pipeline in a calibrator using cross-val
    # We re-fit on train so calibration doesn't touch test set.
    import copy
    prep = copy.deepcopy(pipe.named_steps["prep"])
    best_name = type(pipe.named_steps["model"]).__name__

    # Map sklearn class name → our candidate key
    name_map = {
        "RandomForestClassifier":    "RandomForest",
        "GradientBoostingClassifier":"GradientBoosting",
        "LogisticRegression":        "LogisticRegression",
    }
    cand_name = name_map.get(best_name, list(CANDIDATES.keys())[0])
    est = CANDIDATES[cand_name](42)
    inner_pipe = Pipeline([("prep", copy.deepcopy(prep)), ("model", est)])

    calibrated = CalibratedClassifierCV(inner_pipe, cv=3, method="sigmoid")
    calibrated.fit(X_train, y_train)

    # Calibration curve (macro average over classes)
    probs = calibrated.predict_proba(X_train)
    print(f"  Calibrated probabilities shape: {probs.shape}")
    print(f"  Mean predicted probability by class: "
          f"{dict(zip(range(probs.shape[1]), probs.mean(axis=0).round(3)))}")
    return calibrated


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 10: SHAP feature importance
# ──────────────────────────────────────────────────────────────────────────────

def run_shap(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
) -> None:
    print_section("SECTION 10 - SHAP FEATURE IMPORTANCE  (test set only)")
    try:
        import shap
    except ImportError:
        print("  [SKIP] shap not installed - pip install shap")
        return

    # Get transformed feature names
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    # Build feature names post-OHE (numeric scalar + k-mer block, then bools, then OHE cats)
    num_names = NUMERIC_ALL[:]
    bool_names = BOOL_FEATURES[:]
    cat_transformer = prep.named_transformers_["cat"]
    ohe = cat_transformer.named_steps["ohe"]
    cat_names = ohe.get_feature_names_out(CAT_FEATURES).tolist()
    all_names = num_names + bool_names + cat_names

    X_test_t = prep.transform(X_test)

    # Sample for speed (PermutationExplainer cost scales with the feature count,
    # which is larger now that the k-mer spectrum is included).
    n_sample = min(150, len(X_test_t))
    idx = np.random.RandomState(42).choice(len(X_test_t), n_sample, replace=False)
    X_sample = X_test_t[idx]

    # shap.Explainer auto-selects the best method.
    # For multiclass GradientBoosting it falls back to PermutationExplainer.
    # We cap background at 30 rows to keep runtime reasonable with ~75 features.
    background = shap.sample(pd.DataFrame(X_test_t, columns=all_names), 30,
                              random_state=42)
    explainer = shap.Explainer(model.predict_proba, background,
                                output_names=le.classes_.tolist())
    shap_values_obj = explainer(pd.DataFrame(X_sample, columns=all_names))
    # shape: (n_samples, n_features, n_classes)
    shap_arr = shap_values_obj.values
    if shap_arr.ndim == 3:
        mean_abs = np.abs(shap_arr).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_arr).mean(axis=0)
    shap_values = None  # sentinel - we already have mean_abs

    # Top-15 features
    top_idx = np.argsort(mean_abs)[::-1][:15]
    top_names = [all_names[i] for i in top_idx]
    top_vals  = mean_abs[top_idx]

    print("\n  Top 15 features by mean |SHAP| (averaged across classes):")
    for rank, (nm, val) in enumerate(zip(top_names, top_vals), 1):
        bar = "█" * int(val / top_vals[0] * 20)
        print(f"    {rank:>2}. {nm:<35} {val:.5f}  {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top_names)))
    ax.barh(range(len(top_names))[::-1], top_vals, color=colors_bar, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (across classes)")
    ax.set_title("SHAP Feature Importance - Hantavirus Pathogenicity Classifier", fontweight="bold")
    plt.tight_layout()
    p = PLOTS_DIR / "04_shap_importance.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"\n  [PLOT] Saved: {p}")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 11: Final test-set evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_test_set(
    model,          # fitted pipeline or calibrated
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
    best_name: str,
) -> dict:
    print_section("SECTION 11 - FINAL TEST-SET EVALUATION  (held-out 20%)")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc   = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

    print(f"\n  Best model   : {best_name}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {f1:.4f}")
    print(f"  Weighted AUC : {auc:.4f}")

    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0,
                                 digits=4))

    # --- Confusion matrix ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=le.classes_,
        colorbar=False, ax=axes[0]
    )
    axes[0].set_title(f"Confusion Matrix - {best_name}", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)

    # --- ROC curves (OvR per class) ---
    colors_roc = ["#e74c3c", "#e67e22", "#2ecc71", "#95a5a6"]
    for cls_id in range(len(le.classes_)):
        y_bin = (y_test == cls_id).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, cls_id])
        cls_auc = roc_auc_score(y_bin, y_prob[:, cls_id])
        axes[1].plot(fpr, tpr, lw=2, color=colors_roc[cls_id % len(colors_roc)],
                     label=f"{le.classes_[cls_id]} (AUC={cls_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves (One-vs-Rest per class)", fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    p = PLOTS_DIR / "05_test_evaluation.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"\n  [PLOT] Saved: {p}")

    return {"accuracy": acc, "macro_f1": f1, "weighted_auc": auc}


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 12: Overfitting deep-dive
# ──────────────────────────────────────────────────────────────────────────────

def overfitting_analysis(results: list[dict]) -> None:
    print_section("SECTION 12 - OVERFITTING ANALYSIS")
    print(f"\n  {'Model':<22} {'Train AUC':>12} {'Val AUC':>12} {'Gap':>10}  {'Status'}")
    print("  " + "-" * 72)

    for r in results:
        gap = r["overfit_gap"]
        if gap < 0.02:
            status = "✓  Healthy"
        elif gap < 0.06:
            status = "~  Mild overfit"
        elif gap < 0.12:
            status = "⚠  Moderate overfit"
        else:
            status = "✗  Severe overfit"
        print(f"  {r['name']:<22}  {r['train_auc_mean']:.4f}       "
              f"{r['val_auc_mean']:.4f}     {gap:+.4f}   {status}")

    best = results[0]
    print(f"\n  Interpretation (best model: {best['name']}):")
    gap = best["overfit_gap"]
    auc = best["val_auc_mean"]
    std = best["val_auc_std"]

    if auc > 0.95 and gap < 0.03:
        print("  → High AUC with low variance and minimal overfit.")
        print("    The model is learning genuine biological structure from the features.")
        print("    Genomic and epidemiological signals are clearly discriminative.")
    elif auc > 0.85:
        print("  → Strong AUC with acceptable stability.")
        print("    Feature engineering (GC content, k-mer entropy, segment type)")
        print("    is capturing meaningful pathogenicity signals.")
    elif auc > 0.70:
        print("  → Moderate performance. Consider:")
        print("    1. Adding k-mer features (4-mers, codon usage)")
        print("    2. Including geographic/temporal features")
        print("    3. A deeper tree (n_estimators=500)")
    else:
        print("  → Below 0.70 AUC - model may be struggling with the 'unknown' class.")
        print("    Consider binary classification (high vs not-high) instead.")

    if std > 0.05:
        print(f"\n  WARNING: High fold variance (±{std:.4f}).")
        print("    This suggests instability - likely driven by the small dataset size.")
        print("    With only ~1,600 training rows, some folds may hit rare classes.")
    else:
        print(f"\n  Stability check: fold variance ±{std:.4f} - within acceptable range.")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 13: Predictions on held-out test rows
# ──────────────────────────────────────────────────────────────────────────────

def predict_and_interpret(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
    df_original: pd.DataFrame,
) -> None:
    print_section("SECTION 13 - PREDICTIONS ON HELD-OUT TEST SET")

    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    cls_idx = {cls: i for i, cls in enumerate(le.classes_)}

    print(f"\n  Showing 15 representative predictions from the test set:\n")
    prob_cols = "".join(f"{'P('+c[:4]+')':>9}" for c in le.classes_)
    header = (f"  {'#':>3}  {'True':>10}  {'Predicted':>10}  {prob_cols}  "
              f"{'Correct':>8}  Notes")
    print(header)
    print("  " + "-" * 110)

    # Sample: 5 correct HIGH, 5 correct other, 5 mistakes
    high_class = cls_idx.get("high", 0)
    mask_array = np.array(y_test)
    pred_array = np.array(y_pred)
    correct_mask  = pred_array == mask_array
    high_mask     = mask_array == high_class

    correct_high_pos  = np.where(correct_mask & high_mask)[0][:5]
    correct_other_pos = np.where(correct_mask & ~high_mask)[0][:5]
    mistake_pos       = np.where(~correct_mask)[0][:5]
    show_pos = list(correct_high_pos) + list(correct_other_pos) + list(mistake_pos)

    for i, pos in enumerate(show_pos, 1):
        true_lbl = le.classes_[mask_array[pos]]
        pred_lbl = le.classes_[pred_array[pos]]
        probs    = y_prob[pos]
        correct  = "✓" if pred_array[pos] == mask_array[pos] else "✗ MISS"
        # Pull metadata from test row (clade is intentionally NOT a feature any more)
        try:
            row = X_test.iloc[pos]
            notes = (f"genus={row.get('genus','?')}  "
                     f"seg={row.get('segment','?')}  "
                     f"gc={float(row.get('gc_content', float('nan'))):.3f}")
        except Exception:
            notes = ""
        prob_str = "".join(f"{p:>9.3f}" for p in probs)
        print(f"  {i:>3}  {true_lbl:>10}  {pred_lbl:>10}  {prob_str}  "
              f"{correct:>8}  {notes}")

    # Realistic interpretation
    print(f"\n  Clinical interpretation of predictions:")
    high_id = cls_idx.get("high", 0)
    n_high_pred  = (pred_array == high_id).sum()
    n_high_true  = (mask_array == high_id).sum()
    missed_high  = ((pred_array != high_id) & (mask_array == high_id)).sum()

    print(f"    High-pathogenicity isolates in test  : {n_high_true}")
    print(f"    Predicted as high                    : {n_high_pred}")
    print(f"    Missed (FN - high predicted as other): {missed_high}  "
          f"({100*missed_high/max(n_high_true,1):.1f}% miss rate)")

    if missed_high == 0:
        print("    → PERFECT recall on HIGH class. No dangerous isolates missed.")
    elif missed_high <= 2:
        print("    → Near-perfect recall. A small number of misses warrants human review.")
    else:
        print("    → HIGH class recall needs improvement before clinical deployment.")
        print("      Consider SMOTE, cost-sensitive loss, or threshold optimization.")

    # Probability histogram plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_roc = ["#e74c3c", "#e67e22", "#2ecc71", "#95a5a6"]
    for cls_id, cls_name in enumerate(le.classes_):
        axes[0].hist(y_prob[:, cls_id], bins=20, alpha=0.55,
                     label=cls_name, color=colors_roc[cls_id % len(colors_roc)])
    axes[0].set_title("Predicted Probability Distributions (test set)", fontweight="bold")
    axes[0].set_xlabel("P(class)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Confidence histogram: max prob over classes
    max_prob = y_prob.max(axis=1)
    axes[1].hist(max_prob, bins=20, color="#3498db", edgecolor="black", alpha=0.8)
    axes[1].axvline(0.7, color="red", ls="--", label="Confidence threshold 0.70")
    axes[1].set_title("Model Confidence (max class probability)", fontweight="bold")
    axes[1].set_xlabel("Max P(class)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    p = PLOTS_DIR / "06_predictions.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"\n  [PLOT] Saved: {p}")


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 13b: Novel-family evaluation (the honest novel-virus number)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_novel_family(
    df_feat: pd.DataFrame,
    best_name: str,
    le: LabelEncoder,
    known_family_acc: float,
) -> dict:
    """Clade-grouped cross-validation: hold out ENTIRE virus families so no relative
    is ever seen in training. This is the honest estimate for the stated business
    case - a genuinely new virus arriving during an outbreak."""
    print_section("SECTION 13b - NOVEL-FAMILY EVALUATION  (clade-grouped CV)")
    print("  The real outbreak scenario: a GENUINELY NEW virus family arrives.")
    print("  We hold out entire clades, so the model has never seen a relative.\n")

    X = df_feat.drop(columns=[TARGET] + GROUP_COLS)
    y = pd.Series(le.transform(df_feat[TARGET]), index=df_feat.index).reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = df_feat["clade"].to_numpy()
    n_clades = int(pd.Series(groups).nunique())
    n_splits = min(5, n_clades)

    preds = np.empty(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        prep = build_preprocessor()
        est = CANDIDATES[best_name](42)
        pipe = Pipeline([("prep", prep), ("model", est)])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = pipe.predict(X.iloc[te])

    acc = accuracy_score(y, preds)
    bal = balanced_accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    baseline = float(y.value_counts(normalize=True).max())

    print(f"  Clades (groups)         : {n_clades}   ({n_splits}-fold clade-grouped CV)")
    print(f"  Majority-class baseline : {baseline:.4f}")
    print(f"  Novel-family accuracy   : {acc:.4f}")
    print(f"  Balanced accuracy       : {bal:.4f}")
    print(f"  Macro F1                : {f1:.4f}")
    if acc <= baseline + 0.05:
        print("\n  → HONEST CONCLUSION: on genuinely novel families the model is at or")
        print("    below the majority baseline. Pathogenicity of an unseen family CANNOT")
        print("    be reliably extrapolated from sequence - a scientific limit, not a")
        print("    code bug. The correct response is to ABSTAIN (next section).")
    else:
        print("\n  → Some generalisation to novel families, but far below the known-family")
        print("    number. Novel-family predictions must be treated as advisory only.")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ["Majority\nbaseline", "Known family\n(isolate split)", "Novel family\n(clade split)"]
    vals = [baseline, known_family_acc, acc]
    colors = ["#95a5a6", "#2ecc71", "#e74c3c"]
    ax.bar(bars, vals, color=colors, edgecolor="black", alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.axhline(baseline, ls="--", color="#7f8c8d", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Honest accuracy - known vs novel virus families", fontweight="bold")
    plt.tight_layout()
    p = PLOTS_DIR / "07_known_vs_novel.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"\n  [PLOT] Saved: {p}")

    return {"accuracy": acc, "balanced_accuracy": bal, "macro_f1": f1,
            "baseline": baseline, "n_clades": n_clades}


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 13c: Novelty / out-of-distribution abstention gate
# ──────────────────────────────────────────────────────────────────────────────

def novelty_abstention_gate(
    final_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
) -> dict:
    """Flag genuinely novel sequences for manual review instead of forcing a class.
    Distance to the nearest training isolate in feature space is the novelty score;
    isolates beyond the 99th-percentile training distance are abstained on."""
    print_section("SECTION 13c - NOVELTY / OUT-OF-DISTRIBUTION ABSTENTION GATE")
    print("  A genuinely novel sequence must be FLAGGED for manual BSL review, not")
    print("  given a confident class. Novelty = distance to the nearest TRAINING")
    print("  isolate in feature space; outliers are abstained on.\n")

    prep = final_pipe.named_steps["prep"]
    Xtr = prep.transform(X_train)
    Xte = prep.transform(X_test)

    nn = NearestNeighbors(n_neighbors=2).fit(Xtr)
    d_tr, _ = nn.kneighbors(Xtr)
    train_nn = d_tr[:, 1]               # 2nd neighbour = nearest OTHER train isolate
    thr = float(np.quantile(train_nn, 0.99))
    d_te, _ = nn.kneighbors(Xte, n_neighbors=1)
    test_nn = d_te[:, 0]
    abstain = test_nn > thr
    n_ab = int(abstain.sum())

    print(f"  Novelty threshold (99th pct of train NN distance): {thr:.3f}")
    print(f"  Test isolates flagged NOVEL → abstain            : "
          f"{n_ab}/{len(test_nn)} ({100*n_ab/max(len(test_nn),1):.1f}%)")

    y_pred = np.array(final_pipe.predict(X_test))
    wrong = y_pred != np.array(y_test)
    if n_ab > 0:
        caught = int((abstain & wrong).sum())
        print(f"  Of the abstained isolates, {caught} would have been MIS-classified "
              f"(caught before any clinical decision).")
    print("\n  → In production a novel sequence returns 'Novel - manual BSL review',")
    print("    never a confident HIGH/MODERATE/LOW. This is the safe, correct behaviour")
    print("    for the genuinely-new-virus case that the novel-family eval exposed.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_nn, bins=30, alpha=0.6, label="train → train NN distance", color="#3498db")
    ax.hist(test_nn, bins=30, alpha=0.6, label="test → train NN distance", color="#e67e22")
    ax.axvline(thr, color="red", ls="--", lw=1.5, label=f"abstain threshold ({thr:.2f})")
    ax.set_xlabel("Nearest-neighbour distance in feature space")
    ax.set_ylabel("Count")
    ax.set_title("Novelty / OOD abstention gate", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p = PLOTS_DIR / "08_novelty_gate.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"\n  [PLOT] Saved: {p}")

    return {"threshold": thr, "abstain_count": n_ab,
            "abstain_rate": n_ab / max(len(test_nn), 1)}


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 14: Final summary
# ──────────────────────────────────────────────────────────────────────────────

def print_final_summary(
    best_name: str,
    results: list[dict],
    test_metrics: dict,
    novel_metrics: dict,
    gate_info: dict,
) -> None:
    print_section("SECTION 14 - PIPELINE COMPLETE: EXECUTIVE SUMMARY")
    best = results[0]
    print(textwrap.dedent(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  HANTAVIRUS PATHOGENICITY CLASSIFIER - SUMMARY (leakage-free)   │
    ├─────────────────────────────────────────────────────────────────┤
    │  Best model     : {best_name:<44}│
    │  CV AUC (OvR)   : {best['val_auc_mean']:.4f} ± {best['val_auc_std']:.4f}{' '*35}│
    │  CV F1 (macro)  : {best['val_f1_mean']:.4f} ± {best['val_f1_std']:.4f}{' '*35}│
    │                                                                 │
    │  KNOWN FAMILY  (held-out 20%, isolate split) - for KNOWN viruses│
    │  Accuracy       : {test_metrics['accuracy']:.4f}{' '*43}│
    │  Macro F1       : {test_metrics['macro_f1']:.4f}{' '*43}│
    │  Weighted AUC   : {test_metrics['weighted_auc']:.4f}{' '*43}│
    │                                                                 │
    │  NOVEL FAMILY  (clade split) - for GENUINELY NEW viruses        │
    │  Accuracy       : {novel_metrics['accuracy']:.4f}  (baseline {novel_metrics['baseline']:.4f}){' '*22}│
    │  Balanced acc   : {novel_metrics['balanced_accuracy']:.4f}{' '*43}│
    │  → abstain gate flags {gate_info['abstain_count']:>3} novel test isolates for manual review {' '*5}│
    │                                                                 │
    │  Plots saved to : ./pipeline_plots/                             │
    │    01_eda_class_distribution.png   05_test_evaluation.png       │
    │    02_eda_sequence_features.png    06_predictions.png           │
    │    03_stability_leaderboard.png    07_known_vs_novel.png        │
    │    04_shap_importance.png          08_novelty_gate.png          │
    └─────────────────────────────────────────────────────────────────┘
    """))

    # Final verdict - judged on BOTH the known-family and novel-family numbers.
    known_auc = test_metrics["weighted_auc"]
    known_f1 = test_metrics["macro_f1"]
    novel_acc = novel_metrics["accuracy"]
    novel_base = novel_metrics["baseline"]

    if known_auc >= 0.95 and known_f1 >= 0.85:
        known_verdict = "STRONG on KNOWN virus families (assisted triage with human review)."
    elif known_auc >= 0.85:
        known_verdict = "MODERATE on known families; expert override required."
    else:
        known_verdict = "WEAK even on known families; needs further feature work."

    if novel_acc <= novel_base + 0.05:
        novel_verdict = ("CANNOT predict GENUINELY NEW families (at/below baseline) - "
                         "the abstention gate must handle these.")
    else:
        novel_verdict = "LIMITED generalisation to new families; treat as advisory only."

    print(f"  Verdict (known families): {known_verdict}")
    print(f"  Verdict (novel families): {novel_verdict}")
    print(f"\n  Key finding: the genome's {KMER_K}-mer composition genuinely separates")
    print(f"  pathogenicity for KNOWN virus families, but does NOT extrapolate to")
    print(f"  unseen families - so the platform's correct, safe behaviour is to abstain")
    print(f"  on novel sequences rather than emit a confident risk class.")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print_project_card()

    df_raw = load_data()
    df_raw = engineer_sequence_features(df_raw)
    df_feat = select_features(df_raw)
    run_eda(df_feat)

    X_train, X_test, y_train, y_test, le = make_split(df_feat)

    describe_preprocessor()
    preprocessor = build_preprocessor()

    stability_results = train_all_candidates(preprocessor, X_train, y_train)
    print_leaderboard(stability_results)

    best_name = stability_results[0]["name"]
    final_pipe = fit_final_model(best_name, preprocessor, X_train, y_train)
    calibrated = calibrate_model(final_pipe, X_train, y_train)

    run_shap(final_pipe, X_test, y_test, le)

    test_metrics = evaluate_test_set(calibrated, X_test, y_test, le, best_name)
    overfitting_analysis(stability_results)

    # Honest novel-virus evaluation + the abstention gate that handles it.
    novel_metrics = evaluate_novel_family(df_feat, best_name, le, test_metrics["accuracy"])
    gate_info = novelty_abstention_gate(final_pipe, X_train, X_test, y_test, le)

    predict_and_interpret(calibrated, X_test, y_test, le, df_raw)
    print_final_summary(best_name, stability_results, test_metrics, novel_metrics, gate_info)


if __name__ == "__main__":
    main()
