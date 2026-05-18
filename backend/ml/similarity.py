"""Training-data similarity index (§17).

Uses sklearn BallTree (L2) rather than FAISS. FAISS crashes with SIGSEGV
inside Celery's prefork pool on macOS due to FAISS's internal threading
conflicting with fork(). BallTree is fork-safe and fast enough for datasets
up to ~100K rows × ~100 features - well within this platform's range.

Scores new samples by their mean distance to the k nearest training neighbors,
normalized to [0, 1] where 1 = very similar.

The index is serialized via pickle and stored via storage.py.
"""

from __future__ import annotations

import io
import logging
import pickle  # nosec - internal serialization

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_K = 5
_NORM_PERCENTILE = 95


class SimilarityIndex:
    """sklearn BallTree L2 index for training-data similarity scoring."""

    def __init__(self, k: int = _DEFAULT_K) -> None:
        self.k = k
        self._tree: object | None = None
        self._norm_cap: float = 1.0
        self._n_features: int | None = None

    def fit(self, X_train: np.ndarray) -> "SimilarityIndex":
        from sklearn.neighbors import BallTree

        X = np.asarray(X_train, dtype=np.float32)
        n, d = X.shape
        self._n_features = d
        logger.info("Building BallTree similarity index on %d samples × %d features", n, d)

        self._tree = BallTree(X, metric="euclidean")

        sample_size = min(n, 2000)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sample = X[sample_idx]

        k_query = min(self.k + 1, n)
        distances, _ = self._tree.query(sample, k=k_query)  # type: ignore[attr-defined]
        nn_distances = distances[:, 1:]  # skip self-match
        self._norm_cap = float(np.percentile(nn_distances, _NORM_PERCENTILE))
        if self._norm_cap == 0.0:
            self._norm_cap = 1.0

        logger.info("Similarity index built. norm_cap=%.4f", self._norm_cap)
        return self

    def score(self, X_new: np.ndarray) -> np.ndarray:
        if self._tree is None:
            raise RuntimeError("SimilarityIndex.fit() must be called before score()")

        X = np.asarray(X_new, dtype=np.float32)

        # Reconcile dimension mismatch between query and index.
        # The index may have been built with a preprocessor fitted on the full
        # X_train while predict_single uses the pipeline's preprocessor (fitted
        # on X_fit, ~60% of train) - OHE categories can differ, producing
        # vectors of different widths.  We read the expected width directly from
        # the BallTree's stored training data so this works for all indexes,
        # including ones serialized before _n_features was introduced.
        expected: int = int(self._tree.data.shape[1])  # type: ignore[attr-defined]
        if X.shape[1] != expected:
            if X.shape[1] < expected:
                pad = np.zeros((X.shape[0], expected - X.shape[1]), dtype=np.float32)
                X = np.concatenate([X, pad], axis=1)
            else:
                X = X[:, :expected]

        distances, _ = self._tree.query(X, k=self.k)  # type: ignore[attr-defined]
        mean_dist = distances.mean(axis=1)
        similarity = 1.0 - np.clip(mean_dist / self._norm_cap, 0.0, 1.0)
        return similarity.astype(np.float32)

    def serialize(self) -> bytes:
        if self._tree is None:
            raise RuntimeError("Index not built - call fit() first")
        return pickle.dumps(  # nosec
            {"tree": self._tree, "norm_cap": self._norm_cap, "k": self.k, "n_features": self._n_features}
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "SimilarityIndex":
        payload = pickle.loads(data)  # nosec
        inst = cls(k=payload["k"])
        inst._norm_cap = payload["norm_cap"]
        inst._tree = payload["tree"]
        inst._n_features = payload.get("n_features")  # None for indexes built before this field
        return inst
