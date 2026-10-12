"""Dense semantic encoder.

Default is latent semantic analysis: TF IDF over word unigrams, bigrams and character
trigrams to five grams (which absorbs typos such as "proflie"), compressed with truncated
SVD and L2 normalised. It runs offline in seconds. If sentence-transformers is installed
and WEIR_DENSE=sbert is set, a MiniLM encoder is used instead; the interface is identical.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

LOG = logging.getLogger("weir_whisper.dense")


class LSAEncoder:
    def __init__(self, n_components: int = 192, seed: int = 0):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"[A-Za-z0-9]{2,}")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self.name = "lsa"

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        X = sparse.hstack([self.word.fit_transform(texts), 0.6 * self.char.fit_transform(texts)]).tocsr()
        k = min(self.svd.n_components, X.shape[1] - 1, X.shape[0] - 1)
        self.svd.n_components = k
        Z = self.svd.fit_transform(X)
        return _l2(Z)

    def encode(self, texts: list[str]) -> np.ndarray:
        X = sparse.hstack([self.word.transform(texts), 0.6 * self.char.transform(texts)]).tocsr()
        return _l2(self.svd.transform(X))


class SBertEncoder:  # pragma: no cover - optional dependency
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = "sbert"

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False))


def _l2(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return (Z / np.maximum(n, 1e-9)).astype(np.float32)


def make_encoder(n_components: int, seed: int):
    if os.environ.get("WEIR_DENSE", "lsa") == "sbert":
        try:
            return SBertEncoder()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("sentence-transformers unavailable (%s), falling back to LSA", exc)
    return LSAEncoder(n_components=n_components, seed=seed)
