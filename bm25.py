"""Okapi BM25 over a sparse term matrix (no external dependency)."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..nlp.entities import tokenize


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.4, b: float = 0.72):
        self.k1, self.b = k1, b
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(texts))
        for i, t in enumerate(texts):
            toks = tokenize(t)
            lengths[i] = len(toks)
            counts: dict[int, int] = {}
            for tok in toks:
                j = vocab.setdefault(tok, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        self.vocab = vocab
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)), dtype=np.float32)
        df = np.bincount(tf.indices, minlength=len(vocab))
        n = len(texts)
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg = lengths.mean() if n else 1.0
        norm = k1 * (1.0 - b + b * lengths / max(avg, 1e-9))
        tf = tf.tocoo()
        weighted = tf.data * (k1 + 1.0) / (tf.data + norm[tf.row])
        self.matrix = sparse.csc_matrix((weighted.astype(np.float32), (tf.row, tf.col)), shape=(n, len(vocab)))

    def scores(self, query: str) -> np.ndarray:
        ids = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not ids:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        sub = self.matrix[:, ids]
        return np.asarray(sub @ self.idf[ids]).ravel()
