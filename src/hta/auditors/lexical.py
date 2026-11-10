# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""TF-IDF + one logistic regression per label, trained on reference-population transcripts.

The demographics enter only through the baseline prior (see LexicalModel). The trained model
is exported to plain JSON (vocabulary, idf, weights) and every prediction,
in the main run and in `hta verify`, goes through the JSON-loaded model. There is one code path,
so the stored scores and the re-computed ones cannot drift apart.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .baseline import BaselineAuditor

PRIOR_CLIP = 1e-3


def prior_logit(baseline: BaselineAuditor, records: list[dict]) -> np.ndarray:
    """(n_records, n_labels) log-odds of the demographics-only baseline prior."""
    p = np.clip(baseline.predict(records), PRIOR_CLIP, 1 - PRIOR_CLIP)
    return np.log(p / (1 - p))


class LexicalModel:
    """Prediction from exported parameters only.

    For label j: logit P(y_j = 1) = w_j . tfidf(text) + a_j * logit(prior_j) + b_j, where prior_j
    is the demographics-only baseline. The auditor and the baseline therefore use exactly the
    same demographic information, and the gain measures what the text adds.
    """

    name = "lexical"

    def __init__(self, params: dict):
        self.p = params
        vocab = params["vocabulary"]
        self.vec = CountVectorizer(vocabulary={t: i for i, t in enumerate(vocab)},
                                   ngram_range=tuple(params["ngram_range"]), lowercase=True)
        self.idf = np.array(params["idf"], dtype=float)
        self.W = np.array(params["coef"], dtype=float)          # (n_labels, n_vocab)
        self.a = np.array(params["prior_coef"], dtype=float)    # (n_labels,)
        self.b = np.array(params["intercept"], dtype=float)     # (n_labels,)
        self.baseline = BaselineAuditor(params["baseline"])

    def tfidf(self, records: list[dict]) -> sp.csr_matrix:
        counts = self.vec.transform([r["text"] for r in records]).astype(float).tocsr()
        counts.data = 1.0 + np.log(counts.data)                 # sublinear tf
        x = counts.multiply(self.idf).tocsr()
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return (sp.diags(1.0 / norms) @ x).tocsr()

    def predict(self, records: list[dict]) -> np.ndarray:
        z = np.asarray(self.tfidf(records) @ self.W.T) + prior_logit(self.baseline, records) * self.a + self.b
        return 1.0 / (1.0 + np.exp(-z))


def fit_lexical(records: list[dict], y: np.ndarray, label_ids: list[str], lcfg: dict,
                setting: str, baseline_params: dict) -> dict:
    """Fit on (record, label-vector) pairs and return JSON-serialisable parameters."""
    tfidf = TfidfVectorizer(ngram_range=tuple(lcfg["ngram_range"]), min_df=lcfg["min_df"],
                            max_features=lcfg["max_features"], sublinear_tf=True, lowercase=True)
    texts = [r["text"] for r in records]
    nonempty = [t for t in texts if t.strip()]
    tfidf.fit(nonempty if nonempty else ["empty"])
    vocab = sorted(tfidf.vocabulary_, key=tfidf.vocabulary_.get)
    nl = len(label_ids)
    params = {
        "kind": "lexical", "setting": setting, "label_ids": label_ids,
        "ngram_range": list(lcfg["ngram_range"]), "vocabulary": vocab,
        "idf": [float(v) for v in tfidf.idf_], "baseline": baseline_params,
        "coef": [[0.0] * len(vocab) for _ in label_ids], "prior_coef": [0.0] * nl,
        "intercept": [0.0] * nl, "constant_labels": [], "not_converged": [],
        "n_train": len(records), "C": lcfg["C"],
    }
    model = LexicalModel(params)
    x_text = model.tfidf(records)
    prior = prior_logit(model.baseline, records)
    for j, lid in enumerate(label_ids):
        yj = y[:, j]
        if yj.min() == yj.max():
            # One class only: predict the training prevalence, clipped away from 0 and 1.
            p = float(np.clip(yj.mean(), PRIOR_CLIP, 1 - PRIOR_CLIP))
            params["intercept"][j] = float(np.log(p / (1 - p)))
            params["constant_labels"].append(lid)
            continue
        xj = sp.hstack([x_text, sp.csr_matrix(prior[:, j:j + 1])]).tocsr()
        clf = LogisticRegression(C=lcfg["C"], solver="lbfgs", max_iter=5000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(xj, yj)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            params["not_converged"].append(lid)
        params["coef"][j] = [float(v) for v in clf.coef_[0][:-1]]
        params["prior_coef"][j] = float(clf.coef_[0][-1])
        params["intercept"][j] = float(clf.intercept_[0])
    return params
