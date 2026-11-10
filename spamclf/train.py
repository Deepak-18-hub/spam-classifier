"""Train and evaluate the spam classifier. Run from the repo root: python -m spamclf.train"""
import json
import platform
from datetime import date

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

from spamclf.data import EXPECTED_ROWS, SEED, load_split
from spamclf.model import MODEL_PATH, SPAM, THRESHOLD, make_pipeline, spam_scores

METRICS_PATH = MODEL_PATH.with_name("metrics.json")

CANDIDATES = {
    "naive_bayes": lambda: MultinomialNB(),
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "logreg_balanced": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
}


def top_spam_terms(model, n=15):
    """Words pushing a message towards spam, for explaining the model."""
    names = model.named_steps["tfidf"].get_feature_names_out()
    clf = model.named_steps["clf"]
    weight = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    return [str(names[i]) for i in np.argsort(weight)[::-1][:n]]


def main():
    # Exact duplicates are removed BEFORE splitting (inside load_split), so no identical message
    # sits in both train and test.
    X_train, X_test, y_train, y_test, n_unique = load_split(SEED)
    n_raw = EXPECTED_ROWS

    # Model selection: 5-fold cross-validation on the TRAIN set only. The test set is not touched here.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    for name, make_clf in CANDIDATES.items():
        scores = cross_val_score(make_pipeline(make_clf()), X_train, y_train, cv=cv, scoring="f1")
        cv_results[name] = {"f1_mean": round(float(scores.mean()), 4), "f1_std": round(float(scores.std()), 4)}
        print(f"CV spam-F1  {name:16s} {scores.mean():.4f} +/- {scores.std():.4f}")
    best = max(cv_results, key=lambda k: cv_results[k]["f1_mean"])
    print("chosen:", best)

    # Fit the winner on the full training set, then score the test set exactly once.
    model = make_pipeline(CANDIDATES[best]()).fit(X_train, y_train)
    scores_test = spam_scores(model, X_test)
    pred = (scores_test >= THRESHOLD).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[SPAM], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    always_ham = float((y_test == 0).mean())
    print(f"TEST spam precision {p[0]:.4f} recall {r[0]:.4f} F1 {f1[0]:.4f} | "
          f"accuracy {(pred == y_test).mean():.4f} vs always-ham {always_ham:.4f}")
    print(f"confusion matrix  TN={tn} FP={fp} FN={fn} TP={tp}")

    wrong = [(t, int(l)) for t, l, pr in zip(X_test, y_test, pred) if l != pr]
    metrics = {
        "trained_on": date.today().isoformat(),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "data": {"raw_rows": n_raw, "after_dedup": n_unique, "train": len(y_train), "test": len(y_test),
                 "spam_share_test": round(float(y_test.mean()), 4)},
        "seed": SEED, "threshold": THRESHOLD,
        "cv_f1_train_only": cv_results, "chosen": best,
        "test": {"spam_precision": round(float(p[0]), 4), "spam_recall": round(float(r[0]), 4),
                 "spam_f1": round(float(f1[0]), 4), "accuracy": round(float((pred == y_test).mean()), 4),
                 "always_ham_accuracy": round(always_ham, 4),
                 "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}},
        "top_spam_terms": top_spam_terms(model),
        "false_positives": [t for t, l in wrong if l == 0][:10],  # ham called spam
        "false_negatives": [t for t, l in wrong if l == 1][:10],  # spam let through
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", MODEL_PATH, f"({MODEL_PATH.stat().st_size / 1024:.0f} KB) and", METRICS_PATH.name)


if __name__ == "__main__":
    main()
