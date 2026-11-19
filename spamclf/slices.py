"""Error analysis by message length plus a privacy check, for the responsible-AI write-up.

Run from the repo root: python -m spamclf.slices   (needs the dataset; uses the saved model, no retraining)
"""
import re

import numpy as np

from spamclf.data import load_split
from spamclf.model import THRESHOLD, load_model, spam_scores

BUCKETS = [("short (<50 chars)", 0, 50), ("medium (50-119)", 50, 120), ("long (120+)", 120, 10**6)]
LONG_DIGITS = re.compile(r"\d{7,}")  # crude proxy for phone numbers, account numbers, etc.


def rate(num, den):
    return f"{100 * num / den:.1f}%" if den else "n/a"


def main():
    X_train, X_test, y_train, y_test, n_unique = load_split()
    pred = (spam_scores(load_model(), X_test) >= THRESHOLD).astype(int)
    lens = np.array([len(t) for t in X_test])

    print(f"Test set: {len(X_test)} messages, {int(y_test.sum())} spam. Error rates by message length:")
    print(f"{'bucket':20s} {'n':>5s} {'spam':>5s} {'FP':>4s} {'FN':>4s} {'false-pos rate':>15s} {'miss rate':>10s}")
    for name, lo, hi in BUCKETS:
        m = (lens >= lo) & (lens < hi)
        y, p = y_test[m], pred[m]
        tp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
        fp, tn = int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
        print(f"{name:20s} {m.sum():5d} {int(y.sum()):5d} {fp:4d} {fn:4d} {rate(fp, fp + tn):>15s} {rate(fn, fn + tp):>10s}")

    texts = X_train + X_test
    labels = np.concatenate([y_train, y_test])
    has_digits = np.array([bool(LONG_DIGITS.search(t)) for t in texts])
    print(f"\nPrivacy check on all {n_unique} unique messages (a message has 7+ consecutive digits):")
    for name, cls in [("ham", 0), ("spam", 1)]:
        sel = labels == cls
        print(f"  {name:5s}: {int(has_digits[sel].sum())} of {int(sel.sum())} ({rate(int(has_digits[sel].sum()), int(sel.sum()))})")


if __name__ == "__main__":
    main()
