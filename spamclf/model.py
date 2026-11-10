"""The model definition and prediction helpers, shared by training and the API."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_model.joblib"
SPAM = 1  # class label for spam in y
THRESHOLD = 0.5  # score at or above this is called spam


def make_pipeline(classifier) -> Pipeline:
    """TF-IDF on word unigrams+bigrams, then the given classifier.

    Keeping the vectorizer inside the pipeline means it is fitted on training data only.
    """
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return Pipeline([("tfidf", tfidf), ("clf", classifier)])


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    # joblib files can run code when loaded: only ever load one you trained yourself (D6).
    return joblib.load(path)


def spam_scores(model: Pipeline, texts: list[str]):
    """Probability-like spam score for each text, looked up via classes_ (never by position)."""
    column = list(model.classes_).index(SPAM)
    return model.predict_proba(texts)[:, column]


def predict_one(model: Pipeline, text: str) -> tuple[str, float]:
    score = float(spam_scores(model, [text])[0])
    return ("spam" if score >= THRESHOLD else "ham"), score
