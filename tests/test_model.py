import json

import sklearn

from spamclf.model import MODEL_PATH, THRESHOLD, load_model, predict_one, spam_scores


def test_spam_scores_rank_spam_above_ham(toy_model):
    spam, ham = spam_scores(toy_model, ["free prize call now", "see you at lunch tomorrow"])
    assert spam > ham


def test_predict_one_label_follows_threshold(toy_model):
    for text in ["free prize call now", "see you at lunch tomorrow", "unseen words entirely"]:
        label, score = predict_one(toy_model, text)
        assert 0 <= score <= 1
        assert label == ("spam" if score >= THRESHOLD else "ham")


# The committed real model: guards the artifact that actually ships.
def test_committed_model_matches_installed_sklearn():
    saved = json.loads(MODEL_PATH.with_name("metrics.json").read_text(encoding="utf-8"))
    assert saved["versions"]["scikit-learn"] == sklearn.__version__, "retrain: the scikit-learn version changed since the model was saved"


def test_committed_model_obvious_cases():
    model = load_model()
    assert predict_one(model, "WINNER!! Call 09061701461 now to claim your free prize")[0] == "spam"
    assert predict_one(model, "Are we still meeting for lunch tomorrow? I can do 1pm.")[0] == "ham"
