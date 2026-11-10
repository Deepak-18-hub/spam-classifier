import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from spamclf.model import make_pipeline

# A tiny, repeated toy corpus so tests need neither data/raw/ nor the real model.
SPAM = ["free prize call now claim", "win free cash call now", "claim your free prize today call"] * 5
HAM = ["see you at lunch tomorrow", "are we meeting for lunch", "call me when you get home"] * 5


@pytest.fixture(scope="session")
def toy_model():
    model = make_pipeline(LogisticRegression(max_iter=1000))
    return model.fit(SPAM + HAM, [1] * len(SPAM) + [0] * len(HAM))


@pytest.fixture(scope="module")
def client(toy_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "toy.joblib"
    joblib.dump(toy_model, path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_PATH", str(path))
        from spamclf.api import app

        with TestClient(app) as c:  # the `with` runs the lifespan (loads the model)
            yield c
