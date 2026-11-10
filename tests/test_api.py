import pytest


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_loaded_at_startup(client):
    assert client.app.state.model is not None


def test_predict_spam(client):
    r = client.post("/predict", json={"text": "free prize call now claim cash"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "spam"
    assert 0.5 <= body["spam_score"] <= 1


def test_predict_ham(client):
    r = client.post("/predict", json={"text": "see you at lunch tomorrow"})
    assert r.status_code == 200
    assert r.json()["label"] == "ham"
    assert 0 <= r.json()["spam_score"] < 0.5


def test_unicode_and_emoji_are_accepted(client):
    assert client.post("/predict", json={"text": "héllo 😀 £1000 prize"}).status_code == 200


def test_text_at_the_limit_is_accepted(client):
    assert client.post("/predict", json={"text": "a" * 1000}).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   \n\t"}, {"text": "a" * 1001}, {}, {"text": 123}, {"wrong": "field"}],
)
def test_bad_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_root_serves_the_web_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Spam Classifier" in r.text
