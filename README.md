# Spam Classifier

A web app that tells you whether a text message is spam, built to learn the full ML path: data, training, evaluation, serving, and deployment.

**Status:** Milestones 1-4 built and verified locally (explore, train, API + tests, web page). The Docker image builds and runs (tested locally with Docker Desktop). M5 compared a fine-tuned DistilBERT against the baseline (mean test F1 0.966 vs 0.920, but about 36x slower on CPU and 256 MB vs 157 KB; details in [docs/PROJECT-REPORT.md](docs/PROJECT-REPORT.md) and `models/metrics_bert.json`); the app still serves the small baseline.

**Live demo:** https://spam-classifier-oko3.onrender.com (Render free tier: it sleeps when idle, so the first visit after a quiet period takes about a minute; no rate limiting).

Plan and decisions: [docs/DECISIONS.md](docs/DECISIONS.md)

## Results (held-out test set, 1,035 messages)

| Metric (spam class) | Value |
|---|---|
| Precision | 0.92 |
| Recall | 0.92 |
| F1 | 0.92 |
| Accuracy | 98.0% (a model that always says "ham" scores 87.3%) |

Model: TF-IDF word 1-2 grams + class-balanced logistic regression, chosen by cross-validation on the training set only. Full numbers, confusion matrix, and example mistakes: [models/metrics.json](models/metrics.json). The data is from one source mix (UK spam forum, Singaporean student texts), so real-world performance will be lower.

## Setup (Windows, PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Get the data (only needed to retrain or run the notebook)

The data is not stored in this repo. Download `sms+spam+collection.zip` from
<https://archive.ics.uci.edu/dataset/228/sms+spam+collection> and unzip it so that the file is at
`data/raw/sms_spam/SMSSpamCollection`.

## Run

```powershell
python -m spamclf.train                        # retrain, evaluate, rewrite models/
python -m pytest                               # tests (need no data and no real model)
uvicorn spamclf.api:app --reload               # then open http://127.0.0.1:8000
```

API: `POST /predict` with `{"text": "..."}` returns `{"label": "spam"|"ham", "spam_score": 0-1}`. Interactive docs at `/docs`.

### Docker

```powershell
docker build -t spam-classifier .
docker run --rm -p 8000:8000 spam-classifier   # then open http://127.0.0.1:8000
```

The container reads its port from `$PORT` (default 8000) and runs as a non-root user. The image is about 559 MB.

The committed `models/spam_model.joblib` was trained with the scikit-learn version pinned in `requirements.txt`. Only load model files you trained yourself (joblib files can run code when loaded).

## Layout

- `spamclf/` package: `data.py` (loader), `model.py` (pipeline and prediction), `train.py` (training + evaluation), `api.py` (FastAPI app), `static/index.html` (web page)
- `notebooks/01_explore.ipynb`: data exploration with saved outputs
- `tests/`: pytest suite
- `models/`: trained model and its metrics

## Data credit

SMS Spam Collection, licensed CC BY 4.0.
Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CC84>
