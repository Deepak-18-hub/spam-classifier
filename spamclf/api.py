"""FastAPI service. Run from the repo root: uvicorn spamclf.api:app --reload"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from spamclf.model import MODEL_PATH, load_model, predict_one

INDEX_HTML = Path(__file__).parent / "static" / "index.html"
MAX_CHARS = 1000  # the longest message in the training data is 910 characters


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once before the first request. MODEL_PATH can be overridden (tests use a tiny model).
    app.state.model = load_model(Path(os.environ.get("MODEL_PATH", MODEL_PATH)))
    yield


app = FastAPI(title="Spam Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class PredictResponse(BaseModel):
    label: Literal["spam", "ham"]
    spam_score: float  # NOT a calibrated probability: the model is trained with class_weight="balanced"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    # Plain `def`: sklearn is CPU-bound, so FastAPI runs it in a thread pool instead of blocking the event loop.
    label, score = predict_one(request.app.state.model, body.text)
    return PredictResponse(label=label, spam_score=round(score, 4))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML)
