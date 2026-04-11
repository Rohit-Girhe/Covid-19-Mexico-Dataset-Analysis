from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "Model"
MODEL_PATH = MODEL_DIR / "ann_died_model.keras"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

APP_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"


class PredictionInput(BaseModel):
    USMER: int = Field(..., ge=0, le=1)
    MEDICAL_UNIT: int = Field(..., ge=1, le=13)
    SEX: int = Field(..., ge=0, le=1)
    PATIENT_TYPE: int = Field(..., ge=0, le=1)
    PNEUMONIA: int = Field(..., ge=0, le=1)
    AGE: float = Field(..., ge=0, le=121)
    DIABETES: int = Field(..., ge=0, le=1)
    COPD: int = Field(..., ge=0, le=1)
    ASTHMA: int = Field(..., ge=0, le=1)
    INMSUPR: int = Field(..., ge=0, le=1)
    HIPERTENSION: int = Field(..., ge=0, le=1)
    OTHER_DISEASE: int = Field(..., ge=0, le=1)
    CARDIOVASCULAR: int = Field(..., ge=0, le=1)
    OBESITY: int = Field(..., ge=0, le=1)
    RENAL_CHRONIC: int = Field(..., ge=0, le=1)
    TOBACCO: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    threshold: float
    model_version: str
    input_features: dict[str, Any]


def _load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    metadata = _load_metadata()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained ANN model not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Saved preprocessor not found: {PREPROCESSOR_PATH}")

    app.state.metadata = metadata
    app.state.model = tf.keras.models.load_model(MODEL_PATH)
    app.state.preprocessor = joblib.load(PREPROCESSOR_PATH)
    app.state.threshold = float(metadata.get("decision_threshold", 0.5))
    app.state.feature_columns = metadata["feature_columns"]

    yield


app = FastAPI(
    title="COVID-19 Mortality Risk Predictor",
    description="FastAPI deployment for the ANN-based COVID-19 mortality risk model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


def _to_frame(payload: PredictionInput, columns: list[str]) -> pd.DataFrame:
    record = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    ordered_record = {column: record[column] for column in columns}
    return pd.DataFrame([ordered_record], columns=columns)


def _risk_band(probability: float, threshold: float) -> str:
    if probability >= max(threshold + 0.20, 0.75):
        return "Critical"
    if probability >= threshold:
        return "High"
    if probability >= max(threshold - 0.15, 0.30):
        return "Moderate"
    return "Low"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    metadata = request.app.state.metadata
    context = {
        "request": request,
        "threshold": request.app.state.threshold,
        "test_metrics": metadata.get("test_metrics", {}),
        "validation_metrics": metadata.get("validation_metrics", {}),
        "feature_columns": metadata.get("feature_columns", []),
        "medical_units": list(range(1, 14)),
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/api/health")
async def healthcheck(request: Request) -> dict[str, Any]:
    metadata = request.app.state.metadata
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": request.app.state.threshold,
        "feature_count": len(request.app.state.feature_columns),
        "test_metrics": metadata.get("test_metrics", {}),
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictionInput, request: Request) -> PredictionResponse:
    try:
        input_frame = _to_frame(payload, request.app.state.feature_columns)
        processed = request.app.state.preprocessor.transform(input_frame).astype("float32")
        raw_probability = request.app.state.model.predict(processed, verbose=0)
        probability = float(np.asarray(raw_probability).ravel()[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    threshold = request.app.state.threshold
    prediction = int(probability >= threshold)
    band = _risk_band(probability, threshold)
    label = f"{band} Mortality Risk"
    record = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability=probability,
        threshold=threshold,
        model_version=app.version,
        input_features=record,
    )
