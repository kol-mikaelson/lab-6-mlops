import pickle
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

STUDENT_NAME = "Purandhar"
STUDENT_ROLL = "2022BCS0179"

app = FastAPI(title="Wine Quality Predictor")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("results.json", "r") as f:
    raw = json.load(f)
    results = raw[0] if isinstance(raw, list) else raw


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    chlorides: float
    total_sulfur_dioxide: float
    density: float
    alcohol: float


@app.get("/")
def root():
    return {"message": "Wine Quality Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    metrics = results.get("metrics", {})
    return {
        "experiment_id": results.get("experiment_id"),
        "model_type": results.get("model_type"),
        "mse": metrics.get("mse"),
        "r2_score": metrics.get("r2_score"),
        "features": results.get("selected_features"),
    }


@app.post("/predict")
def predict(wine: WineFeatures):
    try:
        features = [[
            wine.fixed_acidity,
            wine.volatile_acidity,
            wine.chlorides,
            wine.total_sulfur_dioxide,
            wine.density,
            wine.alcohol,
        ]]
        prediction = model.predict(features)[0]
        return {
            "name": STUDENT_NAME,
            "roll_no": STUDENT_ROLL,
            "wine_quality": round(float(prediction), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))