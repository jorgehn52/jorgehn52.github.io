from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow.pytorch
import torch
import torch.nn.functional as F
import os
import time

app = FastAPI(title="Demo MLOps API")

# Model will be loaded on first request
model = None
MODEL_URI = "models:/demo_mlops/1"

# MLflow tracking URI from environment variable
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Pydantic model for request
class PredictRequest(BaseModel):
    data: List[float]

def get_model(retries=5, delay=5):
    global model
    attempt = 0
    while model is None and attempt < retries:
        try:
            model = mlflow.pytorch.load_model(MODEL_URI)
            model.eval()
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                raise HTTPException(status_code=503, detail=f"Model not available: {e}")
            time.sleep(delay)
    return model

@app.get("/")
def read_root():
    return {"message": "Welcome to the Demo MLOps API!"}

@app.post("/predict")
def predict(req: PredictRequest):
    model_instance = get_model()
    x = torch.tensor([req.data], dtype=torch.float32)
    with torch.no_grad():
        logits = model_instance(x)
        probs = F.softmax(logits, dim=1).numpy().tolist()
    return {"predictions": probs}