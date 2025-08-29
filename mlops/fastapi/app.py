from fastapi import FastAPI
import mlflow.pytorch
import torch
import torch.nn.functional as F
import numpy as np

app = FastAPI(title="Demo MLOps API")

# Load the latest model from MLflow
model_uri = "models:/demo_mlops/1"  # assumes first registered version
model = mlflow.pytorch.load_model(model_uri)
model.eval()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Demo MLOps API!"}

@app.post("/predict")
def predict(data: list):
    """Expects data as a list of 20 floats (features)"""
    x = torch.tensor([data], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).numpy().tolist()
    return {"predictions": probs}