import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

# ------------------------
# Dummy dataset
# ------------------------
X = torch.randn(500, 20)
y = torch.randint(0, 2, (500,))

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ------------------------
# Simple model
# ------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# MLflow experiment
# ------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("demo_mlops")

with mlflow.start_run() as run:
    for epoch in range(5):
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - loss: {avg_loss:.4f}")
        mlflow.log_metric("loss", avg_loss, step=epoch)

    # ------------------------
    # Log and register model
    # ------------------------
    try:
        mlflow.pytorch.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri, "demo_mlops")
        print(f"✅ Model logged and registered as 'demo_mlops' version {result.version}")
    except Exception as e:
        print("Error logging model:", e)