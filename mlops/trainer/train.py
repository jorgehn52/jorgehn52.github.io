import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ------------------------
# Prepare dummy dataset
# ------------------------
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ------------------------
# Define simple model
# ------------------------
class SimpleModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# MLflow experiment
# ------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("demo_mlops")

with mlflow.start_run():
    for epoch in range(5):
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - loss: {total_loss/len(train_loader):.4f}")
        mlflow.log_metric("loss", total_loss/len(train_loader), step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "model")