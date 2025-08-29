import mlflow
import pandas as pd
from sklearn.datasets import make_classification
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

# ------------------------
# Generate dummy reference & current data
# ------------------------
X_ref, _ = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
X_curr, _ = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=7)

df_ref = pd.DataFrame(X_ref, columns=[f"feature_{i}" for i in range(X_ref.shape[1])])
df_curr = pd.DataFrame(X_curr, columns=[f"feature_{i}" for i in range(X_curr.shape[1])])

# ------------------------
# Generate Evidently dashboard
# ------------------------
dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(df_ref, df_curr)
dashboard.save("evidently_report.html")
print("Evidently report saved to evidently_report.html")