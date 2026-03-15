import json
import pickle
import numpy as np
import os
import urllib.request
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Download dataset ──────────────────────────────────────────
os.makedirs("dataset", exist_ok=True)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
urllib.request.urlretrieve(url, "dataset/winequality-white.csv")

data = []
with open("dataset/winequality-white.csv") as f:
    headers = f.readline().strip().split(";")
    headers = [h.strip('"') for h in headers]
    for line in f:
        data.append([float(x) for x in line.strip().split(";")])

data = np.array(data)
X = data[:, :-1]
y = data[:, -1]

feature_names = headers[:-1]

# ── Correlation-based feature selection (top 6) ───────────────
correlations = np.abs(np.corrcoef(X.T, y)[-1, :-1])
top_indices = np.argsort(correlations)[::-1][:6]
selected_features = [feature_names[i] for i in top_indices]
X_selected = X[:, top_indices]

# ── Train/test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# ── Experiments ───────────────────────────────────────────────
experiments = [
    {
        "experiment_id": "EXP-01",
        "model": LinearRegression(),
        "model_type": "LinearRegression",
        "preprocessing": "none",
        "X_train": X_train,
        "X_test": X_test,
    },
    {
        "experiment_id": "EXP-02",
        "model": Ridge(alpha=1.0),
        "model_type": "Ridge",
        "preprocessing": "standardization",
        "X_train": StandardScaler().fit_transform(X_train),
        "X_test": StandardScaler().fit_transform(X_test),
    },
    {
        "experiment_id": "EXP-03",
        "model": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        "model_type": "RandomForest",
        "preprocessing": "none",
        "X_train": X_train,
        "X_test": X_test,
    },
    {
        "experiment_id": "EXP-04",
        "model": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        "model_type": "RandomForest",
        "preprocessing": "none",
        "X_train": X_train,
        "X_test": X_test,
    },
]

results = []
best_model = None
best_r2 = -999
best_result = None

for exp in experiments:
    exp["model"].fit(exp["X_train"], y_train)
    preds = exp["model"].predict(exp["X_test"])
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    result = {
        "experiment_id": exp["experiment_id"],
        "model_type": exp["model_type"],
        "preprocessing": exp["preprocessing"],
        "selected_features": selected_features,
        "num_features": len(selected_features),
        "metrics": {"mse": round(mse, 6), "r2_score": round(r2, 6)},
    }
    results.append(result)
    print(f"{exp['experiment_id']} | MSE: {mse:.4f} | R2: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = exp["model"]
        best_result = result

# ── Save artifacts ────────────────────────────────────────────
os.makedirs("app/artifacts", exist_ok=True)

with open("app/artifacts/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("app/artifacts/results.json", "w") as f:
    json.dump(results, f, indent=2)

# metrics.json — used by Jenkinsfile for accuracy comparison
metrics = {
    "mse": best_result["metrics"]["mse"],
    "r2_score": best_result["metrics"]["r2_score"],
    "experiment_id": best_result["experiment_id"],
    "model_type": best_result["model_type"],
}
with open("app/artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nBest Model: {best_result['experiment_id']}")
print(f"MSE: {metrics['mse']} | R2: {metrics['r2_score']}")