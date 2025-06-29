import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.append(parent_dir)
from utils.dataset_util import create_sliding_dataset, stratified_timesplit

# 설정
WINDOW_SIZE = 30
EPOCHS = 500
BATCH_SIZE = 64

# 데이터 로딩
DATA_PATH = "../data/refined/dataset.csv"
EMBED_PATH = "../data/refined/dataset_sbert_ae.csv"
output_dir = "../output_FEnML_sbert_mlp_ensemble"
os.makedirs(output_dir, exist_ok=True)

# 데이터 로딩 및 병합
df_main = pd.read_csv(DATA_PATH, parse_dates=["date"]).drop(columns=["press_titles"], errors="ignore")
df_embed = pd.read_csv(EMBED_PATH, parse_dates=["date"])
df = pd.merge(df_main, df_embed, on="date")

# 피처 분리 및 정규화
ae_cols = [c for c in df.columns if c.startswith("press_emb")]
base_cols = [c for c in df.columns if c not in ae_cols + ["date", "attendences"]]
df_all = df[["date", "attendences"] + base_cols + ae_cols]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_all.drop(columns=["date", "attendences"]))
df_scaled = pd.DataFrame(features_scaled, columns=base_cols + ae_cols)
df_scaled["attendences"] = df_all["attendences"]
df_scaled["date"] = df_all["date"]

# Sliding window dataset
X_seq, y_seq, date_seq = create_sliding_dataset(df_scaled, window=WINDOW_SIZE)
X_mlp = X_seq.reshape(X_seq.shape[0], -1)
date_seq = pd.to_datetime(date_seq)

# Train/Test split
X_train_mlp, X_test_mlp, y_train_raw, y_test_raw, dates_train, dates_test = stratified_timesplit(X_mlp, y_seq, date_seq)

# 🎯 Target 정규화
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

# MLP 모델 정의
class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.model(x)

# 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(DeepMLP(X_train_mlp.shape[1])).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_mlp, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True
)

# 학습 루프
print("🧠 Training MLP...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3} | Loss: {total_loss / len(train_loader):.4f}")

# 예측 및 역변환
model.eval()
with torch.no_grad():
    y_pred_mlp_scaled = model(torch.tensor(X_test_mlp, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    y_pred_mlp = y_scaler.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).flatten()

# 기존 ML 모델 학습
models = {
    "LinearRegression": (LinearRegression(), {}, True),
    "DecisionTree": (DecisionTreeRegressor(random_state=42), {"max_depth": [3, 5, 10, None]}, False),
    "RandomForest": (RandomForestRegressor(random_state=42), {"n_estimators": [100], "max_depth": [5, 10]}, False),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}, True),
    "SVR": (SVR(), {"C": [0.1, 1, 10], "kernel": ["rbf"]}, True),
    "XGBoost": (XGBRegressor(random_state=42), {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.1]}, False)
}

results = []
pred_dict = {}

def evaluate(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    nrmse = rmse / np.mean(y_true)
    print(f"[{name}] MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.2f}, NRMSE={nrmse:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "NRMSE": nrmse}

# ML용 슬라이딩 데이터 재생성
X_ml, y_ml, dates_ml = create_sliding_dataset(df_main, window=WINDOW_SIZE)
dates_ml = pd.to_datetime(dates_ml)

X_train_ml, X_test_ml, y_train_ml, y_test_ml, dates_train_ml, dates_test_ml = stratified_timesplit(X_ml, y_ml, dates_ml)

for name, (model_ml, params, scale) in models.items():
    print(f"🔍 Training {name}...")
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_ml)
        X_test_scaled = scaler.transform(X_test_ml)
    else:
        X_train_scaled, X_test_scaled = X_train_ml, X_test_ml

    if params:
        grid = GridSearchCV(model_ml, params, cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(X_train_scaled, y_train_ml)
        best_model = grid.best_estimator_
    else:
        best_model = model_ml.fit(X_train_scaled, y_train_ml)

    y_pred_ml = best_model.predict(X_test_scaled)
    pred_dict[name] = y_pred_ml
    results.append(evaluate(y_test_ml, y_pred_ml, name))

# MLP 평가
results.append(evaluate(y_test_ml, y_pred_mlp, "MLP_AE+Base (Window)"))

# 앙상블 평가
for name, y_ml in pred_dict.items():
    y_ens = (y_ml + y_pred_mlp) / 2
    results.append(evaluate(y_test_ml, y_ens, f"Ensemble_{name}_+_MLP"))

# 결과 저장 및 시각화
df_result = pd.DataFrame(results)
df_result.to_csv(os.path.join(output_dir, "sliding_combined_mlp_results.csv"), index=False)

# 예측 곡선
plt.figure(figsize=(12, 6))
plt.plot(dates_test_ml, y_test_ml, label="Actual", linestyle="--")
plt.plot(dates_test_ml, y_pred_mlp, label="MLP")
for name in ["XGBoost", "RandomForest"]:
    plt.plot(dates_test_ml, pred_dict[name], label=name)
plt.legend()
plt.title("Prediction Curve")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sliding_prediction_curve.png"))
plt.close()

# bar plot
plt.figure(figsize=(12, 6))
bar_width = 0.25
x = np.arange(len(df_result["Model"]))
plt.bar(x - bar_width/2, df_result["MAE"], width=bar_width, label="MAE")
plt.bar(x + bar_width/2, df_result["RMSE"], width=bar_width, label="RMSE")
plt.xticks(x, df_result["Model"], rotation=45)
plt.title("Sliding Window Model Performance")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sliding_model_comparison.png"))
plt.close()
