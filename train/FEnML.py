
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 데이터 로드
df = pd.read_csv("../data/refined/dataset.csv", encoding="utf-8-sig", parse_dates=["date"])

# Feature Engineering - 논문 기반 간단 적용
drop_cols = ['press_titles']  
df = df.drop(columns=drop_cols, errors='ignore')

# Feature/Target 분리
X = df.drop(columns=["date", "attendences"])
y = df["attendences"]

# 시계열 split
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 모델 정의 + 필요 시 스케일링 포함
models = {
    "LinearRegression": (LinearRegression(), {}, True),
    "DecisionTree": (DecisionTreeRegressor(random_state=42), {"max_depth": [3, 5, 10, None]}, False),
    "RandomForest": (RandomForestRegressor(random_state=42), {"n_estimators": [100], "max_depth": [5, 10]}, False),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}, True),
    "SVR": (SVR(), {"C": [0.1, 1, 10], "kernel": ["rbf"]}, True),
    "XGBoost": (XGBRegressor(random_state=42), {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.1]}, False)
}

results = []
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

for name, (model, params, needs_scaling) in models.items():
    print(f"🔍 Training {name}...")

    # 스케일링
    if needs_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # 모델 학습
    if params:
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(model, param_grid=params, scoring="neg_mean_squared_error", cv=tscv, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    # 예측 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"].iloc[train_size:], y_test.values, label="Actual")
    plt.plot(df["date"].iloc[train_size:], y_pred, label="Predicted")
    plt.title(f"{name} Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_prediction_enhanced.png")
    plt.close()

# 성능 저장
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/model_performance_enhanced.csv", index=False)

# 비교 그래프
plt.figure(figsize=(10, 6))
bar_width = 0.25
x = np.arange(len(results_df["Model"]))
plt.bar(x - bar_width, results_df["MAE"], width=bar_width, label="MAE")
plt.bar(x, results_df["RMSE"], width=bar_width, label="RMSE")
plt.bar(x + bar_width, results_df["R2"], width=bar_width, label="R2")
plt.xticks(x, results_df["Model"])
plt.ylabel("Score")
plt.title("Enhanced Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/model_comparison_enhanced.png")
plt.close()

print("✅ 향상된 Feature Engineering 및 모델 학습 완료.")
