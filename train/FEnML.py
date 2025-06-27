
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

# 데이터 로드
df = pd.read_csv("../data/refined/dataset.csv", encoding="utf-8-sig", parse_dates=["date"])

# Feature/Target 분리
ignore_cols = ["date", "press_titles"]
X = df.drop(columns=ignore_cols + ["attendences"])
y = df["attendences"]
#print(X.head())

# Train/Test split (시계열 유지)
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 모델 정의 및 그리드서치 파라미터 설정
models = {
    "LinearRegression": (LinearRegression(), {}),
    "DecisionTree": (DecisionTreeRegressor(random_state=42), {"max_depth": [3, 5, 10, None]}),
    "RandomForest": (RandomForestRegressor(random_state=42), {"n_estimators": [50, 100], "max_depth": [5, 10, None]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}),
    "SVR": (SVR(), {"C": [0.1, 1, 10], "kernel": ["rbf"]}),
    "XGBoost": (XGBRegressor(random_state=42), {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
}

results = []
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

for name, (model, params) in models.items():
    print(f"🔍 Training {name}...")
    if params:
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(model, param_grid=params, scoring="neg_mean_squared_error", cv=tscv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    # 예측 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"].iloc[train_size:], y_test, label="Actual")
    plt.plot(df["date"].iloc[train_size:], y_pred, label="Predicted")
    plt.title(f"{name} Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_prediction.png")
    plt.close()

# 성능 저장
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/model_performance.csv", index=False)

# 성능 비교 그래프
plt.figure(figsize=(10, 6))
bar_width = 0.25
x = np.arange(len(results_df["Model"]))
plt.bar(x - bar_width, results_df["MAE"], width=bar_width, label="MAE")
plt.bar(x, results_df["RMSE"], width=bar_width, label="RMSE")
plt.bar(x + bar_width, results_df["R2"], width=bar_width, label="R2")
plt.xticks(x, results_df["Model"])
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/model_comparison_barplot.png")
plt.close()

print("✅ 모든 모델 학습 완료 및 평가 저장됨.")
