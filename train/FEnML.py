import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

WINDOW_SIZE = 30

DATA_PATH = "../data/refined/dataset.csv"

output_dir = "../output_FEnML"
os.makedirs(output_dir, exist_ok=True)
OUTPUT_PATH = f"{output_dir}/fenml_sliding_results.csv"

# 1. 데이터 로딩
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["date"])
df = df.drop(columns=["press_titles"], errors="ignore")

# 2. 슬라이딩 윈도우 데이터 생성 (X: WINDOW_SIZE일, y: 그 다음 날)
def create_sliding_dataset(df, window=30):
    features, targets, dates = [], [], []
    feature_cols = [col for col in df.columns if col not in ["date", "attendences"]]
    for i in range(len(df) - window):
        X_window = df[feature_cols].iloc[i:i + window].mean().values
        y = df["attendences"].iloc[i + window]
        date = df["date"].iloc[i + window]
        features.append(X_window)
        targets.append(y)
        dates.append(date)
    return np.array(features), np.array(targets), dates

X, y, dates = create_sliding_dataset(df, window=WINDOW_SIZE)
dates = pd.to_datetime(dates)

# 3. Train/Test Split (시계열 기반)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

# 4. 모델 구성
models = {
    "LinearRegression": (LinearRegression(), {}, True),
    "DecisionTree": (DecisionTreeRegressor(random_state=42), {"max_depth": [3, 5, 10, None]}, False),
    "RandomForest": (RandomForestRegressor(random_state=42), {"n_estimators": [100], "max_depth": [5, 10]}, False),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}, True),
    "SVR": (SVR(), {"C": [0.1, 1, 10], "kernel": ["rbf"]}, True),
    "XGBoost": (XGBRegressor(random_state=42), {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.1]}, False)
}

results = []

# 5. 학습 및 평가
for name, (model, params, scale) in models.items():
    print(f"🔍 Training {name}...")
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    if params:
        grid = GridSearchCV(model, params, cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    nrmse = rmse / np.mean(y_test)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "NRMSE": nrmse
    })

    # 예측 시각화 저장
    plt.figure(figsize=(10, 5))
    plt.plot(dates_test, y_test, label="Actual")
    plt.plot(dates_test, y_pred, label="Predicted")
    plt.title(f"{name} Prediction (sliding)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_sliding_prediction.png")
    plt.close()

# 6. 결과 저장
df_result = pd.DataFrame(results)
df_result.to_csv(OUTPUT_PATH, index=False)
print("✅ Sliding 기반 FEnML 모델 학습 및 예측 완료")
print(df_result)


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
