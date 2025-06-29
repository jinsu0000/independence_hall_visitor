import pandas as pd
import numpy as np
from typing import Tuple

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
        print(f"Sliding {i} ~ {i + window}")
    return np.array(features), np.array(targets), dates
    
def stratified_timesplit(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    test_ratio: float = 0.2,
    bins: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    방문객 수 분포를 기반으로 stratified sampling을 수행하면서,
    시간적 순서도 어느 정도 유지하도록 양쪽 구간에서 고르게 테스트 데이터를 추출하는 함수.

    Parameters:
        X (np.ndarray): feature matrix
        y (np.ndarray): target vector
        dates (np.ndarray): 날짜 정보 (정렬되어 있다고 가정)
        test_ratio (float): 테스트셋 비율
        bins (int): target을 균등분할할 구간 수
        random_state (int): 랜덤 시드

    Returns:
        X_train, X_test, y_train, y_test, dates_train, dates_test
    """
    np.random.seed(random_state)
    df = pd.DataFrame(X)
    df["target"] = y
    df["date"] = dates
    df["bin"] = pd.qcut(df["target"], q=bins, labels=False, duplicates="drop")

    test_idx = []

    for b in sorted(df["bin"].unique()):
        df_bin = df[df["bin"] == b]
        n_sample = max(1, int(len(df_bin) * test_ratio))
        idx = df_bin.sample(n=n_sample, random_state=random_state).index
        test_idx.extend(idx)

    test_idx = sorted(test_idx)
    train_idx = sorted(set(range(len(df))) - set(test_idx))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    dates_train, dates_test = dates[train_idx], dates[test_idx]

    return X_train, X_test, y_train, y_test, dates_train, dates_test