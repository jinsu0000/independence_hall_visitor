
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 데이터 로드
df = pd.read_csv("../data/refined/dataset.csv", encoding="utf-8-sig", parse_dates=["date"])

# 저장 디렉토리 설정
output_dir = "../data/figure"
os.makedirs(output_dir, exist_ok=True)

# 1. 상관계수 - Pearson/Spearman/Kendall 모두 시각화
methods = ["pearson", "spearman", "kendall"]
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

for method in methods:
    corr = df[numeric_cols].corr(method=method)["attendences"].drop("attendences").sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr.values, y=corr.index)
    plt.title(f"{method.capitalize()} Correlation with Attendences")
    plt.xlabel(f"{method.capitalize()} Correlation Coefficient")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_with_attendences_{method}.png")
    plt.close()

# 2. press_count - 로그 스케일 시각화
plt.figure(figsize=(7, 5))
sns.scatterplot(x=df["press_count"], y=df["attendences"])
plt.xscale("symlog")  # log but handles 0
plt.xlabel("press_count (log scale)")
plt.ylabel("Attendences")
plt.title("Attendences vs press_count (log scale)")
plt.tight_layout()
plt.savefig(f"{output_dir}/attendences_vs_press_count_log.png")
plt.close()

# 3. 월별로 관람객 수 평균이 높은 달 기준, 주요 feature 평균값 시각화
month_feature_cols = ['avg_temp', 'max_temp', 'min_temp', 'precipitation', 'max_hourly_rain', 'press_count']
df['month_name'] = df['date'].dt.month_name().str[:3]  # Jan, Feb ...
monthly_summary = df.groupby('month_name')[['attendences'] + month_feature_cols].mean()
monthly_order = monthly_summary['attendences'].sort_values(ascending=False).index.tolist()

plt.figure(figsize=(12, 6))
monthly_summary.loc[monthly_order, month_feature_cols].plot(kind="bar", figsize=(12, 6))
plt.title("Monthly Mean of Features (Sorted by Attendences)")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/monthly_feature_means_by_attendence.png")
plt.close()

# 4. 시계열 추이 (재시각화)
plt.figure(figsize=(14, 4))
plt.plot(df["date"], df["attendences"], label="Attendences")
plt.title("Daily Attendences Over Time")
plt.xlabel("Date")
plt.ylabel("Attendences")
plt.tight_layout()
plt.savefig(f"{output_dir}/attendences_timeseries.png")
plt.close()

# 5. 월별 Feature 평균 구하기
df['year_month'] = df['date'].dt.to_period("M")
features = ['avg_temp', 'max_temp', 'min_temp', 'precipitation', 'max_hourly_rain', 'press_count']
monthly_mean = df.groupby("year_month")[features + ['attendences']].mean().reset_index()
monthly_mean['year_month'] = monthly_mean['year_month'].astype(str)

# 각 feature별로 저장
for feat in features:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # 관람객 수 (상단)
    ax1.plot(monthly_mean['year_month'], monthly_mean['attendences'], color='deeppink')
    ax1.set_title("Monthly Average Attendences")
    ax1.set_ylabel("Attendences")
    ax1.grid(True)

    # 선택 feature (하단)
    ax2.plot(monthly_mean['year_month'], monthly_mean[feat], color='steelblue')
    ax2.set_title(f"Monthly Average of {feat}")
    ax2.set_xlabel("Year-Month")
    ax2.set_ylabel(feat)
    ax2.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 저장
    save_path = os.path.join(output_dir, f"monthly_{feat}_with_attendences.png")
    plt.savefig(save_path)
    plt.close()

# 분리된 그래프 생성
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

# 상단: 관람객 수
sns.lineplot(data=monthly_mean, x='year_month', y='attendences', ax=ax1, color='deeppink')
ax1.set_title("Monthly Average Attendences")
ax1.set_ylabel("Attendences")
ax1.grid(True)

# 하단: 나머지 feature
for col in ['avg_temp', 'max_temp', 'min_temp', 'precipitation', 'max_hourly_rain', 'press_count']:
    ax2.plot(monthly_mean['year_month'], monthly_mean[col], label=col)

ax2.set_title("Monthly Average of Features (Excluding Attendences)")
ax2.set_xlabel("Year-Month")
ax2.set_ylabel("Value")
ax2.legend()
ax2.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# 저장
plt.savefig(f"{output_dir}/monthly_average_split_lineplots.png")
plt.close()



print("✅ 전체 월별 평균값 시각화 완료 ")
