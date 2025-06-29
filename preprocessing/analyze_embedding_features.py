import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr

# 설정
INPUT_CSV = "../data/refined/dataset_ae_per_day.csv"
SAVE_DIR = "../data/figure/embed_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. 데이터 로딩
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
embedding_cols = [col for col in df.columns if col.startswith("press_emb_")]

X = df[embedding_cols].values
y = df["visitor_count"].values
dates = df["date"]

print(f"✅ 로딩 완료: {X.shape[0]}개 날짜, {X.shape[1]}차원 임베딩")

# 2. PCA
print("📊 PCA 분석 중...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=30)
plt.colorbar(sc, label="Visitor Count")
plt.title("PCA of Day Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/pca_visitors.png")
plt.close()

# 3. t-SNE (옵션: 느릴 수 있음)
print("📊 t-SNE 분석 중...")
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="plasma", s=30)
plt.colorbar(sc, label="Visitor Count")
plt.title("t-SNE of Day Embeddings")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/tsne_visitors.png")
plt.close()

# 4. 상관관계 분석 (visitor_count vs press_emb_i)
print("📈 visitor_count와 임베딩 차원 상관관계 계산")
correlations = [pearsonr(df[col], y)[0] for col in embedding_cols]

plt.figure(figsize=(10, 4))
sns.barplot(x=list(range(len(correlations))), y=correlations)
plt.title("Correlation between Embedding Dimensions and Visitor Count")
plt.xlabel("Embedding Dimension")
plt.ylabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/corr_visitor_vs_dim.png")
plt.close()

# 5. 임베딩 내부 상호 상관 히트맵
print("🧠 임베딩 차원 간 상호 상관관계 히트맵")
corr_matrix = pd.DataFrame(X, columns=embedding_cols).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True)
plt.title("Inter-Dimension Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/embedding_corr_heatmap.png")
plt.close()

# 6. PC1/PC2와 방문객 수 비교
print("📈 주성분 vs 방문객 수")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], y, c="blue", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("Visitor Count")
plt.title("PC1 vs Visitor Count")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 1], y, c="green", alpha=0.7)
plt.xlabel("PC2")
plt.ylabel("Visitor Count")
plt.title("PC2 vs Visitor Count")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/pc_vs_visitors.png")
plt.close()

print("✅ 분석 완료! 결과는 폴더에 저장됨:", SAVE_DIR)
