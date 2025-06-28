import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from tqdm import tqdm

# 설정
INPUT_CSV = "../data/refined/dataset.csv"
OUTPUT_CSV = "../data/refined/dataset_sbert_embed.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# 1. Dataset 불러오기
df = pd.read_csv(INPUT_CSV)
df["press_titles"] = df["press_titles"].fillna("")

# 2. SBERT 모델 로딩
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

# 3. MLP 정의 (384 -> 128 -> 32)
class EmbeddingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        return self.mlp(x)

mlp_model = EmbeddingMLP().to(DEVICE)

# 4. 제목 임베딩 및 MLP 투영
def encode_titles(title_list):
    return sbert_model.encode(title_list, convert_to_numpy=True)

def project_embeddings(embeddings):
    with torch.no_grad():
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
        projected = mlp_model(tensor)
        return projected.cpu().numpy()

projected_vectors = []

for titles in tqdm(df["press_titles"], desc="Embedding + MLP Projection"):
    title_list = titles.split(" | ") if titles.strip() else [" "]
    embeddings = encode_titles(title_list)
    avg_embedding = np.mean(embeddings, axis=0, keepdims=True)
    projected = project_embeddings(avg_embedding)
    projected_vectors.append(projected[0])

projected_array = np.array(projected_vectors)
projected_df = pd.DataFrame(projected_array, columns=[f"press_emb_{i}" for i in range(projected_array.shape[1])])

# 5. 병합 및 저장
df_concat = pd.concat([df, projected_df], axis=1)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_concat.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ 저장 완료: {OUTPUT_CSV}")
