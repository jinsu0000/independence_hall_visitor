import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.logger import print_once

# 설정
INPUT_CSV = "../data/refined/dataset.csv"
OUTPUT_CSV = "../data/refined/dataset_sbert_ae.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 30
EMBED_DIM = 384
BOTTLENECK_DIM = 32
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-3

# 1. 데이터 로딩
df = pd.read_csv(INPUT_CSV)
df["press_titles"] = df["press_titles"].fillna("")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
dates = df["date"].unique()

# 2. SBERT + AttentionPooling
sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: (N, D)
        weights = self.attn(x)  # (N, 1)
        weights = torch.softmax(weights, dim=0)
        return (x * weights).sum(dim=0)  # (D,)

attn_pool = AttentionPooling(EMBED_DIM).to(DEVICE)

print("📌 Step 1: 날짜별 attention pooled 임베딩 생성")
daily_embeds = []
for date in tqdm(dates):
    titles = df[df["date"] == date]["press_titles"].tolist()
    sentences = []
    for t in titles:
        sentences.extend(t.split(" | "))
    if not sentences:
        sentences = [" "]
    embs = sbert.encode(sentences, convert_to_numpy=True)
    embs_tensor = torch.tensor(embs, dtype=torch.float32).to(DEVICE)
    pooled = attn_pool(embs_tensor).detach().cpu().numpy()
    daily_embeds.append(pooled)

X_all = np.stack(daily_embeds)  # shape: (N_dates, 384)
X_tensor = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)

# 3. AE 정의 및 sliding window 기반 학습
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, BOTTLENECK_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, EMBED_DIM)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

model = AE().to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# 4. Sliding Window 구성: 30일 평균 → 31일째 벡터 예측
print("📌 Step 2: Forecast-AE용 윈도우 구성")
X_input, X_target = [], []
for i in range(len(X_tensor) - WINDOW_SIZE):
    window = X_tensor[i:i+WINDOW_SIZE]         # (30, 384)
    future = X_tensor[i+WINDOW_SIZE]           # (384,)
    window_vec = window.mean(dim=0)
    X_input.append(window_vec.unsqueeze(0))
    X_target.append(future.unsqueeze(0))

X_input = torch.cat(X_input, dim=0)    # (N, 384)
X_target = torch.cat(X_target, dim=0)  # (N, 384)
print(f"Input shape: {X_input.shape}, Target shape: {X_target.shape}")

# 5. AE 학습
print("📌 Step 3: Forecast-AE 학습 시작")
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_input, X_target), batch_size=BATCH_SIZE, shuffle=True)
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        z, recon = model(xb)
        loss = criterion(recon, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(loader):.6f}")

# 6. 임베딩 생성 (예측기반 z 생성)
print("📌 Step 4: 날짜별 임베딩 → z(32차원) 생성")
model.eval()
with torch.no_grad():
    encoder = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder
    Z = encoder(X_tensor).cpu().numpy()

# 5. 저장
df_out = pd.DataFrame(Z, columns=[f"press_emb_{i}" for i in range(BOTTLENECK_DIM)])
df_out.insert(0, "date", dates)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ 저장 완료: {OUTPUT_CSV}")
