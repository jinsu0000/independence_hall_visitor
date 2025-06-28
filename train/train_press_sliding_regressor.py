
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CSV_PATH = '../data/refined/dataset.csv'

output_dir = "../output_sliding"
os.makedirs(output_dir, exist_ok=True)

OUTPUT_PRED_CSV = f'{output_dir}/pred_press_sliding.csv'
OUTPUT_MODEL_VIS = '../output_sliding/press_sliding_model.png'
WINDOW_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4

# 1. Dataset 정의
class SlidingPressDataset(Dataset):
    def __init__(self, df, model, window=30):
        self.df = df.reset_index(drop=True)
        self.window = window
        self.model = model
        self.valid_indices = [
            i for i in range(window, len(df))
            if all(isinstance(val, str) and val.strip() != '' for val in df["press_titles"].iloc[i-window:i])
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        past_titles = self.df["press_titles"].iloc[real_idx - self.window:real_idx].tolist()
        all_titles = []
        for entry in past_titles:
            all_titles.extend(entry.split(" | "))
        embeddings = self.model.encode(all_titles, convert_to_tensor=True, show_progress_bar=False)
        emb_avg = torch.mean(embeddings, dim=0)  # shape: [384]
        y = self.df["attendences"].iloc[real_idx]
        date = self.df["date"].iloc[real_idx]
        return emb_avg, torch.tensor(y, dtype=torch.float32), date

# 2. 모델 정의
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze()

# 3. 데이터 로딩
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["press_titles", "attendences"])
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]

print("✅ Dataset loaded.")

# SBERT 모델 로드
sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

# Sliding dataset
print("📦 Encoding dataset using sliding window...")
train_dataset = SlidingPressDataset(df_train, sbert, window=WINDOW_SIZE)
test_dataset = SlidingPressDataset(df_test, sbert, window=WINDOW_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 모델 준비 (멀티 GPU 지원)
model = MLPRegressor()
if torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model.to(DEVICE)

print(model)

# 시각화를 위한 구조 출력
with open(OUTPUT_MODEL_VIS.replace(".png", ".txt"), "w") as f:
    f.write(str(model))

# 학습 준비
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# 학습 진행
print("🧠 Training model...")
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"📉 Epoch {epoch+1} Loss: {avg_loss:.4f}")

# Loss 그래프 저장
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_MODEL_VIS)
plt.close()

# 예측
model.eval()
preds, targets, dates = [], [], []

with torch.no_grad():
    for xb, yb, date in tqdm(test_loader, desc="🔍 Predicting"):
        xb = xb.to(DEVICE)
        pred = model(xb).cpu().numpy()
        preds.extend(pred)
        targets.extend(yb.numpy())
        dates.extend(date)

# 저장 및 평가
df_result = pd.DataFrame({
    "date": dates,
    "attendences": targets,
    "pred_from_press_sliding": preds
})
os.makedirs("../output", exist_ok=True)
df_result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

mae = mean_absolute_error(targets, preds)
rmse = mean_squared_error(targets, preds, squared=False)
r2 = r2_score(targets, preds)

print(f"✅ Done! MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
print(f"📁 Prediction result saved to {OUTPUT_CSV}")
