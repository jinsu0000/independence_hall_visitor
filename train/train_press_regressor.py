
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_CSV = '../data/refined/dataset.csv'

output_dir = "../output_sbert_regressor"
os.makedirs(output_dir, exist_ok=True)

OUTPUT_PRED_CSV = f'{output_dir}/pred_press_only.csv'
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4

# 1. 데이터셋 정의
class PressDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["press_titles"].fillna("제목 없음").tolist()
        self.y = df["attendences"].values.astype(np.float32)
        self.model = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        titles = self.texts[idx].split(" | ")
        embeddings = self.model.encode(titles, convert_to_tensor=True)
        emb_avg = torch.mean(embeddings, dim=0)  # [384]
        return emb_avg, self.y[idx]

# 2. 모델 정의
class RegressionMLP(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

# 3. 데이터 준비
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["press_titles", "attendences"]).reset_index(drop=True)

split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]

sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
train_dataset = PressDataset(df_train, tokenizer=sbert)
test_dataset = PressDataset(df_test, tokenizer=sbert)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. 학습
model = RegressionMLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
 
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss/len(train_loader):.4f}")

# 5. 예측
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        pred = model(xb).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(pred)

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"✅ PRESS 기반 회귀모델 성능 → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

# 결과 저장
df_result = df.iloc[split_idx:].copy()
df_result["pred_from_press"] = y_pred
os.makedirs("../output", exist_ok=True)
df_result[["date", "attendences", "pred_from_press"]].to_csv(OUTPUT_PRED_CSV, index=False, encoding="utf-8-sig")
