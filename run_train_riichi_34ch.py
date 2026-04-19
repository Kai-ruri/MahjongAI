# run_train_riichi_34ch.py
# リーチ判断モデル (2クラス: 0=ダマテン, 1=リーチ) の学習スクリプト
# 入力: dataset_riichi_33ch.pkl (34ch tensor, label 0/1)
# 出力: riichi_34ch_best.pth

import pickle
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from mahjong_model import MahjongResNet_34ch

# ============================================================
# 設定
# ============================================================
DATASET_PATH = r"G:\マイドライブ\MahjongAI\dataset_riichi_v2.pkl"
SAVE_PATH    = r"G:\マイドライブ\MahjongAI\riichi_34ch_best.pth"
LOCAL_SAVE   = rr"G:\マイドライブ\MahjongAI\riichi_34ch_best.pth"

NUM_CLASSES  = 2   # 0=ダマテン, 1=リーチ
NUM_EPOCHS   = 40
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
VAL_RATIO    = 0.1
LABEL_SMOOTH = 0.05
NUM_WORKERS  = 0
SEED         = 42

# ============================================================
# 再現性
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Dataset
# ============================================================
class RiichiDataset(Dataset):
    def __init__(self, records):
        self.tensors = [r["tensor"] for r in records]
        self.labels  = [int(r["label"]) for r in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.tensors[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.long)
        return x, y

# ============================================================
# データ読み込み & 分割
# ============================================================
print(f"データ読み込み中: {DATASET_PATH}")
t0 = time.time()
with open(DATASET_PATH, "rb") as f:
    all_records = pickle.load(f)
print(f"  -> {len(all_records)} 件 ({time.time()-t0:.1f}s)")

label_counts = Counter(int(r["label"]) for r in all_records)
print(f"  ラベル分布: {dict(sorted(label_counts.items()))}")
total = sum(label_counts.values())
class_weights = torch.tensor(
    [total / (NUM_CLASSES * label_counts.get(c, 1)) for c in range(NUM_CLASSES)],
    dtype=torch.float32
)
print(f"  クラス重み: {class_weights.tolist()}")

all_game_ids = list(set(r.get("meta_log_id", str(i)) for i, r in enumerate(all_records)))
random.shuffle(all_game_ids)
n_val = max(1, int(len(all_game_ids) * VAL_RATIO))
val_ids = set(all_game_ids[:n_val])
train_records = [r for r in all_records if r.get("meta_log_id", "") not in val_ids]
val_records   = [r for r in all_records if r.get("meta_log_id", "") in val_ids]
print(f"  train: {len(train_records)} 件  val: {len(val_records)} 件")

train_ds = RiichiDataset(train_records)
val_ds   = RiichiDataset(val_records)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ============================================================
# モデル・損失・最適化
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"デバイス: {device}")

model = MahjongResNet_34ch(num_classes=NUM_CLASSES, num_blocks=3, channels=128,
                            dropout_p_res=0.1, dropout_p_fc=0.3).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"パラメータ数: {total_params:,}")

class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

# ============================================================
# 学習ループ
# ============================================================
best_val_acc = 0.0
print("\n学習開始!\n")

for epoch in range(1, NUM_EPOCHS + 1):
    t_epoch = time.time()
    # --- Train ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_correct += (logits.argmax(1) == y).sum().item()
        train_total += x.size(0)
    scheduler.step()

    # --- Val ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits, y).item() * x.size(0)
            preds = logits.argmax(1)
            val_correct += (preds == y).sum().item()
            val_total += x.size(0)
            for c in range(NUM_CLASSES):
                mask = (y == c)
                class_correct[c] += (preds[mask] == y[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    t_loss = train_loss / train_total
    t_acc  = train_correct / train_total * 100
    v_loss = val_loss / val_total
    v_acc  = val_correct / val_total * 100
    elapsed = time.time() - t_epoch

    class_acc_str = " | ".join(
        f"C{c}:{class_correct[c]/max(class_total[c],1)*100:.1f}%" for c in range(NUM_CLASSES)
    )
    print(f"[Epoch {epoch:3d}/{NUM_EPOCHS}] "
          f"train loss={t_loss:.4f} acc={t_acc:.2f}%  "
          f"val loss={v_loss:.4f} acc={v_acc:.2f}%  "
          f"[{class_acc_str}]  {elapsed:.1f}s")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), LOCAL_SAVE)
        try:
            torch.save(model.state_dict(), SAVE_PATH)
        except Exception:
            pass
        print(f"  ★ ベストモデル保存 (val acc={v_acc:.2f}%)")

print(f"\n学習完了! ベスト val acc: {best_val_acc:.2f}%")
print(f"モデル保存: {LOCAL_SAVE}")
