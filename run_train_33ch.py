"""
run_train_33ch.py
33チャンネルテンソルによる打牌モデル（MahjongResNet_UltimateV3）の再訓練スクリプト。

事前準備:
  1. build_supervised_dataset.py を実行して
     dataset_intermediate_phoenix.pkl を生成しておく。
  2. mahjong_model.py の MahjongResNet_UltimateV3 が in_channels=33 になっていること。

出力:
  discard_33ch_best.pth  --- 検証top-1が最高のモデル
"""

import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mahjong_model import MahjongResNet_UltimateV3

# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SAVE_PATH   = r"G:\マイドライブ\MahjongAI\discard_33ch_best.pth"
BATCH_SIZE  = 256
NUM_EPOCHS  = 40
INIT_LR     = 5e-4
VAL_RATIO   = 0.10   # ゲーム単位で10%をバリデーションに使用
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
class DiscardDataset(Dataset):
    def __init__(self, records):
        self.tensors = np.array([r["tensor"] for r in records], dtype=np.float32)
        self.labels  = np.array([r["label"]  for r in records], dtype=np.int64)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return torch.tensor(self.tensors[idx]), torch.tensor(self.labels[idx])


def load_pkl(paths):
    all_records = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [skip] {path} -- not found")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        all_records.extend(data)
        print(f"  loaded {len(data):>8,} records from {path}")
    return all_records


def split_by_game(records, val_ratio=0.10, seed=42):
    """対局単位でtrain/valに分割（データリーク防止）"""
    game_groups = defaultdict(list)
    for r in records:
        game_groups[r.get("meta_log_id", "unknown")].append(r)

    game_ids = sorted(game_groups.keys())
    random.seed(seed)
    random.shuffle(game_ids)

    n_val    = max(1, int(len(game_ids) * val_ratio))
    val_ids  = set(game_ids[:n_val])
    train_ids = set(game_ids[n_val:])

    train_records = [r for gid in train_ids for r in game_groups[gid]]
    val_records   = [r for gid in val_ids   for r in game_groups[gid]]

    print(f"  split: train {len(train_ids)} games ({len(train_records):,} records), "
          f"val {len(val_ids)} games ({len(val_records):,} records)")
    return train_records, val_records


def verify_channel_count(records, expected=33, n_check=10):
    """テンソルのチャンネル数が想定通りか確認"""
    for r in records[:n_check]:
        shape = r["tensor"].shape
        if shape[0] != expected:
            raise ValueError(
                f"テンソルのチャンネル数が {shape[0]} です。"
                f"期待値 {expected} と一致しません。\n"
                "build_supervised_dataset.py を再実行してください。"
            )
    print(f"  channel check OK: shape = {records[0]['tensor'].shape}")


# ---------------------------------------------------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct_top1 = correct_top3 = correct_top5 = total_n = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            probs = F.softmax(logits, dim=1)
            _, top1 = probs.topk(1, dim=1)
            _, top3 = probs.topk(min(3, logits.size(1)), dim=1)
            _, top5 = probs.topk(min(5, logits.size(1)), dim=1)

            correct_top1 += (top1[:, 0] == y).sum().item()
            correct_top3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
            correct_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total_n += len(y)

    return {
        "loss":  total_loss  / total_n,
        "top1":  correct_top1 / total_n * 100,
        "top3":  correct_top3 / total_n * 100,
        "top5":  correct_top5 / total_n * 100,
    }


def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_val_top1 = 0.0
    best_epoch = 0

    header = f"{'Epoch':>5} {'TrLoss':>8} {'TrTop1':>8} {'VaLoss':>8} {'VaTop1':>8} {'VaTop3':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = total_correct = total_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss   += loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n      += len(y)

        scheduler.step()
        tr_loss = total_loss  / total_n
        tr_top1 = total_correct / total_n * 100

        val = evaluate(model, val_loader, criterion)
        mark = ""
        if val["top1"] > best_val_top1:
            best_val_top1 = val["top1"]
            best_epoch = epoch
            torch.save(model.state_dict(), SAVE_PATH)
            mark = " [best]"

        elapsed = time.time() - t0
        print(f"{epoch:>5} {tr_loss:>8.4f} {tr_top1:>7.2f}%"
              f" {val['loss']:>8.4f} {val['top1']:>7.2f}% {val['top3']:>7.2f}%"
              f"  {elapsed:.0f}s{mark}")

    print(f"\n[result] best epoch = {best_epoch}, val top-1 = {best_val_top1:.2f}%")
    print(f"[saved]  {SAVE_PATH}")
    return best_val_top1


# ---------------------------------------------------------------------------
def main():
    print("\n=== run_train_33ch.py: 33チャンネル打牌モデル再訓練 ===\n")

    # 1. データ読み込み
    print("[1] データ読み込み...")
    records = load_pkl(["dataset_intermediate_phoenix.pkl"])
    if not records:
        print("データが見つかりません。先に build_supervised_dataset.py を実行してください。")
        return

    verify_channel_count(records, expected=33)

    # 2. Train/Val 分割（ゲーム単位）
    print("\n[2] Train/Val 分割...")
    train_records, val_records = split_by_game(records, val_ratio=VAL_RATIO, seed=RANDOM_SEED)

    if not val_records:
        print("バリデーションデータが空です。データを確認してください。")
        return

    train_ds = DiscardDataset(train_records)
    val_ds   = DiscardDataset(val_records)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n  DataLoader: train {len(train_ds):,}, val {len(val_ds):,}")

    # 3. チャンネル統計確認
    print("\n[3] チャンネル統計...")
    sample = np.array([r["tensor"] for r in val_records[:500]], dtype=np.float32)
    for ch_idx, ch_name in [(0, "手牌"), (20, "自己捨て"), (22, "残り牌"), (23, "シャンテン"),
                             (24, "受け入れ"), (25, "EVダマ"), (28, "着順"), (30, "赤5m")]:
        mean_val = sample[:, ch_idx, :].mean()
        nonzero  = (sample[:, ch_idx, :] != 0).mean() * 100
        print(f"  CH{ch_idx:02d} ({ch_name:6s}): mean={mean_val:.4f}, non-zero={nonzero:.1f}%")

    # 4. モデル初期化
    print("\n[4] モデル初期化...")
    model = MahjongResNet_UltimateV3().to(device)

    # 旧モデル（25ch）からの転移学習は不可なのでランダム初期化
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  MahjongResNet_UltimateV3 (33ch): {total_params:,} parameters")
    print(f"  ランダム初期化（旧25chモデルとは互換性なし）")

    # 5. 訓練
    print(f"\n[5] 訓練開始 ({NUM_EPOCHS} epochs, lr={INIT_LR}, batch={BATCH_SIZE})...")
    train(model, train_loader, val_loader)

    # 6. 最終テスト（val で評価）
    print("\n[6] 最終評価...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    criterion = nn.CrossEntropyLoss()
    final = evaluate(model, val_loader, criterion)
    print(f"  Val top-1: {final['top1']:.2f}%  top-3: {final['top3']:.2f}%  top-5: {final['top5']:.2f}%")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
