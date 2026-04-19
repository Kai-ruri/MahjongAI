"""
[DEPRECATED] run_b3_train.py - B-3: 残り枚数マップ(CH21)を含む25chデータで再訓練（旧25ch時代の遺物）

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! 警告: このスクリプトは現在のコードベースと互換性がなく、実行すると       !!
!!        RuntimeError (shape mismatch) で失敗します。                       !!
!!        使用しないでください。                                             !!
!!                                                                            !!
!! 代替手順:                                                                  !!
!!   1. python build_supervised_dataset.py  # 33ch データセット生成          !!
!!   2. python run_train_33ch.py            # 33ch モデル学習                !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

【非推奨となった理由】
  このスクリプトは MahjongResNet_UltimateV3 を使用するが、同モデルの入力層は
  旧25ch時代に in_channels=25 で設計されたのち、33ch へ移行した (commit 50d33da)。
  一方、このスクリプトが読み込む dataset_b3_*.pkl は shape=(25, 34) の旧形式のままであり
  モデルの期待する (33, 34) と一致しない。

  CH22（残り牌枚数マップ）は現在 MahjongStateV5.to_tensor() 内で正確に計算されるため、
  このスクリプトが行っていた CH21 への後付けエンリッチ処理も不要になった。
  （関連: run_b3_enrich.py にも同様の廃止注記あり）

【旧設計メモ（参考のみ）】
  戦略:
    1. discard_b1_best.pth (64.13%) から転移学習
    2. CH21に残り枚数マップが入った dataset_b3_*.pkl で学習
    3. 最良モデルを discard_b3_best.pth として保存
"""

import sys
print("="*70)
print("[DEPRECATED] run_b3_train.py は旧25ch時代のスクリプトです。")
print("  現在の MahjongResNet_UltimateV3 は 33ch 入力を期待しますが、")
print("  このスクリプトが読み込む pkl ファイルは shape=(25,34) のため")
print("  RuntimeError が発生します。")
print("")
print("  代わりに以下を実行してください:")
print("    1. python build_supervised_dataset.py")
print("    2. python run_train_33ch.py")
print("="*70)
sys.exit(1)


import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import time
import random

from mahjong_model import MahjongResNet_UltimateV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class DiscardDataset(Dataset):
    def __init__(self, records):
        self.tensors = np.array([r["tensor"] for r in records], dtype=np.float32)
        self.labels = np.array([r["label"] for r in records], dtype=np.int64)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return torch.tensor(self.tensors[idx]), torch.tensor(self.labels[idx])


def load_pkl(paths):
    all_records = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [skip] {path}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        all_records.extend(data)
        print(f"  loaded {len(data):>7,} records from {path}")
    return all_records


def evaluate_model(model, loader):
    model.eval()
    correct_top1 = correct_top3 = correct_top5 = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

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
            total += len(y)

    return {
        "loss": total_loss / total,
        "top1": correct_top1 / total * 100,
        "top3": correct_top3 / total * 100,
        "top5": correct_top5 / total * 100,
    }


def train(model, train_loader, val_loader, num_epochs=30, lr=2e-4, save_path="discard_b3_best.pth"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_top1 = 0.0
    best_epoch = 0

    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainTop1':>10} {'ValLoss':>9} {'ValTop1':>9} {'ValTop3':>9}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = total_correct = total_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += len(y)

        scheduler.step()
        train_loss = total_loss / total_n
        train_top1 = total_correct / total_n * 100

        val = evaluate_model(model, val_loader)

        mark = ""
        if val["top1"] > best_val_top1:
            best_val_top1 = val["top1"]
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            mark = " [best]"

        print(f"{epoch:>5} {train_loss:>10.4f} {train_top1:>9.2f}% {val['loss']:>9.4f} "
              f"{val['top1']:>8.2f}% {val['top3']:>8.2f}%{mark}")

    print(f"\n[best] epoch {best_epoch}, val top-1 = {best_val_top1:.2f}%")
    return best_val_top1


def main():
    print("\n=== B-3: 残り枚数マップ(CH21)を活用した再訓練 ===\n")

    # --- データ読み込み ---
    print("[1] データ読み込み...")
    orig_train = load_pkl(["dataset_b3_part1.pkl"])
    downloaded = load_pkl(["dataset_b3_downloaded.pkl"])

    if downloaded:
        game_groups = defaultdict(list)
        for r in downloaded:
            game_groups[r['meta_log_id']].append(r)
        game_ids = sorted(game_groups.keys())
        random.seed(42)
        random.shuffle(game_ids)
        n_val = max(1, len(game_ids) // 10)
        val_ids = set(game_ids[:n_val])
        train_ids = set(game_ids[n_val:])
        dl_train = [r for gid in train_ids for r in game_groups[gid]]
        dl_val   = [r for gid in val_ids   for r in game_groups[gid]]
        print(f"  downloaded: train {len(train_ids)} games ({len(dl_train):,}), val {len(val_ids)} games ({len(dl_val):,})")
        train_records = orig_train + dl_train
        val_records = dl_val
    else:
        train_records = orig_train
        val_records = load_pkl(["dataset_b3_part2_test.pkl"])

    test_records = load_pkl(["dataset_b3_part3_test.pkl"])

    print(f"\n  train: {len(train_records):,}, val: {len(val_records):,}, test: {len(test_records):,}")

    # CH21の統計確認
    ch21_mean = np.mean([r["tensor"][21].mean() for r in test_records[:100]])
    print(f"  CH21 (test, first 100): mean={ch21_mean:.3f} (0=残り0枚, 1=残り4枚)")

    train_ds = DiscardDataset(train_records)
    val_ds   = DiscardDataset(val_records)
    test_ds  = DiscardDataset(test_records)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

    # --- ベースライン評価 ---
    print("\n[2] ベースライン評価 (discard_b1_best.pth)...")
    model_base = MahjongResNet_UltimateV3().to(device)
    if os.path.exists("discard_b1_best.pth"):
        model_base.load_state_dict(torch.load("discard_b1_best.pth", map_location=device))
        model_base.eval()
        acc_base = evaluate_model(model_base, test_loader)
        print(f"  B-1モデル (CH21=0 のデータで評価): top-1={acc_base['top1']:.2f}%, top-3={acc_base['top3']:.2f}%")
    del model_base

    # --- 転移学習 ---
    print("\n[3] 転移学習 (discard_b1_best.pth -> B-3)...")
    model = MahjongResNet_UltimateV3().to(device)
    base_path = "discard_b1_best.pth"
    if os.path.exists(base_path):
        model.load_state_dict(torch.load(base_path, map_location=device))
        print(f"  {base_path} から初期化 (転移学習)")
        INIT_LR = 2e-4
    else:
        print("  ランダム初期化")
        INIT_LR = 1e-3

    NUM_EPOCHS = 30
    train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=INIT_LR,
          save_path="discard_b3_best.pth")

    # --- 最終テスト ---
    print("\n[4] 最終テスト評価 (discard_b3_best.pth)...")
    final_model = MahjongResNet_UltimateV3().to(device)
    final_model.load_state_dict(torch.load("discard_b3_best.pth", map_location=device))
    final = evaluate_model(final_model, test_loader)

    print("\n" + "=" * 45)
    print("B-3 結果 (CH21=残り枚数マップ)")
    print("=" * 45)
    print(f"  top-1: {final['top1']:.2f}%")
    print(f"  top-3: {final['top3']:.2f}%")
    print(f"  top-5: {final['top5']:.2f}%")

    if os.path.exists("discard_b1_best.pth"):
        model_b1 = MahjongResNet_UltimateV3().to(device)
        model_b1.load_state_dict(torch.load("discard_b1_best.pth", map_location=device))
        b1 = evaluate_model(model_b1, test_loader)
        diff = final['top1'] - b1['top1']
        sign = "+" if diff >= 0 else ""
        print(f"  vs B-1: {sign}{diff:.2f}pt")
    print("=" * 45)
    print("\n[完了] discard_b3_best.pth を保存しました")


if __name__ == "__main__":
    main()
