"""
[DEPRECATED] run_b1_train.py - B-1: 打牌モデル精度向上トレーニング（旧25ch時代の遺物）

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
  一方、このスクリプトが読み込む dataset_b1_downloaded.pkl 等は shape=(25, 34) の
  旧形式のままであり、モデルの期待する (33, 34) と一致しない。

【旧設計メモ（参考のみ）】
  目標: top-1 精度 60%超
  戦略:
    1. supervised_full.pth (61.15%) をベースに転移学習で追加改善
    2. label smoothing / cosine LR / gradient clip など最適化を強化
    3. 最良モデルを discard_b1_best.pth として保存
  注意: CH21/CH22 (シャンテン/受け入れ) のリアルタイム計算は処理コストが非常に高いため
         ここでは既存の25チャンネル特徴のまま学習していた。
"""

import sys
print("="*70)
print("[DEPRECATED] run_b1_train.py は旧25ch時代のスクリプトです。")
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import time

from mahjong_model import MahjongResNet_UltimateV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# =========================================
# 1. Dataset クラス
# =========================================
class DiscardDataset(Dataset):
    def __init__(self, records):
        self.tensors = np.array([r["tensor"] for r in records], dtype=np.float32)
        self.labels = np.array([r["label"] for r in records], dtype=np.int64)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return torch.tensor(self.tensors[idx]), torch.tensor(self.labels[idx])


# =========================================
# 2. データ読み込み
# =========================================
def load_pkl(paths):
    all_records = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [skip] {path}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        all_records.extend(data)
        print(f"  loaded {len(data):>6} records from {path}")
    return all_records


# =========================================
# 3. 精度評価
# =========================================
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


# =========================================
# 4. 学習ループ
# =========================================
def train(model, train_loader, val_loader, num_epochs=30, lr=3e-4, save_path=r"G:\マイドライブ\MahjongAI\discard_b1_best.pth"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    # label_smoothing: モデルが自信過剰になるのを防ぐ
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

    print(f"\n最良: epoch {best_epoch}, val top-1 = {best_val_top1:.2f}%")
    return best_val_top1


# =========================================
# 5. メイン
# =========================================
def main():
    print("\n=== B-1: 打牌モデル精度向上 ===\n")

    # データ読み込み
    print("[1] データ読み込み...")
    orig_train = load_pkl(["dataset_phoenix_closed_precall_part1.pkl"])
    downloaded = load_pkl(["dataset_b1_downloaded.pkl"])  # ダウンロード済み

    # ダウンロードデータをゲームIDで train / val に分割（新規ゲームのみ）
    if downloaded:
        import random
        # ゲームIDでグループ化
        from collections import defaultdict
        game_groups = defaultdict(list)
        for r in downloaded:
            game_groups[r['meta_log_id']].append(r)
        game_ids_sorted = sorted(game_groups.keys())
        random.seed(42)
        random.shuffle(game_ids_sorted)
        n_val_games = max(1, len(game_ids_sorted) // 10)  # 10%を検証に
        val_game_ids = set(game_ids_sorted[:n_val_games])
        train_game_ids = set(game_ids_sorted[n_val_games:])
        dl_train = [r for gid in train_game_ids for r in game_groups[gid]]
        dl_val   = [r for gid in val_game_ids   for r in game_groups[gid]]
        print(f"  ダウンロード: train {len(train_game_ids)} games ({len(dl_train)} rec), "
              f"val {len(val_game_ids)} games ({len(dl_val)} rec)")
        train_records = orig_train + dl_train
        # 検証: ダウンロードの一部 (新規ゲームのみ)
        val_records = dl_val if dl_val else load_pkl(["dataset_phoenix_closed_precall_part2_test.pkl"])
    else:
        train_records = orig_train
        val_records   = load_pkl(["dataset_phoenix_closed_precall_part2_test.pkl"])
        print("  ダウンロードデータなし。既存データのみで学習。")

    test_records  = load_pkl(["dataset_phoenix_closed_precall_part3_test.pkl"])

    train_ds = DiscardDataset(train_records)
    val_ds   = DiscardDataset(val_records)
    test_ds  = DiscardDataset(test_records)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

    print(f"\n  学習:{len(train_ds)}, 検証:{len(val_ds)}, テスト:{len(test_ds)}")

    # ベースライン評価
    print("\n[2] ベースライン評価...")
    for path in [r"G:\マイドライブ\MahjongAI\mahjong_ultimate_ai_v5_master.pth", "mahjong_ultimate_ai_v5_supervised_full.pth"]:
        if not os.path.exists(path):
            continue
        m = MahjongResNet_UltimateV3().to(device)
        try:
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            acc = evaluate_model(m, test_loader)
            print(f"  {path}: top-1={acc['top1']:.2f}%, top-3={acc['top3']:.2f}%")
        except RuntimeError as e:
            print(f"  {path}: スキップ ({e})")

    # 転移学習: supervised_full.pth から開始
    print("\n[3] 転移学習 (supervised_full.pth ベース)...")
    model = MahjongResNet_UltimateV3().to(device)
    base_path = r"G:\マイドライブ\MahjongAI\mahjong_ultimate_ai_v5_supervised_full.pth"
    if os.path.exists(base_path):
        model.load_state_dict(torch.load(base_path, map_location=device))
        print(f"  {base_path} から初期化")
        INIT_LR = 3e-4  # 転移学習は小さいLR
    else:
        print("  ランダム初期化")
        INIT_LR = 1e-3

    NUM_EPOCHS = 30
    train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=INIT_LR, save_path=r"G:\マイドライブ\MahjongAI\discard_b1_best.pth")

    # 最終テスト評価
    print("\n[4] 最終テスト評価 (discard_b1_best.pth)...")
    final_model = MahjongResNet_UltimateV3().to(device)
    final_model.load_state_dict(torch.load(r"G:\マイドライブ\MahjongAI\discard_b1_best.pth", map_location=device))
    final = evaluate_model(final_model, test_loader)

    print("\n" + "=" * 45)
    print("B-1 結果")
    print("=" * 45)
    print(f"  top-1: {final['top1']:.2f}%")
    print(f"  top-3: {final['top3']:.2f}%")
    print(f"  top-5: {final['top5']:.2f}%")
    if final["top1"] >= 60.0:
        print("  [OK] 目標 60% 達成!")
    else:
        print(f"  [NG] 目標まで {60.0 - final['top1']:.2f}% 不足")
    print("=" * 45)
    print("\n保存先: discard_b1_best.pth")
    print("本番反映: python benchmark.py --ai-model discard_b1_best.pth")


if __name__ == "__main__":
    main()
