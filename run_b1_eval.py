"""
run_b1_eval.py - B-1: 打牌モデルの現状精度測定

現在のモデルのtop-1 / top-3 / top-5 精度を計測する。
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os

from mahjong_model import MahjongResNet_UltimateV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================================
# 1. データ読み込み
# =========================================
def load_dataset(paths):
    all_tensors, all_labels = [], []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [skip] {path}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        for r in data:
            t = np.array(r["tensor"], dtype=np.float32)
            all_tensors.append(t)
            all_labels.append(r["label"])
        print(f"  loaded {len(data)} records from {path}")
    return np.array(all_tensors), np.array(all_labels, dtype=np.int64)

# =========================================
# 2. 精度評価
# =========================================
def evaluate(model, tensors, labels, batch_size=512):
    model.eval()
    n = len(tensors)
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.tensor(tensors[i:i+batch_size]).to(device)
            y = torch.tensor(labels[i:i+batch_size]).to(device)

            logits_discard, _ = model(x)
            probs = F.softmax(logits_discard, dim=1)

            # top-k 精度
            _, top1_idx = probs.topk(1, dim=1)
            _, top3_idx = probs.topk(3, dim=1)
            _, top5_idx = probs.topk(5, dim=1)

            correct_top1 += (top1_idx[:, 0] == y).sum().item()
            correct_top3 += (top3_idx == y.unsqueeze(1)).any(dim=1).sum().item()
            correct_top5 += (top5_idx == y.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "top1": correct_top1 / n * 100,
        "top3": correct_top3 / n * 100,
        "top5": correct_top5 / n * 100,
        "n": n,
    }

# =========================================
# 3. メイン
# =========================================
def main():
    # テストデータ読み込み
    print("\n[1] テストデータを読み込み中...")
    test_tensors, test_labels = load_dataset([
        "dataset_phoenix_closed_precall_part2_test.pkl",
        "dataset_phoenix_closed_precall_part3_test.pkl",
    ])
    print(f"  合計テストサンプル: {len(test_tensors)}")

    # 評価するモデルファイル一覧
    model_files = [
        r"G:\マイドライブ\MahjongAI\mahjong_ultimate_ai_v5_master.pth",
        r"G:\マイドライブ\MahjongAI\mahjong_ultimate_ai_v5_supervised_full.pth",
        r"G:\マイドライブ\MahjongAI\mahjong_ultimate_ai_v5_supervised_5000.pth",
        r"G:\マイドライブ\MahjongAI\mahjong_ultimate_ai_v5_1game.pth",
        r"G:\マイドライブ\MahjongAI\discard_b1_best.pth",  # B-1新規学習モデル
    ]

    print("\n[2] 各モデルを評価中...")
    results = []
    for path in model_files:
        if not os.path.exists(path):
            print(f"  [skip] {path}")
            continue
        model = MahjongResNet_UltimateV3().to(device)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except RuntimeError as e:
            print(f"  [skip] {path}: アーキテクチャ不一致 ({e})")
            continue
        model.eval()
        acc = evaluate(model, test_tensors, test_labels)
        results.append((path, acc))
        print(f"  {path}: top-1={acc['top1']:.2f}%, top-3={acc['top3']:.2f}%, top-5={acc['top5']:.2f}%")

    # ランダムベースラインも計算
    print("\n[3] ランダムベースライン（参考）")
    # 各ラベルの分布
    unique, counts = np.unique(test_labels, return_counts=True)
    max_freq = counts.max() / len(test_labels) * 100
    random_top1 = 100.0 / 34
    print(f"  ランダム top-1 = {random_top1:.2f}%")
    print(f"  最頻ラベル選択 top-1 = {max_freq:.2f}%")

    # ラベル分布確認
    print("\n[4] ラベル（正解打牌）分布 (top-10)")
    tile_names_short = [
        "1m","2m","3m","4m","5m","6m","7m","8m","9m",
        "1p","2p","3p","4p","5p","6p","7p","8p","9p",
        "1s","2s","3s","4s","5s","6s","7s","8s","9s",
        "東","南","西","北","白","發","中"
    ]
    label_counts = np.bincount(test_labels, minlength=34)
    sorted_idx = np.argsort(-label_counts)
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"  {tile_names_short[idx]}: {label_counts[idx]} ({label_counts[idx]/len(test_labels)*100:.1f}%)")

    print("\n==============================")
    print("B-1 ベースライン評価 完了")
    print(f"目標: top-1 > 60%")
    if results:
        best = max(results, key=lambda x: x[1]["top1"])
        print(f"現在最良: {best[0]} → top-1 = {best[1]['top1']:.2f}%")
        gap = 60.0 - best[1]["top1"]
        if gap > 0:
            print(f"ギャップ: {gap:.2f}% の改善が必要")
        else:
            print("既に目標達成！")
    print("==============================")

if __name__ == "__main__":
    main()
