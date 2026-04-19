"""
run_full_pipeline.py
全工程を順番に自動実行するパイプラインスクリプト

実行順:
  1. データ抽出 (build_supervised_dataset.py 相当 + 再開機能付き)
  2. 打牌モデル学習 (run_b1_train.py 相当 + dataset_intermediate_phoenix.pkl 追加)
  3. 10,000ゲームベンチマーク
  4. 天鳳オンライン対局
"""

import gzip
import json
import os
import pickle
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ========================================================
# 設定
# ========================================================
LOGS_DIR           = "logs"
PROGRESS_FILE      = "dataset_intermediate_progress.json"  # 処理済みIDを追跡
INTERMEDIATE_PKL   = "dataset_intermediate_phoenix.pkl"
SAVE_MODEL_PATH    = r"G:\マイドライブ\MahjongAI\discard_b1_best.pth"
NUM_EPOCHS         = 30
BATCH_SIZE         = 256
LR                 = 3e-4
BENCHMARK_GAMES    = 10000
TENHOU_GAMES       = 30  # 天鳳での対局数
CHECKPOINT_EVERY   = 100  # 100対局ごとに中間保存

# ========================================================
# STEP 1: データ抽出（再開機能付き）
# ========================================================
def step1_extract_data():
    print("\n" + "=" * 60)
    print("STEP 1: データ抽出")
    print("=" * 60)

    from build_supervised_dataset import extract_dataset_from_xml

    # .gz ファイルから鳳凰卓対局IDを収集
    existing_files = sorted([f for f in os.listdir(LOGS_DIR) if f.endswith('.gz')])
    pattern = re.compile(r'log=(\d{10}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{8})')
    print(f"  .gz ファイル: {len(existing_files)} 個")

    phoenix_log_ids = []
    for filename in existing_files:
        with gzip.open(os.path.join(LOGS_DIR, filename), 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "四鳳" in line:
                    for log_id in pattern.findall(line):
                        phoenix_log_ids.append(log_id)

    # 重複除去
    phoenix_log_ids = list(dict.fromkeys(phoenix_log_ids))
    print(f"  鳳凰卓対局ID: {len(phoenix_log_ids)} 件")

    # 進捗ファイルから処理済みIDを読み込む（再開機能）
    processed_ids = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
            processed_ids = set(progress.get("processed_ids", []))
        print(f"  再開: {len(processed_ids)} 件処理済み → 残り {len(phoenix_log_ids) - len(processed_ids)} 件")

    # 既存データを読み込む
    all_records = []
    if os.path.exists(INTERMEDIATE_PKL) and processed_ids:
        with open(INTERMEDIATE_PKL, 'rb') as f:
            all_records = pickle.load(f)
        print(f"  既存データ読み込み: {len(all_records)} 件")

    # 未処理IDのみ処理
    remaining_ids = [lid for lid in phoenix_log_ids if lid not in processed_ids]
    print(f"  ダウンロード開始: {len(remaining_ids)} 件\n")

    for idx, log_id in enumerate(remaining_ids):
        global_idx = len(processed_ids) + idx + 1
        print(f"  [{global_idx}/{len(phoenix_log_ids)}] {log_id}", flush=True)
        try:
            url = f"https://tenhou.net/0/log/?{log_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
            try:
                xml_string = gzip.decompress(raw).decode('utf-8')
            except Exception:
                xml_string = raw.decode('utf-8')

            records = extract_dataset_from_xml(xml_string, log_id=log_id)
            all_records.extend(records)
            processed_ids.add(log_id)
            print(f"    -> {len(records)} 件 (累計: {len(all_records)} 件)", flush=True)
            time.sleep(1.0)

            # CHECKPOINT_EVERY ごとに中間保存
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                with open(INTERMEDIATE_PKL, 'wb') as f:
                    pickle.dump(all_records, f)
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump({"processed_ids": list(processed_ids)}, f)
                print(f"  [チェックポイント保存: {len(all_records)} 件]", flush=True)

        except Exception as e:
            print(f"    [!] スキップ: {e}", flush=True)
            processed_ids.add(log_id)  # エラーでもスキップ済みとしてマーク
            continue

    # 最終保存
    with open(INTERMEDIATE_PKL, 'wb') as f:
        pickle.dump(all_records, f)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"processed_ids": list(processed_ids)}, f)

    print(f"\n  抽出完了: 合計 {len(all_records)} 件")
    return len(all_records)


# ========================================================
# STEP 2: 打牌モデル学習
# ========================================================
class DiscardDataset(Dataset):
    def __init__(self, records):
        self.tensors = np.array([r["tensor"] for r in records], dtype=np.float32)
        self.labels  = np.array([r["label"]  for r in records], dtype=np.int64)
    def __len__(self):          return len(self.tensors)
    def __getitem__(self, idx): return torch.tensor(self.tensors[idx]), torch.tensor(self.labels[idx])


def evaluate_model(model, loader, device):
    model.eval()
    correct1 = correct3 = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            _, top1 = logits.topk(1, dim=1)
            _, top3 = logits.topk(min(3, logits.size(1)), dim=1)
            correct1 += (top1[:, 0] == y).sum().item()
            correct3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += len(y)
    return {"loss": total_loss/total, "top1": correct1/total*100, "top3": correct3/total*100}


def load_pkl(paths):
    all_records = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [skip] {path}")
            continue
        with open(path, 'rb') as f:
            data = pickle.load(f)
        all_records.extend(data)
        print(f"  loaded {len(data):>7,} records  <- {path}")
    return all_records


def step2_train():
    print("\n" + "=" * 60)
    print("STEP 2: 打牌モデル学習")
    print("=" * 60)

    from mahjong_model import MahjongResNet_UltimateV3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # データ読み込み
    print("\n[1] データ読み込み...")
    part1      = load_pkl(["dataset_phoenix_closed_precall_part1.pkl"])
    downloaded = load_pkl(["dataset_b1_downloaded.pkl"])
    intermediate = load_pkl([INTERMEDIATE_PKL])

    # downloaded を game_id で train/val 分割
    from collections import defaultdict
    import random
    game_groups = defaultdict(list)
    for r in downloaded:
        game_groups[r.get('meta_log_id', 'unknown')].append(r)
    game_ids = sorted(game_groups.keys())
    random.seed(42)
    random.shuffle(game_ids)
    n_val = max(1, len(game_ids) // 10)
    val_ids   = set(game_ids[:n_val])
    train_ids = set(game_ids[n_val:])
    dl_train = [r for gid in train_ids for r in game_groups[gid]]
    dl_val   = [r for gid in val_ids   for r in game_groups[gid]]
    print(f"  downloaded split: train {len(dl_train):,} / val {len(dl_val):,}")

    # intermediate も同様に分割
    if intermediate:
        ig = defaultdict(list)
        for r in intermediate:
            ig[r.get('meta_log_id', 'unk')].append(r)
        ig_ids = sorted(ig.keys())
        random.shuffle(ig_ids)
        n_iv = max(1, len(ig_ids) // 10)
        iv_val   = set(ig_ids[:n_iv])
        iv_train = set(ig_ids[n_iv:])
        int_train = [r for gid in iv_train for r in ig[gid]]
        int_val   = [r for gid in iv_val   for r in ig[gid]]
        print(f"  intermediate split: train {len(int_train):,} / val {len(int_val):,}")
    else:
        int_train, int_val = [], []

    train_records = part1 + dl_train + int_train
    val_records   = dl_val + int_val
    test_records  = load_pkl(["dataset_phoenix_closed_precall_part3_test.pkl"])

    print(f"\n  学習: {len(train_records):,}  検証: {len(val_records):,}  テスト: {len(test_records):,}")

    train_loader = DataLoader(DiscardDataset(train_records), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(DiscardDataset(val_records),   batch_size=512,        shuffle=False, num_workers=0)
    test_loader  = DataLoader(DiscardDataset(test_records),  batch_size=512,        shuffle=False, num_workers=0)

    # 転移学習
    print("\n[2] 転移学習開始 (discard_b1_best.pth ベース)...")
    model = MahjongResNet_UltimateV3().to(device)
    base = r"G:\マイドライブ\MahjongAI\discard_b1_best.pth"
    if os.path.exists(base):
        model.load_state_dict(torch.load(base, map_location=device))
        print(f"  {base} から初期化")
    else:
        print("  ランダム初期化")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_val_top1 = 0.0

    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainTop1':>10} {'ValLoss':>9} {'ValTop1':>9} {'ValTop3':>9}")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)
        scheduler.step()

        val = evaluate_model(model, val_loader, device)
        mark = ""
        if val["top1"] > best_val_top1:
            best_val_top1 = val["top1"]
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            mark = " [best]"
        print(f"{epoch:>5} {total_loss/total:>10.4f} {correct/total*100:>9.2f}%"
              f" {val['loss']:>9.4f} {val['top1']:>8.2f}% {val['top3']:>8.2f}%{mark}", flush=True)

    # 最終テスト評価
    print("\n[3] テスト評価...")
    final = MahjongResNet_UltimateV3().to(device)
    final.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
    res = evaluate_model(final, test_loader, device)
    print(f"  top-1: {res['top1']:.2f}%  top-3: {res['top3']:.2f}%")
    print(f"\n  保存: {SAVE_MODEL_PATH}")
    return res['top1']


# ========================================================
# STEP 3: 10,000ゲームベンチマーク
# ========================================================
def step3_benchmark():
    print("\n" + "=" * 60)
    print("STEP 3: 10,000ゲームベンチマーク")
    print("=" * 60)
    print(f"  {BENCHMARK_GAMES} ゲーム実行中...\n", flush=True)

    result = subprocess.run(
        [sys.executable, "benchmark.py", "--games", str(BENCHMARK_GAMES)],
        cwd=os.getcwd(),
        capture_output=False,
    )
    if result.returncode != 0:
        print("[!] ベンチマーク失敗")
    else:
        print("\n  ベンチマーク完了")


# ========================================================
# STEP 4: 天鳳オンライン対局
# ========================================================
def step4_tenhou():
    print("\n" + "=" * 60)
    print("STEP 4: 天鳳オンライン対局")
    print("=" * 60)

    if not os.path.exists("tenhou_client.py"):
        print("  [!] tenhou_client.py が見つかりません。スキップします。")
        return

    for i in range(1, TENHOU_GAMES + 1):
        print(f"\n  [{i}/{TENHOU_GAMES}] 対局開始...", flush=True)
        result = subprocess.run(
            [sys.executable, "tenhou_client.py"],
            cwd=os.getcwd(),
            capture_output=False,
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"  [!] 対局 {i} 終了 (returncode={result.returncode})")
        else:
            print(f"  対局 {i} 完了")
        time.sleep(10)  # 対局間のインターバル


# ========================================================
# メイン
# ========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MahjongAI フルパイプライン実行")
    print("=" * 60)
    start_all = time.time()

    # STEP 1
    t1 = time.time()
    n_records = step1_extract_data()
    print(f"\n  [STEP 1 完了] {n_records:,} 件  ({(time.time()-t1)/60:.1f} 分)")

    # STEP 2
    t2 = time.time()
    top1 = step2_train()
    print(f"\n  [STEP 2 完了] top-1={top1:.2f}%  ({(time.time()-t2)/60:.1f} 分)")

    # STEP 3
    t3 = time.time()
    step3_benchmark()
    print(f"\n  [STEP 3 完了] ({(time.time()-t3)/60:.1f} 分)")

    # STEP 4
    t4 = time.time()
    step4_tenhou()
    print(f"\n  [STEP 4 完了] ({(time.time()-t4)/60:.1f} 分)")

    total = (time.time() - start_all) / 60
    print(f"\n{'=' * 60}")
    print(f"  全工程完了  総所要時間: {total:.1f} 分")
    print(f"{'=' * 60}")
