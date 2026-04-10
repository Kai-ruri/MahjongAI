import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

# =========================================
# 0. 環境準備とデータ読み込み
# =========================================
DATASET_FILE = "./dataset_oshibiki_phase7a.pkl"

if not os.path.exists(DATASET_FILE):
    print(f"❌ エラー: '{DATASET_FILE}' が見つかりません。")
    sys.exit()

with open(DATASET_FILE, 'rb') as f:
    all_records = pickle.load(f)

# 🌟 中間(1)を除外し、0(完全オリ)と2(無筋押し)で明確な2値分類タスクを作成
# 0 -> 0(オリ), 2 -> 1(押し) にマッピング
binary_records = [r for r in all_records if r["label_oshiki"] in [0, 2]]
y_true = np.array([1 if r["label_oshiki"] == 2 else 0 for r in binary_records])

print(f"📊 評価対象データ (純粋な押し/オリ): {len(binary_records)} 件")
print(f"  - 押し(1): {sum(y_true)} 件 ({sum(y_true)/len(y_true)*100:.1f}%)")
print(f"  - オリ(0): {len(y_true) - sum(y_true)} 件 ({(len(y_true) - sum(y_true))/len(y_true)*100:.1f}%)")

# =========================================
# 1. ルールベースの定義
# =========================================
# Base 1: 全オリ
pred_base1 = np.zeros_like(y_true)

# Base 2: テンパイ押し、それ以外オリ
pred_base2 = np.array([1 if r["shanten_before"] <= 0 else 0 for r in binary_records])

# Base 3: テンパイ押し ＋ 1向聴(親 or ドラ2以上)押し、それ以外オリ
pred_base3 = []
for r in binary_records:
    if r["shanten_before"] <= 0:
        pred_base3.append(1)
    elif r["shanten_before"] == 1 and (r["is_oya"] or r["dora_count"] >= 2):
        pred_base3.append(1)
    else:
        pred_base3.append(0)
pred_base3 = np.array(pred_base3)

# =========================================
# 2. 評価・出力エンジン
# =========================================
def evaluate(name, y_t, y_p, indices=None):
    if indices is not None:
        y_t = y_t[indices]
        y_p = y_p[indices]
    if len(y_t) == 0: return

    acc = accuracy_score(y_t, y_p)
    p = precision_score(y_t, y_p, zero_division=0)
    r = recall_score(y_t, y_p, zero_division=0)
    f1 = f1_score(y_t, y_p, zero_division=0)

    print(f"[{name}] (n={len(y_t)})")
    print(f"  Accuracy: {acc*100:.1f}% | Precision(押し): {p*100:.1f}% | Recall(押し): {r*100:.1f}% | F1(押し): {f1*100:.1f}%")

print("\n" + "="*70)
print("👑 フェーズ7B: 押し引き ルールベース検証結果")
print("="*70)
evaluate("Base 1 (全オリ)", y_true, pred_base1)
evaluate("Base 2 (テンパイ押し)", y_true, pred_base2)
evaluate("Base 3 (テンパイ押し + 1向聴の親/高打点は押し)", y_true, pred_base3)

# =========================================
# 3. 条件別（サブグループ）検証 (Base 3 を対象)
# =========================================
idx_tenpai = [i for i, r in enumerate(binary_records) if r["shanten_before"] <= 0]
idx_1shan = [i for i, r in enumerate(binary_records) if r["shanten_before"] == 1]
idx_2shan = [i for i, r in enumerate(binary_records) if r["shanten_before"] >= 2]

idx_early = [i for i, r in enumerate(binary_records) if r["turn"] <= 6]
idx_mid = [i for i, r in enumerate(binary_records) if 7 <= r["turn"] <= 11]
idx_late = [i for i, r in enumerate(binary_records) if r["turn"] >= 12]

idx_oya = [i for i, r in enumerate(binary_records) if r["is_oya"]]
idx_ko = [i for i, r in enumerate(binary_records) if not r["is_oya"]]

print("\n🔍 【条件別検証 (Base 3 のパフォーマンス)】")
print("--- シャンテン別 ---")
evaluate("テンパイ", y_true, pred_base3, idx_tenpai)
evaluate("1シャンテン", y_true, pred_base3, idx_1shan)
evaluate("2シャンテン以上", y_true, pred_base3, idx_2shan)

print("\n--- 巡目別 ---")
evaluate("序盤 (0-6巡)", y_true, pred_base3, idx_early)
evaluate("中盤 (7-11巡)", y_true, pred_base3, idx_mid)
evaluate("終盤 (12巡-)", y_true, pred_base3, idx_late)

print("\n--- 立場別 ---")
evaluate("親", y_true, pred_base3, idx_oya)
evaluate("子", y_true, pred_base3, idx_ko)
print("="*70)