import os
import pickle
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# 自作モジュール
from mahjong_engine import tile_names, calculate_shanten

# =========================================
# 0. 環境準備とデータ読み込み
# =========================================
DATASET_FILE = "./dataset_riichi_phase6a.pkl"

if not os.path.exists(DATASET_FILE):
    print(f"❌ エラー: '{DATASET_FILE}' が見つかりません。")
    sys.exit()

print(f"📂 リーチ抽出データを読み込んでいます...")
with open(DATASET_FILE, 'rb') as f:
    all_records = pickle.load(f)

valid_records = [r for r in all_records if r["wait_count"] > 0]
print(f"📊 有効データ読み込み完了: {len(valid_records)} 件")

# =========================================
# 1. 特徴量構築 & Dataset
# =========================================
X_tensor, X_aux, y = [], [], []
for r in valid_records:
    X_tensor.append(r["tensor"])
    high_score_dama = 1.0 if (r["meta_dora_count"] >= 2 and r["meta_dama_legal"]) else 0.0
    aux = np.array([
        float(r["wait_count"]),
        1.0 if r["is_ryankei"] else 0.0,
        float(r["meta_turn"]),
        1.0 if r["meta_enemy_riichi"] else 0.0,
        float(r["meta_dora_count"]),
        high_score_dama,
        1.0 if r["meta_dama_legal"] else 0.0,
        float(r["meta_my_rank"]),
        float(r["meta_point_diff"]) / 10000.0
    ], dtype=np.float32)
    X_aux.append(aux)
    y.append(np.float32(r["label_riichi"]))

X_t, X_a, y = np.array(X_tensor, dtype=np.float32), np.array(X_aux, dtype=np.float32), np.array(y).reshape(-1, 1)
indices = np.arange(len(y))

# 前回と同じシード(42)で分割
X_t_tr, X_t_val, X_a_tr, X_a_val, y_tr, y_val, idx_tr, idx_val = train_test_split(X_t, X_a, y, indices, test_size=0.2, random_state=42, stratify=y)

class RiichiCNNDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

train_loader = DataLoader(RiichiCNNDataset(X_t_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
val_loader = DataLoader(RiichiCNNDataset(X_t_val, X_a_val, y_val), batch_size=256, shuffle=False)

# =========================================
# 2. CNN-A 高速再学習
# =========================================
class RiichiCNN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 9, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RiichiCNN_A().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n🚀 モデルを高速再学習中... (数分お待ちください)")
best_loss, best_wts, patience_cnt = float('inf'), None, 0
for epoch in range(30):
    model.train()
    for t, a, l in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(t.to(device), a.to(device)), l.to(device))
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = sum(criterion(model(t.to(device), a.to(device)), l.to(device)).item() * t.size(0) for t, a, l in val_loader) / len(y_val)
    if val_loss < best_loss: best_loss, best_wts, patience_cnt = val_loss, copy.deepcopy(model.state_dict()), 0
    elif (patience_cnt := patience_cnt + 1) >= 5: break

model.load_state_dict(best_wts)

# =========================================
# 3. 予測と最適閾値の計算
# =========================================
model.eval()
with torch.no_grad():
    cnn_probs = torch.sigmoid(model(torch.tensor(X_t_val).to(device), torch.tensor(X_a_val).to(device))).cpu().numpy().flatten()

best_acc, best_thresh = 0.0, 0.5
for thresh in np.arange(0.1, 0.9, 0.02):
    acc = accuracy_score(y_val, (cnn_probs >= thresh).astype(int))
    if acc > best_acc: best_acc, best_thresh = acc, thresh

# =========================================
# 4. エラー出力ヘルパー（🌟 待ち牌の復元ロジック追加！）
# =========================================
_shanten_cache = {}
def get_shanten_cached(counts):
    t = tuple(counts)
    if t not in _shanten_cache:
        _shanten_cache[t] = calculate_shanten(list(counts))
    return _shanten_cache[t]

def get_waits(hand_counts):
    waits = []
    for i in range(34):
        if hand_counts[i] < 4:
            hand_counts[i] += 1
            if get_shanten_cached(hand_counts) == -1:
                waits.append(i)
            hand_counts[i] -= 1
    return waits

def get_wait_type_str(waits):
    if len(waits) >= 3: return "多面張"
    elif len(waits) == 2:
        w1, w2 = sorted(waits)
        if w1 < 27 and w2 < 27 and w1 // 9 == w2 // 9 and w2 - w1 == 3: return "両面"
        return "シャンポン等(2種)"
    return "愚形(カンチャン/ペンチャン/単騎)"

def format_case(r):
    rec = r["rec"]
    prob = r["prob"]
    label = r["label"]
    
    # テンソルから14枚の手牌を復元
    counts = [int(round(rec["tensor"][0][i] * 4.0)) for i in range(34)]
    hand_str = "".join([f"[{tile_names[i]}]" * c for i, c in enumerate(counts) if c > 0])
    
    # 🌟 ここで「切った後の13枚」の形にして待ちを再計算！
    counts_13 = list(counts)
    counts_13[rec["actual_discard_34"]] -= 1
    waits_list = get_waits(counts_13)
    
    # メタ情報の整形
    waits_str = " ".join([f"[{tile_names[w]}]" for w in waits_list])
    wait_type = get_wait_type_str(waits_list)
    is_oya = "親" if rec["tensor"][20][0] == 1.0 else "子" # テンソルの自風ch(東=0)で判定
    high_dama = "〇" if rec["meta_dora_count"] >= 2 and rec["meta_dama_legal"] else "×"
    enemy_r = "あり" if rec["meta_enemy_riichi"] else "なし"
    pred_str = "リーチ" if prob >= best_thresh else "ダマ"
    label_str = "リーチ" if label == 1 else "ダマ"
    
    return f"""  🀄 打牌: [{tile_names[rec["actual_discard_34"]]}] | 巡目: {rec["meta_turn"]}巡 | 順位: {int(rec["meta_my_rank"])}位(トップ差 {int(rec["meta_point_diff"])}) | {is_oya}
  手牌: {hand_str}
  待ち: {waits_str} ({wait_type} / {rec["wait_count"]}枚)
  予測: {pred_str} (確率: {prob*100:.1f}%) | 正解(人間): {label_str}
  ドラ: {int(rec["meta_dora_count"])}枚 | 高打点ダマ候補: {high_dama} | 他家リーチ: {enemy_r}"""

results = []
for i, orig_idx in enumerate(idx_val):
    results.append({"prob": cnn_probs[i], "label": int(y_val[i][0]), "rec": valid_records[orig_idx]})

# =========================================
# 5. 指定された4パターンの抽出と出力
# =========================================
print("\n" + "="*70)
print("🔍 リーチ判断 CNN-A エラー分析レポート")
print(f"   (全体最適閾値: {best_thresh:.2f})")
print("="*70)

# 【1】FN: AIダマ / 人間リーチ (AIが慎重すぎた例)
fn_cases = [r for r in results if r["label"] == 1 and r["prob"] < best_thresh]

print("\n🚨 【FN: AIダマ / 人間リーチ】 (AIが慎重すぎた / 愚形即リーや打点上昇狙い)")
print("  --- 高信頼度で誤った例 (AIは絶対にダマだと思った) ---")
for r in sorted(fn_cases, key=lambda x: x["prob"])[:3]: # 確率が極端に低い順
    print(format_case(r))
    print("-" * 50)
    
print("  --- 境界付近で迷った例 ---")
for r in sorted(fn_cases, key=lambda x: abs(x["prob"] - best_thresh))[:3]:
    print(format_case(r))
    print("-" * 50)

# 【2】FP: AIリーチ / 人間ダマ (AIが押しすぎた例)
fp_cases = [r for r in results if r["label"] == 0 and r["prob"] >= best_thresh]

print("\n🚨 【FP: AIリーチ / 人間ダマ】 (AIが押しすぎた / 変化待ちや高打点・守備ダマ)")
print("  --- 高信頼度で誤った例 (AIは絶対にリーチだと思った) ---")
for r in sorted(fp_cases, key=lambda x: x["prob"], reverse=True)[:3]: # 確率が極端に高い順
    print(format_case(r))
    print("-" * 50)
    
print("  --- 境界付近で迷った例 ---")
for r in sorted(fp_cases, key=lambda x: abs(x["prob"] - best_thresh))[:3]:
    print(format_case(r))
    print("-" * 50)

print("="*70)