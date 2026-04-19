import os
import pickle
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys

# 自作モジュール
from mahjong_engine import tile_names

# =========================================
# 0. 環境準備とデータ読み込み
# =========================================
DATASET_FILE = "./dataset_oshibiki_phase7a.pkl"

if not os.path.exists(DATASET_FILE):
    print(f"❌ エラー: '{DATASET_FILE}' が見つかりません。")
    sys.exit()

print(f"📂 押し引き抽出データを読み込んでいます...")
with open(DATASET_FILE, 'rb') as f:
    all_records = pickle.load(f)

# 中間(1)を除外し、0(完全オリ)と2(無筋押し)の2値分類(0と1)にマッピング
binary_records = [r for r in all_records if r["label_oshiki"] in [0, 2]]
print(f"📊 有効データ読み込み完了: {len(binary_records)} 件")

# =========================================
# 1. 特徴量構築 & Dataset
# =========================================
X_tensor, X_aux, y = [], [], []
for r in binary_records:
    X_tensor.append(r["tensor"])
    high_value_proxy = 1.0 if r["dora_count"] >= 2 else 0.0
    aux = np.array([
        float(r["shanten_before"]),
        1.0 if r["shanten_before"] <= 0 else 0.0,
        1.0 if r["shanten_before"] == 1 else 0.0,
        1.0 if r["is_maintained"] else 0.0,
        float(r["enemy_riichi_count"]),
        float(r["turn"]),
        1.0 if r["is_oya"] else 0.0,
        float(r["dora_count"]),
        high_value_proxy,
        float(r["my_rank"]),
        float(r["diff_top"]) / 10000.0,
        float(r["diff_last"]) / 10000.0,
        1.0 if r["ooras"] else 0.0
    ], dtype=np.float32)
    X_aux.append(aux)
    y.append(np.float32(1.0 if r["label_oshiki"] == 2 else 0.0))

X_t, X_a, y = np.array(X_tensor, dtype=np.float32), np.array(X_aux, dtype=np.float32), np.array(y).reshape(-1, 1)
indices = np.arange(len(y))

# 前回と同じシード(42)で分割
X_t_tr, X_t_val, X_a_tr, X_a_val, y_tr, y_val, idx_tr, idx_val = train_test_split(X_t, X_a, y, indices, test_size=0.2, random_state=42, stratify=y)

class OshibikiCNNDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

train_loader = DataLoader(OshibikiCNNDataset(X_t_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
val_loader = DataLoader(OshibikiCNNDataset(X_t_val, X_a_val, y_val), batch_size=256, shuffle=False)

# =========================================
# 2. CNN-A 高速再学習
# =========================================
class OshibikiCNN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 13, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OshibikiCNN_A().to(device)

pos_weight = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n🚀 モデルを高速再学習中... (数分お待ちください)")
best_loss, best_wts, patience_cnt = float('inf'), None, 0
for epoch in range(25):
    model.train()
    for t, a, l in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(t.to(device), a.to(device)), l.to(device))
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = sum(criterion(model(t.to(device), a.to(device)), l.to(device)).item() * t.size(0) for t, a, l in val_loader) / len(y_val)
    if val_loss < best_loss: best_loss, best_wts, patience_cnt = val_loss, copy.deepcopy(model.state_dict()), 0
    elif (patience_cnt := patience_cnt + 1) >= 4: break

model.load_state_dict(best_wts)

# =========================================
# 3. 予測と最適閾値の計算
# =========================================
model.eval()
with torch.no_grad():
    cnn_probs = torch.sigmoid(model(torch.tensor(X_t_val).to(device), torch.tensor(X_a_val).to(device))).cpu().numpy().flatten()

best_f1, best_thresh = 0.0, 0.5
for thresh in np.arange(0.1, 0.9, 0.02):
    f1 = f1_score(y_val, (cnn_probs >= thresh).astype(int), zero_division=0)
    if f1 > best_f1: best_f1, best_thresh = f1, thresh

# =========================================
# 4. エラー出力ヘルパー
# =========================================
def format_case(r):
    rec = r["rec"]
    prob = r["prob"]
    label = r["label"]
    
    # テンソルから14枚の手牌を復元
    counts = [int(round(rec["tensor"][0][i] * 4.0)) for i in range(34)]
    hand_str = "".join([f"[{tile_names[i]}]" * c for i, c in enumerate(counts) if c > 0])
    
    is_oya = "親" if rec["is_oya"] else "子"
    maintain_str = "〇" if rec["is_maintained"] else "× (手戻り)"
    pred_str = "押し" if prob >= best_thresh else "オリ"
    label_str = "押し(無筋)" if label == 1 else "オリ(現物)"
    
    return f"""  🀄 打牌: [{tile_names[rec["discard_34"]]}] | 巡目: {rec["turn"]}巡 | 順位: {int(rec["my_rank"])}位(トップ差 {int(rec["diff_top"])}) | {is_oya}
  手牌: {hand_str}
  向聴: {rec["shanten_before"]} (打牌後の形維持: {maintain_str}) | リーチ者: {rec["enemy_riichi_count"]}人 | ドラ: {int(rec["dora_count"])}枚
  予測: {pred_str} (確率: {prob*100:.1f}%) | 正解(人間): {label_str}"""

results = []
for i, orig_idx in enumerate(idx_val):
    results.append({"prob": cnn_probs[i], "label": int(y_val[i][0]), "rec": binary_records[orig_idx]})

# =========================================
# 5. 指定されたパターンの抽出と出力
# =========================================
print("\n" + "="*70)
print("🔍 押し引き CNN-A エラー分析レポート")
print(f"   (F1最適閾値: {best_thresh:.2f})")
print("="*70)

# 【1】FP: AI押し / 人間オリ (AIが危険を軽視、または価値を過大評価)
fp_cases = [r for r in results if r["label"] == 0 and r["prob"] >= best_thresh]

print("\n🚨 【FP: AI 押し / 人間 オリ】 (AIが押しすぎた / 人間は安全牌を優先・局収支引き)")
print("  --- 高信頼度で誤った例 (AIは絶対に押すべきだと思った) ---")
for r in sorted(fp_cases, key=lambda x: x["prob"], reverse=True)[:3]:
    print(format_case(r))
    print("-" * 50)
    
print("  --- 境界付近で迷った例 ---")
for r in sorted(fp_cases, key=lambda x: abs(x["prob"] - best_thresh))[:3]:
    print(format_case(r))
    print("-" * 50)

# 【2】FN: AIオリ / 人間押し (AIが価値を軽視、または危険を過大評価)
fn_cases = [r for r in results if r["label"] == 1 and r["prob"] < best_thresh]

print("\n🚨 【FN: AI オリ / 人間 押し】 (AIが慎重すぎた / 人間は打点・好形・親番で勝負した)")
print("  --- 高信頼度で誤った例 (AIは絶対に降りるべきだと思った) ---")
for r in sorted(fn_cases, key=lambda x: x["prob"])[:3]:
    print(format_case(r))
    print("-" * 50)
    
print("  --- 境界付近で迷った例 ---")
for r in sorted(fn_cases, key=lambda x: abs(x["prob"] - best_thresh))[:3]:
    print(format_case(r))
    print("-" * 50)

print("="*70)