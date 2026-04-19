import os
import pickle
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

# =========================================
# 0. 環境準備とデータ読み込み
# =========================================
DATASET_FILE = "./dataset_oshibiki_phase7a.pkl"

if not os.path.exists(DATASET_FILE):
    print(f"❌ エラー: '{DATASET_FILE}' が見つかりません。")
    sys.exit()

print(f"📂 押し引きデータを読み込んでいます...")
with open(DATASET_FILE, 'rb') as f:
    all_records = pickle.load(f)

# 中間(1)を除外し、0(完全オリ)と2(無筋押し)の2値分類(0と1)にマッピング
binary_records = [r for r in all_records if r["label_oshiki"] in [0, 2]]
print(f"📊 評価対象データ (純粋な押し/オリ): {len(binary_records)} 件")

# =========================================
# 1. 特徴量構築 & Dataset
# =========================================
X_tensor, X_aux, y = [], [], []
for r in binary_records:
    X_tensor.append(r["tensor"])
    
    # Base3相当の高打点proxy
    high_value_proxy = 1.0 if r["dora_count"] >= 2 else 0.0
    
    # 🌟 レビューアー指示の価値特徴 + 守備特徴 (13次元)
    aux = np.array([
        float(r["shanten_before"]),               # シャンテン数
        1.0 if r["shanten_before"] <= 0 else 0.0, # テンパイフラグ
        1.0 if r["shanten_before"] == 1 else 0.0, # 1シャンテンフラグ
        1.0 if r["is_maintained"] else 0.0,       # 打牌後テンパイ維持か(シャンテン戻しなし)
        float(r["enemy_riichi_count"]),           # 他家リーチ人数
        float(r["turn"]),                         # 巡目
        1.0 if r["is_oya"] else 0.0,              # 親フラグ
        float(r["dora_count"]),                   # ドラ枚数
        high_value_proxy,                         # 高打点proxy
        float(r["my_rank"]),                      # 順位
        float(r["diff_top"]) / 10000.0,           # トップとの点差(正規化)
        float(r["diff_last"]) / 10000.0,          # ラスとの点差(正規化)
        1.0 if r["ooras"] else 0.0                # オーラスフラグ
    ], dtype=np.float32)
    
    X_aux.append(aux)
    # ラベル: 2(無筋押し) -> 1(Push), 0(現物オリ) -> 0(Fold)
    y.append(np.float32(1.0 if r["label_oshiki"] == 2 else 0.0))

X_t, X_a, y = np.array(X_tensor, dtype=np.float32), np.array(X_aux, dtype=np.float32), np.array(y).reshape(-1, 1)
indices = np.arange(len(y))

# 不均衡データなので stratify で分割
X_t_tr, X_t_val, X_a_tr, X_a_val, y_tr, y_val, idx_tr, idx_val = train_test_split(X_t, X_a, y, indices, test_size=0.2, random_state=42, stratify=y)
meta_val = [binary_records[i] for i in idx_val]

class OshibikiCNNDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

train_loader = DataLoader(OshibikiCNNDataset(X_t_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
val_loader = DataLoader(OshibikiCNNDataset(X_t_val, X_a_val, y_val), batch_size=256, shuffle=False)

# =========================================
# 2. CNN-A モデル定義と本番学習
# =========================================
class OshibikiCNN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 + 13, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, t, a):
        return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OshibikiCNN_A().to(device)

# 押し(1)の割合が少ないため、pos_weightで重み付け
pos_weight = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n🚀 【フェーズ7C】押し引き CNN-A 学習開始！ (Train: {len(y_tr)}件, Valid: {len(y_val)}件)")
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
    print(f"Epoch {epoch+1:02d} | Valid Loss: {val_loss:.4f}")
    if val_loss < best_loss: 
        best_loss, best_wts, patience_cnt = val_loss, copy.deepcopy(model.state_dict()), 0
    elif (patience_cnt := patience_cnt + 1) >= 5: 
        print("🛑 Early stopping triggered.")
        break

model.load_state_dict(best_wts)

# =========================================
# 3. 評価と Base3 との比較
# =========================================
model.eval()
with torch.no_grad():
    cnn_probs = torch.sigmoid(model(torch.tensor(X_t_val).to(device), torch.tensor(X_a_val).to(device))).cpu().numpy().flatten()

# 最適閾値の探索 (F1スコア最大化)
best_f1, best_thresh = 0.0, 0.5
for thresh in np.arange(0.1, 0.9, 0.02):
    f1 = f1_score(y_val, (cnn_probs >= thresh).astype(int), zero_division=0)
    if f1 > best_f1: best_f1, best_thresh = f1, thresh

pred_cnn = (cnn_probs >= best_thresh).astype(int)

# Base 3 (ルールベース) の予測を Valid データに対して作成
pred_base3 = []
for r in meta_val:
    if r["shanten_before"] <= 0: pred_base3.append(1)
    elif r["shanten_before"] == 1 and (r["is_oya"] or r["dora_count"] >= 2): pred_base3.append(1)
    else: pred_base3.append(0)
pred_base3 = np.array(pred_base3)

def get_metrics(y_t, y_p):
    if len(y_t) == 0: return 0.0, 0.0, 0.0, 0.0
    return (accuracy_score(y_t, y_p)*100, precision_score(y_t, y_p, zero_division=0)*100, 
            recall_score(y_t, y_p, zero_division=0)*100, f1_score(y_t, y_p, zero_division=0)*100)

acc_c, p_c, r_c, f1_c = get_metrics(y_val, pred_cnn)
acc_b, p_b, r_b, f1_b = get_metrics(y_val, pred_base3)

print("\n" + "="*70)
print("👑 フェーズ7C: 押し引き CNN-A 結果発表")
print("="*70)
print(f"■ 【全体指標】 最適閾値: {best_thresh:.2f}")
print(f"  [CNN-A] Acc: {acc_c:.1f}% | Precision: {p_c:.1f}% | Recall: {r_c:.1f}% | F1: {f1_c:.1f}%  <-- 🎯 目標: 32.5%超え！")
print(f"  [Base 3] Acc: {acc_b:.1f}% | Precision: {p_b:.1f}% | Recall: {r_b:.1f}% | F1: {f1_b:.1f}%")

# 条件別評価用ヘルパー
def eval_cond(name, condition_fn):
    idx = [i for i, m in enumerate(meta_val) if condition_fn(m)]
    if not idx: return
    y_t_sub, p_c_sub, p_b_sub = y_val[idx], pred_cnn[idx], pred_base3[idx]
    _, _, _, f1_cnn = get_metrics(y_t_sub, p_c_sub)
    _, _, _, f1_base = get_metrics(y_t_sub, p_b_sub)
    diff = f1_cnn - f1_base
    sign = "+" if diff >= 0 else ""
    print(f"  [{name}] (n={len(idx)})")
    print(f"    CNN F1: {f1_cnn:.1f}% | Base3 F1: {f1_base:.1f}% (差: {sign}{diff:.1f}%)")

print("\n🔍 【条件別 F1スコア 比較 (CNN vs Base3)】")
eval_cond("1シャンテン", lambda m: m["shanten_before"] == 1)
eval_cond("2シャンテン以上", lambda m: m["shanten_before"] >= 2)
eval_cond("親", lambda m: m["is_oya"])
eval_cond("終盤 (12巡-)", lambda m: m["turn"] >= 12)
eval_cond("他家リーチ複数", lambda m: m["enemy_riichi_count"] >= 2)
print("="*70)