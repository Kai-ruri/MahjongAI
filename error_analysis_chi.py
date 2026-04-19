import os
import pickle
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import sys

# 自作モジュール
from mahjong_engine import tile_names, calculate_shanten

# =========================================
# 0. 環境準備とデータ読み込み (抽出スキップ！)
# =========================================
DATASET_FILE = "./dataset_chi_phase5b_large.pkl"

if not os.path.exists(DATASET_FILE):
    print(f"❌ エラー: '{DATASET_FILE}' が見つかりません。")
    sys.exit()

print(f"📂 保存された大規模データセットを読み込んでいます...")
with open(DATASET_FILE, 'rb') as f:
    all_records = pickle.load(f)
print(f"📊 データ読み込み完了: {len(all_records)} 件")

# =========================================
# 1. 特徴量構築 & Dataset (前回と完全同一)
# =========================================
X_tensor, X_aux, y = [], [], []
for r in all_records:
    X_tensor.append(r["tensor"])
    target_onehot = np.zeros(34, dtype=np.float32)
    target_onehot[r["target_tile_34"]] = 1.0
    aux = np.concatenate([
        target_onehot,
        np.array(r["chi_mask"], dtype=np.float32), 
        np.array([
            1.0 if r["meta_is_dora"] else 0.0, float(r["meta_my_shanten"]), 1.0 if r["meta_is_improved"] else 0.0,
            1.0 if r["meta_is_ryoukei"] else 0.0, 1.0 if r["meta_is_gukei"] else 0.0, float(r["meta_freedom"]),
            1.0 if r["meta_is_defense"] else 0.0
        ], dtype=np.float32)
    ])
    X_aux.append(aux)
    y.append(np.float32(r["label_chi"]))

X_t, X_a, y = np.array(X_tensor, dtype=np.float32), np.array(X_aux, dtype=np.float32), np.array(y).reshape(-1, 1)
indices = np.arange(len(y))
# 乱数シード(42)を固定しているため、前回と全く同じValidationデータに分かれます
X_t_tr, X_t_val, X_a_tr, X_a_val, y_tr, y_val, idx_tr, idx_val = train_test_split(X_t, X_a, y, indices, test_size=0.2, random_state=42, stratify=y)

class ChiCNNDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

train_loader = DataLoader(ChiCNNDataset(X_t_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
val_loader = DataLoader(ChiCNNDataset(X_t_val, X_a_val, y_val), batch_size=256, shuffle=False)

# =========================================
# 2. CNN-A 再学習 (高速)
# =========================================
class ChiCNN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 44, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChiCNN_A().to(device)

pos_weight = float((len(y_tr)-y_tr.sum())/y_tr.sum())
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
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
# 3. 評価と抽出準備
# =========================================
model.eval()
with torch.no_grad():
    cnn_probs = torch.sigmoid(model(torch.tensor(X_t_val).to(device), torch.tensor(X_a_val).to(device))).cpu().numpy().flatten()

precisions, recalls, thresholds = precision_recall_curve(y_val, cnn_probs)
f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
best_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5

results = []
for i, orig_idx in enumerate(idx_val):
    results.append({"prob": cnn_probs[i], "label": int(y_val[i][0]), "rec": all_records[orig_idx]})

# 抽出用ヘルパー関数
def get_s_after(counts, target):
    temp = list(counts)
    temp[target] += 1
    best_s = 99
    for i in range(34):
        if temp[i] > 0 and i != target:
            temp[i] -= 1
            s = calculate_shanten(temp)
            if s < best_s: best_s = s
            temp[i] += 1
    return best_s

def format_case(r):
    rec = r["rec"]
    prob = r["prob"]
    label = r["label"]
    
    hand_counts = [int(round(rec["tensor"][0][i] * 4.0)) for i in range(34)]
    hand_str = "".join([f"[{tile_names[i]}]" * count for i, count in enumerate(hand_counts) if count > 0])
    
    chi_cands = []
    if rec["chi_mask"][0]: chi_cands.append("左")
    if rec["chi_mask"][1]: chi_cands.append("中")
    if rec["chi_mask"][2]: chi_cands.append("右")
    
    s_before = rec["meta_my_shanten"]
    s_after = get_s_after(hand_counts, rec["target_tile_34"])
    
    return f"""  🀄 対象牌: [{tile_names[rec["target_tile_34"]]}] | 巡目: {rec["meta_turn"]}巡 | 候補: {"/".join(chi_cands)}
  手牌: {hand_str}
  予測: {"CHI" if prob >= best_thresh else "PASS"} (確率: {prob*100:.1f}%) | 正解: {"CHI" if label == 1 else "PASS"}
  ｼｬﾝﾃﾝ: {s_before} -> {s_after} | ドラ: {"〇" if rec["meta_is_dora"] else "×"} | 良形化: {"〇" if rec["meta_is_ryoukei"] else "×"} | 打牌自由度: {int(rec["meta_freedom"])}"""

def print_category(cases, reverse_sort, name):
    print(f"\n✨ 【{name}】")
    print("  --- 高信頼度 (AIの強い確信) ---")
    for r in sorted(cases, key=lambda x: x["prob"], reverse=reverse_sort)[:3]:
        print(format_case(r))
        print("-" * 40)
    print("  --- 境界付近 (AIの迷い) ---")
    for r in sorted(cases, key=lambda x: abs(x["prob"] - best_thresh))[:3]:
        print(format_case(r))
        print("-" * 40)

# =========================================
# 4. 指定された4パターンの抽出と出力
# =========================================
print("\n" + "="*60)
print("🔍 CNN-A モデル チー判断 エラー分析レポート")
print(f"   (全体最適閾値: {best_thresh:.3f})")
print("="*60)

# 1. 非ドラTP (正解:CHI, 予測:CHI) -> 高い順
tp_cases = [r for r in results if not r["rec"]["meta_is_dora"] and r["label"]==1 and r["prob"]>=best_thresh]
print_category(tp_cases, reverse_sort=True, name="1. 非ドラTP (AIが見抜いた『鳴いても苦しくない』チー)")

# 2. 非ドラFP (正解:PASS, 予測:CHI) -> 高い順
fp_cases = [r for r in results if not r["rec"]["meta_is_dora"] and r["label"]==0 and r["prob"]>=best_thresh]
print_category(fp_cases, reverse_sort=True, name="2. 非ドラFP (AIが飛びついた『無駄鳴き候補』)")

# 3. ドラ絡みFN (正解:CHI, 予測:PASS) -> 低い順
fn_dora_cases = [r for r in results if r["rec"]["meta_is_dora"] and r["label"]==1 and r["prob"]<best_thresh]
print_category(fn_dora_cases, reverse_sort=False, name="3. ドラ絡みFN (勝負手なのに守りすぎた/過小評価した例)")

# 4. 1シャンテンFN (正解:CHI, 予測:PASS) -> 低い順
fn_s1_cases = [r for r in results if r["rec"]["meta_my_shanten"]==1 and r["label"]==1 and r["prob"]<best_thresh]
print_category(fn_s1_cases, reverse_sort=False, name="4. 1シャンテンFN (あと一歩でテンパイなのに踏み込めなかった例)")

print("="*60)