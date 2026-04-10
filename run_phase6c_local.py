import os
import pickle
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import torch.nn.functional as F

class BinaryFocalLossWithWeight(nn.Module):
    def __init__(self, pass_weight=1.5, gamma=2.0):
        super().__init__()
        self.pass_weight = pass_weight 
        self.call_weight = 1.0         
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = torch.where(targets == 0, self.pass_weight, self.call_weight)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()

# =========================================
# 0. 環境準備とデータ読み込み
# =========================================
DATASET_FILE = "./dataset_riichi_phase6a.pkl"

if not os.path.exists(DATASET_FILE):
    print(f"❌ エラー: '{DATASET_FILE}' が見つかりません。先にフェーズ6Aを実行してください。")
    sys.exit()

print(f"📂 保存されたリーチ抽出データを読み込んでいます...")
with open(DATASET_FILE, 'rb') as f:
    all_records = pickle.load(f)

# オリ・テンパイ崩しを除外した有効局面のみを抽出
valid_records = [r for r in all_records if r["wait_count"] > 0]
print(f"📊 有効データ読み込み完了: {len(valid_records)} 件")

# =========================================
# 1. 特徴量構築 & Dataset
# =========================================
X_tensor, X_aux, y, meta_val = [], [], [], []
for r in valid_records:
    X_tensor.append(r["tensor"])
    
    # 🌟 レビューアー指示の最小補助特徴量セット (9次元)
    # 高打点ダマproxy: ドラ2以上かつダマで和了可能
    high_score_dama = 1.0 if (r["meta_dora_count"] >= 2 and r["meta_dama_legal"]) else 0.0
    
    aux = np.array([
        float(r["wait_count"]),                # 待ち枚数
        1.0 if r["is_ryankei"] else 0.0,       # 良形フラグ
        float(r["meta_turn"]),                 # 巡目
        1.0 if r["meta_enemy_riichi"] else 0.0,# 他家リーチ有無
        float(r["meta_dora_count"]),           # ドラ枚数
        high_score_dama,                       # 高打点ダマproxy
        1.0 if r["meta_dama_legal"] else 0.0,  # ダマ合法フラグ(役あり)
        float(r["meta_my_rank"]),              # 順位
        float(r["meta_point_diff"]) / 10000.0  # トップとの点差(スケール調整)
    ], dtype=np.float32)
    
    X_aux.append(aux)
    y.append(np.float32(r["label_riichi"]))

X_t, X_a, y = np.array(X_tensor, dtype=np.float32), np.array(X_aux, dtype=np.float32), np.array(y).reshape(-1, 1)
indices = np.arange(len(y))

# 訓練:テスト = 80:20
X_t_tr, X_t_val, X_a_tr, X_a_val, y_tr, y_val, idx_tr, idx_val = train_test_split(X_t, X_a, y, indices, test_size=0.2, random_state=42, stratify=y)
meta_val = [valid_records[i] for i in idx_val]

class RiichiCNNDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

train_loader = DataLoader(RiichiCNNDataset(X_t_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
val_loader = DataLoader(RiichiCNNDataset(X_t_val, X_a_val, y_val), batch_size=256, shuffle=False)

# =========================================
# 2. CNN-A モデル定義と本番学習
# =========================================
class RiichiCNN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # 入力: 空間64次元 + 補助特徴9次元 = 73次元
        self.mlp = nn.Sequential(
            nn.Linear(64 + 9, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, t, a):
        return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RiichiCNN_A().to(device)

# リーチの割合は約40%なので、今回はpos_weightの強い補正は不要（自然なクロスエントロピーで学習）
criterion = nn.BCEWithLogitsLoss() # ←これをコメントアウトまたは削除
# criterion = BinaryFocalLossWithWeight(pass_weight=1.5, gamma=2.0) # ←これに差し替え！
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n🚀 【フェーズ6C】リーチ判断 CNN-A 学習開始！ (Train: {len(y_tr)}件, Valid: {len(y_val)}件)")
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
# 3. 評価と条件別（サブグループ）分析
# =========================================
model.eval()
with torch.no_grad():
    cnn_probs = torch.sigmoid(model(torch.tensor(X_t_val).to(device), torch.tensor(X_a_val).to(device))).cpu().numpy().flatten()

# 最適閾値の探索（今回はAccuracyが最大になる閾値を探す）
best_acc, best_thresh = 0.0, 0.5
for thresh in np.arange(0.1, 0.9, 0.02):
    acc = accuracy_score(y_val, (cnn_probs >= thresh).astype(int))
    if acc > best_acc: best_acc, best_thresh = acc, thresh

pred_opt = (cnn_probs >= best_thresh).astype(int)

roc_auc = roc_auc_score(y_val, cnn_probs)
p_all = precision_score(y_val, pred_opt, zero_division=0)
r_all = recall_score(y_val, pred_opt, zero_division=0)
f1_all = f1_score(y_val, pred_opt, zero_division=0)

print("\n" + "="*60)
print("👑 フェーズ6C: リーチ判断 CNN-A 結果発表")
print("="*60)
print(f"■ 【全体指標】 ROC-AUC: {roc_auc:.4f}")
print(f"■ 最適閾値: {best_thresh:.2f} (全体Accuracy: {best_acc*100:.2f}%)  <-- 🎯 目標: 60.8%超え！")
print(f"  [Precision: {p_all*100:.1f}% | Recall: {r_all*100:.1f}% | F1: {f1_all*100:.1f}%]")

# 条件別（サブグループ）の評価用ヘルパー
def evaluate_condition(name, condition_idx, y_true, y_pred, meta_list):
    if len(condition_idx) == 0: return
    y_t_sub = y_true[condition_idx]
    y_p_sub = y_pred[condition_idx]
    acc = accuracy_score(y_t_sub, y_p_sub)
    
    # 同一データに対する Base 2 (良形ならリーチ) の正解率も計算して比較する
    base2_correct = sum(1 for i in condition_idx if (meta_list[i]["is_ryankei"] and y_true[i]==1) or (not meta_list[i]["is_ryankei"] and y_true[i]==0))
    base2_acc = base2_correct / len(condition_idx)
    
    diff = acc - base2_acc
    sign = "+" if diff >= 0 else ""
    print(f"  [{name}] (n={len(condition_idx)})")
    print(f"    CNN: {acc*100:.1f}% | Base2: {base2_acc*100:.1f}% (差: {sign}{diff*100:.1f}%)")

print("\n🔍 【条件別 Accuracy 比較 (CNN vs Base2)】")

# 1. 待ちの質 (良形 / 愚形)
idx_ryo = np.array([i for i, m in enumerate(meta_val) if m["is_ryankei"]])
idx_gu = np.array([i for i, m in enumerate(meta_val) if not m["is_ryankei"]])
evaluate_condition("良形テンパイ", idx_ryo, y_val, pred_opt, meta_val)
evaluate_condition("愚形テンパイ", idx_gu, y_val, pred_opt, meta_val)

# 2. 他家リーチ (あり / なし)
idx_enemy_yes = np.array([i for i, m in enumerate(meta_val) if m["meta_enemy_riichi"]])
idx_enemy_no = np.array([i for i, m in enumerate(meta_val) if not m["meta_enemy_riichi"]])
evaluate_condition("他家リーチ【あり】", idx_enemy_yes, y_val, pred_opt, meta_val)
evaluate_condition("他家リーチ【なし】", idx_enemy_no, y_val, pred_opt, meta_val)

# 3. 高打点ダマ候補
idx_high_dama = np.array([i for i, m in enumerate(meta_val) if m["meta_dora_count"] >= 2 and m["meta_dama_legal"]])
evaluate_condition("高打点ダマ候補(ドラ2以上役あり)", idx_high_dama, y_val, pred_opt, meta_val)

# 4. 巡目 (序盤0-6, 中盤7-11, 終盤12+)
idx_early = np.array([i for i, m in enumerate(meta_val) if m["meta_turn"] <= 6])
idx_mid = np.array([i for i, m in enumerate(meta_val) if 7 <= m["meta_turn"] <= 11])
idx_late = np.array([i for i, m in enumerate(meta_val) if m["meta_turn"] >= 12])
evaluate_condition("序盤 (0-6巡)", idx_early, y_val, pred_opt, meta_val)
evaluate_condition("中盤 (7-11巡)", idx_mid, y_val, pred_opt, meta_val)
evaluate_condition("終盤 (12巡-)", idx_late, y_val, pred_opt, meta_val)

print("="*60)