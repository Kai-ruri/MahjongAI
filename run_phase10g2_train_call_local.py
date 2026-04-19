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

# =========================================
# 🌟 0. 鳴き暴走を止める最強のLoss関数
# =========================================
class BinaryFocalLossWithWeight(nn.Module):
    def __init__(self, pass_weight=1.5, gamma=2.0):
        super().__init__()
        self.pass_weight = pass_weight # Pass(0)に対する重み(門前重視)
        self.call_weight = 1.0         # Call(1)に対する重み
        self.gamma = gamma             # Hard Negativeへのペナルティ強度

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = torch.where(targets == 0, self.pass_weight, self.call_weight)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()

# =========================================
# 📂 1. 環境準備とデータ読み込み (ポン＆チー合体)
# =========================================
# ⭐️ ポンとチーの大規模データセットを両方指定します
DATASET_FILES = [
    "./dataset_pon_phase5a_large.pkl",
    "./dataset_chi_phase5b_large.pkl"
]

all_records = []
print(f"📂 保存された【ポン】および【チー】のデータを合体して読み込んでいます...")

for file_path in DATASET_FILES:
    if not os.path.exists(file_path):
        print(f"❌ エラー: '{file_path}' が見つかりません。")
        sys.exit()
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        all_records.extend(data)
        print(f"  - {file_path} から {len(data)} 件読み込みました")

print(f"📊 データ合体完了！ 総データ数: {len(all_records)} 件")

# =========================================
# ⚙️ 2. 特徴量構築 & Dataset
# =========================================
X_tensor, X_aux, y = [], [], []

for r in all_records:
    X_tensor.append(r["tensor"])
    
    # ⭐️ 補助特徴量 (10次元) の構築
    # 過去のデータセットに "aux" が10次元で保存されていればそれを使う
    if "aux" in r and len(r["aux"]) == 10:
        aux = np.array(r["aux"], dtype=np.float32)
    else:
        # 保存されていない場合は、rの中身から10次元を手作りする
        aux = np.array([
            float(r.get("wait_count", 0)), 
            float(r.get("is_ryankei", 0)), 
            float(r.get("meta_turn", 0)), 
            float(r.get("meta_enemy_riichi", 0)),
            float(r.get("meta_dora_count", 0)), 
            float(r.get("meta_dama_legal", 0)), 
            float(r.get("meta_my_rank", 0)), 
            float(r.get("meta_point_diff", 0)),
            float(r.get("is_pon_chance", 1.0)), # ポンかチーかの識別用ダミー等
            float(r.get("meta_score_proxy", 0))
        ], dtype=np.float32)
    
    X_aux.append(aux)
    
    # ⭐️ 正解ラベルの取得 (ポンデータならlabel_pon、チーデータならlabel_chiを拾う)
    # どちらも無い場合は "label" を探す、という安全な取得方法
    label = r.get("label_pon", r.get("label_chi", r.get("label", 0)))
    y.append(np.float32(label))

X_t, X_a, y = np.array(X_tensor, dtype=np.float32), np.array(X_aux, dtype=np.float32), np.array(y).reshape(-1, 1)
indices = np.arange(len(y))

# 訓練:テスト = 80:20
X_t_tr, X_t_val, X_a_tr, X_a_val, y_tr, y_val = train_test_split(X_t, X_a, y, test_size=0.2, random_state=42, stratify=y)

class CallCNNDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

train_loader = DataLoader(CallCNNDataset(X_t_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
val_loader = DataLoader(CallCNNDataset(X_t_val, X_a_val, y_val), batch_size=256, shuffle=False)

# =========================================
# 🧠 3. call_net モデル定義と本番学習
# =========================================
class ActionCNN(nn.Module):
    def __init__(self, aux_dim=10): # ⭐️ call_net用に10次元を指定
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 + aux_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, t, a):
        return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActionCNN(10).to(device)

# ⭐️ レビューアー推奨の本命Loss: Focal Loss + Pass Weight
criterion = BinaryFocalLossWithWeight(pass_weight=1.0, gamma=1.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n🚀 【Phase 10G-2】call_net (鳴きモデル) 再学習開始！ (Train: {len(y_tr)}件, Valid: {len(y_val)}件)")
print("🚨 設定: Binary Focal Loss (gamma=1.5, pass_weight=1.0)")

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

# 最高性能の重みを復元し、上書き保存！
model.load_state_dict(best_wts)
torch.save(model.state_dict(), r"G:\マイドライブ\MahjongAI\call_best.pth")
print("✅ 新しい r'G:\マイドライブ\MahjongAI\call_best.pth' を保存しました！")

# =========================================
# 📊 4. 再学習後の単体評価（ミニレポート）
# =========================================
model.eval()
with torch.no_grad():
    cnn_probs = torch.sigmoid(model(torch.tensor(X_t_val).to(device), torch.tensor(X_a_val).to(device))).cpu().numpy().flatten()

best_acc, best_thresh = 0.0, 0.5
for thresh in np.arange(0.1, 0.9, 0.02):
    acc = accuracy_score(y_val, (cnn_probs >= thresh).astype(int))
    if acc > best_acc: best_acc, best_thresh = acc, thresh

pred_opt = (cnn_probs >= best_thresh).astype(int)

roc_auc = roc_auc_score(y_val, cnn_probs)
p_all = precision_score(y_val, pred_opt, zero_division=0)
r_all = recall_score(y_val, pred_opt, zero_division=0) # CallのRecall
r_pass = recall_score(y_val == 0, pred_opt == 0, zero_division=0) # ⭐️ PassのRecall(重要)

print("\n" + "="*60)
print("👑 Phase 10G-2: call_net 再学習 結果発表")
print("="*60)
print(f"■ 【全体指標】 ROC-AUC: {roc_auc:.4f}")
print(f"■ 最適閾値: {best_thresh:.2f} (全体Accuracy: {best_acc*100:.2f}%)")
print(f"  [Call Precision: {p_all*100:.1f}% | Call Recall: {r_all*100:.1f}%]")
print(f"  [⭐️ Pass Recall (門前維持力): {r_pass*100:.1f}%]")
print("="*60)
print("👉 次は 'run_phase10g3_eval_local.py' を実行して、副露率が改善したか確認してください！")