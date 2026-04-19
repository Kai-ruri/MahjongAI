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

with open(DATASET_FILE, 'rb') as f:
    all_records = pickle.load(f)

# 中間(1)を除外し、0(完全オリ)と2(無筋押し)の2値分類(0と1)にマッピング
binary_records = [r for r in all_records if r["label_oshiki"] in [0, 2]]
print(f"📊 評価対象データ (純粋な押し/オリ): {len(binary_records)} 件")

# =========================================
# 1. 新特徴量の構築 (E0, E1, E2, E3用)
# =========================================
X_tensor, y = [], []
X_e0, X_e1, X_e2, X_e3 = [], [], [], []

for r in binary_records:
    X_tensor.append(r["tensor"])
    y.append(np.float32(1.0 if r["label_oshiki"] == 2 else 0.0))
    
    # --- E0: 現行ベースライン (フェーズ7C仕様 + レビューアー推奨のフラグ明示化) ---
    high_value_proxy = 1.0 if r["dora_count"] >= 2 else 0.0
    is_top = 1.0 if r["my_rank"] == 1 else 0.0
    is_last = 1.0 if r["my_rank"] == 4 else 0.0
    is_early = 1.0 if r["turn"] <= 6 else 0.0
    is_late = 1.0 if r["turn"] >= 12 else 0.0
    is_multi_riichi = 1.0 if r["enemy_riichi_count"] >= 2 else 0.0
    
    feat_e0 = [
        float(r["shanten_before"]), 1.0 if r["shanten_before"] <= 0 else 0.0, 1.0 if r["shanten_before"] == 1 else 0.0,
        float(r["enemy_riichi_count"]), float(r["turn"]), 1.0 if r["is_oya"] else 0.0,
        float(r["dora_count"]), high_value_proxy, float(r["my_rank"]),
        float(r["diff_top"]) / 10000.0, float(r["diff_last"]) / 10000.0, 1.0 if r["ooras"] else 0.0,
        is_top, is_last, is_early, is_late, is_multi_riichi # レビューアー推奨の強調フラグ
    ]
    
    # --- E1: 打牌後の手価値特徴 ---
    # 抽出時点でshanten_afterを保存していなかったため、is_maintainedから疑似的に復元
    # (※厳密には再計算が必要ですが、アブレーションの傾向を見るためのproxyとします)
    shanten_after_proxy = float(r["shanten_before"]) if r["is_maintained"] else float(r["shanten_before"] + 1)
    is_tenpai_maintained = 1.0 if (r["shanten_before"] <= 0 and r["is_maintained"]) else 0.0
    is_1shan_maintained = 1.0 if (r["shanten_before"] == 1 and r["is_maintained"]) else 0.0
    is_te_modori = 1.0 if not r["is_maintained"] else 0.0
    
    feat_e1 = [shanten_after_proxy, is_tenpai_maintained, is_1shan_maintained, is_te_modori]
    
    # --- E2: 安牌保有枚数 proxy ---
    # 現物枚数は、テンソル(25ch)のうち、自手(0ch)とリーチ者の河(6ch,10ch,14ch等)の重複から簡易算出
    hand_counts = np.round(r["tensor"][0] * 4.0).astype(int)
    # リーチ者の判定(21-24ch)
    riichi_seats = [i for i in range(4) if r["tensor"][21+i][0] == 1.0]
    
    genbutsu_count = 0
    jihai_count = sum(hand_counts[27:34]) # 字牌を簡易的な安牌候補としてカウント
    
    for tile_idx in range(34):
        if hand_counts[tile_idx] > 0:
            # 全リーチ者に対して現物かチェック
            is_genbutsu = True
            for pov_seat in riichi_seats:
                # 対象者の河chは 2 + pov_seat*4
                if r["tensor"][2 + pov_seat*4][tile_idx] == 0.0:
                    is_genbutsu = False
                    break
            if is_genbutsu:
                genbutsu_count += hand_counts[tile_idx]
                
    feat_e2 = [float(genbutsu_count), float(jihai_count)]
    
    # --- 結合 ---
    X_e0.append(np.array(feat_e0, dtype=np.float32))
    X_e1.append(np.concatenate([feat_e0, feat_e1]).astype(np.float32))
    X_e2.append(np.concatenate([feat_e0, feat_e2]).astype(np.float32))
    X_e3.append(np.concatenate([feat_e0, feat_e1, feat_e2]).astype(np.float32))

X_t = np.array(X_tensor, dtype=np.float32)
y = np.array(y).reshape(-1, 1)

indices = np.arange(len(y))
idx_tr, idx_val = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
meta_val = [binary_records[i] for i in idx_val]

class OshibikiDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

# =========================================
# 2. モデル訓練と評価エンジン
# =========================================
class OshibikiCNN_Dynamic(nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + aux_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

def train_and_evaluate(name, X_aux_list):
    X_a_tr = np.array([X_aux_list[i] for i in idx_tr])
    X_a_val = np.array([X_aux_list[i] for i in idx_val])
    y_tr_split, y_val_split = y[idx_tr], y[idx_val]
    
    train_loader = DataLoader(OshibikiDataset(X_t[idx_tr], X_a_tr, y_tr_split), batch_size=256, shuffle=True)
    val_loader = DataLoader(OshibikiDataset(X_t[idx_val], X_a_val, y_val_split), batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OshibikiCNN_Dynamic(aux_dim=X_a_tr.shape[1]).to(device)
    
    pos_weight = float((len(y_tr_split) - y_tr_split.sum()) / max(1, y_tr_split.sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss, best_wts, patience = float('inf'), None, 0
    for epoch in range(15): # アブレーションのため高速化
        model.train()
        for t, a, l in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(t.to(device), a.to(device)), l.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = sum(criterion(model(t.to(device), a.to(device)), l.to(device)).item() * t.size(0) for t, a, l in val_loader) / len(y_val_split)
        if val_loss < best_loss: best_loss, best_wts, patience = val_loss, copy.deepcopy(model.state_dict()), 0
        elif (patience := patience + 1) >= 3: break
        
    model.load_state_dict(best_wts)
    model.eval()
    with torch.no_grad():
        cnn_probs = torch.sigmoid(model(torch.tensor(X_t[idx_val]).to(device), torch.tensor(X_a_val).to(device))).cpu().numpy().flatten()
    
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.3, 0.8, 0.02):
        f1 = f1_score(y_val_split, (cnn_probs >= thresh).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_thresh = f1, thresh
        
    pred_opt = (cnn_probs >= best_thresh).astype(int)
    acc = accuracy_score(y_val_split, pred_opt)
    p = precision_score(y_val_split, pred_opt, zero_division=0)
    r = recall_score(y_val_split, pred_opt, zero_division=0)
    
    # エラー分析指標 (FP: 押しすぎ, FN: 慎重すぎ)
    def calc_error_rate(condition_fn, error_type):
        idx = [i for i, m in enumerate(meta_val) if condition_fn(m)]
        if not idx: return 0.0
        y_t_sub = y_val_split[idx].flatten()
        y_p_sub = pred_opt[idx]
        if error_type == "FP": # 正解0なのに1と予測 (押しすぎ)
            target_cases = sum(1 for yt, yp in zip(y_t_sub, y_p_sub) if yt == 0 and yp == 1)
            base_cases = sum(1 for yt in y_t_sub if yt == 0)
        else: # 正解1なのに0と予測 (慎重すぎ)
            target_cases = sum(1 for yt, yp in zip(y_t_sub, y_p_sub) if yt == 1 and yp == 0)
            base_cases = sum(1 for yt in y_t_sub if yt == 1)
        return (target_cases / base_cases * 100) if base_cases > 0 else 0.0

    return {
        "name": name, "acc": acc*100, "f1": best_f1*100, "p": p*100, "r": r*100,
        "fp_top": calc_error_rate(lambda m: m["my_rank"] == 1, "FP"),           # トップ目の押しすぎ率
        "fp_multi": calc_error_rate(lambda m: m["enemy_riichi_count"] >= 2, "FP"), # 2軒リーチの押しすぎ率
        "fn_early": calc_error_rate(lambda m: m["turn"] <= 6, "FN"),            # 序盤の慎重すぎ率
        "fn_last": calc_error_rate(lambda m: m["my_rank"] == 4 and m["turn"] >= 12, "FN") # ラス目終盤の慎重すぎ率
    }

print("\n🚀 【フェーズ7E】アブレーション・スタディ開始！")
results = []
results.append(train_and_evaluate("E0 (現行CNN-A)", X_e0))
results.append(train_and_evaluate("E1 (+ 打牌後価値)", X_e1))
results.append(train_and_evaluate("E2 (+ 安牌枚数proxy)", X_e2))
results.append(train_and_evaluate("E3 (両方追加・完全体)", X_e3))

print("\n" + "="*90)
print("👑 フェーズ7E: 押し引き アブレーション・スタディ 結果発表")
print("="*90)
print(f"{'モデル':<22} | {'F1 (%)':<6} | {'Acc (%)':<7} | {'Recall':<6} | {'トップFP↓':<10} | {'2軒R_FP↓':<10} | {'序盤FN↓':<10} | {'ラス終盤FN↓':<10}")
print("-" * 90)
for r in results:
    print(f"{r['name']:<22} | {r['f1']:>6.1f} | {r['acc']:>7.1f} | {r['r']:>6.1f} | {r['fp_top']:>9.1f}% | {r['fp_multi']:>9.1f}% | {r['fn_early']:>9.1f}% | {r['fn_last']:>10.1f}%")
print("="*90)
print("※ FP↓: 押しすぎエラー率(低いほど良い) / FN↓: 慎重すぎエラー率(低いほど良い)")