import os
import re
import gzip
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import copy
import sys

# 自作モジュール
from mahjong_engine import MahjongStateV5, calculate_shanten

LOGS_DIR = "./logs"
TARGET_KYOKU_COUNT = 500  # リアルなテンソル生成を行うため500局のサブセットで学習

# 🚀 高速化：シャンテン計算キャッシュ
_shanten_cache = {}
def get_shanten_cached(counts):
    t = tuple(counts)
    if t not in _shanten_cache:
        _shanten_cache[t] = calculate_shanten(list(counts))
    return _shanten_cache[t]

def get_dora_from_indicator(indicator_34):
    if indicator_34 < 27: return (indicator_34 // 9) * 9 + (indicator_34 % 9 + 1) % 9
    elif indicator_34 < 31: return 27 + (indicator_34 - 27 + 1) % 4
    else: return 31 + (indicator_34 - 31 + 1) % 3

# =========================================
# 1. リアルEVデータ抽出エンジン (本物のテンソル付き)
# =========================================
def extract_real_ev_data():
    print(f"📁 {LOGS_DIR} からEVデータ(本物テンソル付き)を抽出中... (目標: {TARGET_KYOKU_COUNT}局)")
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.gz')]
    
    all_records = []
    kyoku_processed = 0

    for filename in log_files:
        log_ids = []
        with gzip.open(os.path.join(LOGS_DIR, filename), 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(r'log=([\w-]+)', line)
                if match: log_ids.append(match.group(1))
        
        for log_id in list(set(log_ids)):
            try:
                req = urllib.request.Request(f"https://tenhou.net/0/log/?{log_id}", headers={'User-Agent': 'Mozilla/5.0'})
                xml_string = urllib.request.urlopen(req).read().decode('utf-8')
                root = ET.fromstring(xml_string)
            except Exception: continue
            
            current_kyoku_states = {0: [], 1: [], 2: [], 3: []}
            hands_136 = {i: [] for i in range(4)}
            discards_136 = {i: [] for i in range(4)}
            is_riichi = {i: False for i in range(4)}
            is_menzen = {i: True for i in range(4)}
            dora_inds = []
            scores = [25000]*4
            turn_counts = [0]*4
            oya_seat = 0
            
            for node in root:
                tag = node.tag
                if tag == "INIT":
                    current_kyoku_states = {0: [], 1: [], 2: [], 3: []}
                    hands_136 = {i: [] for i in range(4)}
                    discards_136 = {i: [] for i in range(4)}
                    is_riichi = {i: False for i in range(4)}
                    is_menzen = {i: True for i in range(4)}
                    turn_counts = [0]*4
                    
                    seed = node.attrib.get("seed", "0,0,0,0,0,0").split(",")
                    dora_inds = [int(seed[5])]
                    scores = [int(x)*100 for x in node.attrib.get("ten", "250,250,250,250").split(",")]
                    oya_seat = int(node.attrib.get("oya", "0"))
                    
                    for i in range(4):
                        hai_str = node.attrib.get(f"hai{i}", "")
                        hands_136[i] = [int(x) for x in hai_str.split(",") if x]
                        
                elif tag[0] in 'TUVW' and tag[1:].isdigit():
                    seat = 'TUVW'.index(tag[0])
                    tile_136 = int(tag[1:])
                    hands_136[seat].append(tile_136)
                    turn_counts[seat] += 1
                    
                    # 🌟 盤面テンソルと特徴量のスナップショット作成
                    state = MahjongStateV5()
                    for t_136 in hands_136[seat]: state.add_tile(0, t_136)
                    for pov in range(4):
                        for t_136 in discards_136[(seat + pov) % 4]: state.discards[pov].append(t_136 // 4)
                        state.riichi_declared[pov] = is_riichi[(seat + pov) % 4]
                    
                    counts_14 = [sum(1 for t in hands_136[seat] if t//4 == i) for i in range(34)]
                    shanten = 3 if not is_menzen[seat] else get_shanten_cached(counts_14) # 簡易シャンテン
                    
                    enemy_r = 1.0 if sum(1 for s in range(4) if s != seat and is_riichi[s]) > 0 else 0.0
                    dora_34_list = [get_dora_from_indicator(d // 4) for d in dora_inds]
                    dora_count = sum(counts_14[d] for d in dora_34_list)
                    my_score = scores[seat]
                    
                    aux = [
                        float(turn_counts[seat]),
                        1.0 if seat == oya_seat else 0.0,
                        float(sum(1 for s in scores if s > my_score) + 1), # 順位
                        (max(scores) - my_score) / 10000.0,                # トップ差
                        (my_score - min(scores)) / 10000.0,                # ラス差
                        float(dora_count),
                        float(shanten),
                        enemy_r,
                        1.0 if is_menzen[seat] else 0.0,
                        1.0 if shanten <= 0 else 0.0                       # テンパイフラグ
                    ]
                    
                    current_kyoku_states[seat].append({
                        "tensor": state.to_tensor(skip_logic=True),
                        "aux": np.array(aux, dtype=np.float32)
                    })
                    
                elif tag[0] in 'DEFGdefg' and tag[1:].isdigit():
                    seat = 'DEFGdefg'.index(tag[0].upper())
                    tile_136 = int(tag[1:])
                    if tile_136 in hands_136[seat]:
                        hands_136[seat].remove(tile_136)
                        discards_136[seat].append(tile_136)
                        
                elif tag == "N":
                    seat = int(node.attrib.get("who"))
                    is_menzen[seat] = False
                elif tag == "REACH" and node.attrib.get("step") == "1":
                    is_riichi[int(node.attrib.get("who"))] = True
                    
                elif tag in ["AGARI", "RYUUKYOKU"]:
                    sc_str = node.attrib.get("sc", "")
                    if sc_str:
                        sc_vals = [int(x)*100 for x in sc_str.split(",")]
                        deltas = {0: sc_vals[1], 1: sc_vals[3], 2: sc_vals[5], 3: sc_vals[7]}
                        for s in range(4):
                            delta = deltas[s]
                            label = 2 if delta > 0 else (1 if delta == 0 else 0) # 0:マイナス, 1:ゼロ, 2:プラス
                            for st in current_kyoku_states[s]:
                                st["label"] = label
                                all_records.append(st)
                    kyoku_processed += 1
                    
            if kyoku_processed >= TARGET_KYOKU_COUNT: break
        if kyoku_processed >= TARGET_KYOKU_COUNT: break

    print(f"✅ 抽出完了！ 取得局面数: {len(all_records)} 件")
    return all_records

# =========================================
# 2. モデル定義 & データ準備
# =========================================
class EV_CNN_A(nn.Module):
    def __init__(self, aux_dim=10):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + aux_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3)) # 3値分類
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

class EV_MLP_Baseline(nn.Module):
    def __init__(self, aux_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(aux_dim, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, t, a): return self.mlp(a)

class EVDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l, dtype=torch.long)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

def train_and_evaluate(model_class, name, X_t_tr, X_a_tr, y_tr, X_t_val, X_a_val, y_val, class_weights):
    print(f"\n🚀 {name} の学習を開始します...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    
    # 🌟 レビューアー推奨: Class Weight付き Loss
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(EVDataset(X_t_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
    val_loader = DataLoader(EVDataset(X_t_val, X_a_val, y_val), batch_size=256, shuffle=False)
    
    best_loss, best_wts = float('inf'), None
    for epoch in range(10): # 高速化のため10エポック
        model.train()
        for t, a, l in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(t.to(device), a.to(device)), l.to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = sum(criterion(model(t.to(device), a.to(device)), l.to(device)).item() * t.size(0) for t, a, l in val_loader) / len(y_val)
        if val_loss < best_loss: best_loss, best_wts = val_loss, copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_wts)
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for t, a, l in val_loader:
            outputs = model(t.to(device), a.to(device))
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(l.numpy())
            
    return np.array(trues), np.array(preds)

# =========================================
# 3. メイン処理 & レポート出力
# =========================================
records = extract_real_ev_data()
X_t = np.array([r["tensor"] for r in records], dtype=np.float32)
X_a = np.array([r["aux"] for r in records], dtype=np.float32)
y = np.array([r["label"] for r in records])

# Class Weights の手動計算 (N_total / (3 * N_class))
class_counts = np.bincount(y)
class_weights = len(y) / (3.0 * class_counts)
print(f"⚖️ Class Weights: マイナス={class_weights[0]:.2f}, ゼロ={class_weights[1]:.2f}, プラス={class_weights[2]:.2f}")

X_t_tr, X_t_val, X_a_tr, X_a_val, y_tr, y_val = train_test_split(X_t, X_a, y, test_size=0.2, random_state=42, stratify=y)

# --- Base: Majority Baseline (常にマイナス=0を予測) ---
y_pred_maj = np.zeros_like(y_val)

# --- Model 1: MLP Baseline (点数状況・シャンテン等のメタ情報のみ) ---
_, y_pred_mlp = train_and_evaluate(EV_MLP_Baseline, "MLP Baseline (メタ特徴量のみ)", X_t_tr, X_a_tr, y_tr, X_t_val, X_a_val, y_val, class_weights)

# --- Model 2: CNN-A (完全体) ---
_, y_pred_cnn = train_and_evaluate(EV_CNN_A, "CNN-A (盤面テンソル + メタ特徴量)", X_t_tr, X_a_tr, y_tr, X_t_val, X_a_val, y_val, class_weights)

# 📊 評価レポート関数
def print_eval_report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(f"\n[{name}]")
    print(f"  Accuracy: {acc*100:.1f}% | Macro F1: {macro_f1*100:.1f}%")
    print("  [クラス別 F1スコア]")
    for i, label in enumerate(["マイナス(0)", "ゼロ(1)", "プラス(2)"]):
        p = precision_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
        r = recall_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
        f = f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
        print(f"    - {label:<7}: Precision {p*100:>4.1f}% | Recall {r*100:>4.1f}% | F1 {f*100:>4.1f}%")
    print("  [Confusion Matrix (行:正解, 列:予測)]")
    print(f"              pred_0  pred_1  pred_2")
    print(f"    true_0: {cm[0][0]:>7} {cm[0][1]:>7} {cm[0][2]:>7}")
    print(f"    true_1: {cm[1][0]:>7} {cm[1][1]:>7} {cm[1][2]:>7}")
    print(f"    true_2: {cm[2][0]:>7} {cm[2][1]:>7} {cm[2][2]:>7}")

print("\n" + "="*80)
print("👑 フェーズ9B: EV予測 3値分類ベースライン 評価結果")
print("="*80)
print_eval_report("1. Majority Baseline (常にマイナス予測)", y_val, y_pred_maj)
print_eval_report("2. MLP Baseline (メタ特徴量のみ)", y_val, y_pred_mlp)
print_eval_report("3. CNN-A (盤面テンソル + メタ特徴量)", y_val, y_pred_cnn)
print("="*80)