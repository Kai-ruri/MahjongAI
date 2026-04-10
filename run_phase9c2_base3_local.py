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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import copy
import pickle

# 自作モジュール
from mahjong_engine import MahjongStateV5, calculate_shanten

LOGS_DIR = "./logs"
TARGET_KYOKU_COUNT = 500
CACHE_FILE = "./dataset_ev_phase9c2_base3.pkl" # 🌟 9C-2専用キャッシュ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_shanten_cache = {}
def get_shanten_cached(counts):
    t = tuple(counts)
    if t not in _shanten_cache: _shanten_cache[t] = calculate_shanten(list(counts))
    return _shanten_cache[t]

def get_dora_from_indicator(indicator_34):
    if indicator_34 < 27: return (indicator_34 // 9) * 9 + (indicator_34 % 9 + 1) % 9
    elif indicator_34 < 31: return 27 + (indicator_34 - 27 + 1) % 4
    else: return 31 + (indicator_34 - 31 + 1) % 3

# =========================================
# 1. データ抽出 (プラス収支ラベル)
# =========================================
def extract_phase9c2_data():
    if os.path.exists(CACHE_FILE):
        print(f"📦 キャッシュファイル {CACHE_FILE} からデータを読み込みます...")
        with open(CACHE_FILE, 'rb') as f: return pickle.load(f)
        
    print(f"📁 {LOGS_DIR} から行動条件付きEVデータ(プラス予測用)を抽出中... (目標: {TARGET_KYOKU_COUNT}局)")
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
            
            current_kyoku_actions = {0: [], 1: [], 2: [], 3: []}
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
                    current_kyoku_actions = {0: [], 1: [], 2: [], 3: []}
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
                        hands_136[i] = [int(x) for x in node.attrib.get(f"hai{i}", "").split(",") if x]
                        
                elif tag[0] in 'TUVW' and tag[1:].isdigit():
                    seat = 'TUVW'.index(tag[0])
                    hands_136[seat].append(int(tag[1:]))
                    turn_counts[seat] += 1
                    
                elif tag[0] in 'DEFGdefg' and tag[1:].isdigit():
                    seat = 'DEFGdefg'.index(tag[0].upper())
                    tile_136 = int(tag[1:])
                    tile_34 = tile_136 // 4
                    
                    state = MahjongStateV5()
                    for t_136 in hands_136[seat]: state.add_tile(0, t_136)
                    for pov in range(4):
                        for t_136_d in discards_136[(seat + pov) % 4]: state.discards[pov].append(t_136_d // 4)
                        state.riichi_declared[pov] = is_riichi[(seat + pov) % 4]
                    tensor_data = state.to_tensor(skip_logic=True)
                    
                    if tile_136 in hands_136[seat]:
                        hands_136[seat].remove(tile_136)
                    discards_136[seat].append(tile_136)
                        
                    counts_14 = [sum(1 for t in hands_136[seat] if t//4 == i) for i in range(34)]
                    counts_14[tile_34] += 1 
                    shanten = 3 if not is_menzen[seat] else get_shanten_cached(counts_14)
                    
                    enemy_r = 1.0 if sum(1 for s in range(4) if s != seat and is_riichi[s]) > 0 else 0.0
                    dora_34_list = [get_dora_from_indicator(d // 4) for d in dora_inds]
                    dora_count = sum(counts_14[d] for d in dora_34_list)
                    my_score = scores[seat]
                    
                    meta_feat = [
                        float(turn_counts[seat]), 1.0 if seat == oya_seat else 0.0,
                        float(sum(1 for s in scores if s > my_score) + 1),
                        (max(scores) - my_score) / 10000.0, (my_score - min(scores)) / 10000.0,
                        float(dora_count), float(shanten), enemy_r,
                        1.0 if is_menzen[seat] else 0.0, 1.0 if shanten <= 0 else 0.0
                    ]
                    act_one_hot = np.zeros(34, dtype=np.float32)
                    act_one_hot[tile_34] = 1.0
                    action_feat = np.concatenate([[0.0], act_one_hot])
                    
                    current_kyoku_actions[seat].append({"tensor": tensor_data, "meta": np.array(meta_feat, dtype=np.float32), "action": action_feat})
                    
                elif tag == "REACH" and node.attrib.get("step") == "1":
                    seat = int(node.attrib.get("who"))
                    is_riichi[seat] = True
                    if current_kyoku_actions[seat]: current_kyoku_actions[seat][-1]["action"][0] = 2.0
                        
                elif tag == "N":
                    seat = int(node.attrib.get("who"))
                    is_menzen[seat] = False
                    if current_kyoku_actions[seat]: current_kyoku_actions[seat][-1]["action"][0] = 1.0
                    
                elif tag in ["AGARI", "RYUUKYOKU"]:
                    sc_str = node.attrib.get("sc", "")
                    if sc_str:
                        sc_vals = [int(x)*100 for x in sc_str.split(",")]
                        for s in range(4):
                            # 🌟 タスク変更: プラス収支(>0)なら1、それ以外(ゼロ・マイナス)なら0
                            label = 1 if sc_vals[s*2 + 1] > 0 else 0
                            for act in current_kyoku_actions[s]:
                                act["label"] = label
                                all_records.append(act)
                    kyoku_processed += 1
                    if kyoku_processed % 100 == 0: print(f"🔄 処理中... {kyoku_processed}局完了")
                    
            if kyoku_processed >= TARGET_KYOKU_COUNT: break
        if kyoku_processed >= TARGET_KYOKU_COUNT: break

    print(f"✅ 抽出完了！ 取得アクション数: {len(all_records)} 件")
    with open(CACHE_FILE, 'wb') as f: pickle.dump(all_records, f)
    return all_records

# =========================================
# 2. モデル定義
# =========================================
class EV_MLP_Base1(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, t, m, a): return self.mlp(m)

class EV_MLP_Base2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(10 + 35, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, m, a): return self.mlp(torch.cat([m, a], dim=1))

class EV_CNN_Base3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp = nn.Sequential(nn.Linear(64 + 10 + 35, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, m, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), m, a], dim=1))

class EVDataset(Dataset):
    def __init__(self, t, m, a, l): self.t, self.m, self.a, self.l = torch.tensor(t), torch.tensor(m), torch.tensor(a), torch.tensor(l, dtype=torch.float32)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.m[i], self.a[i], self.l[i]

def train_eval(model_class, name, X_t_tr, X_m_tr, X_a_tr, y_tr, X_t_val, X_m_val, X_a_val, y_val):
    print(f"\n🚀 {name} 学習開始...")
    model = model_class().to(device)
    pos_weight = (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(EVDataset(X_t_tr, X_m_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
    val_loader = DataLoader(EVDataset(X_t_val, X_m_val, X_a_val, y_val), batch_size=256, shuffle=False)
    
    best_loss, best_wts = float('inf'), None
    for epoch in range(12):
        model.train()
        for t, m, a, l in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(t.to(device), m.to(device), a.to(device)), l.unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = sum(criterion(model(t.to(device), m.to(device), a.to(device)), l.unsqueeze(1).to(device)).item() * t.size(0) for t, m, a, l in val_loader) / len(y_val)
        if val_loss < best_loss: best_loss, best_wts = val_loss, copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_wts)
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for t, m, a, l in val_loader:
            outputs = torch.sigmoid(model(t.to(device), m.to(device), a.to(device)))
            probs.extend(outputs.cpu().numpy().flatten())
            trues.extend(l.numpy())
    return np.array(probs)

# =========================================
# 3. 実行 & レポート
# =========================================
records = extract_phase9c2_data()
X_t = np.array([r["tensor"] for r in records], dtype=np.float32)
X_m = np.array([r["meta"] for r in records], dtype=np.float32)
X_a = np.array([r["action"] for r in records], dtype=np.float32)
y = np.array([r["label"] for r in records])

print(f"📊 データ分布: プラス(1)={sum(y)}件 ({sum(y)/len(y)*100:.1f}%) | 非プラス(0)={len(y)-sum(y)}件")

idx = np.arange(len(y))
idx_tr, idx_val = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

probs_b1 = train_eval(EV_MLP_Base1, "Base 1 (メタのみMLP)", X_t[idx_tr], X_m[idx_tr], X_a[idx_tr], y[idx_tr], X_t[idx_val], X_m[idx_val], X_a[idx_val], y[idx_val])
probs_b2 = train_eval(EV_MLP_Base2, "Base 2 (メタ+行動MLP)", X_t[idx_tr], X_m[idx_tr], X_a[idx_tr], y[idx_tr], X_t[idx_val], X_m[idx_val], X_a[idx_val], y[idx_val])
probs_b3 = train_eval(EV_CNN_Base3, "Base 3 (盤面+メタ+行動 CNN-A)", X_t[idx_tr], X_m[idx_tr], X_a[idx_tr], y[idx_tr], X_t[idx_val], X_m[idx_val], X_a[idx_val], y[idx_val])

def print_metrics(name, y_true, probs):
    preds = (probs >= 0.5).astype(int)
    print(f"\n[{name}]")
    print(f"  Accuracy: {accuracy_score(y_true, preds)*100:.1f}% | ROC-AUC: {roc_auc_score(y_true, probs):.4f} | PR-AUC: {average_precision_score(y_true, probs):.4f}")
    print(f"  [Target=1 (プラス) 予測] F1: {f1_score(y_true, preds, zero_division=0)*100:.1f}% (Pre: {precision_score(y_true, preds, zero_division=0)*100:.1f}%, Rec: {recall_score(y_true, preds, zero_division=0)*100:.1f}%)")

print("\n" + "="*80)
print("👑 フェーズ9C-2: EV予測 (プラス vs 非プラス) Base 1〜3 比較レポート")
print("="*80)
print_metrics("Base 1 (メタのみ MLP)", y[idx_val], probs_b1)
print_metrics("Base 2 (メタ+行動 MLP)", y[idx_val], probs_b2)
print_metrics("Base 3 (盤面テンソル+メタ+行動 CNN-A)", y[idx_val], probs_b3)
print("="*80)