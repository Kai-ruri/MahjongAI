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

# 自作モジュール
from mahjong_engine import calculate_shanten

LOGS_DIR = "./logs"
TARGET_KYOKU_COUNT = 500  # プロトタイプとして500局

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
# 1. データ抽出 (メタ特徴 + 行動特徴)
# =========================================
def extract_phase9c_data():
    print(f"📁 {LOGS_DIR} から行動条件付きEVデータを抽出中... (目標: {TARGET_KYOKU_COUNT}局)")
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
                    # 🌟 実際の打牌行動をトラッキング
                    seat = 'DEFGdefg'.index(tag[0].upper())
                    tile_136 = int(tag[1:])
                    tile_34 = tile_136 // 4
                    
                    if tile_136 in hands_136[seat]:
                        hands_136[seat].remove(tile_136)
                        
                    counts_14 = [sum(1 for t in hands_136[seat] if t//4 == i) for i in range(34)]
                    counts_14[tile_34] += 1 # 打牌前の形を復元
                    shanten = 3 if not is_menzen[seat] else get_shanten_cached(counts_14)
                    
                    enemy_r = 1.0 if sum(1 for s in range(4) if s != seat and is_riichi[s]) > 0 else 0.0
                    dora_34_list = [get_dora_from_indicator(d // 4) for d in dora_inds]
                    dora_count = sum(counts_14[d] for d in dora_34_list)
                    my_score = scores[seat]
                    
                    # メタ特徴量 (10次元)
                    meta_feat = [
                        float(turn_counts[seat]), 1.0 if seat == oya_seat else 0.0,
                        float(sum(1 for s in scores if s > my_score) + 1),
                        (max(scores) - my_score) / 10000.0, (my_score - min(scores)) / 10000.0,
                        float(dora_count), float(shanten), enemy_r,
                        1.0 if is_menzen[seat] else 0.0, 1.0 if shanten <= 0 else 0.0
                    ]
                    
                    # 行動特徴量: [行動種別(打牌=0, 鳴き=1, リーチ=2), 打牌牌ID]
                    action_feat = [0.0, float(tile_34)] 
                    
                    current_kyoku_actions[seat].append({"meta": meta_feat, "action": action_feat})
                    
                elif tag == "REACH" and node.attrib.get("step") == "1":
                    seat = int(node.attrib.get("who"))
                    is_riichi[seat] = True
                    # リーチ行動として最後に記録した行動情報を上書き
                    if current_kyoku_actions[seat]:
                        current_kyoku_actions[seat][-1]["action"][0] = 2.0
                        
                elif tag == "N":
                    seat = int(node.attrib.get("who"))
                    is_menzen[seat] = False
                    if current_kyoku_actions[seat]:
                        current_kyoku_actions[seat][-1]["action"][0] = 1.0
                    
                elif tag in ["AGARI", "RYUUKYOKU"]:
                    sc_str = node.attrib.get("sc", "")
                    if sc_str:
                        sc_vals = [int(x)*100 for x in sc_str.split(",")]
                        for s in range(4):
                            delta = sc_vals[s*2 + 1]
                            # 🌟 タスク: マイナス(1) vs 非マイナス(0)
                            label = 1 if delta < 0 else 0
                            for act in current_kyoku_actions[s]:
                                act["label"] = label
                                all_records.append(act)
                    kyoku_processed += 1
                    
            if kyoku_processed >= TARGET_KYOKU_COUNT: break
        if kyoku_processed >= TARGET_KYOKU_COUNT: break

    print(f"✅ 抽出完了！ 取得アクション数: {len(all_records)} 件")
    return all_records

# =========================================
# 2. モデル定義 & データセット準備
# =========================================
class EV_MLP_Base1(nn.Module):
    """ Base 1: メタ特徴量のみ """
    def __init__(self, meta_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(meta_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, m, a): return self.mlp(m)

class EV_MLP_Base2(nn.Module):
    """ Base 2: メタ特徴量 + 行動条件付き特徴量 """
    def __init__(self, meta_dim=10, action_dim=35): # action_dim = 種別1 + tile_one_hot34
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(meta_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, m, a): return self.mlp(torch.cat([m, a], dim=1))

class EVActionDataset(Dataset):
    def __init__(self, m, a, l): self.m, self.a, self.l = torch.tensor(m), torch.tensor(a), torch.tensor(l, dtype=torch.float32)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.m[i], self.a[i], self.l[i]

def train_and_evaluate(model_class, name, X_m_tr, X_a_tr, y_tr, X_m_val, X_a_val, y_val):
    print(f"\n🚀 {name} の学習を開始します...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    
    # 不均衡データ（マイナスが多数派）への対応
    pos_weight = (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(EVActionDataset(X_m_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
    val_loader = DataLoader(EVActionDataset(X_m_val, X_a_val, y_val), batch_size=256, shuffle=False)
    
    best_loss, best_wts = float('inf'), None
    for epoch in range(12): # 高速化
        model.train()
        for m, a, l in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(m.to(device), a.to(device)), l.unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = sum(criterion(model(m.to(device), a.to(device)), l.unsqueeze(1).to(device)).item() * m.size(0) for m, a, l in val_loader) / len(y_val)
        if val_loss < best_loss: best_loss, best_wts = val_loss, copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_wts)
    model.eval()
    
    probs, trues = [], []
    with torch.no_grad():
        for m, a, l in val_loader:
            outputs = torch.sigmoid(model(m.to(device), a.to(device)))
            probs.extend(outputs.cpu().numpy().flatten())
            trues.extend(l.numpy())
            
    return np.array(trues), np.array(probs)

# =========================================
# 3. メイン処理 & 評価レポート
# =========================================
records = extract_phase9c_data()
X_m = np.array([r["meta"] for r in records], dtype=np.float32)

# 行動特徴量を One-hot 化して構築
X_a = []
for r in records:
    act_type = r["action"][0]
    tile_id = int(r["action"][1])
    tile_one_hot = np.zeros(34, dtype=np.float32)
    if 0 <= tile_id < 34: tile_one_hot[tile_id] = 1.0
    X_a.append(np.concatenate([[act_type], tile_one_hot]))
X_a = np.array(X_a, dtype=np.float32)

y = np.array([r["label"] for r in records])

print(f"📊 データ分布: マイナス収支(1)={sum(y)}件 ({sum(y)/len(y)*100:.1f}%) | 非マイナス(0)={len(y)-sum(y)}件")

X_m_tr, X_m_val, X_a_tr, X_a_val, y_tr, y_val = train_test_split(X_m, X_a, y, test_size=0.2, random_state=42, stratify=y)

# 学習の実行
_, probs_base1 = train_and_evaluate(EV_MLP_Base1, "Base 1 (メタ特徴のみ MLP)", X_m_tr, X_a_tr, y_tr, X_m_val, X_a_val, y_val)
_, probs_base2 = train_and_evaluate(EV_MLP_Base2, "Base 2 (メタ + 行動条件付き MLP)", X_m_tr, X_a_tr, y_tr, X_m_val, X_a_val, y_val)

def print_metrics(name, y_true, probs):
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    cm = confusion_matrix(y_true, preds)
    
    print(f"\n[{name}]")
    print(f"  Accuracy: {acc*100:.1f}% | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"  [Target=1 (マイナス収支) 予測]")
    print(f"    - Precision: {p*100:.1f}% | Recall: {r*100:.1f}% | F1: {f1*100:.1f}%")
    print(f"  [Confusion Matrix]")
    print(f"    True 0 (非マイナス): Pred 0={cm[0][0]}, Pred 1={cm[0][1]}")
    print(f"    True 1 (マイナス)  : Pred 0={cm[1][0]}, Pred 1={cm[1][1]}")

# 全てマイナスと予測する Majority Baseline の計算
maj_preds = np.ones_like(y_val)
print("\n" + "="*80)
print("👑 フェーズ9C: 行動条件付きEV予測 (マイナス vs 非マイナス)")
print("="*80)
print(f"[Majority Baseline (すべてマイナスと予測)]")
print(f"  Accuracy: {accuracy_score(y_val, maj_preds)*100:.1f}% | F1: {f1_score(y_val, maj_preds)*100:.1f}%")

print_metrics("Base 1 (メタ特徴のみ MLP)", y_val, probs_base1)
print_metrics("Base 2 (メタ + 行動条件付き MLP)", y_val, probs_base2)
print("="*80)