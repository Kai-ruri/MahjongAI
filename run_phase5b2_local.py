import os
import re
import time
import pickle
import urllib.request
import sys
import gzip
import xml.etree.ElementTree as ET
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve

from mahjong_engine import tile_names, MahjongStateV5, calculate_shanten

# =========================================
# 0. 設定と環境準備
# =========================================
TARGET_RECORDS = 30000
LOGS_DIR = "./logs"  # 前回と同じ70個の .html.gz が入っているフォルダ
OUTPUT_FILE = "./dataset_chi_phase5b2_large.pkl" # 新しいファイル名

if not os.path.exists(LOGS_DIR):
    print(f"❌ エラー: '{LOGS_DIR}' フォルダが見つかりません。")
    sys.exit()

# =========================================
# 🛡️ 1. 特徴量計算ヘルパー (打点・役の追加！)
# =========================================
def get_ukeire_and_shanten(counts):
    s = calculate_shanten(counts)
    u_count = 0
    for i in range(34):
        if counts[i] < 4:
            counts[i] += 1
            if calculate_shanten(counts) < s:
                u_count += (4 - counts[i] + 1)
            counts[i] -= 1
    return s, u_count

def evaluate_chi_features_v2(hand_counts, target_tile, bakaze, jikaze, dora_34_list):
    s_before, u_before = get_ukeire_and_shanten(hand_counts)
    temp = list(hand_counts)
    temp[target_tile] += 1
    
    best_s_after = 99
    best_u_after = 0
    for i in range(34):
        if temp[i] > 0 and i != target_tile:
            temp[i] -= 1
            s_after, u_after = get_ukeire_and_shanten(temp)
            if s_after < best_s_after or (s_after == best_s_after and u_after > best_u_after):
                best_s_after = s_after
                best_u_after = u_after
            temp[i] += 1
            
    # 【スピード・形】
    freedom = sum(1 for x in temp if x > 0) - 1
    is_improved = 1.0 if (best_s_after < s_before) else 0.0
    is_ryoukei = 1.0 if (is_improved and ((best_s_after == 0 and best_u_after >= 5) or (best_s_after > 0 and best_u_after >= 8))) else 0.0
    is_gukei = 1.0 if (is_improved and ((best_s_after == 0 and best_u_after <= 4) or (best_s_after > 0 and best_u_after <= 7))) else 0.0
    
    # 【新設1】待ち質低下 (単騎化など: 受け入れ3枚以下)
    bad_wait_after_chi = 1.0 if (is_improved and best_u_after <= 3) else 0.0
    
    # 【新設2】門前価値消失 (喰い下がりリスク: 門前で1シャンテン以下)
    menzen_value_loss = 1.0 if s_before <= 1 else 0.0
    
    # 【新設3】ドラカウント
    dora_count = sum(temp[d] for d in dora_34_list)
    
    # 【新設4】役牌対子数
    yakuhai_pairs = 0
    for t in set([27+bakaze, 27+jikaze, 31, 32, 33]):
        if temp[t] >= 2: yakuhai_pairs += 1
        
    # 【新設5】ホンイツポテンシャル
    m, p, s, z = sum(temp[0:9]), sum(temp[9:18]), sum(temp[18:27]), sum(temp[27:34])
    honitsu_potential = (max(m, p, s) + z) / 14.0
    
    # 【新設6】チャンタポテンシャル (1,9,字牌の比率)
    yaochu_cnt = sum(temp[i] for i in [0, 8, 9, 17, 18, 26] + list(range(27, 34)))
    chanta_potential = yaochu_cnt / 14.0
    
    # 【新設7】三色ポテンシャル
    sanshoku_score = 0
    for num in range(9):
        c = (1 if temp[num]>0 else 0) + (1 if temp[9+num]>0 else 0) + (1 if temp[18+num]>0 else 0)
        if c >= 2: sanshoku_score += (c - 1)
    sanshoku_potential = min(1.0, sanshoku_score / 3.0)
    
    # 【新設8】確定役フラグ (役牌トイツ, 完全タンヤオ, 濃厚ホンイツ)
    has_yaku = 1.0 if (yakuhai_pairs > 0 or yaochu_cnt == 0 or honitsu_potential >= 0.85) else 0.0
    
    return {
        "s_before": float(s_before), "is_imp": is_improved, "is_ryo": is_ryoukei, "is_gu": is_gukei, "free": float(freedom),
        "bad_wait": bad_wait_after_chi, "menzen_loss": menzen_value_loss, "dora_cnt": float(dora_count), 
        "yaku_pairs": float(yakuhai_pairs), "honitsu_pot": honitsu_potential, "chanta_pot": chanta_potential, 
        "sanshoku_pot": sanshoku_potential, "has_yaku": has_yaku
    }

def get_dora_from_indicator(indicator_34):
    if indicator_34 < 27: return (indicator_34 // 9) * 9 + (indicator_34 % 9 + 1) % 9
    elif indicator_34 < 31: return 27 + (indicator_34 - 27 + 1) % 4
    else: return 31 + (indicator_34 - 31 + 1) % 3

# =========================================
# 🛡️ 2. 大規模抽出処理
# =========================================
class GlobalReplayTrackerChi:
    def __init__(self): self.is_broken = False
    def apply_event(self, event):
        e_type = event["type"]
        if e_type == "INIT":
            self.is_broken = False 
            self.hands_136 = {i: event["hands"][i].copy() for i in range(4)}
            self.dora_indicators = [event["dora_indicator"]]
            self.bakaze, self.oya = event["bakaze"], event["oya"]
            self.riichi_declared = {i: False for i in range(4)}
            self.discards_136 = {i: [] for i in range(4)}
        elif not self.is_broken:
            if e_type == "DRAW": self.hands_136[event["seat"]].append(event["tile_136"])
            elif e_type == "DISCARD":
                if event["tile_136"] in self.hands_136[event["seat"]]:
                    self.hands_136[event["seat"]].remove(event["tile_136"])
                    self.discards_136[event["seat"]].append(event["tile_136"])
                else: self.is_broken = True
            elif e_type == "REACH" and event["step"] == 1: self.riichi_declared[event["seat"]] = True
            elif e_type == "CALL": self.is_broken = True

def extract_chi_dataset(xml_string, log_id="unknown"):
    root = ET.fromstring(xml_string)
    events = []
    for node in root:
        tag = node.tag
        if tag == "INIT":
            seed = node.attrib.get("seed", "0,0,0,0,0,0").split(",")
            events.append({"type": "INIT", "bakaze": int(seed[0]) // 4, "dora_indicator": int(seed[5]), "oya": int(node.attrib.get("oya", "0")), "hands": {i: [int(x) for x in node.attrib.get(f"hai{i}", "").split(",") if x] for i in range(4)}})
        elif tag[0] in 'TUVW' and tag[1:].isdigit(): events.append({"type": "DRAW", "seat": 'TUVW'.index(tag[0]), "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
        elif tag[0] in 'DEFGdefg' and tag[1:].isdigit(): events.append({"type": "DISCARD", "seat": 'DEFGdefg'.index(tag[0].upper()), "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
        elif tag == "REACH": events.append({"type": "REACH", "seat": int(node.attrib.get("who")), "step": int(node.attrib.get("step"))})
        elif tag == "N": events.append({"type": "CALL", "seat": int(node.attrib.get("who")), "m": int(node.attrib.get("m"))})
            
    tracker = GlobalReplayTrackerChi()
    dataset_records = []
    for i, current_event in enumerate(events):
        tracker.apply_event(current_event)
        if tracker.is_broken: continue
        if current_event["type"] == "DISCARD":
            discard_seat = current_event["seat"]
            discard_tile_34 = current_event["tile_34"]
            for my_seat in range(4):
                if my_seat == discard_seat: continue
                if (discard_seat - my_seat + 4) % 4 == 3 and discard_tile_34 < 27:
                    my_hand_34 = [t // 4 for t in tracker.hands_136[my_seat]]
                    hand_counts = [my_hand_34.count(idx) for idx in range(34)]
                    t, num = discard_tile_34, discard_tile_34 % 9
                    chi_left = (num <= 6 and hand_counts[t+1] > 0 and hand_counts[t+2] > 0)
                    chi_mid  = (1 <= num <= 7 and hand_counts[t-1] > 0 and hand_counts[t+1] > 0)
                    chi_right= (num >= 2 and hand_counts[t-2] > 0 and hand_counts[t-1] > 0)
                    
                    if chi_left or chi_mid or chi_right:
                        label_chi = 0
                        for j in range(i + 1, len(events)):
                            next_ev = events[j]
                            if next_ev["type"] in ["DRAW", "DISCARD", "CALL"]:
                                if next_ev["type"] == "CALL" and next_ev["seat"] == my_seat and (next_ev["m"] & 0x0004) != 0: label_chi = 1
                                break
                        state = MahjongStateV5()
                        for t_136 in tracker.hands_136[my_seat]: state.add_tile(0, t_136)
                        for pov in range(4):
                            for t_136 in tracker.discards_136[(my_seat + pov) % 4]: state.discards[pov].append(t_136 // 4)
                            state.riichi_declared[pov] = tracker.riichi_declared[(my_seat + pov) % 4]
                        
                        dora_34_list = [get_dora_from_indicator(d // 4) for d in tracker.dora_indicators]
                        jikaze = (my_seat - tracker.oya + 4) % 4
                        
                        # 🌟 打点-aware 新特徴量の計算！
                        feats = evaluate_chi_features_v2(hand_counts, discard_tile_34, tracker.bakaze, jikaze, dora_34_list)
                        
                        dataset_records.append({
                            "tensor": state.to_tensor(skip_logic=True),
                            "target_tile_34": discard_tile_34,
                            "chi_mask": [int(chi_left), int(chi_mid), int(chi_right)],
                            "meta_is_dora": (discard_tile_34 in dora_34_list),
                            "meta_is_defense": any(state.riichi_declared[pov] for pov in range(1,4)),
                            "meta_turn": len(tracker.discards_136[my_seat]),
                            "feats": feats, # まとめて格納
                            "label_chi": label_chi
                        })
    return dataset_records

print(f"📁 {LOGS_DIR} からファイルを読み込みます...")
log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.gz')]
log_ids = []

for filename in log_files:
    with gzip.open(os.path.join(LOGS_DIR, filename), 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r'log=([\w-]+)', line)
            if match: log_ids.append(match.group(1))

log_ids = list(set(log_ids))
print(f"🎯 合計 {len(log_ids)} 件の牌譜リンクを発見！ 新特徴量付き抽出を開始します。")

all_records = []
for i, log_id in enumerate(log_ids):
    try:
        req = urllib.request.Request(f"https://tenhou.net/0/log/?{log_id}", headers={'User-Agent': 'Mozilla/5.0'})
        xml_string = urllib.request.urlopen(req).read().decode('utf-8')
        all_records.extend(extract_chi_dataset(xml_string, log_id))
        if (i + 1) % 5 == 0: 
            print(f"🔄 {i + 1}/{len(log_ids)}件処理... 現在: {len(all_records)} 件")
        if len(all_records) >= TARGET_RECORDS: break
        time.sleep(0.5)
    except: continue

print(f"\n💾 データを保存中... ({OUTPUT_FILE})")
with open(OUTPUT_FILE, 'wb') as f: pickle.dump(all_records, f)

# =========================================
# 3. 特徴量構築 & Dataset (Aux 52次元へ拡張！)
# =========================================
X_tensor, X_aux, y, meta_val = [], [], [], []
for r in all_records:
    X_tensor.append(r["tensor"])
    target_onehot = np.zeros(34, dtype=np.float32)
    target_onehot[r["target_tile_34"]] = 1.0
    f = r["feats"]
    
    # Target(34) + Mask(3) + 既存/新設(15) = 52次元
    aux = np.concatenate([
        target_onehot,
        np.array(r["chi_mask"], dtype=np.float32), 
        np.array([
            1.0 if r["meta_is_dora"] else 0.0, f["s_before"], f["is_imp"], f["is_ryo"], f["is_gu"], f["free"],
            1.0 if r["meta_is_defense"] else 0.0,
            f["bad_wait"], f["menzen_loss"], f["dora_cnt"], f["yaku_pairs"], f["honitsu_pot"], 
            f["chanta_pot"], f["sanshoku_pot"], f["has_yaku"]
        ], dtype=np.float32)
    ])
    X_aux.append(aux)
    y.append(np.float32(r["label_chi"]))

X_t, X_a, y = np.array(X_tensor, dtype=np.float32), np.array(X_aux, dtype=np.float32), np.array(y).reshape(-1, 1)
indices = np.arange(len(y))
X_t_tr, X_t_val, X_a_tr, X_a_val, y_tr, y_val, idx_tr, idx_val = train_test_split(X_t, X_a, y, indices, test_size=0.2, random_state=42, stratify=y)
meta_val = [all_records[i] for i in idx_val]

class ChiCNNDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

train_loader = DataLoader(ChiCNNDataset(X_t_tr, X_a_tr, y_tr), batch_size=256, shuffle=True)
val_loader = DataLoader(ChiCNNDataset(X_t_val, X_a_val, y_val), batch_size=256, shuffle=False)

# =========================================
# 4. CNN-A モデル定義と本番学習 (入力52次元)
# =========================================
class ChiCNN_A_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        # 🌟 入力次元を 64 + 44 から 64 + 52 に拡張！
        self.mlp = nn.Sequential(nn.Linear(64 + 52, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChiCNN_A_v2().to(device)

dynamic_pos_weight = float((len(y_tr)-y_tr.sum())/y_tr.sum())
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([dynamic_pos_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n🚀 【打点-aware】CNN-A 本番学習開始！ (Train: {len(y_tr)}件, Valid: {len(y_val)}件)")
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
    if val_loss < best_loss: best_loss, best_wts, patience_cnt = val_loss, copy.deepcopy(model.state_dict()), 0
    elif (patience_cnt := patience_cnt + 1) >= 5: 
        print("🛑 Early stopping triggered.")
        break

model.load_state_dict(best_wts)

# =========================================
# 5. 評価発表 (レビューアー重点評価軸)
# =========================================
def find_best_threshold(y_true, y_prob):
    if len(np.unique(y_true)) < 2: return 0.5, 0.0
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5, f1_scores[best_idx]

def p_metric(y_t, y_p, name):
    if len(y_t) == 0: return
    p = precision_score(y_t, y_p, zero_division=0)
    r = recall_score(y_t, y_p, zero_division=0)
    f = f1_score(y_t, y_p, zero_division=0)
    print(f"[{name}] (n={len(y_t)}) Precision: {p*100:.2f}% | Recall: {r*100:.2f}% | F1: {f*100:.2f}%")

model.eval()
with torch.no_grad():
    cnn_probs = torch.sigmoid(model(torch.tensor(X_t_val).to(device), torch.tensor(X_a_val).to(device))).cpu().numpy().flatten()

pr_auc = average_precision_score(y_val, cnn_probs)
best_thresh, _ = find_best_threshold(y_val, cnn_probs)
pred_opt = (cnn_probs >= best_thresh).astype(int)

print("\n" + "="*60)
print("👑 フェーズ5B-2: 打点-aware CNN-A 結果発表")
print("="*60)
print(f"■ 【全体指標】 PR-AUC: {pr_auc:.4f} (前回 0.2327からの向上に注目！)")
p_metric(y_val, pred_opt, f"全体 (最適閾値 {best_thresh:.3f})")

print("\n🔍 【レビューアー重点・内訳分析】")
idx_dora = np.array([i for i, m in enumerate(meta_val) if m["meta_is_dora"]])
idx_non_dora = np.array([i for i, m in enumerate(meta_val) if not m["meta_is_dora"]])
p_metric(y_val[idx_dora], pred_opt[idx_dora], "ドラ絡みチー (FN減/Recall増に期待)")
p_metric(y_val[idx_non_dora], pred_opt[idx_non_dora], "非ドラチー　 (FP抑止/Precision維持に期待)")

idx_s1 = np.array([i for i, m in enumerate(meta_val) if m["feats"]["s_before"] == 1])
p_metric(y_val[idx_s1], pred_opt[idx_s1], "1シャンテン (役の概念でF1向上に期待)")

print("="*60)