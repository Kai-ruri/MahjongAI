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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from mahjong_engine import tile_names, MahjongStateV5, calculate_shanten

# =========================================
# 0. 設定と環境準備
# =========================================
TARGET_RECORDS = 20000
LOGS_DIR = "./logs"
OUTPUT_FILE = "./dataset_riichi_phase6e.pkl"

if not os.path.exists(LOGS_DIR):
    print(f"❌ エラー: '{LOGS_DIR}' が見つかりません。")
    sys.exit()

_shanten_cache = {}
def get_shanten_cached(counts):
    t = tuple(counts)
    if t not in _shanten_cache:
        _shanten_cache[t] = calculate_shanten(list(counts))
    return _shanten_cache[t]

# =========================================
# 🛡️ 1. 新規特徴量計算ヘルパー
# =========================================
def get_waits(hand_counts):
    waits = []
    for i in range(34):
        if hand_counts[i] < 4:
            hand_counts[i] += 1
            if get_shanten_cached(hand_counts) == -1: waits.append(i)
            hand_counts[i] -= 1
    return waits

def get_wait_features(waits, wait_count):
    # One-hot: [両面, シャンポン, 愚形(カンチャン/ペンチャン/単騎), 多面張]
    w_type = [0, 0, 0, 0]
    if len(waits) >= 3: w_type[3] = 1
    elif len(waits) == 2:
        w1, w2 = sorted(waits)
        if w1 < 27 and w2 < 27 and w1//9 == w2//9 and w2-w1 == 3: w_type[0] = 1
        else: w_type[1] = 1
    else: w_type[2] = 1

    # 枚数バケット: [1枚, 2枚, 3-4枚, 5枚以上]
    w_cnt = [0, 0, 0, 0]
    if wait_count <= 1: w_cnt[0] = 1
    elif wait_count == 2: w_cnt[1] = 1
    elif 3 <= wait_count <= 4: w_cnt[2] = 1
    else: w_cnt[3] = 1
    return w_type + w_cnt

def get_score_features(scores, seat, oya, kyoku):
    ooras = 1.0 if kyoku >= 7 else 0.0 # 南4局以降をオーラスと判定
    is_oya = 1.0 if seat == oya else 0.0
    my_score = scores[seat]
    my_rank = sum(1 for s in scores if s > my_score) + 1
    is_top = 1.0 if my_rank == 1 else 0.0
    diff_top = (max(scores) - my_score) / 10000.0
    diff_last = (my_score - min(scores)) / 10000.0
    return [ooras, is_oya, is_top, diff_top, diff_last]

def calc_dama_legal_proxy(hand_counts, bakaze, jikaze):
    yaochu = [0, 8, 9, 17, 18, 26] + list(range(27, 34))
    is_tanyao = all(hand_counts[t] == 0 for t in yaochu)
    yakuhai_tiles = [27 + bakaze, 27 + jikaze, 31, 32, 33]
    has_yakuhai = any(hand_counts[t] >= 2 for t in yakuhai_tiles)
    m, p, s, z = sum(hand_counts[0:9]), sum(hand_counts[9:18]), sum(hand_counts[18:27]), sum(hand_counts[27:34])
    is_honitsu = (max(m, p, s) + z) >= 13
    return is_tanyao or has_yakuhai or is_honitsu

def get_dora_from_indicator(indicator_34):
    if indicator_34 < 27: return (indicator_34 // 9) * 9 + (indicator_34 % 9 + 1) % 9
    elif indicator_34 < 31: return 27 + (indicator_34 - 27 + 1) % 4
    else: return 31 + (indicator_34 - 31 + 1) % 3

# =========================================
# 🛡️ 2. データ抽出器 (Phase 6E対応版)
# =========================================
def extract_riichi_dataset_v2(xml_string, log_id="unknown"):
    root = ET.fromstring(xml_string)
    events = []
    for node in root:
        tag = node.tag
        if tag == "INIT":
            seed = node.attrib.get("seed", "0,0,0,0,0,0").split(",")
            scores = [int(x) for x in node.attrib.get("ten", "250,250,250,250").split(",")]
            events.append({"type": "INIT", "bakaze": int(seed[0]) // 4, "kyoku": int(seed[0]) % 4 + (4 if "南" in xml_string else 0),
                           "oya": int(node.attrib.get("oya", "0")), "scores": scores,
                           "dora_indicator": int(seed[5]),
                           "hands": {i: [int(x) for x in node.attrib.get(f"hai{i}", "").split(",") if x] for i in range(4)}})
        elif tag[0] in 'TUVW' and tag[1:].isdigit(): events.append({"type": "DRAW", "seat": 'TUVW'.index(tag[0]), "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
        elif tag[0] in 'DEFGdefg' and tag[1:].isdigit(): events.append({"type": "DISCARD", "seat": 'DEFGdefg'.index(tag[0].upper()), "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
        elif tag == "REACH": events.append({"type": "REACH", "seat": int(node.attrib.get("who")), "step": int(node.attrib.get("step"))})
        elif tag == "N": events.append({"type": "CALL", "seat": int(node.attrib.get("who"))})
            
    hands_136 = {i: [] for i in range(4)}; discards_136 = {i: [] for i in range(4)}
    is_menzen = {i: True for i in range(4)}; is_riichi = {i: False for i in range(4)}
    dora_inds = []; bakaze = oya = kyoku = 0; scores = [250, 250, 250, 250]
    dataset_records = []
    
    for i, ev in enumerate(events):
        if ev["type"] == "INIT":
            hands_136 = copy.deepcopy(ev["hands"]); discards_136 = {s: [] for s in range(4)}
            is_menzen = {s: True for s in range(4)}; is_riichi = {s: False for s in range(4)}
            dora_inds = [ev["dora_indicator"]]
            bakaze, oya, scores, kyoku = ev["bakaze"], ev["oya"], ev["scores"], ev["kyoku"]
        elif ev["type"] == "CALL": is_menzen[ev["seat"]] = False
        elif ev["type"] == "DRAW":
            seat = ev["seat"]
            hands_136[seat].append(ev["tile_136"])
            if is_menzen[seat] and not is_riichi[seat]:
                hand_34 = [t // 4 for t in hands_136[seat]]
                counts_14 = [hand_34.count(t) for t in range(34)]
                if sum(counts_14) == 14 and get_shanten_cached(counts_14) <= 0:
                    label_riichi = 0; actual_discard_34 = None
                    for j in range(i + 1, len(events)):
                        if events[j]["type"] == "REACH" and events[j]["seat"] == seat and events[j]["step"] == 1: label_riichi = 1
                        elif events[j]["type"] == "DISCARD" and events[j]["seat"] == seat:
                            actual_discard_34 = events[j]["tile_34"]
                            break
                    if actual_discard_34 is not None:
                        counts_13 = list(counts_14)
                        counts_13[actual_discard_34] -= 1
                        waits = get_waits(counts_13)
                        
                        if len(waits) > 0:
                            wait_count = sum((4 - counts_13[w]) for w in waits)
                            dora_34_list = [get_dora_from_indicator(d // 4) for d in dora_inds]
                            dora_count = sum(counts_13[d] for d in dora_34_list)
                            dama_legal = calc_dama_legal_proxy(counts_13, bakaze, (seat - oya + 4) % 4)
                            
                            # 🌟 新特徴量の計算
                            wait_feats = get_wait_features(waits, wait_count)
                            score_feats = get_score_features(scores, seat, oya, kyoku)
                            
                            state = MahjongStateV5()
                            for t_136 in hands_136[seat]: state.add_tile(0, t_136)
                            for pov in range(4):
                                for t_136 in discards_136[(seat + pov) % 4]: state.discards[pov].append(t_136 // 4)
                                state.riichi_declared[pov] = is_riichi[(seat + pov) % 4]
                            
                            dataset_records.append({
                                "tensor": state.to_tensor(skip_logic=True),
                                "label_riichi": label_riichi,
                                "base_feats": [wait_count, 1.0 if (wait_count >= 5 or len(waits) >= 2) else 0.0, len(discards_136[seat]), 
                                               1.0 if any(is_riichi[s] for s in range(4) if s != seat) else 0.0,
                                               dora_count, 1.0 if (dora_count >= 2 and dama_legal) else 0.0, 1.0 if dama_legal else 0.0],
                                "wait_feats": wait_feats,
                                "score_feats": score_feats
                            })
        elif ev["type"] == "DISCARD":
            seat = ev["seat"]
            if ev["tile_136"] in hands_136[seat]:
                hands_136[seat].remove(ev["tile_136"])
                discards_136[seat].append(ev["tile_136"])
        elif ev["type"] == "REACH" and ev["step"] == 1: is_riichi[ev["seat"]] = True
    return dataset_records

if not os.path.exists(OUTPUT_FILE):
    print(f"📁 {LOGS_DIR} から新特徴量を含めてデータを抽出します...")
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.gz')]
    all_records = []
    for idx, filename in enumerate(log_files):
        log_ids = []
        with gzip.open(os.path.join(LOGS_DIR, filename), 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(r'log=([\w-]+)', line)
                if match: log_ids.append(match.group(1))
        for log_id in list(set(log_ids)):
            try:
                req = urllib.request.Request(f"https://tenhou.net/0/log/?{log_id}", headers={'User-Agent': 'Mozilla/5.0'})
                xml_string = urllib.request.urlopen(req).read().decode('utf-8')
                all_records.extend(extract_riichi_dataset_v2(xml_string, log_id))
                if len(all_records) >= TARGET_RECORDS: break
                time.sleep(0.3)
            except Exception as e: continue
        print(f"🔄 処理中... 現在: {len(all_records)} 件")
        if len(all_records) >= TARGET_RECORDS: break
    with open(OUTPUT_FILE, 'wb') as f: pickle.dump(all_records, f)
else:
    print(f"📂 既存のデータセット {OUTPUT_FILE} を読み込みます...")
    with open(OUTPUT_FILE, 'rb') as f: all_records = pickle.load(f)

print(f"📊 有効データ準備完了: {len(all_records)} 件")

# =========================================
# 3. アブレーション・スタディ準備
# =========================================
X_tensor, y = [], []
X_base, X_e1, X_e2, X_e3 = [], [], [], []

for r in all_records:
    X_tensor.append(r["tensor"])
    y.append(np.float32(r["label_riichi"]))
    
    b_feat = np.array(r["base_feats"], dtype=np.float32) # 7次元
    w_feat = np.array(r["wait_feats"], dtype=np.float32) # 8次元
    s_feat = np.array(r["score_feats"], dtype=np.float32) # 5次元
    
    X_base.append(b_feat)
    X_e1.append(np.concatenate([b_feat, w_feat]))
    X_e2.append(np.concatenate([b_feat, s_feat]))
    X_e3.append(np.concatenate([b_feat, w_feat, s_feat]))

X_t = np.array(X_tensor, dtype=np.float32)
y = np.array(y).reshape(-1, 1)

indices = np.arange(len(y))
idx_tr, idx_val = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
meta_val = [all_records[i] for i in idx_val]

class RiichiCNNDataset(Dataset):
    def __init__(self, t, a, l): self.t, self.a, self.l = torch.tensor(t), torch.tensor(a), torch.tensor(l)
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.t[i], self.a[i], self.l[i]

# =========================================
# 4. モデル訓練と評価エンジン
# =========================================
class RiichiCNN_Dynamic(nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + aux_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

def train_and_evaluate(name, X_aux):
    X_a_tr = np.array([X_aux[i] for i in idx_tr])
    X_a_val = np.array([X_aux[i] for i in idx_val])
    y_tr_split = y[idx_tr]; y_val_split = y[idx_val]
    
    train_loader = DataLoader(RiichiCNNDataset(X_t[idx_tr], X_a_tr, y_tr_split), batch_size=256, shuffle=True)
    val_loader = DataLoader(RiichiCNNDataset(X_t[idx_val], X_a_val, y_val_split), batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiichiCNN_Dynamic(aux_dim=X_a_tr.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    best_loss, best_wts, patience = float('inf'), None, 0
    for epoch in range(15): # アブレーションのため高速化(15Epoch)
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
    
    best_acc, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.3, 0.7, 0.02):
        acc = accuracy_score(y_val_split, (cnn_probs >= thresh).astype(int))
        if acc > best_acc: best_acc, best_thresh = acc, thresh
        
    pred_opt = (cnn_probs >= best_thresh).astype(int)
    roc = roc_auc_score(y_val_split, cnn_probs)
    f1 = f1_score(y_val_split, pred_opt, zero_division=0)
    
    # サブグループ精度計算
    def sub_acc(condition_fn):
        idx = [i for i, m in enumerate(meta_val) if condition_fn(m)]
        if not idx: return 0.0
        return accuracy_score(y_val_split[idx], pred_opt[idx]) * 100

    def sub_riichi_rate(condition_fn):
        idx = [i for i, m in enumerate(meta_val) if condition_fn(m)]
        if not idx: return 0.0
        return sum(pred_opt[idx]) / len(idx) * 100

    return {
        "name": name, "acc": best_acc*100, "roc": roc, "f1": f1*100, "thresh": best_thresh,
        "acc_gukei": sub_acc(lambda m: m["base_feats"][1] == 0.0), # 良形フラグ==0
        "acc_enemy": sub_acc(lambda m: m["base_feats"][3] == 1.0), # 他家リーチ==1
        "acc_high_dama": sub_acc(lambda m: m["base_feats"][5] == 1.0), # 高打点ダマ==1
        "acc_late": sub_acc(lambda m: m["base_feats"][2] >= 12), # 巡目>=12
        "acc_ooras": sub_acc(lambda m: m["score_feats"][0] == 1.0), # オーラス
        "acc_oya": sub_acc(lambda m: m["score_feats"][1] == 1.0), # 親
        "rate_high_score_riichi": sub_riichi_rate(lambda m: m["base_feats"][5] == 1.0), # 満貫級のリーチ率
        "rate_oya_riichi": sub_riichi_rate(lambda m: m["score_feats"][1] == 1.0) # 親のリーチ率
    }

print("\n🚀 【フェーズ6E】アブレーション・スタディ開始！")
results = []
results.append(train_and_evaluate("Base (6C仕様)", X_base))
results.append(train_and_evaluate("E1 (待ち種別追加)", X_e1))
results.append(train_and_evaluate("E2 (点数状況追加)", X_e2))
results.append(train_and_evaluate("E3 (両方追加・完全体)", X_e3))

print("\n" + "="*80)
print("👑 フェーズ6E: アブレーション・スタディ 結果発表")
print("="*80)
print(f"{'モデル':<15} | {'Acc (%)':<7} | {'ROC-AUC':<7} | {'F1 (%)':<7} | {'愚形Acc':<7} | {'他家立直':<7} | {'高打点ダマ':<7} | {'オーラス':<7}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<15} | {r['acc']:>7.2f} | {r['roc']:>7.4f} | {r['f1']:>7.2f} | {r['acc_gukei']:>7.2f} | {r['acc_enemy']:>7.2f} | {r['acc_high_dama']:>7.2f} | {r['acc_ooras']:>7.2f}")

print("\n🚨 【過剰ダマ警戒チェック (ダマ偏重になっていないか？)】")
for r in results:
    print(f"[{r['name']}] 高打点時リーチ率: {r['rate_high_score_riichi']:.1f}% | 親リーチ率: {r['rate_oya_riichi']:.1f}%")
print("="*80)