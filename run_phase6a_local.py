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

# 自作モジュール
from mahjong_engine import tile_names, MahjongStateV5, calculate_shanten

# =========================================
# 0. 設定と環境準備
# =========================================
TARGET_RECORDS = 20000
LOGS_DIR = "./logs"
OUTPUT_FILE = "./dataset_riichi_phase6a.pkl"

if not os.path.exists(LOGS_DIR):
    print(f"❌ エラー: '{LOGS_DIR}' が見つかりません。")
    sys.exit()

# 🚀 【超高速化】シャンテン計算のキャッシュ（メモ化）
_shanten_cache = {}
def get_shanten_cached(counts):
    t = tuple(counts)
    if t not in _shanten_cache:
        _shanten_cache[t] = calculate_shanten(list(counts))
    return _shanten_cache[t]

# =========================================
# 🛡️ 1. リーチ用特徴量計算ヘルパー
# =========================================
def get_waits(hand_counts):
    waits = []
    for i in range(34):
        if hand_counts[i] < 4:
            hand_counts[i] += 1
            # キャッシュ版を使って爆速化
            if get_shanten_cached(hand_counts) == -1:
                waits.append(i)
            hand_counts[i] -= 1
    return waits

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
# 🛡️ 2. リーチ専用データ抽出器
# =========================================
def extract_riichi_dataset(xml_string, log_id="unknown"):
    root = ET.fromstring(xml_string)
    events = []
    for node in root:
        tag = node.tag
        if tag == "INIT":
            seed = node.attrib.get("seed", "0,0,0,0,0,0").split(",")
            scores = [int(x) for x in node.attrib.get("ten", "250,250,250,250").split(",")]
            events.append({"type": "INIT", "bakaze": int(seed[0]) // 4, "kyoku": int(seed[0]) % 4,
                           "oya": int(node.attrib.get("oya", "0")), "scores": scores,
                           "dora_indicator": int(seed[5]),
                           "hands": {i: [int(x) for x in node.attrib.get(f"hai{i}", "").split(",") if x] for i in range(4)}})
        elif tag[0] in 'TUVW' and tag[1:].isdigit(): events.append({"type": "DRAW", "seat": 'TUVW'.index(tag[0]), "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
        elif tag[0] in 'DEFGdefg' and tag[1:].isdigit(): events.append({"type": "DISCARD", "seat": 'DEFGdefg'.index(tag[0].upper()), "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
        elif tag == "REACH": events.append({"type": "REACH", "seat": int(node.attrib.get("who")), "step": int(node.attrib.get("step"))})
        elif tag == "N": events.append({"type": "CALL", "seat": int(node.attrib.get("who"))})
            
    hands_136 = {i: [] for i in range(4)}
    discards_136 = {i: [] for i in range(4)}
    is_menzen = {i: True for i in range(4)}
    is_riichi = {i: False for i in range(4)}
    dora_inds = []
    bakaze = oya = 0
    scores = [250, 250, 250, 250]
    
    dataset_records = []
    
    for i, ev in enumerate(events):
        if ev["type"] == "INIT":
            hands_136 = copy.deepcopy(ev["hands"])
            discards_136 = {s: [] for s in range(4)}
            is_menzen = {s: True for s in range(4)}
            is_riichi = {s: False for s in range(4)}
            dora_inds = [ev["dora_indicator"]]
            bakaze, oya, scores = ev["bakaze"], ev["oya"], ev["scores"]
            
        elif ev["type"] == "CALL":
            is_menzen[ev["seat"]] = False
            
        elif ev["type"] == "DRAW":
            seat = ev["seat"]
            hands_136[seat].append(ev["tile_136"])
            
            if is_menzen[seat] and not is_riichi[seat]:
                hand_34 = [t // 4 for t in hands_136[seat]]
                counts_14 = [hand_34.count(t) for t in range(34)]
                
                # 🛡️ 安全装置：暗槓などで14枚じゃない場合はスキップ！
                if sum(counts_14) == 14:
                    # キャッシュ版を使って爆速判定
                    if get_shanten_cached(counts_14) <= 0:
                        label_riichi = 0
                        actual_discard_34 = None
                        for j in range(i + 1, len(events)):
                            if events[j]["type"] == "REACH" and events[j]["seat"] == seat and events[j]["step"] == 1:
                                label_riichi = 1
                            elif events[j]["type"] == "DISCARD" and events[j]["seat"] == seat:
                                actual_discard_34 = events[j]["tile_34"]
                                break
                                
                        if actual_discard_34 is not None:
                            counts_13 = list(counts_14)
                            counts_13[actual_discard_34] -= 1
                            
                            waits = get_waits(counts_13)
                            enemy_riichi = any(is_riichi[s] for s in range(4) if s != seat)
                            turn = len(discards_136[seat])
                            jikaze = (seat - oya + 4) % 4
                            dora_34_list = [get_dora_from_indicator(d // 4) for d in dora_inds]
                            dora_count = sum(counts_13[d] for d in dora_34_list)
                            dama_legal = calc_dama_legal_proxy(counts_13, bakaze, jikaze)
                            
                            wait_count = sum((4 - counts_13[w]) for w in waits)
                            is_ryankei = (wait_count >= 5 or len(waits) >= 2)
                            
                            my_score = scores[seat]
                            my_rank = sum(1 for s in scores if s > my_score) + 1
                            point_diff = max(scores) - my_score
                            
                            state = MahjongStateV5()
                            for t_136 in hands_136[seat]: state.add_tile(0, t_136)
                            for pov in range(4):
                                for t_136 in discards_136[(seat + pov) % 4]: state.discards[pov].append(t_136 // 4)
                                state.riichi_declared[pov] = is_riichi[(seat + pov) % 4]
                            
                            dataset_records.append({
                                "tensor": state.to_tensor(skip_logic=True),
                                "label_riichi": label_riichi,
                                "actual_discard_34": actual_discard_34,
                                "wait_count": wait_count,
                                "is_ryankei": is_ryankei,
                                "meta_dama_legal": dama_legal,
                                "meta_dora_count": dora_count,
                                "meta_turn": turn,
                                "meta_enemy_riichi": enemy_riichi,
                                "meta_my_rank": my_rank,
                                "meta_point_diff": point_diff,
                                "meta_kyoku": ev.get("kyoku", 0)
                            })
                        
        elif ev["type"] == "DISCARD":
            seat = ev["seat"]
            if ev["tile_136"] in hands_136[seat]:
                hands_136[seat].remove(ev["tile_136"])
                discards_136[seat].append(ev["tile_136"])
        elif ev["type"] == "REACH" and ev["step"] == 1:
            is_riichi[ev["seat"]] = True

    return dataset_records

print(f"📁 {LOGS_DIR} からファイルを読み込みます...")
log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.gz')]
all_records = []

for idx, filename in enumerate(log_files):
    log_ids = []
    with gzip.open(os.path.join(LOGS_DIR, filename), 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r'log=([\w-]+)', line)
            if match: log_ids.append(match.group(1))
    
    log_ids = list(set(log_ids))
    
    for log_id in log_ids:
        try:
            req = urllib.request.Request(f"https://tenhou.net/0/log/?{log_id}", headers={'User-Agent': 'Mozilla/5.0'})
            xml_string = urllib.request.urlopen(req).read().decode('utf-8')
            all_records.extend(extract_riichi_dataset(xml_string, log_id))
            if len(all_records) >= TARGET_RECORDS: break
            time.sleep(0.3)
        except Exception as e:
            print(f"  ⚠️ 牌譜 {log_id} でエラー発生: {e}")
            continue
            
    print(f"🔄 処理中... 現在: {len(all_records)} 件のテンパイ局面を抽出")
    if len(all_records) >= TARGET_RECORDS: break

print(f"\n💾 データを保存中... ({OUTPUT_FILE})")
with open(OUTPUT_FILE, 'wb') as f: pickle.dump(all_records, f)

# =========================================
# 📊 3. フェーズ6A：抽出レポート＆ルールベース検証
# =========================================
total = len(all_records)

if total == 0:
    print("❌ 抽出されたデータが0件でした。")
    sys.exit()

valid_records = [r for r in all_records if r["wait_count"] > 0]
valid_total = len(valid_records)

if valid_total == 0:
    print("❌ 有効なテンパイ局面が0件でした。")
    sys.exit()

riichi_count = sum(r["label_riichi"] for r in valid_records)
ryankei_count = sum(1 for r in valid_records if r["is_ryankei"])
enemy_riichi_count = sum(1 for r in valid_records if r["meta_enemy_riichi"])
high_score_dama_count = sum(1 for r in valid_records if r["label_riichi"]==0 and r["meta_dora_count"]>=2 and r["meta_dama_legal"])

print("\n" + "="*60)
print("👑 フェーズ6A: リーチ判断用データ 抽出レポート")
print("="*60)
print(f"■ 総抽出局面 (テンパイ時): {total} 件")
print(f"■ 有効局面 (テンパイ維持): {valid_total} 件 (※オリ・テンパイ崩しを除外)")
print(f"  - 実リーチ率      : {riichi_count / valid_total * 100:.1f} %")
print(f"  - 良形(両面等)率  : {ryankei_count / valid_total * 100:.1f} %")
print(f"  - 他家リーチ遭遇率: {enemy_riichi_count / valid_total * 100:.1f} %")
print(f"  - 高打点ダマ率    : {high_score_dama_count / max(1, valid_total - riichi_count) * 100:.1f} % (※ダマ選択中の割合)")

print("\n🔍 【フェーズ6B: ルールベースの輪郭確認】")
base1_acc = riichi_count / valid_total
base2_correct = sum(1 for r in valid_records if (r["is_ryankei"] and r["label_riichi"]==1) or (not r["is_ryankei"] and r["label_riichi"]==0))

base3_correct = 0
for r in valid_records:
    pred = 1
    if not r["is_ryankei"]:
        if r["meta_dora_count"] >= 2 and r["meta_dama_legal"]: pred = 0
        elif r["meta_enemy_riichi"]: pred = 0
    if pred == r["label_riichi"]: base3_correct += 1

print(f"  [Base 1] 全リーチ基準 一致率     : {base1_acc * 100:.1f} %")
print(f"  [Base 2] 良形リーチ基準 一致率   : {base2_correct / valid_total * 100:.1f} %")
print(f"  [Base 3] 良形＋打点/守備考慮 一致率: {base3_correct / valid_total * 100:.1f} %")
print("="*60)