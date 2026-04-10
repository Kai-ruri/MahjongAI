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
TARGET_RECORDS = 20000  # 輪郭を見るために2万件抽出
LOGS_DIR = "./logs"
OUTPUT_FILE = "./dataset_oshibiki_phase7a.pkl"

if not os.path.exists(LOGS_DIR):
    print(f"❌ エラー: '{LOGS_DIR}' が見つかりません。")
    sys.exit()

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
# 🛡️ 1. 押し引き専用データ抽出器
# =========================================
def extract_oshibiki_dataset(xml_string, log_id="unknown"):
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
            
    hands_136 = {i: [] for i in range(4)}
    discards_136 = {i: [] for i in range(4)}
    is_riichi = {i: False for i in range(4)}
    dora_inds = []
    bakaze = oya = kyoku = 0
    scores = [250, 250, 250, 250]
    
    dataset_records = []
    
    for i, ev in enumerate(events):
        if ev["type"] == "INIT":
            hands_136 = copy.deepcopy(ev["hands"])
            discards_136 = {s: [] for s in range(4)}
            is_riichi = {s: False for s in range(4)}
            dora_inds = [ev["dora_indicator"]]
            bakaze, oya, scores, kyoku = ev["bakaze"], ev["oya"], ev["scores"], ev["kyoku"]
            
        elif ev["type"] == "DRAW":
            hands_136[ev["seat"]].append(ev["tile_136"])
            
        elif ev["type"] == "DISCARD":
            seat = ev["seat"]
            discard_tile_136 = ev["tile_136"]
            discard_tile_34 = discard_tile_136 // 4
            
            # 【抽出トリガー】自分は未和了(未リーチ)、かつ他家にリーチ者がいる局面の「打牌」
            enemy_riichi_seats = [s for s in range(4) if s != seat and is_riichi[s]]
            
            if not is_riichi[seat] and len(enemy_riichi_seats) > 0:
                hand_34 = [t // 4 for t in hands_136[seat]]
                counts_14 = [hand_34.count(t) for t in range(34)]
                
                # 暗槓等イレギュラー除外
                if sum(counts_14) == 14:
                    shanten_before = get_shanten_cached(counts_14)
                    
                    counts_13 = list(counts_14)
                    counts_13[discard_tile_34] -= 1
                    shanten_after = get_shanten_cached(counts_13)
                    
                    # 🌟 危険度判定ロジック (0:オリ寄り, 1:中間, 2:押し寄り)
                    genbutsu_sets = {s: set(t // 4 for t in discards_136[s]) for s in enemy_riichi_seats}
                    is_genbutsu_all = all(discard_tile_34 in genbutsu_sets[s] for s in enemy_riichi_seats)
                    
                    if is_genbutsu_all:
                        label = 0  # 全員に対する現物 -> 完全オリ
                    elif discard_tile_34 >= 27:
                        label = 1  # 字牌(現物以外) -> 中間
                    else:
                        is_musuji_to_anyone = False
                        for s in enemy_riichi_seats:
                            genbutsu = genbutsu_sets[s]
                            if discard_tile_34 in genbutsu: continue
                            
                            # 簡易スジ判定
                            num = discard_tile_34 % 9
                            color = discard_tile_34 // 9
                            suji = False
                            if num == 0 and (color*9 + 3) in genbutsu: suji = True
                            elif num == 1 and (color*9 + 4) in genbutsu: suji = True
                            elif num == 2 and (color*9 + 5) in genbutsu: suji = True
                            elif num == 3 and ((color*9 + 0) in genbutsu or (color*9 + 6) in genbutsu): suji = True
                            elif num == 4 and ((color*9 + 1) in genbutsu or (color*9 + 7) in genbutsu): suji = True
                            elif num == 5 and ((color*9 + 2) in genbutsu or (color*9 + 8) in genbutsu): suji = True
                            elif num == 6 and (color*9 + 3) in genbutsu: suji = True
                            elif num == 7 and (color*9 + 4) in genbutsu: suji = True
                            elif num == 8 and (color*9 + 5) in genbutsu: suji = True
                            
                            if suji: continue
                            is_musuji_to_anyone = True # 現物でもスジでもない -> 無筋
                            break
                        
                        if is_musuji_to_anyone: label = 2  # 1人にでも無筋 -> 押し寄り
                        else: label = 1                    # スジ・ワンチャンス等 -> 中間
                    
                    # メタデータ計算
                    turn = len(discards_136[seat])
                    my_score = scores[seat]
                    dora_34_list = [get_dora_from_indicator(d // 4) for d in dora_inds]
                    dora_count = sum(counts_14[d] for d in dora_34_list)
                    
                    state = MahjongStateV5()
                    for t_136 in hands_136[seat]: state.add_tile(0, t_136)
                    for pov in range(4):
                        for t_136 in discards_136[(seat + pov) % 4]: state.discards[pov].append(t_136 // 4)
                        state.riichi_declared[pov] = is_riichi[(seat + pov) % 4]

                    dataset_records.append({
                        "tensor": state.to_tensor(skip_logic=True),
                        "label_oshiki": label,
                        "discard_34": discard_tile_34,
                        "shanten_before": shanten_before,
                        "is_maintained": (shanten_after <= shanten_before), # テンパイ維持/シャンテン維持か？
                        "enemy_riichi_count": len(enemy_riichi_seats),
                        "turn": turn,
                        "is_oya": (seat == oya),
                        "my_rank": sum(1 for s in scores if s > my_score) + 1,
                        "diff_top": max(scores) - my_score,
                        "diff_last": my_score - min(scores),
                        "ooras": (kyoku >= 7),
                        "dora_count": dora_count
                    })
            
            # 打牌を河に反映
            if discard_tile_136 in hands_136[seat]:
                hands_136[seat].remove(discard_tile_136)
                discards_136[seat].append(discard_tile_136)
                
        elif ev["type"] == "REACH" and ev["step"] == 1:
            is_riichi[ev["seat"]] = True

    return dataset_records

print(f"📁 {LOGS_DIR} から他家リーチ局面を抽出します...")
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
            all_records.extend(extract_oshibiki_dataset(xml_string, log_id))
            if len(all_records) >= TARGET_RECORDS: break
            time.sleep(0.3)
        except Exception as e: continue
        
    print(f"🔄 処理中... 現在: {len(all_records)} 件")
    if len(all_records) >= TARGET_RECORDS: break

print(f"\n💾 データを保存中... ({OUTPUT_FILE})")
with open(OUTPUT_FILE, 'wb') as f: pickle.dump(all_records, f)

# =========================================
# 📊 3. フェーズ7A：押し引き抽出レポート
# =========================================
total = len(all_records)
if total == 0:
    print("❌ 抽出されたデータが0件でした。")
    sys.exit()

l0 = sum(1 for r in all_records if r["label_oshiki"] == 0)
l1 = sum(1 for r in all_records if r["label_oshiki"] == 1)
l2 = sum(1 for r in all_records if r["label_oshiki"] == 2)

s_tenpai = sum(1 for r in all_records if r["shanten_before"] <= 0)
s_1 = sum(1 for r in all_records if r["shanten_before"] == 1)
s_2 = sum(1 for r in all_records if r["shanten_before"] >= 2)

t_early = sum(1 for r in all_records if r["turn"] <= 6)
t_mid = sum(1 for r in all_records if 7 <= r["turn"] <= 11)
t_late = sum(1 for r in all_records if r["turn"] >= 12)

oya_cnt = sum(1 for r in all_records if r["is_oya"])
ko_cnt = total - oya_cnt

r1_cnt = sum(1 for r in all_records if r["enemy_riichi_count"] == 1)
r2_cnt = sum(1 for r in all_records if r["enemy_riichi_count"] >= 2)

print("\n" + "="*60)
print("👑 フェーズ7A: 押し引き近似ラベル 抽出レポート")
print("="*60)
print(f"■ 総抽出件数: {total} 件 (他家リーチあり、自身未和了の打牌局面)")
print(f"■ ラベル分布")
print(f"  - オリ寄り(0) [現物等]  : {l0} 件 ({l0/total*100:.1f}%)")
print(f"  - 中間    (1) [スジ等]  : {l1} 件 ({l1/total*100:.1f}%)")
print(f"  - 押し寄り(2) [無筋]    : {l2} 件 ({l2/total*100:.1f}%)")

print("\n🔍 【状況別内訳】")
print(f"  [シャンテン別]")
print(f"    テンパイ: {s_tenpai/total*100:.1f}% | 1シャンテン: {s_1/total*100:.1f}% | 2シャンテン+: {s_2/total*100:.1f}%")
print(f"  [巡目別]")
print(f"    序盤(0-6): {t_early/total*100:.1f}% | 中盤(7-11): {t_mid/total*100:.1f}% | 終盤(12-): {t_late/total*100:.1f}%")
print(f"  [立場別]")
print(f"    親: {oya_cnt/total*100:.1f}% | 子: {ko_cnt/total*100:.1f}%")
print(f"  [リーチ者数]")
print(f"    1人: {r1_cnt/total*100:.1f}% | 2人以上: {r2_cnt/total*100:.1f}%")
print("="*60)