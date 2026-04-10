import os
import re
import gzip
import urllib.request
import xml.etree.ElementTree as ET
import pickle
import numpy as np

# 自作モジュール
from mahjong_engine import MahjongStateV5

LOGS_DIR = "./logs"
OUTPUT_FILE = "./dataset_ev_phase9a.pkl"
TARGET_KYOKU_COUNT = 5000  # まずは5000局分のEVデータを抽出

def extract_ev_dataset():
    print(f"📁 {LOGS_DIR} からEV（局収支）データを抽出します...")
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
            
            # 1局の中の一時保存用リスト
            current_kyoku_states = {0: [], 1: [], 2: [], 3: []}
            scores = [25000, 25000, 25000, 25000]
            turn_counts = [0, 0, 0, 0]
            oya_seat = 0
            
            for node in root:
                tag = node.tag
                if tag == "INIT":
                    current_kyoku_states = {0: [], 1: [], 2: [], 3: []}
                    turn_counts = [0, 0, 0, 0]
                    scores = [int(x)*100 for x in node.attrib.get("ten", "250,250,250,250").split(",")]
                    oya_seat = int(node.attrib.get("oya", "0"))
                    
                elif tag[0] in 'TUVW' and tag[1:].isdigit():
                    seat = 'TUVW'.index(tag[0])
                    turn_counts[seat] += 1
                    
                    # 🌟 本来はここで完全な盤面テンソルを生成しますが、
                    # 今回は抽出テストのためダミーテンソルとメタデータを記録します
                    my_score = scores[seat]
                    my_rank = sum(1 for s in scores if s > my_score) + 1
                    
                    state_info = {
                        "tensor": np.zeros((1, 25, 34), dtype=np.float32), # モック
                        "turn": turn_counts[seat],
                        "is_oya": (seat == oya_seat),
                        "my_rank": my_rank,
                        "seat": seat
                    }
                    current_kyoku_states[seat].append(state_info)
                    
                elif tag in ["AGARI", "RYUUKYOKU"]:
                    # 局終了！ sc属性から各プレイヤーの点数変動（局収支）を取得
                    # 例: sc="250,10,250,-10,250,0,250,0" -> [旧スコア, 変動, 旧スコア, 変動...] (単位は100点)
                    sc_str = node.attrib.get("sc", "")
                    if sc_str:
                        sc_vals = [int(x)*100 for x in sc_str.split(",")]
                        deltas = {0: sc_vals[1], 1: sc_vals[3], 2: sc_vals[5], 3: sc_vals[7]}
                        
                        # 抽出した未来の収支を、その局のすべての局面にラベルとして付与
                        for s in range(4):
                            result_type = "Hora" if tag == "AGARI" and int(node.attrib.get("who", "-1")) == s else \
                                          "Houju" if tag == "AGARI" and int(node.attrib.get("fromWho", "-1")) == s else \
                                          "Ryukyoku" if tag == "RYUUKYOKU" else "Other"
                                          
                            for state in current_kyoku_states[s]:
                                state["score_delta"] = deltas[s]
                                state["result_type"] = result_type
                                all_records.append(state)
                                
                    kyoku_processed += 1
                    if kyoku_processed % 500 == 0:
                        print(f"🔄 処理中... 現在 {kyoku_processed} 局のEVデータを抽出")
                        
            if kyoku_processed >= TARGET_KYOKU_COUNT: break
        if kyoku_processed >= TARGET_KYOKU_COUNT: break

    print(f"\n💾 データを保存中... ({OUTPUT_FILE})")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_records, f)
        
    # 抽出レポート
    total_states = len(all_records)
    positive_ev = sum(1 for r in all_records if r["score_delta"] > 0)
    negative_ev = sum(1 for r in all_records if r["score_delta"] < 0)
    zero_ev = sum(1 for r in all_records if r["score_delta"] == 0)
    
    print("\n" + "="*70)
    print("👑 フェーズ9A: EV（局収支）データ 抽出レポート")
    print("="*70)
    print(f"■ 抽出局数: {kyoku_processed} 局")
    print(f"■ 総抽出局面（テンソル）数: {total_states} 件")
    print(f"\n📊 【局収支 (Target EV) の分布】")
    print(f"  - プラス収支 (>0) : {positive_ev} 件 ({positive_ev/max(1, total_states)*100:.1f}%)")
    print(f"  - マイナス収支 (<0): {negative_ev} 件 ({negative_ev/max(1, total_states)*100:.1f}%)")
    print(f"  - ゼロ収支 (==0)  : {zero_ev} 件 ({zero_ev/max(1, total_states)*100:.1f}%)")
    print("="*70)

if __name__ == "__main__":
    extract_ev_dataset()