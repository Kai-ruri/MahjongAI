import os
import re
import gzip
import urllib.request
import xml.etree.ElementTree as ET
import random

LOGS_DIR = "./logs"
TARGET_KYOKU_COUNT = 200  # 傾向を見るため200局でテスト

# =========================================
# 📊 1. 評価トラッカー定義
# =========================================
class MetricsTracker:
    def __init__(self, name, alpha, beta):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.stats = {
            "riichi_opps": 0, "ai_riichi": 0, "human_riichi": 0,
            "call_opps": 0, "ai_calls": 0, "human_calls": 0,
            "overrides": 0,
            "override_details": {"dama->riichi": 0, "riichi->dama": 0, "pass->call": 0, "call->pass": 0}
        }

    def print_summary(self):
        s = self.stats
        r_opps = max(1, s['riichi_opps'])
        c_opps = max(1, s['call_opps'])
        total_opps = r_opps + c_opps
        
        print(f"\n[{self.name}] α={self.alpha:.2f}, β={self.beta:.2f}")
        print(f"  [行動分布] リーチ率: AI {s['ai_riichi']/r_opps*100:.1f}% (人 {s['human_riichi']/r_opps*100:.1f}%) | 副露率: AI {s['ai_calls']/c_opps*100:.1f}% (人 {s['human_calls']/c_opps*100:.1f}%)")
        print(f"  [EV介入] Override率: {s['overrides']/max(1, total_opps)*100:.1f}% ({s['overrides']}回)")
        print(f"    - 攻めの補正: dama->riichi: {s['override_details']['dama->riichi']}回 | pass->call: {s['override_details']['pass->call']}回")
        print(f"    - 守りの補正: riichi->dama: {s['override_details']['riichi->dama']}回 | call->pass: {s['override_details']['call->pass']}回")

# =========================================
# ⚖️ 2. EV再ランキング・ルーター (モック推論)
# =========================================
def evaluate_action(candidates, human_choice, alpha, beta):
    best_base_action, best_base_score = None, -float('inf')
    best_rr_action, best_rr_score = None, -float('inf')

    for act in candidates:
        # ⭐️ 修正: Policyの差を縮め、迷っている(拮抗している)状態をシミュレート
        pol = random.uniform(0.45, 0.55) if act == human_choice else random.uniform(0.40, 0.50)
        
        ev_plus = random.uniform(0.0, 1.0)
        ev_minus = random.uniform(0.0, 1.0)
        
        # リーチや鳴きはリターンもリスクも高い傾向
        if act in ["riichi", "chi", "pon"]: 
            ev_plus = min(1.0, ev_plus + 0.3)
            ev_minus = min(1.0, ev_minus + 0.3)

        rr_score = pol + (alpha * ev_plus) - (beta * ev_minus)

        if pol > best_base_score:
            best_base_score, best_base_action = pol, act
        if rr_score > best_rr_score:
            best_rr_score, best_rr_action = rr_score, act

    return best_base_action, best_rr_action

# =========================================
# 🎬 3. バッチシミュレーション (全係数同時実行)
# =========================================
def run_grid_search():
    print(f"📁 {LOGS_DIR} から牌譜を読み込み、EV係数のグリッドサーチを開始します...")
    
    # 5つのAI（係数パターン）を同時に走らせる
    configs = [
        ("弱 (微修正)", 0.05, 0.05),
        ("中 (標準)", 0.10, 0.10),
        ("強 (EV重視)", 0.20, 0.20),
        ("守備偏重", 0.05, 0.15),
        ("攻撃偏重", 0.15, 0.05)
    ]
    trackers = [MetricsTracker(name, a, b) for name, a, b in configs]
    
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.gz')]
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
            
            target_seat = 0
            is_menzen = {i: True for i in range(4)}
            turn_counts = [0]*4
            
            for node in root:
                tag = node.tag
                if tag == "INIT":
                    is_menzen = {i: True for i in range(4)}
                    turn_counts = [0]*4
                    
                elif tag[0] in 'TUVW' and tag[1:].isdigit():
                    seat = 'TUVW'.index(tag[0])
                    turn_counts[seat] += 1
                    
                    # リーチ機会の抽出と全パターンの評価
                    if seat == target_seat and is_menzen[seat] and turn_counts[seat] >= 6:
                        # 確率で人間がリーチしたとみなす
                        human_action = "riichi" if random.random() < 0.15 else "dama"
                        
                        for tracker in trackers:
                            tracker.stats["riichi_opps"] += 1
                            if human_action == "riichi": tracker.stats["human_riichi"] += 1
                            
                            base_act, rr_act = evaluate_action(["riichi", "dama"], human_action, tracker.alpha, tracker.beta)
                            if rr_act == "riichi": tracker.stats["ai_riichi"] += 1
                            
                            if base_act != rr_act:
                                tracker.stats["overrides"] += 1
                                tracker.stats["override_details"][f"{base_act}->{rr_act}"] += 1

                elif tag[0] in 'DEFGdefg' and tag[1:].isdigit():
                    seat = 'DEFGdefg'.index(tag[0].upper())
                    if seat != target_seat and is_menzen[target_seat]:
                        # 鳴き機会の抽出と全パターンの評価
                        if random.random() < 0.1: # 簡易的な鳴き機会発生率
                            human_action = "chi" if random.random() < 0.3 else "pass"
                            
                            for tracker in trackers:
                                tracker.stats["call_opps"] += 1
                                if human_action != "pass": tracker.stats["human_calls"] += 1
                                
                                base_act, rr_act = evaluate_action(["chi", "pass"], human_action, tracker.alpha, tracker.beta)
                                if rr_act != "pass": tracker.stats["ai_calls"] += 1
                                
                                if base_act != rr_act:
                                    tracker.stats["overrides"] += 1
                                    # "chi" を "call" に読み替えて集計
                                    b_mapped = "call" if base_act == "chi" else base_act
                                    r_mapped = "call" if rr_act == "chi" else rr_act
                                    tracker.stats["override_details"][f"{b_mapped}->{r_mapped}"] += 1
                                    
                elif tag == "N":
                    is_menzen[int(node.attrib.get("who"))] = False

                elif tag in ["AGARI", "RYUUKYOKU"]:
                    kyoku_processed += 1
                    if kyoku_processed % 50 == 0: print(f"🔄 処理中... {kyoku_processed}局完了")
                    
            if kyoku_processed >= TARGET_KYOKU_COUNT: break
        if kyoku_processed >= TARGET_KYOKU_COUNT: break

    print("\n" + "="*80)
    print("👑 フェーズ10B: EV係数 グリッドサーチ結果 (AI性格診断)")
    print("="*80)
    for tracker in trackers:
        tracker.print_summary()
    print("="*80)

if __name__ == "__main__":
    run_grid_search()