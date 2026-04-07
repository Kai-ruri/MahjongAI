import os
import re
import gzip
import urllib.request
import xml.etree.ElementTree as ET
import time
import numpy as np

LOGS_DIR = "./logs"
TARGET_RECORDS = 500  # まずは500局(数千アクション)をプロファイリング

# =========================================
# 📊 フェーズ8C: 行動分布トラッカー
# =========================================
class MetricsTracker:
    def __init__(self):
        self.stats = {
            "turn_count": 0,
            
            # 鳴き分布
            "call_opps": 0, "ai_calls": 0, "human_calls": 0,
            "ai_pon": 0, "ai_chi": 0, "human_pon": 0, "human_chi": 0,
            
            # リーチ分布
            "riichi_opps": 0, "ai_riichi": 0, "human_riichi": 0,
            "ai_menzen_turns": 0, "human_menzen_turns": 0,
            
            # 押し引き分布 (他家リーチ時)
            "enemy_riichi_turns": 0,
            "ai_push": 0, "ai_fold": 0, "ai_neutral": 0,
            "ai_push_tenpai": 0, "ai_push_1shan": 0,
            "suppressed_call_by_fold": 0, "suppressed_riichi_by_fold": 0,
            
            # 乖離率
            "action_divergence": 0
        }

    def print_report(self):
        s = self.stats
        print("\n" + "="*80)
        print("👑 フェーズ8C: 統合AI 行動分布プロファイリング結果 (人間 vs AI)")
        print("="*80)
        print(f"■ 総評価アクション数: {s['turn_count']} 回")
        
        print("\n📊 【マクロ行動分布 (AI vs 人間)】")
        call_opps = max(1, s['call_opps'])
        riichi_opps = max(1, s['riichi_opps'])
        
        print(f"  [副露率(機会ベース)] AI: {s['ai_calls']/call_opps*100:.1f}% | 人間: {s['human_calls']/call_opps*100:.1f}%")
        print(f"    - ポン率: AI {s['ai_pon']/call_opps*100:.1f}% | 人間 {s['human_pon']/call_opps*100:.1f}%")
        print(f"    - チー率: AI {s['ai_chi']/call_opps*100:.1f}% | 人間 {s['human_chi']/call_opps*100:.1f}%")
        
        print(f"  [リーチ率(機会ベース)] AI: {s['ai_riichi']/riichi_opps*100:.1f}% | 人間: {s['human_riichi']/riichi_opps*100:.1f}%")
        print(f"    - 門前維持率: AI {s['ai_menzen_turns']/max(1, s['turn_count'])*100:.1f}% | 人間 {s['human_menzen_turns']/max(1, s['turn_count'])*100:.1f}%")
        
        print("\n🛡️ 【押し引き分布 (他家リーチ局面: n={})】".format(s['enemy_riichi_turns']))
        if s['enemy_riichi_turns'] > 0:
            e_turns = s['enemy_riichi_turns']
            print(f"  [AI 判定比率] Push: {s['ai_push']/e_turns*100:.1f}% | Fold: {s['ai_fold']/e_turns*100:.1f}% | Neutral: {s['ai_neutral']/e_turns*100:.1f}%")
            print(f"  [抑制機能] Foldによる鳴き抑制: {s['suppressed_call_by_fold']}回 | リーチ抑制: {s['suppressed_riichi_by_fold']}回")
        else:
            print("  他家リーチ局面なし")

        print("\n⚠️ 【行動の乖離 (HumanとAIで選択が分かれた割合)】")
        print(f"  全体アクション乖離率: {s['action_divergence']/max(1, s['turn_count'])*100:.1f}%")
        print("="*80)

# =========================================
# 🧠 本番統合ルーター (本番モデル読み込みラッパー)
# =========================================
class IntegratedAIRouter:
    def __init__(self):
        # 🌟 TODO: 実際はここで torch.load() を行い、各モデルをデバイスにマウントします
        print("🤖 本番CNN統合ルーターをスタンバイ...")
        
    def decide_action(self, ctx):
        log = {"push_fold": "-", "call": "-", "riichi": "-", "suppressed": []}
        final_action = "discard" # デフォルト

        # 1. 押し引き判定 (他家リーチ時のみ)
        pf_state = "neutral"
        if ctx["enemy_riichi"] > 0:
            # 🌟 TODO: Oshibiki CNN-A Inference
            r = np.random.rand()
            if r < 0.25: pf_state = "push"
            elif r < 0.70: pf_state = "fold"
            log["push_fold"] = pf_state

        # 2. 鳴き判定
        if ctx["valid_calls"]:
            if pf_state == "fold":
                log["suppressed"].append("call")
            else:
                # 🌟 TODO: Call CNN Inference
                if np.random.rand() < 0.20: # 本番約20%の鳴き率を想定
                    call_type = ctx["valid_calls"][0]
                    log["call"] = call_type
                    return call_type, log, pf_state
        
        # 3. リーチ判定
        if ctx["can_riichi"]:
            if pf_state == "fold":
                log["suppressed"].append("riichi")
            else:
                # 🌟 TODO: Riichi CNN Inference
                if np.random.rand() < 0.45: # テンパイ時の本番リーチ率想定
                    log["riichi"] = "riichi"
                    return "riichi", log, pf_state
        
        # 4. 打牌判定
        # 🌟 TODO: Discard CNN Inference (Fold時は現物から強制選択するロジックを追加)
        return final_action, log, pf_state

# =========================================
# 🎬 シミュレーター・コア
# =========================================
def run_batch_simulation():
    print(f"📁 {LOGS_DIR} から牌譜を読み込み、バッチシミュレーションを開始します...")
    
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.gz')]
    router = IntegratedAIRouter()
    metrics = MetricsTracker()
    
    games_processed = 0

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
            except: continue
            
            target_seat = 0 # Seat 0 をプロファイリング対象とする
            enemy_riichi = 0
            is_menzen_ai = True
            is_menzen_hu = True
            
            for node in root:
                tag = node.tag
                if tag == "INIT":
                    enemy_riichi = 0
                    is_menzen_ai = True
                    is_menzen_hu = True
                    
                elif tag[0] in 'TUVW' and tag[1:].isdigit():
                    seat = 'TUVW'.index(tag[0])
                    if seat == target_seat:
                        metrics.stats["turn_count"] += 1
                        if is_menzen_ai: metrics.stats["ai_menzen_turns"] += 1
                        if is_menzen_hu: metrics.stats["human_menzen_turns"] += 1
                        
                        # コンテキスト生成
                        ctx = {"enemy_riichi": enemy_riichi, "valid_calls": [], "can_riichi": (is_menzen_hu and np.random.rand()<0.05)}
                        
                        # AI推論
                        ai_action, log, pf_state = router.decide_action(ctx)
                        
                        # トラッキング更新
                        if ctx["enemy_riichi"] > 0:
                            metrics.stats["enemy_riichi_turns"] += 1
                            metrics.stats[f"ai_{pf_state}"] += 1
                            if "call" in log["suppressed"]: metrics.stats["suppressed_call_by_fold"] += 1
                            if "riichi" in log["suppressed"]: metrics.stats["suppressed_riichi_by_fold"] += 1
                        
                        if ctx["can_riichi"]:
                            metrics.stats["riichi_opps"] += 1
                            if ai_action == "riichi": metrics.stats["ai_riichi"] += 1
                            # (※簡易的に、人間がリーチ宣言したノードをここでカウントするロジックが必要ですが今回はモック統計)
                            if np.random.rand() < 0.40: metrics.stats["human_riichi"] += 1
                            
                elif tag == "N" and int(node.attrib.get("who")) == target_seat:
                    is_menzen_hu = False
                    metrics.stats["human_calls"] += 1
                    # 簡易判定
                    if np.random.rand() > 0.5: metrics.stats["human_pon"] += 1
                    else: metrics.stats["human_chi"] += 1

                elif tag == "REACH" and node.attrib.get("step") == "1":
                    if int(node.attrib.get("who")) != target_seat: enemy_riichi += 1
            
            # AI側のモック鳴き機会と集計
            for _ in range(15): # 1局の平均他家打牌数
                ctx = {"enemy_riichi": enemy_riichi, "valid_calls": ["chi" if np.random.rand() > 0.5 else "pon"], "can_riichi": False}
                metrics.stats["call_opps"] += 1
                ai_action, log, _ = router.decide_action(ctx)
                if ai_action in ["pon", "chi"]:
                    metrics.stats["ai_calls"] += 1
                    metrics.stats[f"ai_{ai_action}"] += 1
                    is_menzen_ai = False

            games_processed += 1
            if games_processed % 50 == 0:
                print(f"🔄 処理中... 現在 {games_processed} 局完了")
            if games_processed >= TARGET_RECORDS: break
        if games_processed >= TARGET_RECORDS: break

    print("✅ バッチシミュレーション完走！")
    metrics.print_report()

if __name__ == "__main__":
    run_batch_simulation()