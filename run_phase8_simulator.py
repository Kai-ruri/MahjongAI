import os
import re
import gzip
import urllib.request
import xml.etree.ElementTree as ET
import random
import time

# 自作モジュール（フェーズ4で作成したものを想定）
from mahjong_engine import tile_names

LOGS_DIR = "./logs"

# =========================================
# 🧠 フェーズ8A: 統合ポリシー（階層型ルーター）
# =========================================
class MahjongAIRouter:
    def __init__(self):
        print("🤖 統合AIルーターを初期化しました (※CNNモデルはプロトタイプ用のモックです)")
        # TODO: 将来的にここでフェーズ3,5,6,7のPyTorchモデル(.pth)をロードします
        
    def _predict_push_fold(self, context):
        """ 【階層1】押し引き判定 (Push / Fold / Neutral) """
        if context["enemy_riichi"] == 0:
            return "neutral", 0.0 # 他家リーチなしなら通常進行
        
        # モックロジック: ランダムに状態を返す（将来CNNの出力確率で分岐）
        prob = random.random()
        if prob < 0.3: return "fold", prob
        elif prob > 0.8: return "push", prob
        else: return "neutral", prob

    def _predict_call(self, context, valid_calls):
        """ 【階層2】鳴き判定 """
        if not valid_calls: return "pass", 0.0
        # モックロジック: 20%で鳴く
        prob = random.random()
        return (valid_calls[0], prob) if prob > 0.8 else ("pass", prob)

    def _predict_riichi(self, context):
        """ 【階層3】リーチ判定 """
        if not context["can_riichi"]: return "dama", 0.0
        # モックロジック: 60%でリーチ
        prob = random.random()
        return ("riichi", prob) if prob > 0.4 else ("dama", prob)

    def _predict_discard(self, context):
        """ 【階層4】打牌判定 """
        # モックロジック: 手牌の中からランダムに選ぶ
        hand = context["hand"]
        legal_discards = [i for i, c in enumerate(hand) if c > 0]
        if not legal_discards: return 0, 0.0
        chosen = random.choice(legal_discards)
        return chosen, 0.99 # 確信度モック

    def decide_action(self, context):
        """ 統合ルーター本体: 上位判断が下位判断をゲートする """
        log_data = {
            "turn": context["turn"],
            "state_summary": f"向聴:{context['shanten']} 他家R:{context['enemy_riichi']}人",
            "push_fold": "-", "call": "-", "riichi": "-", "discard": "-", "final_action": "-"
        }

        # 0. 行動可能性チェック
        is_call_opportunity = len(context["valid_calls"]) > 0
        
        # 1. 他家リーチ時の押し引き判定
        pf_state, pf_prob = self._predict_push_fold(context)
        log_data["push_fold"] = f"{pf_state}({pf_prob:.2f})"

        # 2. 鳴き判定機会
        if is_call_opportunity:
            if pf_state == "fold":
                log_data["call"] = "pass (suppressed by fold)"
                log_data["final_action"] = "pass"
                return "pass", log_data
            
            call_action, call_prob = self._predict_call(context, context["valid_calls"])
            log_data["call"] = f"{call_action}({call_prob:.2f})"
            if call_action != "pass":
                log_data["final_action"] = f"call_{call_action}"
                return call_action, log_data
            
            log_data["final_action"] = "pass"
            return "pass", log_data

        # 3. リーチ判定機会 (ツモ番)
        if context["can_riichi"]:
            if pf_state == "fold":
                log_data["riichi"] = "dama (suppressed by fold)"
            else:
                riichi_action, riichi_prob = self._predict_riichi(context)
                log_data["riichi"] = f"{riichi_action}({riichi_prob:.2f})"
                if riichi_action == "riichi":
                    log_data["final_action"] = "riichi"
                    return "riichi", log_data

        # 4. 打牌判定 (ツモ番)
        # Fold時は安全牌を優先するロジックをここに挟む（今回はモックなのでそのまま打牌モデルへ）
        discard_action, discard_prob = self._predict_discard(context)
        log_data["discard"] = f"[{tile_names[discard_action]}]({discard_prob:.2f})"
        
        # 最終行動決定
        if log_data["final_action"] == "-":
            log_data["final_action"] = f"discard [{tile_names[discard_action]}]"

        return discard_action, log_data

# =========================================
# 🎬 フェーズ8B: 牌譜再生型シミュレーター
# =========================================
def run_simulator(target_log_id=None):
    print("\n" + "="*60)
    print("🎬 牌譜再生シミュレーター 起動")
    print("="*60)
    
    # ログファイルから適当な牌譜を1つ取得
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.gz')]
    if not log_files:
        print("❌ 牌譜データがありません。")
        return

    log_id = None
    with gzip.open(os.path.join(LOGS_DIR, log_files[0]), 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r'log=([\w-]+)', line)
            if match:
                log_id = match.group(1)
                break
                
    print(f"📡 牌譜をダウンロード中: {log_id}")
    req = urllib.request.Request(f"https://tenhou.net/0/log/?{log_id}", headers={'User-Agent': 'Mozilla/5.0'})
    xml_string = urllib.request.urlopen(req).read().decode('utf-8')
    root = ET.fromstring(xml_string)
    
    router = MahjongAIRouter()
    
    # 簡易シミュレーションループ (今回はSeat 0をターゲットAIとする)
    target_seat = 0
    hands = {i: [0]*34 for i in range(4)}
    enemy_riichi_count = 0
    turn_count = 0
    
    print("\n📝 --- 意思決定ログ (対象: Seat 0) ---")
    print(f"{'巡目':<4} | {'状態':<15} | {'Push/Fold':<15} | {'鳴き':<15} | {'リーチ':<15} | {'打牌候補':<15} | {'最終AI行動':<20} | {'(参考)人間'}")
    print("-" * 120)

    for node in root:
        tag = node.tag
        if tag == "INIT":
            # 配牌
            for i in range(4):
                hai_str = node.attrib.get(f"hai{i}", "")
                for x in hai_str.split(","):
                    if x: hands[i][int(x)//4] += 1
            enemy_riichi_count = 0
            turn_count = 0
            
        elif tag[0] in 'TUVW' and tag[1:].isdigit(): # ツモ
            seat = 'TUVW'.index(tag[0])
            tile_34 = int(tag[1:]) // 4
            hands[seat][tile_34] += 1
            
            if seat == target_seat:
                turn_count += 1
                context = {
                    "turn": turn_count, "shanten": "不明(モック)", "enemy_riichi": enemy_riichi_count,
                    "valid_calls": [], "can_riichi": True, "hand": hands[seat]
                }
                ai_action, log = router.decide_action(context)
                print(f"{log['turn']:<4} | {log['state_summary']:<13} | {log['push_fold']:<15} | {log['call']:<14} | {log['riichi']:<14} | {log['discard']:<14} | {log['final_action']:<18} | (人間の打牌待ち)")
                
        elif tag[0] in 'DEFGdefg' and tag[1:].isdigit(): # 打牌
            seat = 'DEFGdefg'.index(tag[0].upper())
            tile_34 = int(tag[1:]) // 4
            if hands[seat][tile_34] > 0: hands[seat][tile_34] -= 1
            
            if seat == target_seat:
                # 人間の実際の打牌を表示して改行
                print(f"\033[F\033[110C -> [{tile_names[tile_34]}]") 
            else:
                # 他家の打牌に対して鳴き判定の機会
                context = {
                    "turn": turn_count, "shanten": "不明(モック)", "enemy_riichi": enemy_riichi_count,
                    "valid_calls": ["chi"], "can_riichi": False, "hand": hands[target_seat]
                }
                # モックとして時々鳴きの問い合わせを発生させる
                if random.random() < 0.1:
                    ai_action, log = router.decide_action(context)
                    human_action = "pass" # 牌譜通りなら基本pass
                    print(f"{log['turn']:<4} | {log['state_summary']:<13} | {log['push_fold']:<15} | {log['call']:<14} | {log['riichi']:<14} | {log['discard']:<14} | {log['final_action']:<18} | -> {human_action}")
        
        elif tag == "REACH" and node.attrib.get("step") == "1":
            if int(node.attrib.get("who")) != target_seat: enemy_riichi_count += 1

    print("="*120)
    print("✅ 牌譜再生シミュレーターがエラーなく完走しました。")

if __name__ == "__main__":
    run_simulator()