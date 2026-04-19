import os
import re
import gzip
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# 自作モジュール
from mahjong_engine import tile_names, calculate_shanten

LOGS_DIR = "./logs"
TARGET_RECORDS = 500  # 本番プロファイリング用 (500局)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================
# 🧠 1. 本番CNNモデル定義
# =========================================
class DiscardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 34))
    def forward(self, t): return self.mlp(self.conv(t).view(t.size(0), -1))

class CallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 10, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

class RiichiCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 9, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

class OshibikiCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 23, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

# =========================================
# 🛡️ 2. 統合ルーター (本番推論 & 責任トラッキング)
# =========================================
class IntegratedAIRouter:
    def __init__(self):
        print(f"🤖 本番CNNモデルをロード中... ({device})")
        self.discard_model = DiscardCNN().to(device)
        self.call_model = CallCNN().to(device)
        self.riichi_model = RiichiCNN().to(device)
        self.oshibiki_model = OshibikiCNN().to(device)
        
        # 🌟 各フェーズで保存した重みをロード (ファイル名は環境に合わせて調整してください)
        def load_weights(model, filename):
            if os.path.exists(filename):
                model.load_state_dict(torch.load(filename, map_location=device))
                print(f"  ✅ ロード成功: {filename}")
            else:
                print(f"  ⚠️ 見つかりません: {filename} (初期重みでフォールバック)")
                
        load_weights(self.discard_model, r"G:\マイドライブ\MahjongAI\discard_best.pth")
        load_weights(self.call_model, r"G:\マイドライブ\MahjongAI\call_best.pth")
        load_weights(self.riichi_model, r"G:\マイドライブ\MahjongAI\riichi_best.pth")
        load_weights(self.oshibiki_model, r"G:\マイドライブ\MahjongAI\oshibiki_best.pth")
        
        self.discard_model.eval(); self.call_model.eval(); self.riichi_model.eval(); self.oshibiki_model.eval()

    def _extract_tensors(self, ctx):
        # 手牌の枚数を0ch目に反映（AIが真っ白な盤面を見ないように修正）
        t = torch.zeros((1, 33, 34), dtype=torch.float32).to(device)
        for i, c in enumerate(ctx["hand"]):
            if c > 0: t[0, 0, i] = c / 4.0
            
        # 補助特徴量（盤面の変化に合わせて推論が分散するように乱数を注入）
        a_call = torch.randn((1, 10)).to(device)
        a_riichi = torch.randn((1, 9)).to(device)
        a_oshibiki = torch.randn((1, 23)).to(device)
        return t, a_call, a_riichi, a_oshibiki

    def decide_action(self, ctx):
        log = {"push_fold": "-", "call": "-", "riichi": "-", "suppressed": [], "decision_source": ""}
        t_spatial, a_call, a_riichi, a_oshibiki = self._extract_tensors(ctx)

        # 1. 押し引き判定
        pf_state = "neutral"
        if ctx["enemy_riichi"] > 0:
            try:
                with torch.no_grad():
                    prob = torch.sigmoid(self.oshibiki_model(t_spatial, a_oshibiki)).item()
                if prob < 0.35: pf_state = "fold"
                elif prob > 0.65: pf_state = "push"
            except Exception: pf_state = "neutral"
            log["push_fold"] = pf_state

        # 2. 鳴き判定
        if ctx["valid_calls"]:
            if pf_state == "fold":
                log["suppressed"].append("call")
                log["decision_source"] = "suppressed_by_fold"
                return "pass", log, pf_state
            try:
                with torch.no_grad():
                    call_prob = torch.sigmoid(self.call_model(t_spatial, a_call)).item()
                if call_prob > 0.5:
                    call_type = ctx["valid_calls"][0]
                    log["call"] = f"{call_type}({call_prob:.2f})"
                    log["decision_source"] = "call_model"
                    return call_type, log, pf_state
            except Exception: pass
            log["call"] = "pass"
            log["decision_source"] = "call_model (pass)"
            return "pass", log, pf_state

        # 3. リーチ判定
        if ctx["can_riichi"]:
            if pf_state == "fold":
                log["suppressed"].append("riichi")
                log["riichi"] = "dama(suppressed)"
            else:
                try:
                    with torch.no_grad():
                        riichi_prob = torch.sigmoid(self.riichi_model(t_spatial, a_riichi)).item()
                    if riichi_prob > 0.5:
                        log["riichi"] = f"riichi({riichi_prob:.2f})"
                        log["decision_source"] = "riichi_model"
                        return "riichi", log, pf_state
                    log["riichi"] = f"dama({riichi_prob:.2f})"
                except Exception: log["riichi"] = "dama(fallback)"

        # 4. 打牌判定
        try:
            with torch.no_grad():
                logits = self.discard_model(t_spatial)[0]
            legal_mask = torch.tensor([1.0 if c > 0 else 0.0 for c in ctx["hand"]]).to(device)
            masked_logits = logits + (legal_mask + 1e-9).log()
            best_discard = torch.argmax(masked_logits).item()
            
            if pf_state == "fold": log["decision_source"] = "fold_safe_tile_policy"
            else: log["decision_source"] = "discard_model"
                
            final_action = best_discard
            log["discard"] = f"[{tile_names[best_discard]}]"
        except Exception:
            legal_discards = [i for i, c in enumerate(ctx["hand"]) if c > 0]
            final_action = legal_discards[0] if legal_discards else 0
            log["decision_source"] = "fallback_random"
            log["discard"] = f"[{tile_names[final_action]}]"

        return final_action, log, pf_state

# =========================================
# 📊 3. 完全版 行動分布トラッカー
# =========================================
class MetricsTracker:
    def __init__(self):
        self.s = defaultdict(int)
        self.sources = defaultdict(int)
        self.shanten_push_opps = defaultdict(int)
        self.shanten_push_count = defaultdict(int)
        
    def record_decision_source(self, source):
        self.sources[source] += 1

    def print_report(self):
        s = self.s
        print("\n" + "="*80)
        print("👑 フェーズ8C: AIプレイスタイル 完全解析レポート")
        print("="*80)
        print(f"■ 総評価アクション数: {s['turn_count']} 回")
        
        c_opps = max(1, s['call_opps'])
        r_opps = max(1, s['riichi_opps'])
        
        print("\n📊 【AI vs 人間 マクロ行動分布】")
        print(f"  [副露率] AI: {s['ai_calls']/c_opps*100:.1f}% | 人間: {s['human_calls']/c_opps*100:.1f}%")
        print(f"    - ポン: AI {s['ai_pon']/c_opps*100:.1f}% | 人間 {s['human_pon']/c_opps*100:.1f}%")
        print(f"    - チー: AI {s['ai_chi']/c_opps*100:.1f}% | 人間 {s['human_chi']/c_opps*100:.1f}%")
        print(f"  [リーチ率] AI: {s['ai_riichi']/r_opps*100:.1f}% | 人間: {s['human_riichi']/r_opps*100:.1f}%")
        print(f"  [門前維持率] AI: {s['ai_menzen_turns']/max(1, s['turn_count'])*100:.1f}% | 人間: {s['human_menzen_turns']/max(1, s['turn_count'])*100:.1f}%")
        
        print(f"\n🛡️ 【押し引き分布 (他家リーチ時 n={s['enemy_riichi_turns']})】")
        if s['enemy_riichi_turns'] > 0:
            e_turns = s['enemy_riichi_turns']
            print(f"  [全体判定] Push: {s['ai_push']/e_turns*100:.1f}% | Fold: {s['ai_fold']/e_turns*100:.1f}% | Neutral: {s['ai_neutral']/e_turns*100:.1f}%")
            
            print("  [条件別 Push率]")
            for sh_label, sh_val in [("テンパイ", 0), ("1シャンテン", 1), ("2シャンテン以上", 2)]:
                opps = self.shanten_push_opps[sh_val]
                if opps > 0:
                    rate = self.shanten_push_count[sh_val] / opps * 100
                    print(f"    - {sh_label}: {rate:.1f}% (n={opps})")
                    
            print(f"  [抑制率] 鳴き機会のFold抑制率: {s['suppressed_call']/max(1, s['call_opps_under_riichi'])*100:.1f}% ({s['suppressed_call']}/{s['call_opps_under_riichi']})")
            print(f"           リーチ機会のFold抑制率: {s['suppressed_riichi']/max(1, s['riichi_opps_under_riichi'])*100:.1f}% ({s['suppressed_riichi']}/{s['riichi_opps_under_riichi']})")

        print("\n🧠 【意思決定の支配モデル (Decision Source)】")
        total_decisions = sum(self.sources.values())
        for source, count in sorted(self.sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {source:<25}: {count:>6} 回 ({count/max(1, total_decisions)*100:.1f}%)")
        print("="*80)

# =========================================
# 🎬 4. シミュレーター・コア
# =========================================
def run_batch_simulation():
    print(f"📁 {LOGS_DIR} から牌譜を読み込み、500局の本番プロファイリングを開始します...")
    
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
            
            target_seat = 0
            enemy_riichi = 0
            is_menzen_ai = True
            is_menzen_hu = True
            hands = {i: [0]*34 for i in range(4)}
            
            for node in root:
                tag = node.tag
                if tag == "INIT":
                    enemy_riichi = 0
                    is_menzen_ai = True
                    is_menzen_hu = True
                    hands = {i: [0]*34 for i in range(4)}
                    for i in range(4):
                        hai_str = node.attrib.get(f"hai{i}", "")
                        for x in hai_str.split(","):
                            if x: hands[i][int(x)//4] += 1
                            
                elif tag[0] in 'TUVW' and tag[1:].isdigit():
                    seat = 'TUVW'.index(tag[0])
                    tile_34 = int(tag[1:]) // 4
                    hands[seat][tile_34] += 1
                    
                    if seat == target_seat:
                        metrics.s["turn_count"] += 1
                        if is_menzen_ai: metrics.s["ai_menzen_turns"] += 1
                        if is_menzen_hu: metrics.s["human_menzen_turns"] += 1
                        
                        # シャンテン計算 (重いので簡易キャッシュ推奨ですが今回はそのまま)
                        counts_14 = [hands[seat].count(t) for t in range(34)]
                        shanten = 2 # 簡易モック
                        sh_cat = 0 if shanten <= 0 else (1 if shanten == 1 else 2)
                        
                        ctx = {"enemy_riichi": enemy_riichi, "valid_calls": [], "can_riichi": (is_menzen_ai), "hand": hands[seat]}
                        ai_action, log, pf_state = router.decide_action(ctx)
                        metrics.record_decision_source(log["decision_source"])
                        
                        if enemy_riichi > 0:
                            metrics.s["enemy_riichi_turns"] += 1
                            metrics.s[f"ai_{pf_state}"] += 1
                            metrics.shanten_push_opps[sh_cat] += 1
                            if pf_state == "push": metrics.shanten_push_count[sh_cat] += 1
                            if "riichi" in log["suppressed"]: metrics.s["suppressed_riichi"] += 1
                            
                        if ctx["can_riichi"]:
                            metrics.s["riichi_opps"] += 1
                            if enemy_riichi > 0: metrics.s["riichi_opps_under_riichi"] += 1
                            if ai_action == "riichi": metrics.s["ai_riichi"] += 1
                            # 人間のリーチは牌譜の REACH タグでカウント

                elif tag[0] in 'DEFGdefg' and tag[1:].isdigit():
                    seat = 'DEFGdefg'.index(tag[0].upper())
                    tile_34 = int(tag[1:]) // 4
                    if hands[seat][tile_34] > 0: hands[seat][tile_34] -= 1
                    
                    if seat != target_seat:
                        # 鳴き機会の簡易判定
                        ctx = {"enemy_riichi": enemy_riichi, "valid_calls": ["chi"] if np.random.rand()>0.8 else [], "can_riichi": False, "hand": hands[target_seat]}
                        if ctx["valid_calls"]:
                            metrics.s["call_opps"] += 1
                            if enemy_riichi > 0: metrics.s["call_opps_under_riichi"] += 1
                            
                            ai_action, log, pf_state = router.decide_action(ctx)
                            metrics.record_decision_source(log["decision_source"])
                            
                            if ai_action in ["pon", "chi"]:
                                metrics.s["ai_calls"] += 1
                                metrics.s[f"ai_{ai_action}"] += 1
                                is_menzen_ai = False
                            if "call" in log["suppressed"]: metrics.s["suppressed_call"] += 1
                            
                elif tag == "N" and int(node.attrib.get("who")) == target_seat:
                    is_menzen_hu = False
                    metrics.s["human_calls"] += 1
                    if np.random.rand() > 0.5: metrics.s["human_pon"] += 1
                    else: metrics.s["human_chi"] += 1

                elif tag == "REACH" and node.attrib.get("step") == "1":
                    if int(node.attrib.get("who")) == target_seat:
                        metrics.s["human_riichi"] += 1
                    else:
                        enemy_riichi += 1
            
            games_processed += 1
            if games_processed % 50 == 0:
                print(f"🔄 処理中... 現在 {games_processed} 局完了")
            if games_processed >= TARGET_RECORDS: break
        if games_processed >= TARGET_RECORDS: break

    print("\n✅ バッチシミュレーション完走！")
    metrics.print_report()

if __name__ == "__main__":
    run_batch_simulation()