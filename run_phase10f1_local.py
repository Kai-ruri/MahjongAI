import os
import random
import numpy as np
import torch
import torch.nn as nn
import time

# =========================================
# 🧠 1. 本番CNNモデル定義
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiscardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 34))
    def forward(self, t): return self.mlp(self.conv(t).view(t.size(0), -1))

class ActionCNN(nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + aux_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

class EV_CNN_Base3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 10 + 35, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, m, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), m, a], dim=1))

class ModelManager:
    def __init__(self):
        self.discard_net = DiscardCNN().to(device)
        self.riichi_net = ActionCNN(9).to(device)
        self.call_net = ActionCNN(10).to(device)
        self.ev_plus_net = EV_CNN_Base3().to(device)
        self.ev_minus_net = EV_CNN_Base3().to(device)
        self._load(self.discard_net, "discard_best.pth")
        self._load(self.riichi_net, "riichi_best.pth")
        self._load(self.call_net, "call_best.pth")
        self._load(self.ev_plus_net, "ev_plus_best.pth")
        self._load(self.ev_minus_net, "ev_minus_best.pth")
        self.discard_net.eval(); self.riichi_net.eval(); self.call_net.eval()
        self.ev_plus_net.eval(); self.ev_minus_net.eval()

    def _load(self, model, filename):
        if os.path.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=device))

# =========================================
# 🛡️ 2. 危険度カテゴリ定義とモック手牌生成
# =========================================
# 0: 現物, 1: 準安全(スジ/字牌), 2: 無スジ(危険牌)
def generate_mock_hand_with_safety():
    """ 手牌14枚と、それぞれのリーチ者に対する危険度を生成 """
    hand = []
    has_genbutsu = False
    has_semi_safe = False
    
    for _ in range(14):
        r = random.random()
        if r < 0.15: 
            danger_level = 0 # 現物 (15%の確率で持っているとする)
            has_genbutsu = True
        elif r < 0.40:
            danger_level = 1 # 準安全牌 (スジ・字牌など 25%)
            has_semi_safe = True
        else:
            danger_level = 2 # 無スジ・危険牌 (60%)
        hand.append(danger_level)
    return hand, has_genbutsu, has_semi_safe

# =========================================
# 🤖 3. 分析用エージェント (Phase 10D 最終ベスト設定)
# =========================================
class AnalysisAgent:
    def __init__(self, agent_id, models):
        self.agent_id = agent_id
        self.models = models
        self.stats = {
            "fold_situations": 0,    # リーチ者あり＆自分2向聴以上の局面数
            "genbutsu_available": 0, # そのうち現物を持っていた回数
            "safe_available": 0,     # 現物or準安全牌を持っていた回数
            
            "missed_genbutsu": 0,    # ⭐️ 現物があるのに現物以外を切った回数
            "missed_safe": 0,        # ⭐️ 安牌があるのに無スジを切った回数
            
            "cut_danger": 0,         # 危険牌(無スジ)を切った回数
            "houju_by_danger": 0,    # ⭐️ 危険牌を切って放銃した回数
            
            "total_houju_in_fold": 0 # fold局面での総放銃回数
        }
        self.reset_kyoku()

    def reset_kyoku(self):
        self.is_riichi = False
        self.shanten = 2 # 基本2向聴以上でスタート
        self.hand, self.has_genbutsu, self.has_semi_safe = [], False, False

    def get_coeffs(self, is_oya, rank):
        """ Phase 10D ベスト設定 (親/子 + ラス目攻撃 + 好形攻撃) """
        alpha = 0.05 if is_oya else 0.03
        beta = 0.10 if is_oya else 0.14
        if rank == 3: alpha += 0.005; beta -= 0.01 # ラス目攻撃
        if self.shanten == 0: alpha += 0.005; beta -= 0.01 # テンパイ攻撃
        elif self.shanten == 1: alpha += 0.002; beta -= 0.005 # 1向聴攻撃
        return max(0.0, alpha), max(0.0, beta)

    def select_discard(self, is_riichi_others):
        """ 本番DiscardCNNによる打牌選択 (モック) """
        # ※実際はテンソル入力ですが、今回は危険度カテゴリからCNNが「どれを選びがちか」をシミュレートします。
        # 今のAIは牌効率重視のため、危険度を無視してランダム(手作りの都合)に選ぶ傾向がある。
        
        # 14枚の候補から1枚を選ぶ
        chosen_idx = random.randint(0, len(self.hand)-1)
        chosen_danger = self.hand[chosen_idx]
        
        # ⚠️ 誤り分析ロジック
        if is_riichi_others and self.shanten >= 2 and not self.is_riichi:
            self.stats["fold_situations"] += 1
            if self.has_genbutsu: self.stats["genbutsu_available"] += 1
            if self.has_genbutsu or self.has_semi_safe: self.stats["safe_available"] += 1
            
            # KPI 1: 現物見落とし
            if self.has_genbutsu and chosen_danger != 0:
                self.stats["missed_genbutsu"] += 1
                
            # KPI 2: 安全牌見落とし (現物or準安牌があるのに無スジを切る)
            if (self.has_genbutsu or self.has_semi_safe) and chosen_danger == 2:
                self.stats["missed_safe"] += 1
                
            # KPI 3: 危険打牌
            if chosen_danger == 2:
                self.stats["cut_danger"] += 1
                # 無スジを切った場合の放銃判定 (モック: 約10%で当たる)
                if random.random() < 0.10:
                    self.stats["houju_by_danger"] += 1
                    self.stats["total_houju_in_fold"] += 1
                    return True # 放銃フラグ
                    
            # 現物・準安全牌での不慮の放銃 (モック: スジ引掛け等 1%未満)
            elif chosen_danger == 1 and random.random() < 0.01:
                self.stats["total_houju_in_fold"] += 1
                return True
                
        return False # 通った

# =========================================
# 🌍 4. 分析用シミュレーター
# =========================================
def run_analysis(num_games=5000):
    models = ModelManager()
    agents = [AnalysisAgent(i, models) for i in range(4)]
    
    print(f"\n🚀 Phase 10F-1: 守備局面 誤り分析シミュレーションを開始 ({num_games}局)")
    print(" 条件: 他家リーチあり × 自分2向聴以上")
    
    start_time = time.time()
    for game in range(num_games):
        for a in agents: a.reset_kyoku()
        
        # 誰かがリーチをかける (モック: 巡目適当に誰かをリーチ状態に)
        riichi_turn = random.randint(5, 12)
        riichi_agent_id = random.randint(0, 3)
        
        for turn in range(70):
            agent = agents[turn % 4]
            agent.hand, agent.has_genbutsu, agent.has_semi_safe = generate_mock_hand_with_safety()
            
            if turn == riichi_turn:
                agents[riichi_agent_id].is_riichi = True
                agents[riichi_agent_id].shanten = 0
            
            is_riichi_others = any(a.is_riichi and a.agent_id != agent.agent_id for a in agents)
            
            # 打牌と放銃判定
            is_houju = agent.select_discard(is_riichi_others)
            
            if is_houju:
                break # 局終了
                
    print(f"✅ シミュレーション完了 ({time.time() - start_time:.1f}秒)\n")
    
    # 統計情報の集約
    target = agents[0].stats # 代表してPlayer 0のスタッツを表示
    f_sit = max(1, target["fold_situations"])
    g_avail = max(1, target["genbutsu_available"])
    s_avail = max(1, target["safe_available"])
    c_danger = max(1, target["cut_danger"])
    
    print("👑 === Phase 10F-2: 守備打牌 誤り分析レポート ===")
    print(f"■ 対象局面 (他家リーチ＆自分2向聴以上): {f_sit} 回")
    print("-" * 60)
    print(f"🚨 KPI 1: 現物見落とし率")
    print(f"   現物を保有していた局面: {g_avail}回")
    print(f"   → なのに現物以外を切った: {target['missed_genbutsu']}回")
    print(f"   【現物見落とし率】: {target['missed_genbutsu'] / g_avail * 100:.1f}%")
    print("-" * 60)
    print(f"🚨 KPI 2: 安全牌見落とし率")
    print(f"   現物or準安牌を保有していた局面: {s_avail}回")
    print(f"   → なのに無スジを切った: {target['missed_safe']}回")
    print(f"   【安牌見落とし率】: {target['missed_safe'] / s_avail * 100:.1f}%")
    print("-" * 60)
    print(f"🚨 KPI 3: 危険打牌放銃率")
    print(f"   無スジを切った回数: {c_danger}回")
    print(f"   → その無スジで放銃した: {target['houju_by_danger']}回")
    print(f"   【危険打牌放銃率】: {target['houju_by_danger'] / c_danger * 100:.1f}%")
    print("-" * 60)
    print(f"📊 総合: fold局面での総放銃回数: {target['total_houju_in_fold']}回")
    print(f"   (対象局面全体の放銃率: {target['total_houju_in_fold'] / f_sit * 100:.1f}%)")

if __name__ == "__main__":
    run_analysis()