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
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 34))
    def forward(self, t): return self.mlp(self.conv(t).view(t.size(0), -1))

class ActionCNN(nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + aux_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

class EV_CNN_Base3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 10 + 35, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, m, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), m, a], dim=1))

class ModelManager:
    def __init__(self):
        self.riichi_net = ActionCNN(9).to(device)
        self.call_net = ActionCNN(10).to(device)
        self.ev_plus_net = EV_CNN_Base3().to(device)
        self.ev_minus_net = EV_CNN_Base3().to(device)
        self._load(self.riichi_net, r"G:\マイドライブ\MahjongAI\riichi_best.pth")
        self._load(self.call_net, r"G:\マイドライブ\MahjongAI\call_best.pth")
        self._load(self.ev_plus_net, r"G:\マイドライブ\MahjongAI\ev_plus_best.pth")
        self._load(self.ev_minus_net, r"G:\マイドライブ\MahjongAI\ev_minus_best.pth")
        self.riichi_net.eval(); self.call_net.eval()
        self.ev_plus_net.eval(); self.ev_minus_net.eval()

    def _load(self, model, filename):
        if os.path.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=device))

# =========================================
# 🤖 2. 状況別エージェント (Shanten-Aware)
# =========================================
class ShantenAwareAgent:
    def __init__(self, agent_id, models, config_name, config_type):
        self.agent_id = agent_id
        self.models = models
        self.config_name = config_name
        self.config_type = config_type # "A", "B", "C"
        
        self.stats = {
            "total_kyoku": 0, "total_score": 0, 
            "riichi_count": 0, "call_count": 0, "riichi_overrides": 0, "call_overrides": 0,
            "agari_count": 0, "houju_count": 0,
            # シャンテン数別の集計
            "kyoku_sh0": 0, "houju_sh0": 0, "score_sh0": 0,
            "kyoku_sh1": 0, "houju_sh1": 0, "score_sh1": 0,
            "kyoku_sh2": 0, "houju_sh2": 0, "score_sh2": 0
        }
        self.reset_kyoku()

    def reset_kyoku(self):
        self.is_riichi = False
        self.has_called = False
        self.shanten = 2 # 局開始時は基本的に2シャンテン以上とする
        self.stats["total_kyoku"] += 1

    def advance_shanten(self):
        """ 手が進む (ツモや鳴きによる有効牌) """
        if self.shanten > 0:
            self.shanten -= 1

    def get_current_coeffs(self, is_oya, rank):
        # 🌟 ベース: 10D-1 (親/子ハイブリッド)
        alpha = 0.05 if is_oya else 0.03
        beta = 0.10 if is_oya else 0.14
        
        # 🌟 ベース: 10D-2 (ラス目特化)
        if rank == 3:
            alpha += 0.005; beta -= 0.01
            
        # 🌟 Phase 10D-3: シャンテン数差分
        if self.config_type in ["B", "C"]:
            if self.shanten == 0:
                alpha += 0.005; beta -= 0.01
            elif self.shanten == 1:
                alpha += 0.002; beta -= 0.005
                
        if self.config_type == "C":
            if self.shanten >= 2:
                alpha -= 0.005; beta += 0.005
                
        return max(0.0, alpha), max(0.0, beta)

    def _get_mock_inputs(self):
        t = (torch.rand((1, 25, 34)) < 0.05).float().to(device)
        m = torch.rand((1, 10)).to(device)
        return t, m

    def act_discard(self, turn_count, is_oya, rank):
        if self.is_riichi or self.has_called or turn_count <= 6: return "discard"
        
        # テンパイ(0シャンテン)の時のみリーチ判断が発生
        if self.shanten != 0: return "discard"

        alpha, beta = self.get_current_coeffs(is_oya, rank)
        t, m = self._get_mock_inputs()
        
        pol_riichi = random.uniform(0.45, 0.55)
        pol_dama = 1.0 - pol_riichi

        a_vec_r = torch.zeros((1, 35)).to(device); a_vec_r[0, 0] = 2.0
        a_vec_d = torch.zeros((1, 35)).to(device); a_vec_d[0, 0] = 0.0

        with torch.no_grad():
            oya_bonus = 0.15 if is_oya else 0.0
            plus_r = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_r)).item() + 0.4 + oya_bonus
            minus_r = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec_r)).item() + 0.3
            plus_d = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_d)).item()
            minus_d = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec_d)).item()

        score_r = pol_riichi + (alpha * plus_r) - (beta * minus_r)
        score_d = pol_dama + (alpha * plus_d) - (beta * minus_d)

        base_choice = "riichi" if pol_riichi > pol_dama else "dama"
        rr_choice = "riichi" if score_r > score_d else "dama"

        if base_choice != rr_choice: self.stats["riichi_overrides"] += 1
        if rr_choice == "riichi":
            self.is_riichi = True
            self.stats["riichi_count"] += 1
            return "riichi_discard"
        return "discard"

    def act_response(self, is_oya, rank):
        if self.is_riichi: return "pass"
        if random.random() < 0.02: return "ron" 
        
        # 鳴き機会は1〜2シャンテンの時に発生しやすいとする
        if self.shanten == 0 or random.random() > 0.15: return "pass" 

        alpha, beta = self.get_current_coeffs(is_oya, rank)
        t, m = self._get_mock_inputs()
        
        pol_call = random.uniform(0.45, 0.55)
        pol_pass = 1.0 - pol_call

        a_vec_c = torch.zeros((1, 35)).to(device); a_vec_c[0, 0] = 1.0
        a_vec_p = torch.zeros((1, 35)).to(device); a_vec_p[0, 0] = 0.0

        with torch.no_grad():
            oya_bonus = 0.1 if is_oya else 0.0
            # 1シャンテンの鳴きは価値が高い(すぐテンパイ)のでEVモックにボーナス
            sh_bonus = 0.1 if self.shanten == 1 else 0.0
            plus_c = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_c)).item() + 0.3 + oya_bonus + sh_bonus
            minus_c = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec_c)).item() + 0.2
            plus_p = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_p)).item()
            minus_p = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec_p)).item()

        score_c = pol_call + (alpha * plus_c) - (beta * minus_c)
        score_p = pol_pass + (alpha * plus_p) - (beta * minus_p)

        base_choice = "call" if pol_call > pol_pass else "pass"
        rr_choice = "call" if score_c > score_p else "pass"

        if base_choice != rr_choice: self.stats["call_overrides"] += 1
        if rr_choice == "call":
            self.has_called = True
            self.stats["call_count"] += 1
            self.advance_shanten() # 鳴いたら手が進む
            return "call"
        return "pass"

# =========================================
# 🌍 3. 環境ランナー
# =========================================
def run_kyoku(agents):
    for a in agents: a.reset_kyoku()
    oya_id = random.randint(0, 3) 
    ranks = [0, 1, 2, 3]
    random.shuffle(ranks)
    
    for turn in range(70):
        agent = agents[turn % 4]
        
        # ツモによって自然に手が進行するモック
        if random.random() < 0.10: agent.advance_shanten()
        
        is_oya = (agent.agent_id == oya_id)
        rank = ranks[agent.agent_id]
        
        # ツモ和了 (テンパイ時のみ)
        if agent.shanten == 0 and random.random() < 0.02: 
            return agent.agent_id, None, True, oya_id, ranks
            
        action = agent.act_discard(turn, is_oya, rank)
        
        for offset in range(1, 4):
            other = agents[(turn + offset) % 4]
            is_other_oya = (other.agent_id == oya_id)
            other_rank = ranks[other.agent_id]
            resp = other.act_response(is_other_oya, other_rank)
            
            if resp == "ron":
                # ロンはテンパイ者のみ可能とみなす (モック)
                other.shanten = 0 
                return other.agent_id, agent.agent_id, False, oya_id, ranks
            elif resp == "call": break
            
    return None, None, False, oya_id, ranks

def distribute_rewards(agents, winner, loser, is_tsumo, oya_id, ranks):
    deltas = {i: 0 for i in range(4)}
    for a in agents:
        if a.is_riichi: deltas[a.agent_id] -= 1000
        # 終了時のシャンテン数を記録
        if a.shanten == 0: a.stats["kyoku_sh0"] += 1
        elif a.shanten == 1: a.stats["kyoku_sh1"] += 1
        else: a.stats["kyoku_sh2"] += 1

    base_points = 8000
    if winner is not None:
        win_points = int(base_points * 1.5) if winner == oya_id else base_points
        if is_tsumo:
            deltas[winner] += win_points
            for i in range(4):
                if i != winner:
                    payment = win_points // 3
                    if winner != oya_id and i == oya_id: payment *= 2
                    deltas[i] -= payment
        else:
            deltas[winner] += win_points
            deltas[loser] -= win_points
            
        agents[winner].stats["agari_count"] += 1
        
        if not is_tsumo:
            agents[loser].stats["houju_count"] += 1
            # ⭐️ レビューアー指定: 放銃時のシャンテン数を記録 (無駄押しの検出)
            if agents[loser].shanten == 0: agents[loser].stats["houju_sh0"] += 1
            elif agents[loser].shanten == 1: agents[loser].stats["houju_sh1"] += 1
            else: agents[loser].stats["houju_sh2"] += 1
            
    else:
        tc = sum([random.choice([True, False]) for _ in range(4)])
        if 0 < tc < 4:
            for i in range(4): deltas[i] += (3000 // tc) if random.choice([True, False]) else -(3000 // (4 - tc))
            
    for a in agents:
        aid = a.agent_id
        a.stats["total_score"] += deltas[aid]
        if a.shanten == 0: a.stats["score_sh0"] += deltas[aid]
        elif a.shanten == 1: a.stats["score_sh1"] += deltas[aid]
        else: a.stats["score_sh2"] += deltas[aid]

# =========================================
# 🎬 4. フルバッチ実行
# =========================================
def evaluate_config(models, config_name, config_type, num_games=5000):
    agents = [
        ShantenAwareAgent(0, models, config_name, config_type),
        ShantenAwareAgent(1, models, "Base", "A"), 
        ShantenAwareAgent(2, models, "Base", "A"),
        ShantenAwareAgent(3, models, "Base", "A")
    ]
    for _ in range(num_games):
        w, l, t, oya, ranks = run_kyoku(agents)
        distribute_rewards(agents, w, l, t, oya, ranks)
        
    return agents[0].stats

def main():
    models = ModelManager()
    num_games = 5000 
    
    print(f"\n🚀 Phase 10D-3a: シャンテン数差分シミュレーションを開始します... (各 {num_games}局)")
    
    configs = [
        ("パターンA: 現行ベース (親/子 + ラス目攻撃)", "A"),
        ("パターンB: A + テンパイ/1向聴で攻撃強化", "B"),
        ("パターンC: B + 2向聴以上で守備強化 (完全連動)", "C")
    ]

    results = []
    start_time = time.time()
    for name, c_type in configs:
        print(f"🔄 評価中: {name} ...")
        stats = evaluate_config(models, name, c_type, num_games)
        results.append((name, stats))
    print(f"✅ 全シミュレーション完了 ({time.time() - start_time:.1f}秒)\n")

    print("👑 === Phase 10D-3a: シャンテン数別 KPI 詳細レポート ===")
    for name, s in results:
        tk = s["total_kyoku"]
        k0, k1, k2 = max(1, s["kyoku_sh0"]), max(1, s["kyoku_sh1"]), max(1, s["kyoku_sh2"])
        
        avg_tot = s["total_score"] / tk
        t_opps = max(1, s["riichi_count"] + s["call_count"] + s["riichi_overrides"] + s["call_overrides"])
        ovr_rate = (s["riichi_overrides"] + s["call_overrides"]) / t_opps * 100
        
        print(f"■ {name}")
        print(f"  [総合] 局収支: {avg_tot:>6.1f} | 和了: {s['agari_count']/tk*100:4.1f}% | 放銃: {s['houju_count']/tk*100:4.1f}% | OVR率: {ovr_rate:4.1f}%")
        print(f"  [0向聴(ﾃﾝﾊﾟｲ)] 局収支: {s['score_sh0']/k0:>6.1f} | 放銃率: {s['houju_sh0']/k0*100:4.1f}%")
        print(f"  [1向聴]        局収支: {s['score_sh1']/k1:>6.1f} | 放銃率: {s['houju_sh1']/k1*100:4.1f}%")
        print(f"  [2向聴以上]    局収支: {s['score_sh2']/k2:>6.1f} | 放銃率: {s['houju_sh2']/k2*100:4.1f}% (※無駄押し指標)")
        print("-" * 90)

if __name__ == "__main__":
    main()