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
        self.riichi_net = ActionCNN(9).to(device)
        self.call_net = ActionCNN(10).to(device)
        self.ev_plus_net = EV_CNN_Base3().to(device)
        self.ev_minus_net = EV_CNN_Base3().to(device)
        self._load(self.riichi_net, "riichi_best.pth")
        self._load(self.call_net, "call_best.pth")
        self._load(self.ev_plus_net, "ev_plus_best.pth")
        self._load(self.ev_minus_net, "ev_minus_best.pth")
        self.riichi_net.eval(); self.call_net.eval()
        self.ev_plus_net.eval(); self.ev_minus_net.eval()

    def _load(self, model, filename):
        if os.path.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=device))

# =========================================
# 🤖 2. 状況別エージェント (修正版: ラス目特化)
# =========================================
class RankAwareAgent:
    def __init__(self, agent_id, models, config_name, use_rank_delta):
        self.agent_id = agent_id
        self.models = models
        self.config_name = config_name
        self.use_rank_delta = use_rank_delta 
        
        self.stats = {
            "total_kyoku": 0, "total_score": 0, 
            "riichi_count": 0, "call_count": 0, "riichi_overrides": 0, "call_overrides": 0,
            "agari_count": 0, "houju_count": 0,
            "top_kyoku": 0, "top_score": 0, "top_agari": 0, "top_houju": 0, "top_riichi": 0,
            "last_kyoku": 0, "last_score": 0, "last_agari": 0, "last_houju": 0, "last_riichi": 0
        }
        self.reset_kyoku()

    def reset_kyoku(self):
        self.is_riichi = False
        self.has_called = False
        self.stats["total_kyoku"] += 1

    def get_current_coeffs(self, is_oya, rank):
        """ 親/子ベース + 順位(Rank)差分の計算 """
        # ベース係数 (10D-1 ハイブリッド設定)
        alpha = 0.05 if is_oya else 0.03
        beta = 0.10 if is_oya else 0.14
        
        if self.use_rank_delta:
            # ⭐️ 修正: トップ目の差分を削除し、ラス目のみ攻撃強化
            if rank == 3: # ラス目
                alpha += 0.005
                beta -= 0.01
                
        return max(0.0, alpha), max(0.0, beta)

    def _get_mock_inputs(self):
        t = (torch.rand((1, 25, 34)) < 0.05).float().to(device)
        m = torch.rand((1, 10)).to(device)
        return t, m

    def act_discard(self, turn_count, is_oya, rank):
        if self.is_riichi or self.has_called or turn_count <= 6: return "discard"
        if random.random() > 0.10: return "discard"

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
            if rank == 0: self.stats["top_riichi"] += 1
            if rank == 3: self.stats["last_riichi"] += 1
            return "riichi_discard"
        return "discard"

    def act_response(self, is_oya, rank):
        if self.is_riichi: return "pass"
        if random.random() < 0.02: return "ron" 
        if random.random() > 0.15: return "pass" 

        alpha, beta = self.get_current_coeffs(is_oya, rank)
        t, m = self._get_mock_inputs()
        
        pol_call = random.uniform(0.45, 0.55)
        pol_pass = 1.0 - pol_call

        a_vec_c = torch.zeros((1, 35)).to(device); a_vec_c[0, 0] = 1.0
        a_vec_p = torch.zeros((1, 35)).to(device); a_vec_p[0, 0] = 0.0

        with torch.no_grad():
            oya_bonus = 0.1 if is_oya else 0.0
            plus_c = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_c)).item() + 0.3 + oya_bonus
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
        is_oya = (agent.agent_id == oya_id)
        rank = ranks[agent.agent_id]
        
        if random.random() < 0.015: return agent.agent_id, None, True, oya_id, ranks
        action = agent.act_discard(turn, is_oya, rank)
        
        for offset in range(1, 4):
            other = agents[(turn + offset) % 4]
            is_other_oya = (other.agent_id == oya_id)
            other_rank = ranks[other.agent_id]
            resp = other.act_response(is_other_oya, other_rank)
            if resp == "ron": return other.agent_id, agent.agent_id, False, oya_id, ranks
            elif resp == "call": break
    return None, None, False, oya_id, ranks

def distribute_rewards(agents, winner, loser, is_tsumo, oya_id, ranks):
    deltas = {i: 0 for i in range(4)}
    for a in agents:
        if a.is_riichi: deltas[a.agent_id] -= 1000

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
        if ranks[winner] == 0: agents[winner].stats["top_agari"] += 1
        if ranks[winner] == 3: agents[winner].stats["last_agari"] += 1
        
        if not is_tsumo:
            agents[loser].stats["houju_count"] += 1
            if ranks[loser] == 0: agents[loser].stats["top_houju"] += 1
            if ranks[loser] == 3: agents[loser].stats["last_houju"] += 1
    else:
        tc = sum([random.choice([True, False]) for _ in range(4)])
        if 0 < tc < 4:
            for i in range(4): deltas[i] += (3000 // tc) if random.choice([True, False]) else -(3000 // (4 - tc))
            
    for a in agents:
        aid = a.agent_id
        a.stats["total_score"] += deltas[aid]
        rank = ranks[aid]
        if rank == 0:
            a.stats["top_kyoku"] += 1
            a.stats["top_score"] += deltas[aid]
        elif rank == 3:
            a.stats["last_kyoku"] += 1
            a.stats["last_score"] += deltas[aid]

# =========================================
# 🎬 4. フルバッチ実行
# =========================================
def evaluate_config(models, config_name, use_rank_delta, num_games=5000):
    agents = [
        RankAwareAgent(0, models, config_name, use_rank_delta),
        RankAwareAgent(1, models, "Base", False), 
        RankAwareAgent(2, models, "Base", False),
        RankAwareAgent(3, models, "Base", False)
    ]
    for _ in range(num_games):
        w, l, t, oya, ranks = run_kyoku(agents)
        distribute_rewards(agents, w, l, t, oya, ranks)
        
    return agents[0].stats

def main():
    models = ModelManager()
    num_games = 5000 
    
    print(f"\n🚀 Phase 10D-2 (修正版): 順位状況 (ラス目特化) 差分シミュレーションを開始します... (各 {num_games}局)")
    
    configs = [
        ("パターンA: 順位差分なし (10D-1ハイブリッド)", False),
        ("パターンB: ラス目のみ攻撃強化 (Δα=+0.005, Δβ=-0.01)", True)
    ]

    results = []
    start_time = time.time()
    for name, use_delta in configs:
        print(f"🔄 評価中: {name} ...")
        stats = evaluate_config(models, name, use_delta, num_games)
        results.append((name, stats))
    print(f"✅ 全シミュレーション完了 ({time.time() - start_time:.1f}秒)\n")

    print("👑 === Phase 10D-2 (修正版): 順位状況 KPI 詳細レポート ===")
    for name, s in results:
        tk = s["total_kyoku"]
        top_k = max(1, s["top_kyoku"])
        last_k = max(1, s["last_kyoku"])
        
        avg_tot = s["total_score"] / tk
        avg_top = s["top_score"] / top_k
        avg_last = s["last_score"] / last_k
        
        t_opps = max(1, s["riichi_count"] + s["call_count"] + s["riichi_overrides"] + s["call_overrides"])
        ovr_rate = (s["riichi_overrides"] + s["call_overrides"]) / t_opps * 100
        
        print(f"■ {name}")
        print(f"  [総合] 局収支: {avg_tot:>6.1f} | OVR率: {ovr_rate:4.1f}%")
        print(f"  [トップ目] 局収支: {avg_top:>6.1f} | 放銃率: {s['top_houju']/top_k*100:4.1f}% | リーチ率: {s['top_riichi']/top_k*100:4.1f}%")
        print(f"  [ラス目]   局収支: {avg_last:>6.1f} | 和了率: {s['last_agari']/last_k*100:4.1f}% | リーチ率: {s['last_riichi']/last_k*100:4.1f}%")
        print("-" * 90)

if __name__ == "__main__":
    main()