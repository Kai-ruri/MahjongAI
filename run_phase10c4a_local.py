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

class EV_CNN_Base3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 10 + 35, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, m, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), m, a], dim=1))

class ModelManager:
    def __init__(self):
        print(f"🤖 本番CNNモデルをロード中... ({device})")
        self.ev_plus_net = EV_CNN_Base3().to(device)
        self.ev_minus_net = EV_CNN_Base3().to(device)
        self._load(self.ev_plus_net, r"G:\マイドライブ\MahjongAI\ev_plus_best.pth")
        self._load(self.ev_minus_net, r"G:\マイドライブ\MahjongAI\ev_minus_best.pth")
        self.ev_plus_net.eval(); self.ev_minus_net.eval()

    def _load(self, model, filename):
        if os.path.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=device))

# =========================================
# 🤖 2. 本番エージェント (正しい確率発生版)
# =========================================
class RealCNNMahjongAgent:
    def __init__(self, agent_id, models, alpha, beta):
        self.agent_id = agent_id
        self.models = models
        self.alpha = alpha
        self.beta = beta
        self.is_riichi = False
        self.has_called = False
        self.stats = {
            "kyoku_count": 0, "agari_count": 0, "houju_count": 0,
            "riichi_count": 0, "call_count": 0, "total_score_delta": 0,
            "riichi_overrides": 0, "call_overrides": 0
        }

    def reset_kyoku(self):
        self.is_riichi = False
        self.has_called = False
        self.stats["kyoku_count"] += 1

    def _predict_ev(self, action_type):
        t = torch.zeros((1, 33, 34), dtype=torch.float32).to(device)
        m = torch.zeros((1, 10), dtype=torch.float32).to(device)
        a_vec = torch.zeros((1, 35), dtype=torch.float32).to(device)
        
        if action_type == "riichi": a_vec[0, 0] = 2.0
        elif action_type == "call": a_vec[0, 0] = 1.0
        elif action_type == "dama" or action_type == "pass": a_vec[0, 0] = 0.0

        with torch.no_grad():
            p_plus = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec)).item()
            p_minus = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec)).item()
        return p_plus, p_minus

    def act_discard(self, turn_count):
        if self.is_riichi or self.has_called or turn_count <= 6:
            return "discard"

        # ⭐️ 修正: テンパイ(リーチ機会)は局の途中で約10%の確率で訪れるとする
        if random.random() > 0.10:
            return "discard"

        # ⭐️ 修正: テンパイした時は、リーチとダマで本気で迷う (0.4 ~ 0.6)
        pol_riichi = random.uniform(0.4, 0.6)
        pol_dama = 1.0 - pol_riichi

        plus_r, minus_r = self._predict_ev("riichi")
        plus_d, minus_d = self._predict_ev("dama")

        score_r = pol_riichi + (self.alpha * plus_r) - (self.beta * minus_r)
        score_d = pol_dama + (self.alpha * plus_d) - (self.beta * minus_d)

        base_choice = "riichi" if pol_riichi > pol_dama else "dama"
        rr_choice = "riichi" if score_r > score_d else "dama"

        if base_choice != rr_choice:
            self.stats["riichi_overrides"] += 1

        if rr_choice == "riichi":
            self.is_riichi = True
            self.stats["riichi_count"] += 1
            return "riichi_discard"
        return "discard"

    def act_response(self):
        if self.is_riichi: return "pass"
        if random.random() < 0.02: return "ron" 

        # ⭐️ 修正: 鳴き機会は他家の打牌時に約15%の確率で訪れるとする
        if random.random() > 0.15:
            return "pass"

        # ⭐️ 修正: 鳴ける時は、鳴くかスルーかで迷う (0.4 ~ 0.6)
        pol_call = random.uniform(0.4, 0.6)
        pol_pass = 1.0 - pol_call
        
        plus_c, minus_c = self._predict_ev("call")
        plus_p, minus_p = self._predict_ev("pass")

        score_c = pol_call + (self.alpha * plus_c) - (self.beta * minus_c)
        score_p = pol_pass + (self.alpha * plus_p) - (self.beta * minus_p)

        base_choice = "call" if pol_call > pol_pass else "pass"
        rr_choice = "call" if score_c > score_p else "pass"

        if base_choice != rr_choice:
            self.stats["call_overrides"] += 1

        if rr_choice == "call":
            self.has_called = True
            self.stats["call_count"] += 1
            return "call"
        return "pass"

# =========================================
# 🌍 3. 自己対戦ループ & 実行部
# =========================================
def run_kyoku(agents):
    for a in agents: a.reset_kyoku()
    for turn in range(70):
        agent = agents[turn % 4]
        if random.random() < 0.015: return agent.agent_id, None, True
        action = agent.act_discard(turn)
        for offset in range(1, 4):
            other = agents[(turn + offset) % 4]
            resp = other.act_response()
            if resp == "ron": return other.agent_id, agent.agent_id, False
            elif resp == "call": break
    return None, None, False

def distribute_rewards(agents, winner, loser, is_tsumo):
    deltas = {i: 0 for i in range(4)}
    kyoutaku = sum([1000 for a in agents if a.is_riichi])
    for a in agents:
        if a.is_riichi: deltas[a.agent_id] -= 1000
    base = 8000
    if winner is not None:
        agents[winner].stats["agari_count"] += 1
        deltas[winner] += base + kyoutaku
        if is_tsumo:
            for i in range(4):
                if i != winner: deltas[i] -= base // 3
        else:
            agents[loser].stats["houju_count"] += 1
            deltas[loser] -= base
    else:
        tc = sum([random.choice([True, False]) for _ in range(4)])
        if 0 < tc < 4:
            for i in range(4): deltas[i] += (3000 // tc) if random.choice([True, False]) else -(3000 // (4 - tc))
    for a in agents: a.stats["total_score_delta"] += deltas[a.agent_id]

def main():
    models = ModelManager()
    
    agents = [
        RealCNNMahjongAgent(0, models, 0.08, 0.12),
        RealCNNMahjongAgent(1, models, 0.0, 0.0),
        RealCNNMahjongAgent(2, models, 0.0, 0.0),
        RealCNNMahjongAgent(3, models, 0.0, 0.0)
    ]
    
    num_games = 1000
    print(f"\n🚀 フェーズ10C-4A: 本番CNN × 固定係数 自己対戦評価を開始 ({num_games}局)")
    
    start_time = time.time()
    for game in range(1, num_games + 1):
        w, l, t = run_kyoku(agents)
        distribute_rewards(agents, w, l, t)
        if game % 250 == 0:
            print(f"🔄 処理中... {game}局完了 ({time.time() - start_time:.1f}秒経過)")

    print("\n" + "="*70)
    print(f"👑 フェーズ10C-4A: 本番CNN統合テスト 結果 ({num_games}局)")
    print("="*70)
    for a in agents:
        s = a.stats
        k = max(1, s["kyoku_count"])
        print(f"👤 [Player {a.agent_id}] (α={a.alpha}, β={a.beta})")
        print(f"  - 和了率: {s['agari_count']/k*100:.1f}% | 放銃率: {s['houju_count']/k*100:.1f}%")
        print(f"  - リーチ率: {s['riichi_count']/k*100:.1f}% | 副露率: {s['call_count']/k*100:.1f}%")
        print(f"  - 平均局収支: {s['total_score_delta']/k:+.1f} 点/局")
        if a.agent_id == 0:
            total_opps = s["riichi_count"] + s["call_count"] + s["riichi_overrides"] + s["call_overrides"]
            override_rate = (s["riichi_overrides"] + s["call_overrides"]) / max(1, total_opps) * 100
            print(f"  ★ Override率: {override_rate:.1f}% (リーチ補正:{s['riichi_overrides']}回, 鳴き補正:{s['call_overrides']}回)")
        print("-" * 70)

if __name__ == "__main__":
    main()