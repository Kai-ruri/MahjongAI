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

# 以前のモデル (リーチ用等)
class ActionCNN(nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + aux_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

# 🌟 新規追加: 再学習した「深い」鳴きモデル専用のクラス
class CallCNN_Deep(nn.Module):
    def __init__(self, aux_dim=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 + aux_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, t, a):
        return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

class EV_CNN_Base3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 10 + 35, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, m, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), m, a], dim=1))

class ModelManager:
    def __init__(self):
        self.riichi_net = ActionCNN(9).to(device)
        self.call_net = CallCNN_Deep(10).to(device)  # ⭐️ ここを深いクラスに変更！
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
# 🛡️ 2. 危険度モック生成
# =========================================
def generate_mock_hand_with_safety():
    hand = []
    has_genbutsu, has_semi_safe = False, False
    for _ in range(14):
        r = random.random()
        if r < 0.15: danger = 0; has_genbutsu = True
        elif r < 0.40: danger = 1; has_semi_safe = True
        else: danger = 2
        hand.append(danger)
    return hand, has_genbutsu, has_semi_safe

# =========================================
# 🤖 3. 評価用エージェント (Optuna Rank1の設定)
# =========================================
class RealEvalAgent:
    def __init__(self, agent_id, models):
        self.agent_id = agent_id
        self.models = models
        
        # Optuna Rank1 (最強設定) のパラメータ
        self.a_base = 0.0256
        self.b_base = 0.1776
        self.a_good = 0.0398
        self.a_last = 0.0066
        self.a_dealer = 0.0095
        
        self.stats = {
            "total_kyoku": 0, "total_score": 0, 
            "riichi_count": 0, "call_count": 0, "riichi_overrides": 0, "call_overrides": 0,
            "agari_count": 0, "houju_count": 0
        }
        self.reset_kyoku()

    def reset_kyoku(self):
        self.is_riichi = False
        self.has_called = False
        self.shanten = 2
        self.stats["total_kyoku"] += 1

    def advance_shanten(self):
        if self.shanten > 0: self.shanten -= 1

    def get_coeffs(self, is_oya, rank):
        alpha = self.a_base
        beta = self.b_base
        if is_oya:
            alpha += self.a_dealer
            beta -= 0.04 
        if rank == 3: 
            alpha += self.a_last
            beta -= 0.01 
        if self.shanten <= 1:
            alpha += self.a_good
            if self.shanten == 0: beta -= 0.01 
            if self.shanten == 1: beta -= 0.005 
        if rank == 0:
            alpha -= 0.005
            beta += 0.01
        return max(0.0, alpha), max(0.0, beta)

    def _get_mock_inputs(self):
        t = (torch.rand((1, 25, 34)) < 0.05).float().to(device)
        m = torch.rand((1, 10)).to(device)
        return t, m

    def act_discard(self, turn_count, is_oya, rank, is_riichi_others):
        if self.is_riichi or self.has_called or turn_count <= 6: return "discard"

        alpha, beta = self.get_coeffs(is_oya, rank)
        t, m = self._get_mock_inputs()
        
        pol_push = random.uniform(0.45, 0.55)
        pol_fold = 1.0 - pol_push
        a_vec_push = torch.zeros((1, 35)).to(device); a_vec_push[0, 0] = 2.0
        a_vec_fold = torch.zeros((1, 35)).to(device); a_vec_fold[0, 0] = 0.0

        with torch.no_grad():
            oya_bonus = 0.15 if is_oya else 0.0
            sh_bonus = 0.1 if self.shanten <= 1 else 0.0
            plus_push = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_push)).item() + 0.4 + oya_bonus + sh_bonus
            minus_push = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec_push)).item() + 0.3
            plus_fold = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_fold)).item() + 0.3
            minus_fold = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec_fold)).item() + 0.1

        score_push = pol_push + (alpha * plus_push) - (beta * minus_push)
        score_fold = pol_fold + (alpha * plus_fold) - (beta * minus_fold)
        is_pushing = score_push > score_fold

        if self.shanten == 0 and is_pushing:
            self.is_riichi = True
            self.stats["riichi_count"] += 1
            return "riichi_discard"

        self.hand, self.has_genbutsu, self.has_semi_safe = generate_mock_hand_with_safety()
        logits = np.random.randn(14) 
        
        is_fold_situation = is_riichi_others and self.shanten >= 2 and not self.is_riichi
        if is_fold_situation and not is_pushing:
            for i in range(14):
                if self.hand[i] == 0: logits[i] += 5.0
                elif self.hand[i] == 1: logits[i] += 2.0
                elif self.hand[i] == 2: logits[i] -= 2.0

        chosen_idx = np.argmax(logits)
        chosen_danger = self.hand[chosen_idx]

        if is_fold_situation:
            if chosen_danger == 2 and random.random() < 0.10: return "houju"
            elif chosen_danger == 1 and random.random() < 0.01: return "houju"

        return "discard"

    def act_response(self, is_oya, rank):
        if self.is_riichi: return "pass"
        if random.random() < 0.02: return "ron" 
        if self.shanten == 0 or random.random() > 0.15: return "pass" 

        alpha, beta = self.get_coeffs(is_oya, rank)
        t, m = self._get_mock_inputs()
        
        # ⭐️ ここが最重要！新しくなった本物の call_net の予測確率を取得
        with torch.no_grad():
            call_logit = self.models.call_net(t, m).item()
            pol_call = torch.sigmoid(torch.tensor(call_logit)).item()
            
        pol_pass = 1.0 - pol_call
        a_vec_c = torch.zeros((1, 35)).to(device); a_vec_c[0, 0] = 1.0
        a_vec_p = torch.zeros((1, 35)).to(device); a_vec_p[0, 0] = 0.0

        with torch.no_grad():
            oya_bonus = 0.1 if is_oya else 0.0
            sh_bonus = 0.1 if self.shanten == 1 else 0.0
            plus_c = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_c)).item() + 0.3 + oya_bonus + sh_bonus
            minus_c = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec_c)).item() + 0.2
            plus_p = torch.sigmoid(self.models.ev_plus_net(t, m, a_vec_p)).item()
            minus_p = torch.sigmoid(self.models.ev_minus_net(t, m, a_vec_p)).item()

        score_c = pol_call + (alpha * plus_c) - (beta * minus_c)
        score_p = pol_pass + (alpha * plus_p) - (beta * minus_p)

        if score_c > score_p:
            self.has_called = True
            self.stats["call_count"] += 1
            self.advance_shanten()
            return "call"
        else:
            if pol_call > pol_pass: self.stats["call_overrides"] += 1
            return "pass"

# =========================================
# 🌍 4. 環境ランナー
# =========================================
def run_kyoku(agents):
    for a in agents: a.reset_kyoku()
    oya_id = random.randint(0, 3) 
    ranks = [0, 1, 2, 3]
    random.shuffle(ranks)
    
    riichi_turn = random.randint(5, 12)
    riichi_agent_id = random.randint(0, 3)
    
    for turn in range(70):
        agent = agents[turn % 4]
        if random.random() < 0.10: agent.advance_shanten()
        
        is_oya = (agent.agent_id == oya_id)
        rank = ranks[agent.agent_id]
        
        if turn == riichi_turn:
            agents[riichi_agent_id].is_riichi = True
            agents[riichi_agent_id].shanten = 0
            
        is_riichi_others = any(a.is_riichi for a in agents if a.agent_id != agent.agent_id)
        
        if agent.shanten == 0 and random.random() < 0.02: 
            return agent.agent_id, None, True, oya_id
            
        action = agent.act_discard(turn, is_oya, rank, is_riichi_others)
        if action == "houju":
            riichis = [a.agent_id for a in agents if a.is_riichi and a.agent_id != agent.agent_id]
            winner = riichis[0] if riichis else (agent.agent_id + 1) % 4
            return winner, agent.agent_id, False, oya_id
        
        for offset in range(1, 4):
            other = agents[(turn + offset) % 4]
            is_other_oya = (other.agent_id == oya_id)
            other_rank = ranks[other.agent_id]
            resp = other.act_response(is_other_oya, other_rank)
            
            if resp == "ron":
                other.shanten = 0 
                return other.agent_id, agent.agent_id, False, oya_id
            elif resp == "call": break
            
    return None, None, False, oya_id

def distribute_rewards(agents, winner, loser, is_tsumo, oya_id):
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
        if not is_tsumo: agents[loser].stats["houju_count"] += 1
    else:
        tc = sum([random.choice([True, False]) for _ in range(4)])
        if 0 < tc < 4:
            for i in range(4): deltas[i] += (3000 // tc) if random.choice([True, False]) else -(3000 // (4 - tc))
            
    for a in agents:
        a.stats["total_score"] += deltas[a.agent_id]

# =========================================
# 🎬 5. メイン実行 (10,000局の検証)
# =========================================
def evaluate_real_model(num_games=10000):
    models = ModelManager()
    agents = [RealEvalAgent(i, models) for i in range(4)]
    for _ in range(num_games):
        w, l, t, oya = run_kyoku(agents)
        distribute_rewards(agents, w, l, t, oya)
    return agents[0].stats

def main():
    print("\n🚀 Phase 10G-3: 再学習済み本番モデル 統合シミュレーション (10,000局)")
    print(" ※ 新しい call_best.pth の『鳴き抑制効果』を測定します...\n")
    
    start_time = time.time()
    stats = evaluate_real_model(10000)
    print(f"✅ 全シミュレーション完了 ({time.time() - start_time:.1f}秒)\n")

    tk = max(1, stats["total_kyoku"])
    avg_tot = stats["total_score"] / tk
    t_opps = max(1, stats["call_count"] + stats["call_overrides"])
    ovr_rate = stats["call_overrides"] / t_opps * 100
    
    print("👑 === Phase 10G-3: 再学習後の統合評価 レポート ===")
    print(f"  [全体] 局収支: {avg_tot:>6.1f} | 和了率: {stats['agari_count']/tk*100:4.1f}% | 放銃率: {stats['houju_count']/tk*100:4.1f}%")
    print(f"  [健全性] リーチ率: {stats['riichi_count']/tk*100:4.1f}% | 副露率: {stats['call_count']/tk*100:4.1f}% | OVR率: {ovr_rate:4.1f}%")
    print("-" * 80)
    
    print("💡 【成功基準チェック】")
    print(f"   ・副露率が 30〜35% 台に下がったか？ -> {'✅' if stats['call_count']/tk*100 < 36 else '❌'}")
    print(f"   ・リーチ率が 5% 以上に回復したか？ -> {'✅' if stats['riichi_count']/tk*100 >= 5.0 else '❌'}")
    print(f"   ・OVR率が 20% 未満に下がったか？ -> {'✅' if ovr_rate < 20.0 else '❌'}")

if __name__ == "__main__":
    main()