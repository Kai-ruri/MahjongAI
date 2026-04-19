import os
import random
import numpy as np
import torch
import torch.nn as nn
import time
import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)

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
# 🤖 3. 拡張最適化用エージェント (5変数探索)
# =========================================
class RLExtendedAgent:
    def __init__(self, agent_id, models, params):
        self.agent_id = agent_id
        self.models = models
        
        # ⭐️ Optunaから渡される5つの探索パラメータ
        self.opt_a_base = params["a_base"]
        self.opt_b_base = params["b_base"]
        self.opt_a_good = params["a_good"]
        self.opt_a_last = params["a_last"]
        self.opt_a_dealer = params["a_dealer"]
        
        self.stats = {
            "total_kyoku": 0, "total_score": 0, 
            "riichi_count": 0, "call_count": 0, "agari_count": 0, "houju_count": 0,
            # 条件別KPI
            "oya_kyoku": 0, "oya_score": 0,
            "last_kyoku": 0, "last_agari": 0,
            "good_shape_situations": 0, "good_shape_riichi": 0
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
        alpha = self.opt_a_base
        beta = self.opt_b_base
        
        # 探索: 親番ボーナス (beta減算は固定値として維持)
        if is_oya:
            alpha += self.opt_a_dealer
            beta -= 0.04 
            
        # 探索: ラス目ボーナス
        if rank == 3: 
            alpha += self.opt_a_last
            beta -= 0.01 
            
        # 探索: 好形(0,1向聴)ボーナス
        if self.shanten <= 1:
            alpha += self.opt_a_good
            if self.shanten == 0: beta -= 0.01 
            if self.shanten == 1: beta -= 0.005 

        # 固定ルール: トップ目守備 (ここは探索から除外)
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

        # KPI記録: 好形(1向聴以上)の局面数
        if self.shanten <= 1:
            self.stats["good_shape_situations"] += 1

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
            self.stats["good_shape_riichi"] += 1 # KPI: 好形時のリーチ
            return "riichi_discard"

        self.hand, self.has_genbutsu, self.has_semi_safe = generate_mock_hand_with_safety()
        logits = np.random.randn(14) 
        
        # 🛡️ 守備ルーター
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
        
        pol_call = random.uniform(0.45, 0.55)
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
            return agent.agent_id, None, True, oya_id, ranks
            
        action = agent.act_discard(turn, is_oya, rank, is_riichi_others)
        if action == "houju":
            riichis = [a.agent_id for a in agents if a.is_riichi and a.agent_id != agent.agent_id]
            winner = riichis[0] if riichis else (agent.agent_id + 1) % 4
            return winner, agent.agent_id, False, oya_id, ranks
        
        for offset in range(1, 4):
            other = agents[(turn + offset) % 4]
            is_other_oya = (other.agent_id == oya_id)
            other_rank = ranks[other.agent_id]
            resp = other.act_response(is_other_oya, other_rank)
            
            if resp == "ron":
                other.shanten = 0 
                return other.agent_id, agent.agent_id, False, oya_id, ranks
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
        if ranks[winner] == 3: agents[winner].stats["last_agari"] += 1
        if not is_tsumo: agents[loser].stats["houju_count"] += 1
    else:
        tc = sum([random.choice([True, False]) for _ in range(4)])
        if 0 < tc < 4:
            for i in range(4): deltas[i] += (3000 // tc) if random.choice([True, False]) else -(3000 // (4 - tc))
            
    for a in agents:
        aid = a.agent_id
        a.stats["total_score"] += deltas[aid]
        if aid == oya_id:
            a.stats["oya_kyoku"] += 1
            a.stats["oya_score"] += deltas[aid]
        if ranks[aid] == 3:
            a.stats["last_kyoku"] += 1

# =========================================
# 🎬 5. Optuna 目的関数
# =========================================
def objective(trial):
    # ⭐️ 探索空間の定義
    params = {
        "a_base": trial.suggest_float("a_base", 0.015, 0.04),
        "b_base": trial.suggest_float("b_base", 0.13, 0.19),
        "a_good": trial.suggest_float("a_good", 0.00, 0.04),
        "a_last": trial.suggest_float("a_last", 0.00, 0.03),
        "a_dealer": trial.suggest_float("a_dealer", 0.00, 0.03)
    }
    
    models = ModelManager()
    num_games = 3000 
    
    agents = [RLExtendedAgent(i, models, params) for i in range(4)]
    
    for _ in range(num_games):
        w, l, t, oya, ranks = run_kyoku(agents)
        distribute_rewards(agents, w, l, t, oya, ranks)
        
    stats = agents[0].stats
    tk = max(1, stats["total_kyoku"])
    oya_k = max(1, stats["oya_kyoku"])
    last_k = max(1, stats["last_kyoku"])
    gs_sit = max(1, stats["good_shape_situations"])
    
    # 評価指標の計算
    avg_score = stats["total_score"] / tk
    
    # サブ指標を記録 (Optunaのログで後から確認できるように)
    trial.set_user_attr("houju_rate", stats["houju_count"] / tk * 100)
    trial.set_user_attr("agari_rate", stats["agari_count"] / tk * 100)
    trial.set_user_attr("oya_score", stats["oya_score"] / oya_k)
    trial.set_user_attr("last_agari_rate", stats["last_agari"] / last_k * 100)
    trial.set_user_attr("gs_riichi_rate", stats["good_shape_riichi"] / gs_sit * 100)
    
    return avg_score

def main():
    print("\n🚀 Phase 10E-2: 拡張パラメータ探索 (5変数) を開始します...")
    print(" ※ 50 Trial 実行します。しばらくお待ちください...\n")
    
    start_time = time.time()
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    n_trials = 50
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n✅ 全最適化完了 ({time.time() - start_time:.1f}秒)")
    print("\n👑 === Phase 10E-2: 拡張最適化 結果レポート ===")
    
    best = study.best_trial
    print(f"【Best Trial Number】: {best.number}")
    print(f"  [全体] 局収支: {best.value:>6.1f} | 和了: {best.user_attrs['agari_rate']:4.1f}% | 放銃: {best.user_attrs['houju_rate']:4.1f}%")
    print(f"  [条件] 親番局収支: {best.user_attrs['oya_score']:>6.1f} | ラス目和了: {best.user_attrs['last_agari_rate']:4.1f}% | 好形リーチ: {best.user_attrs['gs_riichi_rate']:4.1f}%")
    
    print("\n【AIが発見した最強のアクセルワーク (5変数)】")
    for key, value in best.params.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()