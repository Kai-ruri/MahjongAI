import random
import numpy as np

# =========================================
# 🤖 1. エージェント定義
# =========================================
class HybridMahjongAgent:
    def __init__(self, agent_id, name, alpha=0.01, beta=0.03):
        self.agent_id = agent_id
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.is_riichi = False
        self.has_called = False
        self.stats = {
            "kyoku_count": 0, "agari_count": 0, "houju_count": 0,
            "riichi_count": 0, "call_count": 0, "total_score_delta": 0
        }

    def reset_kyoku(self):
        self.is_riichi = False
        self.has_called = False
        self.stats["kyoku_count"] += 1

    def act_discard(self, turn_count):
        # β(守備)が高いとリーチしにくく、α(攻撃)が高いとリーチしやすい
        riichi_prob = 0.05 + (self.alpha * 0.5) - (self.beta * 0.5)
        if not self.is_riichi and not self.has_called and turn_count > 6 and random.random() < max(0.001, riichi_prob):
            self.is_riichi = True
            self.stats["riichi_count"] += 1
            return "riichi_discard"
        return "discard"

    def act_response(self, discarded_by, is_riichi_turn):
        if discarded_by == self.agent_id: return "pass"
        ron_prob = 0.02
        call_prob = 0.05 + (self.alpha * 0.5) - (self.beta * 0.5)
        
        if random.random() < ron_prob: return "ron"
        elif not self.is_riichi and random.random() < max(0.001, call_prob):
            self.has_called = True
            self.stats["call_count"] += 1
            return "call"
        return "pass"

# =========================================
# 🌍 2. 自己対戦環境 (RL Env)
# =========================================
class MahjongSelfPlayEnv:
    def __init__(self, agents):
        self.agents = agents

    def step_kyoku(self):
        for a in self.agents: a.reset_kyoku()
        turn_count = 0
        current_player = 0
        
        while turn_count < 70:
            agent = self.agents[current_player]
            if random.random() < 0.015:
                return self._distribute_rewards(winner_id=agent.agent_id, loser_id=None, is_tsumo=True)

            action = agent.act_discard(turn_count)
            is_riichi_turn = (action == "riichi_discard")
            
            for offset in range(1, 4):
                other_id = (current_player + offset) % 4
                other_agent = self.agents[other_id]
                response = other_agent.act_response(discarded_by=agent.agent_id, is_riichi_turn=is_riichi_turn)
                if response == "ron":
                    return self._distribute_rewards(winner_id=other_agent.agent_id, loser_id=agent.agent_id, is_tsumo=False)
                elif response == "call":
                    current_player = other_agent.agent_id
                    break 

            current_player = (current_player + 1) % 4
            turn_count += 1
            
        return self._distribute_rewards(winner_id=None, loser_id=None, is_tsumo=False)

    def _distribute_rewards(self, winner_id, loser_id, is_tsumo):
        deltas = {0: 0, 1: 0, 2: 0, 3: 0}
        kyoutaku = 0
        for a in self.agents:
            if a.is_riichi:
                deltas[a.agent_id] -= 1000
                kyoutaku += 1000

        base_points = 8000 
        if winner_id is not None:
            self.agents[winner_id].stats["agari_count"] += 1
            if is_tsumo:
                deltas[winner_id] += base_points + kyoutaku
                for i in range(4):
                    if i != winner_id: deltas[i] -= base_points // 3
            else:
                self.agents[loser_id].stats["houju_count"] += 1
                deltas[winner_id] += base_points + kyoutaku
                deltas[loser_id] -= base_points
        else:
            tenpai_status = [random.choice([True, False]) for _ in range(4)]
            tenpai_count = sum(tenpai_status)
            if 0 < tenpai_count < 4:
                receive = 3000 // tenpai_count
                pay = 3000 // (4 - tenpai_count)
                for i in range(4):
                    deltas[i] += receive if tenpai_status[i] else -pay
                    
        for a in self.agents:
            a.stats["total_score_delta"] += deltas[a.agent_id]
        return deltas

# =========================================
# 🎬 3. フェーズ10C-3: グリッド探索＆上位再評価
# =========================================
def evaluate_target(alpha, beta, num_games):
    """ Target(探索対象) 1人 vs Base(固定) 3人 のシミュレーション """
    agents = [
        HybridMahjongAgent(0, "Target", alpha=alpha, beta=beta),
        HybridMahjongAgent(1, "Base 1", alpha=0.02, beta=0.03), # 基準となる強い固定ボット
        HybridMahjongAgent(2, "Base 2", alpha=0.02, beta=0.03),
        HybridMahjongAgent(3, "Base 3", alpha=0.02, beta=0.03)
    ]
    env = MahjongSelfPlayEnv(agents)
    for _ in range(num_games):
        env.step_kyoku()
        
    s = agents[0].stats
    k = s["kyoku_count"]
    return {
        "alpha": alpha, "beta": beta,
        "avg_score": s["total_score_delta"] / k,
        "agari_rate": s["agari_count"] / k * 100,
        "houju_rate": s["houju_count"] / k * 100,
        "riichi_rate": s["riichi_count"] / k * 100,
        "call_rate": s["call_count"] / k * 100
    }

def print_result_row(res, rank=""):
    print(f" {rank:<3} | ({res['alpha']:4.2f}, {res['beta']:4.2f}) | {res['avg_score']:>7.1f} | {res['agari_rate']:>5.1f}% | {res['houju_rate']:>5.1f}% | {res['riichi_rate']:>5.1f}% | {res['call_rate']:>5.1f}%")

def run_phase10c3_optimization():
    # 🌟 探索候補 (レビューアーの提案レンジを中心に)
    candidates = [
        (0.00, 0.00), (0.01, 0.03), (0.01, 0.05), (0.02, 0.03), 
        (0.02, 0.05), (0.03, 0.05), (0.02, 0.08), (0.04, 0.08), 
        (0.05, 0.10), (0.08, 0.12)
    ]
    
    print("🚀 [Phase 1: 粗評価] 各設定 3,000局のシミュレーションを開始します...")
    rough_results = []
    for a, b in candidates:
        res = evaluate_target(a, b, 3000)
        rough_results.append(res)
        
    # 🏆 目的関数(局収支)で降順ソート
    rough_results.sort(key=lambda x: x["avg_score"], reverse=True)
    
    print("\n--- 粗評価結果ランキング (3,000局) ---")
    print(" Rank| 設定(α, β)   | 局収支  | 和了率 | 放銃率 | リーチ | 副露率")
    print("-" * 75)
    for i, res in enumerate(rough_results):
        print_result_row(res, f"#{i+1}")
        
    top_k = 3
    print(f"\n🚀 [Phase 2: 本評価] 上位 {top_k} 設定について 10,000局の再評価を開始します...")
    final_results = []
    for res in rough_results[:top_k]:
        final_res = evaluate_target(res["alpha"], res["beta"], 10000)
        final_results.append(final_res)
        
    final_results.sort(key=lambda x: x["avg_score"], reverse=True)
    
    print("\n👑 --- 最終結果 (10,000局) ---")
    print(" Rank| 設定(α, β)   | 局収支  | 和了率 | 放銃率 | リーチ | 副露率")
    print("=" * 75)
    for i, res in enumerate(final_results):
        print_result_row(res, f"#{i+1}")
    print("=" * 75)

if __name__ == "__main__":
    run_phase10c3_optimization()