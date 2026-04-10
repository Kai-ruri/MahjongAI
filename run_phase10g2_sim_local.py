import random
import numpy as np
import time

# =========================================
# 🤖 1. 再学習シミュレーション用エージェント
# =========================================
class ReTrainedAgent:
    def __init__(self, agent_id, exp_name, base_call_bias):
        self.agent_id = agent_id
        self.exp_name = exp_name
        # ⭐️ 本番のcall_net再学習による「鳴き意欲の低下」を擬似的に表現するパラメータ
        self.base_call_bias = base_call_bias 
        
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

    def act_discard(self, turn_count, is_oya, rank, is_riichi_others):
        if self.is_riichi or self.has_called or turn_count <= 6: return "discard"

        # EVによる押し引き判定 (モック)
        is_pushing = random.random() < 0.6 # 60%で押し判定
        
        # テンパイ時に押し判定ならリーチ (鳴いているとリーチできない)
        if self.shanten == 0 and is_pushing and not self.has_called:
            self.is_riichi = True
            self.stats["riichi_count"] += 1
            return "riichi_discard"

        # 守備ルーターによる放銃回避
        is_fold_situation = is_riichi_others and self.shanten >= 2 and not self.is_riichi
        if is_fold_situation:
            if random.random() < 0.05: return "houju" # ルーターが効いていて放銃率は低め
        else:
            if random.random() < 0.08: return "houju"

        return "discard"

    def act_response(self, is_oya, rank):
        if self.is_riichi: return "pass"
        if random.random() < 0.02: return "ron" 
        if self.shanten == 0 or random.random() > 0.15: return "pass" 

        # 1. ベースモデル(call_net)の出力 (モック)
        # 再学習で pass_weight を上げるほど、この base_prob が下がる
        base_prob = self.base_call_bias + random.uniform(-0.1, 0.1)
        base_wants_to_call = base_prob > 0.5

        # 2. EVモデルの判断 (モック: 冷静な期待値判断)
        ev_wants_to_call = random.random() < 0.35 # 本来鳴くべき局面は約35%と仮定

        # 3. 統合判断 (EVがベースの判断を上書き = Override)
        final_call = ev_wants_to_call
        
        # OVR(オーバーライド)の記録
        if base_wants_to_call != ev_wants_to_call:
            self.stats["call_overrides"] += 1

        if final_call:
            self.has_called = True
            self.stats["call_count"] += 1
            self.advance_shanten()
            return "call"
            
        return "pass"

# =========================================
# 🌍 2. 環境ランナー
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
# 🎬 3. メイン実行 (実験パターンの比較)
# =========================================
def evaluate_config(exp_name, base_call_bias, num_games=3000):
    agents = [ReTrainedAgent(i, exp_name, base_call_bias) for i in range(4)]
    for _ in range(num_games):
        w, l, t, oya = run_kyoku(agents)
        distribute_rewards(agents, w, l, t, oya)
    return agents[0].stats

def main():
    print("\n🚀 Phase 10G-2: 統合シミュレータ ミニ評価 (再学習の効果予測)")
    
    # ⭐️ ベースモデルの鳴き意欲 (再学習でこれを適正化していく)
    configs = [
        ("実験A: 現行loss (暴走状態)", 0.85), # 異常に鳴きたがる
        ("実験B: Focal Lossのみ", 0.60),      # Hard Negativeを罰して少し落ち着く
        ("実験C: Focal + pass_weight=1.5", 0.45), # バランスが良くなる (本命)
        ("実験D: Focal + pass_weight=1.8", 0.30)  # かなり門前寄りになる
    ]

    results = []
    start_time = time.time()
    for name, bias in configs:
        print(f"🔄 評価中: {name} ...")
        stats = evaluate_config(name, bias, 5000)
        results.append((name, stats))
    print(f"✅ 全シミュレーション完了 ({time.time() - start_time:.1f}秒)\n")

    print("👑 === Phase 10G-2: 統合評価 KPI 予測レポート ===")
    for name, s in results:
        tk = max(1, s["total_kyoku"])
        avg_tot = s["total_score"] / tk
        
        # OVR率の計算 (鳴き機会におけるEVとベースの不一致率)
        t_opps = max(1, s["call_count"] + s["call_overrides"])
        ovr_rate = s["call_overrides"] / t_opps * 100
        
        print(f"■ {name}")
        print(f"  副露率: {s['call_count']/tk*100:4.1f}% | リーチ率: {s['riichi_count']/tk*100:4.1f}% | OVR率: {ovr_rate:4.1f}%")
        print(f"  局収支: {avg_tot:>6.1f} | 和了率: {s['agari_count']/tk*100:4.1f}% | 放銃率: {s['houju_count']/tk*100:4.1f}%")
        print("-" * 80)

if __name__ == "__main__":
    main()