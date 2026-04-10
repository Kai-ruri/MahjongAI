import random
import numpy as np

# =========================================
# 🤖 1. エージェント定義 (固定方策＋スタッツ管理)
# =========================================
class HybridMahjongAgent:
    def __init__(self, agent_id, name, alpha=0.01, beta=0.03):
        self.agent_id = agent_id
        self.name = name
        self.alpha = alpha  # 攻めEVの係数
        self.beta = beta    # 守りEVの係数
        
        # 状態フラグ
        self.is_riichi = False
        self.has_called = False
        
        # 📊 評価用スタッツ
        self.stats = {
            "kyoku_count": 0,
            "agari_count": 0,    # 和了回数
            "houju_count": 0,    # 放銃回数
            "riichi_count": 0,   # リーチ回数
            "call_count": 0,     # 副露回数
            "total_score_delta": 0 # 累積局収支
        }

    def reset_kyoku(self):
        self.is_riichi = False
        self.has_called = False
        self.stats["kyoku_count"] += 1

    def act_discard(self, turn_count):
        """ EV係数に応じた打牌行動のモック """
        # β(守備)が高いとリーチしにくく、α(攻撃)が高いとリーチしやすい
        riichi_prob = 0.05 + (self.alpha * 0.5) - (self.beta * 0.5)
        
        if not self.is_riichi and not self.has_called and turn_count > 6 and random.random() < max(0.01, riichi_prob):
            self.is_riichi = True
            self.stats["riichi_count"] += 1
            return "riichi_discard"
        return "discard"

    def act_response(self, discarded_by, is_riichi_turn):
        """ EV係数に応じた他家打牌への反応モック """
        if discarded_by == self.agent_id: return "pass"
            
        # 誰かがリーチしている場合のロン確率 (危険な場)
        ron_prob = 0.02
        
        # β(守備)が高いと鳴きにくい、α(攻撃)が高いと鳴きやすい
        call_prob = 0.05 + (self.alpha * 0.5) - (self.beta * 0.5)
        
        if random.random() < ron_prob:
            return "ron"
        elif not self.is_riichi and random.random() < max(0.01, call_prob):
            self.has_called = True
            self.stats["call_count"] += 1
            return "call"
        return "pass"

# =========================================
# 🌍 2. 自己対戦環境 (ゼロサム完全対応版)
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
            
            # 1. ツモ和了判定
            if random.random() < 0.015:
                return self._distribute_rewards(winner_id=agent.agent_id, loser_id=None, is_tsumo=True)

            # 2. 打牌行動
            action = agent.act_discard(turn_count)
            is_riichi_turn = (action == "riichi_discard")
            
            # 3. 他家の反応
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
            
        # 4. 流局処理
        return self._distribute_rewards(winner_id=None, loser_id=None, is_tsumo=False)

    def _distribute_rewards(self, winner_id, loser_id, is_tsumo):
        """ 完全なゼロサム局収支計算 """
        deltas = {0: 0, 1: 0, 2: 0, 3: 0}
        kyoutaku = 0
        
        # リーチ棒の支払い (1000点)
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
                    
        # 収支を各エージェントのスタッツに反映
        for a in self.agents:
            a.stats["total_score_delta"] += deltas[a.agent_id]
            
        return deltas

# =========================================
# 🎬 3. 大規模対戦実行と評価レポート
# =========================================
def run_phase10c2_evaluation():
    # Target(0) は守備寄り、他は標準で設定
    agents = [
        HybridMahjongAgent(0, "Target ボット (守備重視)", alpha=0.01, beta=0.08),
        HybridMahjongAgent(1, "Base ボット 1 (標準)", alpha=0.02, beta=0.03),
        HybridMahjongAgent(2, "Base ボット 2 (標準)", alpha=0.02, beta=0.03),
        HybridMahjongAgent(3, "Base ボット 3 (標準)", alpha=0.02, beta=0.03)
    ]
    
    env = MahjongSelfPlayEnv(agents)
    num_games = 10000  # 🌟 1万局の高速シミュレーション
    
    print(f"🚀 フェーズ10C-2: 固定方策の自己対戦評価を開始します... ({num_games}局)")
    
    for game in range(1, num_games + 1):
        env.step_kyoku()
        if game % 2500 == 0:
            print(f"🔄 処理中... {game}局完了")

    print("\n" + "="*70)
    print("👑 フェーズ10C-2: 10,000局 自己対戦スタッツレポート")
    print("="*70)
    
    for a in agents:
        s = a.stats
        k = s["kyoku_count"]
        agari_rate = s["agari_count"] / k * 100
        houju_rate = s["houju_count"] / k * 100
        riichi_rate = s["riichi_count"] / k * 100
        call_rate = s["call_count"] / k * 100
        avg_score = s["total_score_delta"] / k
        
        print(f"👤 [Player {a.agent_id}] {a.name}")
        print(f"  - 和了率: {agari_rate:.1f}% | 放銃率: {houju_rate:.1f}%")
        print(f"  - リーチ率: {riichi_rate:.1f}% | 副露率: {call_rate:.1f}%")
        print(f"  - 平均局収支: {avg_score:+.1f} 点/局")
        print("-" * 70)

if __name__ == "__main__":
    run_phase10c2_evaluation()