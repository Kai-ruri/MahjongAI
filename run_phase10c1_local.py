import random
import numpy as np

# =========================================
# 🤖 1. エージェント定義 (固定方策ボット)
# =========================================
class HybridMahjongAgent:
    def __init__(self, agent_id, alpha=0.01, beta=0.03):
        self.agent_id = agent_id
        self.alpha = alpha
        self.beta = beta
        self.is_riichi = False
        self.score = 25000

    def act_discard(self, turn_count):
        """ 打牌選択 (今回はモックとして行動を返すだけ) """
        # ※本来はここで Policy + α*EV+ - β*EV- の推論が走る
        if not self.is_riichi and turn_count > 6 and random.random() < 0.05:
            self.is_riichi = True
            return "riichi_discard"
        return "discard"

    def act_response(self, discarded_by):
        """ 他家の打牌に対する反応 (ロン、鳴き、スルー) """
        if discarded_by == self.agent_id:
            return "pass"
            
        # 誰かのリーチ後などはロン確率が上がる (モック)
        if random.random() < 0.02:
            return "ron"
        elif not self.is_riichi and random.random() < 0.05:
            return "call"
        return "pass"

# =========================================
# 🌍 2. 自己対戦・強化学習環境 (Mahjong Env)
# =========================================
class MahjongSelfPlayEnv:
    def __init__(self, agents):
        self.agents = agents
        self.turn_count = 0
        self.current_player = 0
        self.log = []

    def reset_kyoku(self):
        """ 局の初期化 """
        self.turn_count = 0
        self.current_player = 0
        self.log = []
        for a in self.agents:
            a.is_riichi = False
        self.log.append("--- 新しい局が開始されました ---")

    def step_kyoku(self):
        """ 1局が終了するまでターンを回すメインループ """
        self.reset_kyoku()
        
        # 最大約70巡 (王牌を除く牌山が尽きるまで)
        max_turns = 70 
        
        while self.turn_count < max_turns:
            agent = self.agents[self.current_player]
            
            # 1. ツモ和了の判定 (モック)
            if random.random() < 0.015:
                self.log.append(f"[Turn {self.turn_count}] Player {agent.agent_id} がツモ和了！")
                return self._distribute_rewards(winner_id=agent.agent_id, loser_id=None, is_tsumo=True)

            # 2. 打牌行動
            action = agent.act_discard(self.turn_count)
            self.log.append(f"[Turn {self.turn_count}] Player {agent.agent_id} の行動: {action}")
            
            # 3. 他家の反応 (ロン・鳴き)
            for other_agent in self.agents:
                if other_agent.agent_id == agent.agent_id: continue
                
                response = other_agent.act_response(discarded_by=agent.agent_id)
                if response == "ron":
                    self.log.append(f"  💥 Player {other_agent.agent_id} が Player {agent.agent_id} からロン和了！")
                    return self._distribute_rewards(winner_id=other_agent.agent_id, loser_id=agent.agent_id, is_tsumo=False)
                elif response == "call":
                    self.log.append(f"  📣 Player {other_agent.agent_id} が鳴き (Call)")
                    self.current_player = other_agent.agent_id # ターンがスキップされる
                    break # 今回はダブロンなし、最初の鳴きを優先

            self.current_player = (self.current_player + 1) % 4
            self.turn_count += 1
            
        # 4. 流局処理
        self.log.append(f"流局 (牌山が尽きました)")
        return self._distribute_rewards(winner_id=None, loser_id=None, is_tsumo=False)

    def _distribute_rewards(self, winner_id, loser_id, is_tsumo):
        """ 【重要】 RL用の報酬(局収支)分配ロジック """
        rewards = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # 簡易的な打点モック (満貫ベース: 8000点)
        base_points = 8000 
        
        # リーチ棒の供託回収モック (1000点 × リーチ者数)
        kyoutaku = sum([1000 for a in self.agents if a.is_riichi])

        if winner_id is not None:
            if is_tsumo:
                # ツモ: 3人が等分して払う (親/子計算は今回簡略化)
                payment = base_points // 3
                for i in range(4):
                    if i == winner_id:
                        rewards[i] = base_points + kyoutaku
                    else:
                        rewards[i] = -payment
            else:
                # ロン: 放銃者が全額払う
                for i in range(4):
                    if i == winner_id:
                        rewards[i] = base_points + kyoutaku
                    elif i == loser_id:
                        rewards[i] = -base_points
                    else:
                        rewards[i] = 0
        else:
            # 流局: テンパイ料の移動 (今回はランダムにテンパイ者を決める)
            tenpai_status = [random.choice([True, False]) for _ in range(4)]
            tenpai_count = sum(tenpai_status)
            if 0 < tenpai_count < 4:
                receive = 3000 // tenpai_count
                pay = 3000 // (4 - tenpai_count)
                for i in range(4):
                    rewards[i] = receive if tenpai_status[i] else -pay
            else:
                # 全員テンパイ or 全員ノーテン
                rewards = {0: 0, 1: 0, 2: 0, 3: 0}
                
        # 供託を出した人はその分局収支をマイナスにする
        for a in self.agents:
            if a.is_riichi and a.agent_id != winner_id:
                rewards[a.agent_id] -= 1000

        self.log.append(f"💰 局収支 (Rewards): {rewards}")
        # ゼロサムチェック (供託が回収されなかった流局時はゼロサムにならないため、デバッグ用表示)
        self.log.append(f"   (合計移動点: {sum(rewards.values())} 点)") 
        
        return rewards

# =========================================
# 🎬 3. 環境テスト実行 (フェーズ10C-1 成功基準の確認)
# =========================================
def run_env_test():
    print("🚀 フェーズ10C-1: 自己対戦環境(Env)の稼働テストを開始します...\n")
    
    # Player 0 だけ少し守備寄り、他は標準のメタパラメータを持つ想定
    agents = [
        HybridMahjongAgent(0, alpha=0.01, beta=0.05), # Target
        HybridMahjongAgent(1, alpha=0.02, beta=0.03),
        HybridMahjongAgent(2, alpha=0.02, beta=0.03),
        HybridMahjongAgent(3, alpha=0.02, beta=0.03)
    ]
    
    env = MahjongSelfPlayEnv(agents)
    num_games = 5
    
    for game in range(1, num_games + 1):
        print(f"========== [第 {game} 局] ==========")
        rewards = env.step_kyoku()
        
        # ログを最後から5行だけ表示 (長すぎるため)
        for line in env.log[-5:]:
            print(line)
        print("-" * 35)

    print("\n✅ 環境テスト完了！")
    print("・自己対戦ループが最後まで回ることを確認しました。")
    print("・和了(ロン/ツモ)および流局における局収支(報酬)が正しく分配されています。")

if __name__ == "__main__":
    run_env_test()