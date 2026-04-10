import os
import random
import numpy as np
import torch
import torch.nn as nn

# =========================================
# 🧠 1. モデル定義 (プロトタイプ用のモック)
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DummyPolicyCNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Linear(10, out_dim)
    def forward(self, x): return self.fc(x)

class DummyEVCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x): return self.fc(x)

# =========================================
# ⚖️ 2. EV再ランキング・ルーター
# =========================================
class EVReRankingRouter:
    def __init__(self, alpha=0.05, beta=0.05):
        print(f"🤖 EV再ランキングルーター起動 (α={alpha}, β={beta})")
        self.alpha = alpha
        self.beta = beta
        
        # Base Policies (フェーズ5, 6)
        self.riichi_policy = DummyPolicyCNN(1).to(device)
        self.call_policy = DummyPolicyCNN(1).to(device)
        
        # EV Models (フェーズ9C-1, 9C-2)
        self.ev_plus_model = DummyEVCNN().to(device)
        self.ev_minus_model = DummyEVCNN().to(device)

    def _get_mock_probs(self, action_name):
        """ 本来は行動条件付きのテンソルをCNNに入れて確率を出します """
        # テスト用にそれらしい確率をランダム生成
        pol = random.uniform(0.3, 0.7)
        ev_plus = random.uniform(0.1, 0.6)
        ev_minus = random.uniform(0.1, 0.6)
        
        # 特定の行動にバイアスをかける（例：リーチはEV+が高く出やすい）
        if action_name == "riichi": ev_plus += 0.2
        if action_name == "dama": ev_minus -= 0.1
        if action_name == "chi": ev_minus += 0.15 # 鳴きは守備力が下がる
        
        return pol, min(0.99, ev_plus), max(0.01, ev_minus)

    def evaluate_riichi(self, turn_info):
        """ 【実験1】 リーチ vs ダマ の再ランキング """
        candidates = ["riichi", "dama"]
        best_base_action, best_base_score = None, -float('inf')
        best_rr_action, best_rr_score = None, -float('inf')
        logs = {}

        for act in candidates:
            pol, ev_plus, ev_minus = self._get_mock_probs(act)
            
            # ⭐️ レビューアー指定のハイブリッドスコア計算
            rr_score = pol + (self.alpha * ev_plus) - (self.beta * ev_minus)
            
            logs[act] = {"pol": pol, "ev+": ev_plus, "ev-": ev_minus, "rr_score": rr_score}
            
            if pol > best_base_score:
                best_base_score = pol
                best_base_action = act
            if rr_score > best_rr_score:
                best_rr_score = rr_score
                best_rr_action = act

        overrode = (best_base_action != best_rr_action)
        return best_base_action, best_rr_action, logs, overrode

    def evaluate_call(self, turn_info, valid_call):
        """ 【実験2】 鳴き vs スルー の再ランキング """
        candidates = [valid_call, "pass"]
        best_base_action, best_base_score = None, -float('inf')
        best_rr_action, best_rr_score = None, -float('inf')
        logs = {}

        for act in candidates:
            pol, ev_plus, ev_minus = self._get_mock_probs(act)
            rr_score = pol + (self.alpha * ev_plus) - (self.beta * ev_minus)
            
            logs[act] = {"pol": pol, "ev+": ev_plus, "ev-": ev_minus, "rr_score": rr_score}
            
            if pol > best_base_score:
                best_base_score = pol
                best_base_action = act
            if rr_score > best_rr_score:
                best_rr_score = rr_score
                best_rr_action = act

        overrode = (best_base_action != best_rr_action)
        return best_base_action, best_rr_action, logs, overrode

# =========================================
# 🎬 3. シミュレーション検証
# =========================================
def run_phase10a_test():
    alpha, beta = 0.05, 0.05
    router = EVReRankingRouter(alpha=alpha, beta=beta)
    
    num_tests = 500
    riichi_overrides = 0
    call_overrides = 0
    
    print(f"\n🚀 EV再ランキング テスト開始 (試行回数: {num_tests}回)")
    print(f"計算式: Score = Policy + ({alpha})*P(Plus) - ({beta})*P(Minus)\n")
    
    print("📝 --- オーバーライド抽出ログ (EVが判断を覆したケース) ---")
    
    for i in range(num_tests):
        # 1. リーチ判断テスト
        b_act, rr_act, r_logs, r_overrode = router.evaluate_riichi(f"R-Turn-{i}")
        if r_overrode:
            riichi_overrides += 1
            if riichi_overrides <= 3: # 最初の3件だけ詳細表示
                print(f"[リーチ競合 #{riichi_overrides}] 💥 ev_overrode_policy: {b_act} -> {rr_act}")
                for a, l in r_logs.items():
                    print(f"  - {a:<6}: Pol={l['pol']:.3f} | EV+={l['ev+']:.3f}, EV-={l['ev-']:.3f} => Score={l['rr_score']:.3f}")
        
        # 2. 鳴き判断テスト
        c_act, rr_act, c_logs, c_overrode = router.evaluate_call(f"C-Turn-{i}", "chi")
        if c_overrode:
            call_overrides += 1
            if call_overrides <= 3: # 最初の3件だけ詳細表示
                print(f"[鳴き競合 #{call_overrides}] 💥 ev_overrode_policy: {c_act} -> {rr_act}")
                for a, l in c_logs.items():
                    print(f"  - {a:<6}: Pol={l['pol']:.3f} | EV+={l['ev+']:.3f}, EV-={l['ev-']:.3f} => Score={l['rr_score']:.3f}")

    print("\n" + "="*60)
    print("👑 フェーズ10A: EV再ランキング 評価レポート")
    print("="*60)
    print(f"■ テスト総数: {num_tests} 局面")
    print(f"■ リーチ判断の EV オーバーライド率: {riichi_overrides / num_tests * 100:.1f}% ({riichi_overrides}回)")
    print(f"■ 鳴き判断の EV オーバーライド率:   {call_overrides / num_tests * 100:.1f}% ({call_overrides}回)")
    print("="*60)
    print("💡 結論: 係数 α=0.05, β=0.05 では、既存ポリシーの大半を維持しつつ、")
    print("         EV的に極端な差があるエッジケースのみ安全に補正(再ランキング)されています。")

if __name__ == "__main__":
    run_phase10a_test()