import os
import torch
import torch.nn as nn
import numpy as np
import time
import random

# =========================================
# 🧠 1. 現行の本番CNNモデル定義 (call_net)
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActionCNN(nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + aux_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, a): 
        return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), a], dim=1))

def load_call_net():
    model = ActionCNN(10).to(device)
    if os.path.exists(r"G:\マイドライブ\MahjongAI\call_best.pth"):
        model.load_state_dict(torch.load(r"G:\マイドライブ\MahjongAI\call_best.pth", map_location=device))
        print("✅ 現行の call_best.pth をロードしました。")
    else:
        print("⚠️ call_best.pth が見つかりません。未学習モデルでテストします。")
    model.eval()
    return model

# =========================================
# 📊 2. 【修正】構造的な疑似手牌ジェネレーター
# =========================================
def generate_structured_batch(batch_size):
    """ 単なる砂嵐ではなく、麻雀の手牌らしい構造(連続した1)を持つテンソルを生成 """
    t = torch.zeros((batch_size, 33, 34), device=device)
    a = torch.zeros((batch_size, 10), device=device)
    
    for i in range(batch_size):
        # 1. 疑似的な手牌の構築 (チャンネル0〜3あたりを手牌情報と仮定)
        # シュンツ(連続する3つの牌)をランダムに配置してAIの畳み込み(Conv1d)を刺激する
        for _ in range(3): 
            # 萬子(0-8), 筒子(9-17), 索子(18-26) のどこかにシュンツを作る
            suit_start = random.choice([0, 9, 18])
            seq_start = suit_start + random.randint(0, 6)
            
            t[i, 0, seq_start] = 1.0
            t[i, 0, seq_start + 1] = 1.0
            t[i, 0, seq_start + 2] = 1.0
            
        # トイツ(雀頭)などの重なりを表現
        pair_idx = random.randint(0, 33)
        t[i, 1, pair_idx] = 1.0
        
        # 2. 補助特徴量 (シャンテン数が低く、門前価値が高い「チャンス手」を偽装)
        a[i, 0] = random.uniform(0.7, 1.0) # 門前価値(高)
        a[i, 1] = random.uniform(0.0, 0.3) # シャンテン数(低 = テンパイや1向聴に近い)
        a[i, 2] = random.uniform(0.5, 1.0) # 打点/ドラ(高)

    return t, a

# =========================================
# 🔎 3. Hard Negative Mining スキャン
# =========================================
def run_hard_negative_mining(model, num_samples=10000):
    print(f"\n🚀 Phase 10G-1 (修正版): 構造化データによる鳴きバイアス・スキャンを開始... (対象: {num_samples}局面)")
    
    batch_size = 1000
    iterations = num_samples // batch_size
    
    stats = {
        "total": 0,
        "prob_over_50": 0,            # call確率 > 50%
        "high_call_prob": 0,          # call確率 > 70% (異常な鳴き意欲)
        "hard_negative_candidates": 0 # 門前価値が高いのに高確率(>70%)でcall判定した
    }
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(iterations):
            t, a = generate_structured_batch(batch_size)
            
            logits = model(t, a)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            monzen_values = a[:, 0].cpu().numpy() # a[:, 0] を疑似的な門前価値とする
            
            for j in range(batch_size):
                stats["total"] += 1
                p = probs[j]
                mv = monzen_values[j]
                
                if p > 0.50:
                    stats["prob_over_50"] += 1
                    
                if p > 0.70:
                    stats["high_call_prob"] += 1
                    # 門前価値が0.8以上の「鳴くべきではないチャンス手」なのに鳴こうとする
                    if mv > 0.80:
                        stats["hard_negative_candidates"] += 1

    print(f"✅ スキャン完了 ({time.time() - start_time:.2f}秒)\n")
    return stats

# =========================================
# 🎬 4. メイン実行
# =========================================
def main():
    model = load_call_net()
    num_samples = 50000 # 5万局
    
    stats = run_hard_negative_mining(model, num_samples)
    
    t = stats["total"]
    p50 = stats["prob_over_50"]
    hcp = stats["high_call_prob"]
    hnc = stats["hard_negative_candidates"]
    
    print("👑 === Phase 10G-1(修正版): call_net 鳴きバイアス分析レポート ===")
    print(f"■ スキャン総局面数: {t:,} 件")
    print("-" * 60)
    print(f"🚨 Call寄り判定 (確率 > 50%): {p50:,} 件 ({p50/t*100:.1f}%)")
    print(f"🚨 異常な高確率Call (確率 > 70%): {hcp:,} 件 ({hcp/t*100:.1f}%)")
    print(f"   ※ 手牌の形が少し整っただけで、AIがどれだけ『鳴きたがるか』の真の数値です。")
    print("-" * 60)
    print(f"🔥 Hard Negative 候補 (門前価値が高いのに高確率Call): {hnc:,} 件")
    print(f"   ※ これが現在のAIが抱える『門前を崩す無駄鳴きの病巣』です！")
    print(f"   ※ この件数が多ければ多いほど、ベースモデル自体を再学習させる決定的な証拠となります。")
    print("-" * 60)

if __name__ == "__main__":
    main()