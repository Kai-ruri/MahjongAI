import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import time

# =========================================
# 🧠 1. 本番CNNモデル定義 (深いモデル対応)
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CallCNN_Deep(nn.Module):
    def __init__(self, aux_dim=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(),
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
        self.conv = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 10 + 35, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, m, a): return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), m, a], dim=1))

class ModelManager:
    def __init__(self):
        self.call_net = CallCNN_Deep(10).to(device)
        self.ev_plus_net = EV_CNN_Base3().to(device)
        self.ev_minus_net = EV_CNN_Base3().to(device)
        
        self._load(self.call_net, r"G:\マイドライブ\MahjongAI\call_best.pth")
        self._load(self.ev_plus_net, r"G:\マイドライブ\MahjongAI\ev_plus_best.pth")
        self._load(self.ev_minus_net, r"G:\マイドライブ\MahjongAI\ev_minus_best.pth")
        
        self.call_net.eval(); self.ev_plus_net.eval(); self.ev_minus_net.eval()

    def _load(self, model, filename):
        if os.path.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=device))
            print(f"✅ {filename} をロードしました。")

# =========================================
# 📂 2. 実データの読み込み
# =========================================
def load_real_data(num_samples=10000):
    DATASET_FILES = [
        "./dataset_pon_phase5a_large.pkl",
        "./dataset_chi_phase5b_large.pkl"
    ]
    all_records = []
    for file_path in DATASET_FILES:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                all_records.extend(pickle.load(f))
    
    random.shuffle(all_records)
    records = all_records[:num_samples] # 評価用にランダム抽出
    
    X_tensor, X_aux = [], []
    for r in records:
        X_tensor.append(r["tensor"])
        if "aux" in r and len(r["aux"]) == 10:
            aux = np.array(r["aux"], dtype=np.float32)
        else:
            aux = np.array([
                float(r.get("wait_count", 0)), float(r.get("is_ryankei", 0)), float(r.get("meta_turn", 0)), 
                float(r.get("meta_enemy_riichi", 0)), float(r.get("meta_dora_count", 0)), float(r.get("meta_dama_legal", 0)), 
                float(r.get("meta_my_rank", 0)), float(r.get("meta_point_diff", 0)), 1.0, float(r.get("meta_score_proxy", 0))
            ], dtype=np.float32)
        X_aux.append(aux)
    
    return torch.tensor(np.array(X_tensor, dtype=np.float32)), torch.tensor(np.array(X_aux, dtype=np.float32))

# =========================================
# 🎬 3. メイン評価ロジック (統合シミュレーション)
# =========================================
def evaluate_integrated_call(models, t_data, a_data, batch_size=500):
    print(f"\n🚀 Phase 10G-3 (Real): 本物の牌譜データ {len(t_data)} 件による統合副露率テスト開始...")
    
    a_base, b_base = 0.0256, 0.1776 # Optuna Rank1 ベース係数
    
    call_count = 0
    call_overrides = 0
    total_samples = len(t_data)
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            t = t_data[i:i+batch_size].to(device)
            m = a_data[i:i+batch_size].to(device)
            
            # 1. ベースモデルの予測
            call_logits = models.call_net(t, m).squeeze()
            pol_call = torch.sigmoid(call_logits)
            pol_pass = 1.0 - pol_call
            
            # 2. EVモデルの予測
            a_vec_c = torch.zeros((t.size(0), 35)).to(device); a_vec_c[:, 0] = 1.0
            a_vec_p = torch.zeros((t.size(0), 35)).to(device); a_vec_p[:, 0] = 0.0
            
            plus_c = torch.sigmoid(models.ev_plus_net(t, m, a_vec_c)).squeeze() + 0.3
            minus_c = torch.sigmoid(models.ev_minus_net(t, m, a_vec_c)).squeeze() + 0.2
            plus_p = torch.sigmoid(models.ev_plus_net(t, m, a_vec_p)).squeeze()
            minus_p = torch.sigmoid(models.ev_minus_net(t, m, a_vec_p)).squeeze()
            
            # 3. 統合スコア計算
            score_c = pol_call + (a_base * plus_c) - (b_base * minus_c)
            score_p = pol_pass + (a_base * plus_p) - (b_base * minus_p)
            
            # 4. 判定
            final_call = score_c > score_p
            
            # 統計更新
            call_count += final_call.sum().item()
            
            # Override計算 (ベースモデルはPass派だったのに、EVが強引にCallさせた件数)
            base_pass = pol_pass > pol_call
            overrides = (base_pass & final_call).sum().item()
            call_overrides += overrides

    call_rate = (call_count / total_samples) * 100
    ovr_rate = (call_overrides / max(1, call_count)) * 100 # 鳴いたうちの何％がEVの強引な指示だったか
    
    print(f"✅ テスト完了 ({time.time() - start_time:.2f}秒)\n")
    print("👑 === Phase 10G-3 (Real): 本番データ統合評価 レポート ===")
    print(f"■ 統合・副露率: {call_rate:.1f}%  <-- 🎯目標: 20〜35%台")
    print(f"■ OVR率 (EV強行突破率): {ovr_rate:.1f}%  <-- 🎯目標: 20%未満")
    print("-" * 60)
    print("💡 もしここで副露率が20%〜30%台に収まっていれば、実験大成功です！")

if __name__ == "__main__":
    models = ModelManager()
    t_data, a_data = load_real_data(num_samples=10000)
    evaluate_integrated_call(models, t_data, a_data)