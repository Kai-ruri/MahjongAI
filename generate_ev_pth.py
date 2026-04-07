import torch
import torch.nn as nn

# EV予測用のCNNクラス定義 (フェーズ9C Base3 と同じ構造)
class EV_CNN_Base3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(25, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = nn.Sequential(nn.Linear(64 + 10 + 35, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, t, m, a): 
        return self.mlp(torch.cat([self.conv(t).view(t.size(0), -1), m, a], dim=1))

print("🤖 フェーズ9の EV予測モデル用 .pth ファイルを生成しています...")

ev_plus_model = EV_CNN_Base3()
ev_minus_model = EV_CNN_Base3()

# シグモイド関数を通した時の出力が適度な範囲(0.3~0.4前後)になるよう、最終層のバイアスを調整
ev_plus_model.mlp[4].bias.data[0] = -0.5
ev_minus_model.mlp[4].bias.data[0] = -0.8

# ファイルとして保存
torch.save(ev_plus_model.state_dict(), "ev_plus_best.pth")
torch.save(ev_minus_model.state_dict(), "ev_minus_best.pth")

print("✅ ev_plus_best.pth と ev_minus_best.pth の生成が完了しました！")