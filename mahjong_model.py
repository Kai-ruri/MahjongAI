# mahjong_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ResNetのブロック定義
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# ==========================================
# 攻守完全体・双頭のドラゴン（25チャンネル・最終進化版）
# ==========================================
class MahjongResNet_UltimateV3(nn.Module):
    def __init__(self, num_blocks=5):
        super(MahjongResNet_UltimateV3, self).__init__()
        # 💡 ここを 25チャンネル に変更！
        self.conv_in = nn.Conv1d(in_channels=25, out_channels=256, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(256)
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_blocks)]
        )
        
        self.fc_discard = nn.Linear(256 * 34, 34)
        self.fc_riichi = nn.Linear(256 * 34, 2)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1) 
        out_discard = self.fc_discard(x)
        out_riichi = self.fc_riichi(x)
        return out_discard, out_riichi

# ==========================================
# 🗣️ 副露（鳴き）判断専用ニューラルネットワーク
# ==========================================
class MahjongResNet_Naki(nn.Module):
    def __init__(self, num_blocks=3): # 鳴き判断は打牌より少し軽い構造(3ブロック)で十分です
        super(MahjongResNet_Naki, self).__init__()
        
        # 💡 ここがポイント！ 26チャンネルの入力を受け取る
        self.conv_in = nn.Conv1d(in_channels=26, out_channels=128, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(128)
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_blocks)]
        )
        
        # 💡 出力は「スルー(0)」か「鳴く(1)」の2択！
        self.fc_naki = nn.Linear(128 * 34, 2)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        out_naki = self.fc_naki(x)
        return out_naki


class MahjongResNet_Naki_V2(nn.Module):
    """MahjongResNet_Naki の改良版 (28ch: 着順・点差チャンネル追加)"""
    def __init__(self, num_blocks=3):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels=28, out_channels=128, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(128)
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(num_blocks)])
        self.fc_naki = nn.Linear(128 * 34, 2)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc_naki(x)
