# mahjong_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ResNetのブロック定義
# ==========================================
class ResidualBlock(nn.Module):
    # [変更] dropout_p を追加（デフォルト0.0 = Dropout無効、旧モデルと互換）
    def __init__(self, channels, dropout_p=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        # [追加] 最初のReLU後に適用するDropout（p=0.0なら実質パススルー）
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)          # [追加] Conv1→BN1→ReLU の直後にDropout
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# ==========================================
# 攻守完全体・双頭のドラゴン（25チャンネル・最終進化版）
# ==========================================
class MahjongResNet_UltimateV3(nn.Module):
    # [変更] dropout_p_res / dropout_p_fc を追加（デフォルト0.0 = Dropout無効、旧モデルと互換）
    def __init__(self, num_blocks=5, dropout_p_res=0.0, dropout_p_fc=0.0):
        super(MahjongResNet_UltimateV3, self).__init__()
        # 33チャンネル入力（旧25ch -> 新33ch）
        self.conv_in = nn.Conv1d(in_channels=33, out_channels=256, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(256)

        # [変更] 各ResidualBlockにdropout_p_resを渡す（p=0.1で残差ブロック内Dropout有効化）
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256, dropout_p=dropout_p_res) for _ in range(num_blocks)]
        )

        # [追加] Flatten後・FC層直前に適用するDropout（p=0.3で全結合層への過学習を抑制）
        self.dropout_fc = nn.Dropout(p=dropout_p_fc)

        self.fc_discard = nn.Linear(256 * 34, 34)
        self.fc_riichi = nn.Linear(256 * 34, 2)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)           # [追加] Flatten直後・FC層直前にDropout
        out_discard = self.fc_discard(x)
        out_riichi = self.fc_riichi(x)
        return out_discard, out_riichi

# ==========================================
# 🗣️ 副露（鳴き）判断専用ニューラルネットワーク
# ==========================================
class MahjongResNet_Naki(nn.Module):
    def __init__(self, num_blocks=3): # 鳴き判断は打牌より少し軽い構造(3ブロック)で十分です
        super(MahjongResNet_Naki, self).__init__()
        
        # 💡 ここがポイント！ 34チャンネルの入力を受け取る（33ch基底 + 1ch捨て牌位置）
        self.conv_in = nn.Conv1d(in_channels=34, out_channels=128, kernel_size=3, padding=1)
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
    """MahjongResNet_Naki の改良版 (34ch: 33ch基底 + 1ch捨て牌位置)"""
    def __init__(self, num_blocks=3):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels=34, out_channels=128, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(128)
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(num_blocks)])
        self.fc_naki = nn.Linear(128 * 34, 2)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc_naki(x)


class MahjongResNet_34ch(nn.Module):
    """汎用モデル (in_channels / num_classes を可変指定)

    用途:
      - 鳴き判断  : in_channels=34, num_classes=3 (0=スルー, 1=ポン, 2=チー)
      - リーチ判断: in_channels=34, num_classes=2 (0=ダマ, 1=リーチ)
      - 押し引き  : in_channels=33, num_classes=2 (0=オリ, 1=押し) ← CH33なし
    """
    def __init__(self, num_classes=2, in_channels=34, num_blocks=3, channels=128, dropout_p_res=0.1, dropout_p_fc=0.3):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, dropout_p=dropout_p_res) for _ in range(num_blocks)]
        )
        self.dropout_fc = nn.Dropout(p=dropout_p_fc)
        self.fc_out = nn.Linear(channels * 34, num_classes)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        return self.fc_out(x)
