# MahjongAI

ResNetベースのニューラルネットワークと手作りヒューリスティックを組み合わせた、リーチ麻雀AIシステムです。天鳳（最高位戦Phoenix卓）の牌譜データから教師あり学習を行い、天鳳オンラインサーバーでの対局に対応しています。

## 概要

- **打牌モデル精度**: top-1 64.1%（天鳳Phoenix卓牌譜との一致率）
- **自己対戦ベンチマーク**: 和了率 21.5%、放銃率 12.8%、平均順位 2.45（AllAI 1000ゲーム）
- **天鳳オンライン対局**: WebSocketクライアント実装済み、実際の人間との対局が可能

## システム構成

```
天鳳牌譜 (HTML.gz)
  → parse_tenhou_log.py       # XMLパース → イベントシーケンス
  → dataset_extractor.py      # イベント → ゲーム状態スナップショット
  → build_supervised_dataset.py  # スナップショット → 教師データ (.pkl)
  → run_b1_train.py           # 転移学習でモデル更新
  → tenhou_client.py          # 天鳳サーバーで対局
```

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `mahjong_engine.py` | 麻雀ルールエンジン（手牌分解・役判定・点数計算・シャンテン数） |
| `mahjong_model.py` | NNモデル定義（ResNet: 打牌・リーチ・副露判断） |
| `hybrid_inference.py` | 推論ロジック（シャンテン計算・危険度評価・EV計算・点数状況判断） |
| `selfplay_minimal.py` | 4人自己対戦シミュレーター（全意思決定パイプライン） |
| `tenhou_client.py` | 天鳳WebSocketクライアント（JSON プロトコル実装） |
| `benchmark.py` | 性能評価スクリプト |

## ニューラルネットワーク

### 入力表現（25チャンネル、1D）
- 自手牌・河・副露・ドラ・残り枚数マップ 等

### アーキテクチャ
```
Conv1d → ResidualBlock × N → Dense → BatchNorm → Output
```

### モデル種別
- **打牌モデル** (`discard_b1_best.pth`): 34クラス softmax
- **副露モデル** (`mahjong_naki_model_master.pth`): PON/CHI 二値分類
- **リーチモデル** (`riichi_best.pth`): 二値分類
- **押し引きモデル** (`oshibiki_best.pth`): 二値分類

## 意思決定パイプライン

```
NN推論 (top-k候補抽出)
  → EV計算 (calculate_true_ev)
  → 危険度評価 (calculate_simple_discard_risk)
  → シャンテン・受け入れ枚数評価
  → 点数状況補正 (compute_game_situation)
  → 最終スコアリングで打牌決定
```

### 点数状況対応（`compute_game_situation`）
- 残り局数と点差から「逆転可能性」を動的に計算
- urgency（逆転急ぎ度）に応じてEV/守備ウェイトを可変調整
- オーラス着順保護・親番連荘・1位逃げ切り等のシナリオ対応

### リーチ判断ロジック（`should_declare_riichi`）
- ダマテン手役がある場合：打点 vs 目標点を比較してリーチ/ダマを選択
- 1位保護：安手逆転圏内では放銃禁止ガード
- 親オーラス：素点積み上げのため積極リーチ
- 待ち枚数閾値：状況に応じて2〜4枚に可変

## 学習フェーズ

| フェーズ | 内容 |
|---|---|
| Phase 5 | PON/CHI副露判断 |
| Phase 6 | リーチ宣言 |
| Phase 7 | 押し引き（危険牌押し） |
| Phase 8-9 | EV（期待値）モデル |
| Phase 10 | 統合・反復改善 |
| B-1 | 天鳳牌譜1148ゲーム追加 + 転移学習（精度 43%→64%） |
| B-3 | 残り枚数マップによる状態表現改善 |

## セットアップ

```bash
# 依存パッケージ
pip install torch numpy scikit-learn websocket-client

# 自己対戦ベンチマーク
python benchmark.py

# 天鳳オンライン対局（30試合連続）
python run_tenhou_games.py --games 30
```

## タイルエンコーディング

```
0-8   : 1m〜9m (萬子)
9-17  : 1p〜9p (筒子)
18-26 : 1s〜9s (索子)
27-33 : 東南西北白発中 (字牌)
```

## 依存関係

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- websocket-client（天鳳クライアントのみ）

モデルの重み（`.pth`）・学習データ（`.pkl`）・牌譜（`logs/`）はサイズのためリポジトリに含めていません。
