"""
run_phase5c_local.py
MahjongResNet_Naki_V2 (28ch) 学習スクリプト

phase5b2 からの改善点:
  1. ラベル再定義: 鳴いた AND 和了した局面のみ正例（有効鳴きラベル）
  2. 特徴量追加: 着順・トップとの点差チャンネル (28ch → to_tensor_for_naki_v2)
  3. ポン・チー両対応のデータ抽出
  4. 出力: mahjong_naki_model_master.pth を上書き
"""
import os
import re
import time
import gzip
import copy
import pickle
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from mahjong_engine import MahjongStateV5
from mahjong_model import MahjongResNet_Naki_V2

# =========================================
# 0. 設定
# =========================================
TARGET_RECORDS = 10_000_000  # 実質無制限（全データを処理）
LOGS_DIR = "./logs"
DATASET_CACHE = "./dataset_naki_v2.pkl"
OUTPUT_MODEL = "./mahjong_naki_model_master.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =========================================
# 1. Tenhou m値デコード
# =========================================
def decode_naki_type(m):
    if m & 0x4:
        return "chi"
    elif m & 0x8:
        return "pon"
    else:
        return "kan"  # 加カン・暗カン（学習対象外）

# =========================================
# 2. 簡易ゲーム状態トラッカー
# =========================================
class NakiTracker:
    def __init__(self):
        self.is_broken = False

    def reset(self, event):
        seed = event.get("seed", "0,0,0,0,0,0").split(",")
        self.hands_136 = {i: event["hands"][i].copy() for i in range(4)}
        self.dora_indicators = [int(seed[5]) // 4]  # 34種に変換
        self.bakaze = int(seed[0]) // 4
        self.oya = event["oya"]
        self.kyoku = event["kyoku"]
        self.riichi_declared = {i: False for i in range(4)}
        self.discards_136 = {i: [] for i in range(4)}
        self.scores = event["scores"].copy()
        self.is_broken = False

    def apply(self, ev):
        if self.is_broken:
            return
        t = ev["type"]
        if t == "DRAW":
            self.hands_136[ev["seat"]].append(ev["tile_136"])
        elif t == "DISCARD":
            tile = ev["tile_136"]
            hand = self.hands_136[ev["seat"]]
            if tile in hand:
                hand.remove(tile)
                self.discards_136[ev["seat"]].append(tile)
            else:
                self.is_broken = True
        elif t == "REACH" and ev["step"] == 1:
            self.riichi_declared[ev["seat"]] = True
        elif t == "CALL":
            self.is_broken = True  # call後は手牌追跡が困難なため打ち切り

# =========================================
# 3. ローカル状態構築
# =========================================
def build_local_state(tracker, my_seat):
    state = MahjongStateV5()
    state.bakaze = tracker.bakaze
    state.jikaze = (my_seat - tracker.oya + 4) % 4
    state.is_oya = (state.jikaze == 0)

    # add_tile() 経由で赤ドラフラグも設定
    for t136 in tracker.hands_136[my_seat]:
        state.add_tile(0, t136)

    for pov in range(4):
        actual = (my_seat + pov) % 4
        for t136 in tracker.discards_136[actual]:
            state.discards[pov].append(t136 // 4)
        state.riichi_declared[pov] = tracker.riichi_declared[actual]
        state.scores[pov] = tracker.scores[actual]

    state.dora_indicators = tracker.dora_indicators[:]
    return state

# =========================================
# 4. チー可能判定
# =========================================
def can_chi(hand_counts, tile_34):
    if tile_34 >= 27:
        return False
    suit = tile_34 // 9
    num = tile_34 % 9
    pats = []
    if num >= 2 and hand_counts[tile_34-2] >= 1 and hand_counts[tile_34-1] >= 1:
        if (tile_34-2) // 9 == suit:
            pats.append(True)
    if 1 <= num <= 7 and hand_counts[tile_34-1] >= 1 and hand_counts[tile_34+1] >= 1:
        if (tile_34-1) // 9 == suit and (tile_34+1) // 9 == suit:
            pats.append(True)
    if num <= 6 and hand_counts[tile_34+1] >= 1 and hand_counts[tile_34+2] >= 1:
        if (tile_34+1) // 9 == suit:
            pats.append(True)
    return len(pats) > 0

# =========================================
# 5. XML 解析とデータ抽出
# =========================================
def extract_naki_records(xml_string):
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return []

    # 全イベントをリスト化（局ごとに分割）
    all_kyoku = []
    cur = []
    for node in root:
        tag = node.tag
        if tag == "INIT":
            if cur:
                all_kyoku.append(cur)
            seed = node.attrib.get("seed", "0,0,0,0,0,0").split(",")
            sc = node.attrib.get("ten", "250,250,250,250").split(",")
            cur = [{"type": "INIT",
                    "seed": node.attrib.get("seed", "0,0,0,0,0,0"),
                    "oya": int(node.attrib.get("oya", "0")),
                    "kyoku": int(seed[0]),
                    "hands": {i: [int(x) for x in node.attrib.get(f"hai{i}", "").split(",") if x]
                              for i in range(4)},
                    "scores": {i: int(sc[i]) * 100 for i in range(4)}}]
        elif tag[0] in "TUVW" and tag[1:].isdigit():
            cur.append({"type": "DRAW", "seat": "TUVW".index(tag[0]),
                        "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
        elif tag[0] in "DEFGdefg" and tag[1:].isdigit():
            cur.append({"type": "DISCARD", "seat": "DEFGdefg".index(tag[0].upper()),
                        "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
        elif tag == "REACH":
            cur.append({"type": "REACH", "seat": int(node.attrib.get("who", 0)),
                        "step": int(node.attrib.get("step", 0))})
        elif tag == "N":
            m = int(node.attrib.get("m", 0))
            cur.append({"type": "CALL", "seat": int(node.attrib.get("who", 0)),
                        "m": m, "naki_type": decode_naki_type(m)})
        elif tag == "AGARI":
            cur.append({"type": "AGARI", "seat": int(node.attrib.get("who", 0))})
        elif tag in ("RYUUKYOKU", "BYE"):
            cur.append({"type": tag})
    if cur:
        all_kyoku.append(cur)

    records = []
    for kyoku_events in all_kyoku:
        if not kyoku_events or kyoku_events[0]["type"] != "INIT":
            continue

        # この局の和了者を先読み
        winner = -1
        for ev in kyoku_events:
            if ev["type"] == "AGARI":
                winner = ev["seat"]
                break

        tracker = NakiTracker()
        tracker.reset(kyoku_events[0])

        for i, ev in enumerate(kyoku_events[1:], 1):
            if tracker.is_broken:
                break

            if ev["type"] == "DISCARD":
                discard_seat = ev["seat"]
                discard_tile = ev["tile_34"]

                # 各プレイヤーの鳴き機会を確認
                for my_seat in range(4):
                    if my_seat == discard_seat:
                        continue
                    if any(tracker.riichi_declared[p] for p in range(4)):
                        continue  # リーチ中は鳴けない

                    my_hand_34 = [t // 4 for t in tracker.hands_136[my_seat]]
                    hand_counts = [my_hand_34.count(t) for t in range(34)]

                    can_pon = hand_counts[discard_tile] >= 2
                    is_kami = (discard_seat - my_seat + 4) % 4 == 3
                    c_chi = is_kami and can_chi(hand_counts, discard_tile)

                    if not (can_pon or c_chi):
                        continue

                    # 次のイベントで実際に鳴いたか確認
                    next_ev = kyoku_events[i + 1] if i + 1 < len(kyoku_events) else None
                    actually_called = (
                        next_ev and
                        next_ev["type"] == "CALL" and
                        next_ev["seat"] == my_seat and
                        next_ev.get("naki_type") in ("chi", "pon")
                    )

                    # ラベル: 鳴いた AND この局で和了 → 1、それ以外 → 0
                    if actually_called:
                        label = 1 if winner == my_seat else 0
                    else:
                        label = 0

                    state = build_local_state(tracker, my_seat)
                    tensor = state.to_tensor_for_naki_v2(discard_tile)

                    records.append({"tensor": tensor, "label": label})

            tracker.apply(ev)

    return records

# =========================================
# 6. データ収集
# =========================================
EXPECTED_CHANNELS = 34  # to_tensor_for_naki_v2 の新チャンネル数

def _cache_valid(path, expected_ch):
    """キャッシュのテンソルチャンネル数が期待値と一致するか確認"""
    try:
        with open(path, "rb") as f:
            sample = pickle.load(f)
        if sample and sample[0]["tensor"].shape[0] != expected_ch:
            print(f"  [警告] キャッシュのチャンネル数 {sample[0]['tensor'].shape[0]} "
                  f"!= 期待値 {expected_ch}。再生成します。")
            return False
        return True
    except Exception:
        return False

if os.path.exists(DATASET_CACHE) and _cache_valid(DATASET_CACHE, EXPECTED_CHANNELS):
    print(f"キャッシュを読み込みます: {DATASET_CACHE}")
    with open(DATASET_CACHE, "rb") as f:
        all_records = pickle.load(f)
    print(f"  {len(all_records)} 件")
else:
    print(f"📁 {LOGS_DIR} からログを読み込みます...")
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".gz")] if os.path.exists(LOGS_DIR) else []
    log_ids = []
    for filename in log_files:
        with gzip.open(os.path.join(LOGS_DIR, filename), "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r"log=([\w-]+)", line)
                if m:
                    log_ids.append(m.group(1))
    log_ids = list(set(log_ids))
    print(f"🎯 {len(log_ids)} 件の牌譜を処理します...")

    all_records = []
    for i, log_id in enumerate(log_ids):
        try:
            req = urllib.request.Request(
                f"https://tenhou.net/0/log/?{log_id}",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            xml_string = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
            all_records.extend(extract_naki_records(xml_string))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(log_ids)} 件処理... 現在 {len(all_records)} レコード")
            if len(all_records) >= TARGET_RECORDS:
                break
            time.sleep(0.5)
        except Exception:
            continue

    print(f"\n💾 キャッシュ保存: {DATASET_CACHE} ({len(all_records)} 件)")
    with open(DATASET_CACHE, "wb") as f:
        pickle.dump(all_records, f)

# =========================================
# 7. Dataset / DataLoader
# =========================================
print("\n📊 データ集計:")
labels = np.array([r["label"] for r in all_records])
pos = labels.sum()
neg = len(labels) - pos
print(f"  正例(有効鳴き): {int(pos):,}  負例(スキップ/失敗鳴き): {int(neg):,}  比率: {pos/len(labels)*100:.1f}%")

X = np.array([r["tensor"] for r in all_records], dtype=np.float32)
y = labels.astype(np.int64)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


class NakiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_loader = DataLoader(NakiDataset(X_tr, y_tr), batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(NakiDataset(X_val, y_val), batch_size=512, shuffle=False, num_workers=0)

# =========================================
# 8. モデル・損失関数
# =========================================
model = MahjongResNet_Naki_V2(num_blocks=3).to(DEVICE)

# クラス不均衡対策: 正例に重みを付ける
pos_weight = torch.tensor([neg / max(pos, 1)]).to(DEVICE)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, float(neg / max(pos, 1))]).to(DEVICE)
)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# =========================================
# 9. 学習ループ
# =========================================
print(f"\n🚀 学習開始 (Train: {len(y_tr):,} / Val: {len(y_val):,})")
best_f1 = 0.0
best_wts = None
patience = 0

for epoch in range(40):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    scheduler.step()

    # 検証
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            out = model(x_batch.to(DEVICE))
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    rec = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    train_loss_avg = total_loss / len(y_tr)

    print(f"Epoch {epoch+1:02d} | Loss: {train_loss_avg:.4f} | "
          f"Prec: {prec*100:.1f}% Rec: {rec*100:.1f}% F1: {f1*100:.2f}%")

    if f1 > best_f1:
        best_f1 = f1
        best_wts = copy.deepcopy(model.state_dict())
        patience = 0
    else:
        patience += 1
        if patience >= 7:
            print("🛑 Early stopping")
            break

# =========================================
# 10. 保存
# =========================================
model.load_state_dict(best_wts)
torch.save(model.state_dict(), OUTPUT_MODEL)
print(f"\n✅ 保存完了: {OUTPUT_MODEL}  (Best Val F1: {best_f1*100:.2f}%)")
