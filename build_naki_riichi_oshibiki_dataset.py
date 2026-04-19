# build_naki_riichi_oshibiki_dataset.py
"""
天鳳牌譜から 鳴き・リーチ・押し引き の3種類の学習データを抽出するスクリプト

===========================================================================
【抽出内容の説明】
===========================================================================

■ 鳴き (Naki) データセット  →  dataset_naki_33ch.pkl
---------------------------------------------------------------------------
  テンソル : (34, 34) = 33ch基底テンソル + 1ch捨て牌位置チャンネル
  ラベル   : 0=スキップ(鳴かない), 1=ポン, 2=チー  ← 3クラス分類

  抽出ロジック:
    他家が牌を捨てた瞬間を起点に、「物理的に鳴ける全プレイヤー」を候補として抽出する。
      - ポン  : 捨て牌と同種牌を2枚以上手牌に持っているプレイヤー (全方向)
      - チー  : 捨て牌と連続する牌2枚を手牌に持つ上家 (左隣プレイヤー)
    実際にポンしたプレイヤー → ラベル1 (正例-ポン)
    実際にチーしたプレイヤー → ラベル2 (正例-チー)
    鳴けたが鳴かなかったプレイヤー → ラベル0 (負例)
    ポン・チー両方可能で鳴いた場合は実際の選択タイプで判定する。

    鳴きを含む局も完全追跡して全局面を対象にする（1局内に複数の鳴きがあっても全て抽出）。
    局所状態は「捨て牌が打たれた直後・鳴きが発生する直前」のスナップショットを使用。
    カン（槓）は除外する（戦略的重要性が低く、データ量も少ない）。

■ リーチ (Riichi) データセット  →  dataset_riichi_33ch.pkl
---------------------------------------------------------------------------
  テンソル : (34, 34) = 33ch基底テンソル + 1ch打牌位置チャンネル
  ラベル   : 0=ダマテン(リーチしない), 1=リーチ宣言

  抽出ロジック:
    鳴きなし局（面前のみ）のみ対象。
    各ツモ→打牌のペアについて、その打牌後の手牌のシャンテン数が -1（テンパイ）かどうかを確認する。
      - テンパイかつ次イベントが REACH step=1  → ラベル1 (リーチ宣言)
      - テンパイかつ次イベントが REACH でない  → ラベル0 (ダマテン選択)
    テンパイでない打牌は抽出しない（リーチの選択肢がないため）。
    自分がすでにリーチ宣言済みの場合も除外する。

    ※ 「どの牌を打つか」ではなく「テンパイ後にリーチするかダマにするか」を学習する。

■ 押し引き (Oshibiki/Push-Fold) データセット  →  dataset_oshibiki_33ch.pkl
---------------------------------------------------------------------------
  テンソル : (34, 34) = 33ch基底テンソル + 1ch打牌位置チャンネル
  ラベル   : 0=オリ（現物など安全牌を選択）, 1=押し（危険牌を打った）

  抽出ロジック:
    鳴きなし局で「少なくとも1人の対戦相手がリーチ宣言中」の局面を対象。
    各打牌について：
      - 打牌がリーチ者全員の「現物」(リーチ後の河にある牌と同種) → ラベル0 (安全牌=オリ)
      - 打牌がいずれかのリーチ者の現物でない             → ラベル1 (危険牌=押し)
    リーチ宣言者本人の打牌は除外する（REACH step=1 直後の打牌はリーチ牌なので別問題）。
    リーチ宣言者がいない局面は除外する。
===========================================================================
"""

import os
import gzip
import re
import pickle
import time
import copy
import xml.etree.ElementTree as ET
import numpy as np

from mahjong_engine import MahjongStateV5, calculate_shanten
# 副露対応シャンテン計算・有効牌列挙は hybrid_inference から利用
import hybrid_inference as _hi


# ===========================================================================
# 1. EnhancedTenhouParser  ― Nタグ の m 属性もデコードする
# ===========================================================================

def _decode_naki_type(m):
    """
    天鳳 N タグの m 属性からポン/チー/カンを判別して返す。
    戻り値: 'chi' | 'pon' | 'kakan' | 'kan' | 'unknown'
    """
    if m & 0x4:
        return 'chi'
    elif m & 0x10:
        return 'kakan'   # 加槓（ポン後に槓）
    elif m & 0x8:
        return 'pon'
    elif m & 0x20:
        return 'kan'     # 大明槓 or 暗槓
    return 'unknown'


class EnhancedTenhouParser:
    """
    天鳳XMLを時系列イベント列に変換する。
    Nタグの m 属性も解析し naki_type を付加する。
    """
    def __init__(self, xml_string):
        self.root = ET.fromstring(xml_string)
        self.events = []

    def parse(self):
        for node in self.root:
            tag = node.tag

            if tag == "INIT":
                seed = node.attrib.get("seed", "0,0,0,0,0,0").split(",")
                oya  = int(node.attrib.get("oya", "0"))
                hands = {
                    i: [int(x) for x in node.attrib.get(f"hai{i}", "").split(",") if x]
                    for i in range(4)
                }
                self.events.append({
                    "type"          : "INIT",
                    "bakaze"        : int(seed[0]) // 4,
                    "kyoku"         : int(seed[0]) % 4,
                    "honba"         : int(seed[1]),
                    "kyotaku"       : int(seed[2]),
                    "dora_indicator": int(seed[5]),
                    "oya"           : oya,
                    "hands"         : hands,
                    "scores"        : [int(x) * 100 for x in
                                       node.attrib.get("ten", "0,0,0,0").split(",")]
                })

            elif tag.startswith(('T','U','V','W')) and tag[1:].isdigit():
                seat = {'T':0,'U':1,'V':2,'W':3}[tag[0]]
                tile_136 = int(tag[1:])
                self.events.append({
                    "type": "DRAW", "seat": seat,
                    "tile_136": tile_136, "tile_34": tile_136 // 4
                })

            elif tag.startswith(('D','E','F','G','d','e','f','g')) and tag[1:].isdigit():
                seat = {'D':0,'E':1,'F':2,'G':3,'d':0,'e':1,'f':2,'g':3}[tag[0]]
                tile_136 = int(tag[1:])
                self.events.append({
                    "type": "DISCARD", "seat": seat,
                    "tile_136": tile_136, "tile_34": tile_136 // 4
                })

            elif tag == "REACH":
                self.events.append({
                    "type": "REACH",
                    "seat": int(node.attrib.get("who")),
                    "step": int(node.attrib.get("step"))
                })

            elif tag == "N":
                m    = int(node.attrib.get("m", "0"))
                seat = int(node.attrib.get("who"))
                naki_type = _decode_naki_type(m)
                self.events.append({
                    "type"     : "CALL",
                    "seat"     : seat,
                    "m"        : m,
                    "naki_type": naki_type
                })

            elif tag == "DORA":
                dora_136 = int(node.attrib.get("hai", "0"))
                self.events.append({
                    "type": "DORA", "tile_136": dora_136, "tile_34": dora_136 // 4
                })

            elif tag == "AGARI":
                self.events.append({"type": "AGARI"})

            elif tag == "RYUUKYOKU":
                self.events.append({"type": "RYUUKYOKU"})

        return self.events


# ===========================================================================
# 2. EnhancedGlobalReplayTracker  ― 鳴きを含む局も追跡し続ける
# ===========================================================================

def _find_tiles_for_pon(hand_136, tile_34):
    """手牌 136 リストから tile_34 と一致する牌を2枚返す。なければ None。"""
    matches = [t for t in hand_136 if t // 4 == tile_34]
    if len(matches) >= 2:
        return matches[0], matches[1]
    return None


def _find_tiles_for_chi(hand_136, discarded_34):
    """
    手牌 136 リストから discarded_34 とチーを構成できる2枚を返す。
    可能なパターン（上がり牌=discarded の場合）:
      A: [d-2, d-1, d]  →  手牌から d-2, d-1
      B: [d-1, d, d+1]  →  手牌から d-1, d+1
      C: [d, d+1, d+2]  →  手牌から d+1, d+2
    最初に成立するパターンを採用する。
    """
    if discarded_34 >= 27:
        return None   # 字牌はチー不可

    suit_base = (discarded_34 // 9) * 9
    num       = discarded_34 % 9         # 0-indexed (0=1, 8=9)

    patterns = []
    if num >= 2:
        patterns.append((discarded_34 - 2, discarded_34 - 1))
    if 1 <= num <= 7:
        patterns.append((discarded_34 - 1, discarded_34 + 1))
    if num <= 6:
        patterns.append((discarded_34 + 1, discarded_34 + 2))

    for t1_34, t2_34 in patterns:
        t1_list = [t for t in hand_136 if t // 4 == t1_34]
        t2_list = [t for t in hand_136 if t // 4 == t2_34]
        if t1_34 == t2_34:
            if len(t1_list) >= 2:
                return t1_list[0], t1_list[1]
        else:
            if t1_list and t2_list:
                return t1_list[0], t2_list[0]
    return None


class EnhancedGlobalReplayTracker:
    """
    全イベント（鳴きを含む）を正確に追跡するグローバルトラッカー。
    鳴きが発生しても is_broken にならずに追跡を続ける。
    カン（槓）や暗槓は追跡が複雑なため is_broken フラグを立てる。
    """

    def __init__(self):
        self.hands_136      = {i: [] for i in range(4)}
        self.discards_136   = {i: [] for i in range(4)}
        self.riichi_declared = {i: False for i in range(4)}
        self.scores         = {i: 25000 for i in range(4)}
        self.dora_indicators = []
        self.bakaze  = 0
        self.kyoku   = 0
        self.honba   = 0
        self.kyotaku = 0
        self.oya     = 0
        self.is_broken = False
        # 直前の捨て牌情報（鳴き判定に使う）
        self._last_discard_seat   = -1
        self._last_discard_136    = -1
        # 鳴きがあった局かどうか（リーチ/押し引きの抽出判断に使う）
        self.has_naki = False

    def apply_event(self, event):
        e_type = event["type"]

        if e_type == "INIT":
            self.hands_136       = {i: event["hands"][i].copy() for i in range(4)}
            self.discards_136    = {i: [] for i in range(4)}
            self.riichi_declared = {i: False for i in range(4)}
            self.scores          = {i: event["scores"][i] for i in range(4)}
            self.dora_indicators = [event["dora_indicator"]]
            self.bakaze  = event["bakaze"]
            self.kyoku   = event["kyoku"]
            self.honba   = event["honba"]
            self.kyotaku = event["kyotaku"]
            self.oya     = event["oya"]
            self.is_broken = False
            self.has_naki  = False
            self._last_discard_seat = -1
            self._last_discard_136  = -1
            return

        if self.is_broken:
            return

        if e_type == "DRAW":
            self.hands_136[event["seat"]].append(event["tile_136"])

        elif e_type == "DISCARD":
            seat = event["seat"]
            tile = event["tile_136"]
            if tile in self.hands_136[seat]:
                self.hands_136[seat].remove(tile)
                self.discards_136[seat].append(tile)
                self._last_discard_seat = seat
                self._last_discard_136  = tile
            else:
                self.is_broken = True

        elif e_type == "REACH":
            if event["step"] == 1:
                seat = event["seat"]
                self.riichi_declared[seat] = True
                self.scores[seat]   -= 1000
                self.kyotaku        += 1

        elif e_type == "CALL":
            naki_type = event.get("naki_type", "unknown")
            caller    = event["seat"]

            # カン・加槓 は追跡が複雑なため破損扱い
            if naki_type in ('kan', 'kakan', 'unknown'):
                self.is_broken = True
                return

            self.has_naki = True
            discarded_34  = self._last_discard_136 // 4 if self._last_discard_136 >= 0 else -1

            # 捨て牌を河から取り除く（鳴かれた牌は河に残らない）
            if self._last_discard_136 in self.discards_136[self._last_discard_seat]:
                self.discards_136[self._last_discard_seat].remove(self._last_discard_136)

            if naki_type == 'pon':
                result = _find_tiles_for_pon(self.hands_136[caller], discarded_34)
                if result:
                    self.hands_136[caller].remove(result[0])
                    self.hands_136[caller].remove(result[1])
                else:
                    self.is_broken = True

            elif naki_type == 'chi':
                result = _find_tiles_for_chi(self.hands_136[caller], discarded_34)
                if result:
                    self.hands_136[caller].remove(result[0])
                    self.hands_136[caller].remove(result[1])
                else:
                    self.is_broken = True

        elif e_type == "DORA":
            self.dora_indicators.append(event["tile_136"])

        elif e_type in ("AGARI", "RYUUKYOKU"):
            pass   # 局終了 — 次の INIT で初期化される


# ===========================================================================
# 3. build_local_state  ― グローバル状態からプレイヤー視点の MahjongStateV5 を生成
# ===========================================================================

def build_local_state(tracker: EnhancedGlobalReplayTracker,
                      target_seat: int) -> MahjongStateV5:
    state = MahjongStateV5()

    for t_136 in tracker.hands_136[target_seat]:
        state.add_tile(0, t_136)

    for pov in range(4):
        actual_seat = (target_seat + pov) % 4
        for t_136 in tracker.discards_136[actual_seat]:
            state.discard_tile(pov, t_136)

    state.dora_indicators = [d // 4 for d in tracker.dora_indicators]  # tile_136 → tile_34
    state.bakaze  = tracker.bakaze
    state.jikaze  = (target_seat - tracker.oya + 4) % 4
    state.honba   = tracker.honba
    state.kyotaku = tracker.kyotaku

    for pov in range(4):
        actual_seat = (target_seat + pov) % 4
        state.riichi_declared[pov] = tracker.riichi_declared[actual_seat]
        state.scores[pov]          = tracker.scores[actual_seat]

    return state


# ===========================================================================
# 4-A-0. 鳴き特徴量ヘルパー
# ===========================================================================

def _n_melds_from_hand(hand):
    """手牌配列の枚数から副露数を推定する (13枚=0副露, 10枚=1副露, ...)"""
    return max(0, (13 - int(sum(hand))) // 3)


def _chi_consumed_patterns(hand, disc_tile):
    """
    手牌 hand で disc_tile（上家捨て牌）をチーできる消費牌ペアを列挙する。
    戻り値: [(a, b), ...]  (hand[a]>0 かつ hand[b]>0 が保証されたペア)
    """
    if disc_tile >= 27:
        return []
    suit = disc_tile // 9
    patterns = []
    for a, b in (
        (disc_tile - 2, disc_tile - 1),
        (disc_tile - 1, disc_tile + 1),
        (disc_tile + 1, disc_tile + 2),
    ):
        if a < 0 or b >= 34:
            continue
        if a // 9 != suit or b // 9 != suit:
            continue
        # 同種牌を2枚使う場合 (a==b は起こらないが念のため)
        if a == b:
            if hand[a] >= 2:
                patterns.append((a, b))
        else:
            if hand[a] >= 1 and hand[b] >= 1:
                patterns.append((a, b))
    return patterns


def _count_ukeire_by_shanten(tmp_state, hand, current_sh):
    """
    シャンテン変化だけで有効牌種類数を数える（スコア計算を使わない安全版）。
    tmp_state.fixed_mentsu に副露数がセットされていれば OK。
    current_sh=-1（テンパイ）でも動作する（さらに-1=アガリになる牌を数える）。
    """
    count = 0
    for t in range(34):
        if hand[t] >= 4:
            continue
        test_h = hand.copy()
        test_h[t] += 1
        sh = _hi.calculate_shanten_unified(tmp_state, test_h)
        if sh < current_sh:
            count += 1
    return count


def _best_discard_shanten(hand_after_call, n_total_melds, local_state,
                           forbidden_tile=-1):
    """
    鳴き後の手牌 hand_after_call から最良打牌を選び (最小シャンテン, 有効牌種類数) を返す。

    hand_after_call : 鳴き後の手牌配列 (捨て牌はまだ含む — ここから1枚選んで切る)
    n_total_melds   : 鳴き後の副露総数
    forbidden_tile  : 食い替え禁止牌 (チー時の捨て牌と同種, -1=制限なし)
    """
    # シャンテン計算用の仮ステート (fixed_mentsu の内容は不要; 数だけ合わせる)
    tmp = copy.copy(local_state)
    tmp.fixed_mentsu = [[]] * n_total_melds

    best_sh = 99
    best_h  = None

    # Step 1: 最小シャンテンになる打牌を探す
    for t in range(34):
        if hand_after_call[t] <= 0:
            continue
        if t == forbidden_tile:
            continue
        h = hand_after_call.copy()
        h[t] -= 1
        sh = _hi.calculate_shanten_unified(tmp, h)
        if sh < best_sh:
            best_sh = sh
            best_h  = h

    # Step 2: 最良打牌での有効牌をシャンテン変化で計算（スコア計算不要）
    best_uk = 0
    if best_h is not None:
        best_uk = _count_ukeire_by_shanten(tmp, best_h, best_sh)

    return best_sh, best_uk


def _compute_naki_features(local_state, disc_tile, rel_discarder):
    """
    鳴き候補に対してシャンテン/テンパイ/有効牌の特徴量を計算する。

    Returns:
        shanten_before  : int  鳴く前シャンテン数
        shanten_after   : int  鳴き後ベストシャンテン数 (ポン/チーの最良)
        is_tenpai_after : bool 鳴いたらテンパイを取れるか
        ukeire_after    : int  鳴き後有効牌種類数
    """
    hand = local_state.hand
    n_exist = _n_melds_from_hand(hand)

    # 鳴く前シャンテン
    tmp_before = copy.copy(local_state)
    tmp_before.fixed_mentsu = [[]] * n_exist
    sh_before = _hi.calculate_shanten_unified(tmp_before, hand)

    best_sh_after = sh_before   # フォールバック: 鳴けない場合は現状維持
    best_uk_after = 0

    # ---- ポン ----
    if hand[disc_tile] >= 2:
        h_pon = hand.copy()
        h_pon[disc_tile] -= 2
        sh_pon, uk_pon = _best_discard_shanten(
            h_pon, n_exist + 1, local_state, forbidden_tile=-1
        )
        if sh_pon < best_sh_after or (sh_pon == best_sh_after and uk_pon > best_uk_after):
            best_sh_after = sh_pon
            best_uk_after = uk_pon

    # ---- チー (上家のみ) ----
    if rel_discarder == 3:
        for a, b in _chi_consumed_patterns(hand, disc_tile):
            h_chi = hand.copy()
            h_chi[a] -= 1
            h_chi[b] -= 1
            # 食い替え禁止: disc_tile と同じタイルは打てない
            sh_chi, uk_chi = _best_discard_shanten(
                h_chi, n_exist + 1, local_state, forbidden_tile=disc_tile
            )
            if sh_chi < best_sh_after or (sh_chi == best_sh_after and uk_chi > best_uk_after):
                best_sh_after = sh_chi
                best_uk_after = uk_chi

    is_tenpai_after = (best_sh_after == 0)
    return sh_before, best_sh_after, is_tenpai_after, best_uk_after


# ===========================================================================
# 4-A. 鳴きデータ抽出
# ===========================================================================

def extract_naki_samples(events, log_id="unknown"):
    """
    全イベント列から鳴き判断のサンプルを抽出する。

    抽出条件:
      - 他家が DISCARD した直後に、各プレイヤーについて can_naki を確認
      - 実際に CALL したプレイヤー → label=1
      - 鳴けたが鳴かなかったプレイヤー → label=0
      - カン は除外

    出力レコード:
      {"tensor": (34,34), "label": 0|1, "meta_log_id": str,
       "meta_naki_type": str, "meta_kyoku": int}
    """
    tracker = EnhancedGlobalReplayTracker()
    records = []

    for idx in range(len(events)):
        event = events[idx]
        tracker.apply_event(event)

        if tracker.is_broken:
            continue

        if event["type"] != "DISCARD":
            continue

        discarder  = event["seat"]
        disc_tile  = event["tile_34"]

        # 次イベントが CALL かどうか確認
        call_event = None
        for j in range(idx + 1, min(idx + 3, len(events))):
            ne = events[j]
            if ne["type"] == "CALL":
                call_event = ne
                break
            elif ne["type"] in ("DRAW", "INIT", "AGARI", "RYUUKYOKU"):
                break
            # REACH(step=1) は DISCARD と CALL の間に挟まれることがある
            # （リーチ宣言の直後に誰かがロン）→ 基本的に CALL の前に REACH は来ない

        caller_seat  = call_event["seat"]   if call_event else -1
        naki_type    = call_event["naki_type"] if call_event else ""

        # カンは学習対象外
        if naki_type in ('kan', 'kakan', 'unknown') and call_event is not None:
            continue

        # 各シートの鳴き可否を判定
        for candidate in range(4):
            if candidate == discarder:
                continue

            # candidate 視点のローカル状態を構築
            local_state = build_local_state(tracker, candidate)

            # candidate が discarder から見た相対位置
            rel_discarder = (discarder - candidate + 4) % 4

            if not local_state.can_naki(disc_tile, rel_discarder):
                continue  # 物理的に鳴けない → スキップ

            # 3クラスラベル: 0=スキップ, 1=ポン, 2=チー
            if call_event is not None and caller_seat == candidate:
                label = 1 if naki_type == 'pon' else 2  # 'chi' → 2
            else:
                label = 0

            # ---- 鳴き特徴量を計算 ----
            try:
                sh_before, sh_after, tenpai_flag, ukeire = _compute_naki_features(
                    local_state, disc_tile, rel_discarder
                )
            except Exception:
                # 計算失敗 → 安全なデフォルト値で続行
                sh_before, sh_after, tenpai_flag, ukeire = 3, 3, False, 0

            # ---- 自明なスルーサンプルを除外（ノイズ削減）----
            # 「手牌が遠い (3+シャンテン) かつ 鳴いてもシャンテンが改善しない」
            # → NN が境界を学ぶ意味のない自明ケースなので除外
            if label == 0:
                sh_improvement = sh_before - sh_after
                if sh_before >= 3 and sh_improvement <= 0:
                    continue

            # ---- v3 テンソル生成 (CH23-26 に有意な値を埋める) ----
            tensor = local_state.to_tensor_for_naki_v3(
                disc_tile, sh_before, sh_after, tenpai_flag, ukeire
            )

            records.append({
                "tensor"          : tensor,
                "label"           : label,
                "meta_log_id"     : log_id,
                "meta_naki_type"  : naki_type if call_event else "none",
                "meta_kyoku"      : tracker.kyoku,
                # デバッグ/分析用メタ情報
                "meta_sh_before"  : sh_before,
                "meta_sh_after"   : sh_after,
                "meta_tenpai"     : tenpai_flag,
                "meta_ukeire"     : ukeire,
            })

    return records


# ===========================================================================
# 4-B. リーチデータ抽出
# ===========================================================================

def extract_riichi_samples(events, log_id="unknown"):
    """
    鳴きなし局から「テンパイ時にリーチするかダマにするか」のサンプルを抽出。

    ※ 天鳳XMLのリーチ宣言イベント順序:
         DRAW → REACH(step=1) → DISCARD → REACH(step=2)
       リーチ宣言は打牌の「前」に記録される。

    抽出ロジック:
      Case A (label=1 リーチ):
        DRAW(S) → REACH(step=1, S) → DISCARD(S, T) のパターンを検出
        → 打牌 T のチャンネルを CH33 に付加してサンプル化

      Case B (label=0 ダマテン):
        DRAW(S) → DISCARD(S, T) のパターンで打牌後のシャンテン数が -1
        → リーチせずにテンパイを維持した選択としてサンプル化

    対象: 鳴きなし局のみ・リーチ宣言済みプレイヤーは除外
    出力レコード: {"tensor": (34,34), "label": 0|1, ...}
    """
    tracker = EnhancedGlobalReplayTracker()
    records = []

    for idx in range(len(events) - 2):
        event      = events[idx]
        next_event = events[idx + 1]
        tracker.apply_event(event)

        if tracker.is_broken or tracker.has_naki:
            continue

        if event["type"] != "DRAW":
            continue

        seat = event["seat"]

        if tracker.riichi_declared[seat]:
            continue

        # --- Case A: DRAW → REACH(step=1) → DISCARD  ← リーチ宣言 ---
        if (next_event["type"] == "REACH" and
                next_event.get("step") == 1 and
                next_event["seat"] == seat):
            # DISCARD は REACH(step=1) の直後にある
            label_tile_34 = None
            for j in range(idx + 2, min(idx + 5, len(events))):
                ne = events[j]
                if ne["type"] == "DISCARD" and ne["seat"] == seat:
                    label_tile_34 = ne["tile_34"]
                    break
                elif ne["type"] in ("DRAW", "CALL", "INIT", "AGARI", "RYUUKYOKU"):
                    break
            if label_tile_34 is None:
                continue
            label = 1

        # --- Case B: DRAW → DISCARD → (REACHなし) テンパイ  ← ダマテン ---
        elif (next_event["type"] == "DISCARD" and
                next_event["seat"] == seat):
            label_tile_34 = next_event["tile_34"]
            hand_counts = [0] * 34
            for t in tracker.hands_136[seat]:
                hand_counts[t // 4] += 1
            hand_counts[label_tile_34] -= 1
            if calculate_shanten(hand_counts) != -1:
                continue   # テンパイでなければスキップ
            label = 0

        else:
            continue

        local_state  = build_local_state(tracker, seat)
        # 34ch: 33ch基底(skip_logic=False でCH23-26も有効) + 1ch打牌位置
        base_tensor  = local_state.to_tensor(skip_logic=False)  # (33, 34)
        disc_channel = np.zeros((1, 34), dtype=np.float32)
        disc_channel[0][label_tile_34] = 1.0
        tensor_input = np.concatenate((base_tensor, disc_channel), axis=0)  # (34, 34)

        records.append({
            "tensor"     : tensor_input,
            "label"      : label,
            "meta_log_id": log_id,
            "meta_kyoku" : tracker.kyoku,
            "meta_seat"  : seat,
        })

    return records


# ===========================================================================
# 4-C. 押し引きデータ抽出
# ===========================================================================

def _get_riichi_genbutsu_34(tracker: EnhancedGlobalReplayTracker,
                             riichi_seat: int) -> set:
    """
    リーチ宣言者 riichi_seat の「現物」牌セット（34-index）を返す。
    現物 = リーチ宣言後に河に捨てられた牌（他のプレイヤーが安全に捨てられる保証がある牌）。
    ここでは簡略化として河全体の牌を現物として扱う。
    """
    return set(t // 4 for t in tracker.discards_136[riichi_seat])


def extract_oshibiki_samples(events, log_id="unknown"):
    """
    鳴きなし局から「リーチ者がいる場面での押し引き」サンプルを抽出。

    抽出条件:
      - 鳴きなし局のみ
      - 少なくとも1人の対戦相手がリーチ宣言中の場面でのみ抽出
      - リーチ宣言者本人の打牌は除外（リーチ後は選択の余地なし）

    ラベル定義:
      - 打牌が「全てのリーチ者の現物」に含まれる → label=0 (安全牌=オリ)
      - 打牌がいずれかのリーチ者の現物でない    → label=1 (危険牌=押し)

    出力レコード:
      {"tensor": (33,34), "label": 0|1, "meta_log_id": str,
       "meta_kyoku": int, "meta_seat": int}
    """
    tracker = EnhancedGlobalReplayTracker()
    records = []

    for idx in range(len(events) - 1):
        event      = events[idx]
        next_event = events[idx + 1]
        tracker.apply_event(event)

        if tracker.is_broken:
            continue

        if tracker.has_naki:
            continue   # 鳴きありの局は除外

        if event["type"] != "DRAW":
            continue
        if next_event["type"] != "DISCARD":
            continue
        if event["seat"] != next_event["seat"]:
            continue

        seat = event["seat"]

        # 自分以外のリーチ宣言者を特定
        opponent_riichi_seats = [
            s for s in range(4)
            if s != seat and tracker.riichi_declared[s]
        ]
        if not opponent_riichi_seats:
            continue   # リーチ者なし → 押し引き局面でない

        disc_tile_34 = next_event["tile_34"]

        # 全リーチ者の現物かどうか判定
        is_all_genbutsu = all(
            disc_tile_34 in _get_riichi_genbutsu_34(tracker, rs)
            for rs in opponent_riichi_seats
        )
        label = 0 if is_all_genbutsu else 1

        local_state  = build_local_state(tracker, seat)
        # 34ch: 33ch基底(skip_logic=False でCH23-26も有効) + 1ch打牌位置
        base_tensor  = local_state.to_tensor(skip_logic=False)  # (33, 34)
        disc_channel = np.zeros((1, 34), dtype=np.float32)
        disc_channel[0][disc_tile_34] = 1.0
        tensor_input = np.concatenate((base_tensor, disc_channel), axis=0)  # (34, 34)

        records.append({
            "tensor"     : tensor_input,
            "label"      : label,
            "meta_log_id": log_id,
            "meta_kyoku" : tracker.kyoku,
            "meta_seat"  : seat,
        })

    return records


# ===========================================================================
# 5. メインループ
# ===========================================================================

SAVE_NAKI_PATH     = "dataset_naki_v3.pkl"   # v3: CH23-26 にシャンテン/有効牌情報あり
SAVE_RIICHI_PATH   = "dataset_riichi_33ch.pkl"
SAVE_OSHIBIKI_PATH = "dataset_oshibiki_33ch.pkl"
SAVE_INTERVAL      = 100   # 何対局ごとに中間保存するか
LOG_DIR            = "logs"


def _load_existing(path):
    """既存 pkl を読み込み (records, processed_ids) を返す。"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            records = pickle.load(f)
        processed = set(r["meta_log_id"] for r in records)
        return records, processed
    return [], set()


def _save(records, path):
    with open(path, "wb") as f:
        pickle.dump(records, f)


if __name__ == "__main__":
    import socket
    import urllib.request

    socket.setdefaulttimeout(30)

    # ---------- 既存データ読み込み（レジューム対応） ----------
    naki_records,     naki_ids     = _load_existing(SAVE_NAKI_PATH)
    riichi_records,   riichi_ids   = _load_existing(SAVE_RIICHI_PATH)
    oshibiki_records, oshibiki_ids = _load_existing(SAVE_OSHIBIKI_PATH)

    processed_ids = naki_ids | riichi_ids | oshibiki_ids
    print(f"既存データ読み込み完了:")
    print(f"  鳴き    : {len(naki_records)} 件 ({len(naki_ids)} 対局)")
    print(f"  リーチ  : {len(riichi_records)} 件 ({len(riichi_ids)} 対局)")
    print(f"  押し引き: {len(oshibiki_records)} 件 ({len(oshibiki_ids)} 対局)")

    # ---------- ログファイルから対局IDを収集 ----------
    existing_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.gz')])
    pattern = re.compile(r'log=(\d{10}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{8})')

    print(f"\n.gz ファイルをスキャン中... ({len(existing_files)} 個)")
    phoenix_log_ids = []
    for filename in existing_files:
        with gzip.open(os.path.join(LOG_DIR, filename), 'rt',
                       encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "四鳳" in line:
                    for log_id in pattern.findall(line):
                        phoenix_log_ids.append(log_id)

    print(f"鳳凰卓の対局ID: {len(phoenix_log_ids)} 件 発見")

    remaining = [lid for lid in phoenix_log_ids if lid not in processed_ids]
    done_cnt  = len(phoenix_log_ids) - len(remaining)
    print(f"残り {len(remaining)} 対局 (済み: {done_cnt}/{len(phoenix_log_ids)})\n")

    # ---------- 対局ごとに抽出 ----------
    for i, log_id in enumerate(remaining):
        global_idx = done_cnt + i + 1
        print(f"  [{global_idx}/{len(phoenix_log_ids)}] {log_id} ...", end="", flush=True)

        try:
            url = f"https://tenhou.net/0/log/?{log_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()

            try:
                xml_string = gzip.decompress(raw).decode('utf-8')
            except Exception:
                xml_string = raw.decode('utf-8')

            parser = EnhancedTenhouParser(xml_string)
            events = parser.parse()

            n_recs  = extract_naki_samples(events,     log_id=log_id)
            r_recs  = extract_riichi_samples(events,   log_id=log_id)
            os_recs = extract_oshibiki_samples(events, log_id=log_id)

            naki_records.extend(n_recs)
            riichi_records.extend(r_recs)
            oshibiki_records.extend(os_recs)

            print(f" 鳴き+{len(n_recs)} リーチ+{len(r_recs)} 押引+{len(os_recs)}"
                  f"  (累計: {len(naki_records)}/{len(riichi_records)}/{len(oshibiki_records)})")

            time.sleep(1.0)

            # 中間保存
            if (i + 1) % SAVE_INTERVAL == 0:
                _save(naki_records,     SAVE_NAKI_PATH)
                _save(riichi_records,   SAVE_RIICHI_PATH)
                _save(oshibiki_records, SAVE_OSHIBIKI_PATH)
                print(f"  [checkpoint] {global_idx} 対局完了 → 中間保存")

        except Exception as e:
            print(f" [!] スキップ: {e}")
            continue

    # ---------- 最終保存 ----------
    _save(naki_records,     SAVE_NAKI_PATH)
    _save(riichi_records,   SAVE_RIICHI_PATH)
    _save(oshibiki_records, SAVE_OSHIBIKI_PATH)

    print("\n===== 抽出完了 =====")
    print(f"  鳴き    : {len(naki_records)} 件  → {SAVE_NAKI_PATH}")
    print(f"  リーチ  : {len(riichi_records)} 件  → {SAVE_RIICHI_PATH}")
    print(f"  押し引き: {len(oshibiki_records)} 件  → {SAVE_OSHIBIKI_PATH}")
