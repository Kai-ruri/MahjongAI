# ==========================================
# 🧠 究極ハイブリッド司令塔（A君＆B君の統合）
# ==========================================
import torch
import torch.nn.functional as F
import numpy as np
from mahjong_engine import calculate_shanten
import copy
from mahjong_engine import calculate_shanten, calculate_final_score, calculate_true_ev
from functools import lru_cache

def build_visible_tiles34(state):
    """
    34種ベースの visible_tiles をその場で作る
    state.visible_tiles に依存しない
    """
    visible = [0] * 34

    # 自分の手牌
    for i, c in enumerate(state.hand):
        visible[i] += c

    # 河
    for who in [0, 1, 2, 3]:
        for d in state.discards[who]:
            # すでに34種ならそのまま、136種IDなら //4
            tile = d if d < 34 else d // 4
            visible[tile] += 1

    # ドラ表示牌
    for d in state.dora_indicators:
        tile = d if d < 34 else d // 4
        visible[tile] += 1

    # 自分の副露
    for meld in getattr(state, "fixed_mentsu", []):
        for t in meld:
            visible[t] += 1

    # 他家の副露（公開情報なので可視牌に含める）
    for pov in [1, 2, 3]:
        for meld in state.melds.get(pov, []):
            for t in meld:
                t34 = t if t < 34 else t // 4
                visible[t34] += 1

    return visible

def is_real_agari_after_draw(state, hand_counts, draw_tile):
    """
    draw_tile を引いた hand_counts が、本当にアガリ形かどうかを判定する
    """
    result = calculate_final_score(
        closed_counts=hand_counts,
        fixed_mentsu=getattr(state, "fixed_mentsu", []),
        win_tile=draw_tile,
        is_tsumo=True,
        bakaze=state.bakaze,
        jikaze=state.jikaze,
        is_oya=getattr(state, "is_oya", False),
        is_riichi=False,
    )
    return result.get("score", 0) > 0

def get_waiting_tiles_with_open_hand(state, hand_counts=None, is_riichi=False):
    """
    副露対応の厳密待ち牌列挙
    hand_counts が None のときは state.hand を使う
    is_riichi=True にすると、リーチ宣言を前提として役チェックする
    戻り値: [draw_tile, ...]
    """
    if hand_counts is None:
        hand_counts = state.hand.copy()

    waiting_tiles = []

    for draw_tile in range(34):
        if hand_counts[draw_tile] >= 4:
            continue

        test_counts = hand_counts.copy()
        test_counts[draw_tile] += 1

        if is_riichi:
            # リーチ前提で役チェック（形さえ正しければ和了可能）
            result = calculate_final_score(
                closed_counts=test_counts,
                fixed_mentsu=getattr(state, "fixed_mentsu", []),
                win_tile=draw_tile,
                is_tsumo=True,
                bakaze=state.bakaze,
                jikaze=state.jikaze,
                is_oya=getattr(state, "is_oya", False),
                is_riichi=True,
            )
            if result.get("score", 0) > 0:
                waiting_tiles.append(draw_tile)
        else:
            if is_real_agari_after_draw(state, test_counts, draw_tile):
                waiting_tiles.append(draw_tile)

    return waiting_tiles

def get_effective_draw_tiles_with_open_hand(state, hand_counts=None):
    """
    副露対応の厳密有効牌列挙
    1枚引いたときにシャンテンが改善する牌を返す
    テンパイ時は待ち牌列挙を返す
    戻り値: [draw_tile, ...]
    """
    if hand_counts is None:
        hand_counts = state.hand.copy()

    fixed_mentsu = getattr(state, "fixed_mentsu", [])
    current_shanten = calculate_shanten_unified(state, hand_counts)

    # テンパイなら有効牌 = 待ち牌
    if current_shanten == 0:
        return get_waiting_tiles_with_open_hand(state, hand_counts)

    effective_tiles = []

    for draw_tile in range(34):
        if hand_counts[draw_tile] >= 4:
            continue

        test_counts = hand_counts.copy()
        test_counts[draw_tile] += 1

        new_shanten = calculate_shanten_unified(state, test_counts)

        if new_shanten < current_shanten:
            effective_tiles.append(draw_tile)

    return effective_tiles

def count_remaining_tiles_for_list(state, hand_counts, tile_list):
    """
    tile_list に含まれる牌の残り枚数合計
    visible_tiles は手牌を含む可視牌カウントなので、さらに hand_counts を引くと二重引きになる
    """
    visible_tiles = build_visible_tiles34(state)

    total = 0
    for t in tile_list:
        remaining = 4 - visible_tiles[t]
        if remaining > 0:
            total += remaining

    return total

@lru_cache(maxsize=200000)
def _open_hand_shanten_dfs(counts_tuple, need_melds, has_pair):
    """
    副露対応通常手シャンテンの内部DFS
    counts_tuple: 手牌34種タプル
    need_melds: まだ必要な面子数
    has_pair: 0 or 1
    """
    counts = list(counts_tuple)

    # 全部使い切ったらシャンテンを返す
    first = -1
    for i in range(34):
        if counts[i] > 0:
            first = i
            break

    if first == -1:
        # 8 - 2*面子 - ターツ - 雀頭 の形に対応
        # ここでは残り必要面子数 need_melds だけ残っている
        # すでにターツは取っていないので、単純化した終端
        return need_melds * 2 + (0 if has_pair else 1) - 1

    best = 99

    # 1. その牌を孤立牌として捨てる
    counts[first] -= 1
    best = min(best, _open_hand_shanten_dfs(tuple(counts), need_melds, has_pair))
    counts[first] += 1

    # 2. 雀頭を取る
    if has_pair == 0 and counts[first] >= 2:
        counts[first] -= 2
        best = min(best, _open_hand_shanten_dfs(tuple(counts), need_melds, 1))
        counts[first] += 2

    # 3. 面子を取る（刻子）
    if need_melds > 0 and counts[first] >= 3:
        counts[first] -= 3
        best = min(best, _open_hand_shanten_dfs(tuple(counts), need_melds - 1, has_pair))
        counts[first] += 3

    # 4. 面子を取る（順子）
    if need_melds > 0 and first < 27:
        num = first % 9
        if num <= 6 and counts[first + 1] > 0 and counts[first + 2] > 0:
            counts[first] -= 1
            counts[first + 1] -= 1
            counts[first + 2] -= 1
            best = min(best, _open_hand_shanten_dfs(tuple(counts), need_melds - 1, has_pair))
            counts[first] += 1
            counts[first + 1] += 1
            counts[first + 2] += 1

    # 5. ターツを取る（対子）
    if need_melds > 0 and counts[first] >= 2:
        counts[first] -= 2
        # ターツは面子1つ分を1進める扱い
        best = min(best, 1 + _open_hand_shanten_dfs(tuple(counts), need_melds - 1, has_pair))
        counts[first] += 2

    # 6. ターツを取る（連続）
    if need_melds > 0 and first < 27:
        num = first % 9

        # 両面
        if num <= 7 and counts[first + 1] > 0:
            counts[first] -= 1
            counts[first + 1] -= 1
            best = min(best, 1 + _open_hand_shanten_dfs(tuple(counts), need_melds - 1, has_pair))
            counts[first] += 1
            counts[first + 1] += 1

        # カンチャン
        if num <= 6 and counts[first + 2] > 0:
            counts[first] -= 1
            counts[first + 2] -= 1
            best = min(best, 1 + _open_hand_shanten_dfs(tuple(counts), need_melds - 1, has_pair))
            counts[first] += 1
            counts[first + 2] += 1

    return best

def calculate_normal_shanten_with_open_hand_exact(hand_counts, fixed_mentsu):
    """
    副露対応の通常手シャンテン厳密版
    七対子・国士は考えず、通常手のみを見る
    """
    open_melds = len(fixed_mentsu)
    need_melds = 4 - open_melds

    if need_melds < 0:
        need_melds = 0

    shanten = _open_hand_shanten_dfs(tuple(hand_counts), need_melds, 0)

    # 通常手シャンテンとして最低 -1 まで許す
    if shanten < -1:
        shanten = -1

    return shanten

def get_riichi_players(state):
    """リーチしている相手プレイヤー番号のリストを返す"""
    riichi_players = []
    for pid, declared in state.riichi_declared.items():
        if declared:
            riichi_players.append(pid)
    return riichi_players


def calculate_simple_discard_risk(state, tile_idx):
    """
    現物＋字牌評価＋筋＋ワンチャンス/ノーチャンス＋壁＋ドラ危険度
    """
    riichi_players = get_riichi_players(state)

    if len(riichi_players) == 0:
        return 0.0

    risk = 0.0

    for pid in riichi_players:
        enemy_discards = state.discards[pid]

        # 1. 現物
        if tile_idx in enemy_discards:
            base_risk = 0.0

        # 2. 字牌
        elif tile_idx >= 27:
            base_risk = calculate_honor_tile_risk(state, tile_idx, enemy_discards)

        else:
            # 3. ノーチャンス / ワンチャンス
            onechance_risk = get_onechance_risk(state, tile_idx)
            if onechance_risk is not None:
                base_risk = onechance_risk

            # 4. 壁
            else:
                kabe_risk = get_kabe_risk(state, tile_idx)
                if kabe_risk is not None:
                    base_risk = kabe_risk

                # 5. 筋
                elif is_suji_for_player(enemy_discards, tile_idx):
                    base_risk = 0.35

                # 6. その他
                else:
                    base_risk = 1.0

        # 7. ドラ危険度補正
        base_risk += get_dora_danger_bonus(state, tile_idx)

        risk += base_risk

    return risk


def infer_rival_pids_from_scores(state):
    """
    点棒配列から、自分と着順争いしている相手 pid のリストを返す
    簡易版:
      - 自分の1つ上の順位相手
      - 自分の1つ下の順位相手
      - 点差が近い相手（8000点以内）
    """
    scores = getattr(state, "scores", None)
    my_pid = getattr(state, "my_pid", 0)

    if scores is None or len(scores) != 4:
        return []

    indexed_scores = [(pid, sc) for pid, sc in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    # 順位表
    ranked_pids = [pid for pid, _ in indexed_scores]
    my_rank_index = ranked_pids.index(my_pid)

    rivals = set()

    # 1つ上
    if my_rank_index - 1 >= 0:
        rivals.add(ranked_pids[my_rank_index - 1])

    # 1つ下
    if my_rank_index + 1 < len(ranked_pids):
        rivals.add(ranked_pids[my_rank_index + 1])

    # 点差が近い相手
    my_score = scores[my_pid]
    for pid, sc in enumerate(scores):
        if pid == my_pid:
            continue
        if abs(sc - my_score) <= 5000:
            rivals.add(pid)

    return sorted(list(rivals))

def estimate_enemy_hand_value_level(state, who):
    """
    相手 who の手牌打点を簡易推定する。
    返り値: (value_level: str, value_mult: float, reasons: list)
      value_level: 'high' / 'mid' / 'low'
      value_mult:  危険度への乗数
    """
    reasons = []
    mult = 1.0

    # リーチ者は打点が高い
    if state.riichi_declared.get(who, False):
        reasons.append("riichi")
        mult *= 1.20
        return 'high', mult, reasons

    # ドラ表示牌から推定
    dora_indicators = getattr(state, 'dora_indicators', [])
    dora_tiles = get_dora_tiles_from_indicators(dora_indicators)

    # 相手の副露牌にドラが見えているか
    fixed_for_who = getattr(state, 'fixed_mentsu', {})
    if isinstance(fixed_for_who, dict):
        fixed_for_who = fixed_for_who.get(who, [])
    else:
        fixed_for_who = []

    visible_dora_in_meld = sum(
        meld.count(d) for meld in fixed_for_who for d in dora_tiles
    )
    if visible_dora_in_meld >= 2:
        reasons.append("dora_meld_high")
        mult *= 1.15
        return 'high', mult, reasons
    elif visible_dora_in_meld == 1:
        reasons.append("dora_meld_mid")
        mult *= 1.08

    # 副露数が多いと打点は低めになりがち
    naki_count = len(fixed_for_who)
    if naki_count >= 2:
        reasons.append("open2+")
        mult *= 0.90
        return 'low', mult, reasons

    return 'mid', mult, reasons


def get_player_danger_weight_with_reason(state, who):
    """
    相手 who に対する危険度重みと理由を返す
    """
    weight = 1.0
    reasons = []

    dealer_pid = getattr(state, "dealer_pid", 0)
    is_orasu_flag = getattr(state, "is_orasu", False)

    rival_pids = getattr(state, "rival_pids", None)
    if rival_pids is None:
        rival_pids = infer_rival_pids_from_scores(state)

    riichi_players = get_riichi_players(state)
    multi_riichi = (len(riichi_players) >= 2)

    # 基本重み
    if who == dealer_pid:
        weight *= 1.20
        reasons.append("dealer")

    if is_orasu_flag and who in rival_pids:
        weight *= 1.15
        reasons.append("orasu rival")

    # 複数リーチ時の追加重み
    if multi_riichi:
        if who == dealer_pid:
            weight *= 1.10
            reasons.append("multi-riichi dealer boost")

        if who in rival_pids:
            weight *= 1.08
            reasons.append("multi-riichi rival boost")

    # 相手打点推定による追加重み
    value_level, value_mult, value_reasons = estimate_enemy_hand_value_level(state, who)
    weight *= value_mult

    reasons.append(f"value:{value_level}")
    for vr in value_reasons:
        reasons.append(f"value-{vr}")

    return weight, reasons

def calculate_simple_discard_risk_with_reason(state, tile_idx):
    """
    危険度と理由を返す版
    戻り値:
      (risk, reasons)
    例:
      (0.35, ["player1: suji"])
      (1.30, ["player1: dora tile"])
    """
    riichi_players = get_riichi_players(state)

    if len(riichi_players) == 0:
        return 0.0, ["no riichi players"]

    total_risk = 0.0
    reasons = []

    for pid in riichi_players:
        enemy_discards = state.discards[pid]

        # 1. 現物
        if tile_idx in enemy_discards:
            base_risk = 0.0
            reasons.append(f"player{pid}: genbutsu")

        # 2. 字牌
        elif tile_idx >= 27:
            base_risk = calculate_honor_tile_risk(state, tile_idx, enemy_discards)

            if base_risk == 0.2:
                reasons.append(f"player{pid}: honor 2+ visible")
            elif base_risk == 0.6:
                reasons.append(f"player{pid}: honor 1 visible")
            else:
                reasons.append(f"player{pid}: honor live")

        else:
            # 3. ノーチャンス / ワンチャンス
            onechance_risk = get_onechance_risk(state, tile_idx)
            if onechance_risk is not None:
                base_risk = onechance_risk
                if onechance_risk == 0.15:
                    reasons.append(f"player{pid}: no-chance")
                else:
                    reasons.append(f"player{pid}: one-chance")

            else:
                # 4. 壁
                kabe_risk = get_kabe_risk(state, tile_idx)
                if kabe_risk is not None:
                    base_risk = kabe_risk
                    reasons.append(f"player{pid}: kabe")

                # 5. 筋
                elif is_suji_for_player(enemy_discards, tile_idx):
                    base_risk = 0.35
                    reasons.append(f"player{pid}: suji")

                # 6. その他
                else:
                    base_risk = 1.0
                    reasons.append(f"player{pid}: dangerous")

        # 7. ドラ危険度補正
        dora_bonus = get_dora_danger_bonus(state, tile_idx)
        if dora_bonus > 0:
            if dora_bonus == 0.30:
                reasons.append("dora tile")
            elif dora_bonus == 0.15:
                reasons.append("near dora")

        # 8. 相手別危険重み
        player_weight, weight_reasons = get_player_danger_weight_with_reason(state, pid)
        weighted_risk = (base_risk + dora_bonus) * player_weight

        if player_weight > 1.0:
            reasons.append(f"player{pid}: danger weight x{player_weight:.2f}")
            for wr in weight_reasons:
                reasons.append(f"player{pid}: {wr}")

        total_risk += weighted_risk

    return total_risk, reasons

def get_suji_candidates(discard_tile):
    """
    1枚の捨て牌から、その筋牌候補のリストを返す
    例:
      4m(牌番号3) -> [1m, 7m] -> [0, 6]
      5p(牌番号13) -> [2p, 8p] -> [10, 16]
    """
    # 字牌は筋なし
    if discard_tile >= 27:
        return []

    suit_base = (discard_tile // 9) * 9   # 0, 9, 18 のどれか
    rank = discard_tile % 9               # 0〜8 （1〜9牌に対応）

    suji = []

    # 4を切っていたら1と7が筋、のような対応
    if rank - 3 >= 0:
        suji.append(suit_base + (rank - 3))
    if rank + 3 <= 8:
        suji.append(suit_base + (rank + 3))

    return suji


def is_suji_for_player(discarded_tiles, tile_idx):
    """
    あるプレイヤーの河 discarded_tiles に対して、
    tile_idx が筋として通りそうかを判定する
    """
    # 字牌は筋判定しない
    if tile_idx >= 27:
        return False

    for d in discarded_tiles:
        if tile_idx in get_suji_candidates(d):
            return True

    return False

def calculate_honor_tile_risk(state, tile_idx, enemy_discards):
    """
    字牌専用の簡易危険度
    ルール:
      - 現物なら 0.0
      - 見えている枚数が2枚以上なら 0.2
      - 1枚見えなら 0.6
      - 生牌なら 1.0
    """
    # 現物なら安全
    if tile_idx in enemy_discards:
        return 0.0

    visible_count = 0

    # 自分の手牌に何枚あるか
    visible_count += state.hand[tile_idx]

    # 全員の河に何枚見えているか
    for pid in [0, 1, 2, 3]:
        visible_count += state.discards[pid].count(tile_idx)

    # 副露に何枚あるかも加算したいなら将来ここで追加可能
    # 今回はまずシンプルにここまで

    if visible_count >= 2:
        return 0.2
    elif visible_count == 1:
        return 0.6
    else:
        return 1.0

def get_visible_count_for_tile(state, tile_idx):
    """
    ある牌が何枚見えているかを数える
    今回は
      - 自分の手牌
      - 全員の河
    を見える枚数として数える
    """
    count = 0

    # 自分の手牌
    count += state.hand[tile_idx]

    # 全員の河
    for pid in [0, 1, 2, 3]:
        count += state.discards[pid].count(tile_idx)

    return count

def get_onechance_targets(tile_idx):
    """
    その牌に対して、安全度判定に使う中央牌候補を返す
    簡易版として、筋に対応する中央側の牌を見る

    例:
      1m -> 4m
      2m -> 5m
      3m -> 6m
      4m -> 1m, 7m ではなく、今回は中央安全度用に 2m,3m,5m,6m のような
           厳密実装は複雑なので、まずは筋ベース拡張として 7m を見る近似ではなく、
           「tile_idx ± 3」を参照する簡易法にする
    """
    if tile_idx >= 27:
        return []

    suit_base = (tile_idx // 9) * 9
    rank = tile_idx % 9

    targets = []

    if rank - 3 >= 0:
        targets.append(suit_base + (rank - 3))
    if rank + 3 <= 8:
        targets.append(suit_base + (rank + 3))

    return targets

def get_onechance_risk(state, tile_idx):
    """
    数牌に対するワンチャンス/ノーチャンスの簡易危険度を返す
    返り値:
      0.15 = ノーチャンス
      0.25 = ワンチャンス
      None = 該当なし
    """
    if tile_idx >= 27:
        return None

    targets = get_onechance_targets(tile_idx)
    if len(targets) == 0:
        return None

    max_visible = 0
    for t in targets:
        visible = get_visible_count_for_tile(state, t)
        if visible > max_visible:
            max_visible = visible

    if max_visible >= 4:
        return 0.15   # ノーチャンス
    elif max_visible >= 3:
        return 0.25   # ワンチャンス
    else:
        return None

def get_kabe_risk(state, tile_idx):
    """
    数牌に対する簡易壁判定
    隣の牌が4枚見えていれば壁として少し安全寄りにする

    返り値:
      0.18 = 壁
      None = 該当なし
    """
    if tile_idx >= 27:
        return None

    suit_base = (tile_idx // 9) * 9
    rank = tile_idx % 9  # 0〜8

    neighbors = []

    # 同じ色の隣牌だけを見る
    if rank - 1 >= 0:
        neighbors.append(suit_base + (rank - 1))
    if rank + 1 <= 8:
        neighbors.append(suit_base + (rank + 1))

    for n in neighbors:
        visible = get_visible_count_for_tile(state, n)
        if visible >= 4:
            return 0.18

    return None

def get_dora_tiles_from_indicators(dora_indicators):
    """
    ドラ表示牌のリストから、実際のドラ牌リストを返す
    牌番号は 0〜33 を想定
    """
    dora_tiles = []

    for ind in dora_indicators:
        # 萬子 1-9
        if 0 <= ind <= 8:
            dora_tiles.append((ind // 9) * 9 + ((ind % 9 + 1) % 9))

        # 筒子 1-9
        elif 9 <= ind <= 17:
            base = 9
            rank = ind - base
            dora_tiles.append(base + ((rank + 1) % 9))

        # 索子 1-9
        elif 18 <= ind <= 26:
            base = 18
            rank = ind - base
            dora_tiles.append(base + ((rank + 1) % 9))

        # 東南西北
        elif 27 <= ind <= 30:
            dora_tiles.append(27 + ((ind - 27 + 1) % 4))

        # 白發中
        elif 31 <= ind <= 33:
            dora_tiles.append(31 + ((ind - 31 + 1) % 3))

    return dora_tiles

def get_dora_danger_bonus(state, tile_idx):
    """
    ドラ・ドラ近辺の危険度補正を返す
    返り値:
      0.30 = ドラそのもの
      0.15 = ドラの隣
      0.0  = それ以外
    """
    dora_tiles = get_dora_tiles_from_indicators(state.dora_indicators)

    # ドラそのもの
    if tile_idx in dora_tiles:
        return 0.30

    # 数牌のみ隣接を判定
    if tile_idx < 27:
        for d in dora_tiles:
            if d >= 27:
                continue

            # 同じ色だけ
            if (tile_idx // 9) != (d // 9):
                continue

            if abs((tile_idx % 9) - (d % 9)) == 1:
                return 0.15

    return 0.0


# ============================================================
# B-2: 状況適応型重み調整ヘルパー関数群
# ゲーム状況（リーチ・巡目・点差・オーラス）に応じて
# EV/リスク/受け入れの重みを動的に調整する
# ============================================================

def get_riichi_players(state):
    """リーチ中の他家プレイヤーリスト（自家POV=0 を除く）"""
    return [p for p in [1, 2, 3] if state.riichi_declared.get(p, False)]


def get_naki_count(state):
    """自家の副露数"""
    return len(state.fixed_mentsu) if hasattr(state, 'fixed_mentsu') else 0


def is_open_hand(state):
    """自家が鳴いているか"""
    return get_naki_count(state) > 0


def is_dealer(state):
    """自家が親か (jikaze==0 が親)"""
    return getattr(state, 'jikaze', 1) == 0


def is_orasu(state):
    """オーラス（最終局）か"""
    bakaze = getattr(state, 'bakaze', 0)
    kyoku  = getattr(state, 'kyoku', 0)
    # bakaze=1(南場), kyoku=3(4局目)がオーラス
    return bakaze >= 1 and kyoku >= 3


def get_junme_stage(state):
    """
    巡目ステージを返す: 'early'(1-5) / 'mid'(6-11) / 'late'(12+)
    巡目は捨て牌数の合計 / 4 で近似
    """
    total_discards = sum(len(d) for d in state.discards.values()) if hasattr(state, 'discards') else 0
    junme = total_discards // 4 + 1
    if junme <= 5:
        return 'early'
    elif junme <= 11:
        return 'mid'
    else:
        return 'late'


def get_score_pressure(state):
    """
    点差プレッシャー: 自家の点数状況を返す
    'safe': トップ or 安全圏 / 'normal' / 'danger': 危機的
    """
    scores = getattr(state, 'scores', [25000, 25000, 25000, 25000])
    my_score = scores[0]
    if my_score >= 35000:
        return 'safe'
    elif my_score >= 20000:
        return 'normal'
    else:
        return 'danger'


def get_placement_pressure(state):
    """
    順位プレッシャー: 自家の推定順位を返す (0=1位, 3=4位)
    """
    scores = getattr(state, 'scores', [25000, 25000, 25000, 25000])
    my_score = scores[0]
    rank = sum(1 for s in scores[1:] if s > my_score)
    return rank


def compute_game_situation(state):
    """
    残り局数・点差・親子・巡目から総合的な戦況コンテキストを計算する。

    Returns dict:
        rank                     : int   自家の現在順位 (0=1位, 3=4位)
        gap_above                : int   直上順位との点差 (1位なら0)
        gap_below                : int   直下順位との点差 (4位なら自スコア)
        kyoku_remaining          : int   推定残り局数 (現在局含む)
        target_rank              : int   現実的な目標順位
        target_score_needed      : int   目標達成に必要な1回の和了点数
        urgency                  : float 0.0〜1.0 (切迫度)
        can_settle_cheap         : bool  安い和了(≤3900)で着順UP可能
        need_mangan_plus         : bool  満貫以上でないと着順改善困難
        rank_improvement_possible: bool  残局×最大打点で着順UP可能
        is_dealer                : bool  自家が親か
        is_orasu                 : bool  オーラス局か
        junme_stage              : str   'early'/'mid'/'late'
        threatened_by_cheap      : bool  [1位時] 安手で逆転されうる (差≤3900)
        threatened_by_mangan     : bool  [1位時] 満貫で逆転されうる (差≤8000)
        second_place_riichi      : bool  [1位時] 2位の相手がリーチ中か
        dealer_orasu_aggressive  : bool  親オーラス1位・満貫圏外→素点積み上げモード
    """
    scores     = getattr(state, 'scores', [25000] * 4)
    bakaze     = getattr(state, 'bakaze', 0)
    kyoku      = getattr(state, 'kyoku',  1)

    my_score    = scores[0]
    sorted_desc = sorted(scores, reverse=True)
    rank        = sum(1 for s in scores[1:] if s > my_score)

    gap_above = sorted_desc[rank - 1] - my_score if rank > 0 else 0
    gap_below = my_score - sorted_desc[rank + 1] if rank < 3 else my_score

    # 残り局数 (東南戦=最大8局)
    # 現在局は「今まさに打っている局」なので残り局数に含む
    kyoku_played    = bakaze * 4 + (kyoku - 1)
    kyoku_remaining = max(1, 8 - kyoku_played)

    # 子満貫ロン(8000)を1局の最大打点基準とする (保守的推定)
    max_per_hand   = 8000
    max_achievable = kyoku_remaining * max_per_hand

    # 現実的な目標順位
    if rank == 0:
        target_rank = 0
    elif gap_above <= max_achievable:
        target_rank = rank - 1  # 1つ上を狙える
    else:
        target_rank = rank      # 現状維持

    # 目標達成に必要な点数
    if target_rank < rank:
        target_score_needed = gap_above + 100
    elif rank < 3:
        # 現状維持: 直下から追い越されないラインを守る
        target_score_needed = max(0, sorted_desc[rank + 1] + 100 - my_score)
    else:
        target_score_needed = 0  # 4位: とにかく上がる

    # urgency (0.0〜1.0)
    urgency = 0.0
    if rank > 0:
        behind  = min(1.0, gap_above / 32000)
        time    = max(0.0, 1.0 - kyoku_remaining / 8.0)
        urgency = min(1.0, behind * 0.6 + time * 0.4 + (0.2 if rank >= 3 else 0.0))

    # 親子・局面情報 (既存ヘルパーを使用)
    _is_dealer  = is_dealer(state)
    _is_orasu   = is_orasu(state)
    _junme_st   = get_junme_stage(state)

    # 1位のとき: 各相手に逆転されるリスク判定
    if rank == 0:
        gap_to_2nd           = my_score - sorted_desc[1]
        threatened_by_cheap  = gap_to_2nd <= 3900
        threatened_by_mangan = gap_to_2nd <= 8000
        # 2位の相手 (=POV上で最高点の相手) がリーチ中か
        second_pov           = max([1, 2, 3], key=lambda p: scores[p])
        second_place_riichi  = bool(
            getattr(state, 'riichi_declared', {}).get(second_pov, False)
        )
    else:
        threatened_by_cheap  = False
        threatened_by_mangan = False
        second_place_riichi  = False

    # 親オーラス素点積み上げモード:
    #   1位 + 親 + オーラス + 満貫圏外 → 積極的に打点を稼ぐ
    dealer_orasu_aggressive = (
        _is_dealer and _is_orasu
        and rank == 0
        and not threatened_by_mangan
    )

    return {
        'rank':                       rank,
        'gap_above':                  gap_above,
        'gap_below':                  gap_below,
        'kyoku_remaining':            kyoku_remaining,
        'target_rank':                target_rank,
        'target_score_needed':        target_score_needed,
        'urgency':                    urgency,
        'can_settle_cheap':           0 < gap_above <= 3900,
        'need_mangan_plus':           gap_above > 8000 and rank > 0,
        'rank_improvement_possible':  gap_above <= max_achievable,
        'is_dealer':                  _is_dealer,
        'is_orasu':                   _is_orasu,
        'junme_stage':                _junme_st,
        'threatened_by_cheap':        threatened_by_cheap,
        'threatened_by_mangan':       threatened_by_mangan,
        'second_place_riichi':        second_place_riichi,
        'dealer_orasu_aggressive':    dealer_orasu_aggressive,
    }


def get_push_fold_mode(state):
    """
    攻守モードを返す: 'attack' / 'balance' / 'defense'

    判定基準:
    - 他家リーチ数と自家シャンテン数でモードを決定
    - 終盤は守備寄りに
    """
    riichi_count = len(get_riichi_players(state))
    junme_stage  = get_junme_stage(state)

    # 複数リーチ → 守備優先
    if riichi_count >= 2:
        return 'defense'

    # 1リーチ + 終盤 → バランス
    if riichi_count == 1:
        if junme_stage == 'late':
            return 'defense'
        return 'balance'

    # 無リーチ
    if junme_stage == 'late':
        return 'balance'
    return 'attack'


def get_mode_weights(mode):
    """
    モードに応じたリランキング重みを返す

    Returns dict with:
      EV_WEIGHT, RISK_WEIGHT, SAFE_BONUS_VALUE, UKEIRE_WEIGHT, FUTURE_WEIGHT
    """
    if mode == 'attack':
        return {
            "EV_WEIGHT":        0.30,
            "RISK_WEIGHT":      0.10,
            "SAFE_BONUS_VALUE": 0.10,
            "UKEIRE_WEIGHT":    0.04,
            "FUTURE_WEIGHT":    0.80,
        }
    elif mode == 'balance':
        return {
            "EV_WEIGHT":        0.25,
            "RISK_WEIGHT":      0.20,
            "SAFE_BONUS_VALUE": 0.25,
            "UKEIRE_WEIGHT":    0.03,
            "FUTURE_WEIGHT":    0.50,
        }
    else:  # defense
        return {
            "EV_WEIGHT":        0.10,
            "RISK_WEIGHT":      0.40,
            "SAFE_BONUS_VALUE": 0.50,
            "UKEIRE_WEIGHT":    0.02,
            "FUTURE_WEIGHT":    0.20,
        }


def adjust_weights_for_multi_riichi(state, weights):
    """複数リーチ時は守備ウェイトをさらに強化"""
    riichi_count = len(get_riichi_players(state))
    if riichi_count >= 2:
        w = dict(weights)
        w["RISK_WEIGHT"]      = min(0.60, w["RISK_WEIGHT"] * 1.5)
        w["SAFE_BONUS_VALUE"] = min(0.80, w["SAFE_BONUS_VALUE"] * 1.5)
        w["EV_WEIGHT"]        = w["EV_WEIGHT"] * 0.5
        return w
    return weights


def adjust_weights_for_junme(state, weights):
    """終盤（12巡目以降）はEVを下げ守備を上げる"""
    if get_junme_stage(state) == 'late':
        w = dict(weights)
        w["EV_WEIGHT"]    = w["EV_WEIGHT"] * 0.7
        w["RISK_WEIGHT"]  = min(0.60, w["RISK_WEIGHT"] * 1.2)
        return w
    return weights


def adjust_weights_for_score_and_dealer(state, weights):
    """点数状況・親かどうかで重みを調整"""
    ctx = compute_game_situation(state)
    w   = dict(weights)

    # urgency が非常に高い (絶望的状況) 場合のみ攻撃的に
    # 閾値を高めに設定して AllAI での過剰攻撃を防ぐ
    if ctx['urgency'] >= 0.75:
        w["EV_WEIGHT"]   = min(0.40, w["EV_WEIGHT"] * 1.3)
        w["RISK_WEIGHT"] = w["RISK_WEIGHT"] * 0.8
    elif ctx['rank'] == 0 and not ctx['threatened_by_mangan']:
        # 安全な1位: 守備的に (旧来の 'safe' 相当)
        w["RISK_WEIGHT"] = min(0.50, w["RISK_WEIGHT"] * 1.1)
    elif get_score_pressure(state) == 'danger':
        # 点棒20000以下の危機的状況 (旧来の 'danger' 相当)
        w["EV_WEIGHT"]   = min(0.40, w["EV_WEIGHT"] * 1.3)
        w["RISK_WEIGHT"] = w["RISK_WEIGHT"] * 0.8

    # 親番: 連荘狙いでやや攻撃的に
    if ctx['is_dealer']:
        w["EV_WEIGHT"] = min(0.45, w["EV_WEIGHT"] * 1.1)

    # 1位 + 満貫逆転脅威あり: 守備補正
    if ctx['rank'] == 0 and ctx['threatened_by_mangan']:
        w["RISK_WEIGHT"]      = min(0.60, w["RISK_WEIGHT"] * 1.3)
        w["SAFE_BONUS_VALUE"] = min(0.80, w["SAFE_BONUS_VALUE"] * 1.2)
        w["EV_WEIGHT"]        = w["EV_WEIGHT"] * 0.85

    return w


def adjust_weights_for_orasu(state, weights):
    """オーラスは順位優先 → compute_game_situation ベースで調整"""
    if not is_orasu(state):
        return weights
    ctx = compute_game_situation(state)
    w   = dict(weights)

    if ctx['dealer_orasu_aggressive']:
        # 親オーラス1位・満貫圏外: 素点積み上げ → 攻撃維持
        w["EV_WEIGHT"]   = min(0.45, w["EV_WEIGHT"] * 1.2)
        w["RISK_WEIGHT"] = w["RISK_WEIGHT"] * 0.85
    elif ctx['rank'] == 0 and ctx['threatened_by_cheap']:
        # 1位・安手逆転圏内: 放銃厳禁
        w["RISK_WEIGHT"]      = min(0.70, w["RISK_WEIGHT"] * 1.6)
        w["SAFE_BONUS_VALUE"] = min(0.90, w["SAFE_BONUS_VALUE"] * 1.5)
        w["EV_WEIGHT"]        = w["EV_WEIGHT"] * 0.5
    elif ctx['rank'] == 0 and ctx['threatened_by_mangan']:
        # 1位・満貫逆転圏内: 守備的
        w["RISK_WEIGHT"]      = min(0.65, w["RISK_WEIGHT"] * 1.5)
        w["SAFE_BONUS_VALUE"] = min(0.85, w["SAFE_BONUS_VALUE"] * 1.4)
        w["EV_WEIGHT"]        = w["EV_WEIGHT"] * 0.6
    elif ctx['rank'] == 0:
        # 1位・大差: 現状維持で守備的
        w["RISK_WEIGHT"]      = min(0.60, w["RISK_WEIGHT"] * 1.4)
        w["SAFE_BONUS_VALUE"] = min(0.80, w["SAFE_BONUS_VALUE"] * 1.3)
        w["EV_WEIGHT"]        = w["EV_WEIGHT"] * 0.6
    elif ctx['rank'] >= 2:
        # 3-4位: 逆転狙いで攻撃的
        w["EV_WEIGHT"]   = min(0.50, w["EV_WEIGHT"] * 1.4)
        w["RISK_WEIGHT"] = w["RISK_WEIGHT"] * 0.7
    # 2位はほぼ現状維持

    return w


def adjust_weights_for_open_hand(state, weights):
    """副露手（鳴きあり）ではukeireとEVを重視、NNへの依存を下げる"""
    if not is_open_hand(state):
        return weights
    w = dict(weights)
    # 副露後はNNスコアより受け入れ枚数・EVを優先
    w["UKEIRE_WEIGHT"] = w["UKEIRE_WEIGHT"] * 1.8  # 待ち牌最大化を強化
    w["EV_WEIGHT"]     = min(0.45, w["EV_WEIGHT"] * 1.3)  # ダマEVをやや重視
    return w


def get_balance_tile_adjustment(risk_value):
    """
    バランスモード時の牌ごとの補正
    危険度が高い牌にペナルティ、安全な牌にボーナス
    """
    if risk_value >= 0.6:
        return -0.15, ["balance_high_risk_penalty"]
    elif risk_value <= 0.2:
        return 0.05, ["balance_safe_bonus"]
    return 0.0, []


# ============================================================
# シャンテン・受け入れ・将来性計算ヘルパー
# ============================================================

def calculate_shanten_unified(state, hand_counts):
    """
    副露対応シャンテン数統合版 (高速版: lru_cacheつきDFSを使用)
    - 副露がある場合は通常手のみ (副露DFS版)
    - 副露がない場合は通常手DFS + 七対子チェック
    """
    fixed_mentsu = getattr(state, 'fixed_mentsu', [])
    if fixed_mentsu:
        return calculate_normal_shanten_with_open_hand_exact(hand_counts, fixed_mentsu)
    else:
        # 通常手DFS (lru_cacheで高速)
        normal_sh = calculate_normal_shanten_with_open_hand_exact(hand_counts, [])
        # 七対子 (対子数で簡易計算)
        pairs = sum(1 for c in hand_counts if c >= 2)
        chiitoi_sh = 6 - pairs
        return min(normal_sh, chiitoi_sh)


def calculate_hand_shanten_after_discard(state, tile_idx):
    """tile_idx を切った後の手牌シャンテン数"""
    temp_h = state.hand.copy()
    if temp_h[tile_idx] <= 0:
        return 8
    temp_h[tile_idx] -= 1
    return calculate_shanten_unified(state, temp_h)


def calculate_wait_count_with_open_hand(state, tile_idx):
    """tile_idx を切ってテンパイのとき、残り待ち牌枚数を返す"""
    temp_h = state.hand.copy()
    if temp_h[tile_idx] <= 0:
        return 0
    temp_h[tile_idx] -= 1
    wait_tiles = get_waiting_tiles_with_open_hand(state, temp_h)
    return count_remaining_tiles_for_list(state, temp_h, wait_tiles)


def calculate_ukeire_count_with_open_hand(state, tile_idx):
    """tile_idx を切った後の有効牌残り枚数を返す"""
    temp_h = state.hand.copy()
    if temp_h[tile_idx] <= 0:
        return 0
    temp_h[tile_idx] -= 1
    eff_tiles = get_effective_draw_tiles_with_open_hand(state, temp_h)
    return count_remaining_tiles_for_list(state, temp_h, eff_tiles)


def calculate_simple_daten_potential(state, tile_idx):
    """
    tile_idx を切った後の打点上昇余地スコアを返す
    (bonus, reasons) のタプル
    ドラ保有数・役牌対子を簡易評価
    """
    dora_indicators = getattr(state, 'dora_indicators', [])
    dora_tiles = get_dora_tiles_from_indicators(dora_indicators)

    temp_h = state.hand.copy()
    if temp_h[tile_idx] > 0:
        temp_h[tile_idx] -= 1

    dora_count = sum(temp_h[t] for t in dora_tiles if t < 34)

    # 役牌の対子ボーナス (場風・自風・三元牌)
    bakaze = getattr(state, 'bakaze', 0)
    jikaze = getattr(state, 'jikaze', 1)
    yakuhai_tiles = {27 + bakaze, 27 + jikaze, 31, 32, 33}
    honor_bonus = sum(0.05 for t in yakuhai_tiles if t < 34 and temp_h[t] >= 2)

    bonus = dora_count * 0.04 + honor_bonus
    reasons = []
    if dora_count > 0:
        reasons.append(f"dora:{dora_count}")
    if honor_bonus > 0:
        reasons.append("yakuhai_pair")
    return bonus, reasons


def calculate_shape_bonus(state, tile_idx):
    """
    tile_idx を切った後の手牌形評価ボーナス
    孤立牌が少なく接続牌が多いほどボーナス大
    """
    temp_h = state.hand.copy()
    if temp_h[tile_idx] <= 0:
        return 0.0, []
    temp_h[tile_idx] -= 1

    isolated_count = 0
    connected_count = 0

    for i in range(34):
        if temp_h[i] == 0:
            continue
        if i >= 27:
            # 字牌: 対子以上なら接続
            if temp_h[i] >= 2:
                connected_count += 1
            else:
                isolated_count += 1
        else:
            suit = i // 9
            has_neighbor = (temp_h[i] >= 2)
            for offset in [-2, -1, 1, 2]:
                n = i + offset
                if 0 <= n < 34 and n // 9 == suit and temp_h[n] > 0:
                    has_neighbor = True
                    break
            if has_neighbor:
                connected_count += 1
            else:
                isolated_count += 1

    bonus = connected_count * 0.01 - isolated_count * 0.02
    reasons = []
    if isolated_count > 0:
        reasons.append(f"isolated:{isolated_count}")
    if connected_count > 0:
        reasons.append(f"connected:{connected_count}")
    return bonus, reasons


def calculate_route_bonus(state, tile_idx):
    """
    tile_idx を切った後の手役経路ボーナス（簡易版）
    返り値: (bonus: float, reasons: dict)
    """
    temp_h = state.hand.copy()
    if temp_h[tile_idx] <= 0:
        return 0.0, {"normal": []}
    temp_h[tile_idx] -= 1

    shanten = calculate_shanten_unified(state, temp_h)
    if shanten <= 0:
        return 0.10, {"normal": ["tenpai_route"]}
    elif shanten == 1:
        return 0.05, {"normal": ["iishanten_route"]}
    return 0.0, {"normal": []}


def calculate_future_potential_bonus_v2_fast(state, tile_idx, max_future_tiles=3):
    """
    tile_idx を切った後の将来性評価（高速版）
    接続牌数でシャンテン改善可能性を近似 (full shanten計算は避ける)
    返り値: (bonus: float, details: list)
    """
    temp_h = state.hand.copy()
    if temp_h[tile_idx] <= 0:
        return 0.0, []
    temp_h[tile_idx] -= 1

    # シャンテン数（calculate_hand_shanten_after_discardと同じだが再計算を避けるため簡略化）
    current_shanten = calculate_shanten_unified(state, temp_h)
    if current_shanten < 0:
        return 0.0, ["agari"]

    # 有効牌の種類数を簡易カウント (接続牌ベース)
    # 各手牌に対してその牌を追加したときにシャンテンが下がるか、隣接牌で近似
    connected_kinds = 0
    for draw in range(34):
        if temp_h[draw] >= 4:
            continue
        is_useful = False
        if draw >= 27:
            # 字牌: 役牌（場風・自風・三元牌）のみ対子ボーナスを評価
            # 非役牌孤立字牌（南・西・北など）は有効牌としてカウントしない
            _fp_yakuhai = {27 + state.bakaze, 27 + state.jikaze, 31, 32, 33}
            if draw in _fp_yakuhai and temp_h[draw] == 1:
                is_useful = True
        else:
            suit = draw // 9
            rank = draw % 9
            # 隣接牌があれば有効（offset=0は除外: 孤立牌を自己ドローで有効判定しない）
            for offset in [-2, -1, 1, 2]:
                n = draw + offset
                if 0 <= n < 34 and n // 9 == suit and temp_h[n] > 0:
                    is_useful = True
                    break
        if is_useful:
            connected_kinds += 1

    # 残り枚数を簡易推定 (visible_tilesは高コストなので省略し平均2枚と近似)
    eff_count_approx = connected_kinds * 2

    potential = min(eff_count_approx / 136.0 / (current_shanten + 2), 0.5)
    details = [f"connected_kinds:{connected_kinds}, shanten:{current_shanten}"]
    return potential, details


def hybrid_ai_decision_v6_rerank_debug(state, ai_model, top_k=5):
    """
    rerank方式のデバッグ版
    流れ:
      1. NNで候補を top_k 枚に絞る
      2. 候補だけ EV + 守備 + シャンテン + 受け入れで再評価
      3. 最終決定する
    """
    # skip_logic=False: CH23-26(シャンテン/受け入れ/EV)を含む完全版テンソル（33ch）
    tensor_data_np = state.to_tensor(skip_logic=False)
    tensor = torch.tensor(tensor_data_np, dtype=torch.float32).unsqueeze(0)

    ai_model.eval()
    with torch.no_grad():
        out_discard, out_riichi = ai_model(tensor)
        raw_discard_probs = F.softmax(out_discard, dim=1)[0].numpy()

    # EV は候補牌ごとに後で個別計算するのでここでは不要
    # (ev_riichi_ch / ev_dama_ch は後のループ内で compute_candidate_ev() を呼ぶ)

    # パラメータ
    NN_WEIGHT = 1.0

    mode = get_push_fold_mode(state)
    junme_stage = get_junme_stage(state)
    score_pressure = get_score_pressure(state)
    dealer = is_dealer(state)
    orasu = is_orasu(state)
    placement_pressure = get_placement_pressure(state)

    mode_weights = get_mode_weights(mode)
    mode_weights = adjust_weights_for_multi_riichi(state, mode_weights)
    mode_weights = adjust_weights_for_junme(state, mode_weights)
    mode_weights = adjust_weights_for_score_and_dealer(state, mode_weights)
    mode_weights = adjust_weights_for_orasu(state, mode_weights)
    mode_weights = adjust_weights_for_open_hand(state, mode_weights)

    EV_WEIGHT = mode_weights["EV_WEIGHT"]
    RISK_WEIGHT = mode_weights["RISK_WEIGHT"]
    SAFE_BONUS_VALUE = mode_weights["SAFE_BONUS_VALUE"]
    UKEIRE_WEIGHT = mode_weights["UKEIRE_WEIGHT"]
    FUTURE_WEIGHT = mode_weights["FUTURE_WEIGHT"]

    # 副露手はリーチEVが使えないためテンパイボーナスを補強
    if is_open_hand(state):
        SHANTEN_BONUS_TENPAI = 0.70
        SHANTEN_BONUS_1SHANTEN = 0.35
    else:
        SHANTEN_BONUS_TENPAI = 0.60
        SHANTEN_BONUS_1SHANTEN = 0.30

    # 手牌にある牌だけ候補にする
    legal_tiles = [i for i in range(34) if state.hand[i] > 0]

    # 禁止打牌を除く
    legal_tiles = [i for i in legal_tiles if i not in state.forbidden_discards]

    if len(legal_tiles) == 0:
        return 0, [0.0] * 34, []

    # NNスコア上位 top_k を候補化
    sorted_by_nn = sorted(
        legal_tiles,
        key=lambda i: raw_discard_probs[i],
        reverse=True
    )
    candidate_tiles = sorted_by_nn[:top_k]

    # テンパイ達成牌は必ず候補に含める（リーチ機会を逃さないため）
    for t in legal_tiles:
        if t not in candidate_tiles:
            temp_h = state.hand.copy()
            temp_h[t] -= 1
            if calculate_shanten_unified(state, temp_h) == 0:
                candidate_tiles.append(t)

    # 1シャンテン時: 有効牌数が最大の打牌も必ず候補に含める
    # NNがトップ5に選ばなくても最大ukeire牌を評価対象にする
    current_shanten_main = calculate_shanten_unified(state, state.hand)
    if current_shanten_main == 1:
        best_ukeire_tile = -1
        best_ukeire_val = -1
        for t in legal_tiles:
            if t not in candidate_tiles:
                val = calculate_ukeire_count_with_open_hand(state, t)
                if val > best_ukeire_val:
                    best_ukeire_val = val
                    best_ukeire_tile = t
        if best_ukeire_tile >= 0:
            candidate_tiles.append(best_ukeire_tile)

    # 2シャンテン以上かつ副露手: シャンテンを最も改善する打牌を必ず候補に含める
    # 副露後の2シャンテンでNNの精度が低い場合の安全網
    if current_shanten_main >= 2 and is_open_hand(state):
        best_shanten_tile = -1
        best_shanten_val = 99
        best_shanten_ukeire = -1
        for t in legal_tiles:
            if t not in candidate_tiles:
                temp_h = state.hand.copy()
                temp_h[t] -= 1
                s = calculate_shanten_unified(state, temp_h)
                eff = len(get_effective_draw_tiles_with_open_hand(state, temp_h))
                if s < best_shanten_val or (s == best_shanten_val and eff > best_shanten_ukeire):
                    best_shanten_val = s
                    best_shanten_ukeire = eff
                    best_shanten_tile = t
        if best_shanten_tile >= 0:
            candidate_tiles.append(best_shanten_tile)

    # 非役牌孤立字牌は必ず候補に含める
    # NNが「切らない」と学習した場合でも、役のない孤立字牌は常に打牌候補として評価する
    _yakuhai_set_cand = {27 + state.bakaze, 27 + state.jikaze, 31, 32, 33}
    for t in legal_tiles:
        if t not in candidate_tiles and t >= 27 and t not in _yakuhai_set_cand:
            if state.hand[t] == 1:  # 孤立（対子でない）字牌のみ
                candidate_tiles.append(t)

    # リーチ者の現物一覧
    safe_tiles = set()
    is_under_attack = False
    for who in [1, 2, 3]:
        if state.riichi_declared[who]:
            is_under_attack = True
            for tile in state.discards[who]:
                safe_tiles.add(tile)

    debug_rows = []
    final_scores = [0.0] * 34

    for i in candidate_tiles:
        nn_score_raw = float(raw_discard_probs[i])
        nn_score = nn_score_raw ** 0.5

        # シャンテン評価 (EV計算の前に必要)
        after_shanten = calculate_hand_shanten_after_discard(state, i)

        # EV: テンパイ時のみ calculate_true_ev を呼ぶ (非テンパイはEV=0)
        ev_value = 0.0
        if after_shanten <= 0:
            temp_h_ev = state.hand.copy()
            temp_h_ev[i] -= 1
            wait_tiles_ev = get_waiting_tiles_with_open_hand(state, temp_h_ev)
            if wait_tiles_ev:
                visible_ev = build_visible_tiles34(state)
                try:
                    _ev_r, _ev_d = calculate_true_ev(
                        temp_h_ev,
                        getattr(state, 'fixed_mentsu', []),
                        wait_tiles_ev,
                        visible_ev,
                        getattr(state, 'dora_indicators', []),
                        state.bakaze,
                        state.jikaze,
                        is_oya=(state.jikaze == 0),
                        honba=getattr(state, 'honba', 0),
                        kyotaku=getattr(state, 'kyotaku', 0),
                    )
                    ev_value = max(_ev_r / 8000.0, _ev_d / 8000.0)
                except Exception:
                    ev_value = 0.0
        ev_bonus = ev_value * EV_WEIGHT

        # 安全牌ボーナス
        safe_bonus = 0.0
        if is_under_attack and i in safe_tiles:
            safe_bonus = SAFE_BONUS_VALUE

        # 守備評価
        risk_value, risk_reasons = calculate_simple_discard_risk_with_reason(state, i)
        risk_penalty = risk_value * RISK_WEIGHT

        # balanceモード時の候補別押し引き補正
        tile_mode_adjustment = 0.0
        tile_mode_reasons = []

        if mode == "balance":
            tile_mode_adjustment, tile_mode_reasons = get_balance_tile_adjustment(risk_value)

        shanten_bonus = 0.0
        if after_shanten <= 0:
            shanten_bonus = SHANTEN_BONUS_TENPAI
        elif after_shanten == 1:
            shanten_bonus = SHANTEN_BONUS_1SHANTEN

        # 受け入れ枚数評価
        # テンパイ時は待ち牌枚数、非テンパイ時は有効牌枚数（ただし重みを1/4に抑制）
        # 非テンパイの有効牌は20〜40枚になりうるため、等重みだとテンパイボーナスを超えてしまう
        if after_shanten <= 0:
            ukeire_count = calculate_wait_count_with_open_hand(state, i)
            ukeire_bonus = ukeire_count * UKEIRE_WEIGHT
        else:
            ukeire_count = calculate_ukeire_count_with_open_hand(state, i)
            ukeire_bonus = ukeire_count * UKEIRE_WEIGHT * 0.25  # 非テンパイは重みを抑制

        # 打点上昇余地
        daten_bonus, daten_reasons = calculate_simple_daten_potential(state, i)

        # 通常手の形評価
        # テンパイ時は shape/route は不要（shanten_bonus が代わりに評価する）
        if after_shanten <= 0:
            shape_bonus, shape_reasons = 0.0, ["tenpai_skip"]
            route_bonus, route_reasons = 0.0, {"normal": ["tenpai_skip"]}
            future_bonus_raw, future_details = 0.0, []
            future_bonus = 0.0
        else:
            shape_bonus, shape_reasons = calculate_shape_bonus(state, i)
            route_bonus, route_reasons = calculate_route_bonus(state, i)
            future_bonus_raw, future_details = calculate_future_potential_bonus_v2_fast(
                state, i, max_future_tiles=3
            )
            future_bonus = future_bonus_raw * FUTURE_WEIGHT

        # 非役牌孤立字牌ボーナス: NNバイアスを上回る強い補正
        # 場風・自風・三元牌以外の孤立字牌（南・西・北など）はほぼ常に最優先で切るべき
        # ただしドラ牌は例外：非役牌でもドラなら切らない
        isolated_honor_bonus = 0.0
        if i >= 27 and state.hand[i] == 1 and after_shanten > 0:
            _yakuhai_set = {27 + state.bakaze, 27 + state.jikaze, 31, 32, 33}
            _actual_doras = get_dora_tiles_from_indicators(getattr(state, 'dora_indicators', []))
            if i not in _yakuhai_set and i not in _actual_doras:
                isolated_honor_bonus = 0.40

        # 5m/5p/5s の自牌価値ペナルティ（赤ドラ候補保護）
        # 赤5m=tile4, 赤5p=tile13, 赤5s=tile22 を切る場合、手牌価値が高いため抑制
        # ※ 黒5でも同様のペナルティが入るが平均的に有益（赤1枚混在の確率が高い）
        aka_value_penalty = 0.0
        if i in (4, 13, 22):  # 5m, 5p, 5s
            # 安全牌として切る場合（低リスクな場面で切ろうとしている）に抑制
            if risk_value <= 0.45:
                aka_value_penalty = 0.12

        # route_bonus はそのままだと強すぎるので少し弱めて混ぜる
        raw_final_score = (
            nn_score * NN_WEIGHT
            + ev_bonus
            + safe_bonus
            + shanten_bonus
            + ukeire_bonus
            + daten_bonus
            + shape_bonus
            + route_bonus * 0.35
            + future_bonus
            + tile_mode_adjustment
            + isolated_honor_bonus
            - risk_penalty
            - aka_value_penalty
        )

        final_score = max(raw_final_score, 0.0)
        final_scores[i] = final_score

        debug_rows.append({
            "tile_idx": i,
            "nn_score_raw": float(nn_score_raw),
            "nn_score": float(nn_score),
            "ev_value": ev_value,
            "ev_bonus": ev_bonus,
            "safe_bonus": safe_bonus,
            "risk_value": float(risk_value),
            "risk_penalty": float(risk_penalty),
            "after_shanten": int(after_shanten),
            "shanten_bonus": float(shanten_bonus),
            "ukeire_count": int(ukeire_count),
            "ukeire_bonus": float(ukeire_bonus),
            "daten_bonus": float(daten_bonus),
            "daten_reasons": daten_reasons,
            "raw_final_score": float(raw_final_score),
            "final_score": float(final_score),
            "risk_reasons": risk_reasons,
            "is_candidate": True,
            "future_bonus_raw": float(future_bonus_raw),
            "future_bonus": float(future_bonus),
            "future_details": future_details,
            "mode":mode,
            "tile_mode_adjustment": float(tile_mode_adjustment),
            "tile_mode_reasons": tile_mode_reasons,
            "num_riichi": float(len(get_riichi_players(state))),
            "junme_stage": junme_stage,
            "score_pressure": score_pressure,
            "dealer": dealer,
            "orasu": orasu,
            "placement_pressure": placement_pressure,
            "naki_count": get_naki_count(state),
            "open_hand": is_open_hand(state),
            "shape_bonus": float(shape_bonus),
            "shape_reasons": shape_reasons,
            "route_bonus": float(route_bonus),
            "route_reasons": route_reasons,
        })

    # 候補外の合法牌もログに残す
    for i in legal_tiles:
        if i in candidate_tiles:
            continue
        debug_rows.append({
            "tile_idx": i,
            "nn_score": float(raw_discard_probs[i]),
            "ev_value": 0.0,
            "ev_bonus": 0.0,
            "safe_bonus": 0.0,
            "risk_value": 0.0,
            "risk_penalty": 0.0,
            "after_shanten": -1,
            "shanten_bonus": 0.0,
            "ukeire_count": 0,
            "ukeire_bonus": 0.0,
            "raw_final_score": 0.0,
            "final_score": 0.0,
            "risk_reasons": ["candidate out"],
            "is_candidate": False,
            "daten_bonus": 0.0,
            "daten_reasons": [],
            "future_bonus": 0.0,
            "future_details": [],
            "mode":mode,
            "tile_mode_adjustment": 0.0,
            "tile_mode_reasons": [],
            "naki_count": get_naki_count(state),
            "open_hand": is_open_hand(state),
        })

    total_score = sum(final_scores)
    final_probs = [0.0] * 34
    if total_score > 0:
        final_probs = [s / total_score for s in final_scores]
    else:
        best_fallback = candidate_tiles[0]
        final_probs[best_fallback] = 1.0

    best_discard = max(range(34), key=lambda x: final_probs[x])

    for row in debug_rows:
        row["final_prob"] = float(final_probs[row["tile_idx"]])

    debug_rows.sort(
        key=lambda x: (x["is_candidate"], x["final_score"], x["nn_score"]),
        reverse=True
    )

    return best_discard, final_probs, debug_rows


def translate_risk_reason(reason):
    """
    英語タグを日本語説明に変換する
    """
    mapping = {
        "no riichi players": "リーチ者なし",
        "genbutsu": "現物",
        "suji": "筋",
        "kabe": "壁",
        "one-chance": "ワンチャンス",
        "no-chance": "ノーチャンス",
        "dangerous": "無筋寄りで危険",
        "dora tile": "ドラ牌",
        "near dora": "ドラ近辺",
        "honor 2+ visible": "字牌で2枚以上見え",
        "honor 1 visible": "字牌で1枚見え",
        "honor live": "字牌の生牌",
        "forbidden discard": "禁止打牌",
    }

    # "player1: suji" のような形式を処理
    if ":" in reason:
        left, right = reason.split(":", 1)
        left = left.strip()
        right = right.strip()
        jp = mapping.get(right, right)
        return f"{left}に対して{jp}"

    return mapping.get(reason, reason)

def build_discard_explanation(best_discard, debug_rows, tile_names=None, top_k=3):
    """
    debug_rows をもとに説明文を作る
    戻り値:
      {
        "summary": ...,
        "details": [...]
      }
    """
    if not debug_rows:
        return {
            "summary": "候補情報がありません。",
            "details": []
        }

    top_rows = debug_rows[:top_k]

    def tile_label(tile_idx):
        if tile_names is not None:
            return tile_names[tile_idx]
        return f"tile_{tile_idx}"

    def build_context_sentence(row):
        parts = []

        mode = row.get("mode", None)
        dealer = row.get("dealer", False)
        score_pressure = row.get("score_pressure", "neutral")
        orasu = row.get("orasu", False)
        placement_pressure = row.get("placement_pressure", "neutral")
        naki_count = row.get("naki_count", 0)
        open_hand = row.get("open_hand", False)

        # オーラス条件
        if orasu:
            if placement_pressure == "need_attack":
                parts.append("オーラスで着順上昇が必要なため、攻撃寄りに評価しています。")
            elif placement_pressure == "protect":
                parts.append("オーラスで現状維持を優先したいため、守備寄りに評価しています。")
            else:
                parts.append("オーラス局面として押し引きを慎重に見ています。")

        # 親番
        if dealer:
            parts.append("親番なので通常より押しやすく評価しています。")

        # 点棒状況
        if score_pressure == "behind":
            parts.append("点棒状況的に攻める必要があるため、前向きに評価しています。")
        elif score_pressure == "ahead":
            parts.append("点棒状況的に守る価値があるため、慎重に評価しています。")

        # 副露状況
        if open_hand:
            if naki_count >= 2:
                parts.append("2副露以上しているため、速度重視で評価しています。")
            else:
                parts.append("副露しているため、門前手より速度重視で評価しています。")

        # 押し引きモード
        if mode == "attack":
            parts.append("局面全体では攻撃寄りのモードです。")
        elif mode == "balance":
            parts.append("局面全体では攻守バランス型のモードです。")
        elif mode == "defense":
            parts.append("局面全体では守備寄りのモードです。")

        return " ".join(parts)

    best_row = top_rows[0]
    best_tile_name = tile_label(best_row["tile_idx"])

    summary_parts = [f"推奨打牌は {best_tile_name} です。"]

    # 局面背景
    context_sentence = build_context_sentence(best_row)
    if context_sentence:
        summary_parts.append(context_sentence)

    # 速度
    if best_row.get("after_shanten", 99) <= 0:
        summary_parts.append("テンパイが取れ、")
    elif best_row.get("after_shanten", 99) == 1:
        summary_parts.append("1シャンテンに進み、")
    elif best_row.get("ukeire_count", 0) >= 40:
        summary_parts.append("受け入れが広く、")

    # EV
    if best_row.get("ev_bonus", 0.0) > 0.2:
        summary_parts.append("EVも高く、")
    elif best_row.get("ev_bonus", 0.0) > 0.05:
        summary_parts.append("EV面でも利点があり、")

    # future / 守備
    if best_row.get("future_bonus", 0.0) > 0.18:
        summary_parts.append("1手先の将来性も高いです。")
    elif best_row.get("future_bonus", 0.0) > 0.08:
        summary_parts.append("次の伸びも見込めます。")
    else:
        if best_row.get("risk_value", 99.0) <= 0.2:
            summary_parts.append("守備面でもかなり安全です。")
        elif best_row.get("risk_value", 99.0) <= 0.4:
            summary_parts.append("守備面でも比較的安定しています。")
        elif best_row.get("risk_value", 99.0) >= 1.0:
            summary_parts.append("危険度は高めですが、総合評価が最も高いです。")
        else:
            summary_parts.append("攻守のバランスで総合評価が最も高いです。")

    summary = "".join(summary_parts)

    details = []
    for idx, row in enumerate(top_rows, start=1):
        tile_name = tile_label(row["tile_idx"])
        jp_risk_reasons = [translate_risk_reason(r) for r in row.get("risk_reasons", [])]

        detail_parts = [f"{idx}位候補は {tile_name}。"]

        # 局面背景
        local_context = build_context_sentence(row)
        if local_context:
            detail_parts.append(local_context)

        # 速度説明
        after_shanten = row.get("after_shanten", None)
        ukeire_count = row.get("ukeire_count", None)

        if after_shanten is not None:
            if after_shanten <= 0:
                detail_parts.append("テンパイが取れます。")
            elif after_shanten == 1:
                detail_parts.append("1シャンテンに進みます。")
            else:
                detail_parts.append(f"{after_shanten}シャンテンです。")
        
        if row.get("open_hand", False):
            if row.get("naki_count", 0) >= 2:
                detail_parts.append("副露数が多いため、和了までの近さを特に重視しています。")
            else:
                detail_parts.append("副露手なので、門前手より速度を重視しています。")

        if ukeire_count is not None:
            if ukeire_count >= 40:
                detail_parts.append(f"受け入れはかなり広く、{ukeire_count}枚です。")
            elif ukeire_count >= 20:
                detail_parts.append(f"受け入れはまずまずで、{ukeire_count}枚です。")
            else:
                detail_parts.append(f"受け入れは{ukeire_count}枚です。")

        # EV説明
        ev_bonus = row.get("ev_bonus", 0.0)
        if ev_bonus > 0.2:
            detail_parts.append("EV補正が大きいです。")
        elif ev_bonus > 0.05:
            detail_parts.append("EV補正があります。")
        else:
            detail_parts.append("EV補正は小さめです。")

        # 打点説明
        daten_bonus = row.get("daten_bonus", 0.0)
        daten_reasons = row.get("daten_reasons", [])
        if daten_bonus > 0.15:
            detail_parts.append("打点上昇余地も高めです。")
        elif daten_bonus > 0.05:
            detail_parts.append("打点上昇余地があります。")

        if daten_reasons:
            detail_parts.append("打点面の理由: " + "、".join(daten_reasons) + "。")

        # future説明
        future_bonus = row.get("future_bonus", 0.0)
        future_details = row.get("future_details", [])
        if future_bonus > 0.18:
            detail_parts.append("1手先の将来性が高いです。")
        elif future_bonus > 0.08:
            detail_parts.append("1手先の伸びが見込めます。")

        if future_details:
            preview = " / ".join(future_details[:2])
            detail_parts.append("将来候補例: " + preview + "。")

        # 守備説明
        risk_value = row.get("risk_value", 0.0)
        if risk_value == 0.0:
            detail_parts.append("守備面では非常に安全です。")
        elif risk_value <= 0.2:
            detail_parts.append("守備面ではかなり安全寄りです。")
        elif risk_value <= 0.4:
            detail_parts.append("守備面では比較的通しやすい牌です。")
        elif risk_value <= 0.8:
            detail_parts.append("守備面ではやや注意が必要です。")
        else:
            detail_parts.append("守備面では危険度が高めです。")

        if jp_risk_reasons:
            detail_parts.append("守備面の理由: " + "、".join(jp_risk_reasons) + "。")

        # 数値
        detail_parts.append(
            f"(NN={row.get('nn_score', 0.0):.3f}, "
            f"EV補正={row.get('ev_bonus', 0.0):.3f}, "
            f"受け入れ={row.get('ukeire_count', 0)}, "
            f"未来={row.get('future_bonus', 0.0):.3f}, "
            f"危険度={row.get('risk_value', 0.0):.2f}, "
            f"最終スコア={row.get('final_score', 0.0):.3f})"
        )

        details.append(" ".join(detail_parts))

    return {
        "summary": summary,
        "details": details
    }

# ------------------------------------------
# 🀄 鳴き役判定ヘルパー関数
# ------------------------------------------

# 喰いタンに使える牌（端牌・字牌を除く2〜8）
_TANYAO_TILES = frozenset([
    1,2,3,4,5,6,7,       # 2m〜8m
    10,11,12,13,14,15,16, # 2p〜8p
    19,20,21,22,23,24,25, # 2s〜8s
])

def _check_yakuhai_pon(state, discard_tile, naki_type):
    """役牌ポンになるか判定（白發中・場風・自風）"""
    if naki_type != "pon":
        return False
    yakuhai_tiles = {31, 32, 33, 27 + state.bakaze, 27 + state.jikaze}
    return discard_tile in yakuhai_tiles

def _check_tanyao_possible(temp_hand, temp_melds):
    """喰いタン: 手牌・副露の全牌が2〜8かチェック"""
    for i, n in enumerate(temp_hand):
        if n > 0 and i not in _TANYAO_TILES:
            return False
    for meld in temp_melds:
        for t in meld:
            if t not in _TANYAO_TILES:
                return False
    return True

def _check_one_suit(temp_hand, temp_melds):
    """ホンイツ/清一色方向: 手牌+副露が1色（+字牌）に集中しているか"""
    suit_has = [False, False, False]  # man, pin, sou
    for i, n in enumerate(temp_hand):
        if n <= 0:
            continue
        if i < 9:
            suit_has[0] = True
        elif i < 18:
            suit_has[1] = True
        elif i < 27:
            suit_has[2] = True
        # 27以上は字牌 → 色扱いしない
    for meld in temp_melds:
        for t in meld:
            if t < 9:
                suit_has[0] = True
            elif t < 18:
                suit_has[1] = True
            elif t < 27:
                suit_has[2] = True
    # 使っている色が1色のみならホンイツ/清一色の可能性あり
    return sum(suit_has) == 1


def _check_dora_pon(state, discard_tile, naki_type):
    """ドラ牌のポンかどうか（シャンテン2以下のときのみ有効）"""
    if naki_type != "pon":
        return False
    dora_indicators = getattr(state, 'dora_indicators', [])
    dora_tiles = get_dora_tiles_from_indicators(dora_indicators)
    return discard_tile in dora_tiles


def _find_best_discard_after_naki(state, hand_counts):
    """副露後の最良打牌を返す（シャンテン最小化 → 有効牌数最大化）"""
    best_tile = -1
    best_shanten = 99
    best_ukeire = 0
    for tile in range(34):
        if hand_counts[tile] <= 0:
            continue
        test_h = hand_counts.copy()
        test_h[tile] -= 1
        s = calculate_shanten_unified(state, test_h)
        eff = len(get_effective_draw_tiles_with_open_hand(state, test_h))
        if s < best_shanten or (s == best_shanten and eff > best_ukeire):
            best_shanten = s
            best_ukeire = eff
            best_tile = tile
    return best_tile


def _estimate_max_tenpai_score(state, hand_counts):
    """副露テンパイ時の最大ロン点数を推定する（役なし=0）"""
    max_score = 0
    dora_indicators = getattr(state, 'dora_indicators', [])
    for draw_tile in range(34):
        if hand_counts[draw_tile] >= 4:
            continue
        test_counts = hand_counts.copy()
        test_counts[draw_tile] += 1
        result = calculate_final_score(
            closed_counts=test_counts,
            fixed_mentsu=list(getattr(state, 'fixed_mentsu', [])),
            win_tile=draw_tile,
            is_tsumo=False,
            bakaze=state.bakaze,
            jikaze=state.jikaze,
            is_oya=getattr(state, 'is_oya', False),
            is_riichi=False,
            dora_indicators=dora_indicators,
        )
        if result and result.get("ron_score", 0) > max_score:
            max_score = result["ron_score"]
    return max_score


# ------------------------------------------
# 🗣️ B君（鳴きモデル）の呼び出し関数
# ------------------------------------------
def hybrid_naki_decision_v5(state, discarded_tile, who_discarded, naki_model):
    # 🚨 守備の絶対ルール：誰かからリーチがかかっていれば絶対に鳴かない
    if any(state.riichi_declared.values()):
        return False, 0.0

    # まず物理的に鳴けるかチェック
    if not state.can_naki(discarded_tile, who_discarded):
        return False, 0.0

    # モデルの入力チャンネル数でv1/v2を自動判定
    # v1: 26ch、v2: 28ch(旧) or 34ch(新33ch基底+1)
    in_channels = getattr(naki_model.conv_in, 'in_channels', 26)
    if in_channels >= 28:
        tensor_data_np = state.to_tensor_for_naki_v2(discarded_tile)
    else:
        tensor_data_np = state.to_tensor_for_naki(discarded_tile)
    tensor = torch.tensor(tensor_data_np, dtype=torch.float32).unsqueeze(0)

    naki_model.eval()
    with torch.no_grad():
        out_naki = naki_model(tensor)
        probs = F.softmax(out_naki, dim=1)[0].numpy()

    naki_prob = probs[1] # 1番目のインデックスが「鳴く(1)」の確率

    # 💡 鳴き確率が40%を超えたら「鳴く！」と判断（閾値緩和: 50%→40%）
    if naki_prob > 0.40:
        return True, naki_prob
    else:
        return False, naki_prob


def generate_chi_patterns(state, discard_tile):
    """
    上家の捨て牌 discard_tile に対して、自家がチーできるパターンを列挙する。
    返り値: [[consumed1, consumed2], ...] — 手牌から使う2牌のリスト
    """
    if discard_tile >= 27:
        return []

    patterns = []
    suit = discard_tile // 9
    num = discard_tile % 9

    # パターンA: [n-2, n-1, n(捨)] → consumed: [n-2, n-1]
    if num >= 2:
        a, b = discard_tile - 2, discard_tile - 1
        if a // 9 == suit and state.hand[a] >= 1 and state.hand[b] >= 1:
            patterns.append([a, b])

    # パターンB: [n-1, n(捨), n+1] → consumed: [n-1, n+1]
    if 1 <= num <= 7:
        a, b = discard_tile - 1, discard_tile + 1
        if a // 9 == suit and b // 9 == suit and state.hand[a] >= 1 and state.hand[b] >= 1:
            patterns.append([a, b])

    # パターンC: [n(捨), n+1, n+2] → consumed: [n+1, n+2]
    if num <= 6:
        a, b = discard_tile + 1, discard_tile + 2
        if a // 9 == suit and b // 9 == suit and state.hand[a] >= 1 and state.hand[b] >= 1:
            patterns.append([a, b])

    return patterns


def decide_naki_action(state, ai_model, discard_tile, who_discarded):
    """
    鳴き（ポン/チー）を行うかスキップするかを判定し、最良の行動を返す。

    返り値: {"best": {"action": "skip"}} または
            {"best": {"action": "naki", "naki_type": "pon"|"chi",
                      "consumed_tiles": [tile, tile]}}
    """
    import copy as _copy

    current_shanten = calculate_shanten_unified(state, state.hand)
    best_result = {"action": "skip"}
    best_improvement = -1
    best_temp_state = None
    best_temp_h = None

    # --- ポン判定 ---
    # 手牌に3枚ある（暗刻）場合はポンしない: 暗刻はそのまま持つ方が強い（暗槓は別途判定）
    if state.hand[discard_tile] == 2:
        # ポン後の仮手牌でシャンテンを計算
        temp_state = _copy.copy(state)
        temp_h = state.hand.copy()
        temp_h[discard_tile] -= 2
        # ポン後は副露数が1増える
        temp_fixed = list(getattr(state, 'fixed_mentsu', [])) + [[discard_tile, discard_tile, discard_tile]]
        temp_state.hand = temp_h
        temp_state.fixed_mentsu = temp_fixed
        new_shanten = calculate_shanten_unified(temp_state, temp_h)
        improvement = current_shanten - new_shanten
        if improvement > best_improvement:
            best_improvement = improvement
            best_result = {
                "action": "naki",
                "naki_type": "pon",
                "consumed_tiles": [discard_tile, discard_tile],
            }
            best_temp_state = temp_state
            best_temp_h = temp_h

    # --- チー判定（上家=who_discarded==3 のみ） ---
    if who_discarded == 3:
        chi_patterns = generate_chi_patterns(state, discard_tile)
        for consumed in chi_patterns:
            temp_state = _copy.copy(state)
            temp_h = state.hand.copy()
            for t in consumed:
                temp_h[t] -= 1
            meld = sorted(consumed + [discard_tile])
            temp_fixed = list(getattr(state, 'fixed_mentsu', [])) + [meld]
            temp_state.hand = temp_h
            temp_state.fixed_mentsu = temp_fixed
            new_shanten = calculate_shanten_unified(temp_state, temp_h)
            improvement = current_shanten - new_shanten
            if improvement > best_improvement:
                best_improvement = improvement
                best_result = {
                    "action": "naki",
                    "naki_type": "chi",
                    "consumed_tiles": consumed,
                }
                best_temp_state = temp_state
                best_temp_h = temp_h

    # シャンテンが悪化する場合はスキップ
    if best_improvement < 0:
        return {"best": {"action": "skip"}}

    naki_type = best_result.get("naki_type", "")
    after_melds = list(getattr(best_temp_state, 'fixed_mentsu', []))

    # 副露後の最良打牌を先に計算し、打牌後手牌で役・点数判定する
    # （打牌前手牌での判定より正確: 端牌を切れば喰いタン成立など）
    best_discard_after = _find_best_discard_after_naki(best_temp_state, best_temp_h)
    if best_discard_after >= 0:
        post_discard_h = best_temp_h.copy()
        post_discard_h[best_discard_after] -= 1
        post_discard_state = _copy.copy(best_temp_state)
        post_discard_state.hand = post_discard_h
    else:
        post_discard_h = best_temp_h
        post_discard_state = best_temp_state

    after_shanten_post = calculate_shanten_unified(post_discard_state, post_discard_h)

    # 役の見通しチェック:
    # ① improvement==0 かつ テンパイ前（従来通り）
    # ② improvement>0 かつ 副露後もテンパイでない（新規: 役なし改善を排除）
    need_yaku_check = (
        (best_improvement == 0 and current_shanten > 0)
        or (best_improvement > 0 and after_shanten_post > 0)
    )
    if need_yaku_check:
        allowed = (
            _check_yakuhai_pon(state, discard_tile, naki_type)                           # ① 役牌ポン
            or (_check_dora_pon(state, discard_tile, naki_type) and current_shanten <= 3) # ② ドラポン（3シャンテン以内）
            or _check_tanyao_possible(post_discard_h, after_melds)                       # ③ 喰いタン（打牌後で判定）
            or _check_one_suit(post_discard_h, after_melds)                              # ④ ホンイツ/清一色方向（打牌後で判定）
        )
        if not allowed:
            # 役の見通しが①〜④のいずれもない場合は常にスキップ
            # （有効牌フォールバックは役なし端牌ポンを許してしまうため削除）
            return {"best": {"action": "skip"}}

    # 【Change 1】副露後打牌の安全性チェック（リーチ者がいる場合のみ）
    # 役牌ポンは役確定のため、リーチ対策でも基本的にスキップしない
    is_yakuhai_pon_c1 = _check_yakuhai_pon(state, discard_tile, naki_type)
    if not is_yakuhai_pon_c1:
        if any(state.riichi_declared.get(p, False) for p in [1, 2, 3]):
            if best_discard_after >= 0:
                risk = calculate_simple_discard_risk(state, best_discard_after)
                if risk > 1.5:
                    return {"best": {"action": "skip"}}

    # 【Change 3】点数フィルタ: 副露後テンパイで最大ロン点数が低すぎなら見送り
    # 役牌ポンは対象外（役確定のため点数が低くても鳴く価値あり）
    is_yakuhai_pon = _check_yakuhai_pon(state, discard_tile, naki_type)
    if not is_yakuhai_pon:
        if after_shanten_post == 0:
            # 打牌後手牌でスコア推定（より正確）
            max_score = _estimate_max_tenpai_score(post_discard_state, post_discard_h)
            # 点数フィルタ: 役なし鳴きを排除（閾値緩和: 3000→2000）
            if max_score < 2000:
                return {"best": {"action": "skip"}}
            # 【Change 4】メンゼン1シャンテン保護: 門前から鳴いてテンパイする局面
            # リーチ宣言権とイーペー・裏ドラ期待値を失うコストが大きいため、
            # 役・打点が確保できない場合はメンゼン維持を優先する
            if (current_shanten == 1
                    and len(getattr(state, 'fixed_mentsu', [])) == 0):
                has_dora = _check_dora_pon(state, discard_tile, naki_type)
                has_tanyao = _check_tanyao_possible(post_discard_h, after_melds)
                has_one_suit = _check_one_suit(post_discard_h, after_melds)
                if not (has_dora or has_tanyao or has_one_suit) and max_score < 3000:
                    return {"best": {"action": "skip"}}

    return {"best": best_result}