import copy
import random
import math
from dataclasses import dataclass, field

import hybrid_inference
from mahjong_engine import MahjongStateV5, tile_names
from mahjong_engine import calculate_final_score


@dataclass
class GlobalRoundState:
    hands: list
    discards: dict
    riichi_declared: dict
    fixed_mentsu: list
    scores: list
    wall: list
    dora_indicators: list
    dealer_pid: int = 0
    bakaze: int = 0
    kyoku: int = 1
    honba: int = 0
    junme: int = 1
    turn_pid: int = 0
    is_orasu: bool = False
    riichi_sticks: int = 0
    last_draw: list = None
    riichi_junme: list = None
    game_over: bool = False
    # 赤ドラ管理: aka_hands[pid]=プレイヤーの赤ドラ枚数, aka_pool=壁牌内の残り赤ドラ
    aka_hands: list = None   # [0,0,0,0]
    aka_pool: dict = None    # {4:1, 13:1, 22:1} 5m/5p/5s
    # 一発フラグ: リーチ宣言後の最初のツモ/ロン機会まで有効
    ippatsu_eligible: dict = None  # {0:False, 1:False, 2:False, 3:False}


def tiles_to_string(hand34):
    """
    手牌34種表現を牌名の文字列にする
    """
    tiles = []
    for i in range(34):
        for _ in range(hand34[i]):
            tiles.append(tile_names[i])
    return " ".join(tiles)


def build_wall_34(seed=None):
    """
    34種表現の山を作る
    0..33 を各4枚
    """
    if seed is not None:
        random.seed(seed)

    wall = []
    for t in range(34):
        wall.extend([t] * 4)

    random.shuffle(wall)
    return wall


def should_declare_riichi(gs: GlobalRoundState, pid: int, discard_tile: int, debug_rows):

    # 既にリーチ
    if gs.riichi_declared[pid]:
        return False

    # 鳴き手は不可
    if len(gs.fixed_mentsu[pid]) > 0:
        return False

    # 点棒不足
    if gs.scores[pid] < 1000:
        return False

    # テンパイしていない
    if not is_tenpai_after_discard(gs, pid, discard_tile):
        return False

    # 終盤はリーチしない（17巡目以降は残り牌が少なくリーチの旨みが薄い）
    if gs.junme >= 17:
        return False

    # ── 待ち牌 & 受け入れ枚数 ────────────────────────────────
    temp_state = build_local_state_for_player(gs, pid)
    temp_hand  = gs.hands[pid].copy()
    temp_hand[discard_tile] -= 1

    wait_tiles    = hybrid_inference.get_waiting_tiles_with_open_hand(
        temp_state, temp_hand, is_riichi=True
    )
    riichi_ukeire = hybrid_inference.count_remaining_tiles_for_list(
        temp_state, temp_hand, wait_tiles
    )

    # ── ダマテン手牌の平均和了点 ─────────────────────────────
    visible = hybrid_inference.build_visible_tiles34(temp_state)
    is_oya  = (pid == gs.dealer_pid)

    dama_scores = []
    for wt in wait_tiles:
        if (4 - visible[wt]) <= 0:
            continue
        t14 = temp_hand.copy()
        t14[wt] += 1
        res = calculate_final_score(
            t14, gs.fixed_mentsu[pid], wt,
            is_tsumo=False,
            bakaze=temp_state.bakaze,
            jikaze=temp_state.jikaze,
            is_oya=is_oya,
            is_riichi=False,
        )
        if res and res.get('score', 0) > 0:
            dama_scores.append(res['score'])

    typical_dama  = sum(dama_scores) / len(dama_scores) if dama_scores else 0
    has_dama_yaku = typical_dama > 0

    # ── 戦況コンテキスト ──────────────────────────────────────
    ctx    = hybrid_inference.compute_game_situation(temp_state)
    target = ctx['target_score_needed']

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIORITY 1: 絶対放銃禁止ガード
    #   1位のとき、リーチで放銃すると着順が動く危険な状況
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ctx['rank'] == 0 and ctx['threatened_by_cheap']:
        # 安手で逆転されうる: 2着がリーチ中なら完全封印
        if ctx['second_place_riichi']:
            return False
        # 2着無リーチでも、親オーラス素点モード以外はダマで安全策
        if not ctx['dealer_orasu_aggressive']:
            return False

    # 1位 + 満貫で逆転されうる + 2着がリーチ中 → 危険
    if ctx['rank'] == 0 and ctx['threatened_by_mangan'] and ctx['second_place_riichi']:
        return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIORITY 2: 親オーラス素点積み上げ
    #   1位・親・オーラス・満貫圏外 → 積極リーチで素点を稼ぐ
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ctx['dealer_orasu_aggressive']:
        if riichi_ukeire >= 2:
            return True  # ダマ打点チェックをスキップして積極リーチ

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIORITY 3: 待ち枚数チェック (状況に応じて閾値を変動)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ctx['need_mangan_plus'] and ctx['urgency'] >= 0.75:
        # 絶望的な逆転待ち: 待ちが薄くてもウラドラに賭ける
        min_ukeire = 2
    elif ctx['is_dealer'] and not ctx['is_orasu']:
        # 親非オーラス: 連荘圧力のため閾値を緩和
        min_ukeire = 3
    else:
        min_ukeire = 4

    if riichi_ukeire < min_ukeire:
        return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIORITY 4: 1着が現実的に届かない → 1位のみ放銃回避
    #   2位以下はリーチで着順争いに参加する方が得
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not ctx['rank_improvement_possible'] and ctx['rank'] == 0 and has_dama_yaku:
        return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIORITY 5: ダマテン打点 vs 目標点 の比較
    #   手役があり、ダマ打点で目標達成できるなら原則ダマ
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if has_dama_yaku and target > 0 and typical_dama >= target:
        # 例外A: urgency高 + 満貫以上必要 → ウラドラに賭ける
        if ctx['need_mangan_plus'] and ctx['urgency'] >= 0.75:
            return True
        # 例外B: 終盤 → 残り牌少なく変化のメリット薄い + 圧力目的
        if ctx['junme_stage'] == 'late':
            return True
        # 原則ダマ (early / mid)
        return False

    return True


def check_tsumo_agari(gs: GlobalRoundState, pid: int, hand_counts_14, drawn_tile: int,
                      is_ippatsu: bool = False):
    """
    pid が drawn_tile をツモった直後の14枚手で和了しているか判定
    hand_counts_14 は draw 後の手牌をそのまま渡す
    """
    fixed_mentsu = gs.fixed_mentsu[pid]
    aka_han = gs.aka_hands[pid] if gs.aka_hands else 0

    result = calculate_final_score(
        closed_counts=hand_counts_14.copy(),
        fixed_mentsu=fixed_mentsu,
        win_tile=drawn_tile,
        is_tsumo=True,
        bakaze=gs.bakaze,
        jikaze=get_player_wind(gs.dealer_pid, pid),
        is_oya=(pid == gs.dealer_pid),
        is_riichi=gs.riichi_declared[pid],
        dora_indicators=gs.dora_indicators,
        aka_dora_count=aka_han,
        is_ippatsu=is_ippatsu,
    )

    if result.get("score", 0) > 0:
        print("[TSUMO DEBUG] pid=", pid, "drawn=", drawn_tile,
              "han=", result.get("han", 0),
              "fu=", result.get("fu", 0),
              "score=", result.get("score", 0))
        return True, result

    return False, result


def check_ron_agari(gs: GlobalRoundState, pid: int, ron_tile: int,
                    is_ippatsu: bool = False):
    """
    pid が他家の捨て牌 ron_tile でロン和了できるか判定する
    """
    test_hand = gs.hands[pid].copy()
    test_hand[ron_tile] += 1

    fixed_mentsu = gs.fixed_mentsu[pid]
    aka_han = gs.aka_hands[pid] if gs.aka_hands else 0

    result = calculate_final_score(
        closed_counts=test_hand,
        fixed_mentsu=fixed_mentsu,
        win_tile=ron_tile,
        is_tsumo=False,
        bakaze=gs.bakaze,
        jikaze=get_player_wind(gs.dealer_pid, pid),
        is_oya=(pid == gs.dealer_pid),
        is_riichi=gs.riichi_declared[pid],
        dora_indicators=gs.dora_indicators,
        aka_dora_count=aka_han,
        is_ippatsu=is_ippatsu,
    )

    if result.get("score", 0) > 0:
        print("[RON DEBUG] pid=", pid, "tile=", ron_tile,
              "han=", result.get("han", 0),
              "fu=", result.get("fu", 0),
              "score=", result.get("score", 0))
        return True, result

    return False, result


def apply_ron_score(gs, winner_pid, loser_pid, agari_result):
    """
    ロン和了の点数移動
    - 放銃者が全額支払う
    - 本場は放銃者が honba * 300 を追加で払う
    - リーチ棒は和了者が総取り
    """
    detail = agari_result.get("score_detail", None)

    if detail is not None:
        ron_score = detail["ron_score"]
        base_score = detail["base_points"]
    else:
        ron_score = int(agari_result.get("score", 0))
        base_score = ron_score

    honba_bonus = gs.honba * 300
    riichi_bonus = gs.riichi_sticks * 1000

    payments = {0: 0, 1: 0, 2: 0, 3: 0}
    payments[loser_pid] -= (ron_score + honba_bonus)
    payments[winner_pid] += (ron_score + honba_bonus + riichi_bonus)

    for pid in range(4):
        gs.scores[pid] += payments[pid]

    taken_riichi = gs.riichi_sticks
    gs.riichi_sticks = 0

    return {
        "winner": winner_pid,
        "loser": loser_pid,
        "ron_or_tsumo": "ron",
        "payments": payments,
        "riichi_sticks_taken": taken_riichi,
        "base_score": base_score,
        "han": agari_result.get("han", 0),
        "fu": agari_result.get("fu", 0),
        "honba": gs.honba,
    }


def check_any_ron(gs, discarder_pid, discard_tile):
    """
    discarder_pid の discard_tile に対して、他家がロンできるか調べる。
    最小版:
    - 順番は discarder の次家から
    - 最初に見つかった1人だけを採用
    戻り値:
        (winner_pid or None, agari_result or None)
    """
    for offset in range(1, 4):
        pid = (discarder_pid + offset) % 4

        # 一発フラグを渡す
        ippatsu = gs.ippatsu_eligible[pid] if gs.ippatsu_eligible else False

        # すでにリーチしていてもロンは可能
        is_agari, agari_result = check_ron_agari(gs, pid, discard_tile, is_ippatsu=ippatsu)
        if is_agari:
            # ロン成立で一発確定 → フラグ消費
            if ippatsu:
                gs.ippatsu_eligible[pid] = False
            return pid, agari_result

    return None, None


def is_tenpai_after_discard(gs: GlobalRoundState, pid: int, discard_tile: int):
    """
    その牌を切ったあとテンパイかどうか
    """
    from hybrid_inference import calculate_shanten_unified

    temp_state = build_local_state_for_player(gs, pid)
    if temp_state.hand[discard_tile] <= 0:
        return False

    temp_state.hand[discard_tile] -= 1
    shanten = calculate_shanten_unified(temp_state, temp_state.hand)
    return shanten == 0


def empty_hand34():
    return [0] * 34


def apply_tsumo_score(gs, winner_pid, agari_result):
    """
    ツモ和了の点数移動
    - 親ツモ / 子ツモを正確に分ける
    - 本場は各相手から100点ずつ
    - リーチ棒は和了者が総取り
    """
    detail = agari_result.get("score_detail", None)
    if detail is None:
        raise ValueError("tsumo agari_result に score_detail がありません")

    is_dealer = (winner_pid == gs.dealer_pid)

    payments = {0: 0, 1: 0, 2: 0, 3: 0}

    if is_dealer:
        each_pay = detail["tsumo_oya_payment"]
        for pid in range(4):
            if pid == winner_pid:
                continue
            pay = each_pay + gs.honba * 100
            payments[pid] -= pay
            payments[winner_pid] += pay
    else:
        ko_pay = detail["tsumo_ko_payment_child"]
        oya_pay = detail["tsumo_ko_payment"]

        for pid in range(4):
            if pid == winner_pid:
                continue

            if pid == gs.dealer_pid:
                pay = oya_pay + gs.honba * 100
            else:
                pay = ko_pay + gs.honba * 100

            payments[pid] -= pay
            payments[winner_pid] += pay

    riichi_bonus = gs.riichi_sticks * 1000
    if riichi_bonus > 0:
        payments[winner_pid] += riichi_bonus

    for pid in range(4):
        gs.scores[pid] += payments[pid]

    taken_riichi = gs.riichi_sticks
    gs.riichi_sticks = 0

    return {
        "winner": winner_pid,
        "is_dealer": is_dealer,
        "ron_or_tsumo": "tsumo",
        "payments": payments,
        "riichi_sticks_taken": taken_riichi,
        "base_score": detail["base_points"],
        "han": agari_result.get("han", 0),
        "fu": agari_result.get("fu", 0),
        "honba": gs.honba,
    }


def format_score_diff(score_movements):
    payments = score_movements["payments"]
    parts = []
    for pid in range(4):
        delta = payments[pid]
        sign = "+" if delta >= 0 else ""
        parts.append(f"P{pid}:{sign}{delta}")
    return "  ".join(parts)


def deal_initial_hands(seed=None, dealer_pid=0):
    """
    4人に13枚ずつ配る
    簡易版なので dead wall 等はまだ厳密に分けない
    """
    wall = build_wall_34(seed=seed)

    hands = [empty_hand34() for _ in range(4)]
    for _ in range(13):
        for pid in range(4):
            t = wall.pop()
            hands[pid][t] += 1

    # 簡易版ドラ表示牌
    dora_indicators = [wall[-1]]

    # 赤ドラ初期配分: 5m(4)/5p(13)/5s(22) 各1枚を配牌時にランダム割り当て
    aka_hands = [0, 0, 0, 0]
    aka_pool = {4: 1, 13: 1, 22: 1}
    for aka_tile in [4, 13, 22]:
        total_in_hands = sum(hands[p][aka_tile] for p in range(4))
        if total_in_hands > 0:
            # P(赤が配牌内に入っている) = total_in_hands / 4
            if random.random() < total_in_hands / 4.0:
                # 持っているプレイヤーの中からランダムに選択（枚数で重み付け）
                holders = [p for p in range(4) if hands[p][aka_tile] > 0]
                weights = [hands[p][aka_tile] for p in holders]
                chosen = random.choices(holders, weights=weights)[0]
                aka_hands[chosen] += 1
                aka_pool[aka_tile] = 0  # 壁牌内の赤はなし

    gs = GlobalRoundState(
        hands=hands,
        discards={0: [], 1: [], 2: [], 3: []},
        riichi_declared={0: False, 1: False, 2: False, 3: False},
        fixed_mentsu=[[], [], [], []],
        scores=[25000, 25000, 25000, 25000],
        wall=wall,
        dora_indicators=dora_indicators,
        dealer_pid=dealer_pid,
        bakaze=0,
        kyoku=1,
        honba=0,
        junme=1,
        turn_pid=dealer_pid,
        is_orasu=False,
        riichi_sticks=0,
        last_draw=[None] * 4,
        riichi_junme=[None] * 4,
        game_over=False,
        aka_hands=aka_hands,
        aka_pool=aka_pool,
        ippatsu_eligible={0: False, 1: False, 2: False, 3: False},
    )
    return gs


def get_player_wind(dealer_pid, pid):
    """
    0=東,1=南,2=西,3=北
    """
    return (pid - dealer_pid) % 4


def draw_tile(gs: GlobalRoundState, pid: int):
    """
    山から1枚ツモる
    """
    if len(gs.wall) == 0:
        return None

    t = gs.wall.pop()
    gs.hands[pid][t] += 1

    # 赤ドラ判定: 5m/5p/5s をツモった場合、壁牌に残っている赤ドラかチェック
    if gs.aka_pool and t in gs.aka_pool and gs.aka_pool[t] > 0:
        remaining = gs.wall.count(t)  # ツモ後の壁内残り枚数
        total_remaining_before = remaining + 1  # ツモ前の壁内残り枚数
        if total_remaining_before > 0 and random.random() < gs.aka_pool[t] / total_remaining_before:
            gs.aka_hands[pid] += 1
            gs.aka_pool[t] -= 1

    return t


def apply_discard(gs: GlobalRoundState, pid: int, tile_idx: int):
    """
    1枚捨てる
    """
    if gs.hands[pid][tile_idx] <= 0:
        raise ValueError(f"Player {pid} cannot discard tile {tile_idx}")

    gs.hands[pid][tile_idx] -= 1
    gs.discards[pid].append(tile_idx)


def build_local_state_for_player(gs: GlobalRoundState, pid: int):
    """
    グローバル状態から、そのプレイヤー視点の MahjongStateV5 を作る

    訓練データ (dataset_extractor.py) と同じ相対POV順で格納する:
      pov 0 = 自分 (pid)
      pov 1 = 下家 (pid+1)%4
      pov 2 = 対面 (pid+2)%4
      pov 3 = 上家 (pid+3)%4
    """
    state = MahjongStateV5()

    state.hand = gs.hands[pid].copy()

    # 相対POVでマッピング
    for pov in range(4):
        actual = (pid + pov) % 4
        state.discards[pov] = list(gs.discards[actual])
        state.riichi_declared[pov] = gs.riichi_declared[actual]
        if pov > 0:
            state.melds[pov] = [list(m) for m in gs.fixed_mentsu[actual]]

    state.dora_indicators = list(gs.dora_indicators)

    # 自分の副露だけ直接持つ
    state.fixed_mentsu = copy.deepcopy(gs.fixed_mentsu[pid])

    state.forbidden_discards = []
    state.bakaze = gs.bakaze
    state.jikaze = get_player_wind(gs.dealer_pid, pid)
    state.junme = gs.junme

    state.is_oya = (pid == gs.dealer_pid)
    state.score_situation = None
    state.is_orasu = gs.is_orasu
    state.placement_pressure = "neutral"

    # 点数・識別子も相対POV順
    state.scores = [gs.scores[(pid + pov) % 4] for pov in range(4)]
    state.my_pid = 0  # 常にPOV 0 = 自分
    state.dealer_pid = (gs.dealer_pid - pid + 4) % 4  # 相対POVでの親番号
    state.rival_pids = None

    # 他家の副露数は守備推定に使う（相対POV順）
    state.enemy_open_counts = {
        pov: len(gs.fixed_mentsu[(pid + pov) % 4]) for pov in range(4)
    }

    state.honba = gs.honba
    state.kyotaku = gs.riichi_sticks

    return state


def choose_discard_with_ai(gs, pid, ai_model, top_k=5, extra_forbidden=None):
    if gs.riichi_declared[pid]:
        drawn_tile = gs.last_draw[pid]
        if drawn_tile is None:
            raise ValueError(f"Player {pid} riichi state but last_draw is None")

        debug_rows = [{
            "tile_idx": drawn_tile,
            "final_score": 0.0,
            "risk_value": 0.0,
            "after_shanten": None,
            "ukeire_count": None,
            "shape_bonus": 0.0,
            "route_bonus": 0.0,
            "shape_reasons": ["riichi_after_tsumogiri_only"],
            "route_reasons": {"normal": ["riichi_after_tsumogiri_only"]},
            "mode": "riichi_locked",
        }]
        return drawn_tile, debug_rows

    local_state = build_local_state_for_player(gs, pid)

    # チー後の食い替え禁止牌を反映
    if extra_forbidden:
        local_state.forbidden_discards = list(extra_forbidden)

    best_discard, final_probs, debug_rows = hybrid_inference.hybrid_ai_decision_v6_rerank_debug(
        local_state,
        ai_model,
        top_k=top_k
    )

    return best_discard, debug_rows


def apply_naki_global(gs, naki_pid, naki_type, consumed_tiles, called_tile):
    """
    鳴きをグローバル状態に適用する。
    consumed_tiles: 鳴いたプレイヤーの手牌から取り除く牌のリスト
    called_tile:    相手が捨てた牌（meldに加える）
    鳴き後の手牌は 13 - 2 = 11 枚になる（その後打牌で10枚）
    """
    for t in consumed_tiles:
        if gs.hands[naki_pid][t] <= 0:
            raise ValueError(f"Player {naki_pid} does not have tile {t} to consume for {naki_type}")
        gs.hands[naki_pid][t] -= 1

    if naki_type == "pon":
        gs.fixed_mentsu[naki_pid].append([called_tile, called_tile, called_tile])
    elif naki_type == "chi":
        meld = sorted(consumed_tiles + [called_tile])
        gs.fixed_mentsu[naki_pid].append(meld)


def find_naki_action(gs, discarder_pid, discard_tile, naki_model, ai_model):
    """
    捨て牌に対して他家がポン/チーするかを判定する。
    優先順位: ポン > チー（同じならターン順で近い方）

    判定方法:
      - naki_model が提供されている場合は MahjongResNet_Naki で一次フィルタ
      - 最終判定は打牌AIモデルで鳴き後の局面を評価する decide_naki_action() を使用

    戻り値: (naki_pid, naki_type, consumed_tiles, forbidden_discards) または None
    """
    if ai_model is None:
        return None  # 鳴き評価にはai_modelが必須

    # ポン判定（discarder以外全員、ターン順）
    for offset in range(1, 4):
        pid = (discarder_pid + offset) % 4
        if gs.riichi_declared[pid]:
            continue
        if gs.hands[pid][discard_tile] < 2:
            continue

        local_state = build_local_state_for_player(gs, pid)
        relative_who = (discarder_pid - pid + 4) % 4

        # 戦況コンテキスト（速度優先判定・NN閾値調整に使用）
        game_ctx = hybrid_inference.compute_game_situation(local_state)
        _is_dealer  = game_ctx.get('is_dealer', False)
        _is_orasu   = game_ctx.get('is_orasu', False)
        _rank       = game_ctx.get('rank', 2)
        _opp_dealer = getattr(local_state, 'dealer_pid', 0) != 0
        speed_priority_nn = _is_dealer or (_is_orasu and _rank == 0) or _opp_dealer

        # MLモデルによる一次フィルタ（利用可能な場合）
        # speed_priority 局面では閾値を緩和（0.40→0.25）、その他は 0.35 に緩和
        naki_prob = 0.0
        if naki_model is not None:
            should_naki, naki_prob = hybrid_inference.hybrid_naki_decision_v5(
                local_state, discard_tile, relative_who, naki_model
            )
            if not should_naki:
                if naki_prob == 0.0:
                    continue  # 絶対ブロック（リーチ中 or 鳴き不可）
                naki_threshold = 0.25 if speed_priority_nn else 0.35
                if naki_prob <= naki_threshold:
                    continue

        # 打牌AIで鳴き後の局面を評価してスキップより有利か確認
        result = hybrid_inference.decide_naki_action(
            local_state, ai_model, discard_tile, relative_who,
            naki_prob=naki_prob, game_ctx=game_ctx,
        )
        best = result.get("best") if result else None
        if best and best["action"] != "skip" and best.get("consumed_tiles"):
            return pid, best["naki_type"], best["consumed_tiles"], []

    # チー判定（上家のみ = discarderの次のプレイヤー）
    if discard_tile < 27:  # 字牌はチー不可
        chi_pid = (discarder_pid + 1) % 4
        if not gs.riichi_declared[chi_pid]:
            local_state = build_local_state_for_player(gs, chi_pid)
            chi_patterns = hybrid_inference.generate_chi_patterns(local_state, discard_tile)

            if chi_patterns:
                game_ctx = hybrid_inference.compute_game_situation(local_state)
                _is_dealer  = game_ctx.get('is_dealer', False)
                _is_orasu   = game_ctx.get('is_orasu', False)
                _rank       = game_ctx.get('rank', 2)
                _opp_dealer = getattr(local_state, 'dealer_pid', 0) != 0
                speed_priority_nn = _is_dealer or (_is_orasu and _rank == 0) or _opp_dealer

                # MLモデルによる一次フィルタ（利用可能な場合）
                naki_prob = 0.0
                if naki_model is not None:
                    should_naki, naki_prob = hybrid_inference.hybrid_naki_decision_v5(
                        local_state, discard_tile, 3, naki_model
                    )
                    if not should_naki:
                        if naki_prob == 0.0:
                            return None  # 絶対ブロック
                        naki_threshold = 0.25 if speed_priority_nn else 0.35
                        if naki_prob <= naki_threshold:
                            return None

                # 打牌AIでチー後の局面を評価
                result = hybrid_inference.decide_naki_action(
                    local_state, ai_model, discard_tile, 3,
                    naki_prob=naki_prob, game_ctx=game_ctx,
                )
                best = result.get("best") if result else None
                if best and best["action"] != "skip" and best.get("consumed_tiles"):
                    forbidden = [discard_tile]  # 食い替え禁止
                    return chi_pid, best["naki_type"], best["consumed_tiles"], forbidden

    return None


def hand_count(hand34):
    return sum(hand34)


def hand_to_tiles(hand34):
    tiles = []
    for i in range(34):
        tiles.extend([i] * hand34[i])
    return tiles


def ceil100(x):
    return int(math.ceil(x / 100.0) * 100)


def calc_base_points_from_han_fu(han, fu):
    """
    符・翻から基本点を計算
    """
    if han >= 13:
        return 8000
    elif han >= 11:
        return 6000
    elif han >= 8:
        return 4000
    elif han >= 6:
        return 3000
    elif han >= 5:
        return 2000
    else:
        base_points = fu * (2 ** (han + 2))
        return min(base_points, 2000)


def calc_ron_score_from_han_fu(han, fu, is_oya):
    base_points = calc_base_points_from_han_fu(han, fu)
    return ceil100(base_points * (6 if is_oya else 4))


def calc_tsumo_payment_from_han_fu(han, fu, is_oya):
    """
    戻り値:
      親ツモ: {"ko": 支払額, "oya": None}
      子ツモ: {"ko": 子の支払額, "oya": 親の支払額}
    """
    base_points = calc_base_points_from_han_fu(han, fu)

    if is_oya:
        return {
            "ko": ceil100(base_points * 2),
            "oya": None,
        }
    else:
        return {
            "ko": ceil100(base_points * 1),
            "oya": ceil100(base_points * 2),
        }


def is_tenpai_hand(gs: GlobalRoundState, pid: int):
    from hybrid_inference import calculate_shanten_unified

    hand = gs.hands[pid].copy()
    local_state = build_local_state_for_player(gs, pid)
    shanten = calculate_shanten_unified(local_state, hand)
    return shanten == 0


def reset_round_for_next_hand(gs: GlobalRoundState):
    """
    次局の配牌を作る。
    点棒・場風・局・本場・親は現在の gs の値を引き継ぐ。
    """
    new_gs = deal_initial_hands(seed=None, dealer_pid=gs.dealer_pid)

    # 局情報を引き継ぐ
    new_gs.scores = gs.scores.copy()
    new_gs.bakaze = gs.bakaze
    new_gs.kyoku = gs.kyoku
    new_gs.honba = gs.honba
    new_gs.is_orasu = gs.is_orasu
    new_gs.riichi_sticks = gs.riichi_sticks
    new_gs.dealer_pid = gs.dealer_pid
    new_gs.game_over = gs.game_over

    # 念のため明示
    new_gs.turn_pid = gs.dealer_pid
    new_gs.junme = 1
    new_gs.last_draw = [None] * 4

    # もし将来使うなら
    if hasattr(new_gs, "riichi_junme"):
        new_gs.riichi_junme = [None] * 4

    return new_gs


def advance_round(gs: GlobalRoundState, dealer_continues: bool, is_ryuukyoku: bool = False):
    """
    親連荘・親流れ・本場処理
    """

    if dealer_continues:
        gs.honba += 1
    else:
        if is_ryuukyoku:
            gs.honba += 1
        else:
            gs.honba = 0

        gs.dealer_pid = (gs.dealer_pid + 1) % 4
        gs.kyoku += 1

        if gs.kyoku > 4:
            gs.kyoku = 1
            gs.bakaze += 1

    # 南4終了後は簡易終局
    if gs.bakaze >= 2:
        gs.game_over = True
        return gs

    gs.is_orasu = (gs.bakaze == 1 and gs.kyoku == 4)

    gs = reset_round_for_next_hand(gs)
    return gs


def run_minimal_selfplay(ai_model, seed=0, max_turns=60, top_k=5, verbose=True, naki_model=None):
    """
    自己対局ループ
    - 4人で順番にツモ打牌
    - ツモ和了/ロン和了/リーチ対応
    - naki_model を渡すとポン/チーも処理する
    """
    gs = deal_initial_hands(seed=seed, dealer_pid=0)
    logs = []

    if verbose:
        print("=== Minimal Selfplay Start ===")
        print("dora indicator =", tile_names[gs.dora_indicators[0]])
        print()

    turn_index = 0

    while len(gs.wall) > 0 and turn_index < max_turns:
        pid = gs.turn_pid

        hand_before_draw = gs.hands[pid].copy()

        # ツモ
        drawn = draw_tile(gs, pid)
        if drawn is None:
            break

        gs.last_draw[pid] = drawn

        hand_after_draw = gs.hands[pid].copy()

        # ツモ和了判定（draw直後の14枚手を渡す）
        # 一発フラグ: リーチ宣言後の最初のツモ機会
        ippatsu_tsumo = gs.ippatsu_eligible[pid] if gs.ippatsu_eligible else False
        if ippatsu_tsumo:
            gs.ippatsu_eligible[pid] = False  # ツモ機会到来で消費（win/lossに関わらず）
        is_agari, agari_result = check_tsumo_agari(gs, pid, hand_after_draw, drawn,
                                                    is_ippatsu=ippatsu_tsumo)
        if is_agari:
            score_movements = apply_tsumo_score(gs, pid, agari_result)

            log_row = {
                "turn": turn_index + 1,
                "pid": pid,
                "junme": gs.junme,
                "draw": drawn,
                "discard": None,
                "result": "tsumo",
                "agari_result": agari_result,
                "hand_after_draw": gs.hands[pid].copy(),
                "score_movements": score_movements,
                "scores_after": gs.scores.copy(),
                "riichi_sticks_after": gs.riichi_sticks,
            }
            logs.append(log_row)

            dealer_continues = (pid == gs.dealer_pid)
            gs = advance_round(gs, dealer_continues=dealer_continues, is_ryuukyoku=False)

            if verbose:
                print(f"Turn {turn_index+1:02d} | P{pid} | junme={log_row['junme']}")
                print("   hand after draw  :", tiles_to_string(log_row["hand_after_draw"]))
                print("   tsumo agari      :", tile_names[drawn])
                print("   han              :", agari_result.get("han", 0))
                print("   fu               :", agari_result.get("fu", 0))
                print("   score diff       :", format_score_diff(score_movements))
                print("   scores after     :", log_row["scores_after"])
                print("   next dealer      :", gs.dealer_pid if not gs.game_over else "END")
                print("   next bakaze/kyoku:", f"{gs.bakaze}-{gs.kyoku}" if not gs.game_over else "END")
                print("   next honba       :", gs.honba if not gs.game_over else "END")
                print()

                print("=== Hand End (tsumo) ===")
                print("winner =", pid)
                print("scores =", gs.scores)
                print("riichi sticks =", gs.riichi_sticks)
                print("next dealer =", gs.dealer_pid)
                print(f"next bakaze/kyoku = {gs.bakaze}-{gs.kyoku}")
                print("next honba =", gs.honba)
                print()

            turn_index += 1
            continue

        # 副露数を考慮した手牌枚数確認 (副露1つにつき3枚が fixed_mentsu に移動)
        num_melds = len(gs.fixed_mentsu[pid])
        expected_after_draw = 14 - 3 * num_melds
        if hand_count(gs.hands[pid]) != expected_after_draw:
            raise ValueError(
                f"Player {pid} hand count must be {expected_after_draw} after draw "
                f"(melds={num_melds}), got {hand_count(gs.hands[pid])}"
            )

        hand_after_draw = gs.hands[pid].copy()

        # AI打牌
        best_discard, debug_rows = choose_discard_with_ai(gs, pid, ai_model, top_k=top_k)

        declare_riichi = should_declare_riichi(gs, pid, best_discard, debug_rows)

        if declare_riichi:
            gs.riichi_declared[pid] = True
            gs.scores[pid] -= 1000
            gs.riichi_sticks += 1
            if gs.ippatsu_eligible is not None:
                gs.ippatsu_eligible[pid] = True  # 一発有効化

        # 打牌適用
        apply_discard(gs, pid, best_discard)

        hand_after_discard = gs.hands[pid].copy()

        # 副露数を考慮した手牌枚数確認
        expected_after_discard = 13 - 3 * num_melds
        if hand_count(gs.hands[pid]) != expected_after_discard:
            raise ValueError(
                f"Player {pid} hand count must be {expected_after_discard} after discard "
                f"(melds={num_melds}), got {hand_count(gs.hands[pid])}"
            )

        log_row = {
            "turn": turn_index + 1,
            "pid": pid,
            "junme": gs.junme,
            "draw": drawn,
            "discard": best_discard,
            "hand_before_draw": hand_before_draw.copy(),
            "hand_after_draw": hand_after_draw.copy(),
            "hand_after_discard": hand_after_discard.copy(),
            "mode": debug_rows[0].get("mode", "N/A") if debug_rows else "N/A",
            "riichi_declared": declare_riichi,
            "top3": [
                {
                    "tile": row["tile_idx"],
                    "score": round(float(row["final_score"]), 4),
                    "risk": round(float(row.get("risk_value", 0.0)), 4),
                    "after_shanten": row.get("after_shanten", None),
                    "ukeire": row.get("ukeire_count", None),
                    "shape": round(float(row.get("shape_bonus", 0.0)), 4),
                    "route": round(float(row.get("route_bonus", row.get("shape_bonus", 0.0))), 4),
                    "shape_reasons": row.get("shape_reasons", []),
                    "route_reasons": row.get("route_reasons", {}),
                }
                for row in debug_rows[:3]
            ],
        }
        logs.append(log_row)

        if verbose:
            print(f"Turn {turn_index+1:02d} | P{pid} | junme={gs.junme}")
            print("   hand before draw :", tiles_to_string(hand_before_draw))
            print("   draw             :", tile_names[drawn])
            print("   hand after draw  :", tiles_to_string(hand_after_draw))
            print("   discard          :", tile_names[best_discard])
            print("   hand after disc. :", tiles_to_string(hand_after_discard))
            print(f"   mode             : {log_row['mode']}")
            print(f"   riichi           : {log_row['riichi_declared']}")
            print("   top3 candidates  :")
            for rank, row in enumerate(log_row["top3"], start=1):
                print(
                    f"      {rank}. {tile_names[row['tile']]} "
                    f"score={row['score']:.4f} risk={row['risk']:.3f} "
                    f"shanten={row['after_shanten']} ukeire={row['ukeire']} "
                    f"shape={row['shape']:.3f} route={row['route']:.3f}"
                )
                if row["shape_reasons"]:
                    print("         shape_reasons:", row["shape_reasons"])
                if row["route_reasons"]:
                    print("         route_reasons:", row["route_reasons"])
            print()

        # ロン和了判定
        ron_winner_pid, ron_agari_result = check_any_ron(gs, pid, best_discard)
        if ron_winner_pid is not None:
            score_movements = apply_ron_score(gs, ron_winner_pid, pid, ron_agari_result)

            ron_log = {
                "turn": turn_index + 1,
                "pid": ron_winner_pid,
                "junme": gs.junme,
                "draw": None,
                "discard": best_discard,
                "result": "ron",
                "from_pid": pid,
                "agari_result": ron_agari_result,
                "win_tile": best_discard,
                "score_movements": score_movements,
                "scores_after": gs.scores.copy(),
                "riichi_sticks_after": gs.riichi_sticks,
            }
            logs.append(ron_log)

            dealer_continues = (ron_winner_pid == gs.dealer_pid)
            gs = advance_round(gs, dealer_continues=dealer_continues, is_ryuukyoku=False)

            if verbose:
                print(f"   RON by P{ron_winner_pid} on {tile_names[best_discard]}")
                print("   han              :", ron_agari_result.get("han", 0))
                print("   fu               :", ron_agari_result.get("fu", 0))
                print("   score diff       :", format_score_diff(score_movements))
                print("   scores after     :", ron_log["scores_after"])
                print("   next dealer      :", gs.dealer_pid if not gs.game_over else "END")
                print("   next bakaze/kyoku:", f"{gs.bakaze}-{gs.kyoku}" if not gs.game_over else "END")
                print("   next honba       :", gs.honba if not gs.game_over else "END")
                print()

                print("=== Hand End (ron) ===")
                print("winner =", ron_winner_pid)
                print("loser  =", pid)
                print("scores =", gs.scores)
                print("riichi sticks =", gs.riichi_sticks)
                print("next dealer =", gs.dealer_pid)
                print(f"next bakaze/kyoku = {gs.bakaze}-{gs.kyoku}")
                print("next honba =", gs.honba)
                print()

            turn_index += 1
            continue

        # ポン/チー判定（ロン成立後はスキップ済み）
        naki_result = find_naki_action(gs, pid, best_discard, naki_model, ai_model)
        if naki_result is not None:
            naki_pid, naki_type, consumed_tiles, forbidden = naki_result

            # 鳴き発生で全プレイヤーの一発を消す
            if gs.ippatsu_eligible is not None:
                for p in range(4):
                    gs.ippatsu_eligible[p] = False

            apply_naki_global(gs, naki_pid, naki_type, consumed_tiles, best_discard)

            if verbose:
                naki_label = "PON" if naki_type == "pon" else "CHI"
                meld_tiles = consumed_tiles + [best_discard]
                print(f"   {naki_label} by P{naki_pid}: {[tile_names[t] for t in meld_tiles]}")

            # 鳴いたプレイヤーが打牌（食い替え禁止牌を渡す）
            naki_discard, naki_debug = choose_discard_with_ai(
                gs, naki_pid, ai_model, top_k=top_k, extra_forbidden=forbidden
            )

            if verbose:
                print(f"   P{naki_pid} discards after {naki_type}: {tile_names[naki_discard]}")

            apply_discard(gs, naki_pid, naki_discard)

            # 鳴き後の打牌に対するロン判定
            ron_winner_pid2, ron_agari_result2 = check_any_ron(gs, naki_pid, naki_discard)
            if ron_winner_pid2 is not None:
                score_movements2 = apply_ron_score(gs, ron_winner_pid2, naki_pid, ron_agari_result2)

                ron_log2 = {
                    "turn": turn_index + 1,
                    "pid": ron_winner_pid2,
                    "junme": gs.junme,
                    "draw": None,
                    "discard": naki_discard,
                    "result": "ron",
                    "from_pid": naki_pid,
                    "agari_result": ron_agari_result2,
                    "win_tile": naki_discard,
                    "score_movements": score_movements2,
                    "scores_after": gs.scores.copy(),
                    "riichi_sticks_after": gs.riichi_sticks,
                }
                logs.append(ron_log2)

                dealer_continues2 = (ron_winner_pid2 == gs.dealer_pid)
                gs = advance_round(gs, dealer_continues=dealer_continues2, is_ryuukyoku=False)

                if verbose:
                    print(f"   RON by P{ron_winner_pid2} on {tile_names[naki_discard]}")
                    print("   scores after     :", ron_log2["scores_after"])
                    print()

                turn_index += 1
                continue

            # 鳴いたプレイヤーの左隣から次の巡目を始める
            gs.turn_pid = (naki_pid + 1) % 4
            turn_index += 1
            continue

        # 次巡へ
        gs.turn_pid = (gs.turn_pid + 1) % 4

        if gs.turn_pid == gs.dealer_pid:
            gs.junme += 1

        turn_index += 1

    # 流局
    dealer_tenpai = is_tenpai_hand(gs, gs.dealer_pid)
    tenpai_players = [pid for pid in range(4) if is_tenpai_hand(gs, pid)]
    noten_players = [pid for pid in range(4) if pid not in tenpai_players]

    scores_before_ryuukyoku = gs.scores.copy()

    # ノーテン罰符
    if 0 < len(tenpai_players) < 4:
        total_noten_payment = 3000

        pay_per_noten = total_noten_payment // len(noten_players)
        receive_per_tenpai = total_noten_payment // len(tenpai_players)

        for pid in noten_players:
            gs.scores[pid] -= pay_per_noten

        for pid in tenpai_players:
            gs.scores[pid] += receive_per_tenpai

    log_row = {
        "turn": turn_index,
        "pid": None,
        "junme": gs.junme,
        "draw": None,
        "discard": None,
        "result": "ryuukyoku",
        "remaining_wall": len(gs.wall),
        "dealer_tenpai": dealer_tenpai,
        "tenpai_players": tenpai_players,
        "noten_players": noten_players,
        "scores_before_ryuukyoku": scores_before_ryuukyoku,
        "scores_after_ryuukyoku": gs.scores.copy(),
    }
    logs.append(log_row)

    gs = advance_round(gs, dealer_continues=dealer_tenpai, is_ryuukyoku=True)

    if verbose:
        print("=== Minimal Selfplay End ===")
        print("result         = ryuukyoku")
        print("remaining wall =", len(gs.wall))
        print("turns played   =", turn_index)
        print("dealer tenpai  =", dealer_tenpai)
        print("tenpai players =", tenpai_players)
        print("noten players  =", noten_players)
        print("scores after ryuukyoku =", gs.scores)

        if not gs.game_over:
            print("next dealer    =", gs.dealer_pid)
            print("next bakaze/kyoku =", f"{gs.bakaze}-{gs.kyoku}")
            print("next honba     =", gs.honba)
        else:
            print("game over      = True")

    return gs, logs