# mahjong_engine.py
import numpy as np
import math
import copy

# 牌の名前リストなどの定数
tile_names = ["一萬", "二萬", "三萬", "四萬", "五萬", "六萬", "七萬", "八萬", "九萬",
              "一筒", "二筒", "三筒", "四筒", "五筒", "六筒", "七筒", "八筒", "九筒",
              "一索", "二索", "三索", "四索", "五索", "六索", "七索", "八索", "九索",
              "東", "南", "西", "北", "白", "發", "中"]

# ==========================================
# フェーズ1〜4で作った各種関数をここにコピペします
# ==========================================
# def decompose_hand_with_naki(...):
# print("🧩 点数計算フェーズ1改：『鳴き（副露）対応版・ブロック分解エンジン』にアップデート中...")

def decompose_hand_with_naki(closed_counts, fixed_mentsu):
    """
    closed_counts: 鳴いていない手の中にある牌の枚数リスト（例：1ポンなら11枚分）
    fixed_mentsu: すでに鳴いて確定している面子のリスト（例：[[31, 31, 31]]）
    """
    valid_decompositions = []
    
    # 探すべき「未確定の面子」の数は、4つから「すでに鳴いている数」を引いたものになる！
    target_mentsu_count = 4 - len(fixed_mentsu)
    
    def search_mentsu(current_counts, current_mentsu_list):
        # 指定された数の面子が見つかれば成功！
        if len(current_mentsu_list) == target_mentsu_count:
            return [current_mentsu_list.copy()]
            
        found_patterns = []
        start_idx = 0
        while start_idx < 34 and current_counts[start_idx] == 0:
            start_idx += 1
            
        if start_idx == 34:
            return []

        # パターンA：暗刻
        if current_counts[start_idx] >= 3:
            current_counts[start_idx] -= 3
            current_mentsu_list.append([start_idx, start_idx, start_idx])
            found_patterns.extend(search_mentsu(current_counts, current_mentsu_list))
            current_mentsu_list.pop()
            current_counts[start_idx] += 3

        # パターンB：順子
        if start_idx < 27 and start_idx % 9 <= 6:
            if current_counts[start_idx] >= 1 and current_counts[start_idx+1] >= 1 and current_counts[start_idx+2] >= 1:
                current_counts[start_idx] -= 1
                current_counts[start_idx+1] -= 1
                current_counts[start_idx+2] -= 1
                current_mentsu_list.append([start_idx, start_idx+1, start_idx+2])
                found_patterns.extend(search_mentsu(current_counts, current_mentsu_list))
                current_mentsu_list.pop()
                current_counts[start_idx] += 1
                current_counts[start_idx+1] += 1
                current_counts[start_idx+2] += 1

        return found_patterns

    # 1. まず、非公開の手牌から雀頭（アタマ）を仮決めする
    for janto_idx in range(34):
        if closed_counts[janto_idx] >= 2:
            temp_counts = closed_counts.copy()
            temp_counts[janto_idx] -= 2
            
            # 2. 残りの手牌から「足りない分の面子」を探し出す
            mentsu_patterns = search_mentsu(temp_counts, [])
            
            # 3. 見つかったパターンに「鳴き面子（fixed_mentsu）」を合体させる！
            for pattern in mentsu_patterns:
                combined_mentsu = pattern + fixed_mentsu
                valid_decompositions.append({
                    "janto": [janto_idx, janto_idx],
                    "mentsu": combined_mentsu,
                    "closed_mentsu": pattern, # 役判定（三暗刻など）のために「自力で作った面子」も記録しておく
                    "fixed_mentsu": fixed_mentsu
                })
                
    return valid_decompositions

# def calculate_fu_with_naki(...):
def calculate_fu_with_naki(janto, closed_mentsu, fixed_mentsu, win_tile, is_tsumo, bakaze, jikaze):
    """
    鳴き（fixed_mentsu）を考慮して、正確な符を計算する
    """
    fu_base = 20
    is_menzen = len(fixed_mentsu) == 0 # 鳴き面子が1つもなければ門前！
    
    # 1. アガリ方の符（ツモは常に2符、門前ロンは10符、鳴きロンは0符！）
    fu_win = 0
    if is_tsumo:
        fu_win = 2
    elif is_menzen:
        fu_win = 10
        
    fu_janto = 0
    fu_mentsu = 0
    yaochu = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    yakuhai = {31, 32, 33, 27 + bakaze, 27 + jikaze}
    
    # 2. 雀頭の符計算（役牌なら+2符、連風牌なら+4符）
    if janto[0] in yakuhai:
        fu_janto += 2
        if janto[0] == 27 + bakaze and janto[0] == 27 + jikaze:
            fu_janto += 2 

    # 3. 鳴いた面子（fixed_mentsu）の計算 ＝ すべて「明刻（ミンコウ）」扱い
    # （※今回はポン・チーの前提で組んでいます。カンが含まれる場合はここでさらに判定を増やします）
    for m in fixed_mentsu:
        if m[0] == m[1] == m[2]: # ポン（明刻）の場合
            is_yaochu = m[0] in yaochu
            base_kou_fu = 2 # 明刻なので基本2符
            fu_mentsu += base_kou_fu * 2 if is_yaochu else base_kou_fu

    # 4. 面前構成部分（closed_mentsu）の計算
    for m in closed_mentsu:
        if m[0] == m[1] == m[2]: # 暗刻（アンコウ）候補
            is_yaochu = m[0] in yaochu
            # ロンアガリで、かつアガリ牌でこの暗刻が完成した場合は「明刻」に格下げされる！
            is_minkou = (not is_tsumo) and (win_tile in m)
            
            base_kou_fu = 2 if is_minkou else 4 # 暗刻なら4符！
            fu_mentsu += base_kou_fu * 2 if is_yaochu else base_kou_fu

    # 5. 待ちの符計算（待ちは自力で作った closed_mentsu と janto にしか存在しない）
    max_wait_fu = 0
    if win_tile in janto:
        max_wait_fu = max(max_wait_fu, 2) # 単騎待ち
        
    for m in closed_mentsu:
        if win_tile in m:
            if m[0] == m[1] == m[2]: continue
            idx = m.index(win_tile)
            if idx == 1: 
                max_wait_fu = max(max_wait_fu, 2) # カンチャン
            elif idx == 0 and m[2] % 9 == 8: 
                max_wait_fu = max(max_wait_fu, 2) # ペンチャン
            elif idx == 2 and m[0] % 9 == 0: 
                max_wait_fu = max(max_wait_fu, 2) # ペンチャン

    total_fu = fu_base + fu_win + fu_janto + fu_mentsu + max_wait_fu

    # 🚨 例外処理
    # ① 門前平和（ピンフ）の特例
    if is_menzen and fu_janto == 0 and fu_mentsu == 0 and max_wait_fu == 0:
        return 20 if is_tsumo else 30
        
    # ② 喰い平和（ピンフ）形のロンアガリは、計算上20符になるが強制的に30符にする
    if not is_menzen and total_fu == 20: 
        return 30

    # 最後に一の位を切り上げる（例: 32符 -> 40符）
    return int(math.ceil(total_fu / 10.0) * 10)

# ============================================================
# 役判定ヘルパー関数
# ============================================================

YAOCHU_TILES = frozenset({0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33})
TERMINAL_TILES = frozenset({0, 8, 9, 17, 18, 26})

def _get_dora_tiles_internal(dora_indicators):
    """ドラ表示牌リストから実際のドラ牌リストを返す（mahjong_engine内部用）"""
    dora_tiles = []
    for ind in dora_indicators:
        if 0 <= ind <= 26:   # 数牌
            suit = ind // 9
            rank = ind % 9
            dora_tiles.append(suit * 9 + (rank + 1) % 9)
        elif 27 <= ind <= 30:  # 風牌
            dora_tiles.append(27 + (ind - 27 + 1) % 4)
        elif 31 <= ind <= 33:  # 三元牌
            dora_tiles.append(31 + (ind - 31 + 1) % 3)
    return dora_tiles


def _count_dora_in_hand(closed_counts, fixed_mentsu, dora_indicators):
    """手牌中のドラ枚数を数える"""
    if not dora_indicators:
        return 0
    dora_tiles = _get_dora_tiles_internal(dora_indicators)
    total = 0
    for d in dora_tiles:
        total += closed_counts[d]
        for meld in fixed_mentsu:
            total += sum(1 for t in meld if t == d)
    return total


def _is_chiitoi_form(closed_counts, fixed_mentsu):
    """七対子形かどうか判定（14枚、7種類の対子、鳴きなし）"""
    if fixed_mentsu:
        return False
    total = sum(closed_counts)
    if total != 14:
        return False
    for c in closed_counts:
        if c != 0 and c != 2:
            return False
    return sum(1 for c in closed_counts if c == 2) == 7


# def evaluate_all_yaku(...):
def check_tanyao(janto, closed_mentsu, fixed_mentsu):
    """断幺九（タンヤオ）：1翻"""
    yaochu = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33} # 1,9,字牌
    all_mentsu = closed_mentsu + fixed_mentsu
    
    if janto[0] in yaochu: return 0
    for m in all_mentsu:
        for tile in m:
            if tile in yaochu: return 0
    return 1 # 全て2〜8の牌なら1翻！

def check_yakuhai(closed_mentsu, fixed_mentsu, bakaze, jikaze):
    """役牌（白發中・場風・自風）：各1翻"""
    han = 0
    all_mentsu = closed_mentsu + fixed_mentsu
    yakuhai_tiles = {31, 32, 33, 27 + bakaze, 27 + jikaze}
    
    for m in all_mentsu:
        if m[0] == m[1] == m[2] and m[0] in yakuhai_tiles:
            han += 1
            # ダブ東などの連風牌は2翻にする処理
            if m[0] == 27 + bakaze and m[0] == 27 + jikaze:
                han += 1
    return han

def check_pinfu(janto, closed_mentsu, fixed_mentsu, win_tile, bakaze, jikaze):
    """平和（ピンフ）：1翻"""
    # 1. 門前（鳴きなし）であること
    if len(fixed_mentsu) > 0: return 0
    
    # 2. 全て順子（階段）であること
    for m in closed_mentsu:
        if m[0] == m[1] == m[2]: return 0 # 暗刻があればダメ
        
    # 3. 雀頭が役牌ではないこと
    yakuhai_tiles = {31, 32, 33, 27 + bakaze, 27 + jikaze}
    if janto[0] in yakuhai_tiles: return 0
    
    # 4. リャンメン待ちであること
    is_ryanmen = False
    for m in closed_mentsu:
        if win_tile in m:
            idx = m.index(win_tile)
            # 順子の両端のどちらかで待っていて、かつペンチャン（[1,2,3]の3待ちなど）ではない
            if (idx == 0 and m[2] % 9 != 8) or (idx == 2 and m[0] % 9 != 0):
                is_ryanmen = True
                break # 1つでもリャンメン解釈があればOK（高点法）
                
    return 1 if is_ryanmen else 0

def check_iipeikou(closed_mentsu, fixed_mentsu):
    """一盃口(1翻) / 二盃口(3翻)：門前のみ"""
    if len(fixed_mentsu) > 0:
        return 0
    shuntsu_list = [f"{m[0]}-{m[1]}-{m[2]}" for m in closed_mentsu if m[0] != m[1]]
    from collections import Counter
    cnt = Counter(shuntsu_list)
    pairs_count = sum(v // 2 for v in cnt.values())
    if pairs_count >= 2:
        return 3  # 二盃口
    elif pairs_count == 1:
        return 1  # 一盃口
    return 0


def check_sanankou(closed_mentsu, win_tile, is_tsumo):
    """三暗刻(2翻)：手中の暗刻が3つ以上"""
    ankou_count = 0
    for m in closed_mentsu:
        if m[0] == m[1] == m[2]:
            # ロン和了でアガリ牌を含む刻子は明刻扱い
            if not is_tsumo and win_tile in m:
                continue
            ankou_count += 1
    return 2 if ankou_count >= 3 else 0


def check_chanta_junchan(janto, closed_mentsu, fixed_mentsu):
    """
    混全帯幺九 chanta (2翻/1翻) または 純全帯幺九 junchan (3翻/2翻) を判定。
    返り値: (han, yaku_name) または (0, "")
    """
    all_mentsu = closed_mentsu + fixed_mentsu

    # 全ブロック(雀頭含む)に么九牌が含まれるか
    for m in all_mentsu:
        if not any(t in YAOCHU_TILES for t in m):
            return 0, ""
    if not any(t in YAOCHU_TILES for t in janto):
        return 0, ""

    # 少なくとも1つの順子が必要（対々和との重複防止）
    has_sequence = any(m[0] != m[1] and m[0] < 27 for m in all_mentsu)
    if not has_sequence:
        return 0, ""

    is_open = len(fixed_mentsu) > 0

    # 字牌が含まれているか
    has_honor = any(t >= 27 for m in all_mentsu for t in m) or any(t >= 27 for t in janto)

    if not has_honor:
        # 純全帯幺九 (老頭牌のみ)
        han = 2 if is_open else 3
        return han, f"純全帯幺九({han}翻)"
    else:
        # 混全帯幺九
        han = 1 if is_open else 2
        return han, f"混全帯幺九({han}翻)"

def check_sanshoku(closed_mentsu, fixed_mentsu):
    """三色同順（サンショクドウジュン）：門前2翻 / 鳴き1翻"""
    all_mentsu = closed_mentsu + fixed_mentsu
    
    # 全ての順子の「開始ナンバー（0〜8）」をスートごとに分類する
    shuntsu_starts = {'manzu': set(), 'pinzu': set(), 'souzu': set()}
    
    for m in all_mentsu:
        if m[0] != m[1]: # 順子の場合のみ
            start_num = m[0] % 9
            if m[0] < 9: shuntsu_starts['manzu'].add(start_num)
            elif m[0] < 18: shuntsu_starts['pinzu'].add(start_num)
            elif m[0] < 27: shuntsu_starts['souzu'].add(start_num)
            
    # 萬子、筒子、索子のすべてに共通する「開始ナンバー」があるか探す！
    common_starts = shuntsu_starts['manzu'] & shuntsu_starts['pinzu'] & shuntsu_starts['souzu']
    
    if len(common_starts) > 0:
        return 2 if len(fixed_mentsu) == 0 else 1
    return 0

def check_ittsuu(closed_mentsu, fixed_mentsu):
    """一気通貫（イッツー）：門前2翻 / 鳴き1翻"""
    all_mentsu = closed_mentsu + fixed_mentsu

    # 各色ごとに「順子の開始位置」を集める
    shuntsu_starts = {
        'manzu': set(),
        'pinzu': set(),
        'souzu': set()
    }

    for m in all_mentsu:
        # 順子だけを見る
        if m[0] != m[1]:
            start = m[0]

            # 萬子
            if 0 <= start <= 6:
                shuntsu_starts['manzu'].add(start)
            # 筒子
            elif 9 <= start <= 15:
                shuntsu_starts['pinzu'].add(start - 9)
            # 索子
            elif 18 <= start <= 24:
                shuntsu_starts['souzu'].add(start - 18)

    # 同じ色で 123, 456, 789 の3つが揃っているか
    for suit in ['manzu', 'pinzu', 'souzu']:
        if {0, 3, 6}.issubset(shuntsu_starts[suit]):
            return 2 if len(fixed_mentsu) == 0 else 1

    return 0


def check_honitsu(janto, closed_mentsu, fixed_mentsu):
    """混一色（ホンイツ）：門前3翻 / 鳴き2翻"""
    all_tiles = []

    # 雀頭
    all_tiles.extend(janto)

    # 面子
    for m in closed_mentsu + fixed_mentsu:
        all_tiles.extend(m)

    suits_used = set()
    has_honor = False

    for t in all_tiles:
        if 0 <= t <= 8:
            suits_used.add('manzu')
        elif 9 <= t <= 17:
            suits_used.add('pinzu')
        elif 18 <= t <= 26:
            suits_used.add('souzu')
        else:
            has_honor = True

    # 数牌がちょうど1色だけ、かつ字牌を含む
    if len(suits_used) == 1 and has_honor:
        return 3 if len(fixed_mentsu) == 0 else 2

    return 0


def check_chinitsu(janto, closed_mentsu, fixed_mentsu):
    """清一色（チンイツ）：門前6翻 / 鳴き5翻"""
    all_tiles = []

    # 雀頭
    all_tiles.extend(janto)

    # 面子
    for m in closed_mentsu + fixed_mentsu:
        all_tiles.extend(m)

    suits_used = set()
    has_honor = False

    for t in all_tiles:
        if 0 <= t <= 8:
            suits_used.add('manzu')
        elif 9 <= t <= 17:
            suits_used.add('pinzu')
        elif 18 <= t <= 26:
            suits_used.add('souzu')
        else:
            has_honor = True

    # 数牌がちょうど1色だけ、かつ字牌なし
    if len(suits_used) == 1 and not has_honor:
        return 6 if len(fixed_mentsu) == 0 else 5

    return 0
  
def check_toitoi(closed_mentsu, fixed_mentsu):
    """対々和（トイトイホー）：2翻"""
    all_mentsu = closed_mentsu + fixed_mentsu
    # 刻子が4つあれば対々和
    return 2 if sum(1 for m in all_mentsu if m[0] == m[1]) == 4 else 0

def evaluate_all_yaku(janto, closed_mentsu, fixed_mentsu, win_tile, is_tsumo, bakaze, jikaze):
    total_han = 0
    yaku_list = []

    # 門前清自摸和（ツモ）
    if is_tsumo and len(fixed_mentsu) == 0:
        total_han += 1
        yaku_list.append("門前清自摸和(1翻)")

    han_tanyao = check_tanyao(janto, closed_mentsu, fixed_mentsu)
    if han_tanyao > 0:
        total_han += han_tanyao
        yaku_list.append(f"断幺九({han_tanyao}翻)")

    han_yakuhai = check_yakuhai(closed_mentsu, fixed_mentsu, bakaze, jikaze)
    if han_yakuhai > 0:
        total_han += han_yakuhai
        yaku_list.append(f"役牌({han_yakuhai}翻)")

    han_pinfu = check_pinfu(janto, closed_mentsu, fixed_mentsu, win_tile, bakaze, jikaze)
    if han_pinfu > 0:
        total_han += han_pinfu
        yaku_list.append(f"平和({han_pinfu}翻)")

    han_iipeikou = check_iipeikou(closed_mentsu, fixed_mentsu)
    if han_iipeikou > 0:
        label = "二盃口" if han_iipeikou == 3 else "一盃口"
        total_han += han_iipeikou
        yaku_list.append(f"{label}({han_iipeikou}翻)")

    han_sanshoku = check_sanshoku(closed_mentsu, fixed_mentsu)
    if han_sanshoku > 0:
        total_han += han_sanshoku
        yaku_list.append(f"三色同順({han_sanshoku}翻)")

    han_ittsuu = check_ittsuu(closed_mentsu, fixed_mentsu)
    if han_ittsuu > 0:
        total_han += han_ittsuu
        yaku_list.append(f"一気通貫({han_ittsuu}翻)")

    han_chanta, chanta_name = check_chanta_junchan(janto, closed_mentsu, fixed_mentsu)
    if han_chanta > 0:
        total_han += han_chanta
        yaku_list.append(chanta_name)

    han_honitsu = check_honitsu(janto, closed_mentsu, fixed_mentsu)
    if han_honitsu > 0:
        total_han += han_honitsu
        yaku_list.append(f"混一色({han_honitsu}翻)")

    han_chinitsu = check_chinitsu(janto, closed_mentsu, fixed_mentsu)
    if han_chinitsu > 0:
        total_han += han_chinitsu
        yaku_list.append(f"清一色({han_chinitsu}翻)")

    han_toitoi = check_toitoi(closed_mentsu, fixed_mentsu)
    if han_toitoi > 0:
        total_han += han_toitoi
        yaku_list.append(f"対々和({han_toitoi}翻)")

    han_sanankou = check_sanankou(closed_mentsu, win_tile, is_tsumo)
    if han_sanankou > 0:
        total_han += han_sanankou
        yaku_list.append(f"三暗刻({han_sanankou}翻)")

    return total_han, yaku_list

# def calculate_final_score(...):
def calculate_final_score(
    closed_counts,
    fixed_mentsu,
    win_tile,
    is_tsumo,
    bakaze,
    jikaze,
    is_oya=False,
    is_riichi=False,
    dora_indicators=None,
    aka_dora_count=0,
    is_ippatsu=False,
    kiriage_mangan=True
):
    def _build_result(han, fu, yaku_list, decomp=None):
        """スコア計算して結果辞書を作る内部ヘルパー"""
        # 一発加算
        if is_ippatsu:
            han += 1
            yaku_list.insert(0, "一発(1翻)")
        # ドラ加算
        if dora_indicators:
            dora_count = _count_dora_in_hand(closed_counts, fixed_mentsu, dora_indicators)
            if dora_count > 0:
                han += dora_count
                yaku_list.append(f"ドラ{dora_count}({dora_count}翻)")
        # 赤ドラ加算
        if aka_dora_count > 0:
            han += aka_dora_count
            yaku_list.append(f"赤ドラ{aka_dora_count}({aka_dora_count}翻)")
        if han == 0:
            return None
        score_info = calc_base_score(han, fu, is_oya=is_oya, is_tsumo=is_tsumo, kiriage_mangan=kiriage_mangan)
        ron_score = score_info["ron_score"]
        if is_tsumo:
            tsumo_total = (score_info["tsumo_oya_payment"] * 3 if is_oya
                           else score_info["tsumo_ko_payment"] + score_info["tsumo_ko_payment_child"] * 2)
            score_for_compare = tsumo_total
        else:
            tsumo_total = 0
            score_for_compare = ron_score
        return {
            "score": score_for_compare,
            "han": han,
            "fu": fu,
            "yaku": yaku_list,
            "interpretation": decomp,
            "ron_score": ron_score,
            "tsumo_total": tsumo_total,
            "tsumo_oya_payment": score_info["tsumo_oya_payment"],
            "tsumo_ko_payment": score_info["tsumo_ko_payment"],
            "tsumo_ko_payment_child": score_info["tsumo_ko_payment_child"],
            "score_detail": score_info,
        }

    null_result = {
        "score": 0, "han": 0, "fu": 0, "yaku": ["アガリ形ではありません"],
        "ron_score": 0, "tsumo_total": 0,
        "tsumo_oya_payment": 0, "tsumo_ko_payment": 0, "tsumo_ko_payment_child": 0,
        "score_detail": None,
    }

    best_result = None

    # ========== 七対子チェック ==========
    if _is_chiitoi_form(closed_counts, fixed_mentsu):
        han_c = 2
        yaku_c = ["七対子(2翻)"]
        if is_riichi:
            han_c += 1
            yaku_c.insert(0, "立直(1翻)")
        r = _build_result(han_c, 25, yaku_c)
        if r and (best_result is None or r["score"] > best_result["score"]):
            best_result = r

    # ========== 通常手分解 ==========
    decompositions = decompose_hand_with_naki(closed_counts, fixed_mentsu)
    for decomp in decompositions:
        janto_d = decomp["janto"]
        cm_d = decomp["closed_mentsu"]

        fu = calculate_fu_with_naki(janto_d, cm_d, fixed_mentsu, win_tile, is_tsumo, bakaze, jikaze)
        han, yaku_list = evaluate_all_yaku(janto_d, cm_d, fixed_mentsu, win_tile, is_tsumo, bakaze, jikaze)

        if is_riichi and len(fixed_mentsu) == 0:
            han += 1
            yaku_list = ["立直(1翻)"] + yaku_list

        r = _build_result(han, fu, list(yaku_list), decomp)
        if r and (best_result is None or r["score"] > best_result["score"] or
                  (r["score"] == best_result["score"] and r["han"] > best_result["han"])):
            best_result = r

    if best_result is None:
        no_yaku = dict(null_result)
        no_yaku["yaku"] = ["役なし"]
        return no_yaku

    return best_result

# def calc_base_score(...):
def calc_base_score(han, fu, is_oya=False, is_tsumo=False, kiriage_mangan=True):
    """
    点数計算の共通関数

    return:
        {
            "base_points": 基本点,
            "ron_score": ロン和了時の総点,
            "tsumo_oya_payment": 親ツモ時のオール額,
            "tsumo_ko_payment": 子ツモ時に親が払う額,
            "tsumo_ko_payment_child": 子ツモ時に子が払う額,
        }
    """
    if han == 0:
        return {
            "base_points": 0,
            "ron_score": 0,
            "tsumo_oya_payment": 0,
            "tsumo_ko_payment": 0,
            "tsumo_ko_payment_child": 0,
        }

    # 満貫以上
    if han >= 13:
        base = 8000
    elif han >= 11:
        base = 6000
    elif han >= 8:
        base = 4000
    elif han >= 6:
        base = 3000
    elif han >= 5:
        base = 2000
    else:
        base = fu * (2 ** (han + 2))

        # 切り上げ満貫
        if kiriage_mangan and ((han == 4 and fu == 30) or (han == 3 and fu == 60)):
            base = 2000

        # 通常満貫上限
        if base > 2000:
            base = 2000

    # ロン点
    ron_score = int(math.ceil((base * (6 if is_oya else 4)) / 100.0) * 100)

    # ツモ点
    if is_oya:
        # 親ツモは全員同額
        tsumo_oya_payment = int(math.ceil((base * 2) / 100.0) * 100)
        tsumo_ko_payment = 0
        tsumo_ko_payment_child = 0
    else:
        # 子ツモは親と子で支払いが違う
        tsumo_ko_payment = int(math.ceil((base * 2) / 100.0) * 100)       # 親支払い
        tsumo_ko_payment_child = int(math.ceil(base / 100.0) * 100)       # 子支払い
        tsumo_oya_payment = 0

    return {
        "base_points": base,
        "ron_score": ron_score,
        "tsumo_oya_payment": tsumo_oya_payment,
        "tsumo_ko_payment": tsumo_ko_payment,
        "tsumo_ko_payment_child": tsumo_ko_payment_child,
    }

def _apply_honba_and_riichi(gs, winner, loser, is_tsumo):
    """
    本場・供託の加算処理
    - 本場は1本場につき300点
      - ロン: 放銃者がまとめて支払う
      - ツモ: 各支払者が100点ずつ加算して支払う
    - 供託(riichi_sticks)は和了者が総取り
    """
    honba = gs.honba
    riichi_bonus = gs.riichi_sticks * 1000

    if is_tsumo:
        # 本場は各支払者100点ずつ
        for pid in range(4):
            if pid == winner:
                continue
            gs.scores[pid] -= honba * 100
            gs.scores[winner] += honba * 100
    else:
        # ロンは放銃者が300点×本場をまとめて払う
        gs.scores[loser] -= honba * 300
        gs.scores[winner] += honba * 300

    # 供託は和了者が総取り
    gs.scores[winner] += riichi_bonus
    gs.riichi_sticks = 0


def apply_agari_result(gs, winner, loser, agari_result, is_tsumo):
    """
    和了結果を点棒に反映する

    Parameters
    ----------
    gs : GlobalRoundState
    winner : int
        和了者pid
    loser : int or None
        ロン放銃者。ツモ時は None
    agari_result : dict
        check_tsumo_agari / check_ron_agari が返す辞書
        想定キー:
          - "han"
          - "fu"
          - "score"
          - "score_detail"  (あれば優先使用)
    is_tsumo : bool
        ツモならTrue、ロンならFalse
    """
    if is_tsumo:
        detail = agari_result.get("score_detail", None)

        if detail is None:
            raise ValueError("tsumo agari_result に score_detail がありません")

        winner_is_oya = (winner == gs.dealer_pid)

        if winner_is_oya:
            pay = detail["tsumo_oya_payment"]
            for pid in range(4):
                if pid == winner:
                    continue
                gs.scores[pid] -= pay
                gs.scores[winner] += pay
        else:
            pay_oya = detail["tsumo_oya_payment"]
            pay_ko = detail["tsumo_ko_payment_child"]

            for pid in range(4):
                if pid == winner:
                    continue
                if pid == gs.dealer_pid:
                    gs.scores[pid] -= pay_oya
                    gs.scores[winner] += pay_oya
                else:
                    gs.scores[pid] -= pay_ko
                    gs.scores[winner] += pay_ko

        _apply_honba_and_riichi(gs, winner, loser=None, is_tsumo=True)

    else:
        if loser is None:
            raise ValueError("ronなのに loser が None です")

        detail = agari_result.get("score_detail", None)

        if detail is not None:
            ron_score = detail["ron_score"]
        else:
            ron_score = agari_result.get("score", None)

        if ron_score is None:
            raise ValueError("ron agari_result に score / score_detail がありません")

        gs.scores[loser] -= ron_score
        gs.scores[winner] += ron_score

        _apply_honba_and_riichi(gs, winner, loser=loser, is_tsumo=False)

    return gs

# def calculate_shanten(...):
def calculate_shanten_normal(hand_counts):
    """
    34種類の牌の枚数リスト(hand_counts)から、一般手(4面子1雀頭)のシャンテン数を計算する
    """
    min_shanten = 8 # 初期値（最大8シャンテン）
    
    # 探索用の再帰関数
    def search(depth, mentsu, taatsu, has_janto, current_hand):
        nonlocal min_shanten
        
        # 枝刈り：すでに現在見つかっている最小シャンテン数より悪くなりそうなら探索打ち切り
        current_shanten = 8 - (mentsu * 2) - taatsu - (1 if has_janto else 0)
        
        # 牌をすべて見終わったか、メンツ+ターツが4つを超えたら評価
        if depth == 34 or (mentsu + taatsu > 4):
            # メンツとターツの合計は最大4つまでしかカウントできない（5ブロック理論）
            valid_taatsu = min(taatsu, 4 - mentsu)
            shanten = 8 - (mentsu * 2) - valid_taatsu - (1 if has_janto else 0)
            if shanten < min_shanten:
                min_shanten = shanten
            return

        # 1. 雀頭（アタマ）を抜くパターン（まだ雀頭がなく、対象の牌が2枚以上ある場合）
        if not has_janto and current_hand[depth] >= 2:
            current_hand[depth] -= 2
            search(depth, mentsu, taatsu, True, current_hand)
            current_hand[depth] += 2 # 元に戻す（バックトラッキング）
            
        # 2. コウツ（刻子：同じ牌3枚）を抜くパターン
        if current_hand[depth] >= 3:
            current_hand[depth] -= 3
            search(depth, mentsu + 1, taatsu, has_janto, current_hand)
            current_hand[depth] += 3

        # 3. シュンツ（順子：階段3枚）を抜くパターン (字牌 depth >= 27 では不可)
        if depth < 27 and depth % 9 <= 6:
            if current_hand[depth] >= 1 and current_hand[depth+1] >= 1 and current_hand[depth+2] >= 1:
                current_hand[depth] -= 1
                current_hand[depth+1] -= 1
                current_hand[depth+2] -= 1
                search(depth, mentsu + 1, taatsu, has_janto, current_hand)
                current_hand[depth] += 1
                current_hand[depth+1] += 1
                current_hand[depth+2] += 1

        # 4. トイツ（対子：同じ牌2枚、ターツ扱い）を抜くパターン
        if current_hand[depth] >= 2:
            current_hand[depth] -= 2
            search(depth, mentsu, taatsu + 1, has_janto, current_hand)
            current_hand[depth] += 2

        # 5. リャンメン・ペンチャン（階段2枚、ターツ扱い）を抜くパターン
        if depth < 27 and depth % 9 <= 7:
            if current_hand[depth] >= 1 and current_hand[depth+1] >= 1:
                current_hand[depth] -= 1
                current_hand[depth+1] -= 1
                search(depth, mentsu, taatsu + 1, has_janto, current_hand)
                current_hand[depth] += 1
                current_hand[depth+1] += 1

        # 6. カンチャン（1つ飛ばし2枚、ターツ扱い）を抜くパターン
        if depth < 27 and depth % 9 <= 6:
            if current_hand[depth] >= 1 and current_hand[depth+2] >= 1:
                current_hand[depth] -= 1
                current_hand[depth+2] -= 1
                search(depth, mentsu, taatsu + 1, has_janto, current_hand)
                current_hand[depth] += 1
                current_hand[depth+2] += 1

        # 7. 何も抜かずに次の牌へ進むパターン（孤立牌として扱う）
        search(depth + 1, mentsu, taatsu, has_janto, current_hand)

    # 探索開始
    search(0, 0, 0, False, copy.copy(hand_counts))
    return min_shanten

# 💡 七対子のシャンテン数計算
def calculate_shanten_chiitoi(counts):
    pairs = sum(1 for c in counts if c >= 2)
    types = sum(1 for c in counts if c >= 1)
    shanten = 6 - pairs
    # ※七対子は同じ牌4枚を2対子として扱えないルールのため、種類数の補正を入れる
    if types < 7:
        shanten += 7 - types
    return shanten

# 💡 国士無双のシャンテン数計算
def calculate_shanten_kokushi(counts):
    yaochu_indices = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33] # 1・9・字牌のインデックス
    yaochu_types = sum(1 for i in yaochu_indices if counts[i] >= 1)
    has_pair = any(counts[i] >= 2 for i in yaochu_indices)
    return 13 - yaochu_types - (1 if has_pair else 0)

# 👑 究極の統合シャンテン数計算機
def calculate_shanten(counts):
    s_normal = calculate_shanten_normal(counts)
    s_chiitoi = calculate_shanten_chiitoi(counts)
    s_kokushi = calculate_shanten_kokushi(counts)
    # 3つのうち、最もアガリに近い（数値が小さい）ものを採用！
    return min(s_normal, s_chiitoi, s_kokushi)

# def get_ukeire(...):
def get_ukeire(counts):
    """
    手牌(counts)を入力すると、「どの牌を引けばシャンテン数が進むか」のリストを返す
    """
    current_shanten = calculate_shanten(counts)
    if current_shanten == -1:
        return [], current_shanten # すでにアガリ
        
    ukeire_tiles = []
    
    # 34種類の牌すべてについて、仮想的に引いてシミュレーションする
    for i in range(34):
        if counts[i] < 4: # 物理的に引ける（まだ手牌に4枚ない）場合のみ
            counts[i] += 1 # 💡 試しに1枚引いてみる
            new_shanten = calculate_shanten(counts)
            counts[i] -= 1 # 💡 すぐに元に戻す
            
            # 引いたことでシャンテン数が減った（進んだ）なら、それは有効牌！
            if new_shanten < current_shanten:
                ukeire_tiles.append(i)
                
    return ukeire_tiles, current_shanten

def calculate_true_ev(
    closed_counts,
    fixed_mentsu,
    wait_tiles,
    visible_tiles,
    dora_indicators,
    bakaze,
    jikaze,
    is_oya=False,
    honba=0,
    kyotaku=0,
    junme=10
):
    """
    テンパイ時の期待値を計算する（改良版 B-2）

    改良点:
      - ツモ和了スコアを追加（ロン60% + ツモ40% の加重平均）
      - リーチEVにもツモを反映し、ウラドラ期待値を正確に適用
      - 本場ボーナスをダマ・リーチ両方に一貫して加算
      - 供託はダマにも加算（役あり・役なしに関わらず）
    """
    ev_dama = 0.0
    ev_riichi = 0.0

    # 見えていない牌の総枚数
    unseen_total = 136 - sum(visible_tiles)
    if unseen_total <= 0:
        return 0.0, 0.0

    can_riichi = len(fixed_mentsu) == 0

    # ロン/ツモ の重み: 統計的に和了の約60%がロン、40%がツモ
    RON_WEIGHT = 0.60
    TSUMO_WEIGHT = 0.40

    # 本場ボーナス: 1本場 = 300点
    honba_bonus = honba * 300

    # ウラドラ期待値率: リーチ和了の約30%でウラドラ1枚乗る
    URADORI_RATE = 0.30

    for wait_tile in wait_tiles:
        remaining = 4 - visible_tiles[wait_tile]
        if remaining <= 0:
            continue

        prob_win = remaining / unseen_total
        counts_14 = closed_counts.copy()
        counts_14[wait_tile] += 1

        # ======== ダマ EV ========
        result_ron = calculate_final_score(
            counts_14, fixed_mentsu, wait_tile,
            is_tsumo=False, bakaze=bakaze, jikaze=jikaze,
            is_oya=is_oya, is_riichi=False
        )
        result_tsumo = calculate_final_score(
            counts_14, fixed_mentsu, wait_tile,
            is_tsumo=True, bakaze=bakaze, jikaze=jikaze,
            is_oya=is_oya, is_riichi=False
        )

        dama_ron_score   = result_ron["score"]
        dama_tsumo_score = result_tsumo["tsumo_total"]

        if dama_ron_score > 0 or dama_tsumo_score > 0:
            # 本場・供託を加算したロン/ツモ期待値
            ron_with_bonus   = dama_ron_score   + honba_bonus + kyotaku
            tsumo_with_bonus = dama_tsumo_score + honba_bonus + kyotaku
            weighted_dama = RON_WEIGHT * ron_with_bonus + TSUMO_WEIGHT * tsumo_with_bonus
            ev_dama += prob_win * weighted_dama

        # ======== リーチ EV ========
        if can_riichi:
            result_r_ron = calculate_final_score(
                counts_14, fixed_mentsu, wait_tile,
                is_tsumo=False, bakaze=bakaze, jikaze=jikaze,
                is_oya=is_oya, is_riichi=True
            )
            result_r_tsumo = calculate_final_score(
                counts_14, fixed_mentsu, wait_tile,
                is_tsumo=True, bakaze=bakaze, jikaze=jikaze,
                is_oya=is_oya, is_riichi=True
            )

            r_ron_score   = result_r_ron["score"]
            r_tsumo_score = result_r_tsumo["tsumo_total"]

            if r_ron_score > 0 or r_tsumo_score > 0:
                base_han = result_r_ron["han"]
                base_fu  = result_r_ron["fu"]

                # ウラドラなし・あり（+1翻）のスコアを計算
                score_ron_no_ura    = calc_base_score(base_han,     base_fu, is_oya)["ron_score"]
                score_ron_with_ura  = calc_base_score(base_han + 1, base_fu, is_oya)["ron_score"]

                info_tsumo_no_ura   = calc_base_score(base_han,     base_fu, is_oya)
                info_tsumo_with_ura = calc_base_score(base_han + 1, base_fu, is_oya)

                if is_oya:
                    score_tsumo_no_ura   = info_tsumo_no_ura["tsumo_oya_payment"]   * 3
                    score_tsumo_with_ura = info_tsumo_with_ura["tsumo_oya_payment"] * 3
                else:
                    score_tsumo_no_ura   = (info_tsumo_no_ura["tsumo_ko_payment"]
                                            + info_tsumo_no_ura["tsumo_ko_payment_child"] * 2)
                    score_tsumo_with_ura = (info_tsumo_with_ura["tsumo_ko_payment"]
                                            + info_tsumo_with_ura["tsumo_ko_payment_child"] * 2)

                # ウラドラ期待値を加味したロン/ツモスコア
                expected_r_ron   = (1 - URADORI_RATE) * score_ron_no_ura   + URADORI_RATE * score_ron_with_ura
                expected_r_tsumo = (1 - URADORI_RATE) * score_tsumo_no_ura + URADORI_RATE * score_tsumo_with_ura

                # 本場・供託加算
                expected_r_ron   += honba_bonus + kyotaku
                expected_r_tsumo += honba_bonus + kyotaku

                weighted_riichi = RON_WEIGHT * expected_r_ron + TSUMO_WEIGHT * expected_r_tsumo
                ev_riichi += prob_win * weighted_riichi

    return ev_riichi, ev_dama

# ==========================================
# 盤面管理と25チャンネルテンソル化クラス（攻守完全体・最終版）
# ==========================================
class MahjongStateV5:
    def __init__(self):
        self.hand = [0] * 34
        self.fixed_mentsu = []
        self.discards = {0: [], 1: [], 2: [], 3: []}
        self.melds = {1: [], 2: [], 3: []}

        self.bakaze = 0
        self.jikaze = 0
        self.dora_indicators = []
        self.riichi_declared = {0: False, 1: False, 2: False, 3: False}
        self.scores = {0: 25000, 1: 25000, 2: 25000, 3: 25000}
        self.honba = 0
        self.kyotaku = 0
        self.forbidden_discards = [] # 今ターン、絶対に捨ててはいけない牌（食い替え防止）

        # 赤ドラフラグ（tile_136: 16=赤五萬, 52=赤五筒, 88=赤五索）
        self.aka_in_hand_5m = False
        self.aka_in_hand_5p = False
        self.aka_in_hand_5s = False

    def add_tile(self, who, tenhou_id):
        tile_type = tenhou_id // 4
        if who == 0:
            self.hand[tile_type] += 1
            if tenhou_id == 16:
                self.aka_in_hand_5m = True
            elif tenhou_id == 52:
                self.aka_in_hand_5p = True
            elif tenhou_id == 88:
                self.aka_in_hand_5s = True
    
    def execute_pon(self, who, tile_type):
        """ポンを厳密に処理し、自分・他家の脳内に晒し牌として記録する"""
        meld = [tile_type, tile_type, tile_type]
        self.melds[who].append(meld) # 全員共通：指定された人の副露リストに追加
        
        if who == 0:
            # 自分が鳴いた場合：手牌から2枚抜き、食い替えを禁止する
            if self.hand[tile_type] >= 2:
                self.hand[tile_type] -= 2
                self.fixed_mentsu.append(meld)
                self.forbidden_discards = [tile_type] # 🚫 食い替え禁止
                return True
            return False
        else:
            # 他家が鳴いた場合：脳内に「あいつがこれを晒した」と記録するだけ（手牌はいじらない）
            return True
    
    def discard_tile(self, who, tenhou_id):
        tile_type = tenhou_id // 4
        self.discards[who].append(tile_type)
        if who == 0:
            self.hand[tile_type] -= 1
            if tenhou_id == 16:
                self.aka_in_hand_5m = False
            elif tenhou_id == 52:
                self.aka_in_hand_5p = False
            elif tenhou_id == 88:
                self.aka_in_hand_5s = False

    @staticmethod
    def _indicator_to_dora(indicator):
        """ドラ表示牌 -> 実際のドラ牌 変換"""
        if indicator < 27:   # 数牌（萬子・筒子・索子）
            suit = indicator // 9
            num  = indicator % 9
            return suit * 9 + (num + 1) % 9
        elif indicator < 31: # 風牌 (東南西北 = 27-30)
            return 27 + (indicator - 27 + 1) % 4
        else:                # 三元牌 (白發中 = 31-33)
            return 31 + (indicator - 31 + 1) % 3

    def to_tensor(self, skip_logic=False):
        """
        33チャンネルテンソル（新設計）
        CH00: 自分の手牌                (count/4.0)
        CH01-03: 他家1-3の捨て牌        (+0.25/枚)
        CH04-06: 他家1-3の副露          (+0.25/枚)
        CH07-09: 他家1-3のリーチ宣言    (全1.0 or 全0.0)
        CH10: 実際のドラマップ           (+0.25/枚, 最大4枚)
        CH11: 場風                       (該当位置=1.0)
        CH12: 自風                       (該当位置=1.0)
        CH13-16: 各プレイヤーの点数      (score/100000.0)
        CH17: 本場                       (honba/10.0)
        CH18: 供托                       (kyotaku/10.0)
        CH19: 自分の副露                 (+0.25/枚)
        CH20: 自分の捨て牌              (+0.25/枚)  [旧設計から新規追加]
        CH21: 巡目進行度                 (1.0=序盤 -> 0.0=終盤)
        CH22: 残り牌枚数マップ           ((4-visible)/4.0)  [CH21バグ修正・専用化]
        CH23: シャンテン数 per discard   (1.0-shanten*0.1)  [skip_logic=Trueなら0]
        CH24: 有効牌数 per discard       (ukeire/40.0)      [skip_logic=Trueなら0]
        CH25: EVダマ per discard         (ev_d/8000.0)      [skip_logic=Trueなら0]
        CH26: EVリーチ per discard       (ev_r/8000.0)      [skip_logic=Trueなら0]
        CH27: 自分のリーチ宣言           (全1.0 or 全0.0)  [新規追加]
        CH28: 自分の着順                 (rank/3.0)         [新規追加]
        CH29: トップとの点差             (gap/30000.0)      [新規追加]
        CH30: 赤五萬保持フラグ           (全1.0 or 全0.0)  [新規追加]
        CH31: 赤五筒保持フラグ           (全1.0 or 全0.0)  [新規追加]
        CH32: 赤五索保持フラグ           (全1.0 or 全0.0)  [新規追加]
        """
        tensor = np.zeros((33, 34), dtype=np.float32)

        # CH0: 自分の手牌
        for i, count in enumerate(self.hand):
            tensor[0][i] = count / 4.0

        # CH1-3: 他家捨て牌 / CH4-6: 他家副露 / CH7-9: 他家リーチ
        for who in [1, 2, 3]:
            for tile_type in self.discards[who]:
                tensor[who][tile_type] += 0.25
            for m in self.melds[who]:
                for tile_type in m:
                    tensor[who + 3][tile_type] += 0.25
            if self.riichi_declared[who]:
                tensor[who + 6].fill(1.0)

        # CH10: 実際のドラマップ（表示牌+1 の実ドラ）
        for ind in self.dora_indicators:
            actual = self._indicator_to_dora(ind)
            tensor[10][actual] += 0.25  # 最大4枚 -> 1.0

        # CH11: 場風 / CH12: 自風
        tensor[11][27 + self.bakaze] = 1.0
        tensor[12][27 + self.jikaze] = 1.0

        # CH13-16: 点数 / CH17: 本場 / CH18: 供托
        for i in range(4):
            tensor[13 + i].fill(self.scores[i] / 100000.0)
        tensor[17].fill(self.honba / 10.0)
        tensor[18].fill(self.kyotaku / 10.0)

        # CH19: 自分の副露
        for m in self.fixed_mentsu:
            for tile_type in m:
                tensor[19][tile_type] += 0.25

        # CH20: 自分の捨て牌（新規追加）
        for tile_type in self.discards[0]:
            tensor[20][tile_type] += 0.25

        # CH21: 巡目進行度
        total_discards = sum(len(d) for d in self.discards.values())
        turn_progress = max(0.0, 1.0 - (total_discards / 70.0))
        tensor[21].fill(turn_progress)

        # CH22: 残り牌枚数マップ（CH21バグを修正した専用チャンネル）
        visible = np.zeros(34, dtype=np.float32)
        for tile_type, cnt in enumerate(self.hand):
            visible[tile_type] += cnt
        for who in range(4):
            for tile_type in self.discards[who]:
                visible[tile_type] += 1
        for m in self.fixed_mentsu:
            for tile_type in m:
                visible[tile_type] += 1
        for who in [1, 2, 3]:
            for m in self.melds[who]:
                for tile_type in m:
                    visible[tile_type] += 1
        for ind in self.dora_indicators:  # 表示牌は表向きなので見える
            visible[ind] += 1
        visible = np.minimum(visible, 4.0)
        tensor[22] = np.maximum(0.0, 4.0 - visible) / 4.0

        # CH27: 自分のリーチ宣言（新規追加）
        if self.riichi_declared[0]:
            tensor[27].fill(1.0)

        # CH28: 着順 / CH29: トップとの点差（新規追加）
        scores_list = [self.scores[i] for i in range(4)]
        my_score = scores_list[0]
        rank = sum(1 for s in scores_list if s > my_score)
        tensor[28].fill(rank / 3.0)
        max_score = max(scores_list)
        score_gap = min(1.0, max(0.0, max_score - my_score) / 30000.0)
        tensor[29].fill(score_gap)

        # CH30-32: 赤ドラ保持フラグ（新規追加）
        if self.aka_in_hand_5m:
            tensor[30].fill(1.0)
        if self.aka_in_hand_5p:
            tensor[31].fill(1.0)
        if self.aka_in_hand_5s:
            tensor[32].fill(1.0)

        # skip_logic=True なら CH23-26 はゼロのまま返す（鳴き判断用高速化）
        if skip_logic:
            return tensor

        # --- CH23-26: per-discard 論理エンジン ---
        for discard_idx in range(34):
            if self.hand[discard_idx] > 0:
                temp_hand = self.hand.copy()
                temp_hand[discard_idx] -= 1

                # CH23: シャンテン数
                shanten = calculate_shanten(temp_hand)
                tensor[23][discard_idx] = max(0.0, 1.0 - (shanten * 0.1))

                # CH24: 有効牌（受け入れ）
                ukeire_tiles, _ = get_ukeire(temp_hand)
                if len(ukeire_tiles) > 0:
                    visible_tiles = [0] * 34
                    for tile_type, count in enumerate(temp_hand):
                        visible_tiles[tile_type] += count
                    for who2 in range(4):
                        for tile_type in self.discards[who2]:
                            visible_tiles[tile_type] += 1
                    for i in range(34):
                        if visible_tiles[i] > 4:
                            visible_tiles[i] = 4
                    for dora_ind in self.dora_indicators:
                        visible_tiles[dora_ind] += 1
                    for m in self.fixed_mentsu:
                        for tile_type in m:
                            visible_tiles[tile_type] += 1
                    for who2 in [1, 2, 3]:
                        for m in self.melds[who2]:
                            for tile_type in m:
                                visible_tiles[tile_type] += 1

                    ukeire_count = sum(max(0, 4 - visible_tiles[ut]) for ut in ukeire_tiles)
                    tensor[24][discard_idx] = min(1.0, ukeire_count / 40.0)

                    # CH25-26: EV
                    ev_r, ev_d = calculate_true_ev(
                        temp_hand,
                        self.fixed_mentsu,
                        ukeire_tiles,
                        visible_tiles,
                        self.dora_indicators,
                        self.bakaze,
                        self.jikaze,
                        is_oya=(self.jikaze == 0),
                        honba=self.honba,
                        kyotaku=self.kyotaku,
                        junme=int(total_discards / 4) + 1
                    )
                    tensor[25][discard_idx] = min(1.0, ev_d / 8000.0)
                    tensor[26][discard_idx] = min(1.0, ev_r / 8000.0)

        return tensor

    def can_naki(self, discarded_tile, who_discarded):
        """今捨てられた牌に対して、物理的にポンやチーが可能か判定する"""
        # 1. ポン判定（誰からでもOK）
        if self.hand[discarded_tile] >= 2:
            return True
            
        # 2. チー判定（上家=3 からのみOK）
        if who_discarded == 3 and discarded_tile < 27:
            num = discarded_tile % 9
            # パターンA: [n-2, n-1, n(捨)]
            if num >= 2 and self.hand[discarded_tile-2] >= 1 and self.hand[discarded_tile-1] >= 1: return True
            # パターンB: [n-1, n(捨), n+1]
            if 1 <= num <= 7 and self.hand[discarded_tile-1] >= 1 and self.hand[discarded_tile+1] >= 1: return True
            # パターンC: [n(捨), n+1, n+2]
            if num <= 6 and self.hand[discarded_tile+1] >= 1 and self.hand[discarded_tile+2] >= 1: return True
            
        return False

    def to_tensor_for_naki(self, discarded_tile):
        """鳴き判断専用の『26チャンネル・神の目』を生成する"""
        # 🚀 スイッチON！激重計算をスキップして爆速でベース画像を取得！
        base_tensor = self.to_tensor(skip_logic=True)

        target_channel = np.zeros((1, 34), dtype=np.float32)
        target_channel[0][discarded_tile] = 1.0

        return np.concatenate((base_tensor, target_channel), axis=0)

    def to_tensor_for_naki_v2(self, discarded_tile):
        """鳴き判断専用の34チャンネルテンソル（33ch基底 + 1ch捨て牌位置）"""
        # 33ch基底テンソル（CH28: 着順, CH29: 点差 はすでに含まれている）
        base_tensor = self.to_tensor(skip_logic=True)  # shape: (33, 34)

        # CH33: 捨て牌の位置
        target_channel = np.zeros((1, 34), dtype=np.float32)
        target_channel[0][discarded_tile] = 1.0

        return np.concatenate((base_tensor, target_channel), axis=0)  # shape: (34, 34)

    def to_tensor_for_naki_v3(self, discarded_tile,
                               shanten_before, shanten_after,
                               is_tenpai_after, ukeire_after):
        """
        鳴き判断専用テンソル（強化版）: CH23-26 にシャンテン/有効牌情報を上書き

        入力チャンネル構成 (34ch = 33ch基底 + 1ch捨て牌):
          CH00-22: 従来の33ch基底 (skip_logic=True と同一)
          CH23: 鳴く前のシャンテン数   = (shanten+1) / 8.0  [0..1]  ← 上書き
          CH24: 鳴き後ベストシャンテン = (shanten+1) / 8.0  [0..1]  ← 上書き
          CH25: 鳴いたらテンパイか     = 0.0 or 1.0 (全ch一律) ← 上書き
          CH26: 鳴き後有効牌種類数     = ukeire / 34.0       [0..1]  ← 上書き
          CH27-32: 従来通り (赤ドラフラグ等)
          CH33: 捨て牌位置 (1-hot)

        引数:
          discarded_tile  : 捨て牌インデックス (0-33)
          shanten_before  : 鳴く前のシャンテン数 (-1=テンパイ, 0=1シャンテン, ...)
          shanten_after   : 鳴き後ベストシャンテン (最良打牌時)
          is_tenpai_after : 鳴き後にテンパイを取れるか (bool)
          ukeire_after    : 鳴き後の有効牌種類数 (int)
        """
        base = self.to_tensor(skip_logic=True)  # (33, 34)  CH23-26 はゼロ

        # CH23: 鳴く前シャンテン (0シャンテン=テンパイで小さい値になる)
        # shanten=-1(テンパイ)→0/8=0.0, shanten=0(1シャンテン)→1/8, ..., 7+→1.0
        base[23, :] = float(min(max(shanten_before + 1, 0), 8)) / 8.0

        # CH24: 鳴き後シャンテン
        base[24, :] = float(min(max(shanten_after + 1, 0), 8)) / 8.0

        # CH25: テンパイ取得フラグ (全タイル位置に同一値を置く)
        base[25, :] = 1.0 if is_tenpai_after else 0.0

        # CH26: 鳴き後有効牌種類数 (0〜34種)
        base[26, :] = float(min(max(ukeire_after, 0), 34)) / 34.0

        # CH33: 捨て牌位置 (1-hot)
        ch33 = np.zeros((1, 34), dtype=np.float32)
        ch33[0][discarded_tile] = 1.0

        return np.concatenate((base, ch33), axis=0)  # shape: (34, 34)
