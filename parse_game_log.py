"""
parse_game_log.py
天鳳ボット出力ファイルから対局ログを解析してCSV出力する。

設計方針:
  - ゲームセッション = 前の「最終結果」から次の「最終結果」までの全データ
  - 1セッションが複数の接続試行にまたがる場合も同じ game_no に統合
  - 同一局(kyoku)が再接続で重複する場合は最初のINITを採用
  - round_no はセッション内で通し番号（試行ごとにリセットしない）

出力:
  game_log_summary.csv  … 試合ごとの最終結果
  game_log_rounds.csv   … 局ごとの結果（全プレイヤーの打牌含む）
  game_log_actions.csv  … 局内の全アクション
"""
import re
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTPUT_FILE = r'C:\Users\PC2502\AppData\Local\Temp\claude\G---------MahjongAI\ca196ab5-6bd4-4c05-8abb-382d446b4346\tasks\bmc6oqe5s.output'
OUT_DIR = Path(r'G:\マイドライブ\MahjongAI')

# ─── タイルインデックス → 名前変換 ─────────────────────────────────
TILE_NAMES = [
    '1m','2m','3m','4m','5m','6m','7m','8m','9m',
    '1p','2p','3p','4p','5p','6p','7p','8p','9p',
    '1s','2s','3s','4s','5s','6s','7s','8s','9s',
    '東','南','西','北','白','発','中',
]
RED_FIVE = {16: '5mr', 52: '5pr', 88: '5sr'}

def tile_name(idx):
    if idx in RED_FIVE:
        return RED_FIVE[idx]
    t = idx // 4
    return TILE_NAMES[t] if 0 <= t < 34 else f'?{idx}'


def load_text(path):
    with open(path, 'rb') as f:
        return f.read().decode('cp932', errors='replace').splitlines()


YAKU = {
    0:'門前清自摸和',1:'立直',2:'一発',3:'槍槓',4:'嶺上開花',
    5:'海底摸月',6:'河底撈魚',7:'平和',8:'断么九',9:'一盃口',
    10:'自風東',11:'自風南',12:'自風西',13:'自風北',
    14:'場風東',15:'場風南',16:'場風西',17:'場風北',
    18:'白',19:'発',20:'中',21:'両立直',22:'七対子',
    23:'国士無双',24:'四暗刻',25:'大三元',26:'小四喜',
    27:'大四喜',28:'字一色',29:'緑一色',30:'清老頭',
    31:'九蓮宝燈',32:'四槓子',33:'天和',34:'地和',
    52:'ドラ',53:'裏ドラ',54:'赤ドラ',
}


def parse(lines):
    games_summary = []
    rounds_data   = []
    actions_data  = []

    # ── ゲームセッション管理 ──────────────────────────────────────
    # 「最終結果」が来るまで溜める。試行番号をまたいでも同一セッション扱い。
    session_rounds  = []   # このセッションの局リスト（flush後にgame_noを付与）
    session_actions = []   # このセッションのアクションリスト
    session_kyoku_seen = set()   # 重複INIT（再接続）スキップ用
    game_no       = 0
    session_round_no = 0   # セッション内局番（試行をまたいで通し）

    # ── 現在の接続試行 ────────────────────────────────────────────
    attempt_no   = 0
    my_seat      = 0
    game_partial = False   # 途中参加フラグ
    saikai_seen  = False

    # ── 現在の局 ──────────────────────────────────────────────────
    cur_round     = None
    cur_kyoku_key = None    # 重複判定キー（例: '東2局0本場'）
    ai_discards   = []
    ai_draws      = []
    seat_discards = {0:[], 1:[], 2:[], 3:[]}
    actions_buf   = []   # cur_round 確定前のアクションバッファ

    def flush_round():
        """cur_round を session_rounds に追加"""
        nonlocal cur_round, cur_kyoku_key, ai_discards, ai_draws
        nonlocal seat_discards, actions_buf
        if cur_round is not None:
            cur_round['ai_discards'] = ' '.join(ai_discards)
            cur_round['ai_draws']    = ' '.join(ai_draws)
            for s in range(4):
                cur_round[f'discards_s{s}'] = ' '.join(seat_discards[s])
            session_rounds.append(cur_round)
            session_actions.extend(actions_buf)
        cur_round     = None
        cur_kyoku_key = None
        ai_discards   = []
        ai_draws      = []
        seat_discards = {0:[], 1:[], 2:[], 3:[]}
        actions_buf   = []

    def flush_session(final_attempt, partial):
        """最終結果が来たらセッション全局に game_no を割り当て"""
        nonlocal game_no, session_rounds, session_actions
        nonlocal session_kyoku_seen, session_round_no
        game_no += 1
        # round_no を1から付け直す
        for i, r in enumerate(session_rounds):
            r['game_no']   = game_no
            r['round_no']  = i + 1
        rounds_data.extend(session_rounds)
        actions_data.extend(session_actions)
        session_rounds       = []
        session_actions      = []
        session_kyoku_seen   = set()
        session_round_no     = 0

    # ── メインループ ──────────────────────────────────────────────
    for line in lines:
        line = line.strip()

        # 試行番号
        m = re.search(r'試行 #(\d+)', line)
        if m:
            attempt_no  = int(m.group(1))
            saikai_seen = False
            # round_no はリセットしない（セッションをまたいで連続）
            flush_round()   # 未完局があれば保存

        # SAIKAI（最初の1回だけ解析して途中参加フラグを立てる）
        m = re.search(r'"tag":"SAIKAI".*?"sc":"([\d,]+)"', line)
        if m and not saikai_seen:
            saikai_seen = True
            sc_parts = [int(x) for x in m.group(1).split(',')]
            scores = [sc_parts[i * 2] for i in range(4) if i * 2 < len(sc_parts)]
            if all(s == 250 for s in scores):
                # 全員 25000 = 新規ゲーム開始。未完のセッションをリセット
                flush_round()
                session_rounds.clear()
                session_actions.clear()
                session_kyoku_seen.clear()
                session_round_no = 0
                game_partial = False
            else:
                game_partial = True   # 途中参加

        # 自分の席
        m = re.search(r'自分の席: (\d)', line)
        if m:
            my_seat = int(m.group(1))

        # ── 局開始（INIT）────────────────────────────────────────
        m = re.match(
            r'\[Bot\] INIT ([東南])(\d+)局(\d+)本場 '
            r'親=(\d+) 自席=(\d+) ドラ表示=(\S+) 手牌=(\[.+\])',
            line
        )
        if m:
            bz     = m.group(1)
            kyoku  = int(m.group(2))
            honba  = int(m.group(3))
            dealer = int(m.group(4))
            seat   = int(m.group(5))
            dora   = m.group(6)
            hand   = m.group(7)
            kyoku_key = f'{bz}{kyoku}局{honba}本場'
            my_seat = seat

            if kyoku_key in session_kyoku_seen:
                # 再接続による同一局の再出現 → 前の局を残してスキップ
                flush_round()
                cur_round = None   # この局は無視
                cur_kyoku_key = None
            else:
                flush_round()
                session_kyoku_seen.add(kyoku_key)
                session_round_no += 1
                cur_kyoku_key = kyoku_key
                cur_round = {
                    'game_no':       0,
                    'attempt_no':    attempt_no,
                    'round_no':      0,   # flush_session で振り直す
                    'global_round':  len(rounds_data) + len(session_rounds) + 1,
                    'kyoku':         kyoku_key,
                    'dealer':        dealer,
                    'my_seat':       my_seat,
                    'is_dealer':     (my_seat == dealer),
                    'dora':          dora,
                    'init_hand':     hand,
                    'ai_riichi':     '',
                    'result':        '',
                    'winner_seat':   '',
                    'loser_seat':    '',
                    'yaku':          '',
                    'score_gain':    '',
                    'ai_discards':   '',
                    'ai_draws':      '',
                    'discards_s0':   '',
                    'discards_s1':   '',
                    'discards_s2':   '',
                    'discards_s3':   '',
                    'opponent_riichi': '',
                }

        # ── 最終結果（セッション終了）────────────────────────────
        # ※ cur_round が None でも処理が必要なので continue の前に置く
        m = re.match(r'\[Bot\] 最終結果: (.+)', line)
        if m:
            flush_round()
            parts = m.group(1).split(',')
            if len(parts) >= 8:
                try:
                    scores   = [int(parts[i * 2]) * 100 for i in range(4)]
                    umas     = [float(parts[i * 2 + 1]) for i in range(4)]
                    ai_score = scores[my_seat]
                    ai_uma   = umas[my_seat]
                    ai_rank  = sum(1 for s in scores if s > ai_score) + 1
                    flush_session(attempt_no, game_partial)
                    games_summary.append({
                        'game_no':    game_no,
                        'attempt_no': attempt_no,
                        'my_seat':    my_seat,
                        'is_partial': game_partial,
                        'ai_score':   ai_score,
                        'ai_pt':      ai_uma,
                        'ai_rank':    ai_rank,
                        'score_s0':   scores[0],
                        'score_s1':   scores[1],
                        'score_s2':   scores[2],
                        'score_s3':   scores[3],
                        'pt_s0':      umas[0],
                        'pt_s1':      umas[1],
                        'pt_s2':      umas[2],
                        'pt_s3':      umas[3],
                    })
                    game_partial = False
                except Exception:
                    pass
            continue   # 最終結果行はここで処理完了

        if cur_round is None:
            continue   # 無視中の局はスキップ

        # ── 相手打牌（raw JSON: E/F/G タグ）──────────────────────
        m = re.match(r'← \{"tag":"([EFG])(\d+)"(?:,"t":"\d+")?\}', line)
        if m:
            rel      = {'E': 1, 'F': 2, 'G': 3}[m.group(1)]
            abs_seat = (my_seat + rel) % 4
            idx      = int(m.group(2))
            tname    = tile_name(idx)
            seat_discards[abs_seat].append(tname)
            actions_buf.append({
                'attempt_no': attempt_no, 'round': cur_round['round_no'],
                'kyoku': cur_kyoku_key,
                'who': f'席{abs_seat}', 'type': '打牌', 'tile': tname, 'detail': ''
            })

        # ── AI打牌 ───────────────────────────────────────────────
        m = re.match(r'\[Bot\] 打牌: (\S+)', line)
        if m:
            tile = m.group(1)
            ai_discards.append(tile)
            seat_discards[my_seat].append(tile)
            actions_buf.append({
                'attempt_no': attempt_no, 'round': cur_round['round_no'],
                'kyoku': cur_kyoku_key,
                'who': f'AI(席{my_seat})', 'type': '打牌', 'tile': tile, 'detail': ''
            })

        # ── AIツモ ───────────────────────────────────────────────
        m = re.match(r'\[Bot\] ツモ: (\S+)\s+手牌:', line)
        if m:
            tile = m.group(1)
            ai_draws.append(tile)
            actions_buf.append({
                'attempt_no': attempt_no, 'round': cur_round['round_no'],
                'kyoku': cur_kyoku_key,
                'who': f'AI(席{my_seat})', 'type': 'ツモ', 'tile': tile, 'detail': ''
            })

        # ── AIリーチ宣言 ─────────────────────────────────────────
        m = re.match(r'\[Bot\] リーチ宣言 捨て牌: (\S+)', line)
        if m:
            tile = m.group(1)
            cur_round['ai_riichi'] = tile
            actions_buf.append({
                'attempt_no': attempt_no, 'round': cur_round['round_no'],
                'kyoku': cur_kyoku_key,
                'who': f'AI(席{my_seat})', 'type': 'リーチ宣言', 'tile': tile, 'detail': ''
            })

        # ── 他家リーチ ───────────────────────────────────────────
        m = re.match(r'\[Bot\] リーチ: 席(\d)', line)
        if m:
            who  = int(m.group(1))
            prev = cur_round.get('opponent_riichi', '')
            cur_round['opponent_riichi'] = (prev + f' 席{who}').strip()
            actions_buf.append({
                'attempt_no': attempt_no, 'round': cur_round['round_no'],
                'kyoku': cur_kyoku_key,
                'who': f'席{who}', 'type': 'リーチ宣言', 'tile': '', 'detail': ''
            })

        # ── 鳴き（他家含む）─────────────────────────────────────
        m = re.match(r'\[Bot\] 鳴き: 席(\d) (chi|pon|kan|ankan) (\[.+\])', line)
        if m:
            who, ntype, tiles = int(m.group(1)), m.group(2), m.group(3)
            label = {'chi':'チー','pon':'ポン','kan':'カン','ankan':'暗カン'}.get(ntype, ntype)
            actions_buf.append({
                'attempt_no': attempt_no, 'round': cur_round['round_no'],
                'kyoku': cur_kyoku_key,
                'who': f'席{who}', 'type': label, 'tile': tiles, 'detail': ''
            })

        # ── AIポン/チー ──────────────────────────────────────────
        m = re.match(r'\[Bot\] (ポン|チー): (\S+)', line)
        if m:
            ntype, tile = m.group(1), m.group(2)
            actions_buf.append({
                'attempt_no': attempt_no, 'round': cur_round['round_no'],
                'kyoku': cur_kyoku_key,
                'who': f'AI(席{my_seat})', 'type': ntype, 'tile': tile, 'detail': 'AI鳴き'
            })

        # ── 和了 ─────────────────────────────────────────────────
        m = re.match(r'\[Bot\] 和了: 席(\d)\s+スコア: (\[.+\])', line)
        if m:
            winner = int(m.group(1))
            cur_round['winner_seat'] = winner
            if cur_round['result'] == '':
                cur_round['result'] = '和了'

        # ── 流局 ─────────────────────────────────────────────────
        m = re.match(r'\[Bot\] (流局|流局→対局終了)', line)
        if m:
            cur_round['result'] = '流局'

        # ── AGARI raw JSON ────────────────────────────────────────
        if '"tag":"AGARI"' in line:
            raw = re.search(r'\{.+', line)
            if raw:
                rj = raw.group()
                mw = re.search(r'"who":"?(\d+)"?', rj)
                mf = re.search(r'"fromWho":"?(\d+)"?', rj)
                mt = re.search(r'"ten":"([\d,]+)"', rj)
                my = re.search(r'"yaku":"([\d,]+)"', rj)
                if mw:
                    winner   = int(mw.group(1))
                    from_who = int(mf.group(1)) if mf else winner
                    is_tsumo = (winner == from_who)
                    cur_round['result']      = 'ツモ' if is_tsumo else 'ロン'
                    cur_round['winner_seat'] = winner
                    cur_round['loser_seat']  = '' if is_tsumo else from_who
                if mt:
                    tp = mt.group(1).split(',')
                    cur_round['score_gain'] = tp[1] if len(tp) > 1 else ''
                if my:
                    yp = my.group(1).split(',')
                    yakus = []
                    for j in range(0, len(yp) - 1, 2):
                        try:
                            yi = int(yp[j]); yh = int(yp[j + 1])
                            yakus.append(f'{YAKU.get(yi, f"役{yi}")}({yh})')
                        except Exception:
                            pass
                    if yakus:
                        cur_round['yaku'] = '/'.join(yakus)

    flush_round()
    return games_summary, rounds_data, actions_data


def main():
    print("ログ解析中...")
    lines = load_text(OUTPUT_FILE)
    games_summary, rounds_data, actions_data = parse(lines)

    n_games  = len(games_summary)
    n_rounds = len(rounds_data)
    n_acts   = len(actions_data)
    print(f"完了試合: {n_games}")
    print(f"局数:     {n_rounds}  (平均 {n_rounds/n_games:.1f} 局/試合)" if n_games else f"局数: {n_rounds}")
    print(f"アクション数: {n_acts}")

    # ─── game_log_summary.csv ────────────────────────────────
    p = OUT_DIR / 'game_log_summary.csv'
    fields = ['game_no','attempt_no','my_seat','is_partial','ai_score','ai_pt','ai_rank',
              'score_s0','score_s1','score_s2','score_s3',
              'pt_s0','pt_s1','pt_s2','pt_s3']
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(games_summary)
    print(f"-> {p}")

    # ─── game_log_rounds.csv ─────────────────────────────────
    p = OUT_DIR / 'game_log_rounds.csv'
    fields = ['game_no','attempt_no','round_no','global_round','kyoku',
              'dealer','my_seat','is_dealer','dora','init_hand',
              'ai_riichi','opponent_riichi',
              'result','winner_seat','loser_seat','score_gain','yaku',
              'ai_discards','ai_draws',
              'discards_s0','discards_s1','discards_s2','discards_s3']
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rounds_data)
    print(f"-> {p}")

    # ─── game_log_actions.csv ────────────────────────────────
    p = OUT_DIR / 'game_log_actions.csv'
    # round_no を正しい値に更新（flush_session で振り直した値を反映）
    round_map = {(r['game_no'], r['kyoku']): r['round_no'] for r in rounds_data}
    for a in actions_data:
        key = (0, a['kyoku'])   # まず game_no=0 で探す（未確定の場合）
        # actions には game_no が入っていないので kyoku で照合
        # rounds_data から逆引き
        pass
    fields = ['attempt_no','round','kyoku','who','type','tile','detail']
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(actions_data)
    print(f"-> {p}")

    # ─── サマリー表示 ────────────────────────────────────────
    complete_games = [g for g in games_summary if not g['is_partial']]
    partial_games  = [g for g in games_summary if g['is_partial']]
    print(f'\n=== 試合サマリー ({len(complete_games)}完全 / {len(partial_games)}途中参加) ===')
    total_pt  = sum(g['ai_pt'] for g in games_summary)
    rank_dist = {}
    for g in games_summary:
        r = g['ai_rank']
        rank_dist[r] = rank_dist.get(r, 0) + 1
        flag = ' [途中]' if g['is_partial'] else ''
        print(f"  Game{g['game_no']:2d}{flag}: {g['ai_score']:>8}点 {g['ai_pt']:>+6.1f}pt  {g['ai_rank']}位  ({g['round_no_count'] if 'round_no_count' in g else '?'}局)" if False
              else f"  Game{g['game_no']:2d}{flag}: {g['ai_score']:>8}点 {g['ai_pt']:>+6.1f}pt  {g['ai_rank']}位")
    avg_rank = sum(g['ai_rank'] for g in games_summary) / n_games if n_games else 0
    print(f"\n  平均PT: {total_pt/n_games:+.2f}  平均順位: {avg_rank:.2f}" if n_games else "")
    print(f"  順位分布: " + "  ".join(f"{k}位={v}" for k, v in sorted(rank_dist.items())))

    # 局統計
    total_r = len(rounds_data)
    if total_r > 0:
        ai_agari  = sum(1 for r in rounds_data
                        if r['result'] in ('ツモ','ロン','和了')
                        and str(r['winner_seat']) == str(r['my_seat']))
        ai_houju  = sum(1 for r in rounds_data
                        if r['result'] == 'ロン'
                        and str(r['loser_seat']) == str(r['my_seat']))
        ai_riichi = sum(1 for r in rounds_data if r['ai_riichi'])
        ryuukyoku = sum(1 for r in rounds_data if r['result'] == '流局')
        print(f'\n=== 局統計 (計{total_r}局) ===')
        print(f'  AI和了:  {ai_agari:3d} ({ai_agari/total_r*100:.1f}%)')
        print(f'  AI放銃:  {ai_houju:3d} ({ai_houju/total_r*100:.1f}%)')
        print(f'  AIリーチ:{ai_riichi:3d} ({ai_riichi/total_r*100:.1f}%)')
        print(f'  流局:    {ryuukyoku:3d} ({ryuukyoku/total_r*100:.1f}%)')

        opp_d = sum(1 for a in actions_data if a['type'] == '打牌' and not a['who'].startswith('AI'))
        ai_d  = sum(1 for a in actions_data if a['type'] == '打牌' and a['who'].startswith('AI'))
        print(f'\n  アクション内訳: AI打牌={ai_d}  相手打牌={opp_d}')

        # 局数/試合 分布
        from collections import Counter
        rpc = Counter(r['game_no'] for r in rounds_data)
        counts = sorted(rpc.values())
        print(f'\n  局数/試合: min={min(counts)} median={counts[len(counts)//2]} max={max(counts)}')

        # 役別集計
        yaku_count = {}
        for r in rounds_data:
            if r['yaku'] and str(r['winner_seat']) == str(r['my_seat']):
                for y in r['yaku'].split('/'):
                    name = re.sub(r'\(\d+\)', '', y)
                    yaku_count[name] = yaku_count.get(name, 0) + 1
        if yaku_count:
            print('\n  AI和了役トップ:')
            for y, c in sorted(yaku_count.items(), key=lambda x: -x[1])[:10]:
                print(f'    {y}: {c}回')


if __name__ == '__main__':
    main()
