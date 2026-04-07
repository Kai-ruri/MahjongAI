"""
benchmark.py - 自己対戦ベンチマーク

使い方:
    python benchmark.py                    # デフォルト: 100局、AIのみ
    python benchmark.py --games 200        # 200局
    python benchmark.py --vs-random        # AIとランダムエージェントの混合対戦
    python benchmark.py --output results/  # 結果の出力先ディレクトリ

出力:
    benchmark_results_<timestamp>.csv  -- 局ごとの詳細統計
    benchmark_summary_<timestamp>.txt  -- サマリーレポート
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime

import torch

from mahjong_engine import MahjongStateV5, tile_names
from mahjong_model import MahjongResNet_UltimateV3, MahjongResNet_Naki, MahjongResNet_Naki_V2
from selfplay_minimal import (
    GlobalRoundState,
    advance_round,
    apply_discard,
    apply_naki_global,
    apply_ron_score,
    apply_tsumo_score,
    build_local_state_for_player,
    check_any_ron,
    check_tsumo_agari,
    choose_discard_with_ai,
    deal_initial_hands,
    draw_tile,
    find_naki_action,
    get_player_wind,
    hand_count,
    is_tenpai_hand,
    should_declare_riichi,
    tile_names,
)

# ==========================================
# エージェント定義
# ==========================================

class AIAgent:
    """学習済みモデルを使うエージェント"""
    def __init__(self, ai_model, top_k=5):
        self.ai_model = ai_model
        self.top_k = top_k
        self.name = "AI"

    def choose_discard(self, gs, pid, extra_forbidden=None):
        return choose_discard_with_ai(
            gs, pid, self.ai_model, top_k=self.top_k, extra_forbidden=extra_forbidden
        )

    def should_riichi(self, gs, pid, discard_tile, debug_rows):
        return should_declare_riichi(gs, pid, discard_tile, debug_rows)


class RandomAgent:
    """ランダムに打牌するエージェント（ベースライン）"""
    name = "Random"

    def choose_discard(self, gs, pid, extra_forbidden=None):
        hand = gs.hands[pid]
        forbidden = extra_forbidden or []
        candidates = [i for i in range(34) if hand[i] > 0 and i not in forbidden]
        if not candidates:
            candidates = [i for i in range(34) if hand[i] > 0]
        tile = random.choice(candidates)
        debug_rows = [{"tile_idx": tile, "final_score": 0.0, "risk_value": 0.0,
                       "after_shanten": None, "ukeire_count": None,
                       "shape_bonus": 0.0, "route_bonus": 0.0,
                       "shape_reasons": [], "route_reasons": {}, "mode": "random"}]
        return tile, debug_rows

    def should_riichi(self, gs, pid, discard_tile, debug_rows):
        return False  # ランダムエージェントはリーチしない


class SimpleRuleAgent:
    """簡単なルールベースエージェント（シャンテン最小化）"""
    name = "Rule"

    def choose_discard(self, gs, pid, extra_forbidden=None):
        from hybrid_inference import calculate_shanten_unified
        hand = gs.hands[pid]
        forbidden = extra_forbidden or []
        candidates = [i for i in range(34) if hand[i] > 0 and i not in forbidden]
        if not candidates:
            candidates = [i for i in range(34) if hand[i] > 0]

        best_tile = candidates[0]
        best_shanten = 999
        local_state = build_local_state_for_player(gs, pid)

        for t in candidates:
            temp = hand.copy()
            temp[t] -= 1
            sh = calculate_shanten_unified(local_state, temp)
            if sh < best_shanten:
                best_shanten = sh
                best_tile = t

        debug_rows = [{"tile_idx": best_tile, "final_score": 0.0, "risk_value": 0.0,
                       "after_shanten": best_shanten, "ukeire_count": None,
                       "shape_bonus": 0.0, "route_bonus": 0.0,
                       "shape_reasons": [], "route_reasons": {}, "mode": "rule"}]
        return best_tile, debug_rows

    def should_riichi(self, gs, pid, discard_tile, debug_rows):
        # テンパイかつ巡目が早ければリーチ
        from selfplay_minimal import is_tenpai_after_discard
        if gs.riichi_declared[pid] or len(gs.fixed_mentsu[pid]) > 0:
            return False
        if gs.scores[pid] < 1000 or gs.junme >= 17:
            return False
        return is_tenpai_after_discard(gs, pid, discard_tile)


# ==========================================
# 局統計クラス
# ==========================================

class KyokuStats:
    """1局の統計情報"""
    def __init__(self, kyoku_id, agent_names):
        self.kyoku_id = kyoku_id
        self.agent_names = agent_names  # [pid -> agent_name]
        self.result = None          # "tsumo" / "ron" / "ryuukyoku"
        self.winner = None          # pid or None
        self.loser = None           # pid or None (ron放銃者)
        self.han = 0
        self.fu = 0
        self.score_delta = {0: 0, 1: 0, 2: 0, 3: 0}
        self.scores_before = None
        self.scores_after = None
        self.riichi_by = []         # リーチしたpidリスト
        self.naki_by = []           # 鳴いたpidリスト (重複あり)
        self.naki_types = []        # 鳴きの種類リスト ("pon"/"chi")
        self.turn_count = 0
        self.dealer = 0
        self.bakaze = 0
        self.kyoku_num = 0


# ==========================================
# ゲームループ（エージェント対応版）
# ==========================================

def run_one_kyoku(agents, gs, naki_model=None, naki_ai_model=None, max_turns=200):
    """
    1局プレイして KyokuStats を返す。
    agents: list[Agent] (pid=0,1,2,3 対応)
    naki_ai_model: 鳴き評価専用のAIモデル（discarderがRandomAgentでも有効に鳴き評価するため）
    """
    stats = KyokuStats(
        kyoku_id=None,
        agent_names=[a.name for a in agents]
    )
    stats.scores_before = gs.scores.copy()
    stats.dealer = gs.dealer_pid
    stats.bakaze = gs.bakaze
    stats.kyoku_num = gs.kyoku

    turn_count = 0

    while len(gs.wall) > 0 and turn_count < max_turns:
        pid = gs.turn_pid

        drawn = draw_tile(gs, pid)
        if drawn is None:
            break
        gs.last_draw[pid] = drawn

        # ツモ和了判定
        is_agari, agari_result = check_tsumo_agari(gs, pid, gs.hands[pid].copy(), drawn)
        if is_agari:
            score_movements = apply_tsumo_score(gs, pid, agari_result)
            stats.result = "tsumo"
            stats.winner = pid
            stats.han = agari_result.get("han", 0)
            stats.fu = agari_result.get("fu", 0)
            stats.score_delta = score_movements["payments"]
            for rp in stats.riichi_by:
                stats.score_delta[rp] -= 1000
            stats.scores_after = gs.scores.copy()
            dealer_continues = (pid == gs.dealer_pid)
            gs = advance_round(gs, dealer_continues=dealer_continues)
            turn_count += 1
            stats.turn_count = turn_count
            return stats, gs

        # リーチ中はツモ切り
        if gs.riichi_declared[pid]:
            best_discard = drawn
            debug_rows = [{"tile_idx": drawn, "final_score": 0.0, "risk_value": 0.0,
                           "after_shanten": None, "ukeire_count": None,
                           "shape_bonus": 0.0, "route_bonus": 0.0,
                           "shape_reasons": ["riichi_locked"], "route_reasons": {}, "mode": "riichi_locked"}]
        else:
            best_discard, debug_rows = agents[pid].choose_discard(gs, pid)

        # リーチ宣言
        declare_riichi = agents[pid].should_riichi(gs, pid, best_discard, debug_rows)
        if declare_riichi and not gs.riichi_declared[pid]:
            gs.riichi_declared[pid] = True
            gs.scores[pid] -= 1000
            gs.riichi_sticks += 1
            stats.riichi_by.append(pid)

        apply_discard(gs, pid, best_discard)

        # ロン判定
        ron_pid, ron_result = check_any_ron(gs, pid, best_discard)
        if ron_pid is not None:
            score_movements = apply_ron_score(gs, ron_pid, pid, ron_result)
            stats.result = "ron"
            stats.winner = ron_pid
            stats.loser = pid
            stats.han = ron_result.get("han", 0)
            stats.fu = ron_result.get("fu", 0)
            stats.score_delta = score_movements["payments"]
            for rp in stats.riichi_by:
                stats.score_delta[rp] -= 1000
            stats.scores_after = gs.scores.copy()
            dealer_continues = (ron_pid == gs.dealer_pid)
            gs = advance_round(gs, dealer_continues=dealer_continues)
            turn_count += 1
            stats.turn_count = turn_count
            return stats, gs

        # ポン/チー判定（naki_ai_modelを優先、なければdiscarderのai_modelを使用）
        _naki_eval_model = naki_ai_model if naki_ai_model is not None else (
            agents[pid].ai_model if hasattr(agents[pid], 'ai_model') else None
        )
        naki_result = find_naki_action(gs, pid, best_discard, naki_model, _naki_eval_model)
        if naki_result is not None:
            naki_pid, naki_type, consumed_tiles, forbidden = naki_result
            apply_naki_global(gs, naki_pid, naki_type, consumed_tiles, best_discard)
            stats.naki_by.append(naki_pid)
            stats.naki_types.append(naki_type)

            naki_discard, _ = agents[naki_pid].choose_discard(gs, naki_pid, extra_forbidden=forbidden)
            apply_discard(gs, naki_pid, naki_discard)

            # 鳴き後のロン判定
            ron_pid2, ron_result2 = check_any_ron(gs, naki_pid, naki_discard)
            if ron_pid2 is not None:
                score_movements2 = apply_ron_score(gs, ron_pid2, naki_pid, ron_result2)
                stats.result = "ron"
                stats.winner = ron_pid2
                stats.loser = naki_pid
                stats.han = ron_result2.get("han", 0)
                stats.fu = ron_result2.get("fu", 0)
                stats.score_delta = score_movements2["payments"]
                for rp in stats.riichi_by:
                    stats.score_delta[rp] -= 1000
                stats.scores_after = gs.scores.copy()
                dealer_continues2 = (ron_pid2 == gs.dealer_pid)
                gs = advance_round(gs, dealer_continues=dealer_continues2)
                turn_count += 1
                stats.turn_count = turn_count
                return stats, gs

            gs.turn_pid = (naki_pid + 1) % 4
            turn_count += 1
            continue

        # 次巡へ
        gs.turn_pid = (gs.turn_pid + 1) % 4
        if gs.turn_pid == gs.dealer_pid:
            gs.junme += 1
        turn_count += 1

    # 流局
    dealer_tenpai = is_tenpai_hand(gs, gs.dealer_pid)
    tenpai_players = [p for p in range(4) if is_tenpai_hand(gs, p)]
    noten_players = [p for p in range(4) if p not in tenpai_players]

    if 0 < len(tenpai_players) < 4:
        pay = 3000 // len(noten_players)
        recv = 3000 // len(tenpai_players)
        delta = {p: 0 for p in range(4)}
        for p in noten_players:
            gs.scores[p] -= pay
            delta[p] -= pay
        for p in tenpai_players:
            gs.scores[p] += recv
            delta[p] += recv
        stats.score_delta = delta

    for rp in stats.riichi_by:
        stats.score_delta[rp] -= 1000

    stats.result = "ryuukyoku"
    stats.scores_after = gs.scores.copy()
    stats.turn_count = turn_count

    gs = advance_round(gs, dealer_continues=dealer_tenpai, is_ryuukyoku=True)
    return stats, gs


# ==========================================
# 半荘シミュレーション
# ==========================================

def run_hanchan(agents, seed=0, naki_model=None, naki_ai_model=None, max_kyoku=8, max_turns_per_kyoku=200, initial_dealer=0):
    """
    半荘（最大8局）を通してプレイし、全局の統計リストを返す。
    initial_dealer: 東1局の起家pid（ゲームごとにローテーションして座席バイアスを除去）
    """
    random.seed(seed)
    gs = deal_initial_hands(seed=seed, dealer_pid=initial_dealer)
    all_stats = []
    kyoku_id = 0

    while not gs.game_over and kyoku_id < max_kyoku:
        stats, gs = run_one_kyoku(agents, gs, naki_model=naki_model, naki_ai_model=naki_ai_model, max_turns=max_turns_per_kyoku)
        stats.kyoku_id = kyoku_id
        all_stats.append(stats)
        kyoku_id += 1

        if gs.game_over:
            break

    return all_stats, gs


# ==========================================
# 統計集計
# ==========================================

def aggregate_stats(all_stats, agent_names, num_games):
    """
    複数半荘の統計を集計してサマリーを返す。
    各エージェント(pid)ごとに集計。
    """
    summary = {}
    for pid in range(4):
        summary[pid] = {
            "agent": agent_names[pid],
            "pid": pid,
            "kyoku_count": 0,
            "tsumo_count": 0,
            "ron_win_count": 0,
            "houju_count": 0,       # 放銃回数
            "riichi_count": 0,
            "naki_count": 0,
            "total_score_delta": 0,
            "first_place": 0,       # 1位回数
            "second_place": 0,
            "third_place": 0,
            "fourth_place": 0,
            "ryuukyoku_tenpai": 0,
        }

    for stats in all_stats:
        for pid in range(4):
            summary[pid]["kyoku_count"] += 1
            if stats.result == "tsumo" and stats.winner == pid:
                summary[pid]["tsumo_count"] += 1
            if stats.result == "ron" and stats.winner == pid:
                summary[pid]["ron_win_count"] += 1
            if stats.result == "ron" and stats.loser == pid:
                summary[pid]["houju_count"] += 1
            if pid in stats.riichi_by:
                summary[pid]["riichi_count"] += 1
            if pid in stats.naki_by:
                summary[pid]["naki_count"] += 1
            if stats.score_delta:
                summary[pid]["total_score_delta"] += stats.score_delta.get(pid, 0)

    return summary


def compute_placement(final_scores, initial_dealer=0):
    """
    最終スコアリストから順位を返す (1-indexed)。
    同点の場合は東1局起家（initial_dealer）からの席順が近い方が上位（天鳳ルール準拠）。
    """
    # (スコア降順, 起家からの距離昇順) でソート
    sorted_pids = sorted(range(4), key=lambda p: (-final_scores[p], (p - initial_dealer) % 4))
    placements = [0] * 4
    for rank, pid in enumerate(sorted_pids):
        placements[pid] = rank + 1
    return placements


# ==========================================
# CSV出力
# ==========================================

CSV_FIELDS = [
    "game_id", "kyoku_id", "bakaze", "kyoku_num", "dealer",
    "result", "winner_pid", "winner_agent", "loser_pid", "loser_agent",
    "han", "fu", "turn_count",
    "riichi_pids", "naki_pids", "naki_types",
    "score_before_p0", "score_before_p1", "score_before_p2", "score_before_p3",
    "score_delta_p0", "score_delta_p1", "score_delta_p2", "score_delta_p3",
    "score_after_p0", "score_after_p1", "score_after_p2", "score_after_p3",
    "agent_p0", "agent_p1", "agent_p2", "agent_p3",
]


def stats_to_row(game_id, stats):
    agent_names = stats.agent_names
    sb = stats.scores_before or [0, 0, 0, 0]
    sa = stats.scores_after or [0, 0, 0, 0]
    sd = stats.score_delta or {0: 0, 1: 0, 2: 0, 3: 0}
    return {
        "game_id": game_id,
        "kyoku_id": stats.kyoku_id,
        "bakaze": stats.bakaze,
        "kyoku_num": stats.kyoku_num,
        "dealer": stats.dealer,
        "result": stats.result or "",
        "winner_pid": stats.winner if stats.winner is not None else "",
        "winner_agent": agent_names[stats.winner] if stats.winner is not None else "",
        "loser_pid": stats.loser if stats.loser is not None else "",
        "loser_agent": agent_names[stats.loser] if stats.loser is not None else "",
        "han": stats.han,
        "fu": stats.fu,
        "turn_count": stats.turn_count,
        "riichi_pids": ";".join(map(str, stats.riichi_by)),
        "naki_pids": ";".join(map(str, stats.naki_by)),
        "naki_types": ";".join(stats.naki_types),
        "score_before_p0": sb[0], "score_before_p1": sb[1],
        "score_before_p2": sb[2], "score_before_p3": sb[3],
        "score_delta_p0": sd.get(0, 0), "score_delta_p1": sd.get(1, 0),
        "score_delta_p2": sd.get(2, 0), "score_delta_p3": sd.get(3, 0),
        "score_after_p0": sa[0], "score_after_p1": sa[1],
        "score_after_p2": sa[2], "score_after_p3": sa[3],
        "agent_p0": agent_names[0], "agent_p1": agent_names[1],
        "agent_p2": agent_names[2], "agent_p3": agent_names[3],
    }


# ==========================================
# サマリーレポート生成
# ==========================================

def format_summary_report(summary, agent_names, total_games, total_kyoku,
                           placement_counts, elapsed_sec, config_desc):
    lines = []
    lines.append("=" * 60)
    lines.append("MahjongAI Benchmark Summary")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Config: {config_desc}")
    lines.append(f"Games: {total_games} | Total kyoku: {total_kyoku} | Time: {elapsed_sec:.1f}s")
    lines.append("=" * 60)

    for pid in range(4):
        s = summary[pid]
        k = max(s["kyoku_count"], 1)
        win_total = s["tsumo_count"] + s["ron_win_count"]
        lines.append(f"\n[P{pid}] {s['agent']}")
        lines.append(f"  局数        : {k}")
        lines.append(f"  和了率      : {win_total / k * 100:.1f}% "
                     f"(ツモ {s['tsumo_count'] / k * 100:.1f}% + ロン {s['ron_win_count'] / k * 100:.1f}%)")
        lines.append(f"  放銃率      : {s['houju_count'] / k * 100:.1f}%")
        lines.append(f"  リーチ率    : {s['riichi_count'] / k * 100:.1f}%")
        lines.append(f"  鳴き率      : {s['naki_count'] / k * 100:.1f}%")
        lines.append(f"  平均得点差  : {s['total_score_delta'] / k:+.0f} pt/局")

        if total_games > 0 and pid in placement_counts:
            pc = placement_counts[pid]
            total_g = max(total_games, 1)
            lines.append(f"  順位分布    : 1位 {pc[1] / total_g * 100:.1f}%  "
                         f"2位 {pc[2] / total_g * 100:.1f}%  "
                         f"3位 {pc[3] / total_g * 100:.1f}%  "
                         f"4位 {pc[4] / total_g * 100:.1f}%")
            avg_rank = (1 * pc[1] + 2 * pc[2] + 3 * pc[3] + 4 * pc[4]) / total_g
            lines.append(f"  平均順位    : {avg_rank:.3f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ==========================================
# メイン
# ==========================================

def load_ai_model(path="mahjong_ultimate_ai_v5_master.pth"):
    model = MahjongResNet_UltimateV3()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"  Loaded AI model: {path}")
    else:
        print(f"  WARNING: {path} not found, using random weights")
    model.eval()
    return model


def load_naki_model(path="mahjong_naki_model_master.pth"):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, naki uses heuristic only")
        return None
    # V2(28ch) を先に試みて、失敗したら V1(26ch) にフォールバック
    for model_cls, label in [(MahjongResNet_Naki_V2, "v2(28ch)"), (MahjongResNet_Naki, "v1(26ch)")]:
        try:
            model = model_cls()
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            print(f"  Loaded naki model: {path} [{label}]")
            return model
        except Exception:
            continue
    print(f"  WARNING: {path} could not be loaded, naki uses heuristic only")
    return None


def main():
    parser = argparse.ArgumentParser(description="MahjongAI Benchmark")
    parser.add_argument("--games", type=int, default=50, help="半荘数 (default: 50)")
    parser.add_argument("--max-kyoku", type=int, default=8, help="1半荘の最大局数 (default: 8)")
    parser.add_argument("--vs-random", action="store_true", help="AI 1人 + Random 3人で対戦")
    parser.add_argument("--vs-rule", action="store_true", help="AI 1人 + Rule 3人で対戦")
    parser.add_argument("--all-random", action="store_true", help="全員Randomで対戦（ベースライン測定）")
    parser.add_argument("--all-rule", action="store_true", help="全員Ruleで対戦（ルールベースライン）")
    parser.add_argument("--output", type=str, default=".", help="結果出力ディレクトリ (default: .)")
    parser.add_argument("--ai-model", type=str, default="discard_b1_best.pth")
    parser.add_argument("--naki-model", type=str, default="mahjong_naki_model_master.pth")
    parser.add_argument("--no-naki", action="store_true", help="ポン/チーを無効化")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード起点 (default: 0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output, f"benchmark_results_{timestamp}.csv")
    summary_path = os.path.join(args.output, f"benchmark_summary_{timestamp}.txt")

    print("=" * 60)
    print("MahjongAI Benchmark")
    print("=" * 60)

    # モデルロード
    print("\nLoading models...")
    ai_model = load_ai_model(args.ai_model)
    naki_model = None if args.no_naki else load_naki_model(args.naki_model)

    # エージェント構成
    ai_agent = AIAgent(ai_model, top_k=5)
    random_agent = RandomAgent()
    rule_agent = SimpleRuleAgent()

    if args.all_random:
        agents = [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]
        config_desc = "AllRandom"
    elif args.all_rule:
        agents = [SimpleRuleAgent(), SimpleRuleAgent(), SimpleRuleAgent(), SimpleRuleAgent()]
        config_desc = "AllRule"
    elif args.vs_random:
        agents = [AIAgent(ai_model), RandomAgent(), RandomAgent(), RandomAgent()]
        config_desc = "AI vs Random x3"
    elif args.vs_rule:
        agents = [AIAgent(ai_model), SimpleRuleAgent(), SimpleRuleAgent(), SimpleRuleAgent()]
        config_desc = "AI vs Rule x3"
    else:
        agents = [AIAgent(ai_model), AIAgent(ai_model), AIAgent(ai_model), AIAgent(ai_model)]
        config_desc = "AllAI"

    agent_names = [a.name for a in agents]
    print(f"\nConfig: {config_desc}")
    print(f"Agents: {agent_names}")
    print(f"Games: {args.games} x {args.max_kyoku} kyoku max")
    print(f"Naki: {'disabled' if args.no_naki else 'enabled'}")
    print()

    # ベンチマーク実行
    all_game_stats = []
    placement_counts = {pid: {1: 0, 2: 0, 3: 0, 4: 0} for pid in range(4)}
    all_summary = None

    start_time = time.time()

    for game_id in range(args.games):
        seed = args.seed + game_id
        initial_dealer = game_id % 4  # ゲームごとに起家をローテーション
        all_stats, final_gs = run_hanchan(
            agents, seed=seed, naki_model=naki_model,
            naki_ai_model=None if args.no_naki else ai_model,
            max_kyoku=args.max_kyoku,
            initial_dealer=initial_dealer,
        )

        # 最終順位を記録
        if final_gs.scores:
            placements = compute_placement(final_gs.scores, initial_dealer=initial_dealer)
            for pid in range(4):
                placement_counts[pid][placements[pid]] += 1

        all_game_stats.extend(all_stats)

        # 進捗表示
        if (game_id + 1) % 10 == 0 or game_id == args.games - 1:
            elapsed = time.time() - start_time
            print(f"  Progress: {game_id + 1}/{args.games} games | "
                  f"Kyoku: {len(all_game_stats)} | "
                  f"Elapsed: {elapsed:.1f}s")

    elapsed_total = time.time() - start_time

    # 統計集計
    all_summary = aggregate_stats(all_game_stats, agent_names, args.games)

    # サマリーレポート出力（先に行い、CSV失敗でも結果が残るようにする）
    report = format_summary_report(
        all_summary, agent_names, args.games, len(all_game_stats),
        placement_counts, elapsed_total, config_desc
    )
    print(report)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Summary saved: {summary_path}")

    # CSV出力
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for game_id_offset, stats in enumerate(all_game_stats):
                row = stats_to_row(stats.kyoku_id // args.max_kyoku, stats)
                writer.writerow(row)
        print(f"CSV saved: {csv_path}")
    except OSError as e:
        print(f"CSV save failed (summary already saved): {e}")


if __name__ == "__main__":
    main()
