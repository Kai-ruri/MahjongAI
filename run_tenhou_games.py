"""
run_tenhou_games.py
ゲストIDを1回取得し、同じIDで複数試合を連続実行する。
REGIDQ レート制限 (res=1006) を回避するため。
"""
import argparse
import sys
import time

import torch

from tenhou_client import TenhouBot, load_models


def main():
    parser = argparse.ArgumentParser(description="天鳳 複数対局ランナー")
    parser.add_argument("--games",    type=int,   default=30,   help="対局数")
    parser.add_argument("--username", default=None,             help="ゲストID（省略時は自動取得）")
    parser.add_argument("--game-type",type=int,   default=9,    help="対局タイプ")
    parser.add_argument("--interval", type=float, default=30.0, help="対局間の待機秒数")
    args = parser.parse_args()

    ai_model, naki_model = load_models()

    # ゲストIDを1回だけ取得
    username = args.username
    if username is None:
        print("ゲストID取得中...")
        tmp_bot = TenhouBot(ai_model=ai_model, naki_model=naki_model)
        username = tmp_bot._get_guest_id()
        print(f"ゲストID取得: {username}")

    print(f"\n{args.games} 試合を ID={username} で実行します\n")

    results = []
    for i in range(1, args.games + 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{args.games}] 対局開始")
        print(f"{'='*50}")

        try:
            bot = TenhouBot(
                username=username,
                ai_model=ai_model,
                naki_model=naki_model,
                game_type=args.game_type,
                verbose=True,
            )
            bot.run()
            results.append("ok")
        except Exception as e:
            print(f"[!] エラー: {e}")
            results.append(f"error: {e}")

        if i < args.games:
            print(f"\n次の対局まで {args.interval} 秒待機...")
            time.sleep(args.interval)

    # 結果サマリー
    ok_count = sum(1 for r in results if r == "ok")
    print(f"\n{'='*50}")
    print(f"全{args.games}試合完了: 成功 {ok_count} / 失敗 {args.games - ok_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
