"""
run_tenhou_games.py
ゲストIDを1回取得し、同じIDで複数試合を連続実行する。
REGIDQ レート制限 (res=1006) を回避するため。

--games で指定した数の「対局完了」が達成されるまで試みを繰り返す。
マッチング不成立（即切断）はカウントしない。
"""
import argparse
import time

from tenhou_client import TenhouBot, load_models


def main():
    parser = argparse.ArgumentParser(description="天鳳 複数対局ランナー")
    parser.add_argument("--games",    type=int,   default=30,   help="完了させたい対局数")
    parser.add_argument("--username", default=None,             help="ゲストID（省略時は自動取得）")
    parser.add_argument("--game-type",type=int,   default=9,    help="対局タイプ")
    parser.add_argument("--interval", type=float, default=30.0, help="対局間の待機秒数（秒）")
    parser.add_argument("--max-attempts", type=int, default=300, help="最大試行回数（無限ループ防止）")
    args = parser.parse_args()

    ai_model, naki_model = load_models()

    # ゲストIDを1回だけ取得
    username = args.username
    if username is None:
        print("ゲストID取得中...")
        tmp_bot = TenhouBot(ai_model=ai_model, naki_model=naki_model)
        username = tmp_bot._get_guest_id()
        print(f"ゲストID取得: {username}")

    print(f"\n目標: {args.games} 試合完了  ID={username}\n")

    completed = 0   # 対局完了カウント
    attempts  = 0   # 試行回数（マッチング不成立も含む）
    errors    = 0

    while completed < args.games and attempts < args.max_attempts:
        attempts += 1
        print(f"\n{'='*50}")
        print(f"[完了 {completed}/{args.games}]  試行 #{attempts}  対局開始")
        print(f"{'='*50}")

        try:
            bot = TenhouBot(
                username=username,
                ai_model=ai_model,
                naki_model=naki_model,
                game_type=args.game_type,
                verbose=True,
            )
            game_done = bot.run()

            if game_done:
                completed += 1
                print(f"\n[OK] 対局完了 ({completed}/{args.games})")
            else:
                print(f"\n[--] マッチング不成立（スキップ）")

        except Exception as e:
            print(f"[!] エラー: {e}")
            errors += 1

        if completed < args.games and attempts < args.max_attempts:
            print(f"\n次の対局まで {args.interval} 秒待機...")
            time.sleep(args.interval)

    # 結果サマリー
    print(f"\n{'='*50}")
    if completed >= args.games:
        print(f"目標達成: {completed} 試合完了  (試行 {attempts} 回 / エラー {errors} 回)")
    else:
        print(f"最大試行回数到達: {completed}/{args.games} 試合完了  (試行 {attempts} 回 / エラー {errors} 回)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
