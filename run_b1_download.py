"""
run_b1_download.py - B-1: 天鳳牌譜データのダウンロードと抽出

ロビーファイルの game_id から天鳳の牌譜XMLをダウンロードし、
打牌学習データとして抽出・保存する。

使い方:
    python run_b1_download.py             # 全1148件取得
    python run_b1_download.py --limit 100 # 100件だけ試す
    python run_b1_download.py --resume    # 途中から再開

出力: dataset_b1_downloaded.pkl（随時チェックポイント保存）
"""

import argparse
import gzip
import os
import pickle
import re
import time
import urllib.request

from dataset_extractor import extract_dataset

# =========================================
# 設定
# =========================================
SLEEP_BETWEEN_REQUESTS = 1.0   # 秒（サーバー負荷軽減のため）
CHECKPOINT_EVERY = 50          # 件ごとにチェックポイント保存
OUTPUT_PKL = "dataset_b1_downloaded.pkl"
PROGRESS_FILE = "dataset_b1_downloaded_progress.txt"
LOGS_DIR = "logs"


# =========================================
# 1. ロビーからゲームIDを収集
# =========================================
def collect_game_ids():
    """ロビーファイル (logs/*.gz) から四鳳ゲームIDを全て収集する"""
    pattern = re.compile(r'log=(\d{10}gm-00a9-[0-9a-f]{4}-[0-9a-f]{8})')
    all_ids = set()
    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.endswith('.gz'):
            continue
        with gzip.open(os.path.join(LOGS_DIR, fname), 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        all_ids.update(pattern.findall(content))
    return sorted(all_ids)


def collect_existing_ids():
    """既存のデータセットに含まれるゲームIDを収集（重複ダウンロードを防ぐ）"""
    existing = set()
    for path in [
        "dataset_phoenix_closed_precall_part1.pkl",
        "dataset_phoenix_closed_precall_part2_test.pkl",
        "dataset_phoenix_closed_precall_part3_test.pkl",
        OUTPUT_PKL,
    ]:
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            data = pickle.load(f)
        existing.update(r['meta_log_id'] for r in data)
    return existing


def collect_processed_ids():
    """進捗ファイルから処理済みIDを読み込む（エラーで空だったゲームも含む）"""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())


# =========================================
# 2. ダウンロードと抽出
# =========================================
def download_game_xml(game_id, timeout=15):
    """天鳳から牌譜XMLをダウンロードしてXML文字列を返す"""
    url = f'https://tenhou.net/0/log/?{game_id}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 MahjongAI-Research'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode('utf-8', errors='ignore')
    # mjloggm タグの XML を抽出
    m = re.search(r'(<mjloggm[^>]*>.*?</mjloggm>)', content, re.DOTALL)
    if m:
        return m.group(1)
    return None


# =========================================
# 3. メイン
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='取得する最大ゲーム数')
    parser.add_argument('--resume', action='store_true', help='進捗ファイルから再開')
    args = parser.parse_args()

    print("=== B-1: 天鳳牌譜ダウンロード ===\n")

    # ゲームIDを収集
    print("[1] ゲームIDを収集中...")
    all_ids = collect_game_ids()
    existing_ids = collect_existing_ids()
    processed_ids = collect_processed_ids() if args.resume else set()
    processed_ids |= existing_ids  # 既存データも処理済みとみなす

    target_ids = [g for g in all_ids if g not in processed_ids]
    if args.limit:
        target_ids = target_ids[:args.limit]

    print(f"  全ゲームID: {len(all_ids)}")
    print(f"  既存/処理済: {len(processed_ids)}")
    print(f"  新規取得対象: {len(target_ids)}")

    if len(target_ids) == 0:
        print("  取得対象がありません。")
        return

    # 既存の出力PKLを読み込む（resumeの場合）
    all_records = []
    if args.resume and os.path.exists(OUTPUT_PKL):
        with open(OUTPUT_PKL, 'rb') as f:
            all_records = pickle.load(f)
        print(f"  resume: 既存 {len(all_records)} レコードを引き継ぎ")

    # ダウンロードループ
    print(f"\n[2] ダウンロード開始 ({len(target_ids)} 件)...")
    print("  (Ctrl+C で中断可。PROGRESS_FILE に進捗を保存します)")
    print()

    ok_count = 0
    err_count = 0
    new_records = 0

    try:
        for i, game_id in enumerate(target_ids):
            try:
                xml_str = download_game_xml(game_id)
                if xml_str is None:
                    print(f"  [{i+1:>4}/{len(target_ids)}] {game_id}: XML未検出 (skip)")
                    processed_ids.add(game_id)
                    with open(PROGRESS_FILE, 'a') as f:
                        f.write(game_id + '\n')
                    err_count += 1
                    time.sleep(SLEEP_BETWEEN_REQUESTS)
                    continue

                records = extract_dataset(xml_str, log_id=game_id)
                all_records.extend(records)
                new_records += len(records)
                processed_ids.add(game_id)
                ok_count += 1

                print(f"  [{i+1:>4}/{len(target_ids)}] {game_id}: {len(records)} records  "
                      f"(合計: {len(all_records)})")

                # 進捗ファイル更新
                with open(PROGRESS_FILE, 'a') as f:
                    f.write(game_id + '\n')

            except Exception as e:
                print(f"  [{i+1:>4}/{len(target_ids)}] {game_id}: ERROR - {e}")
                err_count += 1

            # チェックポイント保存
            if (i + 1) % CHECKPOINT_EVERY == 0:
                with open(OUTPUT_PKL, 'wb') as f:
                    pickle.dump(all_records, f)
                print(f"  [checkpoint] {len(all_records)} records を {OUTPUT_PKL} に保存")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    except KeyboardInterrupt:
        print("\n  中断されました。これまでのデータを保存します...")

    # 最終保存
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(all_records, f)

    print(f"\n=== ダウンロード完了 ===")
    print(f"  成功: {ok_count} ゲーム")
    print(f"  エラー: {err_count} ゲーム")
    print(f"  新規レコード: {new_records}")
    print(f"  合計レコード: {len(all_records)}")
    print(f"  保存先: {OUTPUT_PKL}")
    print()
    print("次のステップ: python run_b1_train.py でモデルを再学習してください")


if __name__ == "__main__":
    main()
