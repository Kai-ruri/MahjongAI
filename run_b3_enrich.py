"""
run_b3_enrich.py - B-3: 既存データセットのCH21（残り枚数マップ）を更新

【廃止注記】
このスクリプトは旧25チャンネルテンソル専用のパッチユーティリティです。
MahjongStateV5.to_tensor() が33チャンネル設計に移行した現在、
CH22（残り牌枚数マップ）は to_tensor() 内で正確に計算されるため
このスクリプトを実行する必要はありません。
新規データは build_supervised_dataset.py で生成してください。

--- 以下は旧25ch向けの旧実装（参考用に残す）---
既存のpklデータに保存済みのtensorは skip_logic=True で生成されているため
CH21はゼロだった。これを残り枚数マップ (4-visible)/4.0 で上書きする。

visible はテンソルの各チャンネルから逆算：
  visible[i] = CH0[i]*4 + (CH1+CH2+CH3)[i]/0.25 + (CH4+CH5+CH6)[i]/0.25
               + CH19[i]/0.25 + CH10[i]
"""

import pickle
import numpy as np
import os
import time


def enrich_ch21(tensor):
    """
    25チャンネルtensorのCH21を残り枚数マップで更新する。
    入力: (25, 34) numpy array
    """
    # visible_tiles を各チャンネルから復元
    visible = np.zeros(34, dtype=np.float32)
    visible += tensor[0] * 4.0           # CH0: 自家手牌 (count/4.0 を戻す)
    for ch in [1, 2, 3]:
        visible += tensor[ch] / 0.25     # CH1-3: 他家捨て牌 (+=0.25 を戻す)
    for ch in [4, 5, 6]:
        visible += tensor[ch] / 0.25     # CH4-6: 他家副露
    visible += tensor[19] / 0.25        # CH19: 自家副露
    visible += tensor[10]               # CH10: ドラ表示牌 (+=1.0)
    visible = np.minimum(visible, 4.0)

    new_tensor = tensor.copy()
    new_tensor[21] = np.maximum(0.0, 4.0 - visible) / 4.0
    return new_tensor


def process_pkl(input_path, output_path):
    print(f"[処理] {input_path}")
    t0 = time.perf_counter()

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    n = len(data)
    print(f"  レコード数: {n:,}")

    enriched = []
    for i, rec in enumerate(data):
        new_rec = dict(rec)
        new_rec["tensor"] = enrich_ch21(rec["tensor"])
        enriched.append(new_rec)

        if (i + 1) % 50000 == 0:
            elapsed = time.perf_counter() - t0
            pct = (i + 1) / n * 100
            print(f"  {i+1:,}/{n:,} ({pct:.0f}%) - {elapsed:.1f}s")

    with open(output_path, "wb") as f:
        pickle.dump(enriched, f)

    elapsed = time.perf_counter() - t0
    print(f"  完了: {elapsed:.1f}s → {output_path}")
    return n


def main():
    print("=== B-3: CH21 (残り枚数マップ) エンリッチ ===\n")

    targets = [
        ("dataset_phoenix_closed_precall_part1.pkl",
         "dataset_b3_part1.pkl"),
        ("dataset_b1_downloaded.pkl",
         "dataset_b3_downloaded.pkl"),
        ("dataset_phoenix_closed_precall_part2_test.pkl",
         "dataset_b3_part2_test.pkl"),
        ("dataset_phoenix_closed_precall_part3_test.pkl",
         "dataset_b3_part3_test.pkl"),
    ]

    total_records = 0
    t_start = time.perf_counter()

    for src, dst in targets:
        if not os.path.exists(src):
            print(f"  [skip] {src} が存在しません")
            continue
        n = process_pkl(src, dst)
        total_records += n
        print()

    elapsed = time.perf_counter() - t_start
    print(f"=== 完了 ===")
    print(f"合計: {total_records:,} レコード, 所要時間: {elapsed:.1f}s")
    print("\n検証:")

    # 検証: CH21が正しく設定されているか確認
    with open("dataset_b3_part3_test.pkl", "rb") as f:
        test_data = pickle.load(f)
    r = test_data[0]
    ch21 = r["tensor"][21]
    print(f"  CH21 min={ch21.min():.3f}, max={ch21.max():.3f}, mean={ch21.mean():.3f}")
    print(f"  ゼロ以外の要素数: {(ch21 > 0).sum()}/34")
    print(f"  (0=残り0枚=全て見えている, 1.0=残り4枚=まだ誰も見ていない)")


if __name__ == "__main__":
    main()
