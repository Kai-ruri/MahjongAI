# build_supervised_dataset.py
import pickle
from parse_tenhou_log import TenhouParser, GlobalReplayTracker
from mahjong_engine import MahjongStateV5

def build_local_state(tracker: GlobalReplayTracker, target_seat: int) -> MahjongStateV5:
    """グローバル状態から、特定のプレイヤー視点のMahjongStateV5を作成"""
    state = MahjongStateV5()
    
    # 手牌
    for t_136 in tracker.hands_136[target_seat]:
        state.add_tile(0, t_136) 
        
    # 河
    for pov in range(4):
        actual_seat = (target_seat + pov) % 4
        for t_136 in tracker.discards_136[actual_seat]:
            state.discard_tile(pov, t_136)
            
    # メタ情報
    state.dora_indicators = tracker.dora_indicators.copy()
    state.bakaze = tracker.bakaze
    state.jikaze = (target_seat - tracker.oya + 4) % 4
    for pov in range(4):
        actual_seat = (target_seat + pov) % 4
        state.riichi_declared[pov] = tracker.riichi_declared[actual_seat]
        state.scores[pov] = tracker.scores[actual_seat]
        
    return state

def extract_dataset_from_xml(xml_string, log_id="unknown"):
    """XMLから「ツモ直後・打牌直前」の学習データを抽出する"""
    parser = TenhouParser(xml_string)
    events = parser.parse()
    
    tracker = GlobalReplayTracker()
    dataset_records = []
    
    for i in range(len(events) - 1):
        current_event = events[i]
        next_event = events[i+1]
        
        # 状態を1歩進める
        tracker.apply_event(current_event)
        
        if tracker.is_broken:
            continue # 鳴きが入るなどして追えなくなった局はスキップ（データ品質優先）
        
        # 🌟 抽出条件：自分がツモった直後で、次のイベントが自分の打牌
        if current_event["type"] == "DRAW" and next_event["type"] == "DISCARD":
            if current_event["seat"] == next_event["seat"]:
                target_seat = current_event["seat"]
                label_discard_34 = next_event["tile_34"]
                
                # V5状態に変換
                local_state = build_local_state(tracker, target_seat)
                
                # skip_logic=False: CH23-26(シャンテン/受け入れ/EV)も含む完全版テンソル
                tensor_input = local_state.to_tensor(skip_logic=False)
                
                record = {
                    "meta_log_id": log_id,
                    "meta_kyoku": tracker.kyoku,
                    "meta_turn": len(tracker.discards_136[target_seat]), 
                    "meta_seat": target_seat,
                    "tensor": tensor_input,
                    "label": label_discard_34
                }
                dataset_records.append(record)
                
    return dataset_records

if __name__ == "__main__":
    import os
    import gzip
    import re
    import time
    import urllib.request

    print("🔥 本番データ抽出を開始します...")

    # HTMLインデックスファイルから鳳凰卓の対局IDを収集
    target_dir = "logs"
    existing_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.gz')])
    pattern = re.compile(r'log=(\d{10}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{8})')

    print(f"📁 フォルダ内の .gz ファイルをスキャン中... (計 {len(existing_files)} 個)")
    phoenix_log_ids = []
    for filename in existing_files:
        with gzip.open(os.path.join(target_dir, filename), 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "四鳳" in line:
                    for log_id in pattern.findall(line):
                        phoenix_log_ids.append(log_id)

    print(f"🎯 鳳凰卓の対局IDを {len(phoenix_log_ids)} 件 発見しました！\n")

    all_dataset_records = []

    for idx, log_id in enumerate(phoenix_log_ids):
        print(f"  -> ダウンロード中 [{idx+1}/{len(phoenix_log_ids)}]: {log_id}")
        try:
            url = f"https://tenhou.net/0/log/?{log_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()

            # レスポンスがgzip圧縮されている場合は展開する
            try:
                xml_string = gzip.decompress(raw).decode('utf-8')
            except Exception:
                xml_string = raw.decode('utf-8')

            records = extract_dataset_from_xml(xml_string, log_id=log_id)
            all_dataset_records.extend(records)
            print(f"     -> {len(records)} 件 抽出（累計: {len(all_dataset_records)} 件）")

            time.sleep(1.0)  # サーバー負荷軽減のため1秒待機

            # 100対局ごとに中間保存
            if (idx + 1) % 100 == 0:
                save_path = "dataset_intermediate_phoenix.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(all_dataset_records, f)
                print(f"  💾 中間保存しました（{idx+1}対局完了 / {len(all_dataset_records)} 件）")

        except Exception as e:
            print(f"     [!] エラー発生（スキップします）: {e}")
            continue

    print(f"\n✨✨ 抽出完了！ 合計 {len(all_dataset_records)} 件 の局面データを獲得しました！ ✨✨")

    # 段階Aの中間形式として保存
    save_path = "dataset_intermediate_phoenix.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_dataset_records, f)
    print(f"💾 中間データを保存しました: {save_path}")