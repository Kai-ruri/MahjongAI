# build_riichi_oshibiki_v2_dataset.py
"""
リーチ・押し引きデータセット v2 (高速並列版)

修正点:
  - to_tensor(skip_logic=True) で高速化 (旧: 3400ms/call → 新: 0ms)
  - ThreadPoolExecutor で並列ダウンロード (4スレッド同時)
  - sleep 廃止 (スレッド数で流量制御)

出力:
  dataset_riichi_v2.pkl   (34ch: 33ch基底 + CH33打牌位置, label 0=ダマ/1=リーチ)
  dataset_oshibiki_v2.pkl (33ch: CH33なし, label 0=オリ/1=押し)
"""

import os, gzip, re, pickle, time
import xml.etree.ElementTree as ET
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, socket, threading

from mahjong_engine import MahjongStateV5, calculate_shanten

# ===========================================================================
# パス設定
# ===========================================================================
LOG_DIR             = r"G:\マイドライブ\MahjongAI\logs"
SAVE_RIICHI_PATH    = r"G:\マイドライブ\MahjongAI\dataset_riichi_v2.pkl"
SAVE_OSHIBIKI_PATH  = r"G:\マイドライブ\MahjongAI\dataset_oshibiki_v2.pkl"
LOCAL_RIICHI_PATH   = "dataset_riichi_v2.pkl"
LOCAL_OSHIBIKI_PATH = "dataset_oshibiki_v2.pkl"
SAVE_INTERVAL       = 500    # 何対局ごとに中間保存するか
MAX_WORKERS         = 4      # 並列ダウンロード数
TIMEOUT             = 20     # 1リクエストのタイムアウト (秒)

# ===========================================================================
# 天鳳XMLパーサー
# ===========================================================================

def _decode_naki_type(m):
    if m & 0x4:    return 'chi'
    elif m & 0x10: return 'kakan'
    elif m & 0x8:  return 'pon'
    elif m & 0x20: return 'kan'
    return 'unknown'


def parse_tenhou_xml(xml_string):
    root = ET.fromstring(xml_string)
    events = []
    for node in root:
        tag = node.tag
        if tag == "INIT":
            seed = node.attrib.get("seed", "0,0,0,0,0,0").split(",")
            oya  = int(node.attrib.get("oya", "0"))
            hands = {
                i: [int(x) for x in node.attrib.get(f"hai{i}", "").split(",") if x]
                for i in range(4)
            }
            events.append({
                "type": "INIT", "bakaze": int(seed[0])//4, "kyoku": int(seed[0])%4,
                "honba": int(seed[1]), "kyotaku": int(seed[2]),
                "dora_indicator": int(seed[5]), "oya": oya, "hands": hands,
                "scores": [int(x)*100 for x in node.attrib.get("ten","0,0,0,0").split(",")]
            })
        elif tag[0] in 'TUVWtuvw' and tag[1:].isdigit():
            s = {'T':0,'U':1,'V':2,'W':3,'t':0,'u':1,'v':2,'w':3}[tag[0]]
            t = int(tag[1:])
            events.append({"type":"DRAW","seat":s,"tile_136":t,"tile_34":t//4})
        elif tag[0] in 'DEFGdefg' and tag[1:].isdigit():
            s = {'D':0,'E':1,'F':2,'G':3,'d':0,'e':1,'f':2,'g':3}[tag[0]]
            t = int(tag[1:])
            events.append({"type":"DISCARD","seat":s,"tile_136":t,"tile_34":t//4})
        elif tag == "REACH":
            events.append({"type":"REACH","seat":int(node.attrib.get("who")),
                           "step":int(node.attrib.get("step"))})
        elif tag == "N":
            m = int(node.attrib.get("m","0"))
            events.append({"type":"CALL","seat":int(node.attrib.get("who")),
                           "m":m,"naki_type":_decode_naki_type(m)})
        elif tag == "DORA":
            d = int(node.attrib.get("hai","0"))
            events.append({"type":"DORA","tile_136":d,"tile_34":d//4})
        elif tag in ("AGARI","RYUUKYOKU"):
            events.append({"type":tag})
    return events


# ===========================================================================
# グローバルトラッカー
# ===========================================================================

def _find_pon(hand_136, tile_34):
    m = [t for t in hand_136 if t//4 == tile_34]
    return (m[0], m[1]) if len(m) >= 2 else None

def _find_chi(hand_136, d34):
    if d34 >= 27: return None
    num = d34 % 9
    pats = []
    if num >= 2: pats.append((d34-2, d34-1))
    if 1 <= num <= 7: pats.append((d34-1, d34+1))
    if num <= 6: pats.append((d34+1, d34+2))
    for t1, t2 in pats:
        l1 = [t for t in hand_136 if t//4==t1]
        l2 = [t for t in hand_136 if t//4==t2]
        if t1==t2:
            if len(l1)>=2: return l1[0],l1[1]
        else:
            if l1 and l2: return l1[0],l2[0]
    return None


class Tracker:
    def __init__(self):
        self.hands_136       = {i:[] for i in range(4)}
        self.discards_136    = {i:[] for i in range(4)}
        self.riichi_declared = {i:False for i in range(4)}
        self.scores          = {i:25000 for i in range(4)}
        self.dora_indicators = []
        self.bakaze = self.kyoku = self.honba = self.kyotaku = self.oya = 0
        self.is_broken = self.has_naki = False
        self._last_disc_seat = self._last_disc_136 = -1

    def apply(self, e):
        t = e["type"]
        if t == "INIT":
            self.hands_136       = {i:e["hands"][i].copy() for i in range(4)}
            self.discards_136    = {i:[] for i in range(4)}
            self.riichi_declared = {i:False for i in range(4)}
            self.scores          = {i:e["scores"][i] for i in range(4)}
            self.dora_indicators = [e["dora_indicator"]]
            self.bakaze=e["bakaze"]; self.kyoku=e["kyoku"]
            self.honba=e["honba"];   self.kyotaku=e["kyotaku"]; self.oya=e["oya"]
            self.is_broken=self.has_naki=False
            self._last_disc_seat=self._last_disc_136=-1
        elif self.is_broken:
            return
        elif t == "DRAW":
            self.hands_136[e["seat"]].append(e["tile_136"])
        elif t == "DISCARD":
            s,tile = e["seat"],e["tile_136"]
            if tile in self.hands_136[s]:
                self.hands_136[s].remove(tile)
                self.discards_136[s].append(tile)
                self._last_disc_seat=s; self._last_disc_136=tile
            else:
                self.is_broken=True
        elif t == "REACH" and e["step"]==1:
            s=e["seat"]; self.riichi_declared[s]=True
            self.scores[s]-=1000; self.kyotaku+=1
        elif t == "CALL":
            nt=e.get("naki_type","unknown")
            if nt in ('kan','kakan','unknown'):
                self.is_broken=True; return
            self.has_naki=True
            d34=self._last_disc_136//4 if self._last_disc_136>=0 else -1
            if self._last_disc_136 in self.discards_136.get(self._last_disc_seat,[]):
                self.discards_136[self._last_disc_seat].remove(self._last_disc_136)
            caller=e["seat"]
            if nt=='pon':
                r=_find_pon(self.hands_136[caller],d34)
                if r: self.hands_136[caller].remove(r[0]); self.hands_136[caller].remove(r[1])
                else: self.is_broken=True
            elif nt=='chi':
                r=_find_chi(self.hands_136[caller],d34)
                if r: self.hands_136[caller].remove(r[0]); self.hands_136[caller].remove(r[1])
                else: self.is_broken=True
        elif t=="DORA":
            self.dora_indicators.append(e["tile_136"])


def build_state(tracker, seat):
    state = MahjongStateV5()
    for t in tracker.hands_136[seat]:
        state.add_tile(0, t)
    for pov in range(4):
        actual = (seat+pov)%4
        for t in tracker.discards_136[actual]:
            state.discard_tile(pov, t)
    state.dora_indicators = [d//4 for d in tracker.dora_indicators]
    state.bakaze  = tracker.bakaze
    state.jikaze  = (seat-tracker.oya+4)%4
    state.honba   = tracker.honba
    state.kyotaku = tracker.kyotaku
    for pov in range(4):
        actual = (seat+pov)%4
        state.riichi_declared[pov] = tracker.riichi_declared[actual]
        state.scores[pov]          = tracker.scores[actual]
    return state


# ===========================================================================
# 抽出関数
# ===========================================================================

def extract_both(events, log_id):
    """1対局分のイベント列からリーチ・押し引きサンプルを抽出"""
    tracker = Tracker()
    riichi_recs = []
    oshi_recs   = []

    for idx in range(len(events)-2):
        ev   = events[idx]
        nev  = events[idx+1]
        tracker.apply(ev)
        if tracker.is_broken or tracker.has_naki:
            continue
        if ev["type"] != "DRAW":
            continue
        seat = ev["seat"]
        if tracker.riichi_declared[seat]:
            continue

        # ---- リーチ抽出 ----
        # Case A: リーチ宣言
        if (nev["type"]=="REACH" and nev.get("step")==1 and nev["seat"]==seat):
            tile34 = None
            for j in range(idx+2, min(idx+5, len(events))):
                ne2 = events[j]
                if ne2["type"]=="DISCARD" and ne2["seat"]==seat:
                    tile34 = ne2["tile_34"]; break
                elif ne2["type"] in ("DRAW","CALL","INIT","AGARI","RYUUKYOKU"):
                    break
            if tile34 is not None:
                state = build_state(tracker, seat)
                base  = state.to_tensor(skip_logic=True)   # (33,34) 高速
                ch33  = np.zeros((1,34), dtype=np.float32)
                ch33[0][tile34] = 1.0
                riichi_recs.append({
                    "tensor": np.concatenate((base,ch33),axis=0),
                    "label": 1, "meta_log_id": log_id,
                })

        # Case B: ダマテン
        elif (nev["type"]=="DISCARD" and nev["seat"]==seat):
            tile34 = nev["tile_34"]
            hc = [0]*34
            for t in tracker.hands_136[seat]:
                hc[t//4] += 1
            hc[tile34] -= 1
            if calculate_shanten(hc) == 0:   # ★ テンパイ = 0
                state = build_state(tracker, seat)
                base  = state.to_tensor(skip_logic=True)
                ch33  = np.zeros((1,34), dtype=np.float32)
                ch33[0][tile34] = 1.0
                riichi_recs.append({
                    "tensor": np.concatenate((base,ch33),axis=0),
                    "label": 0, "meta_log_id": log_id,
                })

        # ---- 押し引き抽出 ----
        if (nev["type"]=="DISCARD" and nev["seat"]==seat
                and not tracker.riichi_declared[seat]):
            opp_riichi = [s for s in range(4) if s!=seat and tracker.riichi_declared[s]]
            if opp_riichi:
                tile34 = nev["tile_34"]
                genbutsu_all = all(
                    tile34 in set(t//4 for t in tracker.discards_136[rs])
                    for rs in opp_riichi
                )
                state = build_state(tracker, seat)
                tensor = state.to_tensor(skip_logic=True)  # (33,34) CH33なし
                oshi_recs.append({
                    "tensor": tensor,
                    "label": 0 if genbutsu_all else 1,
                    "meta_log_id": log_id,
                })

    return riichi_recs, oshi_recs


# ===========================================================================
# ダウンロード関数 (スレッドセーフ)
# ===========================================================================

def download_and_extract(log_id):
    """1対局をダウンロードして抽出。失敗時は ([], []) を返す"""
    try:
        url = f"https://tenhou.net/0/log/?{log_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
        try:
            xml_string = gzip.decompress(raw).decode('utf-8')
        except Exception:
            xml_string = raw.decode('utf-8')
        events = parse_tenhou_xml(xml_string)
        return extract_both(events, log_id)
    except Exception as e:
        return [], []


# ===========================================================================
# 保存
# ===========================================================================

_save_lock = threading.Lock()

def _save(records, local_path, gd_path):
    with _save_lock:
        with open(local_path, "wb") as f:
            pickle.dump(records, f)
        try:
            with open(gd_path, "wb") as f:
                pickle.dump(records, f)
        except Exception:
            pass

def _load_existing(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            records = pickle.load(f)
        return records, set(r["meta_log_id"] for r in records)
    return [], set()


# ===========================================================================
# メイン
# ===========================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    socket.setdefaulttimeout(TIMEOUT)

    print("=== リーチ・押し引きデータセット v2 (並列版) 抽出開始 ===", flush=True)

    riichi_records,   riichi_ids   = _load_existing(LOCAL_RIICHI_PATH)
    oshibiki_records, oshibiki_ids = _load_existing(LOCAL_OSHIBIKI_PATH)
    processed_ids = riichi_ids | oshibiki_ids

    print(f"既存データ: リーチ {len(riichi_records)} 件 / 押し引き {len(oshibiki_records)} 件", flush=True)

    # ログID収集
    pattern = re.compile(r'log=(\d{10}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{8})')
    gz_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.gz')])
    phoenix_log_ids = []
    for filename in gz_files:
        try:
            with gzip.open(os.path.join(LOG_DIR,filename),'rt',encoding='utf-8',errors='ignore') as f:
                for line in f:
                    if "四鳳" in line:
                        for lid in pattern.findall(line):
                            phoenix_log_ids.append(lid)
        except Exception:
            pass

    remaining = [lid for lid in phoenix_log_ids if lid not in processed_ids]
    done_cnt  = len(phoenix_log_ids) - len(remaining)
    total     = len(phoenix_log_ids)
    print(f"対局ID: {total} 件 / 残り {len(remaining)} 件 (済み: {done_cnt})", flush=True)
    print(f"並列数: {MAX_WORKERS} スレッド\n", flush=True)

    t_start    = time.time()
    completed  = 0
    lock       = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_and_extract, lid): lid for lid in remaining}

        for future in as_completed(futures):
            log_id = futures[future]
            r_recs, os_recs = future.result()

            with lock:
                riichi_records.extend(r_recs)
                oshibiki_records.extend(os_recs)
                completed += 1
                elapsed = time.time() - t_start
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                global_idx = done_cnt + completed
                print(f"  [{global_idx}/{total}] {log_id}"
                      f"  リーチ+{len(r_recs)} 押引+{len(os_recs)}"
                      f"  (累計: {len(riichi_records)}/{len(oshibiki_records)})"
                      f"  [{rate:.0f} 局/分]", flush=True)

                if completed % SAVE_INTERVAL == 0:
                    _save(riichi_records,   LOCAL_RIICHI_PATH,   SAVE_RIICHI_PATH)
                    _save(oshibiki_records, LOCAL_OSHIBIKI_PATH, SAVE_OSHIBIKI_PATH)
                    print(f"  [checkpoint] {global_idx} 局完了 → 中間保存", flush=True)

    _save(riichi_records,   LOCAL_RIICHI_PATH,   SAVE_RIICHI_PATH)
    _save(oshibiki_records, LOCAL_OSHIBIKI_PATH, SAVE_OSHIBIKI_PATH)

    print(f"\n===== 抽出完了 =====", flush=True)
    print(f"  リーチ  : {len(riichi_records)} 件  → {SAVE_RIICHI_PATH}", flush=True)
    print(f"  押し引き: {len(oshibiki_records)} 件  → {SAVE_OSHIBIKI_PATH}", flush=True)
