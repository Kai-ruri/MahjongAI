#!/usr/bin/env python3
"""
tenhou_client.py - 天鳳オンライン対局クライアント (JSON WebSocket)

接続先: wss://b-wk.mjv.jp
プロトコル: JSON (タイル番号はタグサフィックスに埋め込み: {"tag":"T136"})

使い方:
  python tenhou_client.py                    # 新規ゲストIDを自動取得
  python tenhou_client.py --username ID....  # 既存ゲストIDを指定

観戦方法: tenhou.net のロビー → 観戦タブ

依存: pip install websocket-client
"""

import json
import threading
import time
import random
import sys
import copy
import argparse
import queue
import urllib.request
from urllib.parse import quote, unquote
import websocket
import ssl

import torch
import numpy as np
from mahjong_engine import tile_names, calculate_final_score, MahjongStateV5
from mahjong_model import MahjongResNet_UltimateV3, MahjongResNet_Naki, MahjongResNet_Naki_V2
import hybrid_inference
from selfplay_minimal import should_declare_riichi, GlobalRoundState, get_player_wind

TENHOU_WS_URL = "wss://b-wk.mjv.jp"
REGIDQ_URL    = "https://b.mjv.jp/regid"

DISCARD_MODEL_PATH = "discard_b1_best.pth"
NAKI_MODEL_PATH    = "mahjong_naki_model_master.pth"

# ============================================================
# タイル変換ユーティリティ
# ============================================================
def t136_to_t34(tile_136):
    return tile_136 // 4

def is_aka_dora(tile_136):
    return tile_136 in (16, 52, 88)

def hand_to_34_counts(hand_136):
    counts = [0] * 34
    for t in hand_136:
        counts[t // 4] += 1
    return counts

def count_aka(hand_136):
    return sum(1 for t in hand_136 if is_aka_dora(t))

def decode_meld_m(m_val):
    m = int(m_val)
    if m & 0x4:
        t0 = (m & 0xFC00) >> 10
        t0 //= 3
        suit  = t0 // 7
        start = t0 % 7
        return "chi", [suit * 9 + start + i for i in range(3)]
    elif m & 0x18:
        t0 = (m & 0xFE00) >> 9
        tile_t34 = t0 // 3
        meld_type = "kakan" if (m & 0x10) else "pon"
        return meld_type, [tile_t34] * (4 if meld_type == "kakan" else 3)
    else:
        t0 = (m & 0xFF00) >> 8
        tile_t34 = t0 // 4
        meld_type = "ankan" if (m & 0x3) == 0 else "daiminkan"
        return meld_type, [tile_t34] * 4

# ============================================================
# ゲーム状態
# ============================================================
class TenhouGameState:
    def __init__(self):
        self.my_seat         = 0
        self.dealer_seat     = 0
        self.bakaze          = 0
        self.kyoku           = 1
        self.honba           = 0
        self.riichi_sticks   = 0
        self.scores          = [25000] * 4
        self.my_hand_136     = []
        self.discards        = {0: [], 1: [], 2: [], 3: []}
        self.fixed_mentsu    = [[], [], [], []]
        self.riichi_declared = {0: False, 1: False, 2: False, 3: False}
        self.dora_indicators = []
        self.last_draw_136   = None
        self.junme           = 1
        self.my_aka_count    = 0
        self.my_ippatsu      = False

    def hand_34(self):
        return hand_to_34_counts(self.my_hand_136)

    def to_mahjong_state_v5(self):
        state = MahjongStateV5()
        state.hand             = self.hand_34()
        state.discards         = {i: list(self.discards[i]) for i in range(4)}
        state.riichi_declared  = dict(self.riichi_declared)
        state.fixed_mentsu     = copy.deepcopy(self.fixed_mentsu[self.my_seat])
        state.dora_indicators  = list(self.dora_indicators)
        state.forbidden_discards = []
        state.bakaze           = self.bakaze
        state.jikaze           = (self.my_seat - self.dealer_seat) % 4
        state.junme            = self.junme
        state.is_oya           = (self.my_seat == self.dealer_seat)
        state.score_situation  = None
        state.is_orasu         = False
        state.placement_pressure = "neutral"
        state.scores           = list(self.scores)
        state.my_pid           = self.my_seat
        state.dealer_pid       = self.dealer_seat
        state.rival_pids       = None
        state.enemy_open_counts = {i: len(self.fixed_mentsu[i]) for i in range(4)}
        state.honba            = self.honba
        state.kyotaku          = self.riichi_sticks
        return state

    def build_global_round_state(self):
        hands = [self.hand_34() if s == self.my_seat else [0]*34 for s in range(4)]
        gs = GlobalRoundState(
            hands=hands,
            discards={i: list(self.discards[i]) for i in range(4)},
            riichi_declared=dict(self.riichi_declared),
            fixed_mentsu=[copy.deepcopy(fm) for fm in self.fixed_mentsu],
            scores=list(self.scores),
            wall=[0] * 70,
            dora_indicators=list(self.dora_indicators),
            dealer_pid=self.dealer_seat,
            bakaze=self.bakaze,
            kyoku=self.kyoku,
            honba=self.honba,
            junme=self.junme,
            turn_pid=self.my_seat,
            is_orasu=False,
            riichi_sticks=self.riichi_sticks,
            last_draw=[None] * 4,
            riichi_junme=[None] * 4,
            game_over=False,
            aka_hands=[self.my_aka_count, 0, 0, 0],
            aka_pool={4: 0, 13: 0, 22: 0},
            ippatsu_eligible={i: False for i in range(4)},
        )
        gs.last_draw[self.my_seat] = (
            t136_to_t34(self.last_draw_136) if self.last_draw_136 is not None else None
        )
        return gs

# ============================================================
# メインボット
# ============================================================
class TenhouBot:
    def __init__(self, username=None, ai_model=None, naki_model=None,
                 game_type=9, verbose=True):
        self.username   = username   # None → REGIDQ で自動取得
        self.ai_model   = ai_model
        self.naki_model = naki_model
        self.game_type  = game_type
        self.verbose    = verbose

        self.ws      = None
        self.gs      = TenhouGameState()
        self.running = False
        self._msg_q  = queue.Queue()

        self._pending_discard_tile136 = None
        self._pending_discard_from    = None
        self._pending_t_attr          = 0
        self._pending_go              = None   # 認証中に届いた GO メッセージ

    def log(self, msg):
        if self.verbose:
            print(f"[Bot] {msg}", flush=True)

    # ----------------------------------------------------------
    # ゲストID取得 (REGIDQ)
    # ----------------------------------------------------------
    def _get_guest_id(self):
        """REGIDQ 2ステップでゲストIDを新規取得。
        レスポンス形式: クエリ文字列 (res=1012&id=IDxxxxxxxx-yyyyyyyy)
        名前制約: 小文字英数字のみ
        """
        import string as _string

        def _regid(params):
            url = f"{REGIDQ_URL}?{params}"
            req = urllib.request.Request(url, headers={
                "Referer": "https://tenhou.net/",
                "User-Agent": "Mozilla/5.0",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode()
            return dict(kv.split("=", 1) for kv in raw.split("&") if "=" in kv)

        # ランダムな小文字7文字の名前で空きを探す
        # res=1006 (レート制限) の場合は1時間待機してリトライ (最大48回 = 2日間)
        RATE_LIMIT_WAIT_SEC = 3600  # 1時間
        MAX_RATE_RETRIES = 48       # 最大48時間待つ
        rate_limit_count = 0

        for _ in range(200):
            name = "".join(random.choices(_string.ascii_lowercase, k=7))
            try:
                r = _regid(f"q=1&uname={quote(name)}")
                res_code = r.get('res')
                self.log(f"REGIDQ check [{name}]: res={res_code}")
                if res_code == "1006":
                    rate_limit_count += 1
                    if rate_limit_count > MAX_RATE_RETRIES:
                        raise RuntimeError(f"REGIDQ: レート制限が続くため中断します (res=1006 x{MAX_RATE_RETRIES})")
                    self.log(f"REGIDQ レート制限 (res=1006)。{RATE_LIMIT_WAIT_SEC//60}分待機後リトライ... ({rate_limit_count}/{MAX_RATE_RETRIES})")
                    time.sleep(RATE_LIMIT_WAIT_SEC)
                    continue
                rate_limit_count = 0  # 成功したらリセット
                if res_code == "0":
                    r2 = _regid(f"uname={quote(name)}")
                    self.log(f"REGIDQ register [{name}]: {r2}")
                    if r2.get("res") == "1012":
                        return r2["id"]
                time.sleep(2)
            except RuntimeError:
                raise
            except Exception as e:
                self.log(f"REGIDQ エラー: {e}")
                time.sleep(5)
        raise RuntimeError("REGIDQ: 空き名前が見つかりません")

    # ----------------------------------------------------------
    # WebSocket
    # ----------------------------------------------------------
    def _connect(self):
        self.log(f"WebSocket接続中: {TENHOU_WS_URL}")
        sslopt = {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
        self.ws = websocket.create_connection(
            TENHOU_WS_URL,
            timeout=30,
            origin="https://tenhou.net",
            sslopt=sslopt,
        )
        self.log("WebSocket接続完了")
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _recv_loop(self):
        while self.running:
            try:
                msg = self.ws.recv()
                if not msg:
                    self.log("切断されました")
                    self.running = False
                    break
                self._msg_q.put(msg)
            except websocket.WebSocketConnectionClosedException:
                self.log("WS接続クローズ")
                self.running = False
                break
            except Exception as e:
                if self.running:
                    self.log(f"recv error: {e}")
                self.running = False
                break

    def _send(self, obj):
        """dict → JSON、文字列はそのまま送信"""
        raw = json.dumps(obj, ensure_ascii=False) if isinstance(obj, dict) else str(obj)
        print(f"  → {raw}", flush=True)
        try:
            self.ws.send(raw)
        except Exception as e:
            self.log(f"send error: {e}")

    def _recv_messages(self, timeout=0.5):
        msgs = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msgs.append(self._msg_q.get(timeout=0.05))
            except queue.Empty:
                break
        return msgs

    def _wait_for(self, keyword, timeout=60):
        """keyword を含むメッセージが来るまでキューを監視"""
        deadline = time.time() + timeout
        result = []
        while time.time() < deadline:
            try:
                raw = self._msg_q.get(timeout=1.0)
                print(f"  ← {raw[:150]}", flush=True)
                result.append(raw)
                if keyword in raw:
                    return result
            except queue.Empty:
                pass
        return result

    # ----------------------------------------------------------
    # 認証
    # ----------------------------------------------------------
    def _authenticate(self):
        """JSON HELO 送信。ゲストIDは追加認証不要。
        再接続時は HELO の前に GO が来る場合がある。
        """
        self._send({"tag": "HELO", "name": self.username, "sx": "M"})
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                raw = self._msg_q.get(timeout=1.0)
                print(f"  ← {raw[:150]}", flush=True)
                m = json.loads(raw)
                tag = m.get("tag", "")
                if tag == "ERR":
                    raise RuntimeError(
                        f"HELO ERR code={m.get('code')} "
                        f"(1004=未登録/認証失敗, 1002=多重ログイン)"
                    )
                elif tag == "HELO":
                    self.log(f"ログイン成功: uname={m.get('uname','?')}")
                    return
                elif tag == "GO":
                    # 再接続: ゲームが継続中
                    self._pending_go = m
                    self.log(f"ゲーム再接続: type={m.get('type')}")
                    return
                # LN や他のメッセージは無視して待ち続ける
            except queue.Empty:
                pass
        self.log("HELO応答なし、続行...")

    # ----------------------------------------------------------
    # キープアライブ
    # ----------------------------------------------------------
    def _keepalive(self):
        while self.running:
            time.sleep(15)
            if self.running:
                try:
                    self.ws.send("<Z/>")  # キープアライブは生XML
                except Exception:
                    pass

    # ----------------------------------------------------------
    # 対局参加
    # ----------------------------------------------------------
    def _join_game(self):
        # 再接続: 認証中に GO を受信済み
        if self._pending_go is not None:
            msg = self._pending_go
            self._pending_go = None
            self.log(f"既存ゲームに再接続 (type={msg.get('type')})")
            self.log(f"観戦URL: https://tenhou.net/4/ → 観戦タブ (ユーザー: {self.username})")
            time.sleep(random.uniform(1, 2))
            self._send({"tag": "GOK"})
            return

        self.log(f"対局参加中 (type={self.game_type})")
        self._send({"tag": "JOIN", "t": f"0,{self.game_type}"})
        while self.running:
            for raw in self._recv_messages(timeout=5.0):
                print(f"  ← {raw[:150]}", flush=True)
                try:
                    m = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                tag = m.get("tag", "")
                if tag == "GO":
                    self.log(f"対局開始! type={m.get('type')}")
                    self.log(
                        f"観戦URL: https://tenhou.net/4/ → ロビー → 観戦タブ"
                        f" (ユーザー: {self.username})"
                    )
                    time.sleep(random.uniform(1, 2))
                    self._send({"tag": "GOK"})
                    return
                elif tag == "REJOIN":
                    t_val = m.get("t", f"0,{self.game_type},r")
                    self.log("対局未マッチ、再試行...")
                    time.sleep(random.uniform(3, 5))
                    self._send({"tag": "JOIN", "t": t_val})
                elif tag == "ERR":
                    raise RuntimeError(f"JOIN ERR: {m}")

    # ----------------------------------------------------------
    # メインエントリ
    # ----------------------------------------------------------
    def run(self):
        if self.username is None:
            self.log("ゲストID取得中...")
            self.username = self._get_guest_id()
            self.log(f"ゲストID: {self.username}")

        self.running = True
        self._connect()
        self._authenticate()

        threading.Thread(target=self._keepalive, daemon=True).start()
        self._join_game()

        try:
            self._game_loop()
        except KeyboardInterrupt:
            self.log("ユーザー中断")
        finally:
            self.running = False
            try:
                if self.ws:
                    self.ws.close()
            except Exception:
                pass

    def _game_loop(self):
        while self.running:
            for raw in self._recv_messages(timeout=0.2):
                print(f"  ← {raw[:150]}", flush=True)
                self._dispatch(raw)

    # ----------------------------------------------------------
    # JSON メッセージ解析・ルーティング
    # ----------------------------------------------------------
    def _dispatch(self, raw_msg):
        raw_msg = raw_msg.strip()
        if not raw_msg:
            return
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            return

        tag = msg.get("tag", "")
        if not tag:
            return

        # タグから数字サフィックスを分離: "T136" → letter="T", tile_num=136
        tile_num = None
        tag_letter = tag[0].upper()
        if len(tag) >= 2 and tag[1:].isdigit():
            tile_num = int(tag[1:])
            letter = tag_letter
        else:
            letter = tag.upper()  # INIT, REACH 等はそのまま

        try:
            # ツモ牌: T=seat0, U=seat1, V=seat2, W=seat3 (絶対席)
            # tile_num なし = 牌非公開 (他家ツモ)
            if tag_letter in "TUVW" and (len(tag) == 1 or tag[1:].isdigit()):
                seat = ord(tag_letter) - ord('T')
                self._on_draw(seat, tile_num)
            # 捨て牌: D/d=seat0, E/e=seat1, F/f=seat2, G/g=seat3 (絶対席)
            elif tag_letter in "DdEeFfGg" and (len(tag) == 1 or tag[1:].isdigit()):
                seat = ord(tag_letter.upper()) - ord('D')
                is_tsumogiri = tag[0].islower()
                t_val = int(msg.get("t", 0))
                self._on_discard(seat, tile_num, is_tsumogiri, t_val)
            else:
                handlers = {
                    "TAIKYOKU":  self._on_taikyoku,
                    "SAIKAI":    self._on_taikyoku,   # エイリアス
                    "KANSEN":    self._on_taikyoku,   # エイリアス
                    "UN":        self._on_un,
                    "INIT":      self._on_init,
                    "REINIT":    self._on_init,        # 再接続時
                    "REACH":     self._on_reach,
                    "N":         self._on_naki,
                    "DORA":      self._on_dora,
                    "AGARI":     self._on_agari,
                    "RYUUKYOKU": self._on_ryuukyoku,
                    "GO":        self._on_go,
                    "ERR":       lambda m: self.log(f"ERR受信: {m}"),
                    "PROF":      lambda m: self._stop("PROF受信"),
                }
                h = handlers.get(letter)
                if h:
                    h(msg)
        except Exception as e:
            self.log(f"dispatch error (tag={tag}): {e}")
            import traceback; traceback.print_exc()

    # ----------------------------------------------------------
    # イベントハンドラ
    # ----------------------------------------------------------
    def _on_taikyoku(self, msg):
        oya = int(msg.get("oya", 0))
        log_id = msg.get("log", "")
        self.log(f"TAIKYOKU oya={oya} log={log_id}")

    def _on_un(self, msg):
        for i in range(4):
            name = unquote(str(msg.get(f"n{i}", "")))
            if name == self.username:
                self.gs.my_seat = i
                self.log(f"自分の席: {i} ({name})")
                break
        self.log(f"段位: {msg.get('dan','')}  レート: {msg.get('rate','')}")

    def _on_init(self, msg):
        gs = self.gs
        gs.discards        = {0: [], 1: [], 2: [], 3: []}
        gs.fixed_mentsu    = [[], [], [], []]
        gs.riichi_declared = {0: False, 1: False, 2: False, 3: False}
        gs.last_draw_136   = None
        gs.junme           = 1
        gs.my_ippatsu      = False

        seed = str(msg.get("seed", "")).split(",")
        if len(seed) >= 6:
            gs.bakaze        = int(seed[0]) // 4
            gs.kyoku         = int(seed[0]) % 4 + 1
            gs.honba         = int(seed[1])
            gs.riichi_sticks = int(seed[2])
            gs.dora_indicators = [t136_to_t34(int(seed[5]))]

        ten_str = str(msg.get("ten", ""))
        if ten_str:
            gs.scores = [int(x) * 100 for x in ten_str.split(",")]

        gs.dealer_seat = int(msg.get("oya", 0))

        hai_str = str(msg.get("hai", ""))
        gs.my_hand_136  = [int(x) for x in hai_str.split(",") if x.strip()]
        gs.my_aka_count = count_aka(gs.my_hand_136)

        bz = "東" if gs.bakaze == 0 else "南"
        dora_name = tile_names[gs.dora_indicators[0]] if gs.dora_indicators else "?"
        self.log(
            f"INIT {bz}{gs.kyoku}局{gs.honba}本場 "
            f"親={gs.dealer_seat} 自席={gs.my_seat} "
            f"ドラ表示={dora_name} "
            f"手牌={[tile_names[t//4] for t in gs.my_hand_136]}"
        )
        time.sleep(random.uniform(1.5, 3.0))
        self._send({"tag": "NEXTREADY"})

    def _on_draw(self, seat, tile_136):
        """ツモイベント (seat = 絶対席 0-3)"""
        gs = self.gs
        if seat != gs.my_seat:
            return  # 他家のツモは無視

        if tile_136 is None:
            return

        gs.junme += 1
        gs.my_hand_136.append(tile_136)
        gs.last_draw_136 = tile_136
        if is_aka_dora(tile_136):
            gs.my_aka_count += 1

        tile_t34 = t136_to_t34(tile_136)
        self.log(
            f"ツモ: {tile_names[tile_t34]}  "
            f"手牌: {[tile_names[t//4] for t in gs.my_hand_136]}"
        )

        if self._check_tsumo_win(tile_136):
            return

        time.sleep(random.uniform(0.5, 1.5))
        self._do_discard()

    def _on_discard(self, seat, tile_136, is_tsumogiri, t_attr):
        """捨て牌イベント (seat = 絶対席 0-3)"""
        gs = self.gs

        tile_t34 = t136_to_t34(tile_136) if tile_136 is not None else None
        if tile_t34 is not None:
            gs.discards[seat].append(tile_t34)

        # 他家リーチ後の捨て牌で自分の一発消滅
        if seat != gs.my_seat and gs.riichi_declared.get(seat):
            gs.my_ippatsu = False

        if seat == gs.my_seat or tile_136 is None:
            return

        # t_attr のビット: 1=chi, 2=pon, 4=kan, 8=ron
        # t_attr > 0 のとき必ずサーバーに応答が必要
        if t_attr > 0:
            if t_attr & 8:
                if self._check_ron_win(tile_136):
                    return

            if t_attr & 3:  # pon or chi の可能性
                relative_from = (seat - gs.my_seat) % 4
                self._pending_discard_tile136 = tile_136
                self._pending_discard_from    = relative_from
                self._pending_t_attr          = t_attr
                time.sleep(random.uniform(0.3, 0.8))
                self._handle_naki_offer()
            else:
                # kan のみなどパス
                time.sleep(random.uniform(0.2, 0.5))
                self._send({"tag": "N"})

    def _on_reach(self, msg):
        who  = int(msg.get("who", 0))  # 絶対席
        step = int(msg.get("step", 1))
        if step == 1:
            self.gs.riichi_declared[who] = True
            if who != self.gs.my_seat:
                self.gs.my_ippatsu = False
            self.log(f"リーチ: 席{who}")

    def _on_naki(self, msg):
        gs = self.gs
        who   = int(msg.get("who", 0))  # 絶対席
        m_val = msg.get("m")
        if m_val:
            meld_type, tiles_t34 = decode_meld_m(m_val)
            gs.fixed_mentsu[who].append(tiles_t34)
            self.log(f"鳴き: 席{who} {meld_type} {[tile_names[t] for t in tiles_t34]}")
        gs.my_ippatsu = False  # 鳴きで全員の一発消滅

    def _on_dora(self, msg):
        hai_val = msg.get("hai")
        if hai_val is not None:
            t34 = t136_to_t34(int(hai_val))
            self.gs.dora_indicators.append(t34)
            self.log(f"新ドラ表示牌: {tile_names[t34]}")

    def _on_agari(self, msg):
        who = int(msg.get("who", 0))
        sc_str = str(msg.get("sc", ""))
        if sc_str:
            parts = sc_str.split(",")
            for i in range(4):
                if i * 2 < len(parts):
                    self.gs.scores[i] = int(parts[i * 2]) * 100
        self.log(f"和了: 席{who}  スコア: {self.gs.scores}")

        if "owari" in msg:
            self.log("対局終了")
            self._show_final_result(msg)
            self.running = False
            return

        time.sleep(random.uniform(3, 5))
        self._send({"tag": "NEXTREADY"})

    def _on_ryuukyoku(self, msg):
        sc_str = str(msg.get("sc", ""))
        if sc_str:
            parts = sc_str.split(",")
            for i in range(4):
                if i * 2 < len(parts):
                    self.gs.scores[i] = int(parts[i * 2]) * 100

        if "owari" in msg:
            self.log("流局→対局終了")
            self._show_final_result(msg)
            self.running = False
            return

        self.log(f"流局 スコア: {self.gs.scores}")
        time.sleep(random.uniform(3, 5))
        self._send({"tag": "NEXTREADY"})

    def _on_go(self, msg):
        # _join_game ループ外で再受信した場合の保険
        self.log(f"GO再受信: type={msg.get('type')}")
        self._send({"tag": "GOK"})

    def _stop(self, reason):
        self.log(f"停止: {reason}")
        self.running = False

    # ----------------------------------------------------------
    # 打牌
    # ----------------------------------------------------------
    def _do_discard(self):
        gs    = self.gs
        state = gs.to_mahjong_state_v5()

        best_tile_t34, _, debug_rows = hybrid_inference.hybrid_ai_decision_v6_rerank_debug(
            state, self.ai_model, top_k=5
        )

        gr_state      = gs.build_global_round_state()
        declare_riichi = should_declare_riichi(
            gr_state, gs.my_seat, best_tile_t34, debug_rows
        )

        discard_136 = self._pick_tile_136(best_tile_t34)

        if declare_riichi and not gs.riichi_declared[gs.my_seat]:
            self.log(f"リーチ宣言 捨て牌: {tile_names[best_tile_t34]}")
            self._send({"tag": "REACH"})
            time.sleep(random.uniform(0.3, 0.6))
            self._remove_from_hand_136(discard_136)
            gs.discards[gs.my_seat].append(best_tile_t34)
            gs.riichi_declared[gs.my_seat] = True
            gs.my_ippatsu = True
            self._send({"tag": "D", "p": discard_136})
            return

        self.log(f"打牌: {tile_names[best_tile_t34]}")
        self._remove_from_hand_136(discard_136)
        gs.discards[gs.my_seat].append(best_tile_t34)
        self._send({"tag": "D", "p": discard_136})

    def _pick_tile_136(self, tile_t34):
        candidates = [t for t in self.gs.my_hand_136 if t // 4 == tile_t34]
        if not candidates:
            raise ValueError(
                f"手牌に {tile_names[tile_t34]} がありません: {self.gs.my_hand_136}"
            )
        normal = [t for t in candidates if not is_aka_dora(t)]
        return normal[0] if normal else candidates[0]

    def _remove_from_hand_136(self, tile_136):
        self.gs.my_hand_136.remove(tile_136)
        if is_aka_dora(tile_136):
            self.gs.my_aka_count -= 1

    # ----------------------------------------------------------
    # ツモ和了
    # ----------------------------------------------------------
    def _check_tsumo_win(self, drawn_136):
        gs       = self.gs
        tile_t34 = t136_to_t34(drawn_136)
        jikaze   = (gs.my_seat - gs.dealer_seat) % 4

        result = calculate_final_score(
            closed_counts   = gs.hand_34(),
            fixed_mentsu    = gs.fixed_mentsu[gs.my_seat],
            win_tile        = tile_t34,
            is_tsumo        = True,
            bakaze          = gs.bakaze,
            jikaze          = jikaze,
            is_oya          = (gs.my_seat == gs.dealer_seat),
            is_riichi       = gs.riichi_declared[gs.my_seat],
            dora_indicators = gs.dora_indicators,
            aka_dora_count  = gs.my_aka_count,
            is_ippatsu      = gs.my_ippatsu,
        )

        if result and result.get("score", 0) > 0:
            self.log(
                f"ツモ和了! {tile_names[tile_t34]} "
                f"{result['han']}翻{result['fu']}符 {result.get('score',0)}点 "
                f"役: {result.get('yaku', [])}"
            )
            time.sleep(random.uniform(0.5, 1.0))
            self._send({"tag": "N", "type": 7})  # ツモ
            return True
        return False

    # ----------------------------------------------------------
    # ロン和了
    # ----------------------------------------------------------
    def _check_ron_win(self, discard_136):
        gs       = self.gs
        tile_t34 = t136_to_t34(discard_136)
        test_hand = gs.hand_34()
        test_hand[tile_t34] += 1
        jikaze   = (gs.my_seat - gs.dealer_seat) % 4

        result = calculate_final_score(
            closed_counts   = test_hand,
            fixed_mentsu    = gs.fixed_mentsu[gs.my_seat],
            win_tile        = tile_t34,
            is_tsumo        = False,
            bakaze          = gs.bakaze,
            jikaze          = jikaze,
            is_oya          = (gs.my_seat == gs.dealer_seat),
            is_riichi       = gs.riichi_declared[gs.my_seat],
            dora_indicators = gs.dora_indicators,
            aka_dora_count  = gs.my_aka_count,
            is_ippatsu      = gs.my_ippatsu,
        )

        if result and result.get("score", 0) > 0:
            self.log(
                f"ロン和了! {tile_names[tile_t34]} "
                f"{result['han']}翻{result['fu']}符 {result.get('score',0)}点 "
                f"役: {result.get('yaku', [])}"
            )
            time.sleep(random.uniform(0.5, 1.0))
            self._send({"tag": "N", "type": 6})  # ロン
            return True
        return False

    # ----------------------------------------------------------
    # ポン / チー オファー
    # ----------------------------------------------------------
    def _handle_naki_offer(self):
        tile_136 = self._pending_discard_tile136
        t_attr   = self._pending_t_attr
        gs       = self.gs

        if tile_136 is None:
            self._send({"tag": "N"})
            return

        tile_t34 = t136_to_t34(tile_136)
        can_pon  = bool(t_attr & 2)
        can_chi  = bool(t_attr & 1)

        if self.naki_model is not None and (can_pon or can_chi):
            state  = gs.to_mahjong_state_v5()
            result = hybrid_inference.decide_naki_action(
                state, self.naki_model, tile_t34, self._pending_discard_from
            )
            best   = result.get("best", {})
            action = best.get("action", "skip")

            if action == "naki":
                naki_type    = best.get("naki_type", "skip")
                consumed_t34 = best.get("consumed_tiles", [])

                if naki_type == "pon" and can_pon:
                    matching = [t for t in gs.my_hand_136 if t // 4 == tile_t34]
                    if len(matching) >= 2:
                        hai0, hai1 = matching[0], matching[1]
                        self.log(f"ポン: {tile_names[tile_t34]}")
                        gs.my_hand_136.remove(hai0)
                        gs.my_hand_136.remove(hai1)
                        gs.fixed_mentsu[gs.my_seat].append(
                            [tile_t34, tile_t34, tile_t34]
                        )
                        self._send({"tag": "N", "type": 1, "hai0": hai0, "hai1": hai1})
                        time.sleep(random.uniform(0.5, 1.2))
                        self._do_discard()
                        return

                elif naki_type == "chi" and can_chi and self._pending_discard_from == 3:
                    if len(consumed_t34) >= 2:
                        ha = next(
                            (t for t in gs.my_hand_136 if t // 4 == consumed_t34[0]),
                            None
                        )
                        hb = next(
                            (t for t in gs.my_hand_136
                             if t // 4 == consumed_t34[1] and t != ha),
                            None
                        )
                        if ha is not None and hb is not None:
                            self.log(f"チー: {tile_names[tile_t34]}")
                            gs.my_hand_136.remove(ha)
                            gs.my_hand_136.remove(hb)
                            seq = sorted([tile_t34, consumed_t34[0], consumed_t34[1]])
                            gs.fixed_mentsu[gs.my_seat].append(seq)
                            self._send({"tag": "N", "type": 3, "hai0": ha, "hai1": hb})
                            time.sleep(random.uniform(0.5, 1.2))
                            self._do_discard()
                            return

        # パス
        self._send({"tag": "N"})

    # ----------------------------------------------------------
    # 対局結果表示
    # ----------------------------------------------------------
    def _show_final_result(self, msg):
        owari = str(msg.get("owari", ""))
        self.log(f"最終結果: {owari}")
        if owari:
            parts = owari.split(",")
            for i in range(min(4, len(parts) // 2)):
                score = int(parts[i * 2]) * 100
                uma   = float(parts[i * 2 + 1])
                mark  = "★" if i == self.gs.my_seat else " "
                self.log(f"  {mark} 席{i}: {score:,}点  {uma:+.1f}")


# ============================================================
# モデル読み込み
# ============================================================
def load_models(device="cpu"):
    print("[モデル読み込み中]")
    sd = torch.load(DISCARD_MODEL_PATH, map_location=device)
    num_blocks = max(int(k.split(".")[1]) for k in sd if k.startswith("res_blocks.")) + 1
    ai_model = MahjongResNet_UltimateV3(num_blocks=num_blocks)
    ai_model.load_state_dict(sd)
    ai_model.eval()
    ai_model.to(device)
    print(f"  打牌モデル: {DISCARD_MODEL_PATH} (num_blocks={num_blocks})")

    naki_model = None
    try:
        sd_naki = torch.load(NAKI_MODEL_PATH, map_location=device)
        naki_blocks = max(int(k.split(".")[1]) for k in sd_naki if k.startswith("res_blocks.")) + 1
        # V2(28ch) / V1(26ch) を自動判定
        in_ch = sd_naki["conv_in.weight"].shape[1]
        if in_ch == 28:
            naki_model = MahjongResNet_Naki_V2(num_blocks=naki_blocks)
            print(f"  鳴きモデル: {NAKI_MODEL_PATH} (V2 28ch)")
        else:
            naki_model = MahjongResNet_Naki(num_blocks=naki_blocks)
            print(f"  鳴きモデル: {NAKI_MODEL_PATH} (V1 26ch)")
        naki_model.load_state_dict(sd_naki)
        naki_model.eval()
        naki_model.to(device)
    except FileNotFoundError:
        print(f"  鳴きモデルなし ({NAKI_MODEL_PATH})")

    return ai_model, naki_model


# ============================================================
# エントリポイント
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="天鳳オンライン対局ボット")
    parser.add_argument("--username", default=None,
                        help="ゲストID (IDxxxxxxxx-xxxxxxxx形式)。未指定時は自動取得")
    parser.add_argument("--game-type", type=int, default=9,
                        help="対局タイプ (9=東南戦赤あり, 1=ボット戦)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 60)
    print("天鳳オンライン対局ボット (JSON WebSocket)")
    if args.username:
        print(f"ユーザー: {args.username}")
    else:
        print("ユーザー: 自動取得 (ゲストID)")
    print(f"対局タイプ: {args.game_type}")
    print("=" * 60)

    ai_model, naki_model = load_models()
    bot = TenhouBot(
        username   = args.username,
        ai_model   = ai_model,
        naki_model = naki_model,
        game_type  = args.game_type,
        verbose    = args.verbose,
    )
    bot.run()


if __name__ == "__main__":
    main()
