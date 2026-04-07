# parse_tenhou_log.py
import xml.etree.ElementTree as ET

class TenhouParser:
    """天鳳のXMLを時系列のイベントリストに変換する"""
    def __init__(self, xml_string):
        self.root = ET.fromstring(xml_string)
        self.events = []

    def parse(self):
        for node in self.root:
            tag = node.tag
            
            if tag == "INIT":
                seed = node.attrib.get("seed", "0,0,0,0,0,0").split(",")
                oya = int(node.attrib.get("oya", "0"))
                hands = {
                    0: [int(x) for x in node.attrib.get("hai0", "").split(",") if x],
                    1: [int(x) for x in node.attrib.get("hai1", "").split(",") if x],
                    2: [int(x) for x in node.attrib.get("hai2", "").split(",") if x],
                    3: [int(x) for x in node.attrib.get("hai3", "").split(",") if x],
                }
                self.events.append({
                    "type": "INIT",
                    "bakaze": int(seed[0]) // 4,
                    "kyoku": int(seed[0]) % 4,
                    "honba": int(seed[1]),
                    "kyotaku": int(seed[2]),
                    "dora_indicator": int(seed[5]) // 4,
                    "oya": oya,
                    "hands": hands,
                    "scores": [int(x) * 100 for x in node.attrib.get("ten", "0,0,0,0").split(",")]
                })
                
            elif tag.startswith(('T', 'U', 'V', 'W')) and tag[1:].isdigit():
                seat = {'T': 0, 'U': 1, 'V': 2, 'W': 3}[tag[0]]
                tile_136 = int(tag[1:])
                self.events.append({"type": "DRAW", "seat": seat, "tile_136": tile_136, "tile_34": tile_136 // 4})
                
            elif tag.startswith(('D', 'E', 'F', 'G')) and tag[1:].isdigit():
                seat = {'D': 0, 'E': 1, 'F': 2, 'G': 3}[tag[0]]
                tile_136 = int(tag[1:])
                self.events.append({"type": "DISCARD", "seat": seat, "tile_136": tile_136, "tile_34": tile_136 // 4})
                
            elif tag == "REACH":
                seat = int(node.attrib.get("who"))
                step = int(node.attrib.get("step"))
                self.events.append({"type": "REACH", "seat": seat, "step": step})
                
            elif tag == "N":
                # 今回は簡略化のため、鳴きが発生した局は以降の抽出をスキップするための目印にします
                seat = int(node.attrib.get("who"))
                self.events.append({"type": "CALL", "seat": seat})
                
        return self.events


class GlobalReplayTracker:
    """イベント列を1巡ずつ進め、卓全体のグローバル状態を正確に管理する"""
    def __init__(self):
        self.hands_136 = {i: [] for i in range(4)}
        self.discards_136 = {i: [] for i in range(4)}
        self.riichi_declared = {i: False for i in range(4)}
        self.scores = {i: 25000 for i in range(4)}
        self.dora_indicators = []
        self.bakaze = 0
        self.kyoku = 0
        self.oya = 0
        self.is_broken = False # 鳴きなど未対応のイベントが来て盤面が追えなくなったフラグ

    def apply_event(self, event):
        if self.is_broken: return
        
        e_type = event["type"]
        
        if e_type == "INIT":
            self.hands_136 = {i: event["hands"][i].copy() for i in range(4)}
            self.dora_indicators = [event["dora_indicator"]]
            self.scores = {i: event["scores"][i] for i in range(4)}
            self.bakaze = event["bakaze"]
            self.kyoku = event["kyoku"]
            self.oya = event["oya"]
            self.riichi_declared = {i: False for i in range(4)}
            self.discards_136 = {i: [] for i in range(4)}

        elif e_type == "DRAW":
            seat = event["seat"]
            self.hands_136[seat].append(event["tile_136"])

        elif e_type == "DISCARD":
            seat = event["seat"]
            tile = event["tile_136"]
            if tile in self.hands_136[seat]:
                self.hands_136[seat].remove(tile)
                self.discards_136[seat].append(tile)
            else:
                self.is_broken = True # 手牌にない牌が捨てられた（鳴き処理の未実装等が原因）
            
        elif e_type == "REACH":
            if event["step"] == 1:
                self.riichi_declared[event["seat"]] = True
                
        elif e_type == "CALL":
            # フェーズ1の最初は安全のため、鳴きが入った局は破棄（スキップ）する
            self.is_broken = True