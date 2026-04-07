# dataset_extractor.py
import xml.etree.ElementTree as ET
import numpy as np
from mahjong_engine import tile_names, MahjongStateV5

# 🌟 追加: hybrid_inference.py から正確な可視牌カウント関数をインポート
from hybrid_inference import build_visible_tiles34

class TenhouParser:
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
                    "type": "INIT", "bakaze": int(seed[0]) // 4, "kyoku": int(seed[0]) % 4,
                    "honba": int(seed[1]), "kyotaku": int(seed[2]), "dora_indicator": int(seed[5]),
                    "oya": oya, "hands": hands, "scores": [int(x) * 100 for x in node.attrib.get("ten", "0,0,0,0").split(",")]
                })
            elif tag.startswith(('T', 'U', 'V', 'W')) and tag[1:].isdigit():
                seat = {'T': 0, 'U': 1, 'V': 2, 'W': 3}[tag[0]]
                self.events.append({"type": "DRAW", "seat": seat, "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
            elif tag.startswith(('D', 'E', 'F', 'G', 'd', 'e', 'f', 'g')) and tag[1:].isdigit():
                seat = {'D':0, 'E':1, 'F':2, 'G':3, 'd':0, 'e':1, 'f':2, 'g':3}[tag[0]]
                self.events.append({"type": "DISCARD", "seat": seat, "tile_136": int(tag[1:]), "tile_34": int(tag[1:]) // 4})
            elif tag == "REACH":
                self.events.append({"type": "REACH", "seat": int(node.attrib.get("who")), "step": int(node.attrib.get("step"))})
            elif tag == "N":
                self.events.append({"type": "CALL", "seat": int(node.attrib.get("who"))})
        return self.events

class GlobalReplayTracker:
    def __init__(self):
        self.is_broken = False
        self.honba = 0
        self.kyotaku = 0

    def apply_event(self, event):
        e_type = event["type"]
        if e_type == "INIT":
            self.is_broken = False 
            self.hands_136 = {i: event["hands"][i].copy() for i in range(4)}
            self.dora_indicators = [event["dora_indicator"]]
            self.scores = {i: event["scores"][i] for i in range(4)}
            self.bakaze = event["bakaze"]
            self.kyoku = event["kyoku"]
            self.oya = event["oya"]
            self.honba = event["honba"]       
            self.kyotaku = event["kyotaku"]   
            self.riichi_declared = {i: False for i in range(4)}
            self.discards_136 = {i: [] for i in range(4)}
            return
            
        if self.is_broken: return
        
        if e_type == "DRAW":
            seat = event["seat"]
            self.hands_136[seat].append(event["tile_136"])
        elif e_type == "DISCARD":
            seat = event["seat"]
            tile = event["tile_136"]
            if tile in self.hands_136[seat]:
                self.hands_136[seat].remove(tile)
                self.discards_136[seat].append(tile)
            else:
                self.is_broken = True
        elif e_type == "REACH":
            if event["step"] == 1: 
                seat = event["seat"]
                self.riichi_declared[seat] = True
                self.scores[seat] -= 1000  
                self.kyotaku += 1          
        elif e_type == "CALL":
            self.is_broken = True

def build_local_state(tracker: GlobalReplayTracker, target_seat: int) -> MahjongStateV5:
    state = MahjongStateV5()
    for t_136 in tracker.hands_136[target_seat]: 
        state.add_tile(0, t_136) 
    for pov in range(4):
        actual_seat = (target_seat + pov) % 4
        for t_136 in tracker.discards_136[actual_seat]: 
            state.discards[pov].append(t_136 // 4)
            
    state.dora_indicators = [d // 4 for d in tracker.dora_indicators]
    state.bakaze = tracker.bakaze
    state.jikaze = (target_seat - tracker.oya + 4) % 4
    state.honba = tracker.honba       
    state.kyotaku = tracker.kyotaku   
    
    for pov in range(4):
        actual_seat = (target_seat + pov) % 4
        state.riichi_declared[pov] = tracker.riichi_declared[actual_seat]
        state.scores[pov] = tracker.scores[actual_seat]
    return state

def extract_dataset(xml_string, log_id="unknown"):
    parser = TenhouParser(xml_string)
    events = parser.parse()
    tracker = GlobalReplayTracker()
    dataset_records = []
    
    for i in range(len(events) - 1):
        current_event = events[i]
        next_event = events[i+1]
        tracker.apply_event(current_event)
        
        if tracker.is_broken: continue
            
        if current_event["type"] == "DRAW" and next_event["type"] == "DISCARD":
            if current_event["seat"] == next_event["seat"]:
                target_seat = current_event["seat"]
                label_discard_34 = next_event["tile_34"]
                
                # 局所的な盤面状態（MahjongStateV5）を作成
                local_state = build_local_state(tracker, target_seat)
                tensor_input = local_state.to_tensor(skip_logic=True)
                
                # 🌟 フェーズ3後半用：メタデータの抽出！
                visible_tiles = build_visible_tiles34(local_state)
                is_defense = any(local_state.riichi_declared.values())
                
                dataset_records.append({
                    "meta_log_id": log_id, 
                    "meta_kyoku": tracker.kyoku,
                    "meta_turn": len(tracker.discards_136[target_seat]), 
                    "meta_seat": target_seat,
                    "tensor": tensor_input, 
                    "label": label_discard_34,
                    # 🌟 追加メタデータ
                    "meta_visible_tiles": visible_tiles, 
                    "meta_is_defense": is_defense        
                })
    return dataset_records