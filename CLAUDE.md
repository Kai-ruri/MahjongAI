# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MahjongAI is a machine learning system that trains neural networks to play Japanese Mahjong (リーチ麻雀) by learning from real Tenhou (天鳳) game logs. It uses a phased training pipeline combining ResNet-based neural networks with hand-crafted heuristics.

## Running Scripts

There is no build system or package manager. Run scripts directly:

```bash
# Parse raw Tenhou game logs into event sequences
python parse_tenhou_log.py

# Extract labeled datasets from parsed logs
python dataset_extractor.py
# or
python build_supervised_dataset.py

# Training phases (iterative, numbered)
python run_phase5_pon.py
python run_phase6_riichi.py
python run_phase10g_integration.py

# Error analysis
python error_analysis_chi.py
python error_analysis_riichi.py
python error_analysis_oshibiki.py

# Self-play evaluation
python selfplay_minimal.py

# Generate model checkpoints
python generate_pth.py
python generate_ev_pth.py
```

**Required dependencies:** `torch`, `numpy`, `scikit-learn` (no requirements.txt — install manually)

## Architecture

### Data Pipeline
```
logs/*.html.gz (Tenhou XML logs)
    → parse_tenhou_log.py     # XML → event sequences
    → dataset_extractor.py    # events → game state snapshots
    → build_supervised_dataset.py  # snapshots → labeled .pkl datasets
```

### Core Files

| File | Purpose |
|---|---|
| `mahjong_engine.py` | Complete Mahjong rule engine — hand decomposition, yaku evaluation, scoring (fu/han), state management via `MahjongStateV5` |
| `mahjong_model.py` | Neural network definitions: `MahjongResNet_UltimateV3` (discard + riichi heads), `MahjongResNet_Naki` (call decisions) |
| `hybrid_inference.py` | Inference utilities — shanten calculation, effective tile counting, visible tile tracking, risk assessment |
| `rerank_logics.py` | Post-NN heuristics — reranks tile choices based on defense, hand structure, ukeire, danger tile penalties |
| `selfplay_minimal.py` | Full 4-player game simulation using all models; decision pipeline: NN → reranking → action |

### Model Files (`.pth`)
- `mahjong_ultimate_ai_v*.pth` — Discard decision models (main family)
- `riichi_best.pth` — Riichi declaration
- `call_best.pth` / `mahjong_naki_model_master.pth` — Meld/call decisions (PON/CHI)
- `discard_best.pth` — Optimized discard
- `ev_plus_best.pth` / `ev_minus_best.pth` — Expected value situational models
- `oshibiki_best.pth` — Late-game aggressive push decisions

### Neural Network Architecture
- **Input**: 25-channel (discard model) or 26-channel (naki model) 1D representation of game state
- **Backbone**: Conv1d → ResidualBlocks → Dense layers with BatchNorm
- **Output**: 34-class softmax (tile types) for discard; binary for riichi/call decisions

### Training Phase Structure
Phase scripts are named `run_phase<N>_<action>.py` and progress:
- **Phase 5**: PON/CHI call decisions
- **Phase 6**: RIICHI declaration
- **Phase 7**: OSHIBIKI (aggressive pushing)
- **Phase 8–9**: EV (expected value) models
- **Phase 10+**: Integration and iterative refinement (variants 10a–10g)

Dataset files (`.pkl`) store `(feature_tensor, label)` pairs for each action type.

## Domain Notes

- Tile encoding: 0–8 = 1–9m (characters), 9–17 = 1–9p (circles), 18–26 = 1–9s (bamboo), 27–33 = honor tiles (winds + dragons)
- **Shanten**: The `hybrid_inference.py` shanten calculation supports open hands (副露対応)
- **Reranking**: `rerank_logics.py` applies domain heuristics *after* NN output — changes to game strategy logic go here
- Code and comments are a mix of Japanese and English
