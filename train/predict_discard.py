#!/usr/bin/env python3
"""Predict discards with the photo-only model (handonly_model.joblib).

This is the inference wrapper the Streamlit app can call: give it the
tiles in hand (plus the dora indicator(s) and how many melds are
open), and it returns the model's ranked discard suggestions,
restricted to tile types actually in the hand.

Importable:
    from predict_discard import predict
    predict(["5m", "5mr", "2p", ...], dora_markers=["8s"], meld_count=0)

CLI:
    python predict_discard.py --hand "5m 5mr 2p 3p 4p 8p 3s 5s 7s 8s 9s W W P" --dora 8s
"""
import argparse

import numpy as np

TYPES = ([f"{n}m" for n in range(1, 10)]
         + [f"{n}p" for n in range(1, 10)]
         + [f"{n}s" for n in range(1, 10)]
         + ["E", "S", "W", "N", "P", "F", "C"])
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
WIND_NEXT = {"E": "S", "S": "W", "W": "N", "N": "E"}
DRAGON_NEXT = {"P": "F", "F": "C", "C": "P"}

_model = None


def base_tile(tile: str) -> str:
    return tile[:-1] if tile.endswith("r") else tile


def type_idx(tile: str) -> int:
    return TYPE_IDX[base_tile(tile)]


def dora_type(marker: str) -> str:
    b = base_tile(marker)
    if b in WIND_NEXT:
        return WIND_NEXT[b]
    if b in DRAGON_NEXT:
        return DRAGON_NEXT[b]
    return f"{int(b[0]) % 9 + 1}{b[1]}"


def build_vector(hand_tiles, dora_markers, meld_count):
    """Feature order must match train_handonly.py exactly:
    [hand counts x34, dora_in_hand, red_fives, meld_count]."""
    hand = np.zeros(34, dtype=np.float32)
    for t in hand_tiles:
        hand[type_idx(t)] += 1
    dora_in_hand = sum(hand[TYPE_IDX[dora_type(m)]] for m in dora_markers)
    red_fives = sum(1 for t in hand_tiles if t.endswith("r"))
    vec = np.concatenate(
        [hand, [dora_in_hand, red_fives, meld_count]]).astype(np.float32)
    return vec, hand


def load_model(model_path="handonly_model.joblib"):
    global _model
    if _model is None:
        import joblib
        _model = joblib.load(model_path)
    return _model


def predict(hand_tiles, dora_markers=(), meld_count=0,
            model_path="handonly_model.joblib", top=5):
    """Return [(tile_type, probability), ...] best first,
    restricted to tile types present in the hand."""
    model = load_model(model_path)
    vec, hand_counts = build_vector(hand_tiles, dora_markers, meld_count)
    proba = model.predict_proba(vec[None, :])[0]
    results = []
    for cls, p in zip(model.classes_, proba):
        if hand_counts[cls] > 0:
            results.append((TYPES[int(cls)], float(p)))
    results.sort(key=lambda r: -r[1])
    return results[:top]


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict discards.")
    ap.add_argument("--hand", required=True,
                    help='Tiles in hand, e.g. "5m 5mr 2p 3p W W P"')
    ap.add_argument("--dora", default="",
                    help='Dora indicator tile(s), e.g. "8s"')
    ap.add_argument("--melds", type=int, default=0,
                    help="Number of open melds (default 0)")
    ap.add_argument("--model", default="handonly_model.joblib")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    hand_tiles = args.hand.split()
    dora_markers = args.dora.split() if args.dora else []
    if len(hand_tiles) % 3 != 2:
        print(f"(note: hand has {len(hand_tiles)} tiles; a discard "
              f"decision normally has 14/11/8/5/2)")
    results = predict(hand_tiles, dora_markers, args.melds,
                      args.model, args.top)
    print("Suggested discards:")
    for tile, p in results:
        note = ""
        if tile[0] == "5" and len(tile) == 2:
            red = tile + "r"
            if red in hand_tiles and tile in hand_tiles:
                note = "  (discard the plain one, keep the red five)"
            elif red in hand_tiles:
                note = "  (only the red five is in hand - think twice)"
        print(f"  {tile}: {p:.1%}{note}")


if __name__ == "__main__":
    main()
