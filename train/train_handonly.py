#!/usr/bin/env python3
"""Train a "photo-only" discard model.

Features use ONLY what a photo of your own hand can provide:
hand tile counts, dora in hand, red fives, own meld count.
No opponents' discards, no scores, no riichi info, no turn number.

This is the version that can actually run inside the Streamlit app.
Expect lower accuracy than the full-state baseline (59.8%): that gap
is the price of not seeing the rest of the table.

Metrics include a "masked" variant, where predictions are restricted
to tile types actually in the hand - which is how the app will use
the model (it can only suggest tiles you hold).

Feature order (predict_discard.py must match):
    [hand counts x34, dora_in_hand, red_fives, meld_count]

Usage:
    python train_handonly.py discard_samples.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

TYPES = ([f"{n}m" for n in range(1, 10)]
         + [f"{n}p" for n in range(1, 10)]
         + [f"{n}s" for n in range(1, 10)]
         + ["E", "S", "W", "N", "P", "F", "C"])
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
WIND_NEXT = {"E": "S", "S": "W", "W": "N", "N": "E"}
DRAGON_NEXT = {"P": "F", "F": "C", "C": "P"}


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


def build_features(df):
    n = len(df)
    hand = np.zeros((n, 34), dtype=np.float32)
    extra = np.zeros((n, 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    hands = df["hand"].tolist()
    markers_col = df["dora_markers"].tolist()
    melds_col = df["melds"].tolist()
    discarded = df["discarded"].tolist()
    for i in range(n):
        for tok in str(hands[i]).split():
            hand[i][type_idx(tok)] += 1
        if markers_col[i]:
            for marker in str(markers_col[i]).split():
                extra[i, 0] += hand[i][TYPE_IDX[dora_type(marker)]]
        extra[i, 1] = sum(1 for tok in str(hands[i]).split()
                          if tok.endswith("r"))
        extra[i, 2] = 0 if not melds_col[i] else str(
            melds_col[i]).count(";") + 1
        y[i] = type_idx(discarded[i])
    return np.hstack([hand, extra]), y, hand


def topk_accuracy(proba, y_true, k):
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def masked_topk_accuracy(proba, hand_counts, y_true, k):
    p = proba.copy()
    p[hand_counts <= 0] = -1.0
    topk = np.argsort(p, axis=1)[:, -k:]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train photo-only model.")
    ap.add_argument("csv", help="discard_samples.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False)
    print(f"{len(df)} rows, {df['game_id'].nunique()} games")
    X, y, hand_counts = build_features(df)
    print(f"Features: {X.shape}")

    games = df["game_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(games)
    n_games = len(games)
    train_g = set(games[: int(0.8 * n_games)])
    val_g = set(games[int(0.8 * n_games): int(0.9 * n_games)])
    gid = df["game_id"]
    train_m = gid.isin(train_g).to_numpy()
    val_m = gid.isin(val_g).to_numpy()
    test_m = ~(train_m | val_m)
    print(f"Split by game: train {train_m.sum()}, val {val_m.sum()}, "
          f"test {test_m.sum()} rows")

    majority = np.bincount(y[train_m], minlength=34).argmax()
    print(f"Majority baseline ({TYPES[majority]}): "
          f"test top-1 = {(y[test_m] == majority).mean():.3f}")

    print("Training HistGradientBoosting (photo-only features)...")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63)
    model.fit(X[train_m], y[train_m])
    print(f"Trained in {time.time() - t0:.0f}s")

    for name, mask in (("val", val_m), ("test", test_m)):
        proba = model.predict_proba(X[mask])
        pred = proba.argmax(axis=1)
        print(f"{name}: top-1 = {accuracy_score(y[mask], pred):.3f}, "
              f"top-3 = {topk_accuracy(proba, y[mask], 3):.3f} | "
              f"masked top-1 = "
              f"{masked_topk_accuracy(proba, hand_counts[mask], y[mask], 1):.3f}, "
              f"masked top-3 = "
              f"{masked_topk_accuracy(proba, hand_counts[mask], y[mask], 3):.3f}")

    import joblib
    joblib.dump(model, "handonly_model.joblib")
    print("Model saved to handonly_model.joblib")


if __name__ == "__main__":
    main()
