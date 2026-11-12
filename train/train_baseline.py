#!/usr/bin/env python3
"""Train a first discard-prediction baseline on discard_samples.csv.

Framing: one row = one decision. Features describe the state
(hand tile counts, visible tile counts, dora in hand, turn, scores,
riichi situation); the label is the TYPE of tile discarded
(34 classes; a red five counts as its plain type).

Games are split 80/10/10 by game_id, never by row, so no game can
leak into both train and test.

Known v1 limitations (fine for a baseline):
- the model may predict a tile type that is not in the hand
  (a per-candidate-tile model fixes this later);
- opponents' melds are not in the dataset, only their discards are,
  so "visible tiles" misses the tiles inside opponents' melds.

Needs: pandas, scikit-learn
    pip install pandas scikit-learn

Usage:
    python train_baseline.py discard_samples.csv
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


def add_counts(vec, tile_str):
    if tile_str:
        for tok in str(tile_str).split():
            vec[type_idx(tok)] += 1


def meld_tiles(meld_str):
    tiles = []
    if meld_str:
        for part in str(meld_str).split(";"):
            part = part.strip()
            if ":" in part:
                tiles += part.split(":", 1)[1].split()
    return tiles


def build_features(df):
    n = len(df)
    hand = np.zeros((n, 34), dtype=np.float32)
    visible = np.zeros((n, 34), dtype=np.float32)
    dora_in_hand = np.zeros(n, dtype=np.float32)
    red_fives = np.zeros(n, dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    hands = df["hand"].tolist()
    melds_col = df["melds"].tolist()
    markers_col = df["dora_markers"].tolist()
    rivers = [df[f"discards_{i}"].tolist() for i in range(4)]
    discarded = df["discarded"].tolist()

    for i in range(n):
        add_counts(hand[i], hands[i])
        for river in rivers:
            add_counts(visible[i], river[i])
        for tok in meld_tiles(melds_col[i]):
            visible[i][type_idx(tok)] += 1
        add_counts(visible[i], markers_col[i])
        if markers_col[i]:
            for marker in str(markers_col[i]).split():
                dora_in_hand[i] += hand[i][TYPE_IDX[dora_type(marker)]]
        red_fives[i] = sum(1 for tok in str(hands[i]).split()
                           if tok.endswith("r"))
        y[i] = type_idx(discarded[i])
        if i % 50000 == 0 and i:
            print(f"  features: {i}/{n} rows")

    # Score / riichi features, vectorised.
    actors = df["actor"].astype(int).to_numpy()
    all_scores = np.array(
        [[int(s) for s in str(x).split()] for x in df["scores"]])
    score_self = all_scores[np.arange(n), actors]
    leader = all_scores.max(axis=1)
    rank = (all_scores > score_self[:, None]).sum(axis=1) + 1
    opp_riichi = np.array(
        [sum(int(c) for j, c in enumerate(str(f).zfill(4)) if j != a)
         for f, a in zip(df["riichi_flags"], actors)], dtype=np.float32)

    extra = np.column_stack([
        dora_in_hand,
        red_fives,
        df["turn"].astype(int),
        df["honba"].astype(int),
        df["kyotaku"].astype(int),
        [int(k[1:]) for k in df["kyoku"]],
        [0 if not m else str(m).count(";") + 1 for m in melds_col],
        score_self,
        rank,
        leader - score_self,
        (df["actor"].astype(int) == df["oya"].astype(int)).astype(int),
        opp_riichi,
        (df["self_riichi"] == "true").astype(int),
    ]).astype(np.float32)

    return np.hstack([hand, visible, extra]), y


def topk_accuracy(proba, y_true, k):
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train discard baseline.")
    ap.add_argument("csv", help="discard_samples.csv")
    args = ap.parse_args()

    print("Loading...")
    df = pd.read_csv(args.csv, keep_default_na=False)
    print(f"{len(df)} rows, {df['game_id'].nunique()} games")

    print("Building features...")
    t0 = time.time()
    X, y = build_features(df)
    print(f"Features: {X.shape}, built in {time.time() - t0:.0f}s")

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

    # Baseline 0: always predict the most common discard type.
    majority = np.bincount(y[train_m], minlength=34).argmax()
    acc_majority = float((y[test_m] == majority).mean())
    print(f"Majority baseline ({TYPES[majority]}): "
          f"test top-1 = {acc_majority:.3f}")

    print("Training HistGradientBoosting...")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63)
    model.fit(X[train_m], y[train_m])
    print(f"Trained in {time.time() - t0:.0f}s")

    for name, mask in (("val", val_m), ("test", test_m)):
        proba = model.predict_proba(X[mask])
        pred = proba.argmax(axis=1)
        print(f"{name}: top-1 = {accuracy_score(y[mask], pred):.3f}, "
              f"top-3 = {topk_accuracy(proba, y[mask], 3):.3f}")

    try:
        import joblib
        joblib.dump(model, "baseline_model.joblib")
        print("Model saved to baseline_model.joblib")
    except Exception as exc:  # noqa: BLE001
        print(f"(model not saved: {exc})")


if __name__ == "__main__":
    main()
