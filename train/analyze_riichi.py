#!/usr/bin/env python3
"""Riichi-rate analysis (Option A, v2) on discard_samples.csv.

Finds every decision point where the player COULD have declared riichi
with the discard they actually made:
  - closed hand (no melds at all; ankan samples are dropped)
  - not already in riichi
  - score >= 1000 (riichi costs 1000)
  - at least 4 tiles left in the wall (approximated from river sizes)
  - the discard leaves the hand in tenpai (shanten 0)

For those points the label is riichi_dahai (did the pro declare?).

v2 prints TWO views:
  1. all decision points (a damaten hand that stays in tenpai is
     counted once per turn, a riichi hand only once - this deflates
     the rate), and
  2. FIRST tenpai point per player per kyoku only - the moment the
     player first reaches tenpai, which is the situation the app
     actually advises on. Prefer this view for app lookups.

Note: opponent melds are not in dataset v1, so "tiles remaining" for
waits is slightly overestimated (melded tiles counted as unseen).

Usage:
    python -m pip install mahjong
    python analyze_riichi.py discard_samples.csv
"""
import argparse
import os
import sys
from collections import defaultdict

import pandas as pd

try:
    from mahjong.shanten import Shanten
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig
    from mahjong.constants import EAST, SOUTH, WEST, NORTH
except ImportError:
    sys.exit("Missing library. Install with:  python -m pip install mahjong")

TYPES = ([f"{n}m" for n in range(1, 10)] + [f"{n}p" for n in range(1, 10)]
         + [f"{n}s" for n in range(1, 10)]
         + ["E", "S", "W", "N", "P", "F", "C"])
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
WIND_CONST = {"E": EAST, "S": SOUTH, "W": WEST, "N": NORTH}
WIND_NEXT = {"E": "S", "S": "W", "W": "N", "N": "E"}
DRAGON_NEXT = {"P": "F", "F": "C", "C": "P"}

SHANTEN = Shanten()
CALC = HandCalculator()


def base_tile(tile):
    return tile[:-1] if tile.endswith("r") else tile


def counts34(tokens):
    c = [0] * 34
    for tok in tokens:
        c[TYPE_IDX[base_tile(tok)]] += 1
    return c


def dora_type(marker):
    b = base_tile(marker)
    if b in WIND_NEXT:
        return WIND_NEXT[b]
    if b in DRAGON_NEXT:
        return DRAGON_NEXT[b]
    return f"{int(b[0]) % 9 + 1}{b[1]}"


def find_waits(counts13):
    """Tile type indices that complete the hand (shanten -> -1)."""
    waits = []
    for i in range(34):
        if counts13[i] >= 4:
            continue
        trial = counts13[:]
        trial[i] += 1
        if SHANTEN.calculate_shanten(trial) == -1:
            waits.append(i)
    return waits


def has_ron_yaku(counts13, waits, jikaze, bakaze):
    """True if any wait wins by ron with a yaku, WITHOUT riichi.

    Aka fives are converted to plain fives (yaku are unaffected) and
    no dora indicators are passed (dora are not yaku either).
    Dealer needs no flag: the dealer's seat wind IS East.
    """
    tiles136 = []
    for idx, n in enumerate(counts13):
        tiles136.extend(idx * 4 + k for k in range(n))
    for w in waits:
        win136 = w * 4 + counts13[w]  # next free copy index
        cfg = HandConfig(is_riichi=False, is_tsumo=False,
                         player_wind=WIND_CONST[jikaze],
                         round_wind=WIND_CONST[bakaze])
        res = CALC.estimate_hand_value(tiles136 + [win136], win136,
                                       melds=[], dora_indicators=[],
                                       config=cfg)
        if res.error is None and res.han and res.han > 0:
            return True
    return False


def print_group(title, groups):
    print(f"\n== {title} ==")
    for label in sorted(groups):
        n, r = groups[label]
        if n == 0:
            continue
        flag = "  (n<30, 參考價值低)" if n < 30 else ""
        print(f"  {label:<28} n={n:>6}  立直 {r:>6}  = {100.0 * r / n:5.1f}%{flag}")


def tabulate(points, heading):
    print(f"\n########## {heading} ##########")
    n = len(points)
    r = sum(p["declared"] for p in points)
    if n == 0:
        print("  (冇樣本)")
        return
    print(f"決策點: {n}  (真立直: {r})  整體立直率: {100.0 * r / n:.1f}%")
    for title, key in [("等牌質素 (種數 x 剩餘張數)", "wait_key"),
                       ("巡目", "turn_key"),
                       ("有冇役", "yaku_key"),
                       ("手牌寶牌數", "dora_key"),
                       ("莊家 vs 閒家", "dealer_key")]:
        g = defaultdict(lambda: [0, 0])
        for p in points:
            g[p[key]][0] += 1
            g[p[key]][1] += p["declared"]
        print_group(title, g)


def main():
    ap = argparse.ArgumentParser(description="Riichi-rate analysis.")
    ap.add_argument("csv", help="discard_samples.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False)
    print(f"{len(df)} rows")

    # pandas may parse "true"/"false" columns as bool - normalize to str
    def norm_bool(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v).strip().lower()
    df["self_riichi"] = df["self_riichi"].map(norm_bool)
    df["riichi_dahai"] = df["riichi_dahai"].map(norm_bool)

    points = []
    seen_first = set()
    n_ankan_dropped = 0

    for i, row in df.iterrows():
        if i % 50000 == 0 and i:
            print(f"  ...{i} rows scanned, {len(points)} eligible so far")
        # ---- cheap eligibility filters ----
        if row["self_riichi"] != "false":
            continue
        melds = str(row["melds"])
        if melds:
            if melds.startswith("ankan:") and ";" not in melds:
                n_ankan_dropped += 1
            continue  # strictly closed hands only
        if int(row["score_self"]) < 1000:
            continue
        rivers = [str(row[f"discards_{k}"]).split() for k in range(4)]
        if sum(len(r) for r in rivers) > 66:  # <4 drawable tiles left
            continue

        hand14 = str(row["hand"]).split()
        hand13 = hand14[:]
        hand13.remove(row["discarded"])
        if len(hand13) != 13:
            continue
        counts13 = counts34(hand13)
        if SHANTEN.calculate_shanten(counts13) != 0:
            continue

        # ---- eligible riichi decision point ----
        declared = 1 if row["riichi_dahai"] == "true" else 0

        waits = find_waits(counts13)
        visible = counts13[:]
        for r in rivers:
            for tok in r:
                visible[TYPE_IDX[base_tile(tok)]] += 1
        markers = str(row["dora_markers"]).split()
        for mk in markers:
            visible[TYPE_IDX[base_tile(mk)]] += 1
        wait_tiles = sum(max(0, 4 - visible[w]) for w in waits)
        wait_types = len(waits)

        turn = int(row["turn"])
        turn_band = "早巡 (<=6)" if turn <= 6 else ("中巡 (7-12)" if turn <= 12 else "晚巡 (13+)")
        tiles_band = "剩 1-4 張" if wait_tiles <= 4 else ("剩 5-8 張" if wait_tiles <= 8 else "剩 9+ 張")
        types_label = "3+ 種等" if wait_types >= 3 else f"{wait_types} 種等"

        yaku = has_ron_yaku(counts13, waits, row["jikaze"],
                            str(row["kyoku"])[0])
        dora_n = sum(counts34(hand14)[TYPE_IDX[dora_type(mk)]] for mk in markers)
        dora_n += sum(1 for tok in hand14 if tok.endswith("r"))

        first_key = (row["game_id"], int(row["round_index"]), int(row["actor"]))
        is_first = first_key not in seen_first
        seen_first.add(first_key)

        points.append({
            "game_id": row["game_id"], "round_index": int(row["round_index"]),
            "actor": int(row["actor"]), "is_first": is_first,
            "wait_key": f"{types_label} / {tiles_band}",
            "turn_key": turn_band,
            "yaku_key": "有役" if yaku else "無役 (得自摸/立直)",
            "dora_key": "寶牌 0" if dora_n == 0 else ("寶牌 1" if dora_n == 1 else "寶牌 2+"),
            "dealer_key": "莊家" if int(row["actor"]) == int(row["oya"]) else "閒家",
            "wait_types": wait_types, "wait_tiles": wait_tiles,
            "turn": turn, "has_yaku": yaku, "dora": dora_n,
            "declared": declared,
        })

    print(f"\n暗槓樣本 (剔除): {n_ankan_dropped}")
    if not points:
        sys.exit("No eligible riichi decision points found - check filters.")

    print(f"\n===== 結果 =====")
    tabulate(points, "全部可立直決策點 (默聽手會重複計)")
    tabulate([p for p in points if p["is_first"]],
             "首次聽牌點 (每局每人只計第一次 - app 查表用呢份)")

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                            "riichi_points.csv")
    pd.DataFrame(points).to_csv(out_path, index=False)
    print(f"\n每個決策點嘅明細已存: {out_path} ({len(points)} rows)")


if __name__ == "__main__":
    main()
