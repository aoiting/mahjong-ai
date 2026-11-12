#!/usr/bin/env python3
"""Discard-pattern analysis on discard_samples.csv.

Two descriptive analyses (no model involved):

  1. Discard frequency -- how often each of the 34 tile types is the
     tile experts actually threw away, plus grouped views
     (honours / terminals / simples, and by suit).

  2. By turn -- how discards change as a hand progresses:
     the tsumogiri rate (discarding the tile just drawn) and the
     honours / terminals / simples mix, per turn and per turn band
     (early 1-6, mid 7-12, late 13+).

Red fives (5mr / 5pr / 5sr) are counted as their plain five.

Usage:
    python analyze_discards.py discard_samples.csv

Prints the full tables (paste them back for charting) and also saves
two small CSVs next to the input file:
    discard_frequency.csv   one row per tile type
    discard_by_turn.csv     one row per turn
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

TYPES = ([f"{n}m" for n in range(1, 10)] + [f"{n}p" for n in range(1, 10)]
         + [f"{n}s" for n in range(1, 10)]
         + ["E", "S", "W", "N", "P", "F", "C"])
NAMES = {"E": "East", "S": "South", "W": "West", "N": "North",
         "P": "White", "F": "Green", "C": "Red"}
HONOURS = {"E", "S", "W", "N", "P", "F", "C"}


def base_tile(tile):
    return tile[:-1] if tile.endswith("r") else tile


def label(tile):
    return f"{tile} {NAMES[tile]}" if tile in NAMES else tile


def category(tile):
    if tile in HONOURS:
        return "honours"
    return "terminals" if tile[0] in "19" else "simples"


def suit_of(tile):
    if tile in HONOURS:
        return "honours"
    return {"m": "man", "p": "pin", "s": "sou"}[tile[1]]


def pct(n, total):
    return 100.0 * n / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description="Discard frequency + by-turn analysis.")
    ap.add_argument("csv", help="discard_samples.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False)
    total = len(df)
    print(f"{total} rows")
    for col in ("discarded", "turn", "tsumogiri"):
        if col not in df.columns:
            sys.exit(f"Missing column {col!r} - is this discard_samples.csv?")

    def norm_bool(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v).strip().lower()

    tiles = [base_tile(str(t)) for t in df["discarded"]]
    turns = [int(t) for t in df["turn"]]
    tsumo = [norm_bool(v) == "true" for v in df["tsumogiri"]]

    # ---------- 1. frequency per tile type ----------
    freq = Counter(tiles)
    print("\n===== 1. Discard frequency per tile type (34 types) =====")
    print(f"{'tile':<12}{'count':>9}{'share':>8}")
    for t in TYPES:
        print(f"{label(t):<12}{freq.get(t, 0):>9}{pct(freq.get(t, 0), total):>7.2f}%")

    ranked = sorted(TYPES, key=lambda t: -freq.get(t, 0))
    print("\nTop 5 most discarded:  "
          + ", ".join(f"{label(t)} {pct(freq.get(t, 0), total):.2f}%" for t in ranked[:5]))
    print("Bottom 5 least discarded: "
          + ", ".join(f"{label(t)} {pct(freq.get(t, 0), total):.2f}%" for t in ranked[-5:]))

    print("\n-- grouped by category --")
    cats = Counter(category(t) for t in tiles)
    for c in ("honours", "terminals", "simples"):
        print(f"  {c:<10} {cats.get(c, 0):>8}  {pct(cats.get(c, 0), total):5.2f}%")
    print("-- grouped by suit --")
    suits = Counter(suit_of(t) for t in tiles)
    for s in ("man", "pin", "sou", "honours"):
        print(f"  {s:<10} {suits.get(s, 0):>8}  {pct(suits.get(s, 0), total):5.2f}%")

    # ---------- 2. by turn ----------
    overall_tsumo = pct(sum(tsumo), total)
    print(f"\nOverall tsumogiri rate: {overall_tsumo:.2f}%")

    def band(turn):
        return "early (1-6)" if turn <= 6 else ("mid (7-12)" if turn <= 12 else "late (13+)")

    bands = defaultdict(lambda: [0, 0, Counter()])
    per_turn = defaultdict(lambda: [0, 0, Counter()])
    for tile, turn, is_t in zip(tiles, turns, tsumo):
        for table, key in ((bands, band(turn)), (per_turn, turn)):
            table[key][0] += 1
            table[key][1] += 1 if is_t else 0
            table[key][2][category(tile)] += 1

    def show(title, table, keys):
        print(f"\n-- {title} --")
        print(f"{'turn':<13}{'n':>9}{'tsumogiri':>11}{'honours':>9}{'terminals':>11}{'simples':>9}")
        for k in keys:
            if k not in table:
                continue
            n, tg, cc = table[k]
            print(f"{str(k):<13}{n:>9}{pct(tg, n):>10.2f}%"
                  f"{pct(cc.get('honours', 0), n):>8.2f}%"
                  f"{pct(cc.get('terminals', 0), n):>10.2f}%"
                  f"{pct(cc.get('simples', 0), n):>8.2f}%")

    show("by turn band", bands, ["early (1-6)", "mid (7-12)", "late (13+)"])
    turn_keys = [t for t in sorted(per_turn) if t <= 18]
    if any(t > 18 for t in per_turn):
        merged = [0, 0, Counter()]
        for t in sorted(per_turn):
            if t > 18:
                merged[0] += per_turn[t][0]
                merged[1] += per_turn[t][1]
                merged[2].update(per_turn[t][2])
        per_turn["19+"] = merged
        turn_keys.append("19+")
    show("by turn number", per_turn, turn_keys)

    # ---------- save small CSVs ----------
    out_dir = os.path.dirname(os.path.abspath(args.csv))
    freq_path = os.path.join(out_dir, "discard_frequency.csv")
    pd.DataFrame([{"tile": t, "name": NAMES.get(t, t), "count": freq.get(t, 0),
                   "share_pct": round(pct(freq.get(t, 0), total), 3),
                   "category": category(t), "suit": suit_of(t)} for t in TYPES]
                 ).to_csv(freq_path, index=False)
    rows = []
    for k in turn_keys:
        n, tg, cc = per_turn[k]
        rows.append({"turn": k, "n": n, "tsumogiri_pct": round(pct(tg, n), 3),
                     "honours_pct": round(pct(cc.get("honours", 0), n), 3),
                     "terminals_pct": round(pct(cc.get("terminals", 0), n), 3),
                     "simples_pct": round(pct(cc.get("simples", 0), n), 3)})
    turn_path = os.path.join(out_dir, "discard_by_turn.csv")
    pd.DataFrame(rows).to_csv(turn_path, index=False)
    print(f"\nSaved: {freq_path}")
    print(f"Saved: {turn_path}")


if __name__ == "__main__":
    main()
