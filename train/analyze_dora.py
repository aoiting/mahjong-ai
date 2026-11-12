#!/usr/bin/env python3
"""Dora discard behaviour on discard_samples.csv.

Descriptive analysis (no model involved). For every discard decision
we know the dora indicator(s) of that moment (dora_markers), the hand
before the discard (hand_before), the tile just drawn (drawn_tile) and
the tile actually discarded (discarded). From the indicator we derive
the dora tile itself: the NEXT tile after the indicator
(suits 1->2 ... 9->1, winds E->S->W->N->E, dragons P->F->C->P).
Red fives (5mr / 5pr / 5sr) are dora in their own right.

Three questions:

  1. What share of all discards are dora tiles?
     (overall and by turn band: early 1-6 / mid 7-12 / late 13+)

  2. When the hand actually contains a dora, how often do experts
     throw the dora away instead of keeping it and discarding
     something else? Compared against the rate expected if the
     discard were picked uniformly at random from the hand.

  3. When a player draws a dora, how often do they throw it straight
     back (tsumogiri the dora) instead of keeping it?

Plus a red-five section: red fives discarded vs plain fives, and how
often a hand containing a red five gives it up.

Tile lists are parsed with a regex, so hand_before / dora_markers may
be space separated ("1m 2p ..."), a python-list repr
("['1m', '2p']") or compact -- all parse the same way.

Usage:
    python analyze_dora.py discard_samples.csv

Prints the full tables (paste them back for charting) and saves
dora_summary.csv next to the input file.
"""
import argparse
import ast
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

TILE_RE = re.compile(r"[1-9][mps]r?|[ESWNPCF]")
WINDS = ["E", "S", "W", "N"]
DRAGONS = ["P", "F", "C"]


def parse_tiles(value):
    """All tile codes found in a raw cell, as a list of strings."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(t) for t in parsed]
    except (ValueError, SyntaxError):
        pass
    return TILE_RE.findall(s)


def base_tile(tile):
    return tile[:-1] if tile.endswith("r") else tile


def is_red(tile):
    return tile.endswith("r")


def next_tile(tile):
    """Dora shown by an indicator tile."""
    t = base_tile(tile)
    if t in WINDS:
        return WINDS[(WINDS.index(t) + 1) % 4]
    if t in DRAGONS:
        return DRAGONS[(DRAGONS.index(t) + 1) % 3]
    n = int(t[0])
    return f"{n % 9 + 1}{t[1]}"


def dora_types(markers):
    return {next_tile(m) for m in markers if m}


def is_dora_tile(tile, doras):
    return is_red(tile) or base_tile(tile) in doras


def pct(n, total):
    return 100.0 * n / total if total else 0.0


def band(turn):
    return "early (1-6)" if turn <= 6 else ("mid (7-12)" if turn <= 12
                                            else "late (13+)")
BANDS = ["early (1-6)", "mid (7-12)", "late (13+)"]


def main():
    ap = argparse.ArgumentParser(description="Dora discard behaviour analysis.")
    ap.add_argument("csv", help="discard_samples.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False)
    total = len(df)
    print(f"{total} rows")

    def pick(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_disc = pick("discarded", "discard", "dahai")
    col_turn = pick("turn")
    col_markers = pick("dora_markers", "dora_marker", "dora_indicators")
    col_hand = pick("hand_before", "hand", "tehai", "hand_tiles")
    col_drawn = pick("drawn_tile", "drawn", "tsumo_tile", "draw_tile")
    missing = [name for name, col in
               (("discarded", col_disc), ("turn", col_turn),
                ("dora_markers", col_markers), ("hand", col_hand),
                ("drawn", col_drawn)) if col is None]
    if missing:
        print(f"All columns found: {list(df.columns)}")
        sys.exit(f"Missing column(s) {missing} - is this discard_samples.csv?")
    print(f"columns used: hand={col_hand!r} drawn={col_drawn!r} "
          f"markers={col_markers!r} discarded={col_disc!r} turn={col_turn!r}")

    # ---------- diagnostics: make the raw formats visible ----------
    print("\n--- format check (first 3 rows, raw values) ---")
    for col in (col_markers, col_hand, col_drawn, col_disc):
        print(f"  {col}: {[str(v) for v in df[col].head(3)]}")
    n_markers = Counter()
    hand_sizes = Counter()
    empty_markers = 0

    # ---------- per-row facts ----------
    rows = []
    for markers_raw, hand_raw, drawn_raw, disc_raw, turn_raw in zip(
            df[col_markers], df[col_hand], df[col_drawn],
            df[col_disc], df[col_turn]):
        markers = parse_tiles(markers_raw)
        if not markers:
            empty_markers += 1
        doras = dora_types(markers)
        hand = parse_tiles(hand_raw)
        drawn_list = parse_tiles(drawn_raw)
        drawn = drawn_list[0] if drawn_list else ""
        # decision hand: hand_before plus the drawn tile if not in yet
        decision = list(hand)
        if drawn and len(decision) % 3 == 1:  # 13 tiles -> add the draw
            decision.append(drawn)
        hand_sizes[len(hand)] += 1
        n_markers[len(markers)] += 1

        discarded = str(disc_raw).strip()
        dora_in_hand = [t for t in decision if is_dora_tile(t, doras)]
        red_in_hand = [t for t in decision if is_red(t)]
        rows.append({
            "turn": int(turn_raw),
            "band": band(int(turn_raw)),
            "discarded": discarded,
            "disc_is_dora": is_dora_tile(discarded, doras),
            "disc_is_red": is_red(discarded),
            "n_hand": len(decision),
            "n_dora_hand": len(dora_in_hand),
            "has_red": bool(red_in_hand),
            "disc_is_the_red": is_red(discarded) and bool(red_in_hand),
            "drawn": drawn,
            "drawn_is_dora": bool(drawn) and is_dora_tile(drawn, doras),
            "threw_drawn": bool(drawn) and discarded == drawn,
        })

    print(f"  markers per row: {dict(sorted(n_markers.items()))}")
    print(f"  hand_before sizes: {dict(sorted(hand_sizes.items()))}")
    print(f"  rows with no dora marker parsed: {empty_markers}")

    r = pd.DataFrame(rows)

    # ---------- 1. share of discards that are dora ----------
    print("\n===== 1. Discards that are dora tiles =====")
    nd = int(r["disc_is_dora"].sum())
    print(f"overall: {nd} / {total} = {pct(nd, total):.2f}%")
    for b in BANDS:
        sub = r[r["band"] == b]
        n = int(sub["disc_is_dora"].sum())
        print(f"  {b:<12} n={len(sub):>7}  dora discards {n:>6}  "
              f"{pct(n, len(sub)):5.2f}%")

    # ---------- 2. hand contains a dora: throw it or keep it? ----------
    print("\n===== 2. Hand contains a dora: throw the dora? =====")
    has = r[r["n_dora_hand"] > 0]
    threw = int(has["disc_is_dora"].sum())
    exp = 100.0 * (has["n_dora_hand"] / has["n_hand"]).mean()
    print(f"decisions with a dora in hand: {len(has)}")
    print(f"  threw the dora:      {threw}  = {pct(threw, len(has)):.2f}%")
    print(f"  kept it (threw other): {len(has) - threw}  "
          f"= {pct(len(has) - threw, len(has)):.2f}%")
    print(f"  if discarding at random from the hand, expected throw-dora "
          f"rate: {exp:.2f}%")
    for b in BANDS:
        sub = has[has["band"] == b]
        if len(sub) == 0:
            continue
        t = int(sub["disc_is_dora"].sum())
        e = 100.0 * (sub["n_dora_hand"] / sub["n_hand"]).mean()
        print(f"  {b:<12} n={len(sub):>7}  threw {pct(t, len(sub)):5.2f}%  "
              f"(random {e:5.2f}%)")

    # ---------- 3. drew a dora: throw it straight back? ----------
    print("\n===== 3. Drew a dora: throw it straight back? =====")
    drew = r[r["drawn_is_dora"]]
    back = int(drew["threw_drawn"].sum())
    print(f"decisions where the drawn tile was a dora: {len(drew)}")
    print(f"  threw it straight back: {back}  = {pct(back, len(drew)):.2f}%")
    print(f"  kept it (threw other):  {len(drew) - back}  "
          f"= {pct(len(drew) - back, len(drew)):.2f}%")
    for b in BANDS:
        sub = drew[drew["band"] == b]
        if len(sub) == 0:
            continue
        t = int(sub["threw_drawn"].sum())
        print(f"  {b:<12} n={len(sub):>6}  threw back {pct(t, len(sub)):5.2f}%")

    # ---------- 4. red fives ----------
    print("\n===== 4. Red fives =====")
    red_disc = int(r["disc_is_red"].sum())
    plain5 = int(((r["discarded"].map(base_tile).isin(["5m", "5p", "5s"]))
                  & (~r["disc_is_red"])).sum())
    print(f"red-five discards:   {red_disc}  = {pct(red_disc, total):.3f}% "
          f"of all discards")
    print(f"plain-five discards: {plain5}  = {pct(plain5, total):.2f}% "
          f"of all discards")
    hr = r[r["has_red"]]
    gave = int(hr["disc_is_the_red"].sum())
    print(f"decisions with a red five in hand: {len(hr)}")
    print(f"  gave up the red five: {gave}  = {pct(gave, len(hr)):.2f}%")

    # ---------- save summary ----------
    out_dir = os.path.dirname(os.path.abspath(args.csv))
    out_path = os.path.join(out_dir, "dora_summary.csv")
    summary = [
        {"metric": "discard_is_dora_pct", "scope": "overall",
         "value": round(pct(nd, total), 3), "n": total},
        {"metric": "hand_has_dora_throw_pct", "scope": "overall",
         "value": round(pct(threw, len(has)), 3), "n": len(has)},
        {"metric": "hand_has_dora_random_pct", "scope": "overall",
         "value": round(exp, 3), "n": len(has)},
        {"metric": "drew_dora_throwback_pct", "scope": "overall",
         "value": round(pct(back, len(drew)), 3), "n": len(drew)},
        {"metric": "red_five_discard_pct", "scope": "overall",
         "value": round(pct(red_disc, total), 3), "n": total},
        {"metric": "hand_has_red_giveup_pct", "scope": "overall",
         "value": round(pct(gave, len(hr)), 3), "n": len(hr)},
    ]
    pd.DataFrame(summary).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
