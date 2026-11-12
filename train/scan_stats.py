#!/usr/bin/env python3
"""Scan Tenhou mjlog files and summarise game results.

Everything here comes straight from the log's result tags
(AGARI / RYUUKYOKU / REACH / N / discard tags) plus the UN element,
so no hand tracking or meld decoding is needed.

Usage:
    python scan_stats.py mjlog
    python scan_stats.py mjlog --out game_stats.csv

Writes one row per game to CSV and prints aggregate totals.
Note: the yaku attribute alternates (yaku_id, han), so total han
is the sum of every second value.
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def han_of(yaku_str):
    if not yaku_str:
        return 0
    parts = [int(x) for x in yaku_str.split(",")]
    return sum(parts[1::2])


DISCARD_SEAT = {"D": 0, "E": 1, "F": 2, "G": 3}


def analyse(path):
    root = ET.parse(path).getroot()
    elems = list(root.iter())
    un = next((e for e in elems if e.tag == "UN"), None)
    dans = un.get("dan", "") if un is not None else ""
    rounds = agari = ryuukyoku = 0
    wins = Counter()
    tsumo = Counter()
    dealin = Counter()
    han_sum = mangan = 0
    riichi = Counter()
    calls = Counter()
    discards = Counter()
    tenpai_ryu = 0
    final_scores = None
    for e in elems:
        tag = e.tag
        if tag == "INIT":
            rounds += 1
        elif tag == "AGARI":
            agari += 1
            who = int(e.get("who"))
            frm = int(e.get("fromWho"))
            wins[who] += 1
            if who == frm:
                tsumo[who] += 1
            else:
                dealin[frm] += 1
            h = han_of(e.get("yaku", ""))
            han_sum += h
            if h >= 5:
                mangan += 1
            sc = e.get("sc", "")
            if sc:
                vals = [int(x) for x in sc.split(",")]
                if len(vals) >= 8:
                    final_scores = [(vals[i] + vals[i + 1]) * 100
                                    for i in range(0, 8, 2)]
        elif tag == "RYUUKYOKU":
            ryuukyoku += 1
            tenpai_ryu += sum(1 for i in range(4)
                              if e.get(f"hai{i}") is not None)
            sc = e.get("sc", "")
            if sc:
                vals = [int(x) for x in sc.split(",")]
                if len(vals) >= 8:
                    final_scores = [(vals[i] + vals[i + 1]) * 100
                                    for i in range(0, 8, 2)]
        elif tag == "REACH" and e.get("step") == "1":
            riichi[int(e.get("who"))] += 1
        elif tag == "N":
            calls[int(e.get("who"))] += 1
        elif len(tag) > 1 and tag[0] in DISCARD_SEAT and tag[1:].isdigit():
            discards[DISCARD_SEAT[tag[0]]] += 1
    return {
        "game_id": path.stem,
        "dans": dans,
        "rounds": rounds,
        "agari": agari,
        "ryuukyoku": ryuukyoku,
        "wins": sum(wins.values()),
        "tsumo_wins": sum(tsumo.values()),
        "dealins": sum(dealin.values()),
        "han_sum": han_sum,
        "mangan_wins": mangan,
        "riichi": sum(riichi.values()),
        "calls": sum(calls.values()),
        "discards": sum(discards.values()),
        "tenpai_in_ryuukyoku": tenpai_ryu,
        "final_scores": final_scores,
    }


def main():
    ap = argparse.ArgumentParser(description="Summarise Tenhou mjlog results.")
    ap.add_argument("folder", help="Folder containing .mjlog files")
    ap.add_argument("--out", default="game_stats.csv",
                    help="Output CSV path (default: game_stats.csv)")
    args = ap.parse_args()

    files = sorted(Path(args.folder).glob("*.mjlog"))
    if not files:
        print(f"No .mjlog files found in {args.folder}")
        sys.exit(1)

    rows = []
    for f in files:
        try:
            rows.append(analyse(f))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"SKIP {f.name}: {exc}")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "dans", "rounds", "agari", "ryuukyoku",
                    "wins", "tsumo_wins", "dealins", "han_sum",
                    "mangan_wins", "riichi", "calls", "discards",
                    "tenpai_in_ryuukyoku",
                    "final_p0", "final_p1", "final_p2", "final_p3"])
        for r in rows:
            fs = r["final_scores"] or [None, None, None, None]
            w.writerow([r["game_id"], r["dans"], r["rounds"], r["agari"],
                        r["ryuukyoku"], r["wins"], r["tsumo_wins"],
                        r["dealins"], r["han_sum"], r["mangan_wins"],
                        r["riichi"], r["calls"], r["discards"],
                        r["tenpai_in_ryuukyoku"], *fs])

    n = len(rows)
    tot_rounds = sum(r["rounds"] for r in rows)
    tot_agari = sum(r["agari"] for r in rows)
    print(f"Games: {n}")
    print(f"Rounds: {tot_rounds} (avg {tot_rounds / n:.1f} per game)")
    print(f"Agari: {tot_agari}, Ryuukyoku: {sum(r['ryuukyoku'] for r in rows)}")
    if tot_agari:
        print(f"Avg winning han: {sum(r['han_sum'] for r in rows) / tot_agari:.2f}")
        print(f"Mangan+ wins: {sum(r['mangan_wins'] for r in rows)} "
              f"({sum(r['mangan_wins'] for r in rows) / tot_agari * 100:.1f}% of wins)")
        print(f"Tsumo share: {sum(r['tsumo_wins'] for r in rows) / tot_agari * 100:.1f}%")
    if tot_rounds:
        print(f"Riichi declarations per round: "
              f"{sum(r['riichi'] for r in rows) / tot_rounds:.2f}")
        print(f"Calls per round: {sum(r['calls'] for r in rows) / tot_rounds:.2f}")
        print(f"Deal-ins per round: {sum(r['dealins'] for r in rows) / tot_rounds:.2f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
