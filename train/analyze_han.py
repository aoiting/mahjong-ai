#!/usr/bin/env python3
"""Winning-hand han distribution (chart data).

Usage: python analyze_han.py mjlog

Scans the same .mjlog folder scan_stats.py used, and for every AGARI
tag computes the winning hand's total han the same way scan_stats.py
does (the yaku attribute alternates (yaku_id, han); total han = sum of
every second value, dora included). Buckets follow the score limits:
5 = mangan, 6-7 haneman, 8-10 baiman, 11-12 sanbaiman, 13+ yakuman /
kazoe. Prints built-in cross-checks against the known batch totals
(5,861 wins, avg 2.91 han, 871 mangan+ = 14.9%): if those three do not
reproduce, the distribution is not trustworthy.

Saves han_distribution.csv next to the script's working directory.
"""
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


BUCKETS = [
    ("1", "1 han", lambda h: h == 1),
    ("2", "2 han", lambda h: h == 2),
    ("3", "3 han", lambda h: h == 3),
    ("4", "4 han", lambda h: h == 4),
    ("5", "5 han - mangan", lambda h: h == 5),
    ("6-7", "6-7 han - haneman", lambda h: 6 <= h <= 7),
    ("8-10", "8-10 han - baiman", lambda h: 8 <= h <= 10),
    ("11-12", "11-12 han - sanbaiman", lambda h: 11 <= h <= 12),
    ("13+", "13+ han - yakuman / kazoe", lambda h: h >= 13),
]


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "mjlog")
    files = sorted(folder.glob("*.mjlog"))
    if not files:
        print(f"No .mjlog files found in {folder}")
        sys.exit(1)

    counts = Counter()
    total = 0
    han_sum = 0
    mangan = 0
    unparsed = 0
    for f in files:
        root = ET.parse(f).getroot()
        for e in root.iter():
            if e.tag != "AGARI":
                continue
            h = han_of(e.get("yaku", ""))
            total += 1
            han_sum += h
            if h >= 5:
                mangan += 1
            if h == 0:
                unparsed += 1
                continue
            for key, _label, test in BUCKETS:
                if test(h):
                    counts[key] += 1
                    break

    print(f"{len(files)} files, {total} wins (AGARI tags)")
    if unparsed:
        print(f"WARNING: {unparsed} wins had no yaku attribute (han 0), "
              "excluded from buckets")

    print("\n--- cross-checks vs known batch totals ---")
    print(f"total wins: {total}  (expected 5861)")
    print(f"avg han:    {han_sum / total:.2f}  (expected 2.91)")
    print(f"mangan+:    {mangan} = {mangan / total * 100:.1f}%  "
          "(expected 871 = 14.9%)")

    print("\n===== Winning-hand han distribution =====")
    rows = []
    for key, label, _test in BUCKETS:
        n = counts[key]
        pct = n / total * 100 if total else 0.0
        rows.append((key, label, n, pct))
        print(f"  {label:<28} {n:>5}   {pct:5.2f}%")
    bucket_total = sum(counts.values())
    print(f"  bucket total: {bucket_total} (wins minus unparsed)")

    out = Path("han_distribution.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "label", "count", "pct_of_wins"])
        for key, label, n, pct in rows:
            w.writerow([key, label, n, f"{pct:.2f}"])
    print(f"\nSaved: {out.resolve()}")


if __name__ == "__main__":
    main()
