#!/usr/bin/env python3
"""Batch-convert Tenhou .mjlog files to mjai JSONL with mjlog2mjai.

The mjai format is one JSON object per line, one per game event
(start_kyoku, tsumo, dahai, chi, pon, kan, reach, hora, ...), with
tiles as human-readable strings like "1m" or "5mr". Converting first
means the later sample-extraction step never has to decode Tenhou's
numeric tile IDs or meld bitfields itself.

Setup (once):
    git clone https://github.com/fstqwq/mjlog2mjai

Usage:
    python convert_batch.py mjlog mjai --tool mjlog2mjai

Each <id>.mjlog becomes <id>.jsonl. Files that fail to convert
(e.g. sanma games, which the converter does not support) are skipped
and listed at the end. If you get "No module named parse", point
--tool at the folder that actually contains parse.py.
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert .mjlog files to mjai JSONL.")
    ap.add_argument("indir", help="Folder containing .mjlog files")
    ap.add_argument("outdir", help="Folder to write .jsonl files into")
    ap.add_argument("--tool", default="mjlog2mjai",
                    help="Folder containing mjlog2mjai's parse.py")
    args = ap.parse_args()

    sys.path.insert(0, args.tool)
    try:
        from parse import load_mjlog, parse_mjlog_to_mjai
    except ImportError as exc:
        print(f"Could not import mjlog2mjai from {args.tool}: {exc}")
        print("Find parse.py (e.g. under the cloned repo) and pass its "
              "folder with --tool.")
        sys.exit(1)

    in_dir, out_dir = Path(args.indir), Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.mjlog"))
    if not files:
        print(f"No .mjlog files found in {in_dir}")
        sys.exit(1)

    failed = []
    for i, f in enumerate(files, 1):
        dest = out_dir / f"{f.stem}.jsonl"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            root = load_mjlog(str(f))
            dest.write_text(parse_mjlog_to_mjai(root), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append((f.name, str(exc)))
        if i % 50 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] converted, {len(failed)} failed")
    if failed:
        print("FAILED files:")
        for name, why in failed:
            print(f"  {name} - {why}")
    print("Done.")


if __name__ == "__main__":
    main()
