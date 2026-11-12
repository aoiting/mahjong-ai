#!/usr/bin/env python3
"""Download Tenhou mjlog files for a list of log IDs, politely.

Tenhou will block aggressive bulk downloading, so this script is
deliberately slow and stops at the first sign of trouble:

- A delay plus random jitter between requests (defaults: 3s + 0-2s).
- A longer pause after every batch (defaults: 60s after every 100 logs).
- Stops immediately on HTTP 429 or 403 instead of retrying.
- Skips files that already exist, so any run can be resumed.
- `--limit N` lets you do a small trial run first (e.g. 50 logs).

Usage:
    python download_logs.py log_ids.txt
    python download_logs.py log_ids.txt --limit 50
    python download_logs.py log_ids.txt --out mjlog --delay 3 --jitter 2

Only the standard library is used: no extra packages needed.
"""
import argparse
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


def download_one(log_id: str, outdir: Path, retries: int = 3) -> str:
    """Return 'ok', 'skip', 'fail', or 'blocked' (429/403: stop everything)."""
    dest = outdir / f"{log_id}.mjlog"
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    url = f"https://tenhou.net/0/log/?{log_id}"
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if not data.startswith(b"<"):
                raise ValueError("response does not look like XML")
            dest.write_bytes(data)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code in (429, 403):
                print(f"BLOCKED (HTTP {e.code}) on {log_id}: stopping now.")
                return "blocked"
            last_err = e
            time.sleep(5)
        except Exception as e:  # noqa: BLE001 - reported and retried
            last_err = e
            time.sleep(5)
    print(f"FAIL {log_id}: {last_err}")
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Tenhou mjlog files, politely.")
    parser.add_argument("id_file", help="Text file with one log ID per line")
    parser.add_argument("--out", default="mjlog", help="Output folder (default: mjlog)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Base seconds between requests (default: 3.0)")
    parser.add_argument("--jitter", type=float, default=2.0,
                        help="Extra random delay up to this many seconds (default: 2.0)")
    parser.add_argument("--batch", type=int, default=100,
                        help="Pause after every N downloads (default: 100, 0 = never)")
    parser.add_argument("--batch-pause", type=float, default=60.0,
                        help="Seconds to pause after each batch (default: 60)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N IDs (default: 0 = all)")
    args = parser.parse_args()

    ids = []
    for line in Path(args.id_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    if args.limit:
        ids = ids[: args.limit]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed = []
    done = 0
    for i, log_id in enumerate(ids, 1):
        result = download_one(log_id, outdir)
        if result == "blocked":
            break
        counts[result] += 1
        done += 1
        if result == "fail":
            failed.append(log_id)
        if i % 50 == 0 or i == len(ids):
            print(f"[{i}/{len(ids)}] ok={counts['ok']} "
                  f"skip={counts['skip']} fail={counts['fail']}")
        # Polite pacing: base delay + random jitter.
        time.sleep(args.delay + random.uniform(0, args.jitter))
        # Longer rest after each batch of successful/attempted downloads.
        if args.batch and done and done % args.batch == 0 and i < len(ids):
            print(f"Batch of {args.batch} done, resting {args.batch_pause}s...")
            time.sleep(args.batch_pause)

    if failed:
        Path("failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} failed IDs written to failed.txt")
    print(f"Done. ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")


if __name__ == "__main__":
    main()
