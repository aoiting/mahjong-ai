"""下載公有領域嘅麻雀牌圖像（給手牌顯示用）。

來源：https://github.com/FluffyStuff/riichi-mahjong-tiles
授權：CC0 1.0（公有領域），可以自由使用、修改、再分發。

用法：
    py tools/fetch_tiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "tiles"
BASE = "https://raw.githubusercontent.com/FluffyStuff/riichi-mahjong-tiles/master/Export/Regular"

MAN = [f"Man{i}" for i in range(1, 10)]
PIN = [f"Pin{i}" for i in range(1, 10)]
SOU = [f"Sou{i}" for i in range(1, 10)]
HONORS = ["Ton", "Nan", "Shaa", "Pei", "Haku", "Hatsu", "Chun"]

NAMES = MAN + PIN + SOU + HONORS


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in NAMES:
        dest = OUT / f"{name}.png"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{BASE}/{name}.png"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"✓ {name}.png ({len(r.content) // 1024} KB)")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, str(exc)))
            print(f"✗ {name}: {exc}")

    print(f"\n完成：{len(NAMES) - len(failed)}/{len(NAMES)} 張 → {OUT}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
