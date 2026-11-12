#!/usr/bin/env python3
"""Check downloaded .mjlog files: valid XML with GO/UN/INIT elements.

Usage:
    python check_downloads.py mjlog

Prints how many files are OK, the total number of rounds, and lists
any bad files (not XML, unparsable, or missing game elements).
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mjlog")
    files = sorted(folder.glob("*.mjlog"))
    print(f"{len(files)} files in {folder}")
    bad = []
    total_init = 0
    for f in files:
        try:
            data = f.read_bytes()
            if not data.startswith(b"<"):
                bad.append((f.name, "not XML"))
                continue
            root = ET.fromstring(data)
            tags = [el.tag for el in root.iter()]
            n_init = tags.count("INIT")
            if "GO" not in tags or "UN" not in tags or n_init == 0:
                bad.append((f.name, f"missing GO/UN/INIT (INIT={n_init})"))
                continue
            total_init += n_init
        except Exception as exc:  # noqa: BLE001 - reported per file
            bad.append((f.name, f"parse error: {exc}"))
    print(f"OK files: {len(files) - len(bad)}, total rounds: {total_init}")
    if bad:
        print("BAD files:")
        for name, why in bad:
            print(f"  {name} - {why}")
    else:
        print("All files look good.")


if __name__ == "__main__":
    main()
