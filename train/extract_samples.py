#!/usr/bin/env python3
"""Extract per-discard decision samples from mjai JSONL files.

One output row = one discard decision:
    the player's closed hand at that moment (including the drawn tile),
    their melds, all four discard rivers, dora markers, scores, riichi
    flags -> and the tile they actually discarded (the label).

State is tracked event by event from each start_kyoku's tehais.
Every tile removal is checked (the tile must actually be in the hand);
a game that fails any check is skipped and reported, so a format
surprise can never silently corrupt the dataset.

Usage:
    python extract_samples.py mjai discard_samples.csv
    python extract_samples.py mjai test_samples.csv --limit 50
"""
import argparse
import csv
import json
from pathlib import Path

SUIT_ORDER = {"m": 0, "p": 1, "s": 2, "z": 3}
HONORS = {"E": 0, "S": 1, "W": 2, "N": 3, "P": 4, "F": 5, "C": 6}
WINDS = ["E", "S", "W", "N"]

COLUMNS = [
    "game_id", "round_index", "kyoku", "honba", "kyotaku",
    "turn", "actor", "jikaze", "oya",
    "scores", "score_self", "dora_markers",
    "hand", "melds", "drawn", "discarded", "tsumogiri",
    "self_riichi", "riichi_dahai", "riichi_flags",
    "discards_0", "discards_1", "discards_2", "discards_3",
]


def tile_key(tile: str):
    """Sort key: man < pin < sou < honors; red five next to plain five."""
    if tile in HONORS:
        return (3, HONORS[tile], 0)
    red = tile.endswith("r")
    base = tile[:-1] if red else tile
    return (SUIT_ORDER[base[-1]], int(base[0]), 1 if red else 0)


def sort_tiles(tiles):
    return sorted(tiles, key=tile_key)


def remove_one(hand, tile, what):
    if tile not in hand:
        raise ValueError(f"{what}: tile {tile} not in hand {hand}")
    hand.remove(tile)


class KyokuState:
    def __init__(self, ev):
        self.bakaze = ev["bakaze"]
        self.kyoku = ev["kyoku"]
        self.honba = ev["honba"]
        self.kyotaku = ev["kyotaku"]
        self.oya = ev["oya"]
        self.scores = list(ev["scores"])
        self.hands = [list(h) for h in ev["tehais"]]
        self.melds = [[], [], [], []]
        self.rivers = [[], [], [], []]
        self.dora_markers = [ev["dora_marker"]]
        self.riichi = [False, False, False, False]
        self.pending_reach = None
        self.turn_count = [0, 0, 0, 0]
        self.last_draw = [None, None, None, None]

    def take_called_tile(self, target, pai):
        river = self.rivers[target]
        if not river or river[-1] != pai:
            raise ValueError(f"called tile {pai} is not the last discard "
                             f"of player {target}: {river}")
        river.pop()


def extract_game(path: Path):
    """Return a list of sample rows for one .jsonl game file."""
    rows = []
    st = None
    round_index = 0
    game_id = path.stem
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            t = ev.get("type")

            if t == "start_kyoku":
                st = KyokuState(ev)
                round_index += 1

            elif t == "tsumo":
                a = ev["actor"]
                st.hands[a].append(ev["pai"])
                st.last_draw[a] = ev["pai"]

            elif t == "dahai":
                a = ev["actor"]
                pai = ev["pai"]
                if pai not in st.hands[a]:
                    raise ValueError(f"dahai: {pai} not in hand of {a}")
                st.turn_count[a] += 1
                rows.append({
                    "game_id": game_id,
                    "round_index": round_index,
                    "kyoku": f"{st.bakaze}{st.kyoku}",
                    "honba": st.honba,
                    "kyotaku": st.kyotaku,
                    "turn": st.turn_count[a],
                    "actor": a,
                    "jikaze": WINDS[(a - st.oya) % 4],
                    "oya": st.oya,
                    "scores": " ".join(str(s) for s in st.scores),
                    "score_self": st.scores[a],
                    "dora_markers": " ".join(st.dora_markers),
                    "hand": " ".join(sort_tiles(st.hands[a])),
                    "melds": "; ".join(st.melds[a]),
                    "drawn": st.last_draw[a] or "",
                    "discarded": pai,
                    "tsumogiri": str(bool(ev.get("tsumogiri"))).lower(),
                    "self_riichi": str(st.riichi[a]).lower(),
                    "riichi_dahai": str(st.pending_reach == a).lower(),
                    "riichi_flags": "".join(
                        "1" if r else "0" for r in st.riichi),
                    "discards_0": " ".join(st.rivers[0]),
                    "discards_1": " ".join(st.rivers[1]),
                    "discards_2": " ".join(st.rivers[2]),
                    "discards_3": " ".join(st.rivers[3]),
                })
                st.hands[a].remove(pai)
                st.rivers[a].append(pai)
                st.last_draw[a] = None
                if st.pending_reach == a:
                    st.pending_reach = None

            elif t in ("chi", "pon"):
                a, target, pai = ev["actor"], ev["target"], ev["pai"]
                for c in ev["consumed"]:
                    remove_one(st.hands[a], c, t)
                st.take_called_tile(target, pai)
                meld = sort_tiles(list(ev["consumed"]) + [pai])
                st.melds[a].append(f"{t}:{' '.join(meld)}")
                st.last_draw[a] = None

            elif t == "daiminkan":
                a, target, pai = ev["actor"], ev["target"], ev["pai"]
                for c in ev["consumed"]:
                    remove_one(st.hands[a], c, t)
                st.take_called_tile(target, pai)
                meld = sort_tiles(list(ev["consumed"]) + [pai])
                st.melds[a].append(f"daiminkan:{' '.join(meld)}")
                st.last_draw[a] = None

            elif t == "ankan":
                a = ev["actor"]
                for c in ev["consumed"]:
                    remove_one(st.hands[a], c, t)
                st.melds[a].append(
                    f"ankan:{' '.join(sort_tiles(ev['consumed']))}")

            elif t == "kakan":
                a = ev["actor"]
                remove_one(st.hands[a], ev["pai"], t)
                base = ev["pai"].rstrip("r")
                for i, m in enumerate(st.melds[a]):
                    if m.startswith("pon:") and base in m:
                        st.melds[a][i] = "kakan:" + m[4:]
                        break
                else:
                    st.melds[a].append(f"kakan:{ev['pai']}")

            elif t == "dora":
                st.dora_markers.append(ev["dora_marker"])

            elif t == "reach":
                st.pending_reach = ev["actor"]

            elif t == "reach_accepted":
                a = ev["actor"]
                st.riichi[a] = True
                st.scores[a] -= 1000
                st.kyotaku += 1

            # hora / ryukyoku / end_kyoku / end_game: nothing to track;
            # the next start_kyoku resets all state from tehais anyway.
    if not rows:
        raise ValueError("no discard samples extracted")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract per-discard samples from mjai JSONL files.")
    ap.add_argument("indir", help="Folder containing .jsonl files")
    ap.add_argument("outfile", help="Output CSV path")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N files (0 = all)")
    args = ap.parse_args()

    files = sorted(Path(args.indir).glob("*.jsonl"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No .jsonl files found in {args.indir}")
        return

    total = 0
    skipped = []
    with open(args.outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i, path in enumerate(files, 1):
            try:
                rows = extract_game(path)
                writer.writerows(rows)
                total += len(rows)
            except Exception as exc:  # noqa: BLE001 - skip and report
                skipped.append((path.name, str(exc)))
            if i % 200 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] games, {total} samples, "
                      f"{len(skipped)} skipped")

    print(f"Done. {total} samples from {len(files) - len(skipped)} games "
          f"written to {args.outfile}")
    if skipped:
        print(f"SKIPPED {len(skipped)} games:")
        for name, why in skipped[:20]:
            print(f"  {name} - {why}")


if __name__ == "__main__":
    main()
