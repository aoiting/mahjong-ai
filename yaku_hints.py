"""規則式役種方向偵測（純本地、零 API）。

唔會改動精算引擎（mahjong_engine）任何向聽／和牌計算，
只根據手牌統計提供：
  1. direction_hints：潛在役種方向（三元牌、混一色、七對子、斷么九）
  2. confirmed_yaku：已經成刻子嘅役牌（白發中、場風、自風）
"""

from __future__ import annotations

from tile_images import expand_hand

HONOR_NAMES = {"1z": "東", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "發", "7z": "中"}
DRAGONS = {"5z": "白", "6z": "發", "7z": "中"}
WIND_Z = {"東": "1z", "南": "2z", "西": "3z", "北": "4z"}
SUIT_NAMES = {"m": "萬子", "p": "筒子", "s": "索子"}
SUIT_SHORT = {"m": "萬", "p": "筒", "s": "索"}
NUMBER_NAMES = {"1": "一", "9": "九"}


def _terminal_honor_name(tile: str) -> str:
    """邊張／字牌嘅中文名，例如 9s → 九索、1z → 東、5z → 白。"""
    if tile.endswith("z"):
        return HONOR_NAMES.get(tile, tile)
    return f"{NUMBER_NAMES.get(tile[0], tile[0])}{SUIT_SHORT.get(tile[-1], tile[-1])}"


def tile_counts(hand: str, meld_tiles: list[str] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tile in expand_hand(hand) + list(meld_tiles or []):
        base = tile[:-1] if tile.endswith("r") else tile
        counts[base] = counts.get(base, 0) + 1
    return counts


def _dragon_status(count: int) -> str:
    if count >= 3:
        return "刻"
    if count == 2:
        return "對"
    if count == 1:
        return "單"
    return "冇"


def direction_hints(
    hand: str,
    bakaze: str = "東",
    jikaze: str = "東",
    meld_count: int = 0,
    meld_tiles: list[str] | None = None,
) -> str:
    """回傳潛在役種方向（純文字）；冇方向就回「無明顯役種方向」。"""
    if not hand and not meld_tiles:
        return "無明顯役種方向"

    counts = tile_counts(hand, meld_tiles)
    hints: list[str] = []

    # ── 三元牌進度 ──
    dragons = [(name, counts.get(z, 0)) for z, name in DRAGONS.items()]
    present = [(name, c) for name, c in dragons if c > 0]
    if len(present) == 3:
        desc = "、".join(f"{name}＝{_dragon_status(c)}" for name, c in dragons)
        weak = [name for name, c in dragons if c < 2]
        route = f"三元牌：{desc} → 小三元路線成立"
        if weak:
            route += f"（關鍵牌：{'、'.join(weak)}，唔好打走）"
        target = weak[0] if weak else present[0][0]
        route += f"；若{target}亦成刻可望大三元"
        hints.append(route)

    # ── 混一色傾向 ──
    suit_counts = {
        suit: sum(c for t, c in counts.items() if t.endswith(suit))
        for suit in "mps"
    }
    used_suits = [s for s, c in suit_counts.items() if c]
    honors = sum(c for t, c in counts.items() if t.endswith("z"))
    if len(used_suits) == 1:
        suit = used_suits[0]
        total = suit_counts[suit] + honors
        if total >= 10:
            hints.append(f"混一色傾向（{SUIT_NAMES[suit]}＋字牌，合共 {total} 隻）")

    # ── 七對子進度（副露咗做唔到七對子，所以有副露唔出）──
    pairs = sum(1 for c in counts.values() if c >= 2)
    if int(meld_count) == 0 and not meld_tiles and pairs >= 4:
        hints.append(f"對子 {pairs} 個，七對子路線可考慮")

    # ── 斷么九：只計邊張（一／九）同字牌（風牌＋三元牌）──
    terminal_honors = [
        (tile, count)
        for tile, count in counts.items()
        if tile.endswith("z") or tile[0] in "19"
    ]
    terminal_honor_total = sum(count for _, count in terminal_honors)
    if terminal_honor_total == 0:
        hints.append("斷么九：全手都係二至八，冇邊張字牌，點和都有斷么九")
    elif terminal_honor_total == 1:
        tile = next(tile for tile, count in terminal_honors if count)
        name = _terminal_honor_name(tile)
        hints.append(
            f"斷么九：淨返一隻 {name} 係邊張／字牌，"
            "切咗佢就全手二至八，入斷么九路線"
        )

    return "；".join(hints) if hints else "無明顯役種方向"


def confirmed_yaku(
    hand: str,
    bakaze: str = "東",
    jikaze: str = "東",
    meld_count: int = 0,
    meld_tiles: list[str] | None = None,
) -> list[str]:
    """已確定役牌（刻子）：白發中、場風、自風。

    注意：app 目前只追蹤副露「組數」，冇副露牌種資料，
    所以呢度只計手牌入面嘅刻子。
    """
    counts = tile_counts(hand, meld_tiles)
    out: list[str] = []
    for z, name in DRAGONS.items():
        if counts.get(z, 0) >= 3:
            out.append(f"{name}（刻子）")
    for wind in dict.fromkeys([bakaze, jikaze]):
        if wind in WIND_Z and counts.get(WIND_Z[wind], 0) >= 3:
            out.append(f"{wind}（刻子）")
    return out


def inspect(
    hand: str,
    bakaze: str = "東",
    jikaze: str = "東",
    meld_count: int = 0,
    meld_tiles: list[str] | None = None,
) -> tuple[str, list[str]]:
    """一次過回傳 (direction_hints, confirmed_yaku)。"""
    return (
        direction_hints(hand, bakaze, jikaze, meld_count, meld_tiles),
        confirmed_yaku(hand, bakaze, jikaze, meld_count, meld_tiles),
    )
