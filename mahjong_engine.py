"""確定性麻雀計算（唔靠 AI 猜）：向聽、有效牌、切牌比較、和牌點數。

用 `mahjong` 套件（pip install mahjong）做精算，AI 只負責解釋結果。
"""

from __future__ import annotations

import re
from typing import Any

from mahjong.constants import EAST, NORTH, SOUTH, WEST
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

WIND_INDEX = {"東": EAST, "南": SOUTH, "西": WEST, "北": NORTH}
HONORS = ["東", "南", "西", "北", "白", "發", "中"]

_SHANTEN = Shanten()
_SUIT_KEY = {"m": "man", "p": "pin", "s": "sou", "z": "honors"}


def tile_label(index34: int) -> str:
    """34 索引 → 記法標籤（例如 0 → 1m、27 → 1z）。"""
    if index34 < 9:
        return f"{index34 + 1}m"
    if index34 < 18:
        return f"{index34 - 8}p"
    if index34 < 27:
        return f"{index34 - 17}s"
    return f"{index34 - 26}z"


def _split(hand: str) -> dict[str, str]:
    out = {"man": "", "pin": "", "sou": "", "honors": ""}
    for digits, suit in re.findall(r"(\d+)([mpsz])", hand or ""):
        out[_SUIT_KEY[suit]] += digits
    return out


def to_136(hand: str) -> list[int]:
    p = _split(hand)
    return TilesConverter.string_to_136_array(
        man=p["man"], pin=p["pin"], sou=p["sou"], honors=p["honors"], has_aka_dora=False
    )


def to_34(hand: str) -> list[int]:
    return TilesConverter.to_34_array(to_136(hand))


def _dora_indicators(dora: str) -> list[int]:
    """把使用者輸入嘅寶牌指示牌轉成 136 索引。"""
    dora = (dora or "").strip()
    if not dora:
        return []
    # 支援「東」等字牌名稱
    for i, name in enumerate(HONORS, start=1):
        dora = dora.replace(name, f"{i}z")
    try:
        tiles = to_136(dora)
    except Exception:  # noqa: BLE001
        return []
    return tiles[:1]


def _ukeire(tiles34: list[int], shanten: int) -> list[tuple[int, int]]:
    """回傳 [(牌索引, 剩餘張數)]，只計可以令向聽下降嘅牌。"""
    result = []
    for i in range(34):
        if tiles34[i] >= 4:
            continue
        probe = list(tiles34)
        probe[i] += 1
        if _SHANTEN.calculate_shanten(probe) < shanten:
            result.append((i, 4 - tiles34[i]))
    return result


def _hand_value(
    hand14: str,
    win_tile_label: str,
    ctx: dict[str, Any],
    is_tsumo: bool,
) -> dict[str, Any] | None:
    """計算和牌點數（假設門清）。"""
    try:
        tiles = to_136(hand14)
        win_tiles = to_136(win_tile_label)
        # 揀一張存在於手牌中嘅同一種牌
        win_tile = next(t for t in tiles if t in win_tiles)
        config = HandConfig(
            is_tsumo=is_tsumo,
            player_wind=WIND_INDEX.get(ctx.get("jikaze", "東")),
            round_wind=WIND_INDEX.get(ctx.get("bakaze", "東")),
            options=OptionalRules(has_open_tanyao=True, has_aka_dora=False),
        )
        res = HandCalculator.estimate_hand_value(
            tiles,
            win_tile,
            dora_indicators=_dora_indicators(ctx.get("dora", "")),
            config=config,
        )
        if res.error:
            return None
        return {
            "yaku": [y.name for y in (res.yaku or [])],
            "han": res.han,
            "fu": res.fu,
            "cost": res.cost,
            "tsumo": is_tsumo,
        }
    except Exception:  # noqa: BLE001
        return None


def format_facts(result: dict[str, Any]) -> str:
    """把精算結果轉成餵給 AI 嘅事實清單（Markdown）。"""
    if result.get("error"):
        return f"【精算結果】無法計算：{result['error']}"

    lines = ["【精算結果（由計算引擎算出，必須遵從）】"]
    lines.append(f"- 手牌張數：{result['tiles']} 張")
    lines.append(f"- 向聽數：{result['shanten']}")

    if result.get("waits"):
        waits = "、".join(f"{t}（剩 {c} 張）" for t, c in result["waits"])
        lines.append(f"- 聽牌待牌：{waits}")

    if result.get("ukeire") and not result.get("waits"):
        uke = "、".join(f"{t}（剩 {c} 張）" for t, c in result["ukeire"])
        lines.append(f"- 有效牌：{uke}")

    if result.get("discards"):
        lines.append("- 切牌比較（由好至差）：")
        for d in result["discards"][:5]:
            uke = "、".join(f"{t}({c})" for t, c in d["ukeire"]) or "無"
            lines.append(
                f"  - 切 {d['tile']} → {d['shanten']} 向聽，有效牌 {d['ukeire_count']} 張（{uke}）"
            )
        best = result.get("best")
        if best:
            lines.append(f"- 最佳切牌：{best['tile']}")

    if result.get("agari_preview"):
        lines.append("- 聽牌後和牌預覽（榮和）：")
        for v in result["agari_preview"]:
            yaku = "、".join(v["yaku"]) or "無役"
            cost = (v.get("cost") or {}).get("main", 0)
            lines.append(
                f"  - 待 {v['wait']}：{yaku}，{v['han']} 翻 {v['fu']} 符，{cost} 點"
            )

    if result.get("agari"):
        lines.append("- 已經和牌：")
        for kind, v in result["agari"].items():
            yaku = "、".join(v["yaku"]) or "無役"
            cost = (v.get("cost") or {})
            lines.append(
                f"  - {kind}：{yaku}，{v['han']} 翻 {v['fu']} 符，"
                f"{cost.get('main', 0)} 點（總 {cost.get('total', 0)}）"
            )

    return "\n".join(lines)


def analyze(hand: str, ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    """精算手牌。

    Args:
        hand: 壓縮記法手牌（13 或 14 張）。
        ctx: 含 bakaze / jikaze / dora。

    Returns:
        dict，包含 shanten、ukeire、discards、agari、summary；失敗時含 error。
    """
    ctx = ctx or {}
    hand = (hand or "").strip()
    if not hand:
        return {"error": "沒有手牌。"}

    try:
        tiles34 = to_34(hand)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"手牌格式無法解析：{exc}"}

    n = sum(tiles34)
    if n not in (13, 14):
        return {"error": f"手牌有 {n} 張，精算只支援 13 或 14 張。"}

    shanten = _SHANTEN.calculate_shanten(tiles34)
    result: dict[str, Any] = {"tiles": n, "shanten": shanten}

    # 已經和牌
    if shanten == -1:
        labels = [tile_label(i) for i in range(34) for _ in range(tiles34[i])]
        win_label = labels[-1] if labels else ""
        agari = {}
        for tsumo in (False, True):
            v = _hand_value(hand, win_label, ctx, tsumo)
            if v:
                agari["自摸" if tsumo else "榮和"] = v
        result["agari"] = agari
        result["summary"] = _summary_agari(agari)
        return result

    # 13 張：直接計有效牌
    if n == 13:
        uke = _ukeire(tiles34, shanten)
        result["ukeire"] = [(tile_label(i), c) for i, c in uke]
        result["waits"] = [(tile_label(i), c) for i, c in uke] if shanten == 0 else []
        result["summary"] = _summary_shanten(shanten, result["ukeire"])
        if shanten == 0:
            result["agari_preview"] = _tenpai_preview(hand, result["ukeire"], ctx)
        return result

    # 14 張：逐張切牌比較
    discards = []
    for i in range(34):
        if tiles34[i] == 0:
            continue
        probe = list(tiles34)
        probe[i] -= 1
        s = _SHANTEN.calculate_shanten(probe)
        uke = _ukeire(probe, s)
        discards.append(
            {
                "tile": tile_label(i),
                "shanten": s,
                "ukeire_count": sum(c for _, c in uke),
                "ukeire": [(tile_label(t), c) for t, c in uke],
            }
        )
    discards.sort(key=lambda d: (d["shanten"], -d["ukeire_count"]))
    result["discards"] = discards

    best = discards[0] if discards else None
    result["best"] = best
    result["summary"] = _summary_discards(shanten, best)

    # 若切牌後聽牌，預覽役種
    if best and best["shanten"] == 0 and best["ukeire"]:
        tiles13 = list(tiles34)
        idx = next(j for j in range(34) if tile_label(j) == best["tile"])
        tiles13[idx] -= 1
        hand13 = _to_compressed(tiles13)
        result["agari_preview"] = _tenpai_preview(hand13, best["ukeire"], ctx)

    return result


def _to_compressed(tiles34: list[int]) -> str:
    parts = []
    for suit in "mpsz":
        digits = []
        for i in range(34):
            if tiles34[i] and tile_label(i).endswith(suit):
                digits.extend([str(i % 9 + 1)] * tiles34[i])
        if digits:
            parts.append("".join(sorted(digits)) + suit)
    return "".join(parts)


def _tenpai_preview(
    hand13: str, waits: list[tuple[str, int]], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """聽牌時，逐張待牌計和牌點數（榮和）。"""
    out = []
    for label, _count in waits:
        try:
            hand14 = _to_compressed(sorted_hand_34(to_34(hand13), label))
        except Exception:  # noqa: BLE001
            continue
        value = _hand_value(hand14, label, ctx, is_tsumo=False)
        if value:
            out.append({"wait": label, **value})
    return out


def sorted_hand_34(tiles34: list[int], add_label: str) -> dict:
    """把 13 張 + 1 張待牌合併成 dict 形式，方便組合字串。"""
    idx = next(i for i in range(34) if tile_label(i) == add_label)
    merged = list(tiles34)
    merged[idx] += 1
    return merged


def _summary_shanten(shanten: int, ukeire: list[tuple[str, int]]) -> str:
    total = sum(c for _, c in ukeire)
    tiles = "、".join(f"{t}({c})" for t, c in ukeire) or "無"
    if shanten == 0:
        return f"**聽牌**（0 向聽）— 待牌：{tiles}，合共 {total} 張"
    return f"**{shanten} 向聽** — 有效牌：{tiles}，合共 {total} 張"


def _summary_discards(shanten: int, best: dict | None) -> str:
    if not best:
        return f"{shanten} 向聽（無法計算切牌）"
    return (
        f"**{best['shanten']} 向聽** — 建議切 **{best['tile']}**，"
        f"有效牌 {best['ukeire_count']} 張"
    )


def _summary_agari(agari: dict[str, Any]) -> str:
    if not agari:
        return "已經和牌（無役或計算失敗）"
    parts = []
    for kind, v in agari.items():
        yaku = "、".join(v["yaku"]) or "無役"
        parts.append(f"{kind}：{yaku}　{v['han']} 翻 {v['fu']} 符")
    return "　|　".join(parts)
