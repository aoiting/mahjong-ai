"""把手牌代碼（1m、5p、3z…）渲染成真實麻雀牌圖像。

牌圖來源：https://github.com/FluffyStuff/riichi-mahjong-tiles（CC0 公有領域）。
如果你想用自己遊戲嘅牌圖，只需將 PNG 放進 `assets/tiles/`，
檔名跟以下規則即可自動覆蓋：
    1m..9m → Man1.png .. Man9.png
    1p..9p → Pin1.png .. Pin9.png
    1s..9s → Sou1.png .. Sou9.png
    1z..7z → Ton / Nan / Shaa / Pei / Haku / Hatsu / Chun（1=東 … 7=中）
"""

from __future__ import annotations

import base64
import functools
import io
import re
from pathlib import Path

from PIL import Image

TILE_DIR = Path(__file__).with_name("assets") / "tiles"

_SUIT_NAMES = {"m": "Man", "p": "Pin", "s": "Sou"}
_HONOR_NAMES = {
    1: "Ton",    # 東
    2: "Nan",    # 南
    3: "Shaa",   # 西
    4: "Pei",    # 北
    5: "Haku",   # 白
    6: "Hatsu",  # 發
    7: "Chun",   # 中
}


def tile_name(code: str) -> str | None:
    """'1m' → 'Man1'；唔合法就回 None。"""
    if len(code) != 2 or not code[0].isdigit():
        return None
    n, suit = int(code[0]), code[1]
    if suit in _SUIT_NAMES:
        return f"{_SUIT_NAMES[suit]}{n}"
    if suit == "z":
        return _HONOR_NAMES.get(n)
    return None


@functools.lru_cache(maxsize=64)
def tile_uri(code: str, height: int = 96) -> str:
    """回傳可直接嵌入 HTML 嘅 data URI；冇圖就回空字串。"""
    name = tile_name(code)
    if not name:
        return ""
    path = TILE_DIR / f"{name}.png"
    if not path.exists():
        return ""
    try:
        img = Image.open(path).convert("RGBA")
        width = max(1, round(img.width * height / img.height))
        img = img.resize((width, height), Image.LANCZOS)
        buf = io.BytesIO()
        try:
            img.save(buf, format="WEBP", quality=88, method=6)
            mime = "webp"
        except Exception:  # noqa: BLE001 — 個別 Pillow 版本冇 WEBP
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            mime = "png"
    except Exception:  # noqa: BLE001
        return ""
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def expand_hand(hand: str) -> list[str]:
    """'112357p5m' → ['1p','1p','2p','3p','5p','7p','5m']（保持輸入次序）。"""
    tiles: list[str] = []
    for digits, suit in re.findall(r"(\d+)([mpsz])", hand or ""):
        for d in digits:
            if d == "0":  # 赤牌當普通 5
                d = "5"
            tiles.append(f"{d}{suit}")
    return tiles


def _img(code: str, height: int, extra_style: str = "") -> str:
    uri = tile_uri(code, height * 2)  # 2x 解析度，高 DPI 都清晰
    if not uri:
        return (
            f'<span style="display:inline-block;padding:2px 6px;border:1px solid #ccc;'
            f'border-radius:6px;font:600 {max(12, height * 0.55)}px/1 monospace;'
            f'background:#fff;color:#333;{extra_style}">{code}</span>'
        )
    return (
        f'<img src="{uri}" alt="{code}" title="{code}" '
        f'style="height:{height}px;width:auto;vertical-align:middle;'
        f'border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.25);'
        f'margin:0 1px;background:#fff;{extra_style}">'
    )


def tiles_html(codes: list[str], height: int = 44, gap: int = 10) -> str:
    """一排牌圖；`gap` 係每張牌之間嘅空隙（px）。"""
    if not codes:
        return ""
    return "".join(_img(c, height) for c in codes)


def hand_html(hand: str, height: int = 48) -> str:
    """整手牌，按花色分組，組與組之間留多啲位。"""
    tiles = expand_hand(hand)
    if not tiles:
        return ""
    groups: list[str] = []
    current_suit = tiles[0][-1]
    buf: list[str] = []
    for code in tiles:
        suit = code[-1]
        if suit != current_suit and buf:
            groups.append("".join(buf))
            buf = []
            current_suit = suit
        buf.append(_img(code, height))
    if buf:
        groups.append("".join(buf))
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;'
        'padding:6px 0 10px;">' + "".join(groups) + "</div>"
    )


def inline_html(hand: str, height: int = 30) -> str:
    """內文用嘅細牌圖（例如「待牌」）。"""
    return "".join(_img(c, height) for c in expand_hand(hand))


def render_hand(hand: str, height: int = 48) -> None:
    """喺 Streamlit 顯示整手牌圖像。"""
    import streamlit as st

    html = hand_html(hand, height)
    if html:
        st.markdown(html, unsafe_allow_html=True)


_CODE_RE = re.compile(r"(?<![0-9A-Za-z])(\d+)([mpsz])(?![0-9A-Za-z])")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)


def decorate_text(text: str, height: int = 26) -> str:
    """將 AI 文字入面嘅牌代碼（例如 `3m`、`123p`）換成牌圖。

    同時會將 Markdown 粗體 `**…**` 轉成 `<strong>`，
    因為牌圖係 HTML，Markdown 唔會再解析 `**<img>**`，會殘留星號。
    """
    if not text:
        return text

    # 牌代碼如果包住喺 `…` 入面，Markdown 會當係程式碼而唔會 render HTML，
    # 所以要拆走呢啲 backtick（只限入面真係牌代碼嘅）。
    text = re.sub(
        r"`([^`\n]*)`",
        lambda m: m.group(1) if _CODE_RE.search(m.group(1)) else m.group(0),
        text,
    )

    def repl(match: re.Match[str]) -> str:
        digits, suit = match.group(1), match.group(2)
        codes = [f"{d}{suit}" for d in digits]
        if any(not tile_uri(c, height * 2) for c in codes):
            return match.group(0)
        return "".join(_img(c, height) for c in codes)

    out = _CODE_RE.sub(repl, text)
    return _BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)


def waits_html(waits: list[tuple[str, int]], height: int = 34) -> str:
    """待牌：牌圖 + 剩餘張數。"""
    items = []
    for tile, count in waits:
        items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f'margin:0 10px 6px 0;">{_img(tile, height)}'
            f'<span style="font-size:0.85rem;color:#555;">剩 {count} 張</span></span>'
        )
    return f'<div style="display:flex;flex-wrap:wrap;align-items:center;">{"".join(items)}</div>'


def discard_table_html(rows: list[dict], tile_height: int = 40, ukeire_height: int = 24) -> str:
    """切牌建議表（用真牌圖，唔用 1m2m 文字）。

    rows: [{"tile": "3m", "shanten": 1, "ukeire_count": 12,
            "ukeire": [("4m", 4), ...]}, ...]
    """
    head = (
        "<tr>"
        + "".join(
            f'<th style="text-align:left;padding:8px 10px;border-bottom:2px solid #ddd;'
            f'font-size:0.85rem;color:#555;white-space:nowrap;">{h}</th>'
            for h in ("切牌", "向聽", "有效牌數", "有效牌")
        )
        + "</tr>"
    )
    body = []
    for row in rows:
        ukeire = row.get("ukeire") or []
        ukeire_html = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:2px;margin:1px 6px 1px 0;">'
            f'{_img(t, ukeire_height)}'
            f'<span style="font-size:0.78rem;color:#666;">{c}</span></span>'
            for t, c in ukeire
        ) or '<span style="color:#999;">—</span>'
        body.append(
            "<tr>"
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">{_img(row["tile"], tile_height)}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">{row["shanten"]}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">{row["ukeire_count"]}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">{ukeire_html}</td>'
            "</tr>"
        )
    return (
        '<table style="width:100%;border-collapse:collapse;margin:6px 0 2px;">'
        f"<thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"
    )

