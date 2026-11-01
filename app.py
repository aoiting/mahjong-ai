import hashlib
import io
import os
import re

import requests
import streamlit as st
from PIL import Image

import ai_client as ai
import mahjong_engine as me
import ml_predict as mlp
import tile_images as ti
import tile_recognizer as tr
import yaku_catalog as yc
import yaku_hints as yh

# OpenRouter 模型（實測後嘅建議）
# 註：minimax-m3.1-flash:free 免費期已於 2026-10-22 結束；Gemini 喺香港被地區封鎖。
DEFAULT_VISION_MODEL = "meta/muse-spark-1.3-contributor"
DEFAULT_ANALYSIS_MODEL = "meta/muse-spark-1.3-contributor"

HONOR_LABELS = {"1z": "東", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "發", "7z": "中"}
SUIT_LABELS = {"m": "萬", "p": "筒", "s": "索"}


def get_secret(*names: str, default: str = "") -> str:
    """由 st.secrets 或環境變數讀取設定。"""
    for name in names:
        try:
            value = st.secrets.get(name)
        except Exception:  # noqa: BLE001 — secrets.toml 可能唔存在
            value = None
        if value:
            return str(value)
        env = os.environ.get(name)
        if env:
            return env
    return default


# ── 手牌代碼解析 ─────────────────────────────────────────────
_ALLOWED_CHARS = set("0123456789mpszr")


def scan_tiles(text: str) -> tuple[list[str] | None, str]:
    """只做語法檢查（唔檢查總數），回傳 (tiles, error)。

    支援連寫（123m）、空格、逗號；`r` 後綴＝赤五（5mr／5pr／5sr）。
    """
    raw = (text or "").strip()
    if not raw:
        return None, "請先輸入手牌。"

    cleaned = re.sub(r"[\s,，、;；]+", "", raw)
    bad = sorted({ch for ch in cleaned if ch not in _ALLOWED_CHARS})
    if bad:
        return None, f"有唔識嘅字元：{''.join(bad)}。請用數字＋花色（m／p／s／z）。"

    tiles: list[str] = []
    i = 0
    n = len(cleaned)
    while i < n:
        if not cleaned[i].isdigit():
            return None, f"「{cleaned[i]}」前面要有一個數字，例如 5m。"
        j = i
        while j < n and cleaned[j].isdigit():
            j += 1
        if j >= n or cleaned[j] not in "mpsz":
            return None, "每組數字後面要跟花色 m／p／s／z。"
        digits = cleaned[i:j]
        suit = cleaned[j]
        i = j + 1

        red = i < n and cleaned[i] == "r"
        if red:
            if digits[-1] != "5" or suit == "z":
                return None, "赤五只可以寫 5mr／5pr／5sr。"
            i += 1

        for pos, ch in enumerate(digits):
            d = int(ch)
            if suit == "z":
                if not 1 <= d <= 7:
                    return None, "字牌只可以係 1z–7z（東南西北白發中）。"
                tiles.append(f"{d}z")
            else:
                if d == 0 or (red and pos == len(digits) - 1):
                    tiles.append(f"5{suit}r")
                else:
                    tiles.append(f"{d}{suit}")

    counts: dict[str, int] = {}
    for tile in tiles:
        base = tile[:-1] if tile.endswith("r") else tile
        counts[base] = counts.get(base, 0) + 1
    over = [b for b, c in counts.items() if c > 4]
    if over:
        return None, f"{'、'.join(over)} 超過 4 隻。"

    return tiles, ""


def parse_hand_codes(
    text: str,
    meld_count: int = 0,
    meld_total: int | None = None,
    meld_codes: list[str] | None = None,
) -> tuple[list[str] | None, str]:
    """手牌隻數＋副露有效隻數必須係 13 或 14；每種牌合共唔可以過 4。

    `meld_total` 係副露「有效隻數」：每組不論 3 隻定 4 隻（槓）都當 3 隻。
    每種牌實際隻數嘅上限檢查仍然用 `meld_codes` 入面嘅真實牌數。
    """
    tiles, err = scan_tiles(text)
    if tiles is None:
        return None, err
    effective_meld_total = (
        int(meld_total) if meld_total is not None else 3 * int(meld_count)
    )
    total = len(tiles) + effective_meld_total
    if total not in (13, 14):
        return None, (
            f"手牌 {len(tiles)} 隻 ＋ 副露 {effective_meld_total} 隻"
            "（每組計 3 隻）＝ "
            f"{total} 隻，應該要係 13 或 14 隻。"
        )
    counts: dict[str, int] = {}
    for tile in list(tiles) + list(meld_codes or []):
        base = tile[:-1] if tile.endswith("r") else tile
        counts[base] = counts.get(base, 0) + 1
    over = [b for b, c in counts.items() if c > 4]
    if over:
        return None, f"{'、'.join(over)} 合共超過 4 隻（手牌＋副露）。"
    return tiles, ""


def hand_to_plain(tiles: list[str]) -> str:
    """tiles（可含 r）→ 精算引擎用嘅標準手牌字串（r 當普通 5）。"""
    return tr.normalize_hand("".join(t[:-1] if t.endswith("r") else t for t in tiles))


def tile_label(tile: str) -> str:
    """'9p' → '9筒'；'5mr' → '赤5萬'；'3z' → '西'。"""
    if not tile:
        return ""
    red = tile.endswith("r")
    base = tile[:-1] if red else tile
    if base.endswith("z"):
        name = HONOR_LABELS.get(base, base)
        return f"赤{name}" if red else name
    prefix = "赤" if red else ""
    return f"{prefix}{base[0]}{SUIT_LABELS.get(base[1], base[1])}"


def display_tile(tile: str, tiles: list[str]) -> str:
    """如果手牌只有赤五、冇普通五，該類型顯示用赤五圖。"""
    if tile in ("5m", "5p", "5s"):
        red = tile + "r"
        if red in tiles and tile not in tiles:
            return red
    return tile


def set_hand(
    tiles: list[str],
    source: str,
    recognized: bool = False,
    drawn: str | None = None,
    meld_tiles: list[list[str]] | None = None,
) -> None:
    st.session_state["hand_tiles"] = list(tiles)
    st.session_state["hand_drawn"] = drawn
    st.session_state["meld_tiles"] = [list(g) for g in (meld_tiles or [])]
    st.session_state["meld_count_pending"] = len(meld_tiles or [])
    st.session_state["hand_source"] = source
    st.session_state.pop("hand_error", None)
    # 手牌一改，舊 AI 解說即刻作廢
    st.session_state.pop("analysis", None)
    st.session_state["recognized_notice"] = bool(recognized)


# ── 辨識結果快取 —— 只快取「成功」嘅結果。─────────────────────
_RECOGNIZE_CACHE: dict[str, tuple[str, str, str | None, list | None]] = {}


def _recognize_now(
    image_bytes: bytes,
    engine: str,
    provider: str,
    api_key: str,
    vision_model: str,
    bottom_pct: int = 100,
    per_tile: bool = True,
    local_first: bool = False,
    status: dict | None = None,
) -> tuple[str | None, str]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = tr.crop_hand_region(image, bottom_pct)

    if engine == "auto":
        engine = "template" if tr.template_status()[0] else "ai"

    if engine == "template":
        return tr.recognize(image, engine="template")

    if provider == "openrouter":
        if per_tile:
            layout = tr.split_hand_layout(image)
            if layout:
                hand_codes, hmsg = ai.recognize_boxes(
                    image, api_key, layout["hand"], vision_model,
                )
                meld_codes, mmsg = ai.recognize_boxes(
                    image, api_key, layout["meld"], vision_model,
                )
                if hand_codes and meld_codes:
                    status = status if status is not None else {}
                    groups = ai.group_meld_tiles(meld_codes)
                    status["meld_tiles"] = groups
                    status["meld_count"] = len(groups)
                    # 有副露時，摸入牌係「手牌區最後一格」（副露區之前），
                    # 唔可以俾之後嘅排序影響佢嘅身份。
                    effective_total = len(hand_codes) + 3 * len(groups)
                    status["drawn"] = (
                        hand_codes[-1] if effective_total == 14 and hand_codes else None
                    )
                    hand = " ".join(ai._sorted_hand_tokens(hand_codes))
                    note = (
                        f"逐張辨識完成（手牌 {len(hand_codes)} 隻 + 副露 {len(meld_codes)} 隻）："
                        f"{hmsg}　副露：{mmsg}"
                    )
                    if any(len(g) not in (3, 4) for g in groups):
                        note += "\n⚠️ 副露分組唔勻（剩 1–2 隻），請逐隻核對。"
                    if effective_total not in (13, 14):
                        note += (
                            f"\n⚠️ 手牌＋副露有效共 {effective_total} 隻"
                            f"（副露 {len(groups)} 組，每組計 3 隻），"
                            "正常應為 13 或 14，請核對。"
                        )
                    return hand, note

            boxes = tr.split_hand_tiles(image)
            if boxes:
                return ai.recognize_hand_per_tile(
                    image, api_key, boxes, vision_model,
                    local_first=local_first, status=status,
                )
            # 切唔到牌 → 用成副辨識，但一定要話俾用戶知準確度較低
            hand, msg = ai.recognize_hand(image, api_key=api_key, model=vision_model)
            warn = (
                "⚠️ 切牌失敗，今次用咗成副辨識，準確度較低，請逐隻核對牌面。"
            )
            return hand, (f"{msg}\n{warn}" if hand else f"{warn}\n{msg}")
        return ai.recognize_hand(image, api_key=api_key, model=vision_model)

    return tr.recognize(image, engine="ollama")


def cached_recognize(
    image_bytes: bytes,
    engine: str,
    provider: str,
    api_key: str,
    vision_model: str,
    force: bool = False,
    bottom_pct: int = 100,
    per_tile: bool = True,
    local_first: bool = False,
    status: dict | None = None,
) -> tuple[str | None, str]:
    """辨識結果快取 —— 失敗唔會入快取，上游恢復就可以即刻重試。"""
    sig = hashlib.md5(
        b"|".join(
            [
                hashlib.md5(image_bytes).digest(),
                engine.encode(),
                provider.encode(),
                vision_model.encode(),
                hashlib.md5(api_key.encode()).digest(),
                str(bottom_pct).encode(),
                b"pertile" if per_tile else b"whole",
                b"localfirst" if local_first else b"ai",
            ]
        )
    ).hexdigest()

    if not force and sig in _RECOGNIZE_CACHE:
        hand, msg, drawn, melds = _RECOGNIZE_CACHE[sig]
        if status is not None:
            status["drawn"] = drawn
            status["meld_tiles"] = melds or []
        return hand, msg

    hand, msg = _recognize_now(
        image_bytes, engine, provider, api_key, vision_model,
        bottom_pct, per_tile, local_first, status=status,
    )
    if hand:
        _RECOGNIZE_CACHE[sig] = (hand, msg, (status or {}).get("drawn"), (status or {}).get("meld_tiles"))
    return hand, msg


@st.cache_data(ttl=30, show_spinner=False)
def local_statuses() -> dict:
    return tr.engines()


@st.cache_data(show_spinner=False, max_entries=50)
def cached_calc(
    hand: str,
    bakaze: str,
    jikaze: str,
    dora: str,
    meld_groups: tuple[tuple[str, ...], ...] = (),
) -> dict:
    return me.analyze(
        hand,
        {"bakaze": bakaze, "jikaze": jikaze, "dora": dora},
        meld_groups=[list(group) for group in meld_groups],
    )


# 高手立直率（來源：天鳳 1194 場高手牌局、首次聽牌點統計）
RICHI_BASE_RATE = 61.4
RICHI_LATE_RATE = 42.8
RICHI_YAKU_RATE = 52.5
RICHI_NO_YAKU_RATE = 74.9


def _riichi_reference(
    hand_tiles: list[str],
    discard_tile: str,
    junme: int,
    meld_count: int,
    bakaze: str,
    jikaze: str,
    dora_text: str,
) -> dict | None:
    """門清、模型首選切完即聽 → 回傳同類等牌嘅高手立直率。"""
    if (
        not hand_tiles
        or not discard_tile
        or int(meld_count) != 0
        or len(hand_tiles) != 14
    ):
        return None

    base = discard_tile[:-1] if discard_tile.endswith("r") else discard_tile
    remain: list[str] = []
    removed = False
    for tile in hand_tiles:
        tb = tile[:-1] if tile.endswith("r") else tile
        if not removed and tb == base:
            removed = True
            continue
        remain.append(tb)
    if not removed:
        return None

    hand13 = tr.normalize_hand("".join(remain))
    calc13 = cached_calc(hand13, bakaze, jikaze, dora_text.strip())
    if calc13.get("error") or calc13.get("shanten") != 0:
        return None

    waits = calc13.get("waits") or []
    kinds = len(waits)
    if kinds == 0:
        return None

    # 剩餘張數：引擎只扣自己手牌；再扣寶牌指示牌（冇牌河資料，屬上限估計）
    indicator_counts: dict[str, int] = {}
    if dora_text.strip():
        for tile in ti.expand_hand(tr.normalize_hand(dora_text)):
            tb = tile[:-1] if tile.endswith("r") else tile
            indicator_counts[tb] = indicator_counts.get(tb, 0) + 1
    remaining = sum(
        max(0, count - indicator_counts.get(wait, 0)) for wait, count in waits
    )
    if remaining <= 0:
        return None

    if kinds == 1:
        rate, bucket = 49.0, "1 種等"
    elif kinds == 2:
        if remaining <= 4:
            rate, bucket = 57.9, "2 種等，剩 1–4 張"
        else:
            rate, bucket = 71.2, "2 種等，剩 5–8 張"
    else:
        if remaining <= 4:
            rate, bucket = 76.6, "3+ 種等，剩 1–4 張"
        else:
            rate, bucket = 78.2, "3+ 種等，剩 5 張以上"

    preview = calc13.get("agari_preview") or []
    has_yaku = any(v.get("yaku") for v in preview)

    return {
        "rate": rate,
        "bucket": bucket,
        "kinds": kinds,
        "remaining": remaining,
        "waits": waits,
        "has_yaku": has_yaku,
        "junme": int(junme),
    }


def _base_tile(tile: str) -> str:
    return tile[:-1] if tile.endswith("r") else tile


def _meld_facts(meld_tiles: list[list[str]]) -> str:
    """副露事實行，例如「副露：碰五索（5sr 5s 5s）」。"""
    if not meld_tiles:
        return ""
    parts = []
    for group in meld_tiles:
        if not group:
            continue
        codes = " ".join(group)
        bases = [_base_tile(t) for t in group]
        if len(group) == 4 and len(set(bases)) == 1:
            call = "槓"
            name = tile_label(bases[0])
        elif len(group) >= 3 and len(set(bases)) == 1:
            call = "碰"
            name = tile_label(bases[0])
        else:
            call = "吃"
            name = "、".join(tile_label(t) for t in group)
        parts.append(f"{call}{name}（{codes}）")
    return "\n副露：" + "；".join(parts) if parts else ""


def _riichi_facts(ref: dict) -> str:
    return (
        "\n【立直率參考】（天鳳 1194 場高手牌局・首次聽牌點統計；"
        "冇牌河資料，剩餘張數係上限估計）\n"
        f"- 同類等牌（{ref['bucket']}，剩約 {ref['remaining']} 張）：{ref['rate']:.1f}%\n"
        f"- 整體基準：{RICHI_BASE_RATE:.1f}%；晚巡（13 巡起）約 {RICHI_LATE_RATE:.1f}%；"
        f"有役約 {RICHI_YAKU_RATE:.1f}%；無役約 {RICHI_NO_YAKU_RATE:.1f}%\n"
        f"- 呢手聽牌有役：{'是' if ref['has_yaku'] else '否'}"
    )


@st.cache_data(ttl=120, show_spinner=False)
def openrouter_connected(key: str) -> bool:
    if not key:
        return False
    try:
        r = requests.get(
            f"{ai.BASE_URL}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=6,
        )
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


# ── 頁面設定 ────────────────────────────────────────────────
st.set_page_config(
    page_title="日本麻雀 AI 牌局分析助手",
    page_icon="🀄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{
  --felt:#0d130f; --panel:#141d16; --panel2:#18241b; --line:#26362a;
  --gold:#d4af37; --gold2:#9a7f2a; --text:#e9e6da; --muted:#9db3a2;
  --danger:#d64545; --green:#2e7d4f; --green-bright:#41b06c;
  --tile:#f7f2df; --tile-edge:#cfc7a8;
}
.stApp{
  background:
    radial-gradient(1200px 600px at 50% -100px, #1a2b1e, var(--felt));
  color:var(--text);
}
[data-testid="stSidebar"]{
  background:var(--panel);
  border-right:1px solid var(--line);
}
h1,h2,h3,h4,h5{color:var(--text) !important; letter-spacing:.2px;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p, label{color:var(--text);}
[data-testid="stCaptionContainer"], small{color:var(--muted) !important;}
hr{border-color:var(--line);}
div[data-testid="stVerticalBlock"][class*="st-key-"]{
  background:var(--panel);
  border-color:var(--line);
}
.stButton>button{
  background:var(--panel2);
  color:var(--text); border:1px solid var(--line);
  border-radius:10px; font-weight:600;
}
.stButton>button:hover{border-color:var(--gold); color:var(--text); background:var(--panel2);}
.stButton>button[kind="primary"]{
  background:var(--gold); color:#1a1405; border:none;
}
.stButton>button[kind="primary"]:hover{background:var(--gold); color:#1a1405;}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea{
  background:var(--panel2); color:var(--text); border:1px solid var(--line); border-radius:10px;
}
button[data-baseweb="tab"]{color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--gold);}
[data-testid="stExpander"] details{
  background:var(--panel); border:1px solid var(--line); border-radius:12px;
}
/* header */
.mj-header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:6px;}
.mj-brand{display:flex;align-items:center;gap:12px;}
.mj-logo{font-size:34px;filter:drop-shadow(0 2px 4px rgba(0,0,0,.4));}
.mj-title{font-size:1.55rem;font-weight:800;color:var(--text);line-height:1.15;margin:0;}
.mj-sub{font-size:.85rem;color:var(--muted);margin:2px 0 0;}
.mj-status{display:flex;gap:8px;flex-wrap:wrap;}
.chip{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;
  font-size:.78rem;border:1px solid var(--line);background:var(--panel2);color:var(--muted);}
.chip.ok{border-color:var(--green);color:var(--green-bright);}
.chip.bad{border-color:var(--danger);color:var(--danger);}
/* steps */
.mj-steps{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:4px 0 14px;color:var(--muted);font-size:.85rem;}
.mj-steps .step{padding:4px 12px;border:1px solid var(--line);border-radius:999px;background:var(--panel2);}
.mj-steps .step.active{border-color:var(--gold);color:var(--text);}
/* mobile summary */
.mj-mobile-summary{display:none;font-size:.82rem;color:var(--gold);
  border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin-bottom:10px;
  background:var(--panel2);}
@media (max-width: 640px){
  .mj-mobile-summary{display:block;}
  .mj-title{font-size:1.2rem;}
  .mj-header{gap:8px;}
}
/* hero */
.st-key-hero_card{
  border:2px solid var(--gold2) !important; border-radius:20px !important;
  box-shadow:0 0 26px rgba(212,175,55,.14);
  background:linear-gradient(135deg,#1c2a1a,var(--panel) 60%) !important;
}
.hero-name{font-size:1.5rem;font-weight:800;color:var(--text);margin:0;}
.hero-pct{font-size:2.6rem;font-weight:800;color:var(--gold);line-height:1.05;margin:2px 0 6px;}
.hero-chip{display:inline-block;padding:4px 11px;border-radius:999px;font-size:.8rem;
  border:1px solid var(--green);color:var(--green-bright);background:var(--panel2);}
/* skeleton */
.sk-line{height:14px;border-radius:7px;background:var(--line);margin:9px 0;}
.sk-line.w90{width:90%;} .sk-line.w70{width:70%;} .sk-line.w50{width:50%;}
/* tile buttons */
[class*="st-key-tilebtn_"] button{height:74px;min-width:46px;width:100%;padding:0 !important;
  background-size:contain;background-repeat:no-repeat;background-position:center;
  background-color:var(--tile);border:1px solid var(--tile-edge);
  border-radius:9px;position:relative;}
[class*="st-key-tilebtn_"] button p{display:none;}
[class*="st-key-tilebtn_"] button:hover{box-shadow:0 0 0 2px var(--gold);}
[class*="st-key-tilebtn_"] button:hover::after{content:"✎";position:absolute;top:-6px;right:-5px;
  background:var(--gold);color:#1a1405;border-radius:50%;width:18px;height:18px;
  font-size:11px;line-height:18px;text-align:center;font-weight:700;}
/* hand row: 14 tiles always one line; 3:4 boxes capped at 64px wide */
[class*="st-key-hand_card"] [data-testid="stHorizontalBlock"]:has(> [class*="st-key-tilebtn_"]) {
  flex-wrap:nowrap !important;
  justify-content:flex-start !important;
  gap:4px !important;
  align-items:flex-end;
  overflow:visible;
}
[class*="st-key-hand_card"] [data-testid="stHorizontalBlock"]:has(> [class*="st-key-tilebtn_"]) > [class*="st-key-tilebtn_"] {
  flex:0 1 64px !important;
  width:64px !important;
  min-width:0 !important;
  max-width:64px !important;
}
[class*="st-key-hand_card"] [data-testid="stHorizontalBlock"]:has(> [class*="st-key-tilebtn_"]) > [data-testid="stLayoutWrapper"] {
  flex:0 1 64px !important;
  width:64px !important;
  min-width:0 !important;
  max-width:64px !important;
}
[class*="st-key-hand_card"] [data-testid="stHorizontalBlock"]:has(> [class*="st-key-tilebtn_"]) > [data-testid="stElementContainer"]:not([class*="st-key-tilebtn_"]) {
  flex:0 0 16px !important;
}
[class*="st-key-hand_card"] [data-testid="stHorizontalBlock"]:has(> [class*="st-key-tilebtn_"]) [class*="st-key-tilebtn_"] button {
  min-width:0 !important;
  min-height:0 !important;
  width:100% !important;
  height:auto !important;
  aspect-ratio:3 / 4 !important;
  background-size:contain !important;
  object-fit:contain;
}
[class*="st-key-hand_card"] [data-testid="stHorizontalBlock"]:has(> [class*="st-key-tilebtn_"]) > [data-testid="stLayoutWrapper"] [data-testid="stElementContainer"] {
  width:100% !important;
  min-width:0 !important;
}
/* hand card columns: give the tile row the width, keep meld area fixed */
[class*="st-key-hand_card"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]) > [data-testid="stColumn"]:first-child {
  flex:1 1 auto !important;
  min-width:0 !important;
}
[class*="st-key-hand_card"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]) > [data-testid="stColumn"]:last-child {
  flex:0 0 150px !important;
  min-width:130px !important;
}
.draw-label{font-size:.7rem;color:var(--gold);text-align:center;margin-bottom:2px;}
[class*="st-key-meldbtn_"] button{height:56px;min-width:28px;width:100%;padding:0 !important;
  background-size:contain;background-repeat:no-repeat;background-position:center;
  background-color:var(--tile);border:1px solid var(--tile-edge);border-radius:7px;}
[class*="st-key-meldbtn_"] button p{display:none;}
[class*="st-key-meldbtn_"] button:hover{box-shadow:0 0 0 2px var(--gold);}
.meld-box{border:1px dashed var(--line);border-radius:12px;padding:10px 12px;
  color:var(--muted);font-size:.8rem;min-height:74px;}
.compact-row{display:flex;align-items:center;gap:8px;margin:5px 0;}
.compact-bar{flex:1;height:6px;background:var(--line);border-radius:4px;overflow:hidden;}
.compact-bar>div{height:100%;background:linear-gradient(90deg,var(--gold2),var(--gold));border-radius:4px;}
</style>
""",
    unsafe_allow_html=True,
)

# ── 側邊欄 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎲 局勢設定")

    col1, col2 = st.columns(2)
    with col1:
        bakaze = st.selectbox("場風", ["東", "南", "西", "北"], index=0)
    with col2:
        jikaze = st.selectbox("自風", ["東", "南", "西", "北"], index=0)

    junme = st.slider("巡目", 1, 18, 1, 1, help="目前牌局進行至第幾巡")
    dora = st.text_input(
        "寶牌指示牌 (Dora)",
        placeholder="例如：3p、5s、東",
        help="輸入牌山翻開嘅指示牌（例如翻開 3p，即代表 4p 係寶牌）",
    )
    if "meld_count_pending" in st.session_state:
        st.session_state["meld_count_input"] = st.session_state.pop("meld_count_pending")
    if "meld_count_input" not in st.session_state:
        st.session_state["meld_count_input"] = 0

    meld_count = st.number_input(
        "已副露組數（鳴牌）",
        min_value=0, max_value=4, step=1,
        key="meld_count_input",
        help="你已經鳴牌（食糊／碰／槓）嘅組數",
    )

    # 用戶人手改組數 → 清走已認到嘅副露牌，退返「得組數」模式
    if st.session_state.get("meld_tiles") and int(meld_count) != len(st.session_state["meld_tiles"]):
        st.session_state["meld_tiles"] = []

    st.markdown("**局勢警報**")
    alert_riichi = st.checkbox("有對手立直")
    alert_meld = st.checkbox("他家有副露")

    if st.button("✨ 載入範例手牌", width="stretch"):
        sample, _err = scan_tiles("34m 4p 1234s 555667z 9p")
        if sample:
            set_hand(sample, "範例手牌")
            st.rerun()

    if st.button("📖 役種一覽", width="stretch"):
        st.session_state["yaku_open"] = True
        st.rerun()

    st.divider()
    with st.expander("⚙️ AI 設定", expanded=False):
        saved_key = get_secret("OPENROUTER_API_KEY")

        provider_label = st.selectbox(
            "AI 供應商",
            ["OpenRouter（雲端）", "本機 Ollama（免費・私隱）"],
            index=0 if saved_key else 1,
        )
        provider = "openrouter" if provider_label.startswith("OpenRouter") else "ollama"

        api_key = ""
        vision_model = DEFAULT_VISION_MODEL
        analysis_model = DEFAULT_ANALYSIS_MODEL

        if provider == "openrouter":
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                value=saved_key,
                placeholder="sk-or-v1-...",
            )
            vision_model = st.text_input(
                "手牌辨識模型",
                value=get_secret("OPENROUTER_VISION_MODEL", default=DEFAULT_VISION_MODEL),
            )
            analysis_model = st.text_input(
                "牌局分析模型",
                value=get_secret("OPENROUTER_ANALYSIS_MODEL", default=DEFAULT_ANALYSIS_MODEL),
            )
            st.caption("辨識被封鎖時會自動轉用 `meta/muse-spark-open`。")

            if st.button("🔌 測試連線", width="stretch"):
                with st.spinner("測試中..."):
                    ok, msg = ai.check_key(api_key, vision_model)
                (st.success if ok else st.error)(msg)
        else:
            ok, msg = local_statuses()["ollama"]
            st.caption(("✅ " if ok else "⚠️ ") + msg)

        st.divider()
        engine_label = st.selectbox(
            "辨識方式",
            ["自動（有模板用模板，否則用 AI）", "AI 視覺辨識", "OpenCV 模板比對"],
            index=0,
        )
        engine = {
            "自動（有模板用模板，否則用 AI）": "auto",
            "AI 視覺辨識": "ai",
            "OpenCV 模板比對": "template",
        }[engine_label]

        tmpl_ok, tmpl_msg = tr.template_status()
        if engine == "template":
            st.caption(("✅ " if tmpl_ok else "⚠️ ") + tmpl_msg)
        elif engine == "auto" and not tmpl_ok:
            st.caption("ℹ️ 模板未齊全，會使用 AI 視覺辨識。")

        per_tile = st.checkbox("先切牌逐張辨識（推薦）", value=True)
        local_first = st.checkbox(
            "⚡ 省 API 模式", value=False, disabled=not per_tile
        )
        st.caption("開啟會降低辨識準確度（本地分析會搶答難牌）")

# ── 頂部狀態 ────────────────────────────────────────────────
or_ok = openrouter_connected(api_key) if provider == "openrouter" and api_key else False
ml_ok = mlp.available()

st.markdown(
    f"""
<div class="mj-header">
  <div class="mj-brand">
    <div class="mj-logo">🀄</div>
    <div>
      <div class="mj-title">日本麻雀 AI 助手</div>
      <div class="mj-sub">入手牌 · 睇建議 · AI 解說</div>
    </div>
  </div>
  <div class="mj-status">
    <span class="chip {'ok' if or_ok else 'bad'}">● OpenRouter：{'已連接' if or_ok else '未連接'}</span>
    <span class="chip {'ok' if ml_ok else 'bad'}">● 高手模型：{'已載入' if ml_ok else '未載入'}</span>
  </div>
</div>
<div class="mj-steps">
  <span class="step active">① 入手牌</span><span>→</span>
  <span class="step">② 睇建議</span><span>→</span>
  <span class="step">③ AI 解說</span>
</div>
""",
    unsafe_allow_html=True,
)

dora_tiles_preview, _ = scan_tiles(dora)
if dora_tiles_preview:
    dora_show = "、".join(tile_label(t) for t in dora_tiles_preview)
else:
    dora_show = dora.strip() or "無"
st.markdown(
    f'<div class="mj-mobile-summary">{bakaze}場 · 自風{jikaze} · {int(junme)} 巡目 · '
    f'{"門清" if int(meld_count) == 0 else f"副露 {int(meld_count)} 組"} · 寶牌 {dora_show}</div>',
    unsafe_allow_html=True,
)

# ── 手牌狀態 ────────────────────────────────────────────────
if "hand_tiles" not in st.session_state:
    st.session_state["hand_tiles"] = []

# ── ① 入手牌 ────────────────────────────────────────────────
with st.container(key="input_card", border=True):
    st.markdown("### ① 入手牌")

    tab_camera, tab_upload, tab_manual = st.tabs(["📸 影相", "🖼️ 上傳圖片", "⌨️ 手入"])

    with tab_camera:
        camera_image = st.camera_input("對準手牌影相", key="camera_input")
        if camera_image is not None:
            st.image(camera_image, caption="擷取畫面", width="stretch")
            data = camera_image.getvalue()
            sig = hashlib.md5(data).hexdigest()
            if st.session_state.get("camera_sig") != sig:
                st.session_state["camera_sig"] = sig
                rec_status: dict = {}
                with st.spinner("即時辨識中..."):
                    hand, msg = cached_recognize(
                        data, engine, provider, api_key, vision_model,
                        force=True, per_tile=per_tile, local_first=local_first,
                        status=rec_status,
                    )
                if hand:
                    set_hand(
                        ti.expand_hand(hand), "即時影相",
                        recognized=True, drawn=rec_status.get("drawn"),
                        meld_tiles=rec_status.get("meld_tiles") or [],
                    )
                    st.session_state["recognize_msg"] = msg
                    st.rerun()
                st.session_state["recognize_msg"] = msg
            if st.button("🔄 重新辨識", key="btn_camera_retry"):
                st.session_state.pop("camera_sig", None)
                st.rerun()

    with tab_upload:
        uploaded_image = st.file_uploader(
            "將截圖或相片拖放至此，或點擊上傳",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            label_visibility="visible",
        )
        st.caption("💡 截圖辨識率最高；實體相片若因光線／角度認錯，可於下方逐隻修改。")

        if uploaded_image is not None:
            st.image(uploaded_image, caption="已上傳的圖片", width="stretch")
            with st.expander("🔍 裁剪設定", expanded=False):
                bottom_pct = st.slider(
                    "只分析畫面底部百分比", 15, 100, 100, 5,
                    help="完整遊戲截圖先需要裁剪；純手牌條通常保持 100。",
                )
            c1, c2 = st.columns(2)
            do_recognize = c1.button("🔍 辨識手牌", key="btn_upload_ocr", width="stretch")
            do_force = c2.button("🔄 重新辨識", key="btn_upload_ocr_force", width="stretch")

            if do_recognize or do_force:
                rec_status: dict = {}
                with st.spinner("辨識中..."):
                    hand, msg = cached_recognize(
                        uploaded_image.getvalue(), engine, provider, api_key,
                        vision_model, force=do_force, bottom_pct=bottom_pct,
                        per_tile=per_tile, local_first=local_first,
                        status=rec_status,
                    )
                st.session_state["recognize_msg"] = msg
                if hand:
                    set_hand(
                        ti.expand_hand(hand), "圖片上傳",
                        recognized=True, drawn=rec_status.get("drawn"),
                        meld_tiles=rec_status.get("meld_tiles") or [],
                    )
                    st.rerun()

    with tab_manual:
        hand_text = st.text_input(
            "手牌代碼",
            placeholder="例如：34m 4p 1234s 555667z 9p",
            key="manual_hand_text",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns(2)
        if c1.button("用呢手牌", type="primary", width="stretch"):
            _meld_groups = st.session_state.get("meld_tiles", [])
            _meld_flat = [t for g in _meld_groups for t in g]
            tiles, err = parse_hand_codes(
                hand_text, int(meld_count),
                meld_total=3 * len(_meld_groups) if _meld_groups else None,
                meld_codes=_meld_flat or None,
            )
            if tiles is None:
                st.session_state["hand_error"] = err
            else:
                set_hand(tiles, "手動輸入")
                st.rerun()
        if c2.button("⌨️ 開啟選牌盤", width="stretch"):
            st.session_state["picker"] = {"mode": "add", "index": None}
            st.rerun()

        st.caption(
            "格式：數字＋花色（m 萬 · p 筒 · s 索 · z 字：1東 2南 3西 4北 5白 6發 7中）；"
            "赤五 5mr；可連寫，如 123m。"
        )

    if st.session_state.get("hand_error"):
        st.error(st.session_state["hand_error"])
    if st.session_state.get("recognize_msg"):
        _msg = st.session_state["recognize_msg"]
        if "切牌失敗" in _msg or "⚠️" in _msg:
            st.warning(_msg)
        else:
            st.caption(_msg)

# ── 推算（手牌、精算、模型）─────────────────────────────────
hand_tiles: list[str] = list(st.session_state.get("hand_tiles", []))
meld_tiles: list[list[str]] = [list(g) for g in st.session_state.get("meld_tiles", [])]
meld_flat: list[str] = [t for g in meld_tiles for t in g]
plain_hand = hand_to_plain(hand_tiles) if hand_tiles else ""

calc: dict = {}
direction_hints, confirmed_yaku = "無明顯役種方向", []
preds: list[dict] | None = None
preds_err = ""

if plain_hand:
    direction_hints, confirmed_yaku = yh.inspect(
        plain_hand, bakaze, jikaze, int(meld_count), meld_flat or None
    )
    calc = cached_calc(
        plain_hand,
        bakaze,
        jikaze,
        dora.strip(),
        tuple(tuple(group) for group in meld_tiles),
    )

if hand_tiles and mlp.available():
    preds, preds_err = mlp.suggest(
        hand_tiles, dora, meld_count=int(meld_count), top=5
    )

model_preds_text = ""
if preds:
    honor_names = {"E": "東", "S": "南", "W": "西", "N": "北", "P": "白板", "F": "發", "C": "中"}
    lines = []
    for rank, item in enumerate(preds, 1):
        label = item["ml"]
        if label in honor_names:
            label = f"{label}（{honor_names[label]}）"
        lines.append(f"{rank}. {label} — {item['p'] * 100:.1f}%")
    model_preds_text = "\n".join(lines)

riichi_ref = None
if preds:
    riichi_ref = _riichi_reference(
        hand_tiles, preds[0]["tile"], int(junme), int(meld_count),
        bakaze, jikaze, dora,
    )

# ── ② 你嘅手牌 ──────────────────────────────────────────────
with st.container(key="hand_card", border=True):
    st.markdown("### ② 你嘅手牌")

    if not hand_tiles:
        st.info("未有手牌。用上面任何一個方法入手牌，或載入範例。")
    else:
        drawn = st.session_state.get("hand_drawn")
        display_tiles = list(hand_tiles)
        index_map = list(range(len(hand_tiles)))
        draw_idx = None

        if len(hand_tiles) % 3 == 2:
            if drawn:
                base = drawn[:-1] if drawn.endswith("r") else drawn
                cands = [
                    i for i, t in enumerate(hand_tiles)
                    if (t[:-1] if t.endswith("r") else t) == base
                ]
                if cands:
                    # 將摸入嗰隻抽出，其餘 13 隻照舊排序，摸入放最右
                    pick = cands[-1]
                    draw_tile = hand_tiles[pick]
                    display_tiles = [t for i, t in enumerate(hand_tiles) if i != pick]
                    index_map = [i for i in range(len(hand_tiles)) if i != pick]
                    display_tiles.append(draw_tile)
                    index_map.append(pick)
                    draw_idx = len(display_tiles) - 1
            else:
                draw_idx = len(display_tiles) - 1

        tile_css = []
        for idx, tile in enumerate(display_tiles):
            uri = ti.tile_uri(tile, 96)
            if uri:
                tile_css.append(
                    f'.st-key-tilebtn_{idx} button{{background-image:url("{uri}");}}'
                )
        if tile_css:
            st.markdown(f"<style>{''.join(tile_css)}</style>", unsafe_allow_html=True)

        meld_css = []
        for gi, group in enumerate(meld_tiles):
            for mi, tile in enumerate(group):
                uri = ti.tile_uri(tile, 96)
                if uri:
                    meld_css.append(
                        f'.st-key-meldbtn_{gi}_{mi} button{{background-image:url("{uri}");}}'
                    )
        if meld_css:
            st.markdown(f"<style>{''.join(meld_css)}</style>", unsafe_allow_html=True)

        left, right = st.columns([5, 1.4], gap="medium")
        with left:
            with st.container(horizontal=True, gap="small", wrap=True, vertical_alignment="bottom"):
                for disp_idx, tile in enumerate(display_tiles):
                    real_idx = index_map[disp_idx]
                    if draw_idx is not None and disp_idx == draw_idx:
                        st.markdown("<div style='width:16px'></div>", unsafe_allow_html=True)
                        with st.container(vertical_alignment="bottom"):
                            st.markdown('<div class="draw-label">摸入</div>', unsafe_allow_html=True)
                            if st.button(
                                " ", key=f"tilebtn_{disp_idx}",
                                help=f"{tile_label(tile)}（點擊修改）",
                            ):
                                st.session_state["picker"] = {"mode": "replace", "index": real_idx}
                                st.rerun()
                    else:
                        if st.button(
                            " ", key=f"tilebtn_{disp_idx}",
                            help=f"{tile_label(tile)}（點擊修改）",
                        ):
                            st.session_state["picker"] = {"mode": "replace", "index": real_idx}
                            st.rerun()

        with right:
            if meld_tiles:
                st.markdown("**副露**")
                for gi, group in enumerate(meld_tiles):
                    with st.container(horizontal=True, gap="small", wrap=True):
                        for mi, tile in enumerate(group):
                            if st.button(
                                " ", key=f"meldbtn_{gi}_{mi}",
                                help=f"{tile_label(tile)}（點擊修改副露）",
                            ):
                                st.session_state["picker"] = {
                                    "mode": "replace", "target": "meld",
                                    "group": gi, "index": mi,
                                }
                                st.rerun()
            elif int(meld_count) == 0:
                st.markdown(
                    '<div class="meld-box">門清 — 冇副露<br>食・碰・槓顯示喺呢度</div>',
                    unsafe_allow_html=True,
                )
            else:
                faces = "".join(
                    '<span style="display:inline-block;width:22px;height:30px;margin:2px;'
                    'border-radius:4px;background:var(--panel2);'
                    'border:1px solid var(--line);"></span>'
                    for _ in range(int(meld_count) * 3)
                )
                st.markdown(
                    f'<div class="meld-box">副露 {int(meld_count)} 組<br>{faces}</div>',
                    unsafe_allow_html=True,
                )

        st.caption("點擊牌面可修正辨識錯誤（✎）。")

# ── 首選大卡 ────────────────────────────────────────────────
if preds:
    top = preds[0]
    second = preds[1] if len(preds) > 1 else None
    top_tile = display_tile(top["tile"], hand_tiles)
    second_tile = display_tile(second["tile"], hand_tiles) if second else None
    diff = (top["p"] - second["p"]) * 100 if second else 0.0
    with st.container(key="hero_card", border=True):
        st.markdown("### 🏆 首選建議")
        c1, c2 = st.columns([1, 3.2], gap="medium")
        with c1:
            st.markdown(ti.inline_html(top_tile, 96), unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="hero-name">{tile_label(top_tile)}</div>'
                f'<div class="hero-pct">{top["p"] * 100:.1f}%</div>'
                + (
                    f'<span class="hero-chip">領先次選（{tile_label(second_tile)}）+{diff:.1f}%</span>'
                    if second
                    else ""
                ),
                unsafe_allow_html=True,
            )
            st.caption("由高手模型（handonly_model）計算，只考慮手牌、寶牌指示及副露數。")
elif preds_err:
    st.info(preds_err)

# ── ③ AI 解說（全闊）─────────────────────────────────────────
with st.container(key="ai_card", border=True):
    head_l, head_r = st.columns([3.2, 1])
    with head_l:
        st.markdown("### ③ AI 解說")
    with head_r:
        analyze_btn = st.button(
            "🚀 AI 深度分析",
            type="primary",
            width="stretch",
            disabled=not bool(hand_tiles),
        )

    ctx = {
        "bakaze": bakaze,
        "jikaze": jikaze,
        "junme": junme,
        "dora": dora.strip(),
        "riichi": alert_riichi,
        "meld": alert_meld,
        "meld_count": int(meld_count),
    }
    facts = (
        me.format_facts(calc)
        .replace("精算結果", "牌理分析結果")
        .replace("精算", "牌理")
        .replace("Pinfu", "平和")
        .replace("Tanyao", "斷么九")
        .replace("Honitsu", "混一色")
        .replace("Chiitoitsu", "七對子")
        .replace("Chiitoi", "七對子")
        .replace("Ittsu", "一氣通貫")
        .replace("Sanshoku", "三色同順")
        if calc
        else ""
    )
    if meld_tiles:
        facts += _meld_facts(meld_tiles)
    if riichi_ref:
        facts += _riichi_facts(riichi_ref)
    analysis_sig = (
        plain_hand,
        bakaze,
        jikaze,
        int(junme),
        dora.strip(),
        bool(alert_riichi),
        bool(alert_meld),
        int(meld_count),
        model_preds_text,
        direction_hints,
        analysis_model,
        provider,
    )

    rendered = False

    if analyze_btn:
        if not hand_tiles:
            st.error("請先輸入手牌。")
        elif provider == "openrouter" and not api_key:
            st.error("雲端模式需要 OpenRouter API Key，請先在左側 AI 設定輸入。")
        else:
            prev = st.session_state.get("analysis")
            if prev and prev.get("sig") == analysis_sig and prev.get("text"):
                st.info("已重用上次解說（手牌／局勢／模型建議冇變），冇再呼叫 API。")
            else:
                ph = st.empty()
                ph.markdown(
                    '<div class="sk-line w90"></div><div class="sk-line w70"></div>'
                    '<div class="sk-line w90"></div><div class="sk-line w50"></div>',
                    unsafe_allow_html=True,
                )
                status: dict = {}
                text, msg = None, ""
                try:
                    ph.empty()
                    with st.spinner("AI 解說生成中…"):
                        with ph.container():
                            streamed = st.write_stream(
                                ai.stream_analysis(
                                    plain_hand, ctx,
                                    backend=provider,
                                    api_key=api_key,
                                    model=analysis_model,
                                    facts=facts,
                                    model_preds=model_preds_text,
                                    direction_hints=direction_hints,
                                    status=status,
                                )
                            )
                    text = (streamed or "").strip()
                    if not text:
                        raise ai.AIError("模型冇回傳內容。")
                    used_model = status.get("model", analysis_model)
                    msg = f"分析完成（{used_model}，串流）"
                    if used_model != analysis_model and status.get("primary_error"):
                        msg += (
                            f"　⚠️ 主模型 {analysis_model} 暫時不可用"
                            f"（{ai._short_reason(status['primary_error'])}），已自動轉用備援。"
                        )
                    ph.empty()
                    with ph.container():
                        st.markdown(
                            ti.decorate_text(text, height=26),
                            unsafe_allow_html=True,
                        )
                    st.caption(msg)
                    rendered = True
                except Exception:  # noqa: BLE001 — 串流失敗 → 自動 fallback 非串流
                    ph.empty()
                    with st.spinner("串流失敗，改用普通模式分析中..."):
                        text, msg = ai.analyze_hand(
                            plain_hand, ctx,
                            backend=provider,
                            api_key=api_key,
                            model=analysis_model,
                            facts=facts,
                            model_preds=model_preds_text,
                            direction_hints=direction_hints,
                        )
                    if text:
                        with ph.container():
                            st.markdown(
                                ti.decorate_text(text, height=26),
                                unsafe_allow_html=True,
                            )
                        st.caption(msg)
                        rendered = True

                st.session_state["analysis"] = {
                    "hand": plain_hand,
                    "text": text,
                    "msg": msg,
                    "provider": provider,
                    "sig": analysis_sig,
                }
                st.session_state["recognized_notice"] = False

    result = st.session_state.get("analysis")
    if not rendered and result:
        if result.get("text"):
            with st.container():
                st.markdown(
                    ti.decorate_text(result["text"], height=26),
                    unsafe_allow_html=True,
                )
            st.caption(result["msg"])
            st.warning(
                "⚠️ AI 解說僅供參考，可能有錯（例如誤判役種、向聽數）。重要決定請自行核對。"
            )
        else:
            st.error(f"分析失敗：{result.get('msg', '未知錯誤')}")
    elif not analyze_btn and not result:
        if st.session_state.get("recognized_notice"):
            st.info(
                "認牌完成 — 請先核對牌面（撳任何一隻牌可以改牌），"
                "確認後再㩒『AI 深度分析』。"
            )
        elif hand_tiles:
            st.caption("撳「🚀 AI 深度分析」，AI 會跟住高手模型第一位同牌理數字講解。")
        else:
            st.caption("入手牌之後，呢度會顯示 AI 解說。")
# ── 底部：Top 5 ＋ 精算 ─────────────────────────────────────
st.write("")
bottom_left, bottom_right = st.columns([1, 1.15], gap="medium")

with bottom_left:
    with st.container(key="top5_card", border=True):
        st.markdown("### 🏆 模型建議 Top 5")
        st.caption("高手實戰選擇（計埋役種價值）")
        if not hand_tiles:
            st.caption("入手牌後顯示。")
        elif preds:
            for rank, item in enumerate(preds, 1):
                disp_tile = display_tile(item["tile"], hand_tiles)
                st.markdown(
                    f'<div class="compact-row">'
                    f'<span style="width:14px;color:var(--muted);font-size:.78rem;">{rank}</span>'
                    f'{ti.inline_html(disp_tile, 26)}'
                    f'<span style="min-width:34px;font-size:.78rem;">{tile_label(disp_tile)}</span>'
                    f'<div class="compact-bar"><div style="width:{item["p"] * 100:.1f}%;"></div></div>'
                    f'<span style="min-width:48px;text-align:right;font-size:.78rem;'
                    f'font-variant-numeric:tabular-nums;">{item["p"] * 100:.1f}%</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        elif preds_err:
            st.caption(preds_err)
        else:
            st.caption("模型未載入。")

with bottom_right:
    with st.container(key="calc_card", border=True):
        st.markdown("### 📐 牌理分析")
        st.caption("純牌效率：向聽＋進張數，唔計役種")
        if not plain_hand:
            st.caption("入手牌後即時計算。")
        elif calc.get("error"):
            st.warning(calc["error"])
        else:
            if calc.get("discards"):
                best_discard = calc["discards"][0]
                best_tile = display_tile(best_discard["tile"], hand_tiles)
                headline = (
                    f"{best_discard['shanten']} 向聽 · 進張最闊："
                    f"切{best_tile}（{best_discard['ukeire_count']} 張）"
                )
            else:
                headline = calc.get("summary", "")
            st.markdown(
                '<div style="background:var(--panel2);'
                'border:1px solid var(--line);border-radius:12px;'
                'padding:10px 14px;margin-bottom:8px;">'
                + ti.decorate_text(headline, height=26)
                + "</div>",
                unsafe_allow_html=True,
            )
            if riichi_ref:
                st.caption(
                    f"📊 同類等牌，高手首次聽牌立直率約 {riichi_ref['rate']:.1f}%"
                )

            if calc.get("discards"):
                rows = [
                    {
                        "tile": display_tile(d["tile"], hand_tiles),
                        "shanten": d["shanten"],
                        "ukeire_count": d["ukeire_count"],
                        "ukeire": d["ukeire"],
                    }
                    for d in calc["discards"][:6]
                ]
                st.markdown(ti.discard_table_html(rows), unsafe_allow_html=True)

            if calc.get("waits"):
                st.markdown("**待牌**")
                st.markdown(ti.waits_html(calc["waits"]), unsafe_allow_html=True)

            if calc.get("agari_preview"):
                st.markdown("**和牌預覽**（榮和、門清）：")
                for v in calc["agari_preview"]:
                    yaku = "、".join(v["yaku"]) or "無役"
                    cost = (v.get("cost") or {}).get("main", 0)
                    st.markdown(
                        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:4px 0;">'
                        f'<span style="color:var(--muted);font-size:0.9rem;">待</span>'
                        f'{ti.inline_html(v["wait"], 28)}'
                        f'<span>{yaku}　{v["han"]} 翻 {v["fu"]} 符　{cost} 點</span></div>',
                        unsafe_allow_html=True,
                    )

            if calc.get("agari"):
                for kind, v in calc["agari"].items():
                    yaku = "、".join(v["yaku"]) or "無役"
                    st.markdown(f"- **{kind}**：{yaku}　{v['han']} 翻 {v['fu']} 符")

            if not calc.get("agari_preview") and not calc.get("agari"):
                st.markdown("**🏆 役種**")
                st.markdown(f"- 已確定役牌：{'、'.join(confirmed_yaku) or '暫無'}")
                st.markdown(f"- 潛在方向：{direction_hints}")

            st.caption("以上由麻雀計算引擎精確算出（唔涉及 AI）。")


# ── 選牌盤 ──────────────────────────────────────────────────
def _apply_pick(tile: str) -> None:
    info = st.session_state.get("picker") or {}
    mode = info.get("mode", "add")
    index = info.get("index")

    if info.get("target") == "meld":
        groups = [list(g) for g in st.session_state.get("meld_tiles", [])]
        gi, mi = info.get("group"), info.get("index")
        if gi is not None and mi is not None and 0 <= gi < len(groups) and 0 <= mi < len(groups[gi]):
            groups[gi][mi] = tile
        st.session_state["meld_tiles"] = groups
        st.session_state["meld_count_pending"] = len(groups)
        st.session_state.pop("analysis", None)
        st.session_state["recognized_notice"] = False
        st.session_state.pop("picker", None)
        st.rerun()
        return

    tiles = list(st.session_state.get("hand_tiles", []))

    base = tile[:-1] if tile.endswith("r") else tile
    count = sum(1 for t in tiles if (t[:-1] if t.endswith("r") else t) == base)
    if mode != "replace" and count >= 4:
        st.warning(f"{base} 已經有 4 隻，唔可以再加。")
        return

    if mode == "replace" and index is not None and 0 <= index < len(tiles):
        tiles[index] = tile
    else:
        if len(tiles) >= 14:
            st.warning("手牌已經有 14 隻。")
            return
        tiles.append(tile)

    st.session_state["hand_tiles"] = tiles
    st.session_state["hand_source"] = "選牌盤修正"
    st.session_state["hand_drawn"] = None
    st.session_state.pop("analysis", None)
    st.session_state["recognized_notice"] = False
    st.session_state.pop("picker", None)
    st.rerun()


@st.dialog("🀄 選牌盤")
def tile_picker_dialog() -> None:
    info = st.session_state.get("picker") or {}
    mode = info.get("mode", "add")
    index = info.get("index")
    if info.get("target") == "meld":
        gi, mi = info.get("group"), info.get("index")
        try:
            current = st.session_state["meld_tiles"][gi][mi]
            st.caption(f"替換副露第 {gi + 1} 組、第 {mi + 1} 隻（{tile_label(current)}）")
        except Exception:  # noqa: BLE001
            st.caption("修改副露")
    elif mode == "replace" and index is not None:
        current = (st.session_state.get("hand_tiles") or [])[index]
        st.caption(f"替換第 {index + 1} 隻（{tile_label(current)}）")
    else:
        st.caption("揀一隻牌加入手牌")

    # 牌掣用真牌圖（per-key CSS background-image），右上角加代碼 badge
    badge_style = (
        "position:absolute;top:2px;right:3px;font-size:9px;line-height:1;"
        "padding:2px 3px;border-radius:4px;background:rgba(13,19,15,.55);"
        "color:#f7f2df;font-weight:600;pointer-events:none;z-index:2;"
    )
    picker_css: list[str] = [
        '[class*="st-key-pick_"] button{'
        "height:auto !important;aspect-ratio:3/4;min-height:0 !important;"
        "width:100%;padding:0 !important;"
        "background-size:contain;background-repeat:no-repeat;background-position:center;"
        "background-color:var(--tile);border:1px solid var(--tile-edge);"
        "border-radius:8px;position:relative;overflow:hidden;"
        "}",
        '[class*="st-key-pick_"] button p{display:none;}',
        '.st-key-pick_m_red,.st-key-pick_p_red,.st-key-pick_s_red{'
        "max-width:56px !important;margin-left:auto !important;margin-right:auto !important;}",
    ]

    def _add_tile_css(key: str, code: str) -> None:
        uri = ti.tile_uri(code, 96)
        if not uri:
            return
        picker_css.append(f'.st-key-{key} button{{background-image:url("{uri}");}}')
        picker_css.append(
            f'.st-key-{key} button::after{{content:"{code}";{badge_style}}}'
        )

    for suit in "mps":
        for n in range(1, 10):
            _add_tile_css(f"pick_{suit}{n}", f"{n}{suit}")

    for n in range(1, 8):
        _add_tile_css(f"pick_z{n}", f"{n}z")

    for suit in "mps":
        _add_tile_css(f"pick_{suit}_red", f"5{suit}r")

    st.markdown(f"<style>{''.join(picker_css)}</style>", unsafe_allow_html=True)

    for suit, title in (("m", "萬子"), ("p", "筒子"), ("s", "索子")):
        st.markdown(f"**{title}**")
        cols = st.columns(9)
        for n in range(1, 10):
            if cols[n - 1].button(f"{n}{suit}", key=f"pick_{suit}{n}", width="stretch"):
                _apply_pick(f"{n}{suit}")

    st.markdown("**字牌**")
    cols = st.columns(7)
    for n, name in enumerate(["東", "南", "西", "北", "白", "發", "中"], 1):
        if cols[n - 1].button(name, key=f"pick_z{n}", width="stretch"):
            _apply_pick(f"{n}z")

    st.markdown("**赤五**")
    cols = st.columns(3)
    for k, suit in enumerate("mps"):
        if cols[k].button(f"赤5{suit}", key=f"pick_{suit}_red", width="stretch"):
            _apply_pick(f"5{suit}r")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🗑️ 清除", width="stretch"):
        info = st.session_state.get("picker") or {}
        if info.get("target") == "meld":
            groups = [list(g) for g in st.session_state.get("meld_tiles", [])]
            gi, mi = info.get("group"), info.get("index")
            if gi is not None and mi is not None and 0 <= gi < len(groups) and 0 <= mi < len(groups[gi]):
                groups[gi].pop(mi)
                if not groups[gi]:
                    groups.pop(gi)
            st.session_state["meld_tiles"] = groups
            st.session_state["meld_count_pending"] = len(groups)
            st.session_state.pop("analysis", None)
            st.session_state["recognized_notice"] = False
            st.session_state.pop("picker", None)
            st.rerun()
        else:
            tiles = list(st.session_state.get("hand_tiles", []))
            if mode == "replace" and index is not None and 0 <= index < len(tiles):
                tiles.pop(index)
            elif tiles:
                tiles.pop()
            st.session_state["hand_tiles"] = tiles
            st.session_state["hand_source"] = "選牌盤修正"
            st.session_state["hand_drawn"] = None
            st.session_state.pop("analysis", None)
            st.session_state["recognized_notice"] = False
            st.session_state.pop("picker", None)
            st.rerun()
    if c2.button("完成", type="primary", width="stretch"):
        st.session_state.pop("picker", None)
        st.rerun()


def _close_yaku_dialog() -> None:
    st.session_state["yaku_open"] = False


@st.dialog(
    "📖 役種一覽",
    width="large",
    on_dismiss=_close_yaku_dialog,
)
def _yaku_dialog() -> None:
    st.markdown(
        """
<style>
[data-testid="stDialog"] section[role="dialog"]{
  max-width:min(900px,94vw) !important;width:min(900px,94vw) !important;
  margin-left:auto !important;margin-right:auto !important;}
[data-testid="stDialog"]{overflow-y:auto !important;}
.yaku-card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  padding:10px 12px;margin:8px 0;}
.yaku-head{display:flex;align-items:center;justify-content:space-between;gap:10px;}
.yaku-name{color:var(--text);font-weight:700;font-size:1rem;}
.yaku-tag{font-size:.72rem;color:var(--gold);border:1px solid var(--gold2);
  border-radius:999px;padding:2px 8px;white-space:nowrap;}
.yaku-desc{color:var(--muted);font-size:.8rem;line-height:1.5;margin-top:4px;}
.yaku-ex{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:6px;}
.yaku-grp{display:inline-flex;gap:1px;padding-right:6px;}
.yaku-foot{color:var(--muted);font-size:.78rem;line-height:1.6;margin-top:14px;
  border-top:1px dashed var(--line);padding-top:10px;}
</style>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "搜尋役種",
        placeholder="搜尋役名或關鍵字，例如：混、順子、字牌",
        label_visibility="collapsed",
        key="yaku_search",
    )
    selected = st.pills(
        "翻數",
        ["全部", "一翻", "二翻", "三翻", "六翻", "役滿", "狀況"],
        default="全部",
        selection_mode="single",
        key="yaku_filter",
    ) or "全部"

    group_for_filter = {
        "一翻": "一翻役",
        "二翻": "二翻役",
        "三翻": "三翻役",
        "六翻": "六翻役",
        "役滿": "役滿",
        "狀況": "狀況役",
    }
    q = (search or "").strip().lower()
    shown = 0

    for group_title, items in yc.YAKU_GROUPS:
        if selected != "全部" and group_title != group_for_filter.get(selected):
            continue
        matched = [
            item for item in items
            if not q or q in f"{item['name']} {item['desc']}".lower()
        ]
        if not matched:
            continue
        st.markdown(f"### {group_title}")
        for item in matched:
            examples = []
            for group in item.get("example") or []:
                imgs = ti.inline_html(group, 40)  # 高度 40px ≈ 闊 30px
                if imgs:
                    examples.append(f'<span class="yaku-grp">{imgs}</span>')
            ex_html = (
                f'<div class="yaku-ex">{"".join(examples)}</div>' if examples else ""
            )
            st.markdown(
                f'<div class="yaku-card">'
                f'<div class="yaku-head"><span class="yaku-name">{item["name"]}</span>'
                f'<span class="yaku-tag">{item["han"]}</span></div>'
                f'<div class="yaku-desc">{item["desc"]}</div>'
                f"{ex_html}</div>",
                unsafe_allow_html=True,
            )
            shown += 1

    if shown == 0:
        st.info("搵唔到符合嘅役種")
    else:
        st.caption(f"顯示 {shown} / {sum(len(items) for _, items in yc.YAKU_GROUPS)} 項")

    st.markdown(f'<div class="yaku-foot">{yc.FOOTER_NOTE}</div>', unsafe_allow_html=True)
    if st.button("關閉", type="primary", width="stretch"):
        st.session_state["yaku_open"] = False
        st.rerun()


if st.session_state.get("picker"):
    tile_picker_dialog()

if st.session_state.get("yaku_open"):
    _yaku_dialog()
