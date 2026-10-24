"""麻將手牌辨識引擎（可插拔、零 API 成本）

重要：easyocr / paddleocr 等「文字 OCR」認唔到麻雀牌圖案，
因為牌面係圖形（筒子圓圈、索子竹節、萬子書法），唔係文字。
本模組提供兩個真正可行的本地後端：

1. ollama   — 本地視覺語言模型（預設 qwen2.5vl:7b）
              可讀遊戲截圖或實體牌照片，零 token 費用。
2. template — OpenCV 模板比對
              只適用於固定遊戲客戶端；需自行於 tile_templates/
              放入 34 張模板圖（檔名如 1m.png、5z.png）。

兩個後端都唔會呼叫任何雲端 API。
"""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

# ── 牌種定義 ────────────────────────────────────────────────
TILE_LABELS = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + [f"{n}z" for n in range(1, 8)]
)

HONOR_TO_Z = {"東": 1, "南": 2, "西": 3, "北": 4, "白": 5, "發": 6, "中": 7}

# ── 後端設定（可用環境變數覆寫）──────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("MAHJONG_VLM_MODEL", "qwen2.5vl:7b")
TEMPLATE_DIR = Path(__file__).with_name("tile_templates")

PROMPT = """你係日本麻雀牌辨識器。請辨識圖片中「手牌」的 13 或 14 張牌。

輸出規則：
- 只用壓縮記法：萬子 m、筒子 p、索子 s、字牌 z
- 數字在前、花色在後，依 m → p → s → z 順序分組
- 字牌 z：1=東 2=南 3=西 4=北 5=白 6=發 7=中
- 只輸出一行純文字，唔要解釋、標點、空格或程式碼區塊
- 如果圖片完全冇手牌，只輸出 NONE

範例：123m456p789s11z
"""


# ── 文字正規化 ──────────────────────────────────────────────
def normalize_hand(text: str) -> str:
    """將引擎或使用者輸入清洗成標準格式，例如 123m456p789s11z。

    容許輸入：數字＋花色、中文花色（萬筒索）、字牌中文（東南西北白發中）、
    赤牌「0」（會視為 5）。輸出永遠按 m → p → s → z 分組並升序排列。
    """
    if not text:
        return ""

    t = str(text).strip()
    # 全形數字轉半形
    t = t.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    # 花色中文 → 英文字母
    t = (
        t.replace("萬", "m")
        .replace("筒", "p")
        .replace("餅", "p")
        .replace("索", "s")
        .replace("條", "s")
    )
    # 字牌中文 → nz
    for ch, n in HONOR_TO_Z.items():
        t = t.replace(ch, f"{n}z")
    # 赤牌標記
    t = t.replace("赤", "0")

    buckets: dict[str, list[int]] = {"m": [], "p": [], "s": [], "z": []}
    for digits, suit in re.findall(r"(\d+)([mpsz])", t.lower()):
        limit = 7 if suit == "z" else 9
        for d in digits:
            n = int(d)
            if n == 0:          # 0 = 赤牌，等同 5
                n = 5
            if 1 <= n <= limit:
                buckets[suit].append(n)

    return "".join(
        f"{''.join(str(n) for n in sorted(buckets[suit]))}{suit}"
        for suit in "mpsz"
        if buckets[suit]
    )


def count_tiles(hand: str) -> int:
    """計算手牌張數。"""
    return sum(len(run) for run in re.findall(r"\d+", hand or ""))


def _validate_hand(hand: str) -> str | None:
    """回傳錯誤訊息；None 代表正常。"""
    if not hand:
        return None
    n = count_tiles(hand)
    if n not in (13, 14):
        return f"辨識到 {n} 張牌（正常應為 13 或 14 張），建議手動修正。"
    # 逐張牌檢查（唔可以整組計，因為 11567z 呢類合法組合會有 5 張字牌）
    for digits, suit in re.findall(r"(\d+)([mpsz])", hand):
        counts: dict[str, int] = {}
        for d in digits:
            counts[d] = counts.get(d, 0) + 1
        over = [d for d, c in counts.items() if c > 4]
        if over:
            shown = "、".join(f"{d}{suit}" for d in over)
            return f"{shown} 超過 4 張，建議手動檢查。"
    return None


# ── 後端狀態 ────────────────────────────────────────────────
def ollama_status() -> tuple[bool, str]:
    """檢查 Ollama 服務與模型是否可用。"""
    try:
        import requests

        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code != 200:
            return False, f"Ollama 回應 HTTP {r.status_code}"
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if not models:
            return False, "Ollama 已啟動，但未下載任何模型"
        base = OLLAMA_MODEL.split(":")[0]
        hit = [m for m in models if m.split(":")[0].startswith(base)]
        if not hit:
            return False, f"Ollama 已啟動，但未安裝 {OLLAMA_MODEL}（可用：{', '.join(models[:5])}）"
        return True, f"Ollama 就緒：{hit[0]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"未偵測到 Ollama（{type(exc).__name__}）"


def template_status() -> tuple[bool, str]:
    """檢查模板資料夾是否齊全。"""
    if not TEMPLATE_DIR.is_dir():
        return False, f"未建立模板資料夾：{TEMPLATE_DIR}"
    found = {p.stem for p in TEMPLATE_DIR.glob("*.png")} & set(TILE_LABELS)
    if not found:
        return False, f"{TEMPLATE_DIR.name}/ 內沒有合法模板（需 34 張，如 1m.png）"
    if len(found) < 34:
        return False, f"模板只有 {len(found)}/34 張"
    return True, f"模板齊全（{len(found)}/34）"


def engines() -> dict[str, tuple[bool, str]]:
    return {"ollama": ollama_status(), "template": template_status()}


# ── 主入口 ──────────────────────────────────────────────────
def recognize(image: Image.Image, engine: str = "auto") -> tuple[str | None, str]:
    """辨識手牌。

    Args:
        image: PIL 圖片。
        engine: "auto" | "ollama" | "template"

    Returns:
        (hand, message)：hand 為標準格式字串或 None；message 為提示訊息。
    """
    if image is None:
        return None, "沒有圖片。"

    img = image.convert("RGB")
    if img.width > 1600:  # 控制 CPU 推理時間
        ratio = 1600 / img.width
        img = img.resize((1600, int(img.height * ratio)))

    if engine == "auto":
        if ollama_status()[0]:
            engine = "ollama"
        elif template_status()[0]:
            engine = "template"
        else:
            return None, "未偵測到可用辨識引擎，請先安裝 Ollama 模型或建立模板。"

    if engine == "ollama":
        return _recognize_ollama(img)
    if engine == "template":
        return _recognize_template(img)
    return None, f"未知引擎：{engine}"


def tile_color_profile(tile: Image.Image) -> dict:
    """本地色彩分析（純 numpy，零成本、零 API），用嚟輔助／糾正 AI 辨識。

    實測（雀魂牌面）：
      - 白（5z）係空白牌 → 墨量極低（只餘邊框）
      - 索子（2s–9s）同「發」係綠色；其他牌綠色等於 0
      - 萬子／筒子／東南西北／白：綠色 0
      - 「中」係大量紅色但近乎冇黑墨
    """
    a = np.asarray(tile.convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    m = max(1, int(min(h, w) * 0.06))  # 只裁最外 6%（保留白牌邊框）
    a = a[m:h - m, m:w - m]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = mx - mn
    tot = float(a.shape[0] * a.shape[1])
    green = float(((g > r + 25) & (g > b + 25) & (sat > 55)).sum()) / tot
    red = float(((r > g + 30) & (r > b + 30) & (sat > 55)).sum()) / tot
    blue = float(((b > r + 25) & (b > g + 15) & (sat > 55)).sum()) / tot
    dark = float(((mx < 120) & (sat <= 55)).sum()) / tot
    return {
        "green": green,
        "red": red,
        "blue": blue,
        "dark": dark,
        "ink": green + red + blue + dark,
    }


BLANK_INK_MAX = 0.05      # 低於此墨量 = 空白牌（白）
GREEN_MIN = 0.02          # 高於此綠量 = 一定係索子或發


def tile_shape_analysis(tile: Image.Image) -> tuple[str, str | None]:
    """用連通元件嘅形狀推斷牌種。

    Returns:
        (提示文字, 高信心答案或 None)
        答案唔係 None 代表本地分析已經足夠肯定。

    實測（雀魂牌面）：
      - 筒子 = 圓形元件（高闊比 ≈ 1.0、填充 ≥ 0.55）
      - 索子（2s–9s）= 窄長直條（高闊比 < 0.7）
      - 萬子 = 紅色「萬」+ 黑色漢數字（填充 < 0.5，唔符合上面兩者）
      - 字牌 = 一個大字元（填充 < 0.5）
    顏色唔可靠（呢個牌庫嘅 5s 係紅色竹枝），所以主要睇形狀。
    """
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return "", None

    profile = tile_color_profile(tile)

    try:
        a = np.asarray(tile.convert("RGB")).astype(np.float32)
        h, w, _ = a.shape
        m = max(1, int(min(h, w) * 0.12))
        a2 = a[m:h - m, m:w - m]
        r, g, b = a2[..., 0], a2[..., 1], a2[..., 2]
        mx, mn = a2.max(axis=2), a2.min(axis=2)
        sat = mx - mn
        ink = (((sat > 55) | (mx < 135))).astype(np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    except Exception:  # noqa: BLE001
        return "", None

    min_area = max(12, 0.0015 * a2.shape[0] * a2.shape[1])
    bars = rounds = wide = 0
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if area < min_area:
            continue
        ar = cw / max(1, ch)
        fill = area / float(cw * ch)
        if ar <= 0.7:
            bars += 1
        elif ar >= 1.8:
            wide += 1
        elif 0.8 <= ar <= 1.25 and fill >= 0.55:
            rounds += 1

    # ── 有綠色 → 索子（2s–9s，直條）或者 1s（雀鳥）或者 發（單一綠色大字）
    if profile["green"] >= GREEN_MIN:
        if bars >= 2:
            return (f"本地形狀分析：偵測到 {bars} 條窄長直條 + 有綠色 → 係索子，"
                    f"數量 {bars}。", f"{bars}s")
        if wide >= 1:
            return "本地形狀分析：有綠色 + 橫向元件 → 應該係 1s（雀鳥）。", "1s"
        if bars == 0 and wide == 0 and rounds <= 2:
            return "本地形狀分析：單一綠色大字 → 應該係「發」(6z)。", "6z"
        return "", None

    # ── 冇綠色
    total = bars + rounds
    if total >= 2 and bars >= 0.75 * total:
        return (f"本地形狀分析：偵測到 {bars} 條窄長直條 → 係索子，數量 {bars}。",
                f"{bars}s")
    if total >= 2 and rounds >= 0.75 * total:
        return (f"本地形狀分析：偵測到 {rounds} 個圓形 → 係筒子，數量 {rounds}。",
                f"{rounds}p")
    if rounds == 1 and bars == 0:
        return "本地形狀分析：偵測到 1 個大圓形 → 應該係 1p。", "1p"
    if profile["red"] >= 0.11 and profile["dark"] >= 0.02 and total == 0:
        return ("本地形狀分析：紅色「萬」+ 黑色漢數字 → 係萬子，"
                "請集中讀上半部分嘅漢數字。", None)
    return "", None


def tile_shape_hint(tile: Image.Image) -> str:
    """只取提示文字（唔要本地答案）。"""
    return tile_shape_analysis(tile)[0]


def split_hand_tiles(
    img: Image.Image, max_tiles: int = 18
) -> list[tuple[int, int]] | None:
    """由「一手牌橫條」自動切出每張牌嘅左右邊界。

    純背景色偵測，完全離線、零成本、零 API。
    做法：四角估背景色 → 每 column 有幾多比例屬背景 → 自相關估牌距 →
    以明顯縫隙分組（摸牌同手牌之間通常有個大空隙）→ 每組平均切開。

    適用於雀魂／天鳳等固定畫面嘅手牌條；
    實體牌相片或完整遊戲截圖會回 None（交由「整條 AI 辨識」處理）。

    Returns:
        [(x0, x1), ...]（由左至右），或 None 代表切唔到。
    """
    try:
        a = np.asarray(img.convert("RGB")).astype(np.int16)
    except Exception:  # noqa: BLE001
        return None
    h, w, _ = a.shape
    if h < 24 or w < 80:
        return None

    # 1) 背景色（四角）
    k = max(2, min(h, w) // 40)
    corners = np.concatenate([
        a[:k, :k].reshape(-1, 3), a[:k, -k:].reshape(-1, 3),
        a[-k:, :k].reshape(-1, 3), a[-k:, -k:].reshape(-1, 3),
    ])
    bg = corners.mean(axis=0)

    # 2) 每個 column 屬背景嘅比例
    col_bg = (np.abs(a - bg).sum(axis=2) < 90).mean(axis=0)

    # 3) 非背景區段（每段可能係 1 張或幾張牌）
    gap = col_bg > 0.55
    segs: list[tuple[int, int]] = []
    start = None
    for i, g in enumerate(gap):
        if not g and start is None:
            start = i
        elif g and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(gap) - 1))
    segs = [(s, e) for s, e in segs if e - s + 1 >= 20]
    if not segs:
        return None

    widths = [e - s + 1 for s, e in segs]

    # 4) 估牌距。區段寬度係牌距嘅整數倍，所以可以用「最窄嘅區段」做候選；
    #    再加「由牌高推算」（麻雀牌大約 1 : 1.58）做第二候選，
    #    揀最能解釋所有區段寬度嘅嗰個。
    def fit_score(p: float) -> float:
        if p < 25:
            return 9e9
        err = 0.0
        for wd in widths:
            r = wd / p
            err += abs(r - round(r))
        return err / len(widths)

    candidates = [float(min(widths)), h / 1.58]
    pitch = min(candidates, key=fit_score)

    if pitch < 25 or pitch > 260:
        return None
    if fit_score(pitch) > 0.18:  # 解釋唔到 → 唔似整齊手牌條
        return None

    # 5) 每段平均切成 n 張
    boxes: list[tuple[int, int]] = []
    for s, e in segs:
        span = e - s + 1
        n = max(1, round(span / pitch))
        step = span / n
        if step < 25:
            return None
        for i in range(n):
            boxes.append((int(round(s + i * step)), int(round(s + (i + 1) * step))))

    # 6) 只接受合理手牌張數
    if not 2 <= len(boxes) <= max_tiles:
        return None
    return boxes


def crop_hand_region(img: Image.Image, bottom_pct: int = 100) -> Image.Image:
    """遊戲截圖入面，手牌通常喺畫面底部。

    bottom_pct < 100 嘅話淨係保留底部嗰個百分比，並且放大，
    令 AI 睇得清楚啲（截圖入面啲牌其實好細，唔裁剪好易睇唔到）。
    100 = 唔裁剪（相片／實體牌用呢個）。
    """
    if bottom_pct >= 100 or not img.height:
        return img

    # 圖片本身已經係一條好扁嘅手牌帶（例如 1324×143，比例 9.3），
    # 再裁只會切走牌，所以直接原圖。只有「完整遊戲截圖」先需要裁。
    if img.height and img.width / img.height > 4.0:
        return img

    top = max(0, int(img.height * (1 - bottom_pct / 100)))
    out = img.crop((0, top, img.width, img.height))

    # 放大到 ~1600 寬（上限 3 倍，避免太慢）
    if out.width < 1600:
        ratio = min(1600 / out.width, 3.0)
        out = out.resize(
            (int(out.width * ratio), int(out.height * ratio)), Image.LANCZOS
        )
    return out


def _extract_hand(raw: str) -> str:
    """由模型回覆中抽出最似手牌的一行。

    新版提示詞會先叫模型「逐張列出」，再輸出壓縮記法，所以：
      1. 先喺全段文字搵「壓縮記法」子字串（花色之間冇空格），
         取第一個正常化後係 13 或 14 張嘅。
      2. 再逐行睇，邊行有 13 或 14 張。
      3. 最後先做全段正常化（可能唔合法，交由 caller 驗證）。
    咁做可以避免「逐張列出」嗰行同「壓縮」嗰行被合併成一個 28 張嘅亂碼。
    """
    cleaned = raw.replace("`", " ").replace("：", ":")

    # 1. 全段搜尋壓縮記法子字串
    for m in re.finditer(r"\d+[mpsz](?:\d+[mpsz])*", cleaned):
        hand = normalize_hand(m.group())
        if hand and count_tiles(hand) in (13, 14):
            return hand

    # 2. 逐行
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    for line in lines:
        body = line.split(":", 1)[-1].strip()
        if re.fullmatch(r"\d+[mpsz](?:\d+[mpsz])*", body):
            hand = normalize_hand(body)
            if hand and count_tiles(hand) in (13, 14):
                return hand
    for line in lines:
        hand = normalize_hand(line)
        if hand and count_tiles(hand) in (13, 14):
            return hand

    # 3. 最後手段
    return normalize_hand(cleaned)


RETRY_PROMPT = """請由左至右逐張數清楚圖片中的手牌，唔要漏任何一張。

請按以下格式輸出（每張牌用空格分隔，一行寫完）：
1m 2m 3m 4p 5p 6p 7s 8s 9s 1z 1z 5z 6z 7z

之後再另起一行輸出壓縮記法，例如：123m456p789s11567z
記法說明：萬子 m、筒子 p、索子 s、字牌 z（1=東 2=南 3=西 4=北 5=白 6=發 7=中）
只輸出上述內容，唔要解釋。
"""


def _ollama_generate(img: Image.Image, prompt: str) -> tuple[str, str | None]:
    """呼叫 Ollama，回傳 (原始回覆, 錯誤訊息)。"""
    import requests

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        # 保持模型留在記憶體，避免每次影相都重新載入 6GB 權重
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_predict": 96,  # 手牌記法很短，限制輸出可大幅提速
        },
    }
    try:
        # CPU 推理 + 首次載入模型可能很久，俾足 10 分鐘
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        return (r.json().get("response") or "").strip(), None
    except Exception as exc:  # noqa: BLE001
        return "", f"Ollama 辨識失敗：{exc}"


def _recognize_ollama(img: Image.Image) -> tuple[str | None, str]:
    raw, err = _ollama_generate(img, PROMPT)
    if err:
        return None, err
    if not raw:
        return None, "模型沒有回應任何內容。"

    hand = _extract_hand(raw)
    if hand and count_tiles(hand) in (13, 14):
        return hand, f"辨識完成（{OLLAMA_MODEL}）"

    # 張數不對 → 用引導式 prompt 再數一次（只會喺失敗時多花時間）
    raw2, err2 = _ollama_generate(img, RETRY_PROMPT)
    hand2 = _extract_hand(raw2) if raw2 else ""

    for candidate in (hand2, hand):
        if candidate and count_tiles(candidate) in (13, 14):
            return candidate, f"辨識完成（{OLLAMA_MODEL}，第二次嘗試）"

    if hand2 or hand:
        best = hand2 or hand
        return best, _validate_hand(best) or f"辨識完成（{OLLAMA_MODEL}）"

    if "NONE" in (raw + raw2).upper():
        return None, "圖片中偵測不到手牌。"
    detail = err2 or f"模型輸出無法解析：{raw[:80]}"
    return None, detail


def _recognize_template(img: Image.Image) -> tuple[str | None, str]:
    try:
        import cv2
    except ImportError:
        return None, "未安裝 opencv-python，無法使用模板比對。"

    ok, _ = template_status()
    if not ok:
        return None, template_status()[1]

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    detections: list[tuple[float, int, int, int, int, str]] = []

    for label in TILE_LABELS:
        path = TEMPLATE_DIR / f"{label}.png"
        tpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        for scale in np.arange(0.6, 1.45, 0.1):
            tw, th = int(tpl.shape[1] * scale), int(tpl.shape[0] * scale)
            if tw < 12 or th < 12 or tw > gray.shape[1] or th > gray.shape[0]:
                continue
            res = cv2.matchTemplate(gray, cv2.resize(tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.80:
                detections.append((max_val, max_loc[0], max_loc[1], tw, th, label))

    # 貪心 NMS：分數高者優先，重疊過多者丟棄
    detections.sort(key=lambda d: d[0], reverse=True)
    kept: list[tuple[float, int, int, int, int, str]] = []
    for det in detections:
        _, x, y, w, h, _lbl = det
        cx, cy = x + w / 2, y + h / 2
        if all(
            abs(cx - (kx + kw / 2)) > min(w, kw) * 0.6
            or abs(cy - (ky + kh / 2)) > min(h, kh) * 0.6
            for _, kx, ky, kw, kh, _ in kept
        ):
            kept.append(det)

    if not kept:
        return None, "模板比對找不到任何牌，請確認模板尺寸與畫面一致。"

    kept.sort(key=lambda d: d[1])  # 由左至右
    buckets: dict[str, list[int]] = {"m": [], "p": [], "s": [], "z": []}
    for _score, _x, _y, _w, _h, lbl in kept:
        buckets[lbl[-1]].append(int(lbl[0]))
    hand = "".join(
        f"{''.join(str(n) for n in sorted(buckets[s]))}{s}"
        for s in "mpsz"
        if buckets[s]
    )
    warn = _validate_hand(hand)
    return hand, warn or f"模板比對完成，共 {len(kept)} 張牌"
