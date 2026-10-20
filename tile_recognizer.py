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
    for digits, suit in re.findall(r"(\d+)([mpsz])", hand):
        if len(digits) > 4:
            return f"{digits}{suit} 超過 4 張，建議手動檢查。"
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


def _extract_hand(raw: str) -> str:
    """由模型回覆中抽出最似手牌的一行。"""
    cleaned = raw.replace("`", " ")
    for line in cleaned.splitlines():
        hand = normalize_hand(line)
        if hand and count_tiles(hand) in (13, 14):
            return hand
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
