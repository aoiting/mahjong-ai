"""OpenRouter 客戶端：手牌視覺辨識 + 牌局分析。

用 `requests` 直接呼叫 OpenRouter 嘅 OpenAI 相容端點。

注意：呢部機嘅 `openai` SDK（經 httpx2 → truststore）喺 Python 3.14 會
觸發 `RecursionError: maximum recursion depth exceeded`，所以刻意唔用 SDK。

亦支援用本機 Ollama 做文字分析，方便同一個介面切換供應商。
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

import tile_recognizer as tr

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "meta/muse-spark-1.3-contributor")

# 免費模型係 reasoning 模型，思考會用掉 token，
# 所以 max_tokens 唔可以設得太細，否則 content 會係空。
# 700 太細：真實截圖會令模型思考好長，thinking 食晒預算就冇 content。
VISION_MAX_TOKENS = 4000
# 免費 reasoning 模型會先用大量 token 思考，上限太低會令答案被截斷
ANALYSIS_MAX_TOKENS = 1200
# 串流模式：解說目標只係 250–350 字，配合 reasoning effort
STREAM_MAX_TOKENS = 1200

# 視覺辨識備援模型（按實測可靠度排序；第一個掛咗就試下一個）
# 註：minimax-m3.1-flash:free 嘅免費期已於 2026-10-22 結束，唔可以再用。
VISION_FALLBACK_MODELS = [
    "meta/muse-spark-1.3-contributor",
    "meta/muse-spark-open",
    "meta/muse-glimmer-30b",
]

# 分析備援模型：優先留在 Muse 家族（1.3-contributor 只有 Meta 一個 provider，
# 部分地區／熄 VPN 後會 403），最後才用免費 Nemotron。
ANALYSIS_FALLBACK_MODELS = [
    "meta/muse-spark-open",
    "meta/muse-glimmer-30b",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

VISION_PROMPT = """你係日本麻雀（立直麻雀）牌面辨識器。請辨識圖片中「手牌」嘅 13 或 14 張牌。

【牌面辨識規則】
1. 萬子（m）：上面係漢數字（一二三四五六七八九），下面係「萬」字。
2. 筒子（p）：數圓形圖案數量。
   - 1筒：一個特大嘅花形圓圈
   - 2～9筒：2 至 9 個細圓圈，按固定排列（5筒排成 X、3筒斜線）
3. 索子（s）：數竹節數量。竹枝可以斜放；八索係 4 直 4 斜嘅編織形，斜竹都要計。
   - 1索：一隻雀鳥圖案
   - 2～9索：2 至 9 支垂直竹枝
4. 風牌：一個大字 —— 東、南、西、北。
5. 三元牌：紅「中」、綠「發」、白（空白或有藍／黑外框）。
6. 赤牌：成隻 5 嘅圖案全部係紅色 → 用 5mr／5pr／5sr 標示；只有中間一個圖案係紅色（五筒中間紅圈、五索中間紅竹）→ 係普通 5，照寫 5p／5s。
7. 只數「自己嘅手牌」，唔要數牌河、其他家、或者 UI 上面嘅圖示。

【輸出格式】
第一行：由左至右逐張列出（用空格分隔）
最後一行：壓縮記法

格式示範（只示格式，唔可以照抄，亦唔係答案）：
逐張：3m 3m 7p
壓縮：33m7p

【壓縮記法規則】
- 萬子 m、筒子 p、索子 s、字牌 z
- 數字在前、花色在後，依 m → p → s → z 分組，同組內由細到大
- 字牌 z：1=東 2=南 3=西 4=北 5=白 6=發 7=中
- 壓縮記法唔可以有空格
- 唔要解釋、唔要 Markdown、唔要程式碼區塊
- 如果圖片完全冇手牌，只輸出 NONE

【最重要】
- 上面嘅格式示範只係教格式，唔係答案。嚴禁照抄。
- 你必須根據圖片真實內容逐張讀出，張數要同圖片一致（通常 13 或 14 張）。
"""


class AIError(RuntimeError):
    """API 呼叫失敗（含友善訊息）。"""


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter 建議帶上，方便佢做用量統計／排名
        "HTTP-Referer": "https://localhost/mahjong-ai",
        "X-Title": "Mahjong AI Analyzer",
    }


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _friendly_error(status: int | None, detail: str = "") -> str:
    """把 HTTP 錯誤轉成人話。"""
    d = (detail or "").lower()
    if status == 402 or "prompt tokens limit" in d or "requires more credits" in d:
        return (
            "OpenRouter 額度不足：你帳戶餘額係 $0，可負擔嘅請求大小會隨用量越縮越細，"
            "最後連細請求都會俾人擋。請去 openrouter.ai/settings/credits 入錢"
            "（最低 US$5 就足夠用很久）。"
        )
    if "guardrail" in d or "training violation" in d:
        return (
            "呢個模型嘅供應商要求「允許用付費請求做訓練」，但你 OpenRouter 帳戶嘅"
            "私隱設定封鎖咗。請去 openrouter.ai/settings/privacy 開啟相關選項，"
            "或者改用其他視覺模型。"
        )
    mapping = {
        400: "請求格式有誤（模型唔支援某個參數）。",
        401: "API Key 無效或已停用，請重新確認 OpenRouter key。",
        402: "OpenRouter 餘額不足，請入錢。",
        403: "存取被拒：可能係 spend limit、key 被停用、或地區限制（例如 Gemini 唔支援香港）。",
        404: "搵唔到模型或暫時無供應商可用，請檢查模型 ID。",
        429: "請求太頻繁（免費模型有速率／每日限制），請等一陣再試。",
        502: "上游模型暫時失敗（唔會扣費），請重試。",
    }
    base = mapping.get(status or 0, f"API 呼叫失敗（HTTP {status}）。")
    detail = (detail or "").strip().replace("\n", " ")[:200]
    return f"{base} {detail}".strip()


def _short_reason(msg: str, limit: int = 40) -> str:
    """把長長嘅錯誤訊息縮成一句人話（用喺「已轉用備援」提示）。"""
    t = (msg or "").strip().replace("\n", " ")
    low = t.lower()
    if "地區" in t or "region" in low:
        return "地區限制"
    if "餘額" in t or "credits" in low or "402" in t:
        return "額度不足"
    if "逾時" in t:
        return "請求逾時"
    if "502" in t:
        return "上游暫時失敗"
    if "429" in t or "頻繁" in t:
        return "請求太頻繁"
    return t[:limit]


def _post(path: str, api_key: str, payload: dict, timeout: float) -> dict:
    """POST 到 OpenRouter，回傳 JSON；失敗拋 AIError。"""
    if not api_key or not api_key.strip():
        raise AIError("未提供 OpenRouter API Key。")

    url = f"{BASE_URL}{path}"
    try:
        r = requests.post(url, headers=_headers(api_key.strip()), json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise AIError(f"請求逾時（{timeout:.0f} 秒），免費模型繁忙時可能較慢，請重試。") from exc
    except requests.RequestException as exc:
        raise AIError(f"網絡錯誤：{exc}") from exc

    if r.status_code != 200:
        detail = ""
        try:
            err = r.json().get("error") or {}
            detail = err.get("message", "")
        except Exception:  # noqa: BLE001
            detail = r.text[:200]
        raise AIError(_friendly_error(r.status_code, detail))

    try:
        return r.json()
    except ValueError as exc:
        raise AIError("伺服器回傳非 JSON 格式。") from exc


def _iter_stream(resp) -> Iterator[str]:
    """由 OpenRouter SSE 回應逐段抽出文字。

    注意：`iter_lines(decode_unicode=True)` 會用 response.encoding
    （OpenRouter 嘅 event-stream 冇 charset，會當 ISO-8859-1），
    令中文變亂碼。所以呢度自己用 UTF-8 解碼 bytes。
    """
    for raw in resp.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        if not data:
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        for choice in obj.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                yield piece


def _open_stream(api_key: str, payload: dict, timeout: float):
    """開始串流請求；非 200 直接拋 AIError。"""
    if not api_key or not api_key.strip():
        raise AIError("未提供 OpenRouter API Key。")
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=_headers(api_key.strip()),
            json=payload,
            stream=True,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise AIError(f"請求逾時（{timeout:.0f} 秒），請重試。") from exc
    except requests.RequestException as exc:
        raise AIError(f"網絡錯誤：{exc}") from exc

    if resp.status_code != 200:
        detail = ""
        try:
            detail = (resp.json().get("error") or {}).get("message", "")
        except Exception:  # noqa: BLE001
            detail = resp.text[:200]
        resp.close()
        raise AIError(_friendly_error(resp.status_code, detail))
    return resp


def _stream_effort(model: str) -> str:
    """串流用嘅 reasoning effort。

    實測：`meta/muse-spark-1.*` 係「強制 reasoning」模型，effort=low 會用盡
    800 tokens 思考而回傳 0 字；effort=minimal 先可以喺 800 內出到內容。
    其他模型照 brief 用 low。
    """
    return "minimal" if "muse-spark-1." in model else "low"


def stream_analysis(
    hand: str,
    ctx: dict,
    backend: str = "openrouter",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    facts: str = "",
    model_preds: str = "",
    direction_hints: str = "無",
    fallback_models: list[str] | None = None,
    status: dict | None = None,
) -> Iterator[str]:
    """逐段 yield 分析文字（OpenRouter 串流）。

    status：可選 dict，完成後會寫入 "model" 同 "primary_error"。
    如果未 yield 任何字就失敗，會拋 AIError，交由 app fallback 去非串流。
    """
    status = status if status is not None else {}
    if backend == "ollama":
        raise AIError("本機 Ollama 模式唔支援串流。")

    prompt = build_analysis_prompt(hand, ctx, facts, model_preds, direction_hints)

    candidates = [model]
    for alt in (fallback_models if fallback_models is not None else ANALYSIS_FALLBACK_MODELS):
        if alt and alt not in candidates:
            candidates.append(alt)

    last_err = ""
    primary_err = ""
    for m in candidates:
        base = {
            "model": m,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": STREAM_MAX_TOKENS,
            "temperature": 0.3,
            "stream": True,
        }
        for use_reasoning in (True, False):
            payload = dict(base)
            if use_reasoning:
                payload["reasoning"] = {"effort": _stream_effort(m)}
            try:
                resp = _open_stream(api_key, payload, timeout=300)
            except AIError as exc:
                last_err = str(exc)
                if m == candidates[0] and not primary_err:
                    primary_err = last_err
                if use_reasoning and ("400" in last_err or "格式" in last_err):
                    continue
                break

            emitted = 0
            try:
                for piece in _iter_stream(resp):
                    emitted += 1
                    yield piece
            finally:
                resp.close()

            if emitted:
                status["model"] = m
                status["primary_error"] = primary_err
                return

            last_err = "模型回傳空內容。"
            if m == candidates[0] and not primary_err:
                primary_err = last_err
            break

    raise AIError(last_err or "分析失敗，請稍後再試。")


def _message_text(data: dict) -> str:
    try:
        return (data["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def _message_reasoning(data: dict) -> str:
    """reasoning 模型會將答案「計」喺 reasoning field，未必寫入 content。"""
    try:
        return (data["choices"][0]["message"].get("reasoning") or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def _empty_reason(data: dict) -> str:
    """解釋點解 content 係空 —— 俾 UI 顯示真正原因，唔好淨係講「請重試」。"""
    choices = data.get("choices") or []
    if not choices:
        return "上游供應商冇回傳任何候選結果（免費模型暫時繁忙，或被路由到失效 provider）。"
    ch = choices[0] or {}
    msg = ch.get("message") or {}
    refusal = (msg.get("refusal") or "").strip()
    if refusal:
        return f"模型拒絕處理呢張圖片：{refusal[:120]}"
    finish = ch.get("finish_reason") or ""
    if finish == "length":
        return "輸出被 token 上限截斷（模型用晒預算去思考，冇留俾答案）。"
    if msg.get("reasoning"):
        return "模型只回傳咗思考過程，冇正式答案。"
    return "模型回傳咗空內容。"


def _rescue_from_reasoning(data: dict) -> str | None:
    """content 空咗，但答案可能已經喺 reasoning 度 —— 試下救返。"""
    text = _message_reasoning(data)
    if not text:
        return None
    hand = tr._extract_hand(text)
    if hand and tr.count_tiles(hand) in (13, 14):
        return hand
    return None


# ── 逐張辨識 ────────────────────────────────────────────────
# 實測：整條 14 張一齊睇，模型會亂噏而且每次唔同；
# 切開逐張放大之後問，準確度大幅提升（同一隻模型都係）。
PER_TILE_PROMPT = """呢張圖係一張日本麻雀牌（單張）。
請只輸出一個代碼，唔要任何解釋、標點或多餘文字：
- 萬子：1m～9m（有紅色「萬」字，上面有漢數字）
- 筒子：1p～9p（圓圈；1p 係一個大花圈）
- 索子：1s～9s（綠色竹枝（可以斜放）；1s 係綠色雀鳥）
- 字牌：1z=東 2z=南 3z=西 4z=北 5z=白 6z=發 7z=中
- 如果張牌完全空白、冇任何圖案 → 答 5z
- 如果成隻 5 嘅圖案全部係紅色，答 5mr／5pr／5sr（赤牌）；如果只有中間一個圖案係紅色（五筒中間一粒紅圈、五索中間一支紅竹），嗰個係普通 5，照答 5p／5s，唔好當赤牌。

【數牌規則】筒子數圈、索子數竹，要逐個數，斜放嘅都要計，唔好靠排版估。
特別注意：八索係編織形 — 4 支直竹＋4 支斜竹織埋一齊，斜嗰啲唔係花紋，總數係 8；
五索冇編織，中間嗰支多數紅色或橫放；七筒係 3 個黑圈斜排＋4 個紅圈；
九筒／九索係 3×3 共 9 個。

格式示範（唔好照抄，只示格式）：3p"""


def _parse_single_tile(text: str) -> str | None:
    """由模型回覆抽出單張牌代碼（保留赤五 r 後綴）。"""
    if not text:
        return None
    m = re.search(r"([0-9])([mpsz])(r?)", text.lower())
    if not m:
        return None
    digit, suit, red = m.group(1), m.group(2), m.group(3)
    if digit == "0":
        digit, red = "5", "r"
    if red and (suit == "z" or digit != "5"):
        red = ""
    return f"{digit}{suit}{red}"


def _post_with_reasoning(
    path: str, api_key: str, payload: dict, timeout: float
) -> dict:
    """先試帶 reasoning effort=low（慳 token），400 就唔帶再試。"""
    import copy

    first = copy.deepcopy(payload)
    first["reasoning"] = {"effort": "low"}
    try:
        return _post(path, api_key, first, timeout)
    except AIError as exc:
        if "400" not in str(exc) and "格式" not in str(exc):
            raise
    return _post(path, api_key, payload, timeout)


# 餘額不足嘅錯誤特徵（OpenRouter 會按 max_tokens 預留額度）
_CREDIT_HINTS = ("餘額不足", "credits", "max_tokens", "afford", "402")


def _post_smart(
    path: str,
    api_key: str,
    payload: dict,
    timeout: float,
    budgets: tuple[int, ...] = (1000, 700, 400, 200),
    effort: str = "low",
) -> dict:
    """強韌版呼叫：

    1. 先試帶 reasoning effort=low（慳 token）
    2. 如果模型唔支援 `reasoning` 參數（HTTP 400）→ 唔帶再試
    3. 如果「餘額不足」（OpenRouter 按 max_tokens 預留額度）→ 自動降低預算再試

    咁樣就算帳戶餘額係 $0 都用得到。
    """
    import copy

    cands = []
    base = payload.get("max_tokens")
    for b in (base, *budgets):
        if b and int(b) not in cands:
            cands.append(int(b))

    last_exc: AIError | None = None
    for budget in cands:
        for use_reasoning in (True, False):
            body = copy.deepcopy(payload)
            body["max_tokens"] = budget
            if use_reasoning:
                body["reasoning"] = {"effort": effort}
            try:
                return _post(path, api_key, body, timeout)
            except AIError as exc:
                msg = str(exc)
                last_exc = exc
                if any(h in msg for h in _CREDIT_HINTS):
                    break  # 呢個預算唔夠 → 試更低
                if use_reasoning and ("400" in msg or "格式" in msg):
                    continue  # 唔支援 reasoning → 唔帶再試
                raise
    if last_exc:
        raise last_exc
    raise AIError("呼叫失敗。")


GATE_PROMPT = """呢張相片或者遊戲截圖入面，有冇麻雀牌（手牌或者副露都計）？
只准回答 YES 或 NO，唔要解釋、標點或者任何其他字。"""


def gate_check(
    image: Image.Image,
    api_key: str,
    model: str,
    fallback_models: list[str] | None = None,
) -> tuple[bool | None, str]:
    """守門：回傳 True=YES、False=NO、None=API 失敗（fail-open）。"""
    img = image.convert("RGB")
    if img.width > 1600:
        ratio = 1600 / img.width
        img = img.resize((1600, max(1, int(img.height * ratio))), Image.LANCZOS)

    models = [model]
    for alt in (fallback_models if fallback_models is not None else VISION_FALLBACK_MODELS):
        if alt and alt not in models:
            models.append(alt)

    last_err = ""
    for m in models:
        try:
            data = _post_smart(
                "/chat/completions",
                api_key,
                {
                    "model": m,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": GATE_PROMPT},
                                {"type": "image_url", "image_url": {"url": _data_uri(img)}},
                            ],
                        }
                    ],
                    # Reasoning models need room before they emit YES/NO.
                    "max_tokens": 400,
                    "temperature": 0,
                },
                timeout=60,
                budgets=(),
                effort="minimal",
            )
        except AIError as exc:
            last_err = str(exc)
            continue

        text = (_message_text(data) or "").strip().upper()
        if text == "YES":
            return True, "YES"
        if text == "NO":
            return False, "NO"
        if text:
            # Strict parse: a non-YES/NO answer is a NO; do not guess.
            return False, f"unparseable: {text[:80]}"
        last_err = _empty_reason(data)

    return None, last_err or "守門模型沒有回應。"


def recognize_tile(
    tile: Image.Image,
    api_key: str,
    model: str,
    hint: str = "",
) -> tuple[str | None, str]:
    """辨識單張牌。回傳 (代碼, 訊息)。

    Args:
        hint: 本地色彩分析嘅提示，會加入提示詞（唔會取代模型判斷）。
    """
    img = tile.convert("RGB")
    scale = 1
    while img.width * scale < 300 and scale < 6:
        scale += 1
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)

    prompt = PER_TILE_PROMPT + (f"\n\n{hint}" if hint else "")

    try:
        data = _post_smart(
            "/chat/completions",
            api_key,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _data_uri(img)}},
                        ],
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0,
            },
            timeout=240,
            effort=_stream_effort(model),
        )
    except AIError as exc:
        return None, str(exc)

    text = _message_text(data) or _message_reasoning(data)
    code = _parse_single_tile(text)
    if code:
        return code, "ok"
    return None, _empty_reason(data)


def local_hint(profile: dict, shape_text: str = "") -> str:
    """由本地分析生成提示，糾正 AI 最常見嘅錯誤。

    形狀（窄長條 vs 圓形）比顏色可靠 ——
    呢個牌庫嘅 5s 係紅色竹枝，所以唔可以用「冇綠色」否定索子；
    但「發」(6z) 喺呢個牌庫一定係綠色（實測 0.187），所以冇綠色就一定唔係發。

    全紅赤五（5mr／5pr／5sr）嘅裝飾紅點會被形狀分析多數一個圓形，
    所以紅色比例高時，唔傳圈數提示，避免 AI 被「6 個圓形」誤導。
    """
    parts = []
    high_red = profile.get("red", 0.0) >= 0.15 and profile.get("green", 0.0) < tr.GREEN_MIN
    if high_red:
        return (
            "本地色彩分析：呢張牌幾乎全紅、冇黑墨，極可能係赤五（5mr／5pr／5sr），"
            "唔係普通五。赤五筒＝4 個紅圈＋中間紅點，總數 5；赤五索＝紅色竹枝。"
            "請逐個數清楚。"
        )
    if shape_text:
        parts.append(shape_text)
    if profile["green"] >= tr.GREEN_MIN:
        parts.append("本地色彩分析：呢張牌有明顯綠色（綠色只出現喺索子或「發」）。")
    else:
        parts.append("本地色彩分析：呢張牌冇綠色，所以唔可能係「發」(6z)。")
    return " ".join(parts)


MAN_NUMERAL_PROMPT = """呢張圖係日本麻雀「萬子」牌嘅上半部分，上面印有一個漢數字。
請只答一個字：一 二 三 四 五 六 七 八 九
唔要解釋，只答一個字。"""

_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9}


def _read_man_numeral(tile: Image.Image, api_key: str, model: str) -> int | None:
    """萬子專用：放大上半部分，集中讀漢數字。

    實測：整張牌交俾模型讀數字好易錯（4m 讀成 1m），
    但單獨放大上半部分讀漢字就準得多。
    """
    top = tile.crop((0, 0, tile.width, max(8, int(tile.height * 0.5))))
    top = top.resize((top.width * 6, top.height * 6), Image.LANCZOS)
    try:
        data = _post_smart(
            "/chat/completions",
            api_key,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": MAN_NUMERAL_PROMPT},
                            {"type": "image_url", "image_url": {"url": _data_uri(top)}},
                        ],
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0,
            },
            timeout=240,
            effort=_stream_effort(model),
        )
    except AIError:
        return None

    text = _message_text(data) or _message_reasoning(data)
    for ch, n in _CN_NUM.items():
        if ch in (text or ""):
            return n
    m = re.search(r"\b([1-9])\b", text or "")
    return int(m.group(1)) if m else None


def _model_unusable(msg: str) -> bool:
    """判斷錯誤係咪「呢個模型完全用唔到」（而唔係一次過嘅 502）。

    呢類錯誤重試 3 次都只係浪費時間，所以直接跳去下一個備援模型。
    """
    t = (msg or "").lower()
    return any(
        k in t
        for k in (
            "guardrail",
            "training violation",
            "0 endpoints",
            "私隱設定封鎖",
            "存取被拒",
            "地區限制",
            "region",
            "搵唔到模型",
            "無供應商",
            "api key 無效",
            "餘額不足",
        )
    )


def _sorted_hand_tokens(tiles: list[str | None]) -> list[str]:
    """按 m→p→s→z、數字穩定排序；赤五 r 唔參與 key。"""
    order = {"m": 0, "p": 1, "s": 2, "z": 3}

    def key(t: str) -> tuple[int, int]:
        suit = t[-2] if t.endswith("r") else t[-1]
        return (order.get(suit, 9), int(t[0]))

    return sorted((t for t in tiles if t), key=key)


def recognize_hand_per_tile(
    image: Image.Image,
    api_key: str,
    boxes: list[tuple[int, int]],
    model: str,
    fallback_models: list[str] | None = None,
    progress=None,
    local_first: bool = False,
    status: dict | None = None,
) -> tuple[str | None, str]:
    """先切牌、再逐張辨識。

    Args:
        boxes: [(x0, x1), ...]，由 tr.split_hand_tiles() 得出。
        progress: 可選 callback(done, total, tiles)。
        local_first: 保留參數以兼容舊呼叫；現時唔會再直接採用本地形狀分析答案，
            形狀分析只會作為提示傳俾 AI。唯一例外：空白牌 ink 判定直接當 5z。
        status: 可選 dict，完成後會寫入 "drawn"（最後一個 box 認到嘅牌）。

    Returns:
        (hand, message)
    """
    models = [model]
    for alt in (fallback_models if fallback_models is not None else VISION_FALLBACK_MODELS):
        if alt and alt not in models:
            models.append(alt)

    total = len(boxes)
    tiles: list[str | None] = [None] * total
    dead: set[str] = set()
    dead_lock = threading.Lock()
    completed = 0

    # 免費模型有嚴格速率限制（免費層約 20 次/分鐘），延遲只用喺萬子二次讀數字；
    # 逐張請求改為並行（最多 5 隻）。
    primary = models[0] if models else ""
    delay = float(
        os.environ.get("MAHJONG_PER_TILE_DELAY")
        or (3.0 if primary.endswith(":free") else 0.5)
    )

    pending: list[tuple[int, int, int, dict, str]] = []
    for idx, (x0, x1) in enumerate(boxes):
        crop = image.crop((x0, 0, x1, image.height))
        profile = tr.tile_color_profile(crop)

        # 本地快速通道：空白牌一定係「白」（5z），唔使入隊問 AI
        if profile["ink"] < tr.BLANK_INK_MAX:
            tiles[idx] = "5z"
            completed += 1
            if progress:
                progress(completed, total, tiles)
            continue

        # 形狀分析只做提示，唔會再直接當答案（避免本地分析搶答難牌）
        hint_text, _local_answer = tr.tile_shape_analysis(crop)
        pending.append((idx, x0, x1, profile, hint_text))

    def _work(item: tuple[int, int, int, dict, str]) -> tuple[int, str | None, str]:
        idx, x0, x1, profile, hint_text = item
        crop = image.crop((x0, 0, x1, image.height))
        hint = local_hint(profile, hint_text)
        code: str | None = None
        last = ""
        used_model = ""
        # 主模型試 3 次（上游常有暫時性 502），失敗先逐個備援模型試 1 次。
        # 如果某模型係「完全用唔到」（例如被 guardrail 封鎖），即刻標記為 dead，
        # 後面嘅牌就唔會再浪費時間重試。
        attempts = [(models[0], 3)] + [(m, 1) for m in models[1:]]
        for m, tries in attempts:
            with dead_lock:
                is_dead = m in dead
            if is_dead:
                continue
            for attempt in range(tries):
                code, msg = recognize_tile(crop, api_key, m, hint=hint)
                if code:
                    used_model = m
                    break
                last = msg
                if _model_unusable(msg):
                    with dead_lock:
                        dead.add(m)
                    break
                time.sleep(1.5 * (attempt + 1))  # 退避重試
            if code:
                break

        # 萬子嘅漢數字最易讀錯 → 單獨放大上半部分再讀一次
        if code and code.endswith("m"):
            time.sleep(min(delay, 1.5))
            n = _read_man_numeral(crop, api_key, used_model or models[0])
            if n:
                code = f"{n}m"

        return idx, code, last

    error_map: dict[int, str] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=min(5, len(pending))) as pool:
            futures = [pool.submit(_work, item) for item in pending]
            for fut in as_completed(futures):
                idx, code, last = fut.result()
                tiles[idx] = code
                completed += 1
                if not code:
                    error_map[idx] = last
                if progress:
                    progress(completed, total, tiles)

    if status is not None:
        status["drawn"] = tiles[-1] if tiles else None

    errors = [f"第 {i + 1} 張：{error_map[i]}" for i in sorted(error_map)]
    good = [t for t in tiles if t]
    if len(good) < total * 0.7:
        detail = errors[0] if errors else "全部失敗"
        return None, f"逐張辨識失敗（只有 {len(good)}/{total} 張成功）。{detail}"

    hand = " ".join(_sorted_hand_tokens(tiles))
    breakdown = " ".join(t or "?" for t in tiles)
    note = f"逐張辨識完成（{len(good)}/{total} 張）：{breakdown}"
    if errors:
        note += f"　⚠️ {len(errors)} 張要用備援／失敗，建議人手核對"
    est = tr.estimate_tile_count(boxes)
    if est and est != total:
        note += (
            f"\n⚠️ 切出 {total} 格，但幅圖估計有 {est} 隻牌，"
            "可能切漏或切多，請逐隻核對。"
        )
    return hand, note


def _crop_box(image: Image.Image, box) -> Image.Image:
    if len(box) == 4:
        x0, x1, y0, y1 = box
    else:
        x0, x1 = box
        y0, y1 = 0, image.height
    return image.crop((x0, y0, x1, y1))


def _is_wide_box(box) -> bool:
    return len(box) == 4 and (box[1] - box[0]) > (box[3] - box[2])


def group_meld_tiles(codes: list[str]) -> list[list[str]]:
    """副露牌順序分組：每 3 隻一組；多出 1 隻併入前一組做槓（4 隻）。"""
    items = [c for c in codes if c]
    groups: list[list[str]] = []
    i = 0
    while i < len(items):
        remain = len(items) - i
        if remain == 1 and groups:
            groups[-1].append(items[i])
            i += 1
        elif remain == 4:
            groups.append(items[i:i + 4])
            i += 4
        else:
            groups.append(items[i:i + 3])
            i += 3
    return groups


def recognize_boxes(
    image: Image.Image,
    api_key: str,
    boxes: list,
    model: str,
    fallback_models: list[str] | None = None,
    progress=None,
    rotate_wide: bool = True,
) -> tuple[list[str], str]:
    """按 box 次序辨識（支援 4-tuple，闊過高會旋轉 90 度）。"""
    models = [model]
    for alt in (fallback_models if fallback_models is not None else VISION_FALLBACK_MODELS):
        if alt and alt not in models:
            models.append(alt)

    total = len(boxes)
    results: list[str | None] = [None] * total
    dead: set[str] = set()
    dead_lock = threading.Lock()
    completed = 0

    pending = []
    for idx, box in enumerate(boxes):
        crop = _crop_box(image, box)
        if tr.tile_color_profile(crop)["ink"] < tr.BLANK_INK_MAX:
            results[idx] = "5z"
            completed += 1
            if progress:
                progress(completed, total, results)
            continue
        pending.append((idx, box, crop))

    def _work(item):
        idx, box, crop = item
        wide = rotate_wide and _is_wide_box(box)
        variants = [crop]
        if wide:
            variants = [crop.transpose(Image.ROTATE_90), crop.transpose(Image.ROTATE_270)]
        hint_text, _ = tr.tile_shape_analysis(crop)
        hint = local_hint(tr.tile_color_profile(crop), hint_text)
        code = None
        last = ""
        used_model = ""
        attempts = [(models[0], 3)] + [(m, 1) for m in models[1:]]
        for variant in variants:
            for m, tries in attempts:
                with dead_lock:
                    is_dead = m in dead
                if is_dead:
                    continue
                for attempt in range(tries):
                    code, msg = recognize_tile(variant, api_key, m, hint=hint)
                    if code:
                        used_model = m
                        break
                    last = msg
                    if _model_unusable(msg):
                        with dead_lock:
                            dead.add(m)
                        break
                    time.sleep(1.5 * (attempt + 1))
                if code:
                    break
            if code:
                break
        if code and code.endswith("m"):
            n = _read_man_numeral(variants[0], api_key, used_model or models[0])
            if n:
                code = f"{n}m"
        return idx, code, last

    if pending:
        with ThreadPoolExecutor(max_workers=min(5, len(pending))) as pool:
            futs = [pool.submit(_work, item) for item in pending]
            for fut in as_completed(futs):
                idx, code, last = fut.result()
                results[idx] = code
                completed += 1
                if progress:
                    progress(completed, total, results)

    good = [c for c in results if c]
    breakdown = " ".join(c or "?" for c in results)
    return [c for c in results if c], f"（{len(good)}/{total}）{breakdown}"


def check_key(api_key: str, model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    """快速驗證 API key 同模型是否可用（會發一個極細請求）。"""
    try:
        data = _post(
            "/chat/completions",
            api_key,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 200,
                "temperature": 0,
            },
            timeout=45,
        )
    except AIError as exc:
        return False, str(exc)

    text = _message_text(data)
    if text:
        return True, f"連線正常（{model}，回覆：{text[:30]}）"
    return True, f"連線成功（{model}），但模型無回覆內容。"


def recognize_hand(
    image: Image.Image,
    api_key: str,
    model: str = DEFAULT_MODEL,
    fallback_models: list[str] | None = None,
) -> tuple[str | None, str]:
    """用 OpenRouter 視覺模型辨識手牌。

    會自動重試同切換備援模型，因為免費模型好容易：
      - 供應商繁忙（HTTP 200 但 choices 係空）
      - 思考食晒 token（content 空，但答案喺 reasoning 度）
      - 速率限制（429）

    Returns:
        (hand, message)；hand 為標準記法或 None。
    """
    try:
        img = image.convert("RGB")
        if img.width > 1600:  # 控制上傳大小與 token 數
            ratio = 1600 / img.width
            img = img.resize((1600, max(1, int(img.height * ratio))))
    except Exception as exc:  # noqa: BLE001
        return None, f"圖片處理失敗：{exc}"

    candidates = [model]
    for alt in (fallback_models if fallback_models is not None else VISION_FALLBACK_MODELS):
        if alt and alt not in candidates:
            candidates.append(alt)

    last_err = ""
    for m in candidates:
        try:
            data = _post_smart(
                "/chat/completions",
                api_key,
                {
                    "model": m,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": VISION_PROMPT},
                                {"type": "image_url", "image_url": {"url": _data_uri(img)}},
                            ],
                        }
                    ],
                    "max_tokens": VISION_MAX_TOKENS,
                    "temperature": 0,
                },
                timeout=240,
                effort=_stream_effort(m),
            )
        except AIError as exc:
            last_err = str(exc)
            continue

        content = _message_text(data)

        if not content:
            # content 空：先試下喺 reasoning 度救返答案
            rescued = _rescue_from_reasoning(data)
            if rescued:
                warn = tr._validate_hand(rescued)
                note = f"辨識完成（{m}，答案取自模型思考過程）"
                if warn:
                    return rescued, f"{note}。{warn}"
                return rescued, note
            last_err = _empty_reason(data)
            continue

        if "NONE" in content.upper():
            return None, "圖片中偵測不到手牌。"

        hand = tr._extract_hand(content)
        if not hand:
            last_err = f"模型輸出無法解析：{content[:100]}"
            continue

        warn = tr._validate_hand(hand)
        tokens = (data.get("usage") or {}).get("total_tokens")
        suffix = f"，{tokens} tokens" if tokens else ""
        msg = f"辨識完成（{m}{suffix}）"
        if warn:
            return hand, f"{msg}。{warn}"
        return hand, msg

    return None, last_err or "所有視覺模型都失敗，請稍後再試。"


def build_analysis_prompt(
    hand: str,
    ctx: dict,
    facts: str = "",
    model_preds: str = "",
    direction_hints: str = "無",
) -> str:
    """組裝牌局分析提示詞（Part 1：模型主導＋役種方向）。"""
    riichi_alert = "有" if ctx.get("riichi") else "冇"
    dora_text = ctx.get("dora") or "無"
    meld_count = int(ctx.get("meld_count", 0) or 0)
    facts_text = (facts or "").strip() or "（未提供）"
    hints_text = (direction_hints or "").strip() or "無"

    if model_preds and model_preds.strip():
        model_block = model_preds.strip()
        model_note = ""
    else:
        model_block = "無（模型未載入）"
        model_note = (
            "\n（今次冇高手模型建議：首選改為跟牌理分析結果講，"
            "並喺解說入面講明冇模型建議。）"
        )

    return f"""你係日本麻雀解說員。你嘅工作唔係自己決定打邊隻牌，而係將
「高手模型嘅建議」同「牌理分析嘅數字」用廣東話解釋俾用家聽。


手牌：{hand}
局勢：場風 {ctx.get("bakaze", "東")}、自風 {ctx.get("jikaze", "東")}、第 {ctx.get("junme", 1)} 巡、寶牌指示 {dora_text}、
副露 {meld_count} 組、他家立直：{riichi_alert}
牌理分析結果：{facts_text}
（向聽數、進張、聽牌、已確定役牌、和牌預覽 — 一律以呢度為準）
高手模型建議（按概率排）：{model_block}{model_note}
潛在方向提示：{hints_text}
（例如三元牌進度、染手傾向、對子數；冇就填「無」）


1. 以模型第一位做首選，解釋點解打佢：孤張、進張闊窄、寶牌、
安全度 — 要對住實際手牌講，唔好空泛。
2. 簡述次選係咩情況下嘅合理選擇。
3. 講明潛在役種方向：只可以講「潛在方向提示」有列出嘅方向，同埋指出要保留嘅
關鍵牌。如果提示係「無明顯役種方向」，最多用一句總括（例如「暫時無明顯役種方向」），
唔准逐個點名排除冇出現嘅役種（例如「無三元牌、做唔到七對子、混一色唔使諗」）。
如果提示有部分材料但未成形，可以一句帶過（例如「三元牌得一對，唔值得特登追」）。
4. 局勢有風險（例如他家立直）先提防守注意；冇就唔使講。


- 唔可以改動牌理數字（向聽、番數、點數、進張），唔可以自己另計。
- 解說入面提到向聽／進張嗰套數字嗰陣，一律叫「牌理分析」，唔准用「精算」呢個詞。
- 唔可以推翻模型首選、另推一隻做第一建議；有唔同意見只可以
寫成「補充睇法」並講明分別。
- 唔可以將有組合關係嘅牌當孤張處理（例如三元牌嘅刻子／對子
夥伴、順子邊張）。
- 唔肯定嘅嘢要講明唔肯定，唔好作數據。
- 如果「牌理分析結果」顯示已經聽牌（有待牌），或者事實列明切完即聽：
  🎯 首選段最後必須加一句「立直判斷：立直／默聽＋理由」。
- 事實入面有立直率就一定要引用嗰個數字（例如「高手喺呢種等牌有約 71% 會即刻立直」），
  再結合巡目（13 巡起屬晚巡）同有冇役（有役／無役修飾）俾明確取向；
  冇立直率就唔准自己作百分比。
- 唔係聽牌局面就唔准出「立直判斷」。
- 役種名一律用中文：平和（唔准寫 Pinfu）、斷么九（唔好寫 Tanyao）、混一色、
  七對子、一氣通貫、三色同順、小三元、大三元、役牌；唔准用任何羅馬拼音役名。


繁體中文、廣東話口語（自然，唔好太俗）；術語用標準名。
總長 250–350 字，固定三段：
🎯 首選：…
🔀 次選與方向：…
⚠️ 注意：…（冇風險就成段省略）"""


def analyze_hand(
    hand: str,
    ctx: dict,
    backend: str = "openrouter",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    facts: str = "",
    model_preds: str = "",
    direction_hints: str = "無",
    fallback_models: list[str] | None = None,
) -> tuple[str | None, str]:
    """分析牌局。

    Args:
        backend: "openrouter" 或 "ollama"
        facts: 本地精算結果（Markdown），會加入提示詞並要求模型遵從。
        model_preds: 高手出牌模型嘅頭 5 個建議（純文字），作為解說主軸。
            傳空字串就同以前一樣，只按精算結果分析。
        direction_hints: 潛在役種方向提示（純文字），冇就傳「無」。
    """
    prompt = build_analysis_prompt(hand, ctx, facts, model_preds, direction_hints)

    if backend == "ollama":
        return _analyze_ollama(prompt)

    candidates = [model]
    for alt in (fallback_models if fallback_models is not None else ANALYSIS_FALLBACK_MODELS):
        if alt and alt not in candidates:
            candidates.append(alt)

    last_err = ""
    primary_err = ""  # 主模型失敗原因（用嚟話俾用戶知點解轉咗 model）
    for m in candidates:
        try:
            data = _post_smart(
                "/chat/completions",
                api_key,
                {
                    "model": m,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": ANALYSIS_MAX_TOKENS,
                    "temperature": 0.3,
                },
                timeout=300,
                effort=_stream_effort(m),
            )
        except AIError as exc:
            last_err = str(exc)
            if m == candidates[0] and not primary_err:
                primary_err = last_err
            continue

        text = _message_text(data)
        tokens = (data.get("usage") or {}).get("total_tokens")
        suffix = f"，{tokens} tokens" if tokens else ""

        if not text:
            last_err = _empty_reason(data)
            if m == candidates[0] and not primary_err:
                primary_err = last_err
            continue

        try:
            finish = data["choices"][0].get("finish_reason") or ""
        except (KeyError, IndexError, TypeError):
            finish = ""
        if finish == "length":
            suffix += "，⚠️ 輸出被 token 上限截斷"

        note = f"分析完成（{m}{suffix}）"
        if m != candidates[0] and primary_err:
            note += (
                f"　⚠️ 主模型 {candidates[0]} 暫時不可用"
                f"（{_short_reason(primary_err)}），已自動轉用備援。"
            )
        return text, note

    return None, last_err or "分析失敗，請稍後再試。"


def _analyze_ollama(prompt: str) -> tuple[str | None, str]:
    """用本機 Ollama 做文字分析（純文字，唔使睇圖）。"""
    payload = {
        "model": tr.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.3, "num_predict": 1200},
    }
    try:
        r = requests.post(f"{tr.OLLAMA_URL}/api/generate", json=payload, timeout=900)
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
    except Exception as exc:  # noqa: BLE001
        return None, f"本機 Ollama 分析失敗：{exc}"

    if not text:
        return None, "本機模型沒有回傳分析內容。"
    return text, f"分析完成（本機 {tr.OLLAMA_MODEL}）"
