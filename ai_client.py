"""OpenRouter 客戶端：手牌視覺辨識 + 牌局分析。

用 `requests` 直接呼叫 OpenRouter 嘅 OpenAI 相容端點。

注意：呢部機嘅 `openai` SDK（經 httpx2 → truststore）喺 Python 3.14 會
觸發 `RecursionError: maximum recursion depth exceeded`，所以刻意唔用 SDK。

亦支援用本機 Ollama 做文字分析，方便同一個介面切換供應商。
"""

from __future__ import annotations

import base64
import io
import os
import re
import time

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
ANALYSIS_MAX_TOKENS = 6000

# 視覺辨識備援模型（按實測可靠度排序；第一個掛咗就試下一個）
# 註：minimax-m3.1-flash:free 嘅免費期已於 2026-10-22 結束，唔可以再用。
VISION_FALLBACK_MODELS = [
    "meta/muse-spark-1.3-contributor",
    "meta/muse-spark-open",
    "meta/muse-glimmer-30b",
]

# 分析備援模型（免費、有正式文字輸出）
ANALYSIS_FALLBACK_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "poolside/laguna-s-2.1:free",
]

VISION_PROMPT = """你係日本麻雀（立直麻雀）牌面辨識器。請辨識圖片中「手牌」嘅 13 或 14 張牌。

【牌面辨識規則】
1. 萬子（m）：上面係漢數字（一二三四五六七八九），下面係「萬」字。
2. 筒子（p）：數圓形圖案數量。
   - 1筒：一個特大嘅花形圓圈
   - 2～9筒：2 至 9 個細圓圈，按固定排列（5筒排成 X、3筒斜線）
3. 索子（s）：數竹節數量。
   - 1索：一隻雀鳥圖案
   - 2～9索：2 至 9 支垂直竹枝
4. 風牌：一個大字 —— 東、南、西、北。
5. 三元牌：紅「中」、綠「發」、白（空白或有藍／黑外框）。
6. 赤牌（紅色 5）一律當作普通 5，唔使特別標示。
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
- 索子：1s～9s（綠色竹枝；1s 係綠色雀鳥）
- 字牌：1z=東 2z=南 3z=西 4z=北 5z=白 6z=發 7z=中
- 如果張牌完全空白、冇任何圖案 → 答 5z
格式示範（唔好照抄，只示格式）：3p"""


def _parse_single_tile(text: str) -> str | None:
    """由模型回覆抽出單張牌代碼。"""
    if not text:
        return None
    hand = tr.normalize_hand(text)
    found = re.findall(r"\d[mpsz]", hand)
    return found[0] if found else None


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
                body["reasoning"] = {"effort": "low"}
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
    """
    parts = []
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
            "搵唔到模型",
            "無供應商",
            "api key 無效",
            "餘額不足",
        )
    )


def recognize_hand_per_tile(
    image: Image.Image,
    api_key: str,
    boxes: list[tuple[int, int]],
    model: str,
    fallback_models: list[str] | None = None,
    progress=None,
    local_first: bool = False,
) -> tuple[str | None, str]:
    """先切牌、再逐張辨識。

    Args:
        boxes: [(x0, x1), ...]，由 tr.split_hand_tiles() 得出。
        progress: 可選 callback(done, total, tiles)。
        local_first: True 就將本地形狀分析夠肯定嘅牌直接採用（省 API）。

    Returns:
        (hand, message)
    """
    models = [model]
    for alt in (fallback_models if fallback_models is not None else VISION_FALLBACK_MODELS):
        if alt and alt not in models:
            models.append(alt)

    tiles: list[str | None] = []
    errors: list[str] = []
    dead: set[str] = set()
    total = len(boxes)

    # 免費模型有嚴格速率限制（免費層約 20 次/分鐘），所以要節流；
    # 付費模型冇呢個問題，可以快好多。
    primary = models[0] if models else ""
    delay = float(
        os.environ.get("MAHJONG_PER_TILE_DELAY")
        or (3.0 if primary.endswith(":free") else 0.5)
    )

    for idx, (x0, x1) in enumerate(boxes):
        crop = image.crop((x0, 0, x1, image.height))

        # 本地快速通道：空白牌一定係「白」（5z），唔使問 AI
        profile = tr.tile_color_profile(crop)
        if profile["ink"] < tr.BLANK_INK_MAX:
            tiles.append("5z")
            if progress:
                progress(idx + 1, total, tiles)
            continue

        hint_text, local_answer = tr.tile_shape_analysis(crop)

        # 省 API 模式：本地分析夠肯定就直接採用，唔使問 AI
        if local_first and local_answer:
            tiles.append(local_answer)
            if progress:
                progress(idx + 1, total, tiles)
            continue

        if idx > 0:
            time.sleep(delay)  # 節流，避免觸發上游速率限制

        hint = local_hint(profile, hint_text)
        code = None
        last = ""
        used_model = ""
        # 主模型試 3 次（上游常有暫時性 502），失敗先逐個備援模型試 1 次。
        # 如果某模型係「完全用唔到」（例如被 guardrail 封鎖），即刻標記為 dead，
        # 後面嘅牌就唔會再浪費時間重試。
        attempts = [(models[0], 3)] + [(m, 1) for m in models[1:]]
        for m, tries in attempts:
            if m in dead:
                continue
            for attempt in range(tries):
                code, msg = recognize_tile(crop, api_key, m, hint=hint)
                if code:
                    used_model = m
                    break
                last = msg
                if _model_unusable(msg):
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

        tiles.append(code)
        if not code:
            errors.append(f"第 {idx + 1} 張：{last}")
        if progress:
            progress(idx + 1, total, tiles)

    good = [t for t in tiles if t]
    if len(good) < total * 0.7:
        detail = errors[0] if errors else "全部失敗"
        return None, f"逐張辨識失敗（只有 {len(good)}/{total} 張成功）。{detail}"

    hand = tr.normalize_hand("".join(t or "" for t in tiles))
    breakdown = " ".join(t or "?" for t in tiles)
    note = f"逐張辨識完成（{len(good)}/{total} 張）：{breakdown}"
    if errors:
        note += f"　⚠️ {len(errors)} 張要用備援／失敗，建議人手核對"
    return hand, note


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


def build_analysis_prompt(hand: str, ctx: dict, facts: str = "") -> str:
    """組裝牌局分析提示詞。"""
    alerts = []
    if ctx.get("riichi"):
        alerts.append("有對手立直")
    if ctx.get("meld"):
        alerts.append("他家有副露（鳴牌）")
    alert_text = "、".join(alerts) if alerts else "無特別警報"

    n = tr.count_tiles(hand)
    discard_hint = (
        "手牌 14 張，請明確指出應該打邊張。"
        if n == 14
        else "手牌 13 張（或更少），請針對進張與路線給建議。"
    )

    return f"""你係日本麻雀（立直麻雀）嘅職業雀士兼牌局分析師。請用繁體中文（香港用語）分析以下局面。

【局面資料】
- 場風：{ctx.get("bakaze", "東")}場
- 自風：{ctx.get("jikaze", "東")}家
- 巡目：第 {ctx.get("junme", 1)} 巡
- 寶牌：{ctx.get("dora") or "未指定"}
- 警報：{alert_text}
- 手牌（{n} 張）：{hand}

{discard_hint}

{facts}

【重要規則】
- 以上「精算結果」係由麻雀計算引擎精確算出，**你必須以此為準**。
- 唔可以否定或修改精算嘅向聽數、有效牌、役種、翻數、符數。
- 你嘅價值在於解釋「點解」同提供策略，而唔係重新計算。
- 如果精算結果同你嘅直覺有衝突，一律以精算結果為準。

【分析要求】
1. 評估牌型強度、和牌可能性
2. 解釋精算建議嘅切牌理由（若已聽牌，說明待牌）
3. 說明進張方向對牌型的意義
4. 評估危險度，特別注意立直他家與副露家
5. 役種：只可引用精算結果；冇提供就寫「未確定」
6. 給出今後 2～3 巡的策略建議

【輸出格式】用 Markdown，依序使用以下標題：
### 🧠 整體評估
### 🎯 切牌建議
### ⚠️ 危險度分析
### 🏆 役種
### 💡 進一步建議

【嚴格禁止】
- 唔准自行計算翻數、符數或點數。只有「精算結果」有提供嘅數字才可以引用。
- 如果精算結果**沒有**提供役種／和牌預覽，即代表手牌未聽牌或役種未確定。
  此時「🏆 役種」一節只可以寫：「役種未確定，待聽牌後由精算引擎確認」，
  唔准自行列舉平和、一盃口、三色、斷么等役種。
- 唔准講「已經立直」，除非精算結果顯示已和牌。

請具體、精簡，避免空泛嘅一般性建議。"""


def analyze_hand(
    hand: str,
    ctx: dict,
    backend: str = "openrouter",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    facts: str = "",
    fallback_models: list[str] | None = None,
) -> tuple[str | None, str]:
    """分析牌局。

    Args:
        backend: "openrouter" 或 "ollama"
        facts: 本地精算結果（Markdown），會加入提示詞並要求模型遵從。
    """
    prompt = build_analysis_prompt(hand, ctx, facts)

    if backend == "ollama":
        return _analyze_ollama(prompt)

    candidates = [model]
    for alt in (fallback_models if fallback_models is not None else ANALYSIS_FALLBACK_MODELS):
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
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": ANALYSIS_MAX_TOKENS,
                    "temperature": 0.3,
                },
                timeout=300,
            )
        except AIError as exc:
            last_err = str(exc)
            continue

        text = _message_text(data)
        tokens = (data.get("usage") or {}).get("total_tokens")
        suffix = f"，{tokens} tokens" if tokens else ""

        if not text:
            last_err = _empty_reason(data)
            continue

        try:
            finish = data["choices"][0].get("finish_reason") or ""
        except (KeyError, IndexError, TypeError):
            finish = ""
        if finish == "length":
            suffix += "，⚠️ 輸出被 token 上限截斷"
        return text, f"分析完成（{m}{suffix}）"

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
