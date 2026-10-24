import hashlib
import io
import os

import streamlit as st
from PIL import Image

import ai_client as ai
import mahjong_engine as me
import tile_images as ti
import tile_recognizer as tr

# OpenRouter 模型（實測後嘅建議）
# 註：minimax-m3.1-flash:free 免費期已於 2026-10-22 結束；Gemini 喺香港被地區封鎖。
DEFAULT_VISION_MODEL = "meta/muse-spark-1.3-contributor"
DEFAULT_ANALYSIS_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


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


# 辨識結果快取 —— 只快取「成功」嘅結果。
# 之前用 st.cache_data 會連失敗都一齊快取，令「token 用盡」嘅錯誤永久定型，
# 點解重試都係同一個錯。改用自己嘅 dict，只存成功結果。
_RECOGNIZE_CACHE: dict[str, tuple[str, str]] = {}


def _recognize_now(
    image_bytes: bytes,
    engine: str,
    provider: str,
    api_key: str,
    vision_model: str,
    bottom_pct: int = 100,
    per_tile: bool = True,
    local_first: bool = False,
) -> tuple[str | None, str]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = tr.crop_hand_region(image, bottom_pct)

    if engine == "auto":
        engine = "template" if tr.template_status()[0] else "ai"

    if engine == "template":
        return tr.recognize(image, engine="template")

    if provider == "openrouter":
        # 先試自動切牌 + 逐張辨識（準確度明顯較高）
        if per_tile:
            boxes = tr.split_hand_tiles(image)
            if boxes:
                return ai.recognize_hand_per_tile(
                    image, api_key, boxes, vision_model, local_first=local_first
                )
        # 切唔到（實體牌相片／完整截圖）→ 用整條辨識
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
) -> tuple[str | None, str]:
    """辨識結果快取 —— 避免 Streamlit rerun 時重複跑（本機模型很耗 CPU）。

    失敗唔會入快取，所以上游一恢復就即刻可以重試。
    """
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
        return _RECOGNIZE_CACHE[sig]

    hand, msg = _recognize_now(
        image_bytes, engine, provider, api_key, vision_model,
        bottom_pct, per_tile, local_first,
    )
    if hand:
        _RECOGNIZE_CACHE[sig] = (hand, msg)
    return hand, msg


@st.cache_data(ttl=30, show_spinner=False)
def local_statuses() -> dict:
    """本機引擎可用狀態（快取 30 秒，避免每次 rerun 都探測 Ollama）。"""
    return tr.engines()


@st.cache_data(show_spinner=False, max_entries=50)
def cached_calc(hand: str, bakaze: str, jikaze: str, dora: str) -> dict:
    """精算結果快取（純本地計算，瞬間完成）。"""
    return me.analyze(hand, {"bakaze": bakaze, "jikaze": jikaze, "dora": dora})


# ── 頁面設定 ────────────────────────────────────────────────
st.set_page_config(
    page_title="日本麻雀 AI 牌局分析助手",
    page_icon="🀄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <div style="text-align: center; padding-bottom: 1rem;">
        <h1>🀄 日本麻雀 AI 牌局分析助手</h1>
        <p style="font-size: 1.1rem; color: #666;">
            輸入手牌或上傳雀魂／天鳳截圖，讓 AI 協助深度分析牌局局勢
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 側邊欄 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("🤖 AI 設定")

    saved_key = get_secret("OPENROUTER_API_KEY")

    provider_label = st.selectbox(
        "AI 供應商",
        ["OpenRouter（雲端・唔使開機）", "本機 Ollama（免費・私隱）"],
        index=0 if saved_key else 1,
        help="雲端模式需要網絡同 API key；本機模式需要開住電腦並安裝 Ollama",
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
            help="唔會寫入程式碼；本機存於 .streamlit/secrets.toml（已被 gitignore）",
        )

        with st.expander("模型設定", expanded=False):
            vision_model = st.text_input(
                "手牌辨識模型（需要視覺能力）",
                value=get_secret("OPENROUTER_VISION_MODEL", default=DEFAULT_VISION_MODEL),
            )
            analysis_model = st.text_input(
                "牌局分析模型",
                value=get_secret("OPENROUTER_ANALYSIS_MODEL", default=DEFAULT_ANALYSIS_MODEL),
            )
            st.caption(
                "預設辨識用 `meta/muse-spark-1.3-contributor`（準、有視覺），"
                "被封鎖時會自動轉用 `meta/muse-spark-open`。"
                "分析用 `nvidia/nemotron-3-ultra-550b-a55b:free`（約 1–2 分鐘）。"
            )

        if st.button("🔌 測試連線", width="stretch"):
            with st.spinner("測試中..."):
                ok, msg = ai.check_key(api_key, vision_model)
            (st.success if ok else st.error)(msg)

        if not api_key:
            st.warning("請輸入 OpenRouter API Key 先可以用雲端功能。")
        elif analysis_model.endswith(":free"):
            st.caption("💡 免費模型約需 1–2 分鐘；加 US$5 credit 可改用更快更準嘅付費模型。")
    else:
        ok, msg = local_statuses()["ollama"]
        st.caption(("✅ " if ok else "⚠️ ") + msg)

    st.divider()
    st.header("⚙️ 局勢設定")

    col1, col2 = st.columns(2)
    with col1:
        bakaze = st.selectbox("場風", ["東", "南", "西", "北"], index=0)
    with col2:
        jikaze = st.selectbox("自風", ["東", "南", "西", "北"], index=0)

    junme = st.slider(
        "巡目",
        min_value=1,
        max_value=18,
        value=1,
        step=1,
        help="目前牌局進行至第幾巡",
    )

    dora = st.text_input(
        "寶牌指示牌 (Dora)",
        placeholder="例如：3p、5s、東",
        help="輸入牌山翻開嘅指示牌（例如翻開 3p，即代表 4p 係寶牌）",
    )

    st.divider()
    st.markdown("**局勢警報**")
    alert_riichi = st.checkbox("是否有對立直")
    alert_meld = st.checkbox("他家副露等警報")

    st.divider()
    st.markdown("**手牌辨識方式**")
    engine_label = st.selectbox(
        "辨識方式",
        ["自動（有模板用模板，否則用 AI）", "AI 視覺辨識", "OpenCV 模板比對"],
        index=0,
        help="模板比對最快但要先建立 34 張模板；AI 視覺通用但較慢",
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

    per_tile = st.checkbox(
        "🔍 先切牌逐張辨識（推薦）",
        value=True,
        help=(
            "自動偵測手牌條上每張牌嘅邊界，然後逐張放大辨識。"
            "實測準確度明顯高過「整條 14 張一齊睇」。"
            "需要 AI 視覺辨識（唔適用於模板比對）。"
            "注意：會用多啲 API 請求次數（每手牌約 10–14 次）。"
        ),
    )

    local_first = st.checkbox(
        "⚡ 省 API 模式（推薦）",
        value=True,
        disabled=not per_tile,
        help=(
            "本地形狀分析（數圓圈／數竹枝／認空白）夠肯定嘅牌，直接採用唔問 AI。"
            "實測喺示範圖 18/18 全對，而且可以省 60% 以上請求次數、避開免費層限流。"
            "剩低真正要 AI 判斷嘅（萬子漢數字、字牌、部分筒子）才會送去問。"
        ),
    )

# ── 手牌輸入 ────────────────────────────────────────────────
st.subheader("🎴 手牌輸入")
tab_manual, tab_image, tab_camera = st.tabs(["手動輸入", "圖片上傳", "即時影相"])

with tab_manual:
    st.markdown("**請輸入你的手牌代碼**")
    hand_text = st.text_area(
        "手牌",
        placeholder="範例：123m456p789s11z",
        height=100,
        help="支援格式：m=萬、p=筒、s=索、z=字",
    )
    st.caption("說明：m = 萬子、p = 筒子、s = 索子、z = 字牌（東南西北白發中）")

with tab_image:
    st.markdown("**上傳雀魂／天鳳截圖進行自動辨識**")
    uploaded_image = st.file_uploader(
        "選擇圖片檔案",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="支援 PNG、JPG 格式的雀魂或天鳳遊戲截圖",
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="已上傳的截圖預覽", width="stretch")

        bottom_pct = st.slider(
            "🔍 只分析畫面底部百分比",
            min_value=15,
            max_value=100,
            value=100,
            step=5,
            help=(
                "雀魂／天鳳截圖入面手牌喺最底。調低呢個值會裁走上面嘅部分並放大，"
                "令 AI 睇得更準（100 = 唔裁剪）。"
            ),
        )

        col1, col2 = st.columns(2)
        with col1:
            do_recognize = st.button(
                "🔍 辨識手牌", key="btn_upload_ocr", width="stretch"
            )
        with col2:
            do_force = st.button(
                "🔄 重新辨識（略過快取）", key="btn_upload_ocr_force", width="stretch"
            )

        if do_recognize or do_force:
            st.session_state.pop("upload_edit", None)
            with st.spinner("辨識中..."):
                hand, msg = cached_recognize(
                    uploaded_image.getvalue(),
                    engine,
                    provider,
                    api_key,
                    vision_model,
                    force=do_force,
                    bottom_pct=bottom_pct,
                    per_tile=per_tile,
                    local_first=local_first,
                )
            st.session_state["upload_hand"] = hand or ""
            st.session_state["upload_msg"] = msg

        if st.session_state.get("upload_msg"):
            if st.session_state.get("upload_hand"):
                st.success(st.session_state["upload_msg"])
            else:
                st.warning(st.session_state["upload_msg"])

        if st.session_state.get("upload_hand"):
            st.session_state["upload_hand"] = st.text_input(
                "辨識結果（可直接修改）",
                value=st.session_state["upload_hand"],
                key="upload_edit",
                help="格式：123m456p789s11z",
            )

with tab_camera:
    st.markdown("**即時影相辨識**")
    st.caption(
        "對準手牌按快門，影完會自動辨識。影相功能需要以 localhost 或 HTTPS 開啟頁面。"
    )

    camera_image = st.camera_input("對準手牌影相", key="camera_input")

    if camera_image is not None:
        st.image(camera_image, caption="擷取畫面", width="stretch")

        data = camera_image.getvalue()
        sig = hashlib.md5(data).hexdigest()
        if st.session_state.get("camera_sig") != sig:
            st.session_state["camera_sig"] = sig
            st.session_state.pop("camera_edit", None)
            with st.spinner("即時辨識中..."):
                hand, msg = cached_recognize(
                    data,
                    engine,
                    provider,
                    api_key,
                    vision_model,
                    force=True,
                    per_tile=per_tile,
                    local_first=local_first,
                )
            st.session_state["camera_hand"] = hand or ""
            st.session_state["camera_msg"] = msg

        if st.button("🔄 重新辨識", key="btn_camera_retry"):
            st.session_state.pop("camera_sig", None)
            st.rerun()

        if st.session_state.get("camera_msg"):
            if st.session_state.get("camera_hand"):
                st.success(st.session_state["camera_msg"])
            else:
                st.warning(st.session_state["camera_msg"])

        if st.session_state.get("camera_hand"):
            st.session_state["camera_hand"] = st.text_input(
                "辨識結果（可直接修改）",
                value=st.session_state["camera_hand"],
                key="camera_edit",
                help="格式：123m456p789s11z",
            )

    with st.expander("🔧 未偵測到本機辨識引擎？安裝方法"):
        st.markdown(
            """
**方案 A：本機視覺模型（免費、圖片不外傳）**

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5vl:7b
```

AMD 內顯（無 CUDA）會用 CPU 推理，每張圖約 10～60 秒。

**方案 B：OpenCV 模板比對（最快，只限固定遊戲畫面）**

1. 截一張包含「一整排手牌」的圖，記下區塊 `x,y,w,h`
2. 用工具自動裁出模板：

```powershell
python tools/make_templates.py screenshot.png --box 120,880,700,110 --labels 123m456p789s11z --debug
```

3. 重複幾次補齊 34 張，模板存於 `tile_templates/`

> ⚠️ `easyocr` / `paddleocr` 係文字 OCR，認唔到麻雀牌圖案。
"""
        )

# ── 彙整手牌（手動 > 圖片 > 即時影相）──────────────────────
manual_hand = tr.normalize_hand(hand_text)
upload_hand = tr.normalize_hand(st.session_state.get("upload_hand", ""))
camera_hand = tr.normalize_hand(st.session_state.get("camera_hand", ""))

effective_hand, hand_source = "", ""
for source, value in (
    ("手動輸入", manual_hand),
    ("圖片上傳", upload_hand),
    ("即時影相", camera_hand),
):
    if value:
        effective_hand, hand_source = value, source
        break

st.divider()
if effective_hand:
    st.markdown(
        f"**目前手牌**（來自{hand_source}）　·　"
        f"{tr.count_tiles(effective_hand)} 張"
    )
    ti.render_hand(effective_hand, height=54)
    with st.expander("顯示牌局代碼（複製／除錯用）"):
        st.code(effective_hand, language=None)
else:
    st.warning("尚未取得手牌，請在上面任何一個頁籤輸入或辨識。")

# ── 精算（本機、即時、零成本）──────────────────────────────
calc: dict = {}
if effective_hand:
    calc = cached_calc(effective_hand, bakaze, jikaze, dora.strip())

    st.markdown("#### 🧮 精算結果（本機即時計算・準確）")
    if calc.get("error"):
        st.warning(calc["error"])
    else:
        st.markdown(
            '<div style="background:#e9f7ef;border:1px solid #b7e2c6;'
            'border-radius:8px;padding:10px 14px;margin-bottom:8px;">'
            + ti.decorate_text(calc.get("summary", ""), height=28)
            + "</div>",
            unsafe_allow_html=True,
        )

        if calc.get("discards"):
            rows = [
                {
                    "tile": d["tile"],
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
            with st.expander("顯示待牌代碼"):
                st.code(
                    " ".join(t for t, _ in calc["waits"]), language=None
                )

        if calc.get("agari_preview"):
            st.markdown("**和牌預覽**（榮和、門清）：")
            for v in calc["agari_preview"]:
                yaku = "、".join(v["yaku"]) or "無役"
                cost = (v.get("cost") or {}).get("main", 0)
                st.markdown(
                    f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;'
                    f'margin:4px 0;">'
                    f'<span style="color:#666;font-size:0.9rem;">待</span>'
                    f'{ti.inline_html(v["wait"], 30)}'
                    f'<span>{yaku}　{v["han"]} 翻 {v["fu"]} 符　{cost} 點</span></div>',
                    unsafe_allow_html=True,
                )

        if calc.get("agari"):
            for kind, v in calc["agari"].items():
                yaku = "、".join(v["yaku"]) or "無役"
                st.markdown(f"- **{kind}**：{yaku}　{v['han']} 翻 {v['fu']} 符")

        st.caption("以上由麻雀計算引擎精確算出（唔涉及 AI，唔會出錯）。")

# ── 分析 ────────────────────────────────────────────────────
st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button(
        "🚀 開始 AI 深度牌局分析",
        type="primary",
        width="stretch",
    )

if analyze_btn:
    if not effective_hand:
        st.error("請先輸入手牌或完成辨識，再開始分析。")
    elif provider == "openrouter" and not api_key:
        st.error("雲端模式需要 OpenRouter API Key，請先在左側輸入。")
    else:
        ctx = {
            "bakaze": bakaze,
            "jikaze": jikaze,
            "junme": junme,
            "dora": dora.strip(),
            "riichi": alert_riichi,
            "meld": alert_meld,
        }
        label = "☁️ 雲端模型" if provider == "openrouter" else "💻 本機模型"
        facts = me.format_facts(calc) if calc else ""
        with st.spinner(f"{label}分析中，請稍候..."):
            text, msg = ai.analyze_hand(
                effective_hand,
                ctx,
                backend=provider,
                api_key=api_key,
                model=analysis_model,
                facts=facts,
            )
        st.session_state["analysis"] = {
            "hand": effective_hand,
            "text": text,
            "msg": msg,
            "provider": provider,
        }

result = st.session_state.get("analysis")
if result:
    if result.get("text"):
        st.success(result["msg"])
        st.subheader("📊 分析結果")
        with st.container(border=True):
            st.markdown(
                ti.decorate_text(result["text"], height=26),
                unsafe_allow_html=True,
            )
        st.caption(result["msg"])
        st.warning(
            "⚠️ AI 分析僅供參考，可能有錯（例如誤判役種、向聽數）。重要決定請自行核對。"
        )
    else:
        st.error(f"分析失敗：{result.get('msg', '未知錯誤')}")
elif not analyze_btn:
    if effective_hand:
        st.info("💡 上方已顯示本機精算結果；如要 AI 深入解說，點擊「開始 AI 深度牌局分析」。")
    else:
        st.info("💡 請完成局勢設定與手牌輸入後，點擊上方按鈕開始 AI 深度牌局分析")
