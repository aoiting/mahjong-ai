import hashlib
import io

import streamlit as st
from PIL import Image

import tile_recognizer as tr


@st.cache_data(show_spinner=False, max_entries=20)
def cached_recognize(image_bytes: bytes, engine: str) -> tuple[str | None, str]:
    """辨識結果快取 —— 避免 Streamlit rerun 時重複跑（本地模型很耗 CPU）。

    同一張圖片 + 同一個引擎只會真正辨識一次。
    """
    image = Image.open(io.BytesIO(image_bytes))
    return tr.recognize(image, engine=engine)


@st.cache_data(ttl=30, show_spinner=False)
def engine_statuses() -> dict:
    """引擎可用狀態（快取 30 秒，避免每次 rerun 都探測 Ollama）。"""
    return tr.engines()


# 頁面設定
st.set_page_config(
    page_title="日本麻雀 AI 牌局分析助手",
    page_icon="🀄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 標題與說明
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

# 側邊欄：局勢設定
with st.sidebar:
    st.header("⚙️ 局勢設定")

    # API Key 輸入（密碼隱藏）
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-...",
        help="請輸入你的 OpenAI API 金鑰",
    )

    st.divider()

    # 場風與自風
    col1, col2 = st.columns(2)
    with col1:
        bakaze = st.selectbox("場風", ["東", "南", "西", "北"], index=0)
    with col2:
        jikaze = st.selectbox("自風", ["東", "南", "西", "北"], index=0)

    # 巡目選擇器
    junme = st.slider(
        "巡目",
        min_value=1,
        max_value=18,
        value=1,
        step=1,
        help="目前牌局進行至第幾巡",
    )

    # 寶牌輸入
    dora = st.text_input(
        "寶牌 (Dora)",
        placeholder="例如：1p、5s、東",
        help="支援格式：數字＋花色（m=萬、p=筒、s=索）或字牌（東南西北白發中）",
    )

    st.divider()

    # 警報複選框
    st.markdown("**局勢警報**")
    alert_riichi = st.checkbox("是否有對立直")
    alert_meld = st.checkbox("他家副露等警報")

    st.divider()

    # 手牌辨識引擎（全部本機執行，零 API 費用）
    st.markdown("**手牌辨識引擎**")
    engine_label = st.selectbox(
        "辨識方式",
        ["自動偵測", "Ollama 本地視覺模型", "OpenCV 模板比對"],
        index=0,
        help="兩種方式都在你的電腦本機執行，不會上傳圖片，也不會產生 token 費用",
    )
    engine = {
        "自動偵測": "auto",
        "Ollama 本地視覺模型": "ollama",
        "OpenCV 模板比對": "template",
    }[engine_label]

    statuses = engine_statuses()
    if engine == "auto":
        ready = [k for k, (ok, _) in statuses.items() if ok]
        if ready:
            st.caption(f"✅ 自動使用：{ready[0]}")
        else:
            st.caption("⚠️ 未偵測到可用引擎，請參考「即時影相」頁的安裝說明")
    else:
        ok, msg = statuses[engine]
        st.caption(("✅ " if ok else "⚠️ ") + msg)

# 主畫面：手牌輸入
st.subheader("🎴 手牌輸入")
tab_manual, tab_image, tab_camera = st.tabs(["手動輸入", "圖片上傳", "即時影相"])

# Tab 1：手動輸入
with tab_manual:
    st.markdown("**請輸入你的手牌代碼**")
    hand_text = st.text_area(
        "手牌",
        placeholder="範例：123m456p789s11z",
        height=100,
        help="支援格式：m=萬、p=筒、s=索、z=字",
    )
    st.caption("說明：m = 萬子、p = 筒子、s = 索子、z = 字牌（東南西北白發中）")

# Tab 2：圖片上傳
with tab_image:
    st.markdown("**上傳雀魂／天鳳截圖進行自動辨識**")
    uploaded_image = st.file_uploader(
        "選擇圖片檔案",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="支援 PNG、JPG 格式的雀魂或天鳳遊戲截圖",
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="已上傳的截圖預覽", use_container_width=True)

        if st.button("🔍 辨識手牌", key="btn_upload_ocr", use_container_width=True):
            st.session_state.pop("upload_edit", None)
            with st.spinner("本機辨識中，首次可能較慢..."):
                hand, msg = cached_recognize(uploaded_image.getvalue(), engine)
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

# Tab 3：即時影相
with tab_camera:
    st.markdown("**即時影相辨識**")
    st.caption(
        "對準手牌按快門，影完會自動在本機辨識（唔會上傳到任何雲端，亦唔會扣 token）。"
        "影相功能需要以 localhost 或 HTTPS 開啟頁面。"
    )

    camera_image = st.camera_input("對準手牌影相", key="camera_input")

    if camera_image is not None:
        st.image(camera_image, caption="擷取畫面", use_container_width=True)

        # 每張新相片只自動辨識一次（用內容 hash 判斷，配合快取避免重複運算）
        data = camera_image.getvalue()
        sig = hashlib.md5(data).hexdigest()
        if st.session_state.get("camera_sig") != sig:
            st.session_state["camera_sig"] = sig
            st.session_state.pop("camera_edit", None)
            with st.spinner("本機即時辨識中..."):
                hand, msg = cached_recognize(data, engine)
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

    with st.expander("🔧 未偵測到辨識引擎？安裝方法"):
        st.markdown(
            """
**方案 A：本地視覺模型（推薦，截圖／實體牌都讀得）**

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5vl:7b
```

裝好後重新整理頁面即可。你部機係 AMD 內顯（無 CUDA），會用 CPU 推理，
每張圖大約 10～60 秒，但**完全免費**且圖片不會離開你的電腦。

**方案 B：OpenCV 模板比對（最快，但只限固定遊戲畫面）**

1. 截一張包含「一整排手牌」的圖，記下該區塊的 `x,y,w,h`
2. 執行工具自動裁出模板（唔使自己逐張剪）：

```powershell
python tools/make_templates.py screenshot.png --box 120,880,700,110 --labels 123m456p789s11z --debug
```

3. 重複幾次補齊 34 張（工具會顯示還缺哪些牌）
4. 模板存於 `tile_templates/`，檔名如 `1m.png`、`5z.png`

> ⚠️ 注意：`easyocr` / `paddleocr` 係**文字** OCR，認唔到麻雀牌圖案，
> 唔適合用嚟讀手牌。

**小提示**：模板比對最快最準，適合日常打機；Ollama 較慢但通用，
適合實體牌或換了主題的畫面。可在左側「辨識方式」手動切換。
"""
        )

# 彙整手牌來源（手動 > 圖片 > 即時影相）
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
        f"**目前手牌**（來自{hand_source}）：`{effective_hand}`　—　"
        f"{tr.count_tiles(effective_hand)} 張"
    )
else:
    st.warning("尚未取得手牌，請在上面任何一個頁籤輸入或辨識。")

# 分析按鈕
st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button(
        "🚀 開始 AI 深度牌局分析",
        type="primary",
        use_container_width=True,
    )

# 分析結果顯示區
if analyze_btn:
    if not effective_hand:
        st.error("請先輸入手牌或完成辨識，再開始分析。")
    else:
        with st.spinner("🔍 AI 正在進行深度牌局分析，請稍候..."):
            st.success(f"分析完成！（手牌：{effective_hand}）")

            st.subheader("📊 分析結果")

            with st.container(border=True):
                st.markdown("### 🧠 整體評估")
                st.info("（此處將顯示：牌型整體評估、局勢強度、和牌機率概觀）")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("### 👂 聽牌分析")
                    st.write("－向聽數：待分析")
                    st.write("－有效牌：待分析")
                    st.write("－和牌期待：待分析")
                with col_b:
                    st.markdown("### 🎯 切牌建議")
                    st.write("－推薦切牌：待分析")
                    st.write("－理由：待分析")
                    st.write("－風險評估：待分析")
                with col_c:
                    st.markdown("### ⚠️ 危險度分析")
                    st.write("－對立直警戒：待分析")
                    st.write("－他家動向：待分析")
                    st.write("－安全牌建議：待分析")

                st.markdown("### 🏆 役種預測")
                st.write("（此處將顯示：可達成役種、翻數預估、打點預測）")

                st.markdown("### 💡 進一步建議")
                st.write("（此處將顯示：根據巡目、寶牌與場風給出的策略建議）")
else:
    st.info("💡 請完成局勢設定與手牌輸入後，點擊上方按鈕開始 AI 深度牌局分析")
