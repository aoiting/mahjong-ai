"""高手出牌預測模型（handonly_model.joblib）嘅接入層。

模型同推論包裝（predict_discard.py）由 `train/` 複製過嚟，
特徵編碼唔好改。呢個模組只負責：
  1. 將 app 嘅手牌格式（5m / 1p / 3z）轉成模型格式（5m / 1p / W）
  2. 將寶牌指示牌文字（例如「3p、東」）轉成模型格式
  3. 包裝錯誤，確保 model 缺失或載入失敗都唔會令 app 崩潰
"""

from __future__ import annotations

from pathlib import Path

import tile_images as ti
import tile_recognizer as tr

MODEL_PATH = Path(__file__).with_name("handonly_model.joblib")

# app 字牌：1z=東 2z=南 3z=西 4z=北 5z=白 6z=發 7z=中
_HONOR_TO_ML = {1: "E", 2: "S", 3: "W", 4: "N", 5: "P", 6: "F", 7: "C"}
_ML_TO_HONOR = {v: k for k, v in _HONOR_TO_ML.items()}


def available() -> bool:
    """模型檔案存在？"""
    return MODEL_PATH.exists()


def to_ml_tile(tile: str) -> str:
    """'1z' → 'E'、'5m' → '5m'、'5mr' → '5mr'（模型靠 r 計赤五特徵）。"""
    t = (tile or "").strip()
    if not t:
        return ""
    if t.endswith("r"):
        return t if t[-2:-1] in ("m", "p", "s") else t[:-1]
    if len(t) == 2 and t[0].isdigit() and t[1] == "z":
        return _HONOR_TO_ML.get(int(t[0]), "")
    return t


def from_ml_tile(tile: str) -> str:
    """模型格式 → app 格式（'E' → '1z'、'5m' → '5m'）。"""
    t = tile[:-1] if tile.endswith("r") else tile
    if t in _ML_TO_HONOR:
        return f"{_ML_TO_HONOR[t]}z"
    return t


def hand_to_ml(hand: str | list[str]) -> list[str]:
    """'5m112357p1567s113z' 或牌 list → ['5m','1p',...,'E','E','W']"""
    if isinstance(hand, (list, tuple)):
        return [x for x in (to_ml_tile(t) for t in hand) if x]
    return [x for x in (to_ml_tile(c) for c in ti.expand_hand(hand)) if x]


def dora_to_ml(text: str) -> list[str]:
    """寶牌指示牌文字 → 模型格式 list；空白回 []。"""
    if not text or not text.strip():
        return []
    normalised = tr.normalize_hand(text)
    return [x for x in (to_ml_tile(c) for c in ti.expand_hand(normalised)) if x]


_HONOR_LABELS = {1: "東", 2: "南", 3: "西", 4: "北", 5: "白", 6: "發", 7: "中"}


def display_name(tile: str) -> str:
    """app 格式 → 易讀名稱：3z → 西，其餘保留 5m / 3p。"""
    if len(tile) == 2 and tile[0].isdigit() and tile[1] == "z":
        return _HONOR_LABELS.get(int(tile[0]), tile)
    return tile


def suggest(
    hand: str | list[str],
    dora_text: str = "",
    meld_count: int = 0,
    top: int = 5,
) -> tuple[list[dict] | None, str]:
    """回傳 (建議 list, 錯誤訊息)。

    建議 list：`[{"tile": app 格式, "ml": 模型格式, "p": 機率}, ...]`
    任何失敗都只會回傳錯誤訊息，唔會拋 exception。
    """
    tiles = hand_to_ml(hand)
    if len(tiles) < 2:
        return None, "手牌張數不足，模型無法給出建議。"

    try:
        from predict_discard import predict  # 延遲 import：缺套件都唔會拖垮 app
    except Exception as exc:  # noqa: BLE001
        return None, f"無法載入推論程式：{exc}"

    try:
        raw = predict(
            tiles,
            dora_markers=dora_to_ml(dora_text),
            meld_count=int(meld_count),
            model_path=str(MODEL_PATH),
            top=top,
        )
    except FileNotFoundError:
        return None, "搵唔到模型檔案 handonly_model.joblib。"
    except Exception as exc:  # noqa: BLE001
        return None, f"模型載入／預測失敗：{exc}"

    results = [
        {"tile": from_ml_tile(t), "ml": t, "p": float(p)}
        for t, p in raw
    ]
    if not results:
        return None, "模型沒有給出建議（可能手牌張數不完整）。"
    return results, ""
