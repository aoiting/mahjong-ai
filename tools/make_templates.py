"""由遊戲截圖裁出麻雀牌模板（供 OpenCV 模板比對使用）。

用法
----
1. 先開啟 `tile_templates/`（不存在會自動建立）
2. 用截圖工具框住「一整排手牌」（每張牌寬度要一致）
3. 記下該區塊的左上角座標與闊高，然後執行：

    python tools/make_templates.py screenshot.png ^
        --box 120,880,700,110 ^
        --labels 123m456p789s11z ^
        --debug

    * --box      x,y,w,h  (手牌區塊)
    * --labels   該排牌由左至右是甚麼（支援 123m、東南西北白發中 等寫法）
    * --debug    額外輸出一張 preview.png，畫出每格切割位置方便核對

輸出的模板會存成 tile_templates/1m.png、tile_templates/5z.png …，
檔名對應 tile_recognizer.TILE_LABELS。

注意：模板只對「同一個遊戲客戶端、同一種縮放／主題」有效。
換了主題、視窗大小或改用實體牌，就要重新做模板（或改用 Ollama）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 讓工具可以直接由專案根目錄執行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tile_recognizer import TEMPLATE_DIR, count_tiles, normalize_hand  # noqa: E402


def expand_labels(labels: str) -> list[str]:
    """把 123m456p789s11z 展開成 ['1m','2m',...,'1z','1z']。"""
    hand = normalize_hand(labels)
    if not hand:
        raise SystemExit(f"無法解析 --labels：{labels!r}")

    tiles: list[str] = []
    import re

    for digits, suit in re.findall(r"(\d+)([mpsz])", hand):
        tiles.extend(f"{d}{suit}" for d in digits)
    return tiles


def imread_unicode(path: Path) -> np.ndarray:
    """支援中文路徑的 cv2.imread。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"讀不到圖片：{path}")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="由截圖裁出麻雀牌模板")
    ap.add_argument("image", help="來源截圖路徑")
    ap.add_argument("--box", required=True, help="手牌區塊 x,y,w,h")
    ap.add_argument("--labels", required=True, help="由左至右的牌，例如 123m456p789s11z")
    ap.add_argument("--inset", type=int, default=2, help="每格向內縮多少像素（預設 2）")
    ap.add_argument("--debug", action="store_true", help="輸出水印預覽圖以核對切割位置")
    args = ap.parse_args()

    src = Path(args.image)
    img = imread_unicode(src)

    try:
        x, y, w, h = (int(v) for v in args.box.split(","))
    except ValueError:
        raise SystemExit("--box 格式錯誤，應為 x,y,w,h（例如 120,880,700,110）")

    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        raise SystemExit(
            f"--box 超出圖片範圍：圖片為 {img.shape[1]}x{img.shape[0]}，你給的是 {x},{y},{w},{h}"
        )

    tiles = expand_labels(args.labels)
    n = len(tiles)
    if n == 0:
        raise SystemExit("--labels 解析後沒有任何牌")

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    cell_w = w / n
    print(f"共 {n} 張牌，每格寬度 {cell_w:.2f}px，高度 {h}px")

    saved: list[str] = []
    skipped: list[str] = []
    preview = img.copy()

    for i, tile in enumerate(tiles):
        cx0 = int(round(x + i * cell_w)) + args.inset
        cx1 = int(round(x + (i + 1) * cell_w)) - args.inset
        cy0 = y + args.inset
        cy1 = y + h - args.inset
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            raise SystemExit("切割出來的格子太小，請檢查 --box")

        cell = img[cy0:cy1, cx0:cx1]
        out = TEMPLATE_DIR / f"{tile}.png"

        if out.exists():
            skipped.append(tile)
        else:
            cv2.imwrite(str(out), cell)
            saved.append(tile)

        cv2.rectangle(preview, (cx0, cy0), (cx1, cy1), (0, 200, 0), 2)
        cv2.putText(
            preview, tile, (cx0 + 2, cy0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
        )

    print(f"已儲存 {len(saved)} 張模板 → {TEMPLATE_DIR}")
    if saved:
        print("  " + ", ".join(saved))
    if skipped:
        print(f"已存在而略過（如需覆寫請先刪除）：{', '.join(sorted(set(skipped)))}")

    have = {p.stem for p in TEMPLATE_DIR.glob("*.png")}
    from tile_recognizer import TILE_LABELS

    missing = [t for t in TILE_LABELS if t not in have]
    print(f"目前模板進度：{34 - len(missing)}/34")
    if missing:
        print("仍缺少：" + ", ".join(missing))

    if args.debug:
        dbg = TEMPLATE_DIR.parent / "template_preview.png"
        cv2.imwrite(str(dbg), preview)
        print(f"預覽圖：{dbg}")


if __name__ == "__main__":
    main()
