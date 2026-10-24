"""手動測試逐張辨識（唔會寫死 API key）。

用法：
    py tools/test_recognize.py <圖片路徑> [模型]

API key 會依次由環境變數 OPENROUTER_API_KEY、.streamlit/secrets.toml 讀取。
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import ai_client as ai
import tile_recognizer as tr

ROOT = Path(__file__).resolve().parent.parent


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：py tools/test_recognize.py <圖片路徑> [模型]")
        raise SystemExit(1)

    path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else ai.DEFAULT_MODEL
    key = load_key()
    if not key:
        print("找唔到 OPENROUTER_API_KEY（環境變數或 .streamlit/secrets.toml）。")
        raise SystemExit(1)

    im = Image.open(path).convert("RGB")
    boxes = tr.split_hand_tiles(im)
    if not boxes:
        print("切唔到手牌，會用整條辨識。")
        hand, msg = ai.recognize_hand(im, key, model)
        print("HAND:", hand)
        print("MSG:", msg)
        return

    print(f"圖片：{path}（{len(boxes)} 張）　模型：{model}", flush=True)

    t0 = time.time()
    hand, msg = ai.recognize_hand_per_tile(
        im,
        key,
        boxes,
        model,
        progress=lambda done, total, tiles: print(f"  {done}/{total}: {tiles[-1]}", flush=True),
        local_first=True,
    )
    print("HAND:", hand)
    print("MSG:", msg)
    print("time: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
