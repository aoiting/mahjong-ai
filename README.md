# Riichi Lens 立直鏡 — Mahjong AI Assistant

**Live app:** https://riichi-lens.streamlit.app/

## Project proposal

Riichi Lens helps Riichi Mahjong players decide which tile to discard. Upload a photo of your hand and the app gives you one clear recommendation, based on a machine-learning model trained on how expert players actually play — not just an opinion from a general chatbot. A local tile-logic engine provides the exact numbers, and an LLM only writes the friendly explanation.

Scope: four-player East-wind games (東風局), the format the model was trained on.

## Steps taken

1. **Data & ETL** — Collected Tenhou (天鳳) expert game logs (2009–2014, East-wind, 7-dan to Tenhoui): 1,194 games / 6,867 rounds. Converted `mjlog → mjai` and extracted **325,684 expert discard decisions** into `discard_samples.csv`, split by game into train / validation / test (260,969 / 32,892 / 31,823) to avoid leakage.
2. **EDA** — Analysed expert play, e.g. honours and terminals are discarded first, the 5s are kept longest, and 68.4% of expert wins are small 1–3 han hands.
3. **Model** — Trained a `HistGradientBoostingClassifier` to predict the expert's discard (34 classes). The hand-only model used in the app (37 features) scores **56.7% Top-1 / 85.9% Top-3** on the test set, versus 59.8% / 88.2% for a full-table model (81 features) and 4.8% for always guessing one tile. Top-1 is agreement with experts, not a win rate.
4. **App** — Built a Streamlit app: a vision-language model recognises the tiles in a photo (about 18–21 seconds, 14/14 tiles), the expert model makes the decision, the local engine calculates, and the LLM explains. The trained model is included as `handonly_model.joblib`.

## Try it online (easiest)

Just open the live app: https://riichi-lens.streamlit.app/

1. Enter your own OpenRouter API key in the sidebar (needed for tile recognition and explanations; charged per use by OpenRouter — the app itself has no subscription).
2. Take a photo of your hand with your phone's camera, then upload it.
3. Read the recommended discard, the tile-logic table, and the explanation.

## Or run it locally

Only needed if you want to run the code yourself (e.g. to inspect or mark the project):

```bash
git clone https://github.com/aoiting/mahjong-ai.git
cd mahjong-ai
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local address Streamlit prints (usually http://localhost:8501). Python 3.12 is recommended. The main file is `app.py`.

## Limitations

East-wind games only. The model recommends a discard only — it does not decide riichi, and its accuracy is agreement with experts, not a proven win-rate improvement.
