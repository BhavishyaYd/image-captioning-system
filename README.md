# Image Captioning System

Minimal, self-contained copy of an image captioning system focused on the inference GUI and core scripts.

What’s included
- `app.py` — Main GUI (Tkinter) app for selecting images and generating captions.
- `gui_template.py` — Optional GUI template demonstrating chatbot/translation integration.
- `chatbot.py` — Gemini/translation chatbot wrapper used by the GUI (requires API keys).
- `model.py`, `predict.py`, `preprocess.py`, `train.py`, `utils.py`, `config.py` — Core training/inference utilities.
- `extracted_features/` — placeholder for precomputed CNN features (not actual binaries by default).

Quickstart (run locally)
1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Tkinter GUI:

```powershell
python app.py
```

Notes and requirements
- This repo does NOT include large binary artifacts: trained model weights, tokenizer pickles, and full extracted feature pickles are not bundled. Place real files under `saved_model/` and `extracted_features/` to enable inference and training.
- The GUI supports optional features (chatbot, translation, speak); to use them set `GOOGLE_API_KEY` (for Gemini) and install any additional optional packages.
- If you prefer a web UI, `gui_template.py` illustrates how a chat/caption integration might work.

Troubleshooting
- If `generate_caption()` fails, ensure `saved_model/caption_model.keras` and `saved_model/tokenizer.pkl` exist and `config.py` paths match your files.
- For model training, follow `preprocess.py` → `train.py` workflows and ensure `extracted_features/` contains real feature pickles.

Licensing & data
- This copy is a trimmed project snapshot for development and testing. Data and pre-trained weights must be supplied by the user.
