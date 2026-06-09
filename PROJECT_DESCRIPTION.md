Image Captioning System
-----------------------

This repository contains a compact, working copy of an image captioning project focused on inference and a Tkinter GUI front end.

Key components and workflows
- Preprocessing: parse captions and build tokenizer (`preprocess.py`).
- Feature extraction: extract and save CNN features for images (`model.extract_features_from_directory`).
- Training: train the captioning model using extracted features (`train.py`).
- Inference: load a trained model + tokenizer and generate captions (`predict.py`).
- GUI: `app.py` launches a Tkinter interface for single- or multi-image captioning; `gui_template.py` demonstrates chat/translation integrations.

Where to put large assets
- `saved_model/` — place `caption_model.keras` and `tokenizer.pkl` here for inference.
- `extracted_features/` — put real feature pickles produced by your encoder here.
- `dataset/` — dataset images and caption files for preprocessing and training.

Notes
- This copy intentionally omits large binaries (model weights and full datasets). Add those files if you need to run inference/train locally.
- Optional chatbot features rely on external APIs (e.g., Gemini) and require API keys and optional dependencies.
- See `README.md` for quickstart and troubleshooting steps.
