# Image Captioning System

This repository contains the main components for an Image Captioning System that uses a CNN encoder (ResNet50/VGG16) and an LSTM decoder to generate captions for images.

Contents:
- `app.py` — Streamlit-based app to upload images and generate captions.
- `model.py` — CNN encoder and caption model definition.
- `predict.py` — Utilities to load a saved model and generate captions for an image.
- `preprocess.py` — Tokenizer building and preprocessing helpers for datasets.
- `train.py` — Training loop and utilities to train the captioning model.
- `utils.py` — Small helpers for JSON/pickle IO.
- `config.py` — Project configuration and dataset registry.
- `requirements.txt` — Python dependencies.

Notes:
- This repository is a focused subset of a larger project. Dataset files, saved model weights, and extracted features are not included here and should be placed under the `dataset/` and `saved_model/` directories respectively.
- To run the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py
```
