Image Captioning System
-----------------------

This project implements an image captioning pipeline combining a pretrained CNN encoder (ResNet50 or VGG16) to extract image features and an LSTM-based decoder to generate natural-language captions.

Primary workflows:
- Preprocessing: parse dataset captions, build tokenizer, save preprocessing metadata (`preprocess.py`).
- Feature extraction: use the CNN encoder to extract features from images and save them (`model.extract_features_from_directory`).
- Training: train the captioning model using extracted features and tokenizer (`train.py`).
- Inference: load the trained model and generate captions for new images (`predict.py`).
- Web UI: simple Streamlit app to upload images and receive generated captions (`app.py`).

Important paths:
- `dataset/` — place dataset images and captions here.
- `saved_model/` — trained model and tokenizer will be saved here.
- `extracted_features/` — cached image features.

This repository contains the core scripts only. For full dataset files, pre-trained weights, and additional utilities, see the original workspace.
