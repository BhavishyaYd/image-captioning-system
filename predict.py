from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import MODEL_CONFIG, MODEL_PATH, PREPROCESS_METADATA_PATH
from model import _load_image_for_encoder, build_cnn_encoder
from preprocess import load_tokenizer
from utils import load_json


def _id_to_word(integer: int, tokenizer) -> str | None:
    return tokenizer.index_word.get(integer)


def _greedy_search(model, feature_vector, tokenizer, max_length: int) -> str:
    in_text = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding="post")

        yhat = model.predict([feature_vector, seq], verbose=0)
        next_id = int(np.argmax(yhat))
        word = _id_to_word(next_id, tokenizer)

        if word is None:
            break
        in_text += f" {word}"

        if word == "endseq":
            break

    words = [w for w in in_text.split() if w not in {"startseq", "endseq"}]
    return " ".join(words)


def generate_caption(image_path: Path, model_path: Path | None = None) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    metadata = load_json(PREPROCESS_METADATA_PATH)
    max_length = int(metadata["max_length"])

    tokenizer = load_tokenizer()
    model = load_model(model_path or MODEL_PATH)

    encoder_name = MODEL_CONFIG.encoder_name.lower()
    encoder = build_cnn_encoder(encoder_name)

    image_tensor = _load_image_for_encoder(image_path, encoder_name)
    feature_vector = encoder.predict(image_tensor, verbose=0)

    caption = _greedy_search(model, feature_vector, tokenizer, max_length)
    return caption
