from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, add

from config import MODEL_CONFIG


ENCODER_OUTPUT_DIM: Dict[str, int] = {
    "resnet50": 2048,
    "vgg16": 4096,
}


def build_cnn_encoder(encoder_name: str | None = None) -> Model:
    name = (encoder_name or MODEL_CONFIG.encoder_name).lower()

    if name == "resnet50":
        base_model = ResNet50(weights="imagenet")
        return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    if name == "vgg16":
        base_model = VGG16(weights="imagenet")
        return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    raise ValueError(f"Unsupported encoder: {name}")


def _load_image_for_encoder(image_path: Path, encoder_name: str) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    array = np.expand_dims(image.astype(np.float32), axis=0)

    if encoder_name == "resnet50":
        return resnet_preprocess(array)
    if encoder_name == "vgg16":
        return vgg_preprocess(array)
    raise ValueError(f"Unsupported encoder: {encoder_name}")


def extract_features_from_directory(image_dir: Path, encoder_name: str | None = None) -> Dict[str, np.ndarray]:
    name = (encoder_name or MODEL_CONFIG.encoder_name).lower()
    encoder_model = build_cnn_encoder(name)

    features: Dict[str, np.ndarray] = {}
    valid_ext = {".jpg", ".jpeg", ".png"}

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in valid_ext:
            continue
        image_tensor = _load_image_for_encoder(image_path, name)
        vector = encoder_model.predict(image_tensor, verbose=0)[0]
        features[image_path.name] = vector

    return features


def build_caption_model(vocab_size: int, max_length: int, feature_dim: int | None = None) -> Model:
    feature_dim = feature_dim or ENCODER_OUTPUT_DIM[MODEL_CONFIG.encoder_name.lower()]

    # Image feature branch
    inputs_img = Input(shape=(feature_dim,), name="image_features")
    x_img = Dropout(MODEL_CONFIG.dropout_rate)(inputs_img)
    x_img = Dense(MODEL_CONFIG.embedding_dim, activation="relu")(x_img)

    # Text sequence branch
    inputs_txt = Input(shape=(max_length,), name="input_sequence")
    x_txt = Embedding(vocab_size, MODEL_CONFIG.embedding_dim, mask_zero=True)(inputs_txt)
    x_txt = Dropout(MODEL_CONFIG.dropout_rate)(x_txt)
    x_txt = LSTM(MODEL_CONFIG.lstm_units)(x_txt)

    merged = add([x_img, x_txt])
    merged = Dense(MODEL_CONFIG.lstm_units, activation="relu")(merged)
    outputs = Dense(vocab_size, activation="softmax", name="next_word")(merged)

    model = Model(inputs=[inputs_img, inputs_txt], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
    )
    return model
