from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    EXTRACTED_FEATURES_DIR,
    MODEL_CONFIG,
    MODEL_PATH,
    PREPROCESS_METADATA_PATH,
    TRAINING_CONFIG,
    get_dataset_config,
)
from model import ENCODER_OUTPUT_DIM, build_caption_model, extract_features_from_directory
from preprocess import load_image_captions, load_tokenizer, split_image_ids
from utils import load_json, load_pickle, save_pickle


def _feature_file_path(dataset_name: str) -> Path:
    return EXTRACTED_FEATURES_DIR / f"{dataset_name}_{MODEL_CONFIG.encoder_name}_features.pkl"


def prepare_features(dataset_name: str) -> Dict[str, np.ndarray]:
    dataset_cfg = get_dataset_config(dataset_name)
    feature_path = _feature_file_path(dataset_name)

    if feature_path.exists():
        return load_pickle(feature_path)

    if not dataset_cfg.images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {dataset_cfg.images_dir}. "
            "Place dataset images and retry."
        )

    features = extract_features_from_directory(
        image_dir=dataset_cfg.images_dir,
        encoder_name=MODEL_CONFIG.encoder_name,
    )
    save_pickle(features, feature_path)
    return features


def data_generator(
    captions_map: Dict[str, List[str]],
    features: Dict[str, np.ndarray],
    tokenizer,
    max_length: int,
    batch_size: int,
) -> Generator[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray], None, None]:
    image_ids = list(captions_map.keys())

    while True:
        np.random.shuffle(image_ids)

        x_img, x_seq, y = [], [], []
        for image_id in image_ids:
            if image_id not in features:
                continue
            image_feature = features[image_id]

            for caption in captions_map[image_id]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]

                    x_img.append(image_feature)
                    x_seq.append(in_seq)
                    y.append(out_seq)

                    if len(x_img) == batch_size:
                        yield (np.array(x_img), np.array(x_seq)), np.array(y)
                        x_img, x_seq, y = [], [], []

        if x_img:
            yield (np.array(x_img), np.array(x_seq)), np.array(y)


def train_model(dataset_name: str | None = None) -> None:
    dataset_cfg = get_dataset_config(dataset_name)
    if not PREPROCESS_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing preprocessing metadata at {PREPROCESS_METADATA_PATH}. "
            "Run preprocess.py first."
        )
    metadata = load_json(PREPROCESS_METADATA_PATH)
    tokenizer = load_tokenizer()

    captions_map = load_image_captions(dataset_cfg.name)
    splits = split_image_ids(captions_map)

    train_caps = {k: captions_map[k] for k in splits["train"]}
    val_caps = {k: captions_map[k] for k in splits["val"]}

    features = prepare_features(dataset_cfg.name)

    max_length = int(metadata["max_length"])
    vocab_size = int(metadata["vocab_size"])
    feature_dim = ENCODER_OUTPUT_DIM[MODEL_CONFIG.encoder_name.lower()]

    model = build_caption_model(vocab_size=vocab_size, max_length=max_length, feature_dim=feature_dim)

    train_gen = data_generator(train_caps, features, tokenizer, max_length, TRAINING_CONFIG.batch_size)
    val_gen = data_generator(val_caps, features, tokenizer, max_length, TRAINING_CONFIG.batch_size)

    steps_train = max(1, sum(len(v) for v in train_caps.values()) // TRAINING_CONFIG.batch_size)
    steps_val = max(1, sum(len(v) for v in val_caps.values()) // TRAINING_CONFIG.batch_size)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=TRAINING_CONFIG.early_stopping_patience, restore_best_weights=True),
        ModelCheckpoint(filepath=str(MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_train,
        validation_steps=steps_val,
        epochs=TRAINING_CONFIG.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print("Training completed and model saved.")
    print(json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2))


if __name__ == "__main__":
    train_model()
