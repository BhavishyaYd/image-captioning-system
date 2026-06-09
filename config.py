from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root_dir: Path
    images_dir: Path
    captions_path: Path
    captions_format: str


@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str = "resnet50"  # Supported: resnet50, vgg16
    decoder_name: str = "lstm"      # Base phase decoder
    embedding_dim: int = 256
    lstm_units: int = 256
    dropout_rate: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    early_stopping_patience: int = 4


@dataclass(frozen=True)
class RuntimeConfig:
    max_vocab_size: int = 10000
    oov_token: str = "<unk>"
    start_token: str = "startseq"
    end_token: str = "endseq"
    min_word_frequency: int = 1


# Change this single value to switch datasets globally.
ACTIVE_DATASET = "flickr8k"  # Options: flickr8k, flickr30k, mscoco


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "flickr8k": DatasetConfig(
        name="flickr8k",
        root_dir=BASE_DIR / "dataset" / "Flickr8k",
        images_dir=BASE_DIR / "dataset" / "Flickr8k" / "Images",
        captions_path=BASE_DIR / "dataset" / "Flickr8k" / "captions.txt",
        captions_format="flickr_token",
    ),
    "flickr30k": DatasetConfig(
        name="flickr30k",
        root_dir=BASE_DIR / "dataset" / "Flickr30k",
        images_dir=BASE_DIR / "dataset" / "Flickr30k" / "Images",
        captions_path=BASE_DIR / "dataset" / "Flickr30k" / "captions.txt",
        captions_format="flickr_token",
    ),
    "mscoco": DatasetConfig(
        name="mscoco",
        root_dir=BASE_DIR / "dataset" / "MSCOCO",
        images_dir=BASE_DIR / "dataset" / "MSCOCO" / "train2017",
        captions_path=BASE_DIR / "dataset" / "MSCOCO" / "captions_train2017.json",
        captions_format="coco_json",
    ),
}


MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
RUNTIME_CONFIG = RuntimeConfig()


SAVED_MODEL_DIR = BASE_DIR / "saved_model"
EXTRACTED_FEATURES_DIR = BASE_DIR / "extracted_features"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"


TOKENIZER_PATH = SAVED_MODEL_DIR / "tokenizer.pkl"
PREPROCESS_METADATA_PATH = SAVED_MODEL_DIR / "preprocess_metadata.json"
MODEL_PATH = SAVED_MODEL_DIR / "caption_model.keras"


def get_dataset_config(dataset_name: str | None = None) -> DatasetConfig:
    name = (dataset_name or ACTIVE_DATASET).lower()
    if name not in DATASET_REGISTRY:
        allowed = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unsupported dataset '{name}'. Allowed: {allowed}")
    return DATASET_REGISTRY[name]


def get_split_ratios() -> Tuple[float, float, float]:
    train_ratio = 1.0 - TRAINING_CONFIG.validation_split - TRAINING_CONFIG.test_split
    if train_ratio <= 0:
        raise ValueError("Invalid split values. Train ratio must be > 0.")
    return train_ratio, TRAINING_CONFIG.validation_split, TRAINING_CONFIG.test_split


def ensure_project_structure() -> None:
    required_dirs = [
        BASE_DIR / "dataset",
        BASE_DIR / "dataset" / "Flickr8k",
        BASE_DIR / "dataset" / "Flickr30k",
        BASE_DIR / "dataset" / "MSCOCO",
        SAVED_MODEL_DIR,
        EXTRACTED_FEATURES_DIR,
        OUTPUTS_DIR,
        STATIC_DIR,
    ]
    for path in required_dirs:
        path.mkdir(parents=True, exist_ok=True)


ensure_project_structure()
