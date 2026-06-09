from __future__ import annotations

import json
import pickle
import random
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from config import (
    PREPROCESS_METADATA_PATH,
    RUNTIME_CONFIG,
    TOKENIZER_PATH,
    TRAINING_CONFIG,
    get_dataset_config,
    get_split_ratios,
)


CaptionMap = Dict[str, List[str]]


def _clean_caption_text(raw_caption: str) -> str:
    text = raw_caption.lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _add_boundary_tokens(caption: str) -> str:
    return f"{RUNTIME_CONFIG.start_token} {caption} {RUNTIME_CONFIG.end_token}"


def _parse_flickr_caption_line(line: str) -> Tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None
    if "\t" not in line:
        return None

    image_token, caption = line.split("\t", maxsplit=1)
    image_name = image_token.split("#", maxsplit=1)[0]
    return image_name, caption


def _load_flickr_captions(captions_path: Path) -> CaptionMap:
    captions_map: CaptionMap = defaultdict(list)
    with captions_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        handle.seek(0)

        # Support newer Kaggle Flickr files using CSV format: image,caption
        if first_line.lower().startswith("image,caption"):
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = (row.get("image") or "").strip()
                caption = (row.get("caption") or "").strip()
                if not image_name or not caption:
                    continue
                cleaned = _clean_caption_text(caption)
                if cleaned:
                    captions_map[image_name].append(_add_boundary_tokens(cleaned))
        else:
            for line in handle:
                parsed = _parse_flickr_caption_line(line)
                if parsed is None:
                    continue
                image_name, caption = parsed
                cleaned = _clean_caption_text(caption)
                if cleaned:
                    captions_map[image_name].append(_add_boundary_tokens(cleaned))
    return dict(captions_map)


def _load_coco_captions(captions_path: Path) -> CaptionMap:
    captions_map: CaptionMap = defaultdict(list)
    payload = json.loads(captions_path.read_text(encoding="utf-8"))

    image_id_to_name = {
        image["id"]: image["file_name"]
        for image in payload.get("images", [])
        if "id" in image and "file_name" in image
    }

    for ann in payload.get("annotations", []):
        image_id = ann.get("image_id")
        caption = ann.get("caption", "")
        file_name = image_id_to_name.get(image_id)
        if file_name is None:
            continue
        cleaned = _clean_caption_text(caption)
        if cleaned:
            captions_map[file_name].append(_add_boundary_tokens(cleaned))

    return dict(captions_map)


def load_image_captions(dataset_name: str | None = None) -> CaptionMap:
    dataset_cfg = get_dataset_config(dataset_name)
    if not dataset_cfg.captions_path.exists():
        raise FileNotFoundError(
            f"Captions file not found: {dataset_cfg.captions_path}. "
            "Place dataset files and retry."
        )

    if dataset_cfg.captions_format == "flickr_token":
        return _load_flickr_captions(dataset_cfg.captions_path)
    if dataset_cfg.captions_format == "coco_json":
        return _load_coco_captions(dataset_cfg.captions_path)

    raise ValueError(f"Unsupported captions format: {dataset_cfg.captions_format}")


def _filter_vocabulary(captions_map: CaptionMap) -> List[str]:
    if RUNTIME_CONFIG.min_word_frequency <= 1:
        return [cap for caps in captions_map.values() for cap in caps]

    counter = Counter()
    for caps in captions_map.values():
        for cap in caps:
            counter.update(cap.split())

    allowed = {word for word, freq in counter.items() if freq >= RUNTIME_CONFIG.min_word_frequency}

    filtered: List[str] = []
    for caps in captions_map.values():
        for cap in caps:
            kept = [token for token in cap.split() if token in allowed]
            filtered.append(" ".join(kept))
    return filtered


def build_tokenizer(captions_map: CaptionMap) -> Tokenizer:
    all_captions = _filter_vocabulary(captions_map)
    tokenizer = Tokenizer(num_words=RUNTIME_CONFIG.max_vocab_size, oov_token=RUNTIME_CONFIG.oov_token, filters="")
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def get_max_caption_length(captions_map: CaptionMap) -> int:
    if not captions_map or all(len(caps) == 0 for caps in captions_map.values()):
        raise ValueError(
            "No captions found after parsing/cleaning. "
            "Verify captions file format and dataset paths in config.py."
        )
    return max(len(caption.split()) for caps in captions_map.values() for caption in caps)


def split_image_ids(captions_map: CaptionMap) -> Dict[str, List[str]]:
    image_ids = list(captions_map.keys())
    random.Random(TRAINING_CONFIG.random_seed).shuffle(image_ids)

    train_ratio, val_ratio, _ = get_split_ratios()
    train_end = int(len(image_ids) * train_ratio)
    val_end = train_end + int(len(image_ids) * val_ratio)

    return {
        "train": image_ids[:train_end],
        "val": image_ids[train_end:val_end],
        "test": image_ids[val_end:],
    }


def subset_captions(captions_map: CaptionMap, image_ids: List[str]) -> CaptionMap:
    return {img_id: captions_map[img_id] for img_id in image_ids if img_id in captions_map}


def create_sequence_sample(
    tokenizer: Tokenizer,
    caption: str,
    max_length: int,
    vocab_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = tokenizer.texts_to_sequences([caption])[0]
    x_seq, y_next = [], []

    for i in range(1, len(seq)):
        in_seq = seq[:i]
        out_seq = seq[i]
        x_seq.append(pad_sequences([in_seq], maxlen=max_length, padding="post")[0])
        y_next.append(out_seq)

    if not x_seq:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    x_image_dummy = np.zeros((len(x_seq), 1), dtype=np.float32)
    return x_image_dummy, np.array(x_seq), np.array(y_next)


def save_tokenizer(tokenizer: Tokenizer, tokenizer_path: Path = TOKENIZER_PATH) -> None:
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    with tokenizer_path.open("wb") as handle:
        pickle.dump(tokenizer, handle)


def load_tokenizer(tokenizer_path: Path = TOKENIZER_PATH) -> Tokenizer:
    with tokenizer_path.open("rb") as handle:
        return pickle.load(handle)


def save_preprocess_metadata(
    dataset_name: str,
    max_length: int,
    vocab_size: int,
    split_sizes: Dict[str, int],
    metadata_path: Path = PREPROCESS_METADATA_PATH,
) -> None:
    payload = {
        "dataset": dataset_name,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "split_sizes": split_sizes,
        "random_seed": TRAINING_CONFIG.random_seed,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_preprocessing(dataset_name: str | None = None) -> Dict[str, object]:
    dataset_cfg = get_dataset_config(dataset_name)
    captions_map = load_image_captions(dataset_cfg.name)

    tokenizer = build_tokenizer(captions_map)
    max_length = get_max_caption_length(captions_map)
    splits = split_image_ids(captions_map)

    vocab_size = min(RUNTIME_CONFIG.max_vocab_size, len(tokenizer.word_index) + 1)
    split_sizes = {k: len(v) for k, v in splits.items()}

    save_tokenizer(tokenizer)
    save_preprocess_metadata(dataset_cfg.name, max_length, vocab_size, split_sizes)

    return {
        "dataset": dataset_cfg.name,
        "total_images": len(captions_map),
        "vocab_size": vocab_size,
        "max_length": max_length,
        "splits": splits,
    }


if __name__ == "__main__":
    stats = run_preprocessing()
    print("Preprocessing complete")
    print(json.dumps({
        "dataset": stats["dataset"],
        "total_images": stats["total_images"],
        "vocab_size": stats["vocab_size"],
        "max_length": stats["max_length"],
        "split_sizes": {k: len(v) for k, v in stats["splits"].items()},
    }, indent=2))
