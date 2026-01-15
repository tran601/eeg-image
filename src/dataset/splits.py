import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import CLIPTextEncoder


def normalize_image_key(name: str) -> str:
    cleaned = name.strip().replace("\\", "/")
    cleaned = cleaned.split("/")[-1]
    cleaned = cleaned.lstrip(".")
    if "." in cleaned:
        cleaned = cleaned[: cleaned.rfind(".")]
    return cleaned


def split_images_by_label(
    images: Sequence[str], train_ratio: float = 0.8
) -> Tuple[Set[str], Set[str]]:
    label_to_images: Dict[str, List[str]] = defaultdict(list)
    for image in images:
        label = image.split("_", 1)[0]
        label_to_images[label].append(image)

    train_images: Set[str] = set()
    val_images: Set[str] = set()

    for items in label_to_images.values():
        if not items:
            continue
        split_index = int(len(items) * train_ratio)
        if split_index <= 0:
            split_index = 1
        if split_index >= len(items) and len(items) > 1:
            split_index = len(items) - 1

        train_images.update(items[:split_index])
        val_images.update(items[split_index:])

    return train_images, val_images


def build_split_payload(
    dataset: Sequence[dict],
    train_images: Iterable[str],
    val_images: Iterable[str],
) -> Tuple[dict, dict, Dict[str, List[str]]]:
    missing_dataset_images: List[str] = []
    unmatched_records: List[str] = []

    train_key_set: Set[str] = {normalize_image_key(image) for image in train_images}
    val_key_set: Set[str] = {normalize_image_key(image) for image in val_images}

    train_dataset: List[dict] = []
    val_dataset: List[dict] = []
    train_dataset_keys: Set[str] = set()
    val_dataset_keys: Set[str] = set()

    for record in dataset:
        record_key = normalize_image_key(record["image"])
        if record_key in train_key_set:
            train_dataset.append(record)
            train_dataset_keys.add(record_key)
        elif record_key in val_key_set:
            val_dataset.append(record)
            val_dataset_keys.add(record_key)
        else:
            unmatched_records.append(record["image"])

    train_images_list: List[str] = []
    for image in train_images:
        normalized = normalize_image_key(image)
        if normalized not in train_dataset_keys:
            missing_dataset_images.append(f"train:{image}")
            continue
        train_images_list.append(image)

    val_images_list: List[str] = []
    for image in val_images:
        normalized = normalize_image_key(image)
        if normalized not in val_dataset_keys:
            missing_dataset_images.append(f"val:{image}")
            continue
        val_images_list.append(image)

    missing_info = {
        "split_images_without_dataset": missing_dataset_images,
        "dataset_records_without_split": unmatched_records,
    }

    return train_dataset, val_dataset, missing_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dataset into train and validation sets by label."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "/home/chengwenjie/datasets/40classes-50images/eeg_data/dataset.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/home/chengwenjie/datasets/40classes-50images/eeg_data",
        help="Directory to store train.pth and val.pth (defaults to dataset directory).",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train split ratio per label."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")

    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    data = torch.load(dataset_path)
    dataset = data.get("dataset")
    images = data.get("images")
    labels = data.get("labels")

    if dataset is None or images is None:
        raise KeyError("dataset.pth must contain 'dataset' and 'images' keys.")

    train_image_set, val_image_set = split_images_by_label(images, args.train_ratio)

    train_images_ordered = [image for image in images if image in train_image_set]
    val_images_ordered = [image for image in images if image in val_image_set]

    train_dataset, val_dataset, missing_info = build_split_payload(
        dataset, train_images_ordered, val_images_ordered
    )
    train_payload = {"dataset": train_dataset, "images": images, "labels": labels}
    val_payload = {"dataset": val_dataset, "images": images, "labels": labels}

    output_dir = args.output_dir or dataset_path.parent
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_payload, output_dir / "train.pth")
    torch.save(val_payload, output_dir / "val.pth")

    print(f"Saved train split with {len(train_payload['images'])} images.")
    print(f"Saved val split with {len(val_payload['images'])} images.")
    missing_dataset = missing_info["split_images_without_dataset"]
    unmatched_records = missing_info["dataset_records_without_split"]
    if missing_dataset:
        print(
            f"Warning: {len(missing_dataset)} split images skipped due to missing dataset entries."
        )
        for name in missing_dataset[:5]:
            print(f"  - {name}")
        if len(missing_dataset) > 5:
            print("  ...")
    if unmatched_records:
        print(
            f"Warning: {len(unmatched_records)} dataset records were not assigned to any split."
        )
        for name in unmatched_records[:5]:
            print(f"  - {name}")
        if len(unmatched_records) > 5:
            print("  ...")


if __name__ == "__main__":
    main()
