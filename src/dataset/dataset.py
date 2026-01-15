import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from scipy import signal
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


def collate_fn_keep_captions(batch):
    # 张量/数值用默认逻辑堆叠
    eeg_data = default_collate([b["eeg_data"] for b in batch])  # [B, ...]
    class_label = default_collate([b["class_label"] for b in batch])  # [B]
    word = default_collate([b["word"] for b in batch])  # [B]
    image_id = default_collate([b["image_id"] for b in batch])  # [B]
    img_path = [b["img_path"] for b in batch]  # list[str], 长度 B
    class_text_embeddings = [b.get("class_text_embedding") for b in batch]
    has_class_text_embeddings = any(emb is not None for emb in class_text_embeddings)
    if has_class_text_embeddings and not all(
        emb is not None for emb in class_text_embeddings
    ):
        raise ValueError("Batch has inconsistent class text embedding availability.")

    # 关键：caption 不再让 default_collate 处理，保持为 list[list[str]]，形状 [B, K]
    captions = [b["caption"] for b in batch]  # [[str]*K]*B
    caption_embeddings = [b.get("caption_embeddings") for b in batch]
    has_caption_embeddings = any(emb is not None for emb in caption_embeddings)
    if has_caption_embeddings and not all(
        emb is not None for emb in caption_embeddings
    ):
        raise ValueError("Batch has inconsistent caption embedding availability.")

    image_embeddings = [b.get("image_embedding") for b in batch]
    has_image_embeddings = any(emb is not None for emb in image_embeddings)
    if has_image_embeddings and not all(emb is not None for emb in image_embeddings):
        raise ValueError("Batch has inconsistent image embedding availability.")

    collated = {
        "eeg_data": eeg_data,
        "class_label": class_label,
        "word": word,
        "image_id": image_id,
        "img_path": img_path,
        "caption": captions,  # [B, K]
    }
    if has_class_text_embeddings:
        collated["class_text_embedding"] = default_collate(class_text_embeddings)
    if has_caption_embeddings:
        collated["caption_embeddings"] = caption_embeddings
    if has_image_embeddings:
        collated["image_embedding"] = default_collate(image_embeddings)
    return collated


class EEGDataset(Dataset):
    def __init__(self, config: Dict, split: str = "train"):
        """
        Args:
            config: dataset section from the YAML config.
            split: one of {"train","val","all"}.
        """

        if split not in {"train", "val", "all"}:
            raise ValueError(f"split={split} not in {{'train','val','all'}}")

        self.root = config["root"]
        splits = config.get("splits", {})
        try:
            data_filename = splits[split]
        except KeyError as exc:
            raise KeyError(f"Split '{split}' not found in config['splits']") from exc

        captions_cfg = config.get("captions", {})
        self.captions_per_sample = max(1, int(captions_cfg.get("per_sample", 1)))
        self.sample_captions_randomly = captions_cfg.get("random", True)
        embedding_file = captions_cfg.get("embedding_file")
        self.caption_embeddings: Optional[Dict[str, torch.Tensor]] = None
        self.caption_embedding_index: Dict[str, List[int]] = {}
        self.image_embeddings: Optional[Dict[str, torch.Tensor]] = None
        self.class_text_embeddings: Optional[Dict[str, torch.Tensor]] = None

        eeg_cfg = config.get("eeg", {})
        self.sample_rate = int(eeg_cfg.get("sample_rate", 1000))
        crop = eeg_cfg.get("crop", [0, None])
        self.crop_start = int(crop[0]) if crop and crop[0] is not None else 0
        self.crop_end = int(crop[1]) if crop and crop[1] is not None else None

        filter_cfg = eeg_cfg.get("filter", {})
        self.filter_order = int(filter_cfg.get("order", 2))
        self.filter_low = float(filter_cfg.get("low_hz", 1.0))
        self.filter_high = float(filter_cfg.get("high_hz", 70.0))
        self.filter_enabled = bool(filter_cfg.get("enabled", True))

        caption_path = os.path.join(self.root, "caption", captions_cfg.get("file", ""))
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file not found: {caption_path}")

        with open(caption_path, "r", encoding="utf-8") as fp:
            self.captions = json.load(fp)

        data_path = os.path.join(self.root, "eeg_data", data_filename)
        data = torch.load(data_path, map_location="cpu")
        self.dataset: Sequence[Dict] = data["dataset"]
        self.labels: List[str] = data["labels"]
        self.image_ids: List[str] = data["images"]
        self.labels_to_words: Dict[str, str] = data["labels_to_words"]

        self.label_to_index: Dict[str, int] = {
            label: idx for idx, label in enumerate(self.labels)
        }
        self.image_to_index: Dict[str, int] = {
            image_id: idx for idx, image_id in enumerate(self.image_ids)
        }

        if embedding_file:
            embedding_path = Path(embedding_file)
            if not embedding_path.is_absolute():
                embedding_path = Path(self.root) / embedding_path
            if not embedding_path.exists():
                raise FileNotFoundError(
                    f"Caption embedding file not found: {embedding_path}"
                )
            payload = torch.load(embedding_path, map_location="cpu")

            caption_embeddings = payload.get("caption_embeddings")
            if isinstance(caption_embeddings, dict):
                self.caption_embeddings = {
                    key: value.to(dtype=torch.float16)
                    for key, value in caption_embeddings.items()
                    if isinstance(value, torch.Tensor)
                }
            elif isinstance(caption_embeddings, torch.Tensor):
                caption_image_paths = payload.get("caption_image_paths")
                caption_mean_embeddings = payload.get("caption_mean_embeddings")
                if not isinstance(caption_image_paths, Sequence):
                    raise ValueError(
                        "caption_image_paths must be provided when caption_embeddings is a tensor."
                    )
                if isinstance(caption_image_paths, (str, bytes)):
                    raise ValueError(
                        "caption_image_paths must be a sequence of strings."
                    )
                self.caption_embedding_index = {}
                for idx, path in enumerate(caption_image_paths):
                    self.caption_embedding_index.setdefault(path, []).append(idx)

            image_embeddings = payload.get("image_embeddings")
            if isinstance(image_embeddings, dict):
                self.image_embeddings = {
                    key: value.to(dtype=torch.float16)
                    for key, value in image_embeddings.items()
                    if isinstance(value, torch.Tensor)
                }
            elif isinstance(image_embeddings, torch.Tensor):
                image_paths = payload.get("image_paths")
                if not isinstance(image_paths, Sequence):
                    raise ValueError(
                        "image_paths must be provided when image_embeddings is a tensor."
                    )
                if isinstance(image_paths, (str, bytes)):
                    raise ValueError("image_paths must be a sequence of strings.")

            class_text_embeddings = payload.get("class_text_embeddings")
            if isinstance(class_text_embeddings, dict):
                self.class_text_embeddings = {
                    key: value.to(dtype=torch.float16)
                    for key, value in class_text_embeddings.items()
                    if isinstance(value, torch.Tensor)
                }
            elif isinstance(class_text_embeddings, torch.Tensor):
                class_labels = payload.get("class_labels")
                if not isinstance(class_labels, Sequence):
                    raise ValueError(
                        "class_labels must be provided when class_text_embeddings is a tensor."
                    )
                if isinstance(class_labels, (str, bytes)):
                    raise ValueError("class_labels must be a sequence of strings.")

    def __len__(self):
        return len(self.dataset)

    def filter(self, eeg_data, hz, low_f, high_f):
        b, a = signal.butter(
            self.filter_order, [low_f * 2 / hz, high_f * 2 / hz], "bandpass"
        )
        eeg_data = signal.lfilter(b, a, eeg_data).copy()
        eeg_data = torch.from_numpy(eeg_data).float()
        return eeg_data

    def __getitem__(self, idx, hz: Optional[int] = None):
        data_item = self.dataset[idx]
        eeg_data = data_item["eeg_data"]
        eeg_data = eeg_data[:, self.crop_start : self.crop_end]

        effective_hz = hz or self.sample_rate
        eeg_data = (
            self.filter(eeg_data, effective_hz, self.filter_low, self.filter_high)
            if self.filter_enabled
            else eeg_data
        )

        label_name = data_item["label"]
        label = self.label_to_index[label_name]
        word = self.labels_to_words[label_name]

        image_stem = os.path.splitext(data_item["image"])[0]
        image_id = self.image_to_index[image_stem]

        img_name = data_item["image"]
        image_path = os.path.join(self.root, "images", str(label_name), img_name)
        full_captions: List[str] = self.captions.get(image_path, [])
        if not full_captions:
            raise KeyError(f"No captions found for image path: {image_path}")

        total_captions = len(full_captions)
        if self.captions_per_sample >= total_captions:
            selected_indices = list(range(total_captions))
        elif self.sample_captions_randomly:
            selected_indices = random.sample(
                range(total_captions), self.captions_per_sample
            )
        else:
            selected_indices = [0]

        captions = [full_captions[idx] for idx in selected_indices]

        item = {
            "eeg_data": eeg_data,
            "caption": captions,
            "class_label": label,
            "image_id": image_id,
            "img_path": image_path,
            "word": word,
        }

        if self.caption_embeddings is not None:
            embeddings_for_image = self.caption_embeddings.get(image_path)
            if embeddings_for_image is None:
                raise KeyError(
                    f"No caption embeddings found for image path: {image_path}"
                )
            if embeddings_for_image.size(0) != len(full_captions):
                raise ValueError(
                    f"Mismatch between captions and embeddings for {image_path}: "
                    f"{len(full_captions)} captions vs {embeddings_for_image.size(0)} embeddings."
                )
            index_tensor = torch.tensor(selected_indices, dtype=torch.long)
            selected_embeddings = embeddings_for_image.index_select(0, index_tensor).to(
                dtype=torch.float32
            )
            item["caption_embeddings"] = selected_embeddings

        if self.image_embeddings is not None:
            image_embedding = self.image_embeddings.get(image_path)
            if image_embedding is None:
                raise KeyError(f"No image embedding found for image path: {image_path}")
            item["image_embedding"] = image_embedding.to(dtype=torch.float32)

        if self.class_text_embeddings is not None:
            class_embedding = self.class_text_embeddings.get(label_name)
            if class_embedding is None:
                raise KeyError(
                    f"No class text embedding found for class label: {label_name}"
                )
            item["class_text_embedding"] = class_embedding.to(dtype=torch.float32)

        return item


from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Subset

if __name__ == "__main__":
    example_config = {
        "root": "/home/chengwenjie/datasets/40classes-50images",
        "splits": {"train": "train.pth", "val": "val.pth", "all": "dataset.pth"},
        "captions": {
            "file": "qwen3vl_multi_caption.json",
            "per_sample": 1,
            "random": True,
        },
        "eeg": {
            "sample_rate": 1000,
            "crop": [20, 460],
            "filter": {"order": 2, "low_hz": 1.0, "high_hz": 70.0},
        },
    }
    dataset = EEGDataset(example_config, split="train")
    indices = torch.load(
        os.path.join(example_config["root"], "eeg_data", "indices.pth"),
        map_location="cpu",
    )
    train_dataset = Subset(dataset, indices["train"])
    eval_dataset = Subset(dataset, indices["eval"])
    dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_keep_captions,
    )
    for item in tqdm(dataloader):
        pass
