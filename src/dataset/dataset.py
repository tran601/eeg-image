import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import torch
from scipy import signal
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


def collate_fn_keep_captions(batch):
    collated = {"eeg_data": default_collate([item["eeg_data"] for item in batch])}
    for key in ("class_label", "image_id", "embedding"):
        if key in batch[0]:
            assert all(key in item for item in batch), f"batch missing key: {key}"
            collated[key] = default_collate([item[key] for item in batch])
    for key in ("word", "img_path", "caption"):
        if key in batch[0]:
            assert all(key in item for item in batch), f"batch missing key: {key}"
            collated[key] = [item[key] for item in batch]
    return collated


def _resolve_path(root: Path, path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _select_caption_indices(total: int, count: int, randomize: bool) -> List[int]:
    assert total >= count, "captions_per_sample must be <= available captions"
    if randomize:
        return random.sample(range(total), count)
    return list(range(count))


def _select_caption_and_embedding(
    full_captions: List[str],
    embeddings: Dict[str, torch.Tensor],
    embedding_type: str,
    captions_per_sample: int,
    image_key: str,
    label_name: str,
) -> Tuple[List[str], torch.Tensor]:
    if embedding_type == "caption_embeddings":
        indices = _select_caption_indices(
            len(full_captions), captions_per_sample, randomize=True
        )
        caption = [full_captions[i] for i in indices]
        embedding = embeddings[image_key][indices]
        return caption, embedding
    if embedding_type == "image_embeddings":
        indices = _select_caption_indices(
            len(full_captions), captions_per_sample, randomize=False
        )
        caption = [full_captions[i] for i in indices]
        embedding = embeddings[image_key]
        return caption, embedding
    indices = _select_caption_indices(
        len(full_captions), captions_per_sample, randomize=False
    )
    caption = [full_captions[i] for i in indices]
    embedding = embeddings[label_name]
    return caption, embedding


class EEG40Dataset(Dataset):
    DEFAULT_ROOT = Path("/home/chengwenjie/datasets/40classes-50images")
    DEFAULT_SPLITS = {"train": "train.pth", "val": "val.pth", "all": "dataset.pth"}
    DEFAULT_EMBEDDING_FILE = "embedding/embeddings.pt"
    DEFAULT_CROP = (20, 460)
    DEFAULT_SAMPLE_RATE = 1000
    DEFAULT_FILTER_ORDER = 2
    DEFAULT_FILTER_LOW = 1.0
    DEFAULT_FILTER_HIGH = 70.0
    DEFAULT_FILTER_ENABLED = True
    ALLOWED_EMBEDDINGS = (
        "caption_embeddings",
        "image_embeddings",
        "class_text_embeddings",
    )

    def __init__(
        self,
        split: str = "train",
        embedding_type: str = "caption_embeddings",
        captions_per_sample: int = 1,
    ) -> None:
        split_map = self.DEFAULT_SPLITS
        assert split in split_map, f"split must be one of {sorted(split_map)}"
        assert captions_per_sample > 0, "captions_per_sample must be > 0"
        assert (
            embedding_type in self.ALLOWED_EMBEDDINGS
        ), f"unsupported embedding_type: {embedding_type}"

        self.root = self.DEFAULT_ROOT
        self.embedding_type = embedding_type
        self.captions_per_sample = int(captions_per_sample)
        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self.crop_start = int(self.DEFAULT_CROP[0])
        self.crop_end = int(self.DEFAULT_CROP[1])
        self.filter_order = self.DEFAULT_FILTER_ORDER
        self.filter_low = self.DEFAULT_FILTER_LOW
        self.filter_high = self.DEFAULT_FILTER_HIGH
        self.filter_enabled = self.DEFAULT_FILTER_ENABLED

        data_path = self.root / "eeg_data" / split_map[split]
        data = torch.load(data_path, map_location="cpu")
        self.dataset: Sequence[Dict] = data["dataset"]
        self.labels: List[str] = data["labels"]
        self.image_ids: List[str] = data["images"]
        self.labels_to_words: Dict[str, str] = data["labels_to_words"]

        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.image_to_index = {
            image_id: idx for idx, image_id in enumerate(self.image_ids)
        }

        embedding_path = _resolve_path(self.root, self.DEFAULT_EMBEDDING_FILE)
        self._load_embeddings(embedding_path)

    def _load_embeddings(self, embedding_path: Path) -> None:
        payload = torch.load(embedding_path, map_location="cpu")
        self.captions = payload["caption_texts"]
        self.embeddings = payload[self.embedding_type]

    def __len__(self) -> int:
        return len(self.dataset)

    def filter(self, eeg_data, hz: int, low_f: float, high_f: float) -> torch.Tensor:
        b, a = signal.butter(
            self.filter_order, [low_f * 2 / hz, high_f * 2 / hz], "bandpass"
        )
        filtered = signal.lfilter(b, a, eeg_data).copy()
        return torch.as_tensor(filtered).float()

    def _prepare_eeg(self, eeg_data: torch.Tensor) -> torch.Tensor:
        eeg_data = eeg_data[:, self.crop_start : self.crop_end]
        if self.filter_enabled:
            return self.filter(
                eeg_data, self.sample_rate, self.filter_low, self.filter_high
            )
        return torch.as_tensor(eeg_data).float()

    def __getitem__(self, idx: int) -> Dict:
        data_item = self.dataset[idx]
        eeg_tensor = self._prepare_eeg(data_item["eeg_data"])

        label_name = data_item["label"]
        label = self.label_to_index[label_name]
        word = self.labels_to_words[label_name]

        image_name = data_item["image"]
        image_stem = Path(image_name).stem
        image_id = self.image_to_index[image_stem]

        image_path = self.root / "images" / str(label_name) / image_name
        image_key = str(image_path)
        full_captions: List[str] = self.captions[image_key]
        assert full_captions, f"missing captions for {image_key}"

        caption, embedding = _select_caption_and_embedding(
            full_captions,
            self.embeddings,
            self.embedding_type,
            self.captions_per_sample,
            image_key,
            label_name,
        )

        item = {
            "eeg_data": eeg_tensor,
            "caption": caption,
            "class_label": label,
            "image_id": image_id,
            "img_path": image_key,
            "word": word,
            "embedding": embedding,
        }

        return item


class EEG4Dataset(Dataset):
    DEFAULT_ROOT = Path("/home/chengwenjie/datasets/4classes-50images")
    DEFAULT_SPLITS = {"train": "train.pth", "val": "val.pth", "all": "dataset.pth"}
    DEFAULT_EMBEDDING_FILE = "embedding/embeddings.pt"
    ALLOWED_EMBEDDINGS = (
        "caption_embeddings",
        "image_embeddings",
        "class_text_embeddings",
    )

    def __init__(
        self,
        split: str = "train",
        embedding_type: str = "caption_embeddings",
        captions_per_sample: int = 1,
    ) -> None:
        split_map = self.DEFAULT_SPLITS
        assert split in split_map, f"split must be one of {sorted(split_map)}"
        assert captions_per_sample > 0, "captions_per_sample must be > 0"
        assert (
            embedding_type in self.ALLOWED_EMBEDDINGS
        ), f"unsupported embedding_type: {embedding_type}"

        self.root = self.DEFAULT_ROOT
        self.embedding_type = embedding_type
        self.captions_per_sample = int(captions_per_sample)

        data_path = self.root / "eeg_data" / split_map[split]
        data = torch.load(data_path, map_location="cpu")
        self.dataset: Sequence[Dict] = data["dataset"]
        self.labels: List[str] = data["labels"]
        self.image_ids: List[str] = data["images"]
        self.labels_to_words: Dict[str, str] = data["labels_to_words"]

        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.image_to_index = {
            image_id: idx for idx, image_id in enumerate(self.image_ids)
        }

        embedding_path = _resolve_path(self.root, self.DEFAULT_EMBEDDING_FILE)
        self._load_embeddings(embedding_path)

    def _load_embeddings(self, embedding_path: Path) -> None:
        payload = torch.load(embedding_path, map_location="cpu")
        self.captions = payload["caption_texts"]
        self.embeddings = payload[self.embedding_type]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        data_item = self.dataset[idx]
        eeg_tensor = data_item["eeg_data"]

        label_name = data_item["label"]
        label = self.label_to_index[label_name]
        word = self.labels_to_words[label_name]

        image_name = data_item["image"]
        image_stem = Path(image_name).stem
        image_id = self.image_to_index[image_stem]

        image_path = self.root / "images" / str(label_name) / image_name
        image_key = str(image_path)
        full_captions: List[str] = self.captions[image_key]
        assert full_captions, f"missing captions for {image_key}"

        caption, embedding = _select_caption_and_embedding(
            full_captions,
            self.embeddings,
            self.embedding_type,
            self.captions_per_sample,
            image_key,
            label_name,
        )

        item = {
            "eeg_data": eeg_tensor,
            "caption": caption,
            "class_label": label,
            "image_id": image_id,
            "img_path": image_key,
            "word": word,
            "embedding": embedding,
        }

        return item
