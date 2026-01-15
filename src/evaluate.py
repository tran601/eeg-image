import argparse
import random
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from tqdm import tqdm
import lpips

from dataset import EEG40Dataset, collate_fn_keep_captions
from models import EEGEncoder, StableDiffusionBridge
from utils import setup_logger


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    return tensor.clamp(0.0, 1.0)


def pil_to_lpips_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = pil_to_tensor(image, device)
    return tensor.mul(2.0).sub(1.0)


def tensor_to_uint8(image_tensor: torch.Tensor) -> torch.Tensor:
    return image_tensor.mul(255.0).clamp(0.0, 255.0).to(torch.uint8)


TARGET_IMAGE_SIZE: Tuple[int, int] = (256, 256)

SWEEP_VARIANTS: Sequence[Dict[str, Any]] = [
    {
        "name": "ip_adapter_on_scale_0_3_text_on",
        "overrides": {
            "stable_diffusion": {
                "ip_adapter": {"enabled": True},
                "ip_adapter_scale": 0.3,
                "text_enabled": True,
            }
        },
    },
    {
        "name": "ip_adapter_on_scale_0_5_text_on",
        "overrides": {
            "stable_diffusion": {
                "ip_adapter": {"enabled": True},
                "ip_adapter_scale": 0.5,
                "text_enabled": True,
            }
        },
    },
    {
        "name": "ip_adapter_on_scale_1_0_text_off",
        "overrides": {
            "stable_diffusion": {
                "ip_adapter": {"enabled": True},
                "ip_adapter_scale": 1.0,
                "text_enabled": False,
            }
        },
    },
    {
        "name": "ip_adapter_off_scale_0_0_text_on",
        "overrides": {
            "stable_diffusion": {
                "ip_adapter": {"enabled": False},
                "ip_adapter_scale": 0.0,
                "text_enabled": True,
            }
        },
    },
]


def resize_image(image: Image.Image) -> Image.Image:
    if image.size == TARGET_IMAGE_SIZE:
        return image
    return image.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)


class Evaluator:
    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        text_ckpt: Path,
        image_ckpt: Path,
        output_dir: Path,
    ) -> None:
        if isinstance(config, (str, Path)):
            with open(config, "r") as fp:
                self.config = yaml.safe_load(fp)
        elif isinstance(config, dict):
            self.config = deepcopy(config)
        else:
            raise TypeError(
                "config must be a path-like object or a configuration dictionary."
            )
        self.device = torch.device(self.config.get("device", "cuda"))

        eval_cfg = self.config.get("evaluation", {})
        self.num_samples = int(eval_cfg.get("num_samples", 200))
        self.batch_size = int(eval_cfg.get("batch_size", 1))
        self.num_generations_per_eeg = int(eval_cfg.get("num_generations_per_eeg", 1))
        self.visualization_interval = int(eval_cfg.get("visualization_interval", 50))
        self.visualization_groups = int(eval_cfg.get("visualization_groups", 8))
        self.class_alpha = eval_cfg["class_alpha"]
        self.visualization_cols = 2
        self.visualization_rows = max(
            1,
            (self.visualization_groups + self.visualization_cols - 1)
            // self.visualization_cols,
        )
        self.output_dir = output_dir
        log_cfg = self.config.get("logging", {})
        log_cfg["dir"] = "evaluation"

        self.logger, self.log_dir = setup_logger(
            name="Evaluator",
            config=self.config.get("logging", {}),
            run_name=f"{self.output_dir}",
        )
        self.logger.info("Config:\n%s", json.dumps(self.config, indent=4))
        self.visualization_dir = self.log_dir / "visualizations"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = EEG40Dataset(split="val")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config["training"]["text"].get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn_keep_captions,
        )

        self.sd_bridge = StableDiffusionBridge(self.config["stable_diffusion"])

        self.text_encoder = EEGEncoder(self.config["model"]["eeg_encoder"]).to(
            self.device
        )
        self.image_encoder = EEGEncoder(self.config["model"]["eeg_encoder"]).to(
            self.device
        )

        self._load_checkpoint(self.text_encoder, text_ckpt)
        self._load_checkpoint(self.image_encoder, image_ckpt)

        self.text_encoder.eval()
        self.image_encoder.eval()

        self.inception = InceptionScore().to(self.device)
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        lpips_net = self.config["evaluation"]["metrics"].get("lpips_net", "alex")
        self.lpips = lpips.LPIPS(net=lpips_net).to(self.device)
        metrics_cfg = self.config["evaluation"]["metrics"]
        clip_model_id = metrics_cfg.get("clip_model_path")
        self.clip_model, self.clip_processor = self._load_clip_model(clip_model_id)

    def _load_checkpoint(self, model: EEGEncoder, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device)
        missing, unexpected = model.load_state_dict(
            payload["model_state_dict"], strict=False
        )
        if missing:
            self.logger.warning(
                "Missing keys when loading %s: %s", checkpoint_path, missing
            )
        if unexpected:
            self.logger.warning(
                "Unexpected keys when loading %s: %s", checkpoint_path, unexpected
            )

    def _load_clip_model(self, model_path: str) -> Tuple[Any, Any]:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers package is required to compute CLIP scores. "
                "Install it via `pip install transformers`."
            ) from exc
        if not model_path:
            raise ValueError("A valid CLIP model path must be provided for evaluation.")
        model = CLIPModel.from_pretrained(model_path).to(self.device)
        processor = CLIPProcessor.from_pretrained(model_path)
        model.eval()
        return model, processor

    def encode_images_with_clip(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        inputs = self.clip_processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        return features

    def evaluate(self) -> Dict[str, float]:
        generated_images: List[Image.Image] = []
        ground_truth_images: List[Image.Image] = []
        clip_similarities: List[float] = []
        lpips_scores: List[float] = []
        ssim_scores: List[float] = []

        processed = 0
        eeg_samples_seen = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                eeg = batch["eeg_data"].to(self.device)
                captions = [caps[0] if caps else "" for caps in batch["caption"]]
                img_paths = batch["img_path"]

                text_embeds = self.text_encoder.encode_text(eeg)
                class_text_embeds = batch.get("class_text_embedding")
                if class_text_embeds is not None:
                    class_text_embeds = class_text_embeds.to(self.device)
                    text_embeds = (
                        1.0 - self.class_alpha
                    ) * text_embeds + self.class_alpha * class_text_embeds
                image_embeds = self.image_encoder.encode_image(eeg)

                for idx, img_path in enumerate(img_paths):
                    if processed >= self.num_samples:
                        break

                    text_embed = text_embeds[idx : idx + 1]
                    image_embed = image_embeds[idx : idx + 1]
                    gt_image = Image.open(img_path).convert("RGB")

                    gt_image_resized = resize_image(gt_image)
                    gt_tensor = pil_to_tensor(gt_image_resized, self.device)
                    gt_tensor_uint8 = tensor_to_uint8(gt_tensor)
                    gt_lpips = pil_to_lpips_tensor(gt_image_resized, self.device)
                    gt_clip = self.encode_images_with_clip([gt_image_resized])

                    per_eeg_count = 0
                    while per_eeg_count < self.num_generations_per_eeg:
                        if processed >= self.num_samples:
                            break

                        generated_batch = self.sd_bridge.generate(
                            text_embeddings=text_embed,
                            image_embeddings=image_embed,
                        )
                        if not generated_batch:
                            break
                        gen_image = generated_batch[0]
                        gen_image_resized = resize_image(gen_image)

                        generated_images.append(gen_image_resized)
                        ground_truth_images.append(gt_image_resized.copy())

                        gen_tensor = pil_to_tensor(gen_image_resized, self.device)
                        gen_tensor_uint8 = tensor_to_uint8(gen_tensor)
                        gen_lpips = pil_to_lpips_tensor(gen_image_resized, self.device)

                        self.inception.update(gen_tensor_uint8.unsqueeze(0))
                        self.fid.update(gt_tensor_uint8.unsqueeze(0), real=True)
                        self.fid.update(gen_tensor_uint8.unsqueeze(0), real=False)

                        ssim_val = structural_similarity_index_measure(
                            gen_tensor.unsqueeze(0),
                            gt_tensor.unsqueeze(0),
                            data_range=1.0,
                        )
                        ssim_scores.append(ssim_val.item())

                        lpips_val = self.lpips(
                            gen_lpips.unsqueeze(0), gt_lpips.unsqueeze(0)
                        )
                        lpips_scores.append(lpips_val.item())

                        gen_clip = self.encode_images_with_clip([gen_image_resized])
                        clip_sim = F.cosine_similarity(gen_clip, gt_clip, dim=-1)
                        clip_similarities.append(clip_sim.item())

                        per_eeg_count += 1
                        processed += 1

                    gt_image.close()
                    eeg_samples_seen += 1

                    if (
                        self.visualization_interval > 0
                        and eeg_samples_seen % self.visualization_interval == 0
                    ):
                        self._visualize_samples(
                            generated_images,
                            ground_truth_images,
                            eeg_samples_seen,
                        )
                        generated_images: List[Image.Image] = []
                        ground_truth_images: List[Image.Image] = []

                if processed >= self.num_samples:
                    break

        is_mean, is_std = self.inception.compute()
        fid_score = self.fid.compute()

        metrics = {
            "inception_score_mean": float(is_mean),
            "inception_score_std": float(is_std),
            "fid": float(fid_score),
            "ssim": float(np.mean(ssim_scores)),
            "lpips": float(np.mean(lpips_scores)),
            "clip_cosine_similarity": float(np.mean(clip_similarities)),
            "num_images": processed,
        }

        metrics_json = json.dumps(metrics, indent=4)
        self.logger.info("Evaluation metrics:\n%s", metrics_json)

        return metrics

    def _visualize_samples(
        self,
        generated_images: Sequence[Image.Image],
        ground_truth_images: Sequence[Image.Image],
        step: int,
    ) -> None:
        required_groups = self.visualization_rows * self.visualization_cols
        if (
            self.visualization_groups <= 0
            or self.num_generations_per_eeg <= 0
            or required_groups <= 0
        ):
            return

        total_groups = len(generated_images) // self.num_generations_per_eeg
        if total_groups < required_groups:
            return

        import matplotlib.pyplot as plt

        try:
            selected_indices = random.sample(range(total_groups), required_groups)
        except ValueError:
            selected_indices = list(range(total_groups))[:required_groups]

        selected_indices.sort()

        unit_images: List[Image.Image] = []
        for group_idx in selected_indices:
            start = group_idx * self.num_generations_per_eeg
            gt_image = ground_truth_images[start]
            gens = generated_images[start : start + self.num_generations_per_eeg]
            unit_images.append(self._compose_unit_image(gt_image, gens))

        fig, axes = plt.subplots(
            self.visualization_rows,
            self.visualization_cols,
            figsize=(9, 6),
        )
        axes_iter = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, ax in enumerate(axes_iter):
            if idx < len(unit_images):
                ax.imshow(unit_images[idx])
                ax.axis("off")
            else:
                ax.axis("off")
        plt.tight_layout()

        viz_path = self.visualization_dir / f"step_{step:05d}.png"
        fig.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _compose_unit_image(
        ground_truth: Image.Image,
        generated_images: Sequence[Image.Image],
        padding: int = 4,
    ) -> Image.Image:
        images = [ground_truth] + list(generated_images)
        uniform_height = max(img.height for img in images)
        resized: List[Image.Image] = []
        for img in images:
            local_img = img if img.mode == "RGB" else img.convert("RGB")
            if local_img.height != uniform_height:
                width = int(local_img.width * (uniform_height / local_img.height))
                local_img = local_img.resize((width, uniform_height), Image.LANCZOS)
            resized.append(local_img)

        total_width = sum(img.width for img in resized) + padding * (len(resized) - 1)
        canvas = Image.new("RGB", (total_width, uniform_height), color=(0, 0, 0))

        offset = 0
        for idx, img in enumerate(resized):
            canvas.paste(img, (offset, 0))
            offset += img.width + padding

        return canvas


def _deep_update(target: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict):
            base_value = target.get(key, {})
            if not isinstance(base_value, dict):
                base_value = {}
            target[key] = _deep_update(base_value, value)
        else:
            target[key] = value
    return target


def _run_param_sweep(args: argparse.Namespace) -> None:
    with open(args.config, "r") as fp:
        base_config = yaml.safe_load(fp)

    sweep_metrics: Dict[str, Dict[str, float]] = {}
    for variant in SWEEP_VARIANTS:
        variant_config = deepcopy(base_config)
        _deep_update(variant_config, variant["overrides"])
        variant_output_dir = args.output_dir / variant["name"]
        evaluator = Evaluator(
            variant_config,
            args.text_checkpoint,
            args.image_checkpoint,
            variant_output_dir,
        )
        sweep_metrics[variant["name"]] = evaluator.evaluate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path("evaluation") / args.output_dir / "param_sweep_metrics.json"
    summary_path.write_text(json.dumps(sweep_metrics, indent=4))
    print(f"Parameter sweep summary written to {summary_path}")

    import gc

    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EEG-driven Stable Diffusion pipeline."
    )
    parser.add_argument("--config", type=str, default="configs/default_40.yaml")

    parser.add_argument(
        "--image-checkpoint",
        type=Path,
        default="checkpoints/image_path/image_path_infonce/epoch_240.pt",
    )
    parser.add_argument(
        "--text-checkpoint",
        type=Path,
        default="checkpoints/text_path/text_path_cosine/epoch_240.pt",
    )
    """
    i_info_t_avg
    i_info_t_eot
    i_cos_t_avg
    i_cos_t_eot
    i_cos_t_cos
    i_info_t_cos
    
    """
    parser.add_argument("--output-dir", type=Path, default=Path("i_info_t_cos"))
    parser.add_argument(
        "--param-sweep",
        default=True,
        help=(
            "Evaluate a fixed set of Stable Diffusion parameter combinations that "
            "toggle the IP-Adapter and text guidance."
        ),
    )
    args = parser.parse_args()

    if args.param_sweep:
        _run_param_sweep(args)
        return

    evaluator = Evaluator(
        args.config, args.text_checkpoint, args.image_checkpoint, args.output_dir
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
