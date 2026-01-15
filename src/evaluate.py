import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from transformers import CLIPImageProcessor, CLIPModel

import lpips

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import EEG40Dataset, EEG4Dataset, collate_fn_keep_captions
from src.models.eeg_encoder import (
    EEGAlign,
    EEGBackbone,
    ImageVectorHead,
    LowRankTextHead,
    TextTokenHead,
)


DATASETS = {
    "eeg4": EEG4Dataset,
    "eeg40": EEG40Dataset,
}


def _make_loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_keep_captions,
    )


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Evaluate EEG-conditioned Stable Diffusion reconstructions."
    )
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="eeg40")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument(
        "--embedding_type",
        choices=["caption_embeddings", "image_embeddings", "class_text_embeddings"],
        default="image_embeddings",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")

    parser.add_argument("--high_ckpt", type=str, default=None)
    parser.add_argument("--low_ckpt", type=str, default=None)

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["high", "low", "both"],
        default=["high", "low", "both"],
    )

    parser.add_argument("--feature_dim", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temporal_conv", type=str, required=True)
    parser.add_argument("--spatial_conv", type=str, required=True)
    parser.add_argument("--ts_conv", type=str, required=True)

    parser.add_argument("--text_hidden_dim", type=int, default=None)
    parser.add_argument("--text_tokens", type=int, default=77)
    parser.add_argument("--text_token_dim", type=int, default=768)
    parser.add_argument("--text_dropout", type=float, default=None)
    parser.add_argument("--text_head", choices=["dense", "lowrank"], default="dense")
    parser.add_argument("--rank", type=int, default=64)

    parser.add_argument("--image_hidden_dim", type=int, default=None)
    parser.add_argument("--image_output_dim", type=int, default=1024)
    parser.add_argument("--image_dropout", type=float, default=None)

    parser.add_argument(
        "--sd_model",
        type=str,
        default="/home/chengwenjie/workspace/models/v1_5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors",
    )
    parser.add_argument(
        "--ip_adapter_root",
        type=str,
        default="/home/chengwenjie/workspace/models/ip-adapter",
    )
    parser.add_argument("--ip_adapter_subfolder", type=str, default="models")
    parser.add_argument("--ip_adapter_weight", type=str, default="ip-adapter_sd15.bin")
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0)
    parser.add_argument("--ip_adapter_scales", type=str, default="0.3,0.5,0.7,1.0")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    return parser.parse_args()


def build_text_model(args: argparse.Namespace) -> EEGAlign:
    backbone = EEGBackbone(feature_dim=args.feature_dim, dropout=args.dropout)
    text_dropout = args.text_dropout if args.text_dropout is not None else args.dropout
    if args.text_head == "lowrank":
        head = LowRankTextHead(
            in_dim=args.feature_dim,
            num_tokens=args.text_tokens,
            token_dim=args.text_token_dim,
            rank=args.rank,
        )
    else:
        hidden = args.text_hidden_dim or args.feature_dim
        head = TextTokenHead(
            in_dim=args.feature_dim,
            num_tokens=args.text_tokens,
            token_dim=args.text_token_dim,
            hidden=hidden,
            dropout=text_dropout,
        )
    return EEGAlign(backbone=backbone, head=head)


def build_image_model(args: argparse.Namespace) -> EEGAlign:
    backbone = EEGBackbone(feature_dim=args.feature_dim, dropout=args.dropout)
    hidden = args.image_hidden_dim or args.feature_dim
    image_dropout = args.image_dropout if args.image_dropout is not None else args.dropout
    head = ImageVectorHead(
        in_dim=args.feature_dim,
        out_dim=args.image_output_dim,
        hidden=hidden,
        dropout=image_dropout,
        layernorm=True,
    )
    return EEGAlign(backbone=backbone, head=head)


def load_align_model(
    model: EEGAlign, ckpt_path: str, device: torch.device
) -> EEGAlign:
    if not ckpt_path:
        raise ValueError("Checkpoint path is required for the selected mode.")
    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict dictionary.")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def parse_scales(raw: str) -> List[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def pick_sample_indices(total: int, count: int, seed: int) -> Set[int]:
    if total <= 0 or count <= 0:
        return set()
    rng = random.Random(seed)
    count = min(count, total)
    return set(rng.sample(range(total), count))


def encode_empty_prompt(
    pipe: StableDiffusionPipeline,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tokens = pipe.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = getattr(tokens, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    outputs = pipe.text_encoder(input_ids, attention_mask=attention_mask)
    embeds = outputs.last_hidden_state
    return embeds.to(dtype=dtype)


def prepare_ip_adapter_embeds(
    image_embeds: torch.Tensor, guidance_scale: float
) -> List[torch.Tensor]:
    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(1)
    assert image_embeds.dim() == 3, "image embeds must be [B,1,D]"
    if guidance_scale > 1.0:
        neg = torch.zeros_like(image_embeds)
        image_embeds = torch.cat([neg, image_embeds], dim=0)
    return [image_embeds]


def resize_to_match(images: Sequence[Image.Image], target: Image.Image) -> List[Image]:
    width, height = target.size
    resized = []
    for image in images:
        if image.size != (width, height):
            resized.append(image.resize((width, height), Image.BICUBIC))
        else:
            resized.append(image)
    return resized


class MetricsTracker:
    def __init__(
        self,
        device: torch.device,
        clip_model: CLIPModel,
        clip_processor: CLIPImageProcessor,
    ) -> None:
        self.device = device
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.is_metric = InceptionScore(splits=10, normalize=True).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = lpips.LPIPS(net="alex").to(device)
        self.lpips.eval()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.lpips_total = 0.0
        self.clip_total = 0.0
        self.count = 0

    def update(
        self,
        generated: torch.Tensor,
        original: torch.Tensor,
        generated_pil: Sequence[Image.Image],
        original_pil: Sequence[Image.Image],
    ) -> None:
        batch_size = generated.size(0)
        self.fid.update(original, real=True)
        self.fid.update(generated, real=False)
        self.is_metric.update(generated)
        self.ssim.update(generated, original)

        lpips_score = self.lpips(generated * 2 - 1, original * 2 - 1)
        self.lpips_total += float(lpips_score.mean().item()) * batch_size

        clip_score = self._clip_similarity(generated_pil, original_pil)
        self.clip_total += float(clip_score.sum().item())
        self.count += batch_size

    def _clip_similarity(
        self, generated: Sequence[Image.Image], original: Sequence[Image.Image]
    ) -> torch.Tensor:
        with torch.inference_mode():
            gen_inputs = self.clip_processor(
                images=list(generated), return_tensors="pt"
            ).to(self.device)
            orig_inputs = self.clip_processor(
                images=list(original), return_tensors="pt"
            ).to(self.device)
            gen_feat = self.clip_model.get_image_features(**gen_inputs)
            orig_feat = self.clip_model.get_image_features(**orig_inputs)
            gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
            orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)
            return (gen_feat * orig_feat).sum(dim=-1)

    def compute(self) -> Dict[str, float]:
        assert self.count > 0, "no samples processed"
        fid = float(self.fid.compute().item())
        is_mean, is_std = self.is_metric.compute()
        ssim = float(self.ssim.compute().item())
        lpips_score = self.lpips_total / self.count
        clip_score = self.clip_total / self.count
        return {
            "fid": fid,
            "inception_score_mean": float(is_mean.item()),
            "inception_score_std": float(is_std.item()),
            "ssim": ssim,
            "lpips": lpips_score,
            "clip_image_score": clip_score,
        }


def save_pairs(
    output_dir: Path,
    indices: Set[int],
    start_index: int,
    generated: Sequence[Image.Image],
    original: Sequence[Image.Image],
) -> None:
    if not indices:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, (gen_img, orig_img) in enumerate(zip(generated, original, strict=True)):
        index = start_index + offset
        if index not in indices:
            continue
        gen_path = output_dir / f"{index:06d}_gen.png"
        orig_path = output_dir / f"{index:06d}_gt.png"
        gen_img.save(gen_path)
        orig_img.save(orig_path)


def build_images_from_paths(paths: Sequence[str]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def run_eval(
    name: str,
    pipe: StableDiffusionPipeline,
    loader: DataLoader,
    device: torch.device,
    high_model: Optional[EEGAlign],
    low_model: Optional[EEGAlign],
    guidance_scale: float,
    num_inference_steps: int,
    ip_adapter_scale: Optional[float],
    save_indices: Set[int],
    output_dir: Path,
    max_samples: Optional[int],
    metrics: MetricsTracker,
    height: Optional[int],
    width: Optional[int],
    seed: int,
) -> Dict[str, float]:
    dtype = pipe.unet.dtype
    total = len(loader.dataset)
    target_total = min(total, max_samples) if max_samples else total
    generator = torch.Generator(device="cpu").manual_seed(seed)

    processed = 0
    progress = tqdm(total=target_total, desc=name)
    for batch in loader:
        if processed >= target_total:
            break

        eeg = batch["eeg_data"].to(device, dtype=torch.float32)
        paths = batch["img_path"]
        batch_size = eeg.size(0)

        remaining = target_total - processed
        if batch_size > remaining:
            eeg = eeg[:remaining]
            paths = paths[:remaining]
            batch_size = remaining

        with torch.inference_mode():
            uncond = encode_empty_prompt(pipe, batch_size, device, dtype)
            if high_model is not None:
                prompt_embeds = high_model(eeg).to(dtype)
                assert prompt_embeds.dim() == 3, "text embeds must be [B,77,768]"
            else:
                prompt_embeds = uncond
            negative_prompt_embeds = uncond

            ip_adapter_embeds = None
            if low_model is not None:
                image_embeds = low_model(eeg).to(dtype)
                assert image_embeds.dim() == 2, "image embeds must be [B,1024]"
                ip_adapter_embeds = prepare_ip_adapter_embeds(
                    image_embeds, guidance_scale
                )

            if ip_adapter_scale is not None:
                pipe.set_ip_adapter_scale(ip_adapter_scale)

            pipe_kwargs = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "ip_adapter_image_embeds": ip_adapter_embeds,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if height is not None:
                pipe_kwargs["height"] = height
            if width is not None:
                pipe_kwargs["width"] = width
            result = pipe(**pipe_kwargs)

        generated = [image.convert("RGB") for image in result.images]
        original = build_images_from_paths(paths)
        resized_original = resize_to_match(original, generated[0])

        generated_tensor = torch.stack(
            [TF.to_tensor(image) for image in generated], dim=0
        ).to(device)
        original_tensor = torch.stack(
            [TF.to_tensor(image) for image in resized_original], dim=0
        ).to(device)

        metrics.update(generated_tensor, original_tensor, generated, resized_original)
        save_pairs(
            output_dir,
            save_indices,
            processed,
            generated,
            original,
        )
        processed += batch_size
        progress.update(batch_size)
    progress.close()

    return metrics.compute()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "cuda requested but not available"
    device = torch.device(args.device)

    dataset_cls = DATASETS[args.dataset]
    dataset = dataset_cls(split=args.split, embedding_type=args.embedding_type)
    loader = _make_loader(dataset, args)

    need_high = "high" in args.modes or "both" in args.modes
    need_low = "low" in args.modes or "both" in args.modes

    high_model = None
    if need_high:
        high_model = load_align_model(build_text_model(args), args.high_ckpt, device)

    low_model = None
    if need_low:
        low_model = load_align_model(build_image_model(args), args.low_ckpt, device)

    pipe_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_single_file(
        args.sd_model,
        torch_dtype=pipe_dtype,
        variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True,
    ).to(device)

    if need_low:
        pipe.load_ip_adapter(
            args.ip_adapter_root,
            subfolder=args.ip_adapter_subfolder,
            weight_name=args.ip_adapter_weight,
        )

    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    total_samples = len(dataset)
    target_total = (
        min(total_samples, args.max_samples) if args.max_samples else total_samples
    )

    def run_mode(
        name: str,
        *,
        high: Optional[EEGAlign],
        low: Optional[EEGAlign],
        ip_scale: Optional[float],
        seed: int,
        out_dir: Path,
    ) -> None:
        save_indices = pick_sample_indices(target_total, args.save_samples, seed)
        metrics = MetricsTracker(device, clip_model, clip_processor)
        results[name] = run_eval(
            name=name,
            pipe=pipe,
            loader=loader,
            device=device,
            high_model=high,
            low_model=low,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            ip_adapter_scale=ip_scale,
            save_indices=save_indices,
            output_dir=out_dir,
            max_samples=args.max_samples,
            metrics=metrics,
            height=args.height,
            width=args.width,
            seed=seed,
        )

    if "high" in args.modes:
        run_mode(
            "high",
            high=high_model,
            low=None,
            ip_scale=None,
            seed=args.seed,
            out_dir=output_dir / "samples" / "high",
        )

    if "low" in args.modes:
        run_mode(
            "low",
            high=None,
            low=low_model,
            ip_scale=args.ip_adapter_scale,
            seed=args.seed + 1,
            out_dir=output_dir / "samples" / "low",
        )

    if "both" in args.modes:
        scales = parse_scales(args.ip_adapter_scales)
        for idx, scale in enumerate(scales):
            run_name = f"both_scale_{scale}"
            run_mode(
                run_name,
                high=high_model,
                low=low_model,
                ip_scale=scale,
                seed=args.seed + 10 + idx,
                out_dir=output_dir / "samples" / run_name,
            )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "results": results,
                "num_samples": target_total,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_inference_steps,
            },
            fp,
            indent=2,
            ensure_ascii=True,
        )


if __name__ == "__main__":
    main()
