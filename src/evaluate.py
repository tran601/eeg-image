import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from compel import Compel
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
from src.utils import set_seed


DATASETS = {
    "eeg4": EEG4Dataset,
    "eeg40": EEG40Dataset,
}

NEGATIVE_PROMPT = (
    "cartoon, illustration, 3d render, plastic, lowres, blurry, jpeg artifacts, "
    "watermark, text, logo, cropped, deformed, bad anatomy, extra limbs, "
    "oversaturated, multiple subjects, clutter, fisheye, motion blur, "
    "studio backdrop, cutout"
)

# Defaults aligned with src/align_high_level.py and src/align_low_level.py.
HIGH_FEATURE_DIM = 512
HIGH_DROPOUT = 0.1
HIGH_HIDDEN_DIM = 512
HIGH_NUM_TOKENS = 77
HIGH_TOKEN_DIM = 768
HIGH_RANK = 64

LOW_FEATURE_DIM = 512
LOW_DROPOUT = 0.1
LOW_HIDDEN_DIM = 512
LOW_OUT_DIM = 1024


def _make_loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
        choices=["caption_embeddings", "image_embeddings", "both"],
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")

    parser.add_argument(
        "--high_ckpt",
        type=str,
        default=(
            "/home/chengwenjie/workspace/masterCode2_simple/checkpoints/"
            "align_high__dataset=eeg40__embedding=caption_embeddings__text_head=dense__pool=mean__loss=mse+cosine+sinkhorn_last.pt"
        ),
    )
    parser.add_argument(
        "--low_ckpt",
        type=str,
        default=(
            "/home/chengwenjie/workspace/masterCode2_simple/checkpoints/"
            "align_low__dataset=eeg40__embedding=image_embeddings__loss=mse+cosine_last.pt"
        ),
    )

    parser.add_argument("--modes", choices=["high", "low", "both"], default="high")

    parser.add_argument("--text_head", choices=["dense", "lowrank"], default="dense")

    parser.add_argument(
        "--sd_model",
        type=str,
        default="/home/chengwenjie/workspace/models/v1_5/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--ip_adapter_root",
        type=str,
        default="/home/chengwenjie/workspace/models/ip-adapter",
    )
    parser.add_argument("--ip_adapter_subfolder", type=str, default="models")
    parser.add_argument("--ip_adapter_weight", type=str, default="ip-adapter_sd15.bin")
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)

    parser.add_argument(
        "--clip_model",
        type=str,
        default="/home/chengwenjie/workspace/models/CLIP-ViT-B-32-laion2B-s34B-b79K",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    return parser.parse_args()


def build_text_model(args: argparse.Namespace) -> EEGAlign:
    backbone = EEGBackbone(feature_dim=HIGH_FEATURE_DIM, dropout=HIGH_DROPOUT)
    if args.text_head == "lowrank":
        head = LowRankTextHead(
            in_dim=HIGH_FEATURE_DIM,
            num_tokens=HIGH_NUM_TOKENS,
            token_dim=HIGH_TOKEN_DIM,
            rank=HIGH_RANK,
        )
    else:
        head = TextTokenHead(
            in_dim=HIGH_FEATURE_DIM,
            num_tokens=HIGH_NUM_TOKENS,
            token_dim=HIGH_TOKEN_DIM,
            hidden=HIGH_HIDDEN_DIM,
            dropout=HIGH_DROPOUT,
        )
    return EEGAlign(backbone=backbone, head=head)


def build_image_model() -> EEGAlign:
    backbone = EEGBackbone(feature_dim=LOW_FEATURE_DIM, dropout=LOW_DROPOUT)
    head = ImageVectorHead(
        in_dim=LOW_FEATURE_DIM,
        out_dim=LOW_OUT_DIM,
        hidden=LOW_HIDDEN_DIM,
        dropout=LOW_DROPOUT,
        layernorm=True,
    )
    return EEGAlign(backbone=backbone, head=head)


def load_align_model(model: EEGAlign, ckpt_path: str, device: torch.device) -> EEGAlign:
    if not ckpt_path:
        raise ValueError("Checkpoint path is required for the selected mode.")
    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict dictionary.")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def pick_sample_indices(total: int, count: int, seed: int) -> Set[int]:
    if total <= 0 or count <= 0:
        return set()
    rng = random.Random(seed)
    count = min(count, total)
    return set(rng.sample(range(total), count))


def encode_empty_prompt(
    compel_proc: Compel,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    embeds = compel_proc([""] * batch_size)
    return embeds.to(device=device, dtype=dtype)


def encode_prompt(
    compel_proc: Compel,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    embeds = compel_proc([prompt])
    return embeds.to(device=device, dtype=dtype)


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


def prepare_text_embedding(text_embeds: torch.Tensor) -> torch.Tensor:
    if text_embeds.dim() == 4:
        assert text_embeds.size(1) == 1, "text embeds must be [B,77,768]"
        text_embeds = text_embeds.squeeze(1)
    assert text_embeds.dim() == 3, "text embeds must be [B,77,768]"
    return text_embeds.float()


def prepare_image_embedding(image_embeds: torch.Tensor) -> torch.Tensor:
    if image_embeds.dim() == 3:
        assert image_embeds.size(1) == 1, "image embeds must be [B,1024]"
        image_embeds = image_embeds.squeeze(1)
    assert image_embeds.dim() == 2, "image embeds must be [B,1024]"
    return image_embeds.float()


def resize_to_match(
    images: Sequence[Image.Image], target: Image.Image
) -> List[Image.Image]:
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


def save_triplets(
    output_dir: Path,
    indices: Set[int],
    start_index: int,
    paths: Sequence[str],
    original: Sequence[Image.Image],
    embedded: Sequence[Image.Image],
    eeg_generated: Sequence[Image.Image],
) -> None:
    if not indices:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, (path, orig_img, embed_img, eeg_img) in enumerate(
        zip(paths, original, embedded, eeg_generated, strict=True)
    ):
        index = start_index + offset
        if index not in indices:
            continue
        gap = 5
        images = [orig_img, embed_img, eeg_img]
        total_width = sum(image.width for image in images) + gap * (len(images) - 1)
        max_height = max(image.height for image in images)
        merged = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for image in images:
            merged.paste(image, (x_offset, 0))
            x_offset += image.width + gap
        merged.save(output_dir / Path(path).name)


def build_images_from_paths(paths: Sequence[str]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def run_eval(
    name: str,
    mode: str,
    pipe: StableDiffusionPipeline,
    compel_proc: Compel,
    negative_prompt_embed: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    high_model: Optional[EEGAlign],
    low_model: Optional[EEGAlign],
    guidance_scale: float,
    num_inference_steps: int,
    ip_adapter_scale: Optional[float],
    save_indices: Set[int],
    output_dir: Path,
    metrics_eeg_original: MetricsTracker,
    metrics_eeg_embedding: MetricsTracker,
    height: Optional[int],
    width: Optional[int],
) -> Dict[str, Dict[str, float]]:
    dtype = pipe.unet.dtype
    if ip_adapter_scale is not None:
        pipe.set_ip_adapter_scale(ip_adapter_scale)

    for batch_idx, batch in enumerate(tqdm(loader, desc=name)):
        eeg = batch["eeg_data"].to(device, dtype=torch.float32)
        paths = batch["img_path"]
        batch_size = eeg.size(0)

        with torch.inference_mode():
            uncond = encode_empty_prompt(compel_proc, batch_size, device, dtype)
            if high_model is not None:
                eeg_prompt_embeds = high_model(eeg).to(dtype)
                assert eeg_prompt_embeds.dim() == 3, "text embeds must be [B,77,768]"
            else:
                eeg_prompt_embeds = uncond
            negative_prompt_embeds = negative_prompt_embed.expand(batch_size, -1, -1)

            eeg_ip_adapter_embeds = None
            if low_model is not None:
                eeg_image_embeds = low_model(eeg).to(dtype)
                assert eeg_image_embeds.dim() == 2, "image embeds must be [B,1024]"
                eeg_ip_adapter_embeds = prepare_ip_adapter_embeds(
                    eeg_image_embeds, guidance_scale
                )

            embed_prompt_embeds = uncond
            embed_ip_adapter_embeds = None
            if mode == "high":
                text_embeds = prepare_text_embedding(batch["embedding"]).to(
                    device, dtype=dtype
                )
                embed_prompt_embeds = text_embeds
            elif mode == "low":
                image_embeds = prepare_image_embedding(batch["embedding"]).to(
                    device, dtype=dtype
                )
                embed_ip_adapter_embeds = prepare_ip_adapter_embeds(
                    image_embeds, guidance_scale
                )
            else:
                text_embeds = prepare_text_embedding(batch["text_embedding"]).to(
                    device, dtype=dtype
                )
                image_embeds = prepare_image_embedding(batch["image_embedding"]).to(
                    device, dtype=dtype
                )
                embed_prompt_embeds = text_embeds
                embed_ip_adapter_embeds = prepare_ip_adapter_embeds(
                    image_embeds, guidance_scale
                )

            pipe_kwargs = {
                "negative_prompt_embeds": negative_prompt_embeds,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
            if height is not None:
                pipe_kwargs["height"] = height
            if width is not None:
                pipe_kwargs["width"] = width
            embed_result = pipe(
                prompt_embeds=embed_prompt_embeds,
                ip_adapter_image_embeds=embed_ip_adapter_embeds,
                **pipe_kwargs,
            )
            eeg_result = pipe(
                prompt_embeds=eeg_prompt_embeds,
                ip_adapter_image_embeds=eeg_ip_adapter_embeds,
                **pipe_kwargs,
            )

        embedded_generated = [image.convert("RGB") for image in embed_result.images]
        eeg_generated = [image.convert("RGB") for image in eeg_result.images]
        original = build_images_from_paths(paths)
        assert (
            embedded_generated[0].size == eeg_generated[0].size
        ), "generated sizes must match"
        resized_original = resize_to_match(original, embedded_generated[0])

        embedded_tensor = torch.stack(
            [TF.to_tensor(image) for image in embedded_generated], dim=0
        ).to(device)
        eeg_tensor = torch.stack(
            [TF.to_tensor(image) for image in eeg_generated], dim=0
        ).to(device)
        original_tensor = torch.stack(
            [TF.to_tensor(image) for image in resized_original], dim=0
        ).to(device)

        metrics_eeg_original.update(
            eeg_tensor, original_tensor, eeg_generated, resized_original
        )
        metrics_eeg_embedding.update(
            eeg_tensor, embedded_tensor, eeg_generated, embedded_generated
        )
        assert loader.batch_size is not None, "loader batch_size is required"
        save_triplets(
            output_dir,
            save_indices,
            batch_idx * loader.batch_size,
            paths,
            resized_original,
            embedded_generated,
            eeg_generated,
        )

    return {
        "eeg_vs_original": metrics_eeg_original.compute(),
        "eeg_vs_embedding": metrics_eeg_embedding.compute(),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "cuda requested but not available"
    set_seed(args.seed)
    device = torch.device(args.device)

    mode = args.modes
    if mode == "high":
        embedding_type = "caption_embeddings"
    elif mode == "low":
        embedding_type = "image_embeddings"
    else:
        embedding_type = "both"
    if args.embedding_type is not None:
        assert (
            args.embedding_type == embedding_type
        ), "embedding_type must match modes"

    dataset_cls = DATASETS[args.dataset]
    dataset = dataset_cls(split=args.split, embedding_type=embedding_type)
    if args.max_samples is not None:
        max_count = min(len(dataset), args.max_samples)
        dataset = Subset(dataset, range(max_count))
    loader = _make_loader(dataset, args)

    need_high = mode in {"high", "both"}
    need_low = mode in {"low", "both"}

    high_model = None
    if need_high:
        high_model = load_align_model(build_text_model(args), args.high_ckpt, device)

    low_model = None
    if need_low:
        low_model = load_align_model(build_image_model(), args.low_ckpt, device)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch.float16,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )

    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    with torch.inference_mode():
        negative_prompt_embed = encode_prompt(
            compel_proc,
            NEGATIVE_PROMPT,
            device,
            pipe.unet.dtype,
        )

    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    target_total = len(dataset)

    if need_low:
        pipe.load_ip_adapter(
            args.ip_adapter_root,
            subfolder=args.ip_adapter_subfolder,
            weight_name=args.ip_adapter_weight,
        )

    save_indices = pick_sample_indices(target_total, args.save_samples, args.seed)
    metrics_eeg_original = MetricsTracker(device, clip_model, clip_processor)
    metrics_eeg_embedding = MetricsTracker(device, clip_model, clip_processor)
    results[mode] = run_eval(
        name=mode,
        mode=mode,
        pipe=pipe,
        compel_proc=compel_proc,
        negative_prompt_embed=negative_prompt_embed,
        loader=loader,
        device=device,
        high_model=high_model if need_high else None,
        low_model=low_model if need_low else None,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        ip_adapter_scale=args.ip_adapter_scale if need_low else None,
        save_indices=save_indices,
        output_dir=output_dir / "samples" / mode,
        metrics_eeg_original=metrics_eeg_original,
        metrics_eeg_embedding=metrics_eeg_embedding,
        height=args.height,
        width=args.width,
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
