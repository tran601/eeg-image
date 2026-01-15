import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import EEG40Dataset, EEG4Dataset, collate_fn_keep_captions
from src.losses import (
    CosineLoss,
    GlobalInfoNCELoss,
    MSELoss,
    TextPathLoss,
    pool_tokens,
)
from src.models.eeg_encoder import EEGAlign, EEGBackbone, LowRankTextHead, TextTokenHead


DATASETS = {
    "eeg4": EEG4Dataset,
    "eeg40": EEG40Dataset,
}


def _prepare_text_target(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.dim() == 4:
        assert embedding.size(1) == 1, "captions_per_sample must be 1"
        embedding = embedding.squeeze(1)
    assert embedding.dim() == 3, "text embeddings must be [B, N, D]"
    return embedding.float()


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="High-level EEG->text alignment")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="eeg40")
    parser.add_argument(
        "--embedding_type",
        choices=["caption_embeddings", "class_text_embeddings"],
        default="caption_embeddings",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=0)

    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text_head", choices=["dense", "lowrank"], default="dense")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_tokens", type=int, default=77)
    parser.add_argument("--token_dim", type=int, default=768)
    parser.add_argument("--rank", type=int, default=64)

    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--contrast_weight", type=float, default=1.0)
    parser.add_argument(
        "--contrastive", choices=["cosine", "infonce"], default="cosine"
    )
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument(
        "--pool_mode", type=str, choices=["mean", "eot"], default="mean"
    )

    parser.add_argument("--sinkhorn_weight", type=float, default=1.0)
    parser.add_argument("--sim_temperature", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--gamma_pos", type=float, default=0.02)
    parser.add_argument("--barycentric_weight", type=float, default=0.5)
    parser.add_argument("--global_cosine_weight", type=float, default=0.2)
    parser.add_argument("--use_sinkhorn_divergence", action="store_true")
    parser.add_argument("--position_metric", type=str, default="l1")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    backbone = EEGBackbone(feature_dim=args.feature_dim, dropout=args.dropout)
    if args.text_head == "dense":
        head = TextTokenHead(
            in_dim=args.feature_dim,
            num_tokens=args.num_tokens,
            token_dim=args.token_dim,
            hidden=args.hidden_dim,
            dropout=args.dropout,
        )
    else:
        head = LowRankTextHead(
            in_dim=args.feature_dim,
            num_tokens=args.num_tokens,
            token_dim=args.token_dim,
            rank=args.rank,
        )
    return EEGAlign(backbone=backbone, head=head)


def build_contrastive_loss(args: argparse.Namespace) -> torch.nn.Module:
    if args.contrastive == "cosine":
        return CosineLoss(l2_normalize=True, reduction="mean")
    return GlobalInfoNCELoss(
        temperature=args.contrastive_temperature, symmetric=True, l2_normalize=True
    )


def build_log_path(args: argparse.Namespace) -> Path:
    loss_parts = ["mse"]
    if args.contrast_weight > 0.0:
        loss_parts.append(args.contrastive)
    else:
        loss_parts.append("contrast-off")
    if args.sinkhorn_weight > 0.0:
        loss_parts.append("sinkhorn")
    else:
        loss_parts.append("sinkhorn-off")

    name = "__".join(
        [
            "align_high",
            f"dataset={args.dataset}",
            f"embedding={args.embedding_type}",
            f"text_head={args.text_head}",
            f"pool={args.pool_mode}",
            f"loss={'+'.join(loss_parts)}",
        ]
    )
    return Path(args.log_dir) / f"{name}.txt"


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mse_loss: torch.nn.Module,
    contrastive_loss: torch.nn.Module,
    sinkhorn_loss: torch.nn.Module,
    args: argparse.Namespace,
    train: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train() if train else model.eval()
    total = 0.0
    mse_total = 0.0
    contrast_total = 0.0
    sinkhorn_total = 0.0
    count = 0

    split = "train" if train else "val"
    pbar = tqdm(loader, desc=f"{split} {epoch}", leave=False)
    for batch in pbar:
        eeg = batch["eeg_data"].to(device, dtype=torch.float32)
        target = _prepare_text_target(batch["embedding"]).to(device)

        with torch.set_grad_enabled(train):
            pred = model(eeg)
            assert pred.shape == target.shape, "pred/target must share shape"

            mse_value = mse_loss(pred, target)
            contrast_value = pred.new_tensor(0.0)
            sinkhorn_value = pred.new_tensor(0.0)

            if args.contrast_weight > 0.0:
                pooled_pred = pool_tokens(pred, mode=args.pool_mode)
                pooled_target = pool_tokens(target, mode=args.pool_mode)
                contrast_value = contrastive_loss(pooled_pred, pooled_target)

            if args.sinkhorn_weight > 0.0:
                sinkhorn_value = sinkhorn_loss(pred, target)

            loss = args.mse_weight * mse_value
            loss = loss + args.contrast_weight * contrast_value
            loss = loss + args.sinkhorn_weight * args.sim_temperature * sinkhorn_value

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = pred.size(0)
        loss_item = float(loss.item())
        mse_item = float(mse_value.item())
        contrast_item = float(contrast_value.item())
        sinkhorn_item = float(sinkhorn_value.item())
        total += loss_item * batch_size
        mse_total += mse_item * batch_size
        contrast_total += contrast_item * batch_size
        sinkhorn_total += sinkhorn_item * batch_size
        count += batch_size
        pbar.set_postfix(
            loss=f"{loss_item:.4f}",
            mse=f"{mse_item:.4f}",
            contrast=f"{contrast_item:.4f}",
            sinkhorn=f"{sinkhorn_item:.4f}",
        )

    assert count > 0, "empty dataloader"
    return {
        "total": total / count,
        "mse": mse_total / count,
        "contrast": contrast_total / count,
        "sinkhorn": sinkhorn_total / count,
    }


def append_log(log_path: Path, epoch: int, split: str, stats: Dict[str, float]) -> None:
    with open(log_path, "a") as f:
        f.write(
            f"{epoch}\t{split}\t{stats['total']:.6f}\t{stats['mse']:.6f}\t"
            f"{stats['contrast']:.6f}\t{stats['sinkhorn']:.6f}\n"
        )


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "cuda requested but not available"
    device = torch.device(args.device)

    dataset_cls = DATASETS[args.dataset]
    train_set = dataset_cls(split="train", embedding_type=args.embedding_type)
    val_set = dataset_cls(split="val", embedding_type=args.embedding_type)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_keep_captions,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_keep_captions,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    mse_loss = MSELoss(reduction="mean")
    contrastive_loss = build_contrastive_loss(args)
    sinkhorn_loss = TextPathLoss(
        epsilon=args.epsilon,
        n_iters=args.n_iters,
        sim_temperature=args.sim_temperature,
        gamma_pos=args.gamma_pos,
        barycentric_weight=args.barycentric_weight,
        global_cosine_weight=args.global_cosine_weight,
        use_sinkhorn_divergence=args.use_sinkhorn_divergence,
        position_metric=args.position_metric,
        reduce="mean",
    )

    log_path = build_log_path(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("epoch\tsplit\ttotal\tmse\tcontrast\tsinkhorn\n")

    ckpt_dir = Path(args.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = build_log_path(args).stem
    best_path = ckpt_dir / f"{ckpt_stem}_best.pt"
    last_path = ckpt_dir / f"{ckpt_stem}_last.pt"
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mse_loss,
            contrastive_loss,
            sinkhorn_loss,
            args,
            train=True,
            epoch=epoch,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            mse_loss,
            contrastive_loss,
            sinkhorn_loss,
            args,
            train=False,
            epoch=epoch,
        )
        print(
            f"epoch {epoch}: "
            f"train_total={train_stats['total']:.6f} "
            f"train_mse={train_stats['mse']:.6f} "
            f"train_contrast={train_stats['contrast']:.6f} "
            f"train_sinkhorn={train_stats['sinkhorn']:.6f} "
            f"val_total={val_stats['total']:.6f} "
            f"val_mse={val_stats['mse']:.6f} "
            f"val_contrast={val_stats['contrast']:.6f} "
            f"val_sinkhorn={val_stats['sinkhorn']:.6f}"
        )
        append_log(log_path, epoch, "train", train_stats)
        append_log(log_path, epoch, "val", val_stats)

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            save_checkpoint(model, best_path)
        save_checkpoint(model, last_path)
        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = ckpt_dir / f"{ckpt_stem}_epoch{epoch}.pt"
            save_checkpoint(model, epoch_path)


if __name__ == "__main__":
    main()
