from __future__ import annotations

import argparse
import atexit
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn

from core.data import (
    DenseBatch,
    DEFAULT_PARTICLE_FEATURES,
    ParticleNormalization,
    compute_particle_normalization,
    discover_root_files,
    iter_dense_batches,
    load_particle_normalization,
    save_particle_normalization,
)
from core.model import SimpleParT


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def enable_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    def _restore_streams() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(_restore_streams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal SimpleParT model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-constituents", type=int, default=128)
    parser.add_argument("--step-size", type=str, default="20 MB")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/checkpoints/simple_part_checkpoint.pt"),
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("artifacts/checkpoints/simple_part_best.pt"),
    )
    parser.add_argument(
        "--normalization-path",
        type=Path,
        default=Path("artifacts/checkpoints/particle_norm.npz"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("artifacts/logs/train.log"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def maybe_limit(paths: list[Path], limit: int) -> list[Path]:
    return paths[:limit] if limit > 0 else paths


def interleave_labeled_paths(
    bb_paths: list[Path],
    gg_paths: list[Path],
) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    max_len = max(len(bb_paths), len(gg_paths))

    for index in range(max_len):
        if index < len(bb_paths):
            paths.append(bb_paths[index])
            labels.append(1)
        if index < len(gg_paths):
            paths.append(gg_paths[index])
            labels.append(0)

    return paths, labels


def iterate_labeled_batches(
    data_dir: Path,
    split: str,
    *,
    max_constituents: int,
    step_size: str,
    limit_files: int,
):
    bb_paths = maybe_limit(discover_root_files(data_dir, split, "HToBB"), limit_files)
    gg_paths = maybe_limit(discover_root_files(data_dir, split, "HToGG"), limit_files)

    paths, labels = interleave_labeled_paths(bb_paths, gg_paths)
    print(f"[{split}] files={len(paths)} (HToBB={len(bb_paths)}, HToGG={len(gg_paths)})")

    return iter_dense_batches(
        paths,
        labels,
        particle_features=DEFAULT_PARTICLE_FEATURES,
        max_constituents=max_constituents,
        step_size=step_size,
        verbose=True,
    )


def concatenate_batches(left: DenseBatch, right: DenseBatch) -> DenseBatch:
    return DenseBatch(
        x_particles=np.concatenate([left.x_particles, right.x_particles], axis=0),
        x_jets=np.concatenate([left.x_jets, right.x_jets], axis=0),
        mask=np.concatenate([left.mask, right.mask], axis=0),
        labels=np.concatenate([left.labels, right.labels], axis=0),
    )


def iterate_balanced_batches(
    data_dir: Path,
    split: str,
    *,
    max_constituents: int,
    step_size: str,
    limit_files: int,
    normalization: ParticleNormalization | None,
):
    bb_paths = maybe_limit(discover_root_files(data_dir, split, "HToBB"), limit_files)
    gg_paths = maybe_limit(discover_root_files(data_dir, split, "HToGG"), limit_files)

    print(f"[{split}] files={len(bb_paths) + len(gg_paths)} (HToBB={len(bb_paths)}, HToGG={len(gg_paths)})")

    bb_iter = iter_dense_batches(
        bb_paths,
        [1] * len(bb_paths),
        particle_features=DEFAULT_PARTICLE_FEATURES,
        max_constituents=max_constituents,
        step_size=step_size,
        verbose=True,
        normalization=normalization,
    )
    gg_iter = iter_dense_batches(
        gg_paths,
        [0] * len(gg_paths),
        particle_features=DEFAULT_PARTICLE_FEATURES,
        max_constituents=max_constituents,
        step_size=step_size,
        verbose=True,
        normalization=normalization,
    )

    while True:
        try:
            bb_batch = next(bb_iter)
            gg_batch = next(gg_iter)
        except StopIteration:
            break
        yield concatenate_batches(bb_batch, gg_batch)


def move_batch_to_device(batch, device: torch.device):
    x_particles = torch.from_numpy(batch.x_particles).to(device)
    padding_mask = torch.from_numpy(batch.mask).to(device)
    labels = torch.from_numpy(batch.labels).to(device)
    return x_particles, padding_mask, labels


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
    normalization: ParticleNormalization | None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "best_val_acc": best_val_acc,
        "particle_features": list(DEFAULT_PARTICLE_FEATURES),
        "max_constituents": args.max_constituents,
        "step_size": args.step_size,
        "limit_files": args.limit_files,
        "normalization_mean": normalization.mean if normalization is not None else None,
        "normalization_std": normalization.std if normalization is not None else None,
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
    normalization: ParticleNormalization | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            next_epoch=next_epoch,
            best_val_acc=best_val_acc,
            args=args,
            normalization=normalization,
        ),
        path,
    )
    print(f"Saved checkpoint to {path}", flush=True)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    next_epoch = int(checkpoint.get("next_epoch", 1))
    best_val_acc = float(checkpoint.get("best_val_acc", float("-inf")))
    print(f"Resumed from {path} at epoch={next_epoch}", flush=True)
    return next_epoch, best_val_acc


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    normalization: ParticleNormalization | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batch_index = 0

    for batch in iterate_balanced_batches(
        args.data_dir,
        "train",
        max_constituents=args.max_constituents,
        step_size=args.step_size,
        limit_files=args.limit_files,
        normalization=normalization,
    ):
        batch_index += 1
        x_particles, padding_mask, labels = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_particles, padding_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        running_loss = total_loss / total_examples
        running_acc = total_correct / total_examples
        print(
            f"[train] batch={batch_index} "
            f"jets={labels.size(0)} total={total_examples} "
            f"loss={running_loss:.4f} acc={running_acc:.4f}",
            flush=True,
        )
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break

    return total_loss / total_examples, total_correct / total_examples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    normalization: ParticleNormalization | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batch_index = 0

    for batch in iterate_balanced_batches(
        args.data_dir,
        "val",
        max_constituents=args.max_constituents,
        step_size=args.step_size,
        limit_files=args.limit_files,
        normalization=normalization,
    ):
        batch_index += 1
        x_particles, padding_mask, labels = move_batch_to_device(batch, device)
        logits = model(x_particles, padding_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        running_loss = total_loss / total_examples
        running_acc = total_correct / total_examples
        print(
            f"[val] batch={batch_index} "
            f"jets={labels.size(0)} total={total_examples} "
            f"loss={running_loss:.4f} acc={running_acc:.4f}",
            flush=True,
        )
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break

    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    args = parse_args()
    enable_file_logging(args.log_path)
    device = torch.device(args.device)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    model = SimpleParT(input_dim=len(DEFAULT_PARTICLE_FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1
    best_val_acc = float("-inf")
    normalization: ParticleNormalization | None = None

    train_bb_paths = maybe_limit(discover_root_files(args.data_dir, "train", "HToBB"), args.limit_files)
    train_gg_paths = maybe_limit(discover_root_files(args.data_dir, "train", "HToGG"), args.limit_files)
    train_paths = [*train_bb_paths, *train_gg_paths]
    val_bb_paths = maybe_limit(discover_root_files(args.data_dir, "val", "HToBB"), args.limit_files)
    val_gg_paths = maybe_limit(discover_root_files(args.data_dir, "val", "HToGG"), args.limit_files)

    print(
        "[config] "
        f"device={args.device} epochs={args.epochs} lr={args.lr} "
        f"max_constituents={args.max_constituents} step_size={args.step_size} "
        f"limit_files={args.limit_files or 'all'}",
        flush=True,
    )
    print(
        "[files] "
        f"train: HToBB={len(train_bb_paths)} HToGG={len(train_gg_paths)} "
        f"val: HToBB={len(val_bb_paths)} HToGG={len(val_gg_paths)}",
        flush=True,
    )
    print(
        "[artifacts] "
        f"best={args.weights_path} checkpoint={args.checkpoint_path} "
        f"norm={args.normalization_path}",
        flush=True,
    )

    if args.normalization_path.exists():
        normalization = load_particle_normalization(args.normalization_path)
        print(
            f"Loaded normalization from {args.normalization_path} "
            f"(features={len(normalization.mean)})",
            flush=True,
        )
    else:
        print("Computing particle normalization from train split...", flush=True)
        normalization = compute_particle_normalization(
            train_paths,
            particle_features=DEFAULT_PARTICLE_FEATURES,
            max_constituents=args.max_constituents,
            step_size=args.step_size,
        )
        args.normalization_path.parent.mkdir(parents=True, exist_ok=True)
        save_particle_normalization(args.normalization_path, normalization)
        print(f"Saved normalization to {args.normalization_path}", flush=True)

    if args.resume:
        if not args.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        start_epoch, best_val_acc = load_checkpoint(
            args.checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            print(f"[epoch] start {epoch}/{args.epochs}", flush=True)
            start = time.perf_counter()
            train_loss, train_acc = train_one_epoch(
                model,
                optimizer,
                criterion,
                device,
                args,
                normalization,
            )
            val_loss, val_acc = evaluate(
                model,
                criterion,
                device,
                args,
                normalization,
            )
            elapsed = time.perf_counter() - start

            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"time_s={elapsed:.1f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                args.weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "particle_features": list(DEFAULT_PARTICLE_FEATURES),
                        "max_constituents": args.max_constituents,
                        "normalization_mean": normalization.mean if normalization is not None else None,
                        "normalization_std": normalization.std if normalization is not None else None,
                    },
                    args.weights_path,
                )
                print(f"Saved best weights to {args.weights_path}", flush=True)

            save_checkpoint(
                args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                next_epoch=epoch + 1,
                best_val_acc=best_val_acc,
                args=args,
                normalization=normalization,
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving checkpoint...", flush=True)
        save_checkpoint(
            args.checkpoint_path,
            model=model,
            optimizer=optimizer,
            next_epoch=current_epoch,
            best_val_acc=best_val_acc,
            args=args,
            normalization=normalization,
        )
        return

    print(f"Training finished. Best weights are in {args.weights_path}", flush=True)


if __name__ == "__main__":
    main()
