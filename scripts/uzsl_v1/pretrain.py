"""Pretrain the Conv1D+Transformer encoder on Slovo RSL before UzSL fine-tuning.

Trains the EXACT same architecture as uzsl_train.py (frame_dim=750) on 1,000 RSL
classes from Slovo (15,300 videos, 194 signers). Only the classification head is
different; the encoder weights are transplanted directly into UzSL fine-tuning.

Slovo has hands-only landmarks (no pose). We zero-fill the POSE slot so the
model architecture is identical; the encoder learns RSL-family hand features.

Usage:
    # Download Slovo first:
    pip install kaggle
    kaggle datasets download kulqkul/slovo-mediapipe-json -p slovo_data
    unzip slovo_data/slovo-mediapipe-json.zip -d slovo_data
    kaggle datasets download -d kapitanov/slovo -f annotations.csv -p slovo_data

    # Then pretrain:
    .venv/Scripts/python.exe uzsl_pretrain.py --slovo-dir slovo_data --epochs 30

The saved checkpoint at --artifact-dir/encoder.pt can then be loaded in
uzsl_train.py with --pretrained-encoder <path>.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from .augment import Augment
from .features import feature_dim
from .model import build_model, require_torch
from .paths import DEFAULT_DATA_DIR
from .progress import ProgressBar
from .slovo_data import SlovoDataset

DEFAULT_ARTIFACT_DIR = Path("artifacts") / "slovo_pretrain"


def collate_slovo(batch):
    torch = require_torch()
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def evaluate(model, loader, device, num_classes, top_k=5) -> dict:
    torch = require_torch()
    model.eval()
    total = correct1 = correctk = 0
    pred_counts = Counter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            k = min(top_k, logits.shape[1])
            top = logits.topk(k, dim=1).indices
            pred = top[:, 0]
            total += y.numel()
            correct1 += (pred == y).sum().item()
            correctk += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            pred_counts.update(pred.cpu().tolist())
    collapse = max(pred_counts.values()) / total if total else 0.0
    return {
        "samples": total,
        "top1": correct1 / total if total else 0.0,
        "top5": correctk / total if total else 0.0,
        "mode_collapse_share": collapse,
    }


def pretrain(args: argparse.Namespace) -> None:
    torch = require_torch()
    torch.manual_seed(args.seed)

    COMPONENTS = "hands_pose"
    frame_dim = feature_dim(COMPONENTS, kinematics=args.kinematics)

    augment = Augment(components=COMPONENTS, seed=args.seed) if args.augment else None

    print("Loading Slovo training set…")
    train_ds = SlovoDataset(
        args.slovo_dir, split="train",
        target_frames=args.target_frames,
        kinematics=args.kinematics,
        augment=augment,
        seed=args.seed,
    )
    val_ds = SlovoDataset(
        args.slovo_dir, split="test",
        target_frames=args.target_frames,
        kinematics=args.kinematics,
        seed=args.seed,
    )
    n_classes = len(train_ds.class_to_idx)
    print(f"train={len(train_ds)}  val={len(val_ds)}  classes={n_classes}  frame_dim={frame_dim}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        input_dim=frame_dim,
        num_classes=n_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        architecture=args.architecture,
        target_frames=args.target_frames,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)

    loader_kw = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_slovo,
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader   = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler  = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and args.amp))

    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = seen = 0
        bar = ProgressBar(len(train_loader), label=f"ep {epoch}/{args.epochs}")
        for bi, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * y.numel()
            seen += y.numel()
            bar.update(bi, suffix=f"loss {running_loss/seen:.4f}")
        bar.finish(suffix=f"loss {running_loss/seen:.4f}")

        m = evaluate(model, val_loader, device, n_classes)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "lr": scheduler.get_last_lr()[0],
            **{f"val_{k}": v for k, v in m.items()},
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if m["top1"] > best_val:
            best_val = m["top1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save full checkpoint (for resuming) and encoder-only weights (for transplant)
    full_ckpt = args.artifact_dir / "pretrain_checkpoint.pt"
    encoder_ckpt = args.artifact_dir / "encoder.pt"

    torch.save({
        "model_state": best_state,
        "class_to_idx": train_ds.class_to_idx,
        "config": {
            "frame_dim": frame_dim,
            "hidden_dim": args.hidden_dim,
            "architecture": args.architecture,
            "target_frames": args.target_frames,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "kinematics": args.kinematics,
            "components": "hands_pose",
        },
        "history": history,
    }, full_ckpt)

    # Extract only encoder weights (everything except the classification head)
    encoder_state = {
        k: v for k, v in best_state.items()
        if not k.startswith("head.")
    }
    torch.save(encoder_state, encoder_ckpt)

    print(f"\nBest val top-1: {best_val:.3f}")
    print(f"Full checkpoint : {full_ckpt}")
    print(f"Encoder weights : {encoder_ckpt}  ← use with --pretrained-encoder")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretrain encoder on Slovo RSL landmarks")
    ap.add_argument("--slovo-dir",    type=Path, required=True, help="Folder with slovo_mediapipe.json + annotations.csv")
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument("--target-frames", type=int,   default=64)
    ap.add_argument("--architecture",  choices=["conv_transformer", "transformer"], default="conv_transformer")
    ap.add_argument("--hidden-dim",    type=int,   default=192)
    ap.add_argument("--n-layers",      type=int,   default=4)
    ap.add_argument("--n-heads",       type=int,   default=4)
    ap.add_argument("--dropout",       type=float, default=0.2)
    ap.add_argument("--epochs",        type=int,   default=30)
    ap.add_argument("--batch-size",    type=int,   default=64)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight-decay",  type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-kinematics", dest="kinematics", action="store_false")
    ap.add_argument("--no-augment",    dest="augment",    action="store_false")
    ap.add_argument("--num-workers",   type=int, default=max(0, min(4, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--seed",          type=int, default=13)
    ap.add_argument("--device",        default="")
    ap.add_argument("--amp",           action="store_true")
    ap.set_defaults(kinematics=True, augment=True)
    args = ap.parse_args()
    pretrain(args)


if __name__ == "__main__":
    main()
