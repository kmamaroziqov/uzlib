from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .augment import Augment
from .features import feature_dim, finalize, prepare_base
from .model import build_model, load_pretrained_encoder, require_torch
from .paths import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, DEFAULT_MANIFEST
from .progress import ProgressBar
from .validate_data import resolve_pose_path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def label_metadata(rows: list[dict[str, str]], train_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    train_sign_ids = {row["sign_id"] for row in train_rows}
    by_sign: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["sign_id"] in train_sign_ids and row["sign_id"] not in by_sign:
            by_sign[row["sign_id"]] = {
                "sign_id": row["sign_id"],
                "label_uz": row["label_uz"],
                "label_ru": row.get("label_ru", ""),
                "category": row.get("category", ""),
            }
    return [by_sign[sign_id] for sign_id in sorted(by_sign)]


# In-memory cache of preprocessed full-length base arrays (load + normalize + trim),
# shared across epochs. Most valuable with num_workers=0 (the default): each worker
# process would otherwise fill its own copy.
_BASE_CACHE: dict[tuple[str, str, float, bool], np.ndarray] = {}
# Pose files that failed to read/parse at access time. A single corrupt file
# (e.g. zero-byte or truncated by a cross-platform copy) must not kill a run.
_BAD_PATHS: set[str] = set()


def pose_file_ok(path: Path) -> bool:
    """Cheap pre-filter: exists and is non-empty (a stat, not a full parse)."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def load_base_cached(pose_path: Path, components: str, trim_threshold: float, wrist_norm: bool = False) -> np.ndarray:
    key = (str(pose_path), components, trim_threshold, wrist_norm)
    cached = _BASE_CACHE.get(key)
    if cached is None:
        cached = prepare_base(pose_path, components, trim_threshold=trim_threshold, wrist_norm=wrist_norm)
        _BASE_CACHE[key] = cached
    return cached


class PoseDataset:
    def __init__(
        self,
        rows: list[dict[str, str]],
        data_dir: Path,
        class_to_idx: dict[str, int],
        signer_to_idx: dict[str, int],
        *,
        target_frames: int,
        trim_threshold: float,
        components: str,
        architecture: str,
        kinematics: bool,
        wrist_norm: bool = False,
        augment: Augment | None = None,
    ) -> None:
        self.rows = [
            row for row in rows
            if row["sign_id"] in class_to_idx
            and pose_file_ok(resolve_pose_path(data_dir, row["pose_path"]))
        ]
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.signer_to_idx = signer_to_idx
        self.target_frames = target_frames
        self.trim_threshold = trim_threshold
        self.components = components
        self.architecture = architecture
        self.kinematics = kinematics
        self.wrist_norm = wrist_norm
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        # Resilient load: if a pose file is unreadable/corrupt, skip to the next
        # valid row instead of crashing the whole run. Bad paths are recorded so
        # repeated draws (WeightedRandomSampler with replacement) skip instantly.
        n = len(self.rows)
        for attempt in range(n):
            row = self.rows[(index + attempt) % n]
            pose_path = resolve_pose_path(self.data_dir, row["pose_path"])
            if str(pose_path) in _BAD_PATHS:
                continue
            try:
                base = load_base_cached(pose_path, self.components, self.trim_threshold, self.wrist_norm)
            except Exception as exc:  # noqa: BLE001 - any read/parse failure is non-fatal
                if str(pose_path) not in _BAD_PATHS:
                    _BAD_PATHS.add(str(pose_path))
                    print(f"warning: skipping unreadable pose {pose_path}: {exc}", flush=True)
                continue
            if self.augment is not None:
                base = self.augment(base)
            features = finalize(base, self.target_frames, kinematics=self.kinematics)
            if self.architecture == "mlp":
                features = features.reshape(-1)
            torch = require_torch()
            x = torch.from_numpy(features)
            y = torch.tensor(self.class_to_idx[row["sign_id"]], dtype=torch.long)
            s = torch.tensor(self.signer_to_idx.get(row.get("signer_id", ""), 0), dtype=torch.long)
            return x, y, s
        raise RuntimeError("All pose files in this dataset failed to load.")


def collate_batch(batch):
    torch = require_torch()
    xs, ys, ss = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(ss)


def evaluate(model, loader, device: str, num_classes: int, top_k: int = 5) -> dict[str, float | int]:
    torch = require_torch()
    model.eval()
    total = 0
    correct1 = 0
    correctk = 0
    predicted_counts = Counter()
    gold_counts = Counter()
    true_positive = Counter()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            k = min(top_k, logits.shape[1])
            top = logits.topk(k, dim=1).indices
            pred = top[:, 0]
            total += y.numel()
            correct1 += (pred == y).sum().item()
            correctk += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            for p, g in zip(pred.cpu().tolist(), y.cpu().tolist()):
                predicted_counts[p] += 1
                gold_counts[g] += 1
                if p == g:
                    true_positive[g] += 1

    f1_values: list[float] = []
    for class_idx in range(num_classes):
        tp = true_positive[class_idx]
        fp = predicted_counts[class_idx] - tp
        fn = gold_counts[class_idx] - tp
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)

    return {
        "samples": total,
        "top1": correct1 / total if total else 0.0,
        "top5": correctk / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        # Largest share of predictions landing on a single class; ~1/n_classes is
        # healthy, near 1.0 means the model collapsed.
        "mode_collapse_share": max(predicted_counts.values()) / total if total else 0.0,
    }


def _tta_logits(model, x, tta_views: int, components: str):
    """Average logits over deterministic views.

    View 1 is the original sequence. View 2 is a horizontal mirror for the
    hands_pose layout, matching the train-time flip convention. Extra views are
    intentionally ignored for now so TTA stays conservative and reproducible.
    """
    if tta_views <= 1 or components != "hands_pose":
        return model(x)
    torch = require_torch()
    flipped = x.clone()
    # hands_pose flattened per frame: [LH(21), POSE(33), RH(21)] * channels.
    # Infer channels so this works with and without kinematics.
    channels = flipped.shape[-1] // 75
    if channels * 75 != flipped.shape[-1]:
        return model(x)
    z = flipped.reshape(flipped.shape[0], flipped.shape[1], 75, channels)
    perm = torch.arange(75, device=flipped.device)
    perm[0:21], perm[54:75] = torch.arange(54, 75, device=flipped.device), torch.arange(0, 21, device=flipped.device)
    pose_pairs = [
        (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
        (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
    ]
    for a, b in pose_pairs:
        perm[21 + a], perm[21 + b] = 21 + b, 21 + a
    z = z[:, :, perm, :]
    x_channels = [0]
    if channels >= 8:
        x_channels.extend([4, 7])  # velocity-x and acceleration-x
    z[..., x_channels] = -z[..., x_channels]
    return (model(x) + model(z.reshape_as(flipped))) / 2.0


def evaluate_tta(model, loader, device: str, num_classes: int, *, components: str, tta_views: int, top_k: int = 5) -> dict[str, float | int]:
    torch = require_torch()
    model.eval()
    total = 0
    correct1 = 0
    correctk = 0
    predicted_counts = Counter()
    gold_counts = Counter()
    true_positive = Counter()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = _tta_logits(model, x, tta_views, components)
            k = min(top_k, logits.shape[1])
            top = logits.topk(k, dim=1).indices
            pred = top[:, 0]
            total += y.numel()
            correct1 += (pred == y).sum().item()
            correctk += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            for p, g in zip(pred.cpu().tolist(), y.cpu().tolist()):
                predicted_counts[p] += 1
                gold_counts[g] += 1
                if p == g:
                    true_positive[g] += 1

    f1_values: list[float] = []
    for class_idx in range(num_classes):
        tp = true_positive[class_idx]
        fp = predicted_counts[class_idx] - tp
        fn = gold_counts[class_idx] - tp
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)

    return {
        "samples": total,
        "top1": correct1 / total if total else 0.0,
        "top5": correctk / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "mode_collapse_share": max(predicted_counts.values()) / total if total else 0.0,
    }


def build_prototypes(model, loader, device: str, num_classes: int):
    torch = require_torch()
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            _, features = model(x, return_features=True)
            features = torch.nn.functional.normalize(features, dim=1)
            if sums is None:
                sums = torch.zeros(num_classes, features.shape[1], dtype=torch.float32, device=device)
            sums.index_add_(0, y, features)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
    if sums is None:
        return None
    prototypes = sums / counts.clamp(min=1.0).unsqueeze(1)
    return torch.nn.functional.normalize(prototypes, dim=1)


def evaluate_prototypes(model, loader, prototypes, device: str, top_k: int = 5) -> dict[str, float | int]:
    torch = require_torch()
    model.eval()
    total = 0
    correct1 = 0
    correctk = 0
    predicted_counts = Counter()
    gold_counts = Counter()
    true_positive = Counter()
    num_classes = prototypes.shape[0]
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            _, features = model(x, return_features=True)
            features = torch.nn.functional.normalize(features, dim=1)
            logits = features @ prototypes.T
            k = min(top_k, logits.shape[1])
            top = logits.topk(k, dim=1).indices
            pred = top[:, 0]
            total += y.numel()
            correct1 += (pred == y).sum().item()
            correctk += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            for p, g in zip(pred.cpu().tolist(), y.cpu().tolist()):
                predicted_counts[p] += 1
                gold_counts[g] += 1
                if p == g:
                    true_positive[g] += 1

    f1_values: list[float] = []
    for class_idx in range(num_classes):
        tp = true_positive[class_idx]
        fp = predicted_counts[class_idx] - tp
        fn = gold_counts[class_idx] - tp
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)

    return {
        "samples": total,
        "top1": correct1 / total if total else 0.0,
        "top5": correctk / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "mode_collapse_share": max(predicted_counts.values()) / total if total else 0.0,
    }


def category_accuracy(model, loader, rows: list[dict[str, str]], class_to_idx: dict[str, int], device: str) -> dict[str, float]:
    torch = require_torch()
    categories_by_class = {class_to_idx[row["sign_id"]]: row.get("category", "") for row in rows if row["sign_id"] in class_to_idx}
    totals = Counter()
    correct = Counter()
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().tolist()
            gold = y.cpu().tolist()
            for p, g in zip(pred, gold):
                category = categories_by_class.get(g, "")
                totals[category] += 1
                if p == g:
                    correct[category] += 1
    return {category: correct[category] / total for category, total in sorted(totals.items()) if category}


def signer_accuracy(model, loader, eval_rows: list[dict[str, str]], device: str) -> dict[str, float]:
    torch = require_torch()
    totals = Counter()
    correct = Counter()
    row_index = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().tolist()
            gold = y.cpu().tolist()
            for p, g in zip(pred, gold):
                signer = eval_rows[row_index].get("signer_id", "")
                row_index += 1
                totals[signer] += 1
                if p == g:
                    correct[signer] += 1
    return {signer: correct[signer] / total for signer, total in sorted(totals.items()) if signer}


def make_grad_reverse(torch):
    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.lam * grad, None

    return _GradReverse.apply


def train(args: argparse.Namespace) -> dict[str, object]:
    torch = require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(args.cpu_threads, 4)))

    rows = read_manifest(args.manifest)
    train_rows = [row for row in rows if row.get(args.split_column) == "train"]
    val_rows   = [row for row in rows if row.get(args.split_column) == "val"]
    test_rows  = [row for row in rows if row.get(args.split_column) == "test"]

    labels = label_metadata(rows, train_rows)
    class_to_idx = {row["sign_id"]: idx for idx, row in enumerate(labels)}
    if not labels:
        raise SystemExit(f"No training labels found using split column {args.split_column!r}")

    # When the manifest carries an eval_vocab column (built by make_loso_splits.py),
    # restrict val/test scoring to classes with >=2 signers so the signer-independent
    # metric is honest (single-signer classes are unlearnable when that signer is held out).
    has_eval_vocab = "eval_vocab" in (rows[0] if rows else {})
    if has_eval_vocab:
        eval_class_ids = {r["sign_id"] for r in rows if r.get("eval_vocab") == "1"}
        val_rows_eval  = [r for r in val_rows  if r["sign_id"] in class_to_idx and r["sign_id"] in eval_class_ids]
        test_rows_eval = [r for r in test_rows if r["sign_id"] in class_to_idx and r["sign_id"] in eval_class_ids]
        dropped_v = len([r for r in val_rows  if r["sign_id"] in class_to_idx]) - len(val_rows_eval)
        dropped_t = len([r for r in test_rows if r["sign_id"] in class_to_idx]) - len(test_rows_eval)
        if dropped_v or dropped_t:
            print(f"eval_vocab filter: dropped {dropped_v} val rows, {dropped_t} test rows (single-signer classes)", flush=True)
        val_rows  = val_rows_eval
        test_rows = test_rows_eval
    signer_to_idx = {signer: idx for idx, signer in enumerate(sorted({row.get("signer_id", "") for row in train_rows}))}

    if args.signer_adversarial and args.architecture == "mlp":
        raise SystemExit("--signer-adversarial requires a sequence architecture (conv_transformer or transformer).")
    if args.signer_adversarial and len(signer_to_idx) < 2:
        print("warning: --signer-adversarial disabled, training split has fewer than 2 signers")
        args.signer_adversarial = False

    augment = (
        Augment(
            components=args.components,
            seed=args.seed,
            body_proportion_p=0.5 if args.body_proportion_aug else 0.0,
        )
        if args.augment
        else None
    )
    dataset_kwargs = {
        "target_frames": args.target_frames,
        "trim_threshold": args.trim_threshold,
        "components": args.components,
        "architecture": args.architecture,
        "kinematics": args.kinematics,
        "wrist_norm": args.wrist_norm,
    }
    train_ds = PoseDataset(train_rows, args.data_dir, class_to_idx, signer_to_idx, augment=augment, **dataset_kwargs)
    val_ds = PoseDataset(val_rows, args.data_dir, class_to_idx, signer_to_idx, **dataset_kwargs)
    test_ds = PoseDataset(test_rows, args.data_dir, class_to_idx, signer_to_idx, **dataset_kwargs)

    if len(train_ds) == 0:
        raise SystemExit("No training samples found after filtering classes.")

    frame_dim = feature_dim(args.components, kinematics=args.kinematics)
    input_dim = frame_dim if args.architecture != "mlp" else frame_dim * args.target_frames
    model = build_model(
        input_dim=frame_dim,
        num_classes=len(labels),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        architecture=args.architecture,
        target_frames=args.target_frames,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but this PyTorch install is CPU-only. "
            "Install a CUDA-enabled PyTorch wheel, then rerun training."
        )
    # Resume from a full checkpoint (model weights only — optimizer/scheduler restart)
    if args.resume_weights:
        ckpt = torch.load(args.resume_weights, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(json.dumps({"resume_weights": str(args.resume_weights), "missing": missing, "unexpected": unexpected}, ensure_ascii=False))

    # Load pretrained encoder from Slovo pretraining if requested
    if args.pretrained_encoder and not args.resume_weights:
        n_loaded = load_pretrained_encoder(model, args.pretrained_encoder)
        print(json.dumps({"pretrained_encoder": str(args.pretrained_encoder), "loaded_tensors": n_loaded}, ensure_ascii=False))
        if n_loaded == 0:
            raise SystemExit(
                "Pretrained encoder loaded 0 tensors. Check that --hidden-dim, "
                "--n-layers, --n-heads, --target-frames, --components, and "
                "--no-kinematics match the pretraining run."
            )
        if args.freeze_encoder_epochs > 0:
            for name, param in model.named_parameters():
                if not name.startswith("head."):
                    param.requires_grad = False
            print(json.dumps({"encoder_frozen_until_epoch": args.freeze_encoder_epochs}, ensure_ascii=False))

    model.to(device)
    use_cuda = device.startswith("cuda")
    use_amp = args.amp and use_cuda
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=args.compile_mode)

    signer_head = None
    grad_reverse = None
    if args.signer_adversarial:
        signer_head = torch.nn.Linear(model.feature_dim, len(signer_to_idx)).to(device)
        grad_reverse = make_grad_reverse(torch)

    print(
        json.dumps(
            {
                "device": device,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "torch_threads": torch.get_num_threads(),
                "num_workers": args.num_workers,
                "pin_memory": use_cuda,
                "amp": use_amp,
                "architecture": args.architecture,
                "components": args.components,
                "kinematics": args.kinematics,
                "frame_dim": frame_dim,
                "augment": bool(augment),
                "mixup_alpha": args.mixup_alpha,
                "label_smoothing": args.label_smoothing,
                "signer_adversarial": args.signer_adversarial,
                "weighted_sampling": args.weighted_sampling,
                "signer_balanced_sampling": args.signer_balanced_sampling,
                "wrist_norm": args.wrist_norm,
                "body_proportion_aug": args.body_proportion_aug,
                "resume_weights": str(args.resume_weights) if args.resume_weights else None,
                "model_parameters": sum(p.numel() for p in model.parameters()),
            },
            ensure_ascii=False,
        )
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_batch,
        "pin_memory": use_cuda,
        "persistent_workers": args.num_workers > 0,
        **({"prefetch_factor": 4} if args.num_workers > 0 else {}),
    }
    if args.weighted_sampling or args.signer_balanced_sampling:
        class_counts = Counter(class_to_idx[row["sign_id"]] for row in train_ds.rows)
        signer_counts = Counter(row.get("signer_id", "") for row in train_ds.rows)
        sample_weights = []
        for row in train_ds.rows:
            w = 1.0
            if args.weighted_sampling:
                w *= 1.0 / max(class_counts[class_to_idx[row["sign_id"]]], 1)
            if args.signer_balanced_sampling:
                w *= 1.0 / max(signer_counts[row.get("signer_id", "")], 1)
            sample_weights.append(w)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs) if len(val_ds) else None
    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, **loader_kwargs) if len(test_ds) else None

    parameters = list(model.parameters())
    if signer_head is not None:
        parameters += list(signer_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    signer_loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mixup_rng = np.random.default_rng(args.seed)

    best_state = None
    best_val = -1.0
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # Unfreeze encoder after the warm-up phase
        if args.pretrained_encoder and args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            # Reset optimizer so unfrozen params get proper momentum
            parameters = list(model.parameters())
            if signer_head is not None:
                parameters += list(signer_head.parameters())
            finetune_lr = args.lr * args.finetune_lr_scale
            optimizer = torch.optim.AdamW(parameters, lr=finetune_lr, weight_decay=args.weight_decay)
            remaining_steps = max(1, (args.epochs - epoch + 1) * len(train_loader))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=finetune_lr, total_steps=remaining_steps)
            print(json.dumps({"encoder_unfrozen_at_epoch": epoch, "lr": finetune_lr, "remaining_steps": remaining_steps}, ensure_ascii=False))

        model.train()
        if signer_head is not None:
            signer_head.train()
        running_loss = 0.0
        seen = 0
        bar = ProgressBar(len(train_loader), label=f"epoch {epoch}/{args.epochs}") if args.progress else None
        batch_index = 0
        for x, y, s in train_loader:
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            s = s.to(device, non_blocking=use_cuda)

            mixup_lam = 1.0
            perm = None
            if args.mixup_alpha > 0 and x.shape[0] > 1 and mixup_rng.random() < args.mixup_p:
                mixup_lam = float(mixup_rng.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(x.shape[0], device=device)
                x = mixup_lam * x + (1.0 - mixup_lam) * x[perm]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if signer_head is not None:
                    logits, features = model(x, return_features=True)
                else:
                    logits = model(x)
                if perm is not None:
                    loss = mixup_lam * loss_fn(logits, y) + (1.0 - mixup_lam) * loss_fn(logits, y[perm])
                else:
                    loss = loss_fn(logits, y)
                if signer_head is not None and epoch > args.grl_warmup_epochs:
                    # DANN-style lambda ramp. grl_lambda is bounded to [0, weight]
                    # so the adversarial contribution never exceeds the weight cap
                    # regardless of adv_loss magnitude — stable with large models.
                    warmup_steps = max(1, args.grl_warmup_epochs * len(train_loader))
                    post_warmup_steps = max(1, total_steps - warmup_steps)
                    post_warmup_global = global_step - warmup_steps
                    progress = max(0.0, post_warmup_global / post_warmup_steps)
                    grl_lambda = args.signer_adversarial_weight * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
                    signer_logits = signer_head(grad_reverse(features, grl_lambda))
                    if perm is not None:
                        adv_loss = mixup_lam * signer_loss_fn(signer_logits, s) + (1.0 - mixup_lam) * signer_loss_fn(signer_logits, s[perm])
                    else:
                        adv_loss = signer_loss_fn(signer_logits, s)
                    loss = loss + adv_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            running_loss += loss.item() * y.numel()
            seen += y.numel()
            batch_index += 1
            if bar:
                bar.update(batch_index, suffix=f"loss {running_loss / seen:.4f}")
        if bar:
            bar.finish(suffix=f"loss {running_loss / seen if seen else 0.0:.4f}")
        epoch_metrics = {"epoch": epoch, "train_loss": running_loss / seen if seen else 0.0, "lr": scheduler.get_last_lr()[0]}
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, len(labels))
            epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["top1"] > best_val:
                best_val = float(val_metrics["top1"])
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics: dict[str, object] = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "classes": len(labels),
        "history": history,
    }
    if val_loader is not None:
        metrics["val"] = evaluate(model, val_loader, device, len(labels))
        metrics["val_category_accuracy"] = category_accuracy(model, val_loader, val_ds.rows, class_to_idx, device)
        metrics["val_signer_accuracy"] = signer_accuracy(model, val_loader, val_ds.rows, device)
        if args.tta_views > 1:
            metrics["val_tta"] = evaluate_tta(model, val_loader, device, len(labels), components=args.components, tta_views=args.tta_views)

    prototypes = None
    if args.eval_prototypes:
        proto_ds = PoseDataset(train_rows, args.data_dir, class_to_idx, signer_to_idx, augment=None, **dataset_kwargs)
        proto_loader = torch.utils.data.DataLoader(proto_ds, shuffle=False, **loader_kwargs)
        prototypes = build_prototypes(model, proto_loader, device, len(labels))
        if prototypes is not None:
            if val_loader is not None:
                metrics["val_prototype"] = evaluate_prototypes(model, val_loader, prototypes, device)

    # Save checkpoint and config before test evaluation so a crash there doesn't lose training.
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "feature_version": 2,
        "input_dim": input_dim,
        "frame_dim": frame_dim,
        "target_frames": args.target_frames,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "split_column": args.split_column,
        "pose_schema": "mediapipe_pose_hands_v1",
        "trim_threshold": args.trim_threshold,
        "components": args.components,
        "kinematics": args.kinematics,
        "architecture": args.architecture,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "wrist_norm": args.wrist_norm,
        "attention_pool": True,
        "prototype_classifier": prototypes is not None,
    }
    checkpoint_path = args.artifact_dir / "checkpoint.pt"
    checkpoint = {
        "model_state": model.state_dict(),
        "labels": labels,
        "config": config,
    }
    if prototypes is not None:
        checkpoint["prototypes"] = prototypes.detach().cpu()
    torch.save(checkpoint, checkpoint_path)
    (args.artifact_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.artifact_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"checkpoint: {checkpoint_path}")

    if test_loader is not None:
        metrics["test"] = evaluate(model, test_loader, device, len(labels))
        metrics["test_category_accuracy"] = category_accuracy(model, test_loader, test_ds.rows, class_to_idx, device)
        metrics["test_signer_accuracy"] = signer_accuracy(model, test_loader, test_ds.rows, device)
        if args.tta_views > 1:
            metrics["test_tta"] = evaluate_tta(model, test_loader, device, len(labels), components=args.components, tta_views=args.tta_views)
        if prototypes is not None:
            metrics["test_prototype"] = evaluate_prototypes(model, test_loader, prototypes, device)

    (args.artifact_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics: {args.artifact_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UzSL v1 isolated sign recognizer.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--split-column", default="split", help="Manifest column that defines train/val/test splits. Any column name is accepted (e.g. dev_split, loso_s05).")
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--trim-threshold", type=float, default=0.03)
    parser.add_argument("--components", choices=["hands_pose", "rec", "full"], default="hands_pose")
    parser.add_argument("--architecture", choices=["conv_transformer", "transformer", "mlp"], default="conv_transformer")
    parser.add_argument("--no-kinematics", dest="kinematics", action="store_false", help="Drop the velocity/acceleration feature channels.")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable train-time augmentation.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for the OneCycle schedule.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="Beta(alpha, alpha) for mixup; 0 disables.")
    parser.add_argument("--mixup-p", type=float, default=0.5, help="Per-batch probability of applying mixup.")
    parser.add_argument("--pretrained-encoder", type=Path, default=None, help="Path to encoder.pt from uzsl_pretrain.py (Slovo pretraining).")
    parser.add_argument("--resume-weights", type=Path, default=None, help="Path to a checkpoint.pt to resume from (loads full model weights; optimizer/LR schedule restarts).")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5, help="Freeze encoder for this many epochs after loading pretrained weights, then unfreeze at 1/10 LR.")
    parser.add_argument("--finetune-lr-scale", type=float, default=0.1, help="Peak LR multiplier after unfreezing a pretrained encoder.")
    parser.add_argument("--signer-adversarial", action="store_true", help="Add a gradient-reversal signer classifier for signer-invariant features.")
    parser.add_argument("--signer-adversarial-weight", type=float, default=0.2, help="GRL lambda ceiling. The adversarial loss is scaled to never exceed this fraction of the sign loss magnitude.")
    parser.add_argument("--grl-warmup-epochs", type=int, default=15, help="Freeze the GRL head for this many epochs so the sign classifier warms up before adversarial training begins.")
    parser.add_argument("--num-workers", type=int, default=0, help="0 keeps the in-memory base-tensor cache in one process (recommended).")
    parser.add_argument("--cpu-threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision when CUDA is available.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode. reduce-overhead uses CUDA graphs (best for training). max-autotune finds fastest kernels but compiles slowly (~5-10 min).")
    parser.add_argument("--no-weighted-sampling", dest="weighted_sampling", action="store_false", help="Disable class-balanced sampling.")
    parser.add_argument("--signer-balanced-sampling", action="store_true", help="Weight samples so each signer contributes equally, regardless of recording count.")
    parser.add_argument("--wrist-norm", action="store_true", help="Normalize each hand block relative to wrist position and hand span (removes arm-length bias).")
    parser.add_argument("--body-proportion-aug", action="store_true", help="Augment arm length per sample to simulate signer body proportion variation.")
    parser.add_argument("--eval-prototypes", action="store_true", help="Build class centroids from train features and report/save a cosine prototype classifier.")
    parser.add_argument("--tta-views", type=int, default=1, help="Report deterministic test-time augmentation metrics; 2 adds mirrored hands_pose logits.")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True, kinematics=True, augment=True, weighted_sampling=True, signer_balanced_sampling=False, wrist_norm=False, body_proportion_aug=False)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
