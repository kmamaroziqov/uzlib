from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .predict import load_checkpoint, predict_scores
from .train import PoseDataset, collate_batch, label_metadata, read_manifest


def _macro_f1(gold_counts: Counter, predicted_counts: Counter, true_positive: Counter, num_classes: int) -> float:
    f1_values: list[float] = []
    for class_idx in range(num_classes):
        tp = true_positive[class_idx]
        fp = predicted_counts[class_idx] - tp
        fn = gold_counts[class_idx] - tp
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def evaluate_ensemble(args: argparse.Namespace) -> dict[str, object]:
    torch = __import__("torch")
    rows = read_manifest(args.manifest)
    train_rows = [row for row in rows if row.get(args.split_column) == "train"]
    eval_rows = [row for row in rows if row.get(args.split_column) == args.split]
    labels = label_metadata(rows, train_rows)
    class_to_idx = {row["sign_id"]: idx for idx, row in enumerate(labels)}
    signer_to_idx = {signer: idx for idx, signer in enumerate(sorted({row.get("signer_id", "") for row in train_rows}))}
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    members = []
    label_ids = [row["sign_id"] for row in labels]
    for checkpoint_path in args.checkpoints:
        model, checkpoint_labels, config, prototypes = load_checkpoint(checkpoint_path, device)
        checkpoint_label_ids = [row["sign_id"] for row in checkpoint_labels]
        if checkpoint_label_ids != label_ids:
            raise SystemExit(f"Label mismatch for {checkpoint_path}")
        if args.classifier in ("prototype", "ensemble") and prototypes is None:
            raise SystemExit(f"{checkpoint_path} does not contain prototypes")
        members.append((model, config, prototypes, str(checkpoint_path)))

    config0 = members[0][1]
    dataset = PoseDataset(
        eval_rows,
        args.data_dir,
        class_to_idx,
        signer_to_idx,
        target_frames=int(config0.get("target_frames", 64)),
        trim_threshold=float(config0.get("trim_threshold", 0.03)),
        components=config0.get("components", "hands_pose"),
        architecture=config0.get("architecture", "conv_transformer"),
        kinematics=bool(config0.get("kinematics", True)),
        wrist_norm=bool(config0.get("wrist_norm", False)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    total = 0
    correct1 = 0
    correctk = 0
    predicted_counts = Counter()
    gold_counts = Counter()
    true_positive = Counter()
    signer_totals = Counter()
    signer_correct = Counter()
    row_index = 0
    top_k = min(args.top_k, len(labels))
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            probs = None
            for model, config, prototypes, _ in members:
                scores = predict_scores(model, x, config, prototypes, args.classifier, args.tta_views)
                if args.classifier == "ensemble":
                    member_probs = scores / scores.sum(dim=1, keepdim=True).clamp(min=1e-12)
                else:
                    member_probs = torch.softmax(scores, dim=1)
                probs = member_probs if probs is None else probs + member_probs
            probs = probs / len(members)
            top = probs.topk(top_k, dim=1).indices
            pred = top[:, 0]
            total += y.numel()
            correct1 += (pred == y).sum().item()
            correctk += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            for p, g in zip(pred.cpu().tolist(), y.cpu().tolist()):
                predicted_counts[p] += 1
                gold_counts[g] += 1
                if p == g:
                    true_positive[g] += 1
            for p, g in zip(pred.cpu().tolist(), y.cpu().tolist()):
                signer = dataset.rows[row_index].get("signer_id", "")
                row_index += 1
                signer_totals[signer] += 1
                if p == g:
                    signer_correct[signer] += 1

    metrics: dict[str, object] = {
        "split": args.split,
        "classifier": args.classifier,
        "tta_views": args.tta_views,
        "checkpoints": [member[3] for member in members],
        "samples": total,
        "classes": len(labels),
        "top1": correct1 / total if total else 0.0,
        "top5": correctk / total if total else 0.0,
        "macro_f1": _macro_f1(gold_counts, predicted_counts, true_positive, len(labels)),
        "mode_collapse_share": max(predicted_counts.values()) / total if total else 0.0,
        "signer_accuracy": {
            signer: signer_correct[signer] / total_count
            for signer, total_count in sorted(signer_totals.items())
            if signer
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a probability ensemble of UzSL checkpoints.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-column", choices=["split", "signer_holdout_split"], default="signer_holdout_split")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--classifier", choices=["softmax", "prototype", "ensemble"], default="softmax")
    parser.add_argument("--tta-views", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate_ensemble(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
