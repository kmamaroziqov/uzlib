from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .extract_poses import extract_video
from .features import finalize, prepare_base
from .model import build_model, require_torch
from .paths import DEFAULT_ARTIFACT_DIR
from .pose_io import normalized_flat_features, normalized_sequence_features, read_pose_file


POSE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def load_checkpoint(checkpoint_path: Path, device: str):
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    labels = checkpoint["labels"]
    model = build_model(
        input_dim=config.get("frame_dim", config["input_dim"]),
        num_classes=len(labels),
        hidden_dim=config.get("hidden_dim", 512),
        dropout=config.get("dropout", 0.25),
        architecture=config.get("architecture", "mlp"),
        target_frames=config.get("target_frames", 64),
        n_layers=config.get("n_layers", 4),
        n_heads=config.get("n_heads", 8),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    prototypes = checkpoint.get("prototypes")
    if prototypes is not None:
        prototypes = prototypes.to(device)
    return model, labels, config, prototypes


def flip_hands_pose_tensor(x):
    torch = require_torch()
    channels = x.shape[-1] // 75
    if channels * 75 != x.shape[-1]:
        return x
    z = x.clone().reshape(x.shape[0], x.shape[1], 75, channels)
    perm = torch.arange(75, device=x.device)
    perm[0:21], perm[54:75] = torch.arange(54, 75, device=x.device), torch.arange(0, 21, device=x.device)
    for a, b in POSE_LR_PAIRS:
        perm[21 + a], perm[21 + b] = 21 + b, 21 + a
    z = z[:, :, perm, :]
    x_channels = [0]
    if channels >= 8:
        x_channels.extend([4, 7])  # velocity-x and acceleration-x
    z[..., x_channels] = -z[..., x_channels]
    return z.reshape_as(x)


def predict_scores(model, x, config: dict, prototypes, classifier: str, tta_views: int):
    torch = require_torch()
    views = [x]
    if tta_views > 1 and config.get("components", "hands_pose") == "hands_pose" and config.get("architecture") != "mlp":
        views.append(flip_hands_pose_tensor(x))

    logits = None
    proto_logits = None
    for view in views:
        if classifier in ("softmax", "ensemble") or prototypes is None:
            current = model(view)
            logits = current if logits is None else logits + current
        if prototypes is not None and classifier in ("prototype", "ensemble"):
            _, features = model(view, return_features=True)
            features = torch.nn.functional.normalize(features, dim=1)
            current_proto = features @ prototypes.T
            proto_logits = current_proto if proto_logits is None else proto_logits + current_proto

    if logits is not None:
        logits = logits / len(views)
    if proto_logits is not None:
        proto_logits = proto_logits / len(views)
    if classifier == "prototype" and proto_logits is not None:
        return proto_logits
    if classifier == "ensemble" and proto_logits is not None and logits is not None:
        return torch.softmax(logits, dim=1) + torch.softmax(proto_logits, dim=1)
    return logits if logits is not None else proto_logits


def predict_pose(
    checkpoint_path: Path,
    pose_path: Path,
    top_k: int,
    device: str,
    *,
    classifier: str = "softmax",
    tta_views: int = 1,
) -> list[dict[str, float | str]]:
    torch = require_torch()
    model, labels, config, prototypes = load_checkpoint(checkpoint_path, device)
    if classifier in ("prototype", "ensemble") and prototypes is None:
        raise SystemExit("This checkpoint does not contain prototypes. Re-train with --eval-prototypes or use --classifier softmax.")
    if int(config.get("feature_version", 1)) >= 2:
        base = prepare_base(
            pose_path,
            config.get("components", "hands_pose"),
            trim_threshold=float(config.get("trim_threshold", 0.03)),
            wrist_norm=bool(config.get("wrist_norm", False)),
        )
        features = finalize(
            base,
            int(config["target_frames"]),
            kinematics=bool(config.get("kinematics", False)),
        )
        if config.get("architecture") == "mlp":
            features = features.reshape(-1)
        x = torch.from_numpy(features).unsqueeze(0).to(device)
    else:
        payload = read_pose_file(pose_path)
        if config.get("architecture", "mlp") == "mlp":
            features = normalized_flat_features(
                payload,
                int(config["target_frames"]),
                trim_threshold=float(config.get("trim_threshold", 0.03)),
                components=config.get("components", "hands_pose"),
            )
        else:
            features = normalized_sequence_features(
                payload,
                int(config["target_frames"]),
                trim_threshold=float(config.get("trim_threshold", 0.03)),
                components=config.get("components", "hands_pose"),
            )
        x = torch.tensor([features], dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = predict_scores(model, x, config, prototypes, classifier, tta_views)
        probabilities = torch.softmax(scores, dim=1)[0] if classifier != "ensemble" else scores[0] / scores[0].sum().clamp(min=1e-8)
        k = min(top_k, probabilities.numel())
        values, indices = probabilities.topk(k)
    predictions = []
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        label = labels[index]
        predictions.append(
            {
                "sign_id": label["sign_id"],
                "label_uz": label["label_uz"],
                "label_ru": label.get("label_ru", ""),
                "category": label.get("category", ""),
                "confidence": float(value),
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict isolated UzSL labels from a video or generated pose JSON.")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--pose", type=Path, default=None, help="Use an existing generated .pose file.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARTIFACT_DIR / "checkpoint.pt")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", choices=["json", "text"], default="json")
    parser.add_argument("--device", default="")
    parser.add_argument("--delegate", choices=["cpu", "gpu"], default="cpu", help="MediaPipe delegate for --video pose extraction.")
    parser.add_argument("--classifier", choices=["softmax", "prototype", "ensemble"], default="softmax")
    parser.add_argument("--tta-views", type=int, default=1, help="2 averages original + mirrored hands_pose views.")
    args = parser.parse_args()

    if args.pose is None and args.video is None:
        raise SystemExit("Provide --video or --pose.")
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}. Train first with uzsl_train.py.")

    torch = require_torch()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.pose is not None:
        predictions = predict_pose(args.checkpoint, args.pose, args.top_k, device, classifier=args.classifier, tta_views=args.tta_views)
    else:
        with tempfile.TemporaryDirectory(prefix="uzsl_predict_") as tmp:
            pose_path = Path(tmp) / f"{args.video.stem}.pose"
            extract_video(args.video, pose_path, delegate=args.delegate)
            predictions = predict_pose(args.checkpoint, pose_path, args.top_k, device, classifier=args.classifier, tta_views=args.tta_views)

    if args.output == "json":
        print(json.dumps({"predictions": predictions}, ensure_ascii=False, indent=2))
    else:
        for item in predictions:
            print(f"{item['confidence']:.4f}\t{item['sign_id']}\t{item['label_uz']}")


if __name__ == "__main__":
    main()
