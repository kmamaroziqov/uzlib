"""
UzSL experiment runner — single entrypoint.

Usage:
  python run.py setup          # generate LOSO splits from manifest
  python run.py e0             # E0: strong baseline (dev split)
  python run.py e1             # E1: + signer adversarial GRL
  python run.py e2             # E2: + body-proportion aug
  python run.py e3             # E3: + pretrained Slovo encoder
  python run.py loso           # Final LOSO (8 folds, all signers as test)
  python run.py status         # print latest epoch from every artifact dir

Edit config.py at the repo root to set DATA_DIR before running anything.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# ── Load the single user config ──────────────────────────────────────────────
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("config", Path(__file__).parent / "config.py")
    _mod = _ilu.module_from_spec(_spec)      # type: ignore[arg-type]
    _spec.loader.exec_module(_mod)           # type: ignore[union-attr]
    DATA_DIR = Path(_mod.DATA_DIR)
except Exception as exc:
    sys.exit(f"Could not load config.py: {exc}\nEdit config.py at the repo root and set DATA_DIR.")

MANIFEST     = Path("experiments/new_dataset/manifests/train_manifest_loso.csv")
MANIFEST_SRC = Path("experiments/new_dataset/manifests/train_manifest_available.csv")
SLOVO_ENCODER = Path("artifacts/slovo_pretrain/encoder.pt")

# ── Detect GPU capability ─────────────────────────────────────────────────────
def _detect_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0

VRAM = _detect_vram_gb()

def _base_capacity() -> list[str]:
    """Scale model size, batch, and workers to available VRAM.

    RTX 5090 / A100 / H100 (>=24 GB):
      hidden 768, 6 layers, 8 heads → ~22M params, fully occupies 32 GB at batch 1024.
      8 data-loading workers keep the GPU fed between batches.
      reduce-overhead compile mode uses CUDA graphs — best throughput for fixed batch sizes.

    Mid-range (8-24 GB):
      hidden 384, 4 workers.

    Low VRAM / CPU:
      hidden 192, no workers (in-memory cache is most efficient single-threaded).
    """
    if VRAM >= 24:
        return [
            "--hidden-dim", "768", "--n-layers", "6", "--n-heads", "8",
            "--batch-size", "1024", "--num-workers", "8",
            "--compile", "--compile-mode", "reduce-overhead",
        ]
    elif VRAM >= 8:
        return [
            "--hidden-dim", "384", "--n-layers", "4", "--n-heads", "4",
            "--batch-size", "256", "--num-workers", "4",
            "--compile", "--compile-mode", "reduce-overhead",
        ]
    else:
        return ["--hidden-dim", "192", "--batch-size", "32"]


def _lr_flags() -> list[str]:
    """Scale peak LR with batch size (sqrt scaling rule for AdamW + OneCycleLR)."""
    if VRAM >= 24:
        return ["--lr", "2e-3"]   # batch 1024 vs default 256 → sqrt(4)× ≈ 2×
    elif VRAM >= 8:
        return ["--lr", "1e-3"]
    else:
        return ["--lr", "5e-4"]


def _device_flags() -> list[str]:
    return ["--amp", "--device", "cuda"] if VRAM > 0 else ["--device", "cpu"]

# ── Shared base flags for every training run ─────────────────────────────────
BASE = [
    sys.executable, "-u", "-m", "scripts.uzsl_v1.train",
    "--manifest", str(MANIFEST),
    "--data-dir", str(DATA_DIR),
    "--wrist-norm", "--signer-balanced-sampling",
    "--epochs", "120",
    "--no-progress",
    *_base_capacity(),
    *_lr_flags(),
    *_device_flags(),
]

def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"[run.py] process exited {ret.returncode}", flush=True)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_setup() -> None:
    """Generate LOSO split columns in the manifest."""
    if not MANIFEST_SRC.exists():
        sys.exit(f"Source manifest not found: {MANIFEST_SRC}")
    run([
        sys.executable, "-m", "scripts.uzsl_v1.make_loso_splits",
        "--in",  str(MANIFEST_SRC),
        "--out", str(MANIFEST),
    ])
    print(f"\nManifest written to {MANIFEST}")
    print("Open it to verify eval_vocab, dev_split, and loso_s0N columns.")


def cmd_e0() -> None:
    """E0: full-data baseline on dev split."""
    run([
        *BASE,
        "--artifact-dir", "artifacts/e0_baseline",
        "--split-column", "dev_split",
    ])


def cmd_e1() -> None:
    """E1: + signer-adversarial GRL, weight 0.02, warmup 15 epochs.

    The GRL is silent for the first 15 epochs so the sign classifier warms up
    before adversarial training begins. The lambda is bounded by the weight so
    it never overwhelms the sign loss regardless of model size.
    """
    run([
        *BASE,
        "--artifact-dir", "artifacts/e1_grl",
        "--split-column", "dev_split",
        "--signer-adversarial",
        "--signer-adversarial-weight", "0.02",
        "--grl-warmup-epochs", "15",
    ])


def cmd_e2() -> None:
    """E2: + body-proportion augmentation (simulates signer body-size variation)."""
    run([
        *BASE,
        "--artifact-dir", "artifacts/e2_body_aug",
        "--split-column", "dev_split",
        "--signer-adversarial", "--signer-adversarial-weight", "0.05",
        "--body-proportion-aug",
    ])


def cmd_e3() -> None:
    """E3: + Slovo RSL encoder transfer (co-pretrained; sequential fine-tune)."""
    if not SLOVO_ENCODER.exists():
        print(f"WARNING: Slovo encoder not found at {SLOVO_ENCODER}")
        print("To pretrain: python -m scripts.uzsl_v1.pretrain [Slovo flags]")
        print("Continuing without pretrained encoder...")
    run([
        *BASE,
        "--artifact-dir", "artifacts/e3_slovo_transfer",
        "--split-column", "dev_split",
        "--signer-adversarial", "--signer-adversarial-weight", "0.05",
        "--body-proportion-aug",
        *(["--pretrained-encoder", str(SLOVO_ENCODER), "--freeze-encoder-epochs", "5"]
          if SLOVO_ENCODER.exists() else []),
    ])


def cmd_loso() -> None:
    """Final LOSO: run all 8 folds with the best config found in E0-E3."""
    best_art = _pick_best_artifact()
    if best_art:
        print(f"Using config from {best_art} as the LOSO template.")
    run([
        sys.executable, "-u", "-m", "scripts.uzsl_v1.run_loso",
        "--manifest", str(MANIFEST),
        "--data-dir", str(DATA_DIR),
        "--out-dir", "artifacts/loso_final",
        "--wrist-norm", "--signer-balanced-sampling",
        "--signer-adversarial", "--signer-adversarial-weight", "0.05",
        "--body-proportion-aug",
        "--epochs", "120",
        "--no-progress",
        *_base_capacity(),
        *_device_flags(),
    ])


def cmd_status() -> None:
    """Print latest epoch and val_top1 for every artifact dir."""
    art_root = Path("artifacts")
    if not art_root.exists():
        print("No artifacts/ directory found.")
        return
    print(f"\n{'Artifact':<35}  {'epoch':>6}  {'val_top1':>9}  {'test_top1':>10}")
    print("-" * 68)
    for log in sorted(art_root.glob("*/train.log")):
        lines = [l for l in log.read_text(encoding="utf-8").splitlines() if '"epoch"' in l]
        if not lines:
            continue
        try:
            d = json.loads(lines[-1])
            ep  = d.get("epoch", "?")
            v1  = f"{d['val_top1']:.4f}" if "val_top1" in d else "  N/A "
        except Exception:
            ep, v1 = "?", "?"
        metrics = log.parent / "metrics.json"
        t1 = "  N/A "
        if metrics.exists():
            try:
                m = json.loads(metrics.read_text())
                t1 = f"{m['test']['top1']:.4f}" if "test" in m else "  N/A "
            except Exception:
                pass
        print(f"  {log.parent.name:<33}  {str(ep):>6}  {v1:>9}  {t1:>10}")


def _pick_best_artifact() -> Path | None:
    best_path, best_top1 = None, -1.0
    for mf in Path("artifacts").glob("*/metrics.json"):
        try:
            m = json.loads(mf.read_text())
            t1 = m.get("test", {}).get("top1", -1.0) or -1.0
            if t1 > best_top1:
                best_top1, best_path = t1, mf.parent
        except Exception:
            pass
    return best_path


# ── Dispatch ──────────────────────────────────────────────────────────────────

COMMANDS = {
    "setup":  cmd_setup,
    "e0":     cmd_e0,
    "e1":     cmd_e1,
    "e2":     cmd_e2,
    "e3":     cmd_e3,
    "loso":   cmd_loso,
    "status": cmd_status,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Commands:", ", ".join(COMMANDS))
        sys.exit(1)
    print(f"DATA_DIR = {DATA_DIR}  |  VRAM = {VRAM:.1f} GB", flush=True)
    COMMANDS[sys.argv[1]]()
