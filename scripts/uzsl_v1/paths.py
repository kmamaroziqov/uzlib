from __future__ import annotations

from pathlib import Path

# Read the single user-editable config (repo root/config.py).
# Falls back gracefully if config is absent (e.g. during CI or fresh clone).
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "project_config",
        Path(__file__).resolve().parents[2] / "config.py",
    )
    _mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    DEFAULT_DATA_DIR: Path = Path(_mod.DATA_DIR)
except Exception:
    DEFAULT_DATA_DIR = Path("uzsl_data")

DEFAULT_MANIFEST = Path("experiments/new_dataset/manifests/train_manifest_loso.csv")
DEFAULT_UNSUPPORTED = DEFAULT_DATA_DIR / "generated" / "manifests" / "unsupported_signs.csv"
DEFAULT_POSE_DIR = DEFAULT_DATA_DIR / "poses"
DEFAULT_ARTIFACT_DIR = Path("artifacts") / "uzsl_v1"


def repo_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def video_stem(video_path: str) -> str:
    return Path(video_path).stem
