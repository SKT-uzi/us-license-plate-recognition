"""Verify shipped model hashes, dependencies, and model loading."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CHECKSUM_FILE = MODELS / "SHA256SUMS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Verify files without importing machine-learning packages",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes() -> None:
    if not CHECKSUM_FILE.is_file():
        raise FileNotFoundError(CHECKSUM_FILE)
    checked = 0
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, relative_name = line.split(maxsplit=1)
        target = MODELS / relative_name.strip().lstrip("*")
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = sha256(target)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"Checksum mismatch: {target.name}")
        checked += 1
        print(f"checksum ok: {target.name}")
    print(f"Verified {checked} model artifacts.")


def verify_dependencies_and_models() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

    import cv2
    import numpy
    import paddleocr
    import PIL
    import torch
    import ultralytics
    from fast_plate_ocr import LicensePlateRecognizer
    from ultralytics import YOLO

    versions = {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "Pillow": PIL.__version__,
        "PaddleOCR": getattr(paddleocr, "__version__", "installed"),
    }
    for package, version in versions.items():
        print(f"{package}: {version}")

    expected_tasks = {
        "pose_baseline.pt": "pose",
        "pose_finetuned.pt": "pose",
        "state_classifier.pt": "classify",
    }
    for filename, expected_task in expected_tasks.items():
        model = YOLO(str(MODELS / filename))
        if model.task != expected_task:
            raise RuntimeError(
                f"Unexpected task for {filename}: {model.task!r}; expected {expected_task!r}"
            )
        print(f"model load ok: {filename} ({model.task})")

    LicensePlateRecognizer(
        device="cpu",
        onnx_model_path=MODELS / "cct_s_v2_global.onnx",
        plate_config_path=MODELS / "cct_s_v2_global_plate_config.yaml",
    )
    print("model load ok: cct_s_v2_global.onnx (plate OCR)")


def main() -> None:
    args = parse_args()
    verify_hashes()
    if not args.hash_only:
        verify_dependencies_and_models()
    print("Installation verification passed.")


if __name__ == "__main__":
    main()
