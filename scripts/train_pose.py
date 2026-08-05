"""Train the US license-plate four-corner pose model."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_YAML = WORKSPACE / "configs" / "plate_pose.yaml"
INITIAL_MODEL = WORKSPACE / "models" / "pose_baseline.pt"
RUNS_DIR = WORKSPACE / "runs"
DEFAULT_RUN_NAME = "plate_pose_100epoch_baseline"

# Keep Ultralytics settings and all generated training artifacts in the workspace.
# This Ultralytics version appends its own "Ultralytics" subdirectory.
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))

ultralytics_source = os.environ.get("ULTRALYTICS_SOURCE")
if ultralytics_source:
    sys.path.insert(0, str(Path(ultralytics_source).expanduser().resolve()))

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.trainer import BaseTrainer  # noqa: E402


def enable_csv_fallback_if_needed() -> None:
    """Allow this source revision to save checkpoints when polars is absent."""
    try:
        import polars  # noqa: F401
    except ModuleNotFoundError:

        def read_results_csv_without_polars(trainer: BaseTrainer) -> dict[str, list]:
            with trainer.csv.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            columns: dict[str, list] = {}
            if not rows:
                return columns

            for column_name in rows[0]:
                values: list = []
                for row in rows:
                    value = row[column_name].strip()
                    try:
                        values.append(float(value))
                    except ValueError:
                        values.append(value)
                columns[column_name] = values
            return columns

        BaseTrainer.read_results_csv = read_results_csv_without_polars
        print("Polars is unavailable; using the standard-library CSV checkpoint fallback.")


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--model", type=Path, default=INITIAL_MODEL)
    parser.add_argument("--project", type=Path, default=RUNS_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    initial_model = args.model.resolve()
    runs_dir = args.project.resolve()
    require_path(data_yaml, "dataset YAML")
    require_path(initial_model, "initial model")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment.")

    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"Ultralytics source:  {Path(ultralytics.__file__).resolve()}")
    print(f"PyTorch version:     {torch.__version__}")
    print(f"CUDA device 0:       {torch.cuda.get_device_name(0)}")
    print(f"Dataset YAML:        {data_yaml}")
    print(f"Initial model:       {initial_model}")
    print(f"Run directory:       {runs_dir / args.name}")
    print(
        "Training parameters: "
        f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
        f"device={args.device}, patience={args.patience}"
    )

    enable_csv_fallback_if_needed()
    model = YOLO(str(initial_model))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        amp=False,
        project=str(runs_dir),
        name=args.name,
        exist_ok=False,
        val=True,
        plots=True,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
