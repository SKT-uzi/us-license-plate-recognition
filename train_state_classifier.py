"""Train a 51-class YOLOv8 US license-plate state classifier."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_DATASET = WORKSPACE / "state_classifier_dataset"
DEFAULT_MODEL = "yolov8s-cls.pt"
RUNS_DIR = WORKSPACE / "runs"
DEFAULT_RUN_NAME = "us_plate_state_yolov8s"

os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(WORKSPACE / "Ultralytics"))

ultralytics_source = os.environ.get("ULTRALYTICS_SOURCE")
if ultralytics_source:
    sys.path.insert(0, str(Path(ultralytics_source).expanduser().resolve()))

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.trainer import BaseTrainer  # noqa: E402
from ultralytics.models.yolo.classify.train import ClassificationTrainer  # noqa: E402


def plot_classification_results_without_polars(
    csv_path: Path, on_plot=None
) -> None:
    """Create the standard four classification curves using only csv/matplotlib."""
    import matplotlib.pyplot as plt

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return

    panels = (
        ("train/loss", "Training loss"),
        ("val/loss", "Validation loss"),
        ("metrics/accuracy_top1", "Top-1 accuracy"),
        ("metrics/accuracy_top5", "Top-5 accuracy"),
    )
    epochs = [float(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), tight_layout=True)
    for axis, (column, title) in zip(axes.ravel(), panels):
        values = [float(row[column]) for row in rows]
        axis.plot(epochs, values, marker=".", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    output_path = csv_path.parent / "results.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    if on_plot:
        on_plot(output_path)


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

        def plot_metrics_without_polars(trainer: ClassificationTrainer) -> None:
            plot_classification_results_without_polars(
                trainer.csv, on_plot=trainer.on_plot
            )

        ClassificationTrainer.plot_metrics = plot_metrics_without_polars
        print(
            "Polars unavailable; enabled CSV checkpoint and matplotlib plot fallbacks."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--project", type=Path, default=RUNS_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    return parser.parse_args()


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    model_source = Path(args.model).resolve() if Path(args.model).is_file() else args.model
    runs_dir = args.project.resolve()
    require_directory(data_path / "train", "Training split")
    require_directory(data_path / "val", "Validation split")
    require_directory(data_path / "test", "Test split")
    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is unavailable in the selected Python environment.")

    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"Ultralytics source:  {Path(ultralytics.__file__).resolve()}")
    print(f"PyTorch version:     {torch.__version__}")
    print(
        "CUDA device:        "
        + (
            torch.cuda.get_device_name(int(args.device))
            if args.device != "cpu"
            else "CPU"
        )
    )
    print(f"Dataset:            {data_path}")
    print(f"Initial model:      {model_source}")
    print(f"Run directory:      {runs_dir / args.name}")
    print(
        "Training parameters: "
        f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
        f"device={args.device}, patience={args.patience}, workers={args.workers}, "
        f"optimizer={args.optimizer}, lr0={args.lr0}, lrf={args.lrf}, "
        f"weight_decay={args.weight_decay}, warmup_epochs={args.warmup_epochs}"
    )

    enable_csv_fallback_if_needed()
    model = YOLO(str(model_source))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(runs_dir),
        name=args.name,
        exist_ok=False,
        val=True,
        plots=True,
        patience=args.patience,
        pretrained=True,
        amp=True,
        seed=42,
        deterministic=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        dropout=0.1,
        flipud=0.0,
        fliplr=0.0,
        degrees=3.0,
        translate=0.05,
        scale=0.15,
    )


if __name__ == "__main__":
    main()
