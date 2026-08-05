"""Build a mixed clean/road-degraded dataset for US plate-state classification.

The original train/val/test files are preserved. Originals are hard-linked into
the new dataset when possible, and only the training split receives synthetic
road-camera variants. Validation and test therefore remain honest clean-image
benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = WORKSPACE / "state_classifier_dataset"
DEFAULT_OUTPUT = WORKSPACE / "state_classifier_dataset_road_aug"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--augment-fraction",
        type=float,
        default=0.75,
        help="Fraction of each class that receives one degraded training variant.",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: str(path).casefold(),
    )


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def stable_seed(path: Path, base_seed: int) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}|{path.as_posix()}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def motion_blur(image: Image.Image, rng: random.Random) -> Image.Image:
    size = rng.choice((3, 5))
    weights = [0.0] * (size * size)
    pattern = rng.choice(("horizontal", "diag_down", "diag_up"))
    for index in range(size):
        if pattern == "horizontal":
            row, column = size // 2, index
        elif pattern == "diag_down":
            row, column = index, index
        else:
            row, column = index, size - 1 - index
        weights[row * size + column] = 1.0
    return image.filter(ImageFilter.Kernel((size, size), weights, scale=1.0))


def add_glare_or_occlusion(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if rng.random() < 0.65:
        stripe_width = max(2, int(width * rng.uniform(0.025, 0.08)))
        start_x = rng.randint(-stripe_width, max(width - 1, 0))
        draw.polygon(
            [
                (start_x, 0),
                (start_x + stripe_width, 0),
                (
                    start_x + stripe_width + int(height * rng.uniform(0.1, 0.45)),
                    height,
                ),
                (start_x + int(height * rng.uniform(0.1, 0.45)), height),
            ],
            fill=(255, 255, 245, rng.randint(45, 105)),
        )
    else:
        radius = max(1, int(min(width, height) * rng.uniform(0.04, 0.11)))
        center_x = rng.randint(radius, max(radius, width - radius))
        center_y = rng.randint(radius, max(radius, height - radius))
        shade = rng.randint(15, 80)
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=(shade, shade, shade, rng.randint(100, 190)),
        )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def road_degrade(source: Path, destination: Path, seed: int) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    with Image.open(source) as loaded:
        image = ImageOps.exif_transpose(loaded).convert("RGB")

    original_size = image.size
    width, height = original_size
    max_side = max(width, height)

    difficulty = rng.random()
    if difficulty < 0.20:
        target_long_side = rng.randint(50, 82)
    elif difficulty < 0.75:
        target_long_side = rng.randint(82, 132)
    else:
        target_long_side = rng.randint(132, 190)
    downscale = min(1.0, target_long_side / max(max_side, 1))
    low_size = (
        max(18, round(width * downscale)),
        max(12, round(height * downscale)),
    )
    image = image.resize(low_size, Image.Resampling.LANCZOS)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.60, 1.22))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.58, 1.38))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.48, 1.22))

    if rng.random() < 0.72:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.35, 1.15)))
    if rng.random() < 0.48:
        image = motion_blur(image, rng)
    if rng.random() < 0.34:
        image = add_glare_or_occlusion(image, rng)

    if rng.random() < 0.60:
        pixels = np.asarray(image, dtype=np.int16)
        sigma = rng.uniform(2.0, 10.0)
        noise = np_rng.normal(0.0, sigma, pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")

    image = jpeg_roundtrip(image, rng.randint(24, 62))
    interpolation = rng.choice(
        (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
    )
    image = image.resize(original_size, interpolation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=82, optimize=True)


def make_preview(pairs: list[tuple[Path, Path]], output: Path) -> None:
    if not pairs:
        return
    tile_width, tile_height = 320, 170
    columns = min(4, len(pairs))
    canvas = Image.new("RGB", (columns * tile_width, 2 * tile_height), "white")
    for column, (clean_path, degraded_path) in enumerate(pairs[:columns]):
        for row, path in enumerate((clean_path, degraded_path)):
            with Image.open(path) as loaded:
                image = ImageOps.contain(
                    ImageOps.exif_transpose(loaded).convert("RGB"),
                    (tile_width - 12, tile_height - 12),
                    Image.Resampling.LANCZOS,
                )
            x = column * tile_width + (tile_width - image.width) // 2
            y = row * tile_height + (tile_height - image.height) // 2
            canvas.paste(image, (x, y))
    canvas.save(output, quality=90)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not 0.0 <= args.augment_fraction <= 1.0:
        raise ValueError("--augment-fraction must be between 0 and 1")
    for split in ("train", "val", "test"):
        if not (source / split).is_dir():
            raise FileNotFoundError(f"Missing source split: {source / split}")
    if output.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite it: {output}"
        )

    staging = output.with_name(f"{output.name}.building-{os.getpid()}")
    staging.mkdir(parents=True)
    started = time.time()
    link_counts = {"hardlink": 0, "copy": 0}
    split_counts: dict[str, dict[str, int]] = {}
    augmentation_jobs: list[tuple[Path, Path, int]] = []

    try:
        for split in ("train", "val", "test"):
            split_source = source / split
            classes = sorted(
                (path for path in split_source.iterdir() if path.is_dir()),
                key=lambda path: path.name.casefold(),
            )
            original_count = 0
            augmented_count = 0
            for class_dir in classes:
                files = image_files(class_dir)
                destination_class = staging / split / class_dir.name
                for source_path in files:
                    relative = source_path.relative_to(class_dir)
                    destination = destination_class / relative
                    link_mode = link_or_copy(source_path, destination)
                    link_counts[link_mode] += 1
                    original_count += 1

                if split == "train" and files:
                    class_rng = random.Random(
                        stable_seed(Path(split) / class_dir.name, args.seed)
                    )
                    selected = files.copy()
                    class_rng.shuffle(selected)
                    selected_count = round(len(selected) * args.augment_fraction)
                    if args.augment_fraction > 0:
                        selected_count = max(1, selected_count)
                    for source_path in selected[:selected_count]:
                        relative = source_path.relative_to(class_dir)
                        augmented_name = (
                            f"{relative.stem}__roadaug_{args.seed}.jpg"
                        )
                        destination = (
                            destination_class
                            / relative.parent
                            / augmented_name
                        )
                        augmentation_jobs.append(
                            (
                                source_path,
                                destination,
                                stable_seed(
                                    Path(split)
                                    / class_dir.name
                                    / relative,
                                    args.seed,
                                ),
                            )
                        )
                        augmented_count += 1
            split_counts[split] = {
                "originals": original_count,
                "augmented": augmented_count,
                "total": original_count + augmented_count,
                "classes": len(classes),
            }

        completed = 0
        total_jobs = len(augmentation_jobs)

        def run_job(job: tuple[Path, Path, int]) -> tuple[Path, Path]:
            source_path, destination, seed = job
            road_degrade(source_path, destination, seed)
            return source_path, destination

        preview_pairs: list[tuple[Path, Path]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for clean_path, degraded_path in executor.map(run_job, augmentation_jobs):
                completed += 1
                if len(preview_pairs) < 4:
                    preview_pairs.append((clean_path, degraded_path))
                if completed % 500 == 0 or completed == total_jobs:
                    print(
                        f"Generated {completed}/{total_jobs} degraded images "
                        f"({completed / max(total_jobs, 1):.1%})",
                        flush=True,
                    )

        manifest = {
            "source": str(source),
            "output": str(output),
            "seed": args.seed,
            "augment_fraction": args.augment_fraction,
            "split_counts": split_counts,
            "original_file_mode": link_counts,
            "elapsed_seconds": round(time.time() - started, 2),
            "degradation": {
                "downsample_long_side_px": "50-190",
                "brightness": "0.60-1.22",
                "contrast": "0.58-1.38",
                "color": "0.48-1.22",
                "gaussian_blur_probability": 0.72,
                "motion_blur_probability": 0.48,
                "glare_or_occlusion_probability": 0.34,
                "noise_probability": 0.60,
                "jpeg_quality": "24-62",
            },
        }
        (staging / "road_augmentation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        make_preview(preview_pairs, staging / "road_augmentation_preview.jpg")
        staging.rename(output)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception:
        print(f"Build failed; incomplete staging directory kept at: {staging}")
        raise


if __name__ == "__main__":
    main()
