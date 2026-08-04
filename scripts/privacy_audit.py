"""Scan tracked release files and checkpoint metadata for common privacy leaks."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
PATH_RE = re.compile(r"(?:\b[A-Za-z]:\\|/(?:Users|home|root|mnt)/)")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
SECRET_RE = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)


def deny_pattern(term: str) -> re.Pattern[str]:
    """Match a name as a standalone ASCII token, including next to CJK text."""
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deny-term",
        action="append",
        default=[],
        help="Case-insensitive company, customer, project, or person term to reject; repeat as needed",
    )
    return parser.parse_args()


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def strings_in(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings_in(key)
            yield from strings_in(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from strings_in(item)


def inspect_text(path: Path, deny_terms: list[str]) -> list[str]:
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path.relative_to(ROOT)}: not valid UTF-8"]
    patterns = (("absolute user path", PATH_RE), ("email address", EMAIL_RE), ("secret token", SECRET_RE))
    for label, pattern in patterns:
        if pattern.search(text):
            findings.append(f"{path.relative_to(ROOT)}: {label}")
    for term in deny_terms:
        if deny_pattern(term).search(text):
            findings.append(f"{path.relative_to(ROOT)}: denied term {term!r}")
    return findings


def inspect_checkpoint(path: Path, deny_terms: list[str]) -> list[str]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return [f"{path.relative_to(ROOT)}: unexpected checkpoint type"]
    values: list[object] = [checkpoint.get("train_args"), checkpoint.get("git")]
    for model_key in ("model", "ema"):
        model = checkpoint.get(model_key)
        if model is not None:
            values.append(getattr(model, "args", None))
    findings: list[str] = []
    for text in strings_in(values):
        if PATH_RE.search(text):
            findings.append(f"{path.relative_to(ROOT)}: absolute user path in metadata")
        if EMAIL_RE.search(text):
            findings.append(f"{path.relative_to(ROOT)}: email address in metadata")
        for term in deny_terms:
            if deny_pattern(term).search(text):
                findings.append(f"{path.relative_to(ROOT)}: denied term {term!r} in metadata")
    return sorted(set(findings))


def inspect_image(path: Path) -> list[str]:
    from PIL import Image

    with Image.open(path) as image:
        if dict(image.getexif()):
            return [f"{path.relative_to(ROOT)}: EXIF metadata is present"]
    return []


def main() -> None:
    args = parse_args()
    deny_terms = [term.strip() for term in args.deny_term if term.strip()]
    findings: list[str] = []
    files = tracked_files()
    for path in files:
        if not path.is_file():
            continue
        relative_name = str(path.relative_to(ROOT)).casefold()
        for term in deny_terms:
            if deny_pattern(term).search(relative_name):
                findings.append(f"{path.relative_to(ROOT)}: denied term {term!r} in filename")
        if path.suffix.casefold() in TEXT_SUFFIXES:
            findings.extend(inspect_text(path, deny_terms))
        elif path.suffix.casefold() == ".pt":
            findings.extend(inspect_checkpoint(path, deny_terms))
        elif path.suffix.casefold() in IMAGE_SUFFIXES:
            findings.extend(inspect_image(path))

    if findings:
        print("Privacy audit failed:")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        raise SystemExit(1)
    print(f"Privacy audit passed for {len(files)} tracked files.")


if __name__ == "__main__":
    main()
