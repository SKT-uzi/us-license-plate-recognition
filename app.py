"""Local web demo for comparing two US license-plate pose models."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


WORKSPACE = Path(__file__).resolve().parent
MODELS_DIR = WORKSPACE / "models"
ORIGINAL_MODEL = MODELS_DIR / "pose_baseline.pt"
NEW_MODEL = MODELS_DIR / "pose_finetuned.pt"
STATE_MODEL = MODELS_DIR / "state_classifier.pt"
INDEX_HTML = WORKSPACE / "demo_static" / "index.html"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
KEYPOINT_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")
DEFAULT_OCR_URL = "http://127.0.0.1:7862"
DEFAULT_NUMBER_OCR_URL = "http://127.0.0.1:7863"
OCR_ACCEPT_THRESHOLD = 0.5
STATE_ACCEPT_THRESHOLD = 0.60
FALLBACK_IMGSZ = 1280
STATE_MIN_CROP_WIDTH = 120
STATE_MIN_CROP_HEIGHT = 40
PLATE_CROP_HORIZONTAL_PADDING = 0.20
PLATE_CROP_VERTICAL_PADDING = 0.08
STATE_DISPLAY_NAMES = {
    "NewHampshire": "New Hampshire",
    "NewJersey": "New Jersey",
    "NewMexico": "New Mexico",
    "NewYork": "New York",
    "NorthCarolina": "North Carolina",
    "NorthDakota": "North Dakota",
    "RhodeIsland": "Rhode Island",
    "SouthCarolina": "South Carolina",
    "SouthDakota": "South Dakota",
    "WashingtonDC": "Washington DC",
    "WestVirginia": "West Virginia",
}
MONTH_STICKERS = {
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
}

os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))

ultralytics_source = os.environ.get("ULTRALYTICS_SOURCE")
if ultralytics_source:
    sys.path.insert(0, str(Path(ultralytics_source).expanduser().resolve()))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def image_to_data_uri(image_rgb: np.ndarray, quality: int = 90) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(
        buffer, format="JPEG", quality=quality, optimize=True
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def save_data_uri(data_uri: str, destination: Path) -> None:
    encoded = data_uri.split(",", 1)[1]
    destination.write_bytes(base64.b64decode(encoded))


class PlateOCRClient:
    def __init__(self, base_url: str, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def recognize(self, image_rgb: np.ndarray) -> dict[str, Any]:
        image_uri = image_to_data_uri(image_rgb, quality=96)
        payload = json.dumps(
            {"image_base64": image_uri.split(",", 1)[1]}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/recognize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(detail).get("error", detail)
            except json.JSONDecodeError:
                message = detail
            raise RuntimeError(f"The OCR service returned an error: {message}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError("The OCR service is unavailable or did not respond.") from error

        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health", timeout=2.0
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
            return {
                "status": result.get("status", "unknown"),
                "model": result.get("model", ""),
                "device": result.get("device", ""),
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return {"status": "unavailable"}


class PlateStateClassifier:
    """Classify a rectified plate into one of 50 states or Washington DC."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        imgsz: int = 224,
        threshold: float = STATE_ACCEPT_THRESHOLD,
        min_crop_width: int = STATE_MIN_CROP_WIDTH,
        min_crop_height: int = STATE_MIN_CROP_HEIGHT,
    ) -> None:
        require_file(model_path, "State classifier")
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.threshold = threshold
        self.min_crop_width = min_crop_width
        self.min_crop_height = min_crop_height
        self.model = YOLO(str(model_path))
        if self.model.task != "classify":
            raise ValueError(f"The state classifier has the wrong task type: {self.model.task}")

    @staticmethod
    def display_name(class_name: str) -> str:
        return STATE_DISPLAY_NAMES.get(class_name, class_name)

    def recognize(
        self,
        image_rgb: np.ndarray,
        quality_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        input_height, input_width = image_rgb.shape[:2]
        quality_width, quality_height = quality_size or (
            input_width,
            input_height,
        )
        quality_accepted = not (
            quality_width < self.min_crop_width
            or quality_height < self.min_crop_height
        )
        quality_reason = ""
        if not quality_accepted:
            quality_reason = (
                f"The original plate crop is low resolution ({quality_width}x{quality_height}; "
                f"at least {self.min_crop_width}x{self.min_crop_height} is recommended). "
                "Appearance-based state classification still ran using the plate's colors, "
                "graphics, and layout; treat the result as a guess."
            )

        started = time.perf_counter()
        prediction = self.model.predict(
            source=Image.fromarray(image_rgb.astype(np.uint8), mode="RGB"),
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        total_ms = (time.perf_counter() - started) * 1000
        if prediction.probs is None:
            raise RuntimeError("The state classifier did not return class probabilities.")

        probabilities = prediction.probs.data.detach().cpu()
        top_indices = [int(index) for index in prediction.probs.top5]
        top5 = [
            {
                "name": str(prediction.names[index]),
                "display_name": self.display_name(str(prediction.names[index])),
                "score": round(float(probabilities[index]), 4),
            }
            for index in top_indices
        ]
        top1 = top5[0]
        inference_ms = float(prediction.speed.get("inference", total_ms))
        classifier_accepted = bool(top1["score"] >= self.threshold)
        return {
            **top1,
            "accepted": bool(classifier_accepted and quality_accepted),
            "threshold": self.threshold,
            "inference_ms": round(inference_ms, 2),
            "top5": top5,
            "model": self.model_path.name,
            "input_width": input_width,
            "input_height": input_height,
            "quality_width": quality_width,
            "quality_height": quality_height,
            "quality_accepted": quality_accepted,
            "quality_reason": quality_reason,
            "low_resolution_guess": not quality_accepted,
            "classification_basis": "plate_appearance",
            "source": "state_classifier",
            "classifier_name": top1["name"],
            "classifier_display_name": top1["display_name"],
            "classifier_score": top1["score"],
            "classifier_accepted": classifier_accepted,
            "ocr_verified": False,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": str(self.model_path),
            "classes": len(self.model.names),
            "imgsz": self.imgsz,
            "threshold": self.threshold,
            "min_crop_width": self.min_crop_width,
            "min_crop_height": self.min_crop_height,
            "low_resolution_policy": "classify_as_unreliable_guess",
            "device": self.device,
        }


def perspective_crop(image_rgb: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    if np.any(points <= 0):
        return None

    top_left, top_right, bottom_right, bottom_left = points.astype(np.float32)
    width = int(
        round(
            max(
                np.linalg.norm(top_right - top_left),
                np.linalg.norm(bottom_right - bottom_left),
            )
        )
    )
    height = int(
        round(
            max(
                np.linalg.norm(bottom_left - top_left),
                np.linalg.norm(bottom_right - top_right),
            )
        )
    )
    if width < 4 or height < 4 or width > 4096 or height > 4096:
        return None

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
    return cv2.warpPerspective(
        image_rgb,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def expand_plate_quad(
    points: np.ndarray,
    image_shape: tuple[int, ...],
    horizontal_padding: float = PLATE_CROP_HORIZONTAL_PADDING,
    vertical_padding: float = PLATE_CROP_VERTICAL_PADDING,
) -> np.ndarray:
    """Expand a plate quadrilateral so slightly inset keypoints do not cut characters."""
    if (
        points.shape != (4, 2)
        or not np.isfinite(points).all()
        or np.any(points <= 0)
    ):
        return points

    top_left, top_right, bottom_right, bottom_left = points.astype(np.float32)
    horizontal_vector = (top_right - top_left) + (bottom_right - bottom_left)
    vertical_vector = (bottom_left - top_left) + (bottom_right - top_right)
    horizontal_norm = float(np.linalg.norm(horizontal_vector))
    vertical_norm = float(np.linalg.norm(vertical_vector))
    if horizontal_norm < 1e-6 or vertical_norm < 1e-6:
        return points

    horizontal_unit = horizontal_vector / horizontal_norm
    vertical_unit = vertical_vector / vertical_norm
    plate_width = max(
        float(np.linalg.norm(top_right - top_left)),
        float(np.linalg.norm(bottom_right - bottom_left)),
    )
    plate_height = max(
        float(np.linalg.norm(bottom_left - top_left)),
        float(np.linalg.norm(bottom_right - top_right)),
    )
    horizontal_delta = horizontal_unit * plate_width * horizontal_padding
    vertical_delta = vertical_unit * plate_height * vertical_padding
    expanded = np.array(
        [
            top_left - horizontal_delta - vertical_delta,
            top_right + horizontal_delta - vertical_delta,
            bottom_right + horizontal_delta + vertical_delta,
            bottom_left - horizontal_delta + vertical_delta,
        ],
        dtype=np.float32,
    )
    image_height, image_width = image_shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 1e-3, max(image_width - 1, 1e-3))
    expanded[:, 1] = np.clip(expanded[:, 1], 1e-3, max(image_height - 1, 1e-3))
    return expanded


def draw_prediction(
    image_rgb: np.ndarray,
    box: np.ndarray,
    confidence: float,
    points: np.ndarray | None,
    color: tuple[int, int, int],
    plate_index: int,
) -> None:
    height, width = image_rgb.shape[:2]
    thickness = max(2, round(min(height, width) / 320))
    font_scale = max(0.5, min(height, width) / 900)
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = np.clip([x1, x2], 0, max(width - 1, 0))
    y1, y2 = np.clip([y1, y2], 0, max(height - 1, 0))

    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, thickness)
    label = f"plate {plate_index + 1}  {confidence:.3f}"
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_top = max(0, y1 - text_height - 12)
    cv2.rectangle(
        image_rgb,
        (x1, label_top),
        (min(width - 1, x1 + text_width + 12), y1),
        color,
        -1,
    )
    cv2.putText(
        image_rgb,
        label,
        (x1 + 6, max(text_height + 1, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (8, 16, 28),
        thickness,
        cv2.LINE_AA,
    )

    if points is None or points.shape != (4, 2):
        return

    visible_points = np.rint(points).astype(int)
    if np.isfinite(points).all() and np.all(points > 0):
        cv2.polylines(
            image_rgb,
            [visible_points.reshape((-1, 1, 2))],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    for point_index, (point_x, point_y) in enumerate(visible_points):
        if point_x <= 0 or point_y <= 0:
            continue
        point_x = int(np.clip(point_x, 0, max(width - 1, 0)))
        point_y = int(np.clip(point_y, 0, max(height - 1, 0)))
        cv2.circle(
            image_rgb,
            (point_x, point_y),
            max(4, thickness * 2),
            (255, 255, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            image_rgb,
            (point_x, point_y),
            max(2, thickness),
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image_rgb,
            str(point_index + 1),
            (point_x + 6, max(12, point_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


class ComparisonEngine:
    def __init__(
        self,
        device: str = "0",
        imgsz: int = 640,
        fallback_imgsz: int = FALLBACK_IMGSZ,
        ocr_url: str = DEFAULT_OCR_URL,
        number_ocr_url: str = DEFAULT_NUMBER_OCR_URL,
        original_model: Path = ORIGINAL_MODEL,
        finetuned_model: Path = NEW_MODEL,
        state_model: Path = STATE_MODEL,
        state_imgsz: int = 224,
        state_threshold: float = STATE_ACCEPT_THRESHOLD,
    ) -> None:
        require_file(original_model, "Baseline pose model")
        require_file(finetuned_model, "Fine-tuned pose model")
        require_file(state_model, "State classifier")
        require_file(INDEX_HTML, "Demo page")
        if not torch.cuda.is_available() and device != "cpu":
            raise RuntimeError("CUDA is unavailable. Check the environment or use --device cpu.")

        self.device = device
        self.imgsz = imgsz
        self.fallback_imgsz = max(imgsz, fallback_imgsz)
        self.ocr = PlateOCRClient(ocr_url)
        self.number_ocr = PlateOCRClient(number_ocr_url)
        self.state_classifier = PlateStateClassifier(
            state_model,
            device=device,
            imgsz=state_imgsz,
            threshold=state_threshold,
        )
        self.lock = threading.Lock()
        self.models = (
            {
                "id": "original",
                "name": "Baseline Pose Model (~1,600 epochs)",
                "short_name": "Baseline Model",
                "path": original_model,
                "color": (47, 214, 178),
                "model": YOLO(str(original_model)),
            },
            {
                "id": "finetuned",
                "name": "Fine-tuned Pose Model (best, epoch 47)",
                "short_name": "Fine-tuned Model",
                "path": finetuned_model,
                "color": (255, 181, 71),
                "model": YOLO(str(finetuned_model)),
            },
        )

    @property
    def device_name(self) -> str:
        if self.device == "cpu":
            return "CPU"
        return torch.cuda.get_device_name(int(self.device))

    def _select_ocr_variant(
        self,
        variants: list[tuple[str, np.ndarray, dict[str, Any]]],
    ) -> tuple[str, np.ndarray, dict[str, Any]] | None:
        """Choose between the original and edge-expanded plate crops."""
        candidates: list[dict[str, Any]] = []
        for variant_name, variant_crop, result in variants:
            normalized_text = self._normalize_plate_text(result.get("text", ""))
            score = float(result.get("score", 0.0))
            mixed = any(character.isalpha() for character in normalized_text) and any(
                character.isdigit() for character in normalized_text
            )
            candidates.append(
                {
                    "name": variant_name,
                    "crop": variant_crop,
                    "result": {**result, "text": normalized_text},
                    "text": normalized_text,
                    "score": score,
                    "plausible": self._is_plausible_plate_number(normalized_text),
                    "trusted": (
                        int(result.get("independent_support", 0)) >= 2
                        and not result.get("review_required", False)
                    ),
                    "rank": (
                        score
                        + (0.05 if mixed else 0.0)
                        + (
                            0.20
                            if "fast-plate-ocr" in str(result.get("model", ""))
                            else 0.0
                        )
                        + min(
                            0.10,
                            0.05 * int(result.get("independent_support", 0)),
                        )
                        - (0.08 if result.get("review_required", False) else 0.0)
                    ),
                }
            )

        if not candidates:
            return None

        plausible = [candidate for candidate in candidates if candidate["plausible"]]
        trusted = [candidate for candidate in plausible if candidate["trusted"]]
        nonempty = [candidate for candidate in candidates if candidate["text"]]
        selection_pool = trusted or plausible or nonempty or candidates
        if len(selection_pool) >= 2:
            by_length = sorted(
                selection_pool,
                key=lambda candidate: len(candidate["text"]),
                reverse=True,
            )
            longer, shorter = by_length[0], by_length[-1]
            if (
                len(longer["text"]) > len(shorter["text"])
                and shorter["text"]
                and shorter["text"] in longer["text"]
                and longer["score"] >= OCR_ACCEPT_THRESHOLD
                and longer["score"] >= shorter["score"] - 0.20
            ):
                chosen = longer
            else:
                chosen = max(selection_pool, key=lambda candidate: candidate["rank"])
        else:
            chosen = selection_pool[0]

        return chosen["name"], chosen["crop"], chosen["result"]

    def _finalize_ocr_evidence(
        self,
        result: dict[str, Any],
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Count support by independent OCR models, not preprocessing repeats."""
        selected_text = self._normalize_plate_text(result.get("text", ""))
        support_by_source: dict[str, set[str]] = {}
        for attempt in attempts:
            source = str(attempt.get("source", "unknown"))
            if source != "number_ensemble" or not attempt.get("candidates"):
                support_by_source.setdefault(source, set()).add(
                    self._normalize_plate_text(attempt.get("text", ""))
                )
            for candidate in attempt.get("candidates", []):
                candidate_source = str(candidate.get("source", "unknown"))
                support_by_source.setdefault(candidate_source, set()).add(
                    self._normalize_plate_text(candidate.get("text", ""))
                )

        supporting_sources = sorted(
            source
            for source, texts in support_by_source.items()
            if selected_text and selected_text in texts
        )
        independent_support = len(supporting_sources)
        plausible = self._is_plausible_plate_number(selected_text)
        review_required = (
            independent_support < 2
            or float(result.get("min_character_score", 1.0)) < 0.60
            or not plausible
        )
        if not plausible:
            selection_reason = "The text length or format is unusual for a US plate; review it manually"
        elif independent_support >= 2:
            selection_reason = f"{independent_support} independent OCR models support this number"
        else:
            selection_reason = "The OCR models disagree; review the number manually"
        return {
            **result,
            "text": selected_text,
            "independent_support": independent_support,
            "supporting_sources": supporting_sources,
            "review_required": review_required,
            "selection_reason": selection_reason,
        }

    @staticmethod
    def _resize_state_ocr_band(
        image_rgb: np.ndarray, target_height: int = 96
    ) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        scale = target_height / max(height, 1)
        return cv2.resize(
            image_rgb,
            (max(4, round(width * scale)), target_height),
            interpolation=cv2.INTER_CUBIC,
        )

    def _match_state_name(self, raw_text: Any) -> dict[str, Any] | None:
        letters = "".join(
            character
            for character in str(raw_text).upper()
            if "A" <= character <= "Z"
        )
        if len(letters) < 4 or len(letters) > 18:
            return None

        names = self.state_classifier.model.names
        class_names = (
            [str(names[index]) for index in sorted(names)]
            if isinstance(names, dict)
            else [str(name) for name in names]
        )
        ranked: list[tuple[float, str]] = []
        for class_name in class_names:
            target = "".join(
                character
                for character in class_name.upper()
                if "A" <= character <= "Z"
            )
            ranked.append(
                (SequenceMatcher(None, letters, target).ratio(), class_name)
            )
        ranked.sort(reverse=True)
        best_score, best_name = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < 0.55 or best_score - second_score < 0.05:
            return None
        return {
            "ocr_letters": letters,
            "name": best_name,
            "display_name": self.state_classifier.display_name(best_name),
            "similarity": round(best_score, 4),
            "margin": round(best_score - second_score, 4),
        }

    def _recognize_state_name(
        self,
        raw_crop: np.ndarray,
        expanded_crop: np.ndarray | None,
    ) -> dict[str, Any]:
        raw_height = raw_crop.shape[0]
        bands: list[tuple[str, np.ndarray]] = [
            (
                "raw_top30",
                raw_crop[: max(4, round(raw_height * 0.30))],
            ),
            (
                "raw_top40",
                raw_crop[: max(4, round(raw_height * 0.40))],
            ),
        ]
        if expanded_crop is not None:
            expanded_height = expanded_crop.shape[0]
            bands.append(
                (
                    "expanded_top35",
                    expanded_crop[
                        : max(4, round(expanded_height * 0.35))
                    ],
                )
            )

        attempts: list[dict[str, Any]] = []
        votes: dict[str, list[dict[str, Any]]] = {}
        total_ms = 0.0
        for band_name, band in bands:
            enlarged = self._resize_state_ocr_band(band)
            try:
                ocr_result = self.ocr.recognize(enlarged)
            except RuntimeError as error:
                attempts.append(
                    {
                        "band": band_name,
                        "error": str(error),
                        "matched": False,
                    }
                )
                continue
            total_ms += float(ocr_result.get("inference_ms", 0.0))
            match = self._match_state_name(ocr_result.get("raw_text", ""))
            attempt = {
                "band": band_name,
                "raw_text": str(ocr_result.get("raw_text", "")),
                "text": str(ocr_result.get("text", "")),
                "ocr_score": round(float(ocr_result.get("score", 0.0)), 4),
                "matched": match is not None,
            }
            if match is not None and attempt["ocr_score"] >= 0.35:
                attempt.update(match)
                votes.setdefault(str(match["name"]), []).append(attempt)
            attempts.append(attempt)

        if not votes:
            return {
                "accepted": False,
                "name": "",
                "display_name": "",
                "votes": 0,
                "confidence": 0.0,
                "inference_ms": round(total_ms, 2),
                "attempts": attempts,
                "reason": "State-name OCR did not produce a usable candidate.",
            }

        winner_name, winner_votes = max(
            votes.items(),
            key=lambda item: (
                len(item[1]),
                sum(
                    float(vote["similarity"]) * float(vote["ocr_score"])
                    for vote in item[1]
                ),
            ),
        )
        vote_count = len(winner_votes)
        average_similarity = sum(
            float(vote["similarity"]) for vote in winner_votes
        ) / vote_count
        average_ocr_score = sum(
            float(vote["ocr_score"]) for vote in winner_votes
        ) / vote_count
        strongest_similarity = max(
            float(vote["similarity"]) for vote in winner_votes
        )
        accepted = bool(
            (vote_count >= 3 and average_similarity >= 0.55)
            or (vote_count >= 2 and average_similarity >= 0.67)
            or (
                strongest_similarity >= 0.85
                and average_ocr_score >= 0.60
            )
        )
        confidence = 0.70 * average_similarity + 0.30 * average_ocr_score
        return {
            "accepted": accepted,
            "name": winner_name,
            "display_name": self.state_classifier.display_name(winner_name),
            "votes": vote_count,
            "confidence": round(confidence, 4),
            "average_similarity": round(average_similarity, 4),
            "average_ocr_score": round(average_ocr_score, 4),
            "inference_ms": round(total_ms, 2),
            "attempts": attempts,
            "reason": (
                f"{vote_count} state-name OCR variants agree on "
                f"{self.state_classifier.display_name(winner_name)}."
            ),
        }

    def _fuse_state_evidence(
        self,
        classifier_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        fused = dict(classifier_result)
        fused["ocr_verification"] = verification
        fused["classifier_name"] = classifier_result.get("name", "")
        fused["classifier_display_name"] = classifier_result.get(
            "display_name", ""
        )
        fused["classifier_score"] = classifier_result.get("score", 0.0)
        fused["classifier_accepted"] = classifier_result.get(
            "accepted", False
        )
        fused["source"] = "state_classifier"
        fused["ocr_verified"] = False
        if not verification.get("accepted"):
            return fused

        verified_name = str(verification.get("name", ""))
        classifier_name = str(classifier_result.get("name", ""))
        if verified_name == classifier_name:
            fused["ocr_verified"] = True
            fused["source"] = "state_classifier+state_name_ocr"
            return fused

        fused.update(
            {
                "name": verified_name,
                "display_name": verification.get("display_name", verified_name),
                "score": verification.get("confidence", 0.0),
                "accepted": True,
                "source": "state_name_ocr_override",
                "ocr_verified": True,
                "classification_conflict": True,
                "fusion_reason": (
                    f"The classifier originally predicted "
                    f"{classifier_result.get('display_name', '')} "
                    f"({float(classifier_result.get('score', 0.0)):.4f}); "
                    f"the state-name OCR majority corrected it to "
                    f"{verification.get('display_name', verified_name)}."
                ),
            }
        )
        return fused

    def _run_one(
        self,
        spec: dict[str, Any],
        image: Image.Image,
        confidence_threshold: float,
        inference_imgsz: int | None = None,
        fallback_used: bool = False,
    ) -> dict[str, Any]:
        run_imgsz = inference_imgsz or self.imgsz
        image_rgb = np.asarray(image.convert("RGB")).copy()
        started = time.perf_counter()
        prediction = spec["model"].predict(
            source=image.copy(),
            imgsz=run_imgsz,
            conf=confidence_threshold,
            iou=0.7,
            device=self.device,
            verbose=False,
        )[0]
        total_ms = (time.perf_counter() - started) * 1000
        inference_ms = float(prediction.speed.get("inference", total_ms))

        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return {
                "id": spec["id"],
                "name": spec["name"],
                "annotated": image_to_data_uri(image_rgb),
                "crops": [],
                "detections": 0,
                "valid_crops": 0,
                "recognized_crops": 0,
                "ocr_ms": 0.0,
                "state_ms": 0.0,
                "inference_ms": round(inference_ms, 2),
                "total_ms": round(total_ms, 2),
                "rows": [],
                "imgsz": run_imgsz,
                "fallback_used": fallback_used,
                "initial_inference_ms": 0.0,
                "fallback_inference_ms": (
                    round(inference_ms, 2) if fallback_used else 0.0
                ),
            }

        box_values = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        keypoint_values: np.ndarray | None = None
        if prediction.keypoints is not None:
            keypoint_values = prediction.keypoints.data.detach().cpu().numpy()

        crops: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        total_ocr_ms = 0.0
        total_state_ms = 0.0
        for index, (box, score) in enumerate(zip(box_values, confidences)):
            points: np.ndarray | None = None
            raw_points: np.ndarray | None = None
            point_scores: list[float] = []
            if keypoint_values is not None and index < len(keypoint_values):
                raw_points = keypoint_values[index, :, :2].astype(np.float32)
                points = expand_plate_quad(raw_points, image_rgb.shape)
                if keypoint_values.shape[2] >= 3:
                    point_scores = [
                        round(float(value), 4) for value in keypoint_values[index, :, 2]
                    ]

            display_box = box.astype(np.float32).copy()
            if points is not None and points.shape == (4, 2):
                display_box[0] = min(display_box[0], float(points[:, 0].min()))
                display_box[1] = min(display_box[1], float(points[:, 1].min()))
                display_box[2] = max(display_box[2], float(points[:, 0].max()))
                display_box[3] = max(display_box[3], float(points[:, 1].max()))

            draw_prediction(
                image_rgb,
                display_box,
                float(score),
                points,
                spec["color"],
                index,
            )

            source_image_rgb = np.asarray(image.convert("RGB"))
            raw_crop = (
                perspective_crop(source_image_rgb, raw_points)
                if raw_points is not None
                else None
            )
            expanded_crop = (
                perspective_crop(source_image_rgb, points)
                if points is not None
                else None
            )
            crop = expanded_crop if expanded_crop is not None else raw_crop
            ocr_result: dict[str, Any] | None = None
            ocr_error_messages: list[str] = []
            ocr_variant = ""
            ocr_attempts: list[dict[str, Any]] = []
            state_result: dict[str, Any] | None = None
            state_error: str | None = None
            if crop is not None:
                try:
                    quality_size = (
                        (raw_crop.shape[1], raw_crop.shape[0])
                        if raw_crop is not None
                        else None
                    )
                    state_result = self.state_classifier.recognize(
                        expanded_crop if expanded_crop is not None else crop,
                        quality_size=quality_size,
                    )
                    total_state_ms += float(state_result.get("inference_ms", 0.0))
                except RuntimeError as error:
                    state_error = str(error)

                ocr_variants: list[tuple[str, np.ndarray, dict[str, Any]]] = []
                variant_crops: list[tuple[str, np.ndarray]] = []
                if raw_crop is not None:
                    variant_crops.append(("raw", raw_crop))
                if expanded_crop is not None and (
                    raw_crop is None
                    or not np.array_equal(expanded_crop, raw_crop)
                ):
                    variant_crops.append(("expanded", expanded_crop))
                for variant_name, variant_crop in variant_crops:
                    try:
                        variant_result = self.ocr.recognize(variant_crop)
                        total_ocr_ms += float(
                            variant_result.get("inference_ms", 0.0)
                        )
                        ocr_variants.append(
                            (
                                f"{variant_name}_paddle_v4",
                                variant_crop,
                                variant_result,
                            )
                        )
                        ocr_attempts.append(
                            {
                                "variant": variant_name,
                                "source": "paddle_v4",
                                "text": self._normalize_plate_text(
                                    variant_result.get("text", "")
                                ),
                                "score": round(
                                    float(variant_result.get("score", 0.0)), 4
                                ),
                            }
                        )
                    except RuntimeError as error:
                        ocr_error_messages.append(
                            f"{variant_name}/paddle_v4: {error}"
                        )
                    try:
                        number_result = self.number_ocr.recognize(variant_crop)
                        total_ocr_ms += float(
                            number_result.get("inference_ms", 0.0)
                        )
                        ocr_variants.append(
                            (variant_name, variant_crop, number_result)
                        )
                        ocr_attempts.append(
                            {
                                "variant": variant_name,
                                "source": "number_ensemble",
                                "text": self._normalize_plate_text(
                                    number_result.get("text", "")
                                ),
                                "score": round(
                                    float(number_result.get("score", 0.0)), 4
                                ),
                                "review_required": bool(
                                    number_result.get("review_required")
                                ),
                                "candidates": number_result.get(
                                    "candidates", []
                                ),
                            }
                        )
                    except RuntimeError as error:
                        ocr_error_messages.append(
                            f"{variant_name}/number_ensemble: {error}"
                        )

                selected_ocr = self._select_ocr_variant(ocr_variants)
                if selected_ocr is not None:
                    ocr_variant, crop, ocr_result = selected_ocr
                    ocr_result = self._finalize_ocr_evidence(
                        ocr_result, ocr_attempts
                    )
                if (
                    state_result is not None
                    and state_result.get("quality_accepted")
                    and raw_crop is not None
                ):
                    state_verification = self._recognize_state_name(
                        raw_crop, expanded_crop
                    )
                    total_ocr_ms += float(
                        state_verification.get("inference_ms", 0.0)
                    )
                    state_result = self._fuse_state_evidence(
                        state_result, state_verification
                    )
                ocr_error = "; ".join(ocr_error_messages) or None
                crops.append(
                    {
                        "label": (
                            f"Plate {index + 1} · confidence {float(score):.3f} · "
                            + (
                                "OCR used the expanded crop"
                                if ocr_variant == "expanded"
                                else "OCR used the original crop"
                            )
                        ),
                        "image": image_to_data_uri(crop, quality=94),
                        "ocr_text": ocr_result.get("text", "") if ocr_result else "",
                        "ocr_raw_text": (
                            ocr_result.get("raw_text", "") if ocr_result else ""
                        ),
                        "ocr_score": (
                            float(ocr_result.get("score", 0.0))
                            if ocr_result
                            else 0.0
                        ),
                        "ocr_ms": (
                            float(ocr_result.get("inference_ms", 0.0))
                            if ocr_result
                            else 0.0
                        ),
                        "ocr_error": ocr_error,
                        "ocr_variant": ocr_variant,
                        "ocr_attempts": ocr_attempts,
                        "ocr_model": (
                            ocr_result.get("model", "") if ocr_result else ""
                        ),
                        "ocr_review_required": (
                            bool(ocr_result.get("review_required"))
                            if ocr_result
                            else False
                        ),
                        "ocr_independent_support": (
                            int(ocr_result.get("independent_support", 0))
                            if ocr_result
                            else 0
                        ),
                        "ocr_selection_reason": (
                            ocr_result.get("selection_reason", "")
                            if ocr_result
                            else ""
                        ),
                        "ocr_candidates": (
                            ocr_result.get("candidates", [])
                            if ocr_result
                            else []
                        ),
                        "state": state_result,
                        "state_error": state_error,
                    }
                )
            else:
                ocr_error = None

            rows.append(
                {
                    "model": spec["short_name"],
                    "plate": index + 1,
                    "confidence": round(float(score), 4),
                    "ocr_text": ocr_result.get("text", "") if ocr_result else "",
                    "ocr_score": (
                        round(float(ocr_result.get("score", 0.0)), 4)
                        if ocr_result
                        else 0.0
                    ),
                    "ocr_error": ocr_error,
                    "ocr_variant": ocr_variant,
                    "ocr_attempts": ocr_attempts,
                    "ocr_model": (
                        ocr_result.get("model", "") if ocr_result else ""
                    ),
                    "ocr_review_required": (
                        bool(ocr_result.get("review_required"))
                        if ocr_result
                        else False
                    ),
                    "ocr_independent_support": (
                        int(ocr_result.get("independent_support", 0))
                        if ocr_result
                        else 0
                    ),
                    "ocr_selection_reason": (
                        ocr_result.get("selection_reason", "")
                        if ocr_result
                        else ""
                    ),
                    "state_name": (
                        state_result.get("display_name", "")
                        if state_result
                        else ""
                    ),
                    "state_score": (
                        round(float(state_result.get("score", 0.0)), 4)
                        if state_result
                        else 0.0
                    ),
                    "state_accepted": (
                        bool(state_result.get("accepted")) if state_result else False
                    ),
                    "state_quality_reason": (
                        state_result.get("quality_reason", "")
                        if state_result
                        else ""
                    ),
                    "state_source": (
                        state_result.get("source", "")
                        if state_result
                        else ""
                    ),
                    "state_classifier_name": (
                        state_result.get("classifier_display_name", "")
                        if state_result
                        else ""
                    ),
                    "state_ocr_verified": (
                        bool(state_result.get("ocr_verified"))
                        if state_result
                        else False
                    ),
                    "state_error": state_error,
                    "box": [round(float(value), 1) for value in display_box],
                    "raw_box": [round(float(value), 1) for value in box],
                    "keypoints": (
                        [
                            {
                                "name": KEYPOINT_NAMES[point_index],
                                "x": round(float(point[0]), 1),
                                "y": round(float(point[1]), 1),
                                "score": (
                                    point_scores[point_index]
                                    if point_index < len(point_scores)
                                    else None
                                ),
                            }
                            for point_index, point in enumerate(points)
                        ]
                        if points is not None
                        else []
                    ),
                    "raw_keypoints": (
                        [
                            {
                                "name": KEYPOINT_NAMES[point_index],
                                "x": round(float(point[0]), 1),
                                "y": round(float(point[1]), 1),
                                "score": (
                                    point_scores[point_index]
                                    if point_index < len(point_scores)
                                    else None
                                ),
                            }
                            for point_index, point in enumerate(raw_points)
                        ]
                        if raw_points is not None
                        else []
                    ),
                }
            )

        return {
            "id": spec["id"],
            "name": spec["name"],
            "annotated": image_to_data_uri(image_rgb),
            "crops": crops,
            "detections": len(box_values),
            "valid_crops": len(crops),
            "recognized_crops": sum(bool(crop["ocr_text"]) for crop in crops),
            "ocr_ms": round(total_ocr_ms, 2),
            "state_ms": round(total_state_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "total_ms": round(total_ms, 2),
            "rows": rows,
            "imgsz": run_imgsz,
            "fallback_used": fallback_used,
            "initial_inference_ms": 0.0,
            "fallback_inference_ms": (
                round(inference_ms, 2) if fallback_used else 0.0
            ),
        }

    def compare(self, image: Image.Image, confidence_threshold: float) -> dict[str, Any]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 16 or image.height < 16:
            raise ValueError("The image is too small.")
        if image.width * image.height > 40_000_000:
            raise ValueError("The image is too large; use an image under 40 megapixels.")

        with self.lock:
            outputs = [
                self._run_one(spec, image, confidence_threshold) for spec in self.models
            ]
            finetuned_index = next(
                index
                for index, output in enumerate(outputs)
                if output["id"] == "finetuned"
            )
            finetuned_output = outputs[finetuned_index]
            if (
                finetuned_output["detections"] == 0
                and self.fallback_imgsz > self.imgsz
            ):
                finetuned_spec = next(
                    spec for spec in self.models if spec["id"] == "finetuned"
                )
                fallback_output = self._run_one(
                    finetuned_spec,
                    image,
                    confidence_threshold,
                    inference_imgsz=self.fallback_imgsz,
                    fallback_used=True,
                )
                initial_inference_ms = float(
                    finetuned_output.get("inference_ms", 0.0)
                )
                fallback_inference_ms = float(
                    fallback_output.get("inference_ms", 0.0)
                )
                fallback_output["initial_inference_ms"] = round(
                    initial_inference_ms, 2
                )
                fallback_output["fallback_inference_ms"] = round(
                    fallback_inference_ms, 2
                )
                fallback_output["inference_ms"] = round(
                    initial_inference_ms + fallback_inference_ms, 2
                )
                fallback_output["total_ms"] = round(
                    float(finetuned_output.get("total_ms", 0.0))
                    + float(fallback_output.get("total_ms", 0.0)),
                    2,
                )
                outputs[finetuned_index] = fallback_output

        return {
            "image": {"width": image.width, "height": image.height},
            "imgsz": self.imgsz,
            "fallback_imgsz": self.fallback_imgsz,
            "adaptive_fallback_used": any(
                output.get("fallback_used", False) for output in outputs
            ),
            "confidence_threshold": confidence_threshold,
            "device": self.device_name,
            "ocr": "fast-plate-ocr CCT-S-v2 + PaddleOCR v5/v4",
            "state_classifier": self.state_classifier.health(),
            "models": outputs,
            "rows": [row for output in outputs for row in output["rows"]],
        }

    def recognize_plate(
        self, image: Image.Image, classify_state: bool = True
    ) -> dict[str, Any]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 16 or image.height < 16:
            raise ValueError("The image is too small.")
        if image.width * image.height > 40_000_000:
            raise ValueError("The image is too large; use an image under 40 megapixels.")

        image_rgb = np.asarray(image).copy()
        state_result: dict[str, Any] | None = None
        state_error: str | None = None
        with self.lock:
            if classify_state:
                try:
                    state_result = self.state_classifier.recognize(image_rgb)
                    if state_result.get("quality_accepted"):
                        state_verification = self._recognize_state_name(
                            image_rgb, None
                        )
                        state_result = self._fuse_state_evidence(
                            state_result, state_verification
                        )
                except RuntimeError as error:
                    state_error = str(error)
            direct_variants: list[
                tuple[str, np.ndarray, dict[str, Any]]
            ] = []
            direct_attempts: list[dict[str, Any]] = []
            direct_errors: list[RuntimeError] = []
            total_direct_ocr_ms = 0.0
            try:
                number_result = self.number_ocr.recognize(image_rgb)
                total_direct_ocr_ms += float(
                    number_result.get("inference_ms", 0.0)
                )
                direct_variants.append(("number_ensemble", image_rgb, number_result))
                direct_attempts.append(
                    {
                        "variant": "direct",
                        "source": "number_ensemble",
                        "text": number_result.get("text", ""),
                        "score": number_result.get("score", 0.0),
                        "candidates": number_result.get("candidates", []),
                    }
                )
            except RuntimeError as error:
                direct_errors.append(error)
            try:
                paddle_v4_result = self.ocr.recognize(image_rgb)
                total_direct_ocr_ms += float(
                    paddle_v4_result.get("inference_ms", 0.0)
                )
                direct_variants.append(
                    ("paddle_v4", image_rgb, paddle_v4_result)
                )
                direct_attempts.append(
                    {
                        "variant": "direct",
                        "source": "paddle_v4",
                        "text": paddle_v4_result.get("text", ""),
                        "score": paddle_v4_result.get("score", 0.0),
                    }
                )
            except RuntimeError as error:
                direct_errors.append(error)
            selected_direct = self._select_ocr_variant(direct_variants)
            if selected_direct is None:
                raise direct_errors[0]
            _, _, ocr_result = selected_direct
            ocr_result = self._finalize_ocr_evidence(
                ocr_result, direct_attempts
            )
            ocr_result["inference_ms"] = round(total_direct_ocr_ms, 2)

        score = float(ocr_result.get("score", 0.0))
        text = self._normalize_plate_text(ocr_result.get("text", ""))
        return {
            "mode": "direct_ocr",
            "image": {
                "width": image.width,
                "height": image.height,
                "data_uri": image_to_data_uri(image_rgb, quality=94),
            },
            "ocr": {
                "text": text,
                "raw_text": str(ocr_result.get("raw_text", "")),
                "score": round(score, 4),
                "inference_ms": round(
                    float(ocr_result.get("inference_ms", 0.0)), 2
                ),
                "accepted": bool(text and score >= OCR_ACCEPT_THRESHOLD),
                "threshold": OCR_ACCEPT_THRESHOLD,
                "model": str(
                    ocr_result.get("model", "en_PP-OCRv4_mobile_rec")
                ),
                "device": str(ocr_result.get("device", "cpu")).upper(),
                "review_required": bool(
                    ocr_result.get("review_required", False)
                ),
                "independent_support": int(
                    ocr_result.get("independent_support", 0)
                ),
                "selection_reason": str(
                    ocr_result.get("selection_reason", "")
                ),
                "candidates": ocr_result.get("candidates", []),
                "state": state_result,
                "state_error": state_error,
            },
            "state": state_result,
        }

    @staticmethod
    def _normalize_plate_text(value: Any) -> str:
        return "".join(
            character
            for character in str(value).upper()
            if "A" <= character <= "Z" or "0" <= character <= "9"
        )

    @staticmethod
    def _is_plausible_plate_number(text: str) -> bool:
        return 4 <= len(text) <= 8 and text not in MONTH_STICKERS

    def recognize_auto(
        self, image: Image.Image, confidence_threshold: float
    ) -> dict[str, Any]:
        """Return one final plate number while preserving comparison diagnostics."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 16 or image.height < 16:
            raise ValueError("The image is too small.")
        if image.width * image.height > 40_000_000:
            raise ValueError("The image is too large; use an image under 40 megapixels.")

        direct = self.recognize_plate(image.copy(), classify_state=False)
        direct_ocr = direct["ocr"]
        direct_text = self._normalize_plate_text(direct_ocr.get("text", ""))
        aspect_ratio = image.width / image.height
        looks_like_plate_crop = (
            1.85 <= aspect_ratio <= 4.5
            and direct_ocr.get("score", 0.0) >= OCR_ACCEPT_THRESHOLD
            and self._is_plausible_plate_number(direct_text)
        )

        if looks_like_plate_crop:
            state_result: dict[str, Any] | None = None
            state_error: str | None = None
            with self.lock:
                try:
                    state_result = self.state_classifier.recognize(
                        np.asarray(image).copy()
                    )
                    if state_result.get("quality_accepted"):
                        state_verification = self._recognize_state_name(
                            np.asarray(image).copy(), None
                        )
                        state_result = self._fuse_state_evidence(
                            state_result, state_verification
                        )
                except RuntimeError as error:
                    state_error = str(error)
            direct["state"] = state_result
            direct["ocr"]["state"] = state_result
            direct["ocr"]["state_error"] = state_error
            return {
                "mode": "auto",
                "image": direct["image"],
                "input_type": "cropped_plate",
                "final": {
                    "text": direct_text,
                    "score": direct_ocr["score"],
                    "inference_ms": direct_ocr["inference_ms"],
                    "accepted": True,
                    "threshold": OCR_ACCEPT_THRESHOLD,
                    "source": "direct_ocr",
                    "source_label": "Direct OCR on the full plate image",
                    "reason": "The input appears to be a cropped plate, so pose detection was skipped.",
                    "review_required": bool(
                        direct_ocr.get("review_required", False)
                    ),
                    "independent_support": int(
                        direct_ocr.get("independent_support", 0)
                    ),
                    "selection_reason": str(
                        direct_ocr.get("selection_reason", "")
                    ),
                    "candidates": direct_ocr.get("candidates", []),
                    "state": state_result,
                },
                "direct": direct,
                "comparison": None,
            }

        comparison = self.compare(image.copy(), confidence_threshold)
        candidates: list[dict[str, Any]] = []
        state_candidates: list[dict[str, Any]] = []
        source_priority = {"finetuned": 0.12, "original": 0.04}
        for model in comparison["models"]:
            for crop in model["crops"]:
                state_result = crop.get("state")
                if state_result:
                    state_candidate = dict(state_result)
                    state_candidate["_rank"] = float(
                        state_result.get("score", 0.0)
                    ) + source_priority.get(model["id"], 0.0)
                    state_candidates.append(state_candidate)
                text = self._normalize_plate_text(crop.get("ocr_text", ""))
                score = float(crop.get("ocr_score", 0.0))
                if (
                    crop.get("ocr_error")
                    or score < OCR_ACCEPT_THRESHOLD
                    or not self._is_plausible_plate_number(text)
                ):
                    continue
                mixed = any(character.isalpha() for character in text) and any(
                    character.isdigit() for character in text
                )
                rank = score + source_priority.get(model["id"], 0.0)
                if mixed:
                    rank += 0.05
                if crop.get("ocr_review_required"):
                    rank -= 0.08
                candidates.append(
                    {
                        "text": text,
                        "score": round(score, 4),
                        "inference_ms": round(float(crop.get("ocr_ms", 0.0)), 2),
                        "accepted": True,
                        "threshold": OCR_ACCEPT_THRESHOLD,
                        "source": f"{model['id']}_pose_crop",
                        "source_label": f"OCR after localization and rectification by {model['name']}",
                        "reason": "The plate was localized in the vehicle image, rectified, and read by OCR.",
                        "review_required": bool(
                            crop.get("ocr_review_required", False)
                        ),
                        "independent_support": int(
                            crop.get("ocr_independent_support", 0)
                        ),
                        "selection_reason": str(
                            crop.get("ocr_selection_reason", "")
                        ),
                        "candidates": crop.get("ocr_candidates", []),
                        "state": state_result,
                        "_rank": rank,
                    }
                )

        if candidates:
            finetuned_candidates = [
                candidate
                for candidate in candidates
                if candidate["source"] == "finetuned_pose_crop"
            ]
            selection_pool = finetuned_candidates or candidates
            final = max(
                selection_pool, key=lambda candidate: candidate["_rank"]
            ).copy()
            final.pop("_rank", None)
        else:
            final = {
                "text": "",
                "score": 0.0,
                "inference_ms": 0.0,
                "accepted": False,
                "threshold": OCR_ACCEPT_THRESHOLD,
                "source": "none",
                "source_label": "No reliable result",
                "reason": "No plausible plate number reached the 0.50 confidence threshold.",
                "state": None,
            }

        if state_candidates:
            best_state = max(
                state_candidates, key=lambda candidate: candidate["_rank"]
            ).copy()
            best_state.pop("_rank", None)
            final["state"] = best_state
        elif not final.get("state"):
            final["state"] = None

        return {
            "mode": "auto",
            "image": direct["image"],
            "input_type": "vehicle",
            "final": final,
            "direct": direct,
            "comparison": comparison,
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    engine: ComparisonEngine
    index_html: bytes
    server_version = "PlatePoseDemo/1.0"

    def log_message(self, message_format: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {message_format % args}")

    def _send_headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_headers(200, "text/html; charset=utf-8", len(self.index_html))
            self.wfile.write(self.index_html)
            return
        if path == "/health":
            self._send_json(
                {
                    "status": "ready",
                    "device": self.engine.device_name,
                    "imgsz": self.engine.imgsz,
                    "fallback_imgsz": self.engine.fallback_imgsz,
                    "ocr": self.engine.ocr.health(),
                    "number_ocr": self.engine.number_ocr.health(),
                    "state_classifier": self.engine.state_classifier.health(),
                }
            )
            return
        self._send_json({"error": "Page not found."}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        request_path = self.path.split("?", 1)[0]
        if request_path not in {"/api/auto", "/api/compare", "/api/ocr"}:
            self._send_json({"error": "Endpoint not found."}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("No upload was received.")
            if content_length > MAX_UPLOAD_BYTES + 1024 * 1024:
                raise ValueError("The uploaded file exceeds the 20 MB limit.")

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                raise ValueError("Invalid upload format.")

            raw_body = self.rfile.read(content_length)
            mime_message = BytesParser(policy=email_policy).parsebytes(
                (
                    f"Content-Type: {content_type}\r\n"
                    "MIME-Version: 1.0\r\n\r\n"
                ).encode("utf-8")
                + raw_body
            )

            image_bytes: bytes | None = None
            confidence_text = "0.25"
            for part in mime_message.iter_parts():
                field_name = part.get_param("name", header="content-disposition")
                if field_name == "image":
                    image_bytes = part.get_payload(decode=True)
                elif field_name == "conf":
                    confidence_text = part.get_content().strip()

            if not image_bytes:
                raise ValueError("Select an image.")
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                raise ValueError("The uploaded file exceeds the 20 MB limit.")

            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                if request_path == "/api/ocr":
                    result = self.engine.recognize_plate(image)
                else:
                    confidence = float(confidence_text)
                    if not 0.01 <= confidence <= 0.95:
                        raise ValueError("The confidence threshold must be between 0.01 and 0.95.")
                    if request_path == "/api/auto":
                        result = self.engine.recognize_auto(image, confidence)
                    else:
                        result = self.engine.compare(image, confidence)
            self._send_json(result)
        except (ValueError, OSError) as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception as error:  # pragma: no cover - last-resort server guard
            print(f"[ERROR] {type(error).__name__}: {error}")
            self._send_json(
                {"error": f"Inference failed: {type(error).__name__}: {error}"}, status=500
            )


def save_self_test(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        key: value for key, value in result.items() if key not in {"models"}
    }
    summary["models"] = []
    for model in result["models"]:
        model_summary = {
            key: value
            for key, value in model.items()
            if key not in {"annotated", "crops"}
        }
        save_data_uri(model["annotated"], output_dir / f"{model['id']}_annotated.jpg")
        crop_files = []
        for index, crop in enumerate(model["crops"], start=1):
            filename = f"{model['id']}_crop_{index}.jpg"
            save_data_uri(crop["image"], output_dir / filename)
            crop_files.append(filename)
        model_summary["crop_files"] = crop_files
        summary["models"].append(model_summary)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--fallback-imgsz", default=FALLBACK_IMGSZ, type=int)
    parser.add_argument("--ocr-url", default=DEFAULT_OCR_URL)
    parser.add_argument("--number-ocr-url", default=DEFAULT_NUMBER_OCR_URL)
    parser.add_argument("--original-model", type=Path, default=ORIGINAL_MODEL)
    parser.add_argument("--finetuned-model", type=Path, default=NEW_MODEL)
    parser.add_argument("--state-model", type=Path, default=STATE_MODEL)
    parser.add_argument("--state-imgsz", default=224, type=int)
    parser.add_argument(
        "--state-threshold", default=STATE_ACCEPT_THRESHOLD, type=float
    )
    parser.add_argument("--self-test", type=Path, help="Compare models on one image and exit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "demo_test_output",
        help="Self-test output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = ComparisonEngine(
        device=args.device,
        imgsz=args.imgsz,
        fallback_imgsz=args.fallback_imgsz,
        ocr_url=args.ocr_url,
        number_ocr_url=args.number_ocr_url,
        original_model=args.original_model,
        finetuned_model=args.finetuned_model,
        state_model=args.state_model,
        state_imgsz=args.state_imgsz,
        state_threshold=args.state_threshold,
    )

    print(f"Device: {engine.device_name}")
    print(f"Input size: {engine.imgsz}")
    print(f"Fallback input size: {engine.fallback_imgsz}")
    print(f"Baseline pose model: {args.original_model}")
    print(f"Fine-tuned pose model: {args.finetuned_model}")
    print(f"State classifier: {args.state_model}")
    print(f"OCR service: {args.ocr_url}")
    print(f"Plate-number OCR service: {args.number_ocr_url}")

    if args.self_test:
        require_file(args.self_test, "Self-test image")
        with Image.open(args.self_test) as image:
            result = engine.compare(image, confidence_threshold=0.25)
        save_self_test(result, args.output_dir)
        print(f"Self-test completed: {args.output_dir.resolve()}")
        return

    DemoRequestHandler.engine = engine
    DemoRequestHandler.index_html = INDEX_HTML.read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    server.daemon_threads = True
    print(f"Demo is running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the service.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the demo...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
