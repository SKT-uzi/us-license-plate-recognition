"""Local ensemble OCR service specialized for US license-plate numbers."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from PIL import Image, ImageOps

from ocr_service import PlateOCREngine


WORKSPACE = Path(__file__).resolve().parent
FAST_MODEL_DIR = WORKSPACE / "models"
FAST_MODEL_PATH = FAST_MODEL_DIR / "cct_s_v2_global.onnx"
FAST_CONFIG_PATH = FAST_MODEL_DIR / "cct_s_v2_global_plate_config.yaml"
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def normalize_plate_text(text: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def plausible_plate(text: str) -> bool:
    return 4 <= len(text) <= 8 and any(character.isalnum() for character in text)


def enhanced_gray(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    target_height = max(64, min(128, height * 3))
    target_width = max(128, round(width * target_height / max(height, 1)))
    resized = cv2.resize(
        image_rgb, (target_width, target_height), interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class NumberOCREngine:
    """Fuse a plate-specific CCT model with PaddleOCR V5 verification."""

    def __init__(self, device: str = "cpu") -> None:
        for path in (FAST_MODEL_PATH, FAST_CONFIG_PATH):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.device = device
        self.fast_lock = threading.Lock()
        self.fast_model = LicensePlateRecognizer(
            device="cpu",
            onnx_model_path=FAST_MODEL_PATH,
            plate_config_path=FAST_CONFIG_PATH,
        )
        self.paddle_v5 = PlateOCREngine(
            model_name="en_PP-OCRv5_mobile_rec", device=device
        )

    def _fast_recognize(
        self, image_rgb: np.ndarray, variant: str
    ) -> dict[str, Any]:
        started = time.perf_counter()
        with self.fast_lock:
            prediction = self.fast_model.run(
                image_rgb, return_confidence=True
            )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        text = normalize_plate_text(prediction.plate)
        character_scores = np.asarray(prediction.char_probs)[
            : max(len(text), 1)
        ]
        return {
            "source": "fast_plate_cct_s_v2",
            "variant": variant,
            "text": text,
            "score": round(float(character_scores.mean()), 4),
            "min_character_score": round(float(character_scores.min()), 4),
            "region": prediction.region or "",
            "region_score": round(float(prediction.region_prob or 0.0), 4),
            "inference_ms": round(elapsed_ms, 2),
        }

    def _paddle_recognize(
        self, image_rgb: np.ndarray, variant: str
    ) -> dict[str, Any]:
        result = self.paddle_v5.recognize(Image.fromarray(image_rgb))
        return {
            "source": "paddle_v5",
            "variant": variant,
            "text": normalize_plate_text(result.get("text", "")),
            "score": round(float(result.get("score", 0.0)), 4),
            "min_character_score": 0.0,
            "region": "",
            "region_score": 0.0,
            "inference_ms": round(float(result.get("inference_ms", 0.0)), 2),
        }

    @staticmethod
    def _choose_fast_candidate(
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        valid = [
            candidate
            for candidate in candidates
            if candidate["source"] == "fast_plate_cct_s_v2"
            and plausible_plate(candidate["text"])
        ]
        if not valid:
            return None
        raw = next(
            (candidate for candidate in valid if candidate["variant"] == "raw"),
            None,
        )
        gray = next(
            (candidate for candidate in valid if candidate["variant"] == "gray"),
            None,
        )
        if raw and gray and raw["text"] == gray["text"]:
            return max(valid, key=lambda candidate: candidate["score"])
        if raw and raw["min_character_score"] >= 0.65:
            return raw
        return max(valid, key=lambda candidate: candidate["min_character_score"])

    def recognize(self, image: Image.Image) -> dict[str, Any]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 4 or image.height < 4:
            raise ValueError("OCR image is too small.")
        if image.width * image.height > 8_000_000:
            raise ValueError("OCR image has too many pixels.")

        started = time.perf_counter()
        raw_rgb = np.asarray(image)
        candidates = [
            self._fast_recognize(raw_rgb, "raw"),
            self._paddle_recognize(raw_rgb, "raw"),
        ]
        initial_agreement = (
            candidates[0]["text"]
            and candidates[0]["text"] == candidates[1]["text"]
        )
        if not initial_agreement:
            gray_rgb = enhanced_gray(raw_rgb)
            candidates.extend(
                [
                    self._fast_recognize(gray_rgb, "gray"),
                    self._paddle_recognize(gray_rgb, "gray"),
                ]
            )

        chosen = self._choose_fast_candidate(candidates)
        if chosen is None:
            plausible = [
                candidate
                for candidate in candidates
                if plausible_plate(candidate["text"])
            ]
            chosen = max(
                plausible or candidates,
                key=lambda candidate: candidate["score"],
            )

        model_texts: dict[str, set[str]] = {}
        for candidate in candidates:
            model_texts.setdefault(candidate["source"], set()).add(
                candidate["text"]
            )
        independent_support = sum(
            chosen["text"] in texts for texts in model_texts.values()
        )
        review_required = (
            independent_support < 2
            or chosen["min_character_score"] < 0.60
        )
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "text": chosen["text"],
            "raw_text": chosen["text"],
            "score": chosen["score"],
            "min_character_score": chosen["min_character_score"],
            "inference_ms": round(total_ms, 2),
            "model": "fast-plate-ocr cct-s-v2 + PaddleOCR v5",
            "device": self.device,
            "agreement": independent_support >= 2,
            "independent_support": independent_support,
            "review_required": review_required,
            "selection_reason": (
                "The plate-specific OCR and PaddleOCR v5 agree"
                if independent_support >= 2
                else "The plate-specific OCR was preferred; PaddleOCR v5 disagrees"
            ),
            "region": chosen["region"],
            "region_score": chosen["region_score"],
            "candidates": candidates,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": "fast-plate-ocr cct-s-v2 + PaddleOCR v5",
            "device": self.device,
            "fast_model": str(FAST_MODEL_PATH),
            "verification_model": "en_PP-OCRv5_mobile_rec",
        }


class NumberOCRRequestHandler(BaseHTTPRequestHandler):
    engine: NumberOCREngine
    server_version = "NumberOCR/1.0"

    def log_message(self, message_format: str, *args: Any) -> None:
        print(f"[Number OCR HTTP] {self.address_string()} - {message_format % args}")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/health":
            self._send_json(self.engine.health())
            return
        self._send_json({"error": "Page not found."}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/recognize":
            self._send_json({"error": "Endpoint not found."}, status=404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_IMAGE_BYTES * 2:
                raise ValueError("Invalid OCR request size.")
            request = json.loads(self.rfile.read(content_length).decode("utf-8"))
            encoded = request.get("image_base64", "")
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded, validate=True)
            if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
                raise ValueError("OCR image is empty or exceeds 5 MB.")
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                result = self.engine.recognize(image)
            self._send_json(result)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception as error:  # pragma: no cover
            print(f"[Number OCR ERROR] {type(error).__name__}: {error}")
            self._send_json(
                {"error": f"Number OCR inference failed: {type(error).__name__}: {error}"},
                status=500,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7863, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-test", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading the plate-specific OCR and PaddleOCR v5...")
    engine = NumberOCREngine(device=args.device)
    if args.self_test:
        with Image.open(args.self_test) as image:
            print(
                json.dumps(
                    engine.recognize(image), ensure_ascii=False, indent=2
                )
            )
        return
    NumberOCRRequestHandler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), NumberOCRRequestHandler)
    server.daemon_threads = True
    print(f"Plate-number OCR service is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the plate-number OCR service...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
