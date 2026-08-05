"""Local PaddleOCR service for rectified US license-plate crops."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "en_PP-OCRv4_mobile_rec"
MODEL_ROOT = Path(
    os.environ.get("PADDLEOCR_MODEL_ROOT", WORKSPACE / "models" / "paddleocr")
).expanduser()
MODEL_DIRECTORIES = {
    "en_PP-OCRv4_mobile_rec": (
        MODEL_ROOT / "en_PP-OCRv4_mobile_rec_infer"
    ),
    "en_PP-OCRv5_mobile_rec": (
        MODEL_ROOT / "en_PP-OCRv5_mobile_rec_infer"
    ),
}
CACHE_DIR = WORKSPACE / ".cache" / "paddleocr"
MAX_IMAGE_BYTES = 5 * 1024 * 1024

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CACHE_DIR))

from paddleocr import TextRecognition  # noqa: E402


def require_model(model_dir: Path) -> None:
    required = ("inference.json", "inference.pdiparams", "inference.yml")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"The OCR model is incomplete: {model_dir}; missing {', '.join(missing)}"
        )


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class PlateOCREngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_dir: Path | None = None,
        device: str = "cpu",
    ) -> None:
        if model_dir is None:
            local_candidate = MODEL_DIRECTORIES.get(model_name)
            if local_candidate is not None and local_candidate.is_dir():
                model_dir = local_candidate
        if model_dir is not None:
            require_model(model_dir)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = device
        self.lock = threading.Lock()
        model_options: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
        }
        if model_dir is not None:
            model_options["model_dir"] = str(model_dir)
            model_options["engine"] = "paddle_static"
        self.model = TextRecognition(**model_options)

    def recognize(self, image: Image.Image) -> dict[str, Any]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 4 or image.height < 4:
            raise ValueError("The OCR image is too small.")
        if image.width * image.height > 8_000_000:
            raise ValueError("The OCR image has too many pixels.")

        started = time.perf_counter()
        with self.lock:
            results = self.model.predict(np.asarray(image), batch_size=1)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if not results:
            return {
                "text": "",
                "raw_text": "",
                "score": 0.0,
                "inference_ms": round(elapsed_ms, 2),
                "model": self.model_name,
                "device": self.device,
            }

        payload = results[0].json
        result = payload.get("res", payload)
        raw_text = str(result.get("rec_text", "")).strip()
        return {
            "text": normalize_plate_text(raw_text),
            "raw_text": raw_text,
            "score": round(float(result.get("rec_score", 0.0)), 4),
            "inference_ms": round(elapsed_ms, 2),
            "model": self.model_name,
            "device": self.device,
        }


class OCRRequestHandler(BaseHTTPRequestHandler):
    engine: PlateOCREngine
    server_version = "PlateOCR/1.0"

    def log_message(self, message_format: str, *args: Any) -> None:
        print(f"[OCR HTTP] {self.address_string()} - {message_format % args}")

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
            self._send_json(
                {
                    "status": "ready",
                    "model": self.engine.model_name,
                    "device": self.engine.device,
                }
            )
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
                raise ValueError("The OCR image is empty or exceeds 5 MB.")

            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                result = self.engine.recognize(image)
            self._send_json(result)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception as error:  # pragma: no cover
            print(f"[OCR ERROR] {type(error).__name__}: {error}")
            self._send_json(
                {"error": f"OCR inference failed: {type(error).__name__}: {error}"},
                status=500,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7862, type=int)
    parser.add_argument("--self-test", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading OCR model: {args.model_name}")
    engine = PlateOCREngine(
        model_name=args.model_name,
        model_dir=args.model_dir,
        device=args.device,
    )

    if args.self_test:
        with Image.open(args.self_test) as image:
            result = engine.recognize(image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    OCRRequestHandler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), OCRRequestHandler)
    server.daemon_threads = True
    print(f"OCR service is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the OCR service...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
