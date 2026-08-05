"""Start the complete local license-plate recognition demo."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ocr-port", type=int, default=7862)
    parser.add_argument("--number-ocr-port", type=int, default=7863)
    parser.add_argument("--device", default="0", help="Pose/state device, for example 0 or cpu")
    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument(
        "--ocr-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable for OCR services; defaults to the current interpreter",
    )
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    return parser.parse_args()


def wait_until_ready(
    name: str,
    url: str,
    process: subprocess.Popen[bytes],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    print(f"Waiting for {name} at {url} ...")
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    print(f"{name} is ready.")
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0)
    raise TimeoutError(f"{name} did not become ready within {timeout:.0f} seconds")


def stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> None:
    args = parse_args()
    processes: list[subprocess.Popen[bytes]] = []
    try:
        ocr_command = [
            str(args.ocr_python),
            str(ROOT / "src" / "ocr_service.py"),
            "--host",
            args.host,
            "--port",
            str(args.ocr_port),
            "--device",
            args.ocr_device,
        ]
        number_ocr_command = [
            str(args.ocr_python),
            str(ROOT / "src" / "number_ocr_service.py"),
            "--host",
            args.host,
            "--port",
            str(args.number_ocr_port),
            "--device",
            args.ocr_device,
        ]
        processes.append(subprocess.Popen(ocr_command, cwd=ROOT))
        processes.append(subprocess.Popen(number_ocr_command, cwd=ROOT))

        wait_until_ready(
            "PaddleOCR service",
            f"http://{args.host}:{args.ocr_port}/health",
            processes[0],
            args.startup_timeout,
        )
        wait_until_ready(
            "plate-number OCR service",
            f"http://{args.host}:{args.number_ocr_port}/health",
            processes[1],
            args.startup_timeout,
        )

        app_command = [
            sys.executable,
            str(ROOT / "src" / "app.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--device",
            args.device,
            "--ocr-url",
            f"http://{args.host}:{args.ocr_port}",
            "--number-ocr-url",
            f"http://{args.host}:{args.number_ocr_port}",
        ]
        app_process = subprocess.Popen(app_command, cwd=ROOT)
        processes.append(app_process)
        print(f"Demo is available at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop all services.")
        exit_code = app_process.wait()
        if exit_code:
            raise RuntimeError(f"The demo exited with code {exit_code}")
    except KeyboardInterrupt:
        print("\nStopping all services...")
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    main()
