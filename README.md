# US License Plate Recognition

An end-to-end local application for US license-plate recognition from full vehicle images or cropped plates. The pipeline combines YOLO four-corner pose detection, perspective rectification, 51-class state classification, and an OCR ensemble.

## What it does

1. Detects a license plate and its four corners in a vehicle image.
2. Rectifies the plate with a perspective transform.
3. Predicts one of the 50 US states or Washington DC from the plate's overall appearance.
4. Uses state-name OCR as supporting evidence when readable.
5. Reads the plate number with a plate-specific CCT model and PaddleOCR.
6. Flags low-confidence or disagreeing OCR results for human review.

The web interface compares the included baseline pose model with the fine-tuned checkpoint. Cropped plate images can skip pose detection and go directly to state classification and OCR.

## Reproducibility scope

The repository contains everything needed to reproduce the included **inference demo** except platform-specific PyTorch/PaddlePaddle wheels and PaddleOCR recognition weights, which are installed or downloaded using the steps below. All custom inference weights are included and checksummed.

The original training images are intentionally **not** included because license-plate images may contain personal or customer data. The training scripts, label layouts, hyperparameters, anonymized metrics, and selected weights are included. Exact retraining therefore requires an independently obtained, appropriately licensed dataset matching the documented layouts.

## Results

| Evaluation | Result |
|---|---:|
| Pose validation box precision / recall | 99.94% / 100.00% |
| Pose validation box mAP50 / mAP50-95 | 99.50% / 91.99% |
| State classifier, clean test set (1,971 images) | 94.06% top-1 / 97.62% top-5 |
| State classifier, road-degraded test set (1,971 images) | 61.54% top-1 / 77.07% top-5 |
| Public OpenALPR road sample, plate localization | 15/15 images |
| Public OpenALPR road sample, exact plate number | 11/15 images |

Detailed anonymized CSV and JSON outputs are in [`benchmarks`](benchmarks). Source images and plate identifiers are not redistributed in this repository.

## Project structure

```text
.
|-- app.py                         # HTTP API and comparison web demo
|-- run_demo.py                    # Starts all three local services
|-- ocr_service.py                 # PaddleOCR service
|-- number_ocr_service.py          # CCT + PaddleOCR ensemble service
|-- train_pose.py                  # Four-corner pose training
|-- train_state_classifier.py      # 51-class state training
|-- prepare_state_dataset.py       # Road-style augmentation utility
|-- demo_static/index.html         # English browser UI
|-- models/                        # Included inference weights and hashes
|-- benchmarks/                    # Anonymized evaluation summaries
|-- scripts/verify_install.py      # Dependency, checksum, and model checks
|-- scripts/privacy_audit.py       # Pre-publication privacy scan
|-- MODEL_CARD.md                  # Performance and limitations
`-- PRIVACY.md                     # Data-handling and release policy
```

## Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/SKT-uzi/us-license-plate-recognition.git
cd us-license-plate-recognition
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Python 3.10 or 3.11 is recommended.

### 2. Install runtime dependencies

Install a PyTorch build for your operating system and CUDA version using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Install PaddlePaddle or PaddlePaddle GPU using the [official PaddlePaddle instructions](https://www.paddlepaddle.org.cn/install/quick). CPU builds are sufficient for a functional reproduction.

Then install the remaining packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Developers using an Ultralytics source checkout may set `ULTRALYTICS_SOURCE` to that checkout instead of installing the package into the environment.

### 3. Verify the checkout

Run a checksum-only test before installing machine-learning packages if desired:

```bash
python scripts/verify_install.py --hash-only
```

After all dependencies are installed, run the full installation check:

```bash
python scripts/verify_install.py
```

The command verifies all shipped model hashes, imports the required packages, and loads the three Ultralytics models plus the plate-specific ONNX OCR model.

### 4. Start the demo

CUDA device 0:

```bash
python run_demo.py --device 0
```

CPU-only:

```bash
python run_demo.py --device cpu
```

If PaddleOCR is installed in a separate environment, pass that interpreter explicitly:

```bash
python run_demo.py --device 0 --ocr-python /path/to/ocr-environment/python
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The first launch can take several minutes while PaddleOCR downloads and initializes its V4 and V5 recognition weights. Press `Ctrl+C` in the terminal to stop all services.

For a fully offline launch, place compatible PaddleOCR inference directories at:

```text
models/paddleocr/en_PP-OCRv4_mobile_rec_infer
models/paddleocr/en_PP-OCRv5_mobile_rec_infer
```

Alternatively, set `PADDLEOCR_MODEL_ROOT` to a directory containing those two folders.

## Run services independently

```bash
python ocr_service.py --port 7862 --device cpu
python number_ocr_service.py --port 7863 --device cpu
python app.py --port 7860 --device 0
```

## HTTP API

All upload endpoints accept `multipart/form-data` with an `image` field.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model and service status |
| `POST /api/auto` | Detect the input type and return one final result |
| `POST /api/plate` | Treat the input as an already cropped plate |
| `POST /api/compare` | Compare baseline and fine-tuned pose models |

The optional `confidence` form field must be between `0.01` and `0.95`.

## Train the pose model

Edit `plate_pose.yaml` so `path` points to a YOLO pose dataset. Each label must contain four keypoints in this order: top-left, top-right, bottom-right, bottom-left.

```bash
python train_pose.py \
  --data plate_pose.yaml \
  --model models/pose_baseline.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0
```

The included fine-tuned checkpoint was selected at epoch 47. Its training history is in `benchmarks/pose_training_results.csv`.

## Train the state classifier

Prepare an Ultralytics classification dataset with identical class folders under `train`, `val`, and `test`:

```text
state_classifier_dataset/
|-- train/Alabama ... WashingtonDC/
|-- val/Alabama ... WashingtonDC/
`-- test/Alabama ... WashingtonDC/
```

Optionally create a road-degraded training copy while preserving the clean validation and test splits:

```bash
python prepare_state_dataset.py \
  --source state_classifier_dataset \
  --output state_classifier_dataset_road_aug \
  --augment-fraction 0.75
```

Then train:

```bash
python train_state_classifier.py \
  --data state_classifier_dataset_road_aug \
  --model yolov8s-cls.pt \
  --epochs 30 \
  --imgsz 224 \
  --batch 64 \
  --device 0
```

## Privacy and limitations

No original training images, customer images, customer names, company names, local user paths, or raw plate identifiers are intended to be included. Uploaded demo images are processed in memory and are not saved by the normal HTTP endpoints. See [`PRIVACY.md`](PRIVACY.md) and run the privacy audit before any release:

```bash
python scripts/privacy_audit.py --deny-term "YOUR_COMPANY_NAME" --deny-term "CUSTOMER_NAME"
```

State classification degrades on tiny, blurry, occluded, or specialty plates. OCR can confuse similar characters, and confidence values are not calibrated probabilities. Read [`MODEL_CARD.md`](MODEL_CARD.md) before relying on outputs.

## License and acknowledgments

This repository is released under the GNU Affero General Public License v3.0. Third-party components and model formats retain their own licenses and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
