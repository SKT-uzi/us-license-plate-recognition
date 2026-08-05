# Model Card

## Intended use

This project demonstrates US license-plate localization, four-corner rectification, appearance-based state classification, and alphanumeric OCR on still images. It is intended for research, prototyping, and human-reviewed workflows.

## Included weights

| File | Task | Notes |
|---|---|---|
| `models/pose_baseline.pt` | Plate bounding box + four corners | Baseline YOLO pose model used for comparison |
| `models/pose_finetuned.pt` | Plate bounding box + four corners | Fine-tuned checkpoint selected at epoch 47 |
| `models/state_classifier.pt` | 51-class state classification | 50 US states plus Washington DC; trained with road-style augmentation |
| `models/cct_s_v2_global.onnx` | Plate-number OCR | CCT-based OCR model used by `fast-plate-ocr` |

PaddleOCR recognition weights are downloaded automatically on first use unless local model directories are supplied.

## Evaluation snapshot

- Pose validation at the selected epoch: box precision 0.9994, box recall 1.0000, box mAP50 0.9950, and box mAP50–95 0.9199.
- State classifier on 1,971 clean held-out crops: top-1 94.06%, top-5 97.62%.
- State classifier on 1,971 synthetically road-degraded held-out crops: top-1 61.54%, top-5 77.07%.
- OpenALPR road-image sample: plate detected in 15/15 images and exact plate text in 11/15 images.

The evaluation sets differ in difficulty and should not be combined into a single accuracy number. The OpenALPR sample provides plate-number labels, not reliable state ground truth.

## Limitations

- State appearance can be ambiguous across editions, specialty designs, glare, occlusion, and low resolution.
- OCR may confuse visually similar characters such as `0/O`, `1/I`, `5/S`, `6/G`, and `8/B`.
- Performance on a clean crop does not predict performance on distant traffic-camera imagery.
- The confidence values are ranking signals, not calibrated probabilities.
- Results marked “review required” should not be treated as final without a human check.

## Responsible use

License plates may be personal data. Follow applicable privacy, retention, access-control, and surveillance laws. Do not use this demo as the sole basis for enforcement or other high-impact decisions.
