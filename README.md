# Instance Segmentation Pre-Annotation Pipeline

> **Accelerate CVAT labelling with zero-shot YOLOv8 pre-annotations.**

---

## Overview

Manually annotating hundreds of images for object detection or instance segmentation is slow and expensive. This pipeline automates the most tedious part: **generating a first-pass set of bounding-box annotations** using a pretrained YOLOv8 model, then exporting them in the exact format CVAT needs so you can open any task and only *correct* boxes rather than drawing them from scratch.

```
Images on disk  →  YOLOv8 inference  →  COCO JSON  →  CVAT pre-annotations
```

### Key features

| Feature | Detail |
|---|---|
| **Zero-shot detection** | Uses `yolov8m.pt` (pretrained on MS-COCO, 80 classes) — no custom training required |
| **CVAT-compatible output** | Produces a valid COCO JSON (`predictions_coco.json`) that imports cleanly via *CVAT → Import annotations → COCO 1.0* |
| **Robust label discovery** | Auto-detects the YOLO run folder regardless of version-specific path nesting quirks |
| **CLI-configurable** | Paths and model weights are all overridable from the command line — no source edits needed |
| **Skip-inference mode** | Re-build the COCO JSON from existing label files without re-running inference |

---

## Project Structure

```
Instance Segmentation Project/
│
├── pre_annotate.py              # Main pipeline script
│
├── requirements.txt             # Python dependencies
│
├── yolov8m.pt                   # YOLOv8 medium weights (not committed to Git)
│
├── labels.json                  # CVAT label configuration (80 COCO classes)
│
├── coco128/                     # Sample dataset (COCO128 — 128 images)
│   ├── images/train2017/        # Input images (.jpg)
│   └── labels/train2017/        # Ground-truth YOLO labels (reference only)
│
├── train2017_yolo_cvat_export/  # YOLO-format dataset export from CVAT
│   ├── obj_train_data/          # Per-image annotation .txt files
│   ├── obj.data
│   ├── obj.names
│   └── train.txt
│
├── predictions_coco.json        # ← Generated output: upload this to CVAT
│
├── cvat_annotations_export.xml  # CVAT XML export (for backup / re-import)
│
└── runs/                        # YOLO inference artefacts (auto-generated)
    └── detect/predict/
        └── labels/              # Per-image YOLO prediction .txt files
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ultralytics` will pull in PyTorch automatically. If you need a specific CUDA version, install PyTorch manually first — see the [PyTorch install guide](https://pytorch.org/get-started/locally/).

### 2 — Download model weights (if not already present)

`yolov8m.pt` is downloaded automatically the first time you run inference if the file is not found locally. Alternatively, download it manually:

```bash
# Automatic (handled by ultralytics)
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### 3 — Run the pipeline

```bash
# Default — run inference on coco128/images/train2017, write predictions_coco.json
python pre_annotate.py

# Custom paths
python pre_annotate.py \
    --model   yolov8m.pt \
    --images  path/to/your/images \
    --output  my_predictions.json

# Skip inference (re-generate JSON from existing label files only)
python pre_annotate.py --skip-inference
```

### 4 — Import into CVAT

1. Open your CVAT task and go to **Actions → Upload annotations**.
2. Choose format **COCO 1.0**.
3. Upload `predictions_coco.json`.
4. CVAT will populate every image with the model's detections — review and correct them.

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | `yolov8m.pt` | Path to the YOLOv8 weights file |
| `--images` | `coco128/images/train2017` | Folder of input images |
| `--output` | `predictions_coco.json` | Destination path for the COCO JSON |
| `--runs-dir` | `runs` | Root folder for YOLO run artefacts |
| `--skip-inference` | _flag_ | Skip inference; rebuild JSON from existing labels |

---

## How It Works

```
┌───────────────────────────────────────────────────────────┐
│  1. YOLO Inference                                         │
│     model.predict(source=images, save_txt=True)           │
│     → writes one .txt file per image under runs/          │
└──────────────────────────┬────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────┐
│  2. Label Discovery                                        │
│     Checks known paths in priority order;                 │
│     falls back to a recursive glob under runs/            │
└──────────────────────────┬────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────┐
│  3. COCO JSON Assembly                                     │
│     For each image:                                        │
│       • record id / filename / width / height             │
│       • parse each YOLO label line                        │
│       • convert normalised (cx, cy, w, h) → COCO          │
│         absolute (x_min, y_min, w, h) bounding box        │
│       • emit annotation with 1-indexed category_id        │
│     Write out predictions_coco.json                       │
└───────────────────────────────────────────────────────────┘
```

### Coordinate conversion

YOLO stores boxes as normalised centre + size:

```
x_min = (x_center × img_width)  − (w × img_width  / 2)
y_min = (y_center × img_height) − (h × img_height / 2)
```

The resulting `[x_min, y_min, bbox_width, bbox_height]` is the COCO bounding-box convention.

---

## Supported Classes

All **80 MS-COCO** categories are supported out of the box (person, bicycle, car, … toothbrush). The full list lives in `COCO_CLASSES` inside `pre_annotate.py` and directly maps to the `labels.json` configuration file used in CVAT.

---

## Requirements

| Package | Minimum version | Purpose |
|---|---|---|
| `ultralytics` | 8.0.0 | YOLOv8 model loading & inference |
| `Pillow` | 10.0.0 | Reading image dimensions |
| `torch` | _(via ultralytics)_ | Deep-learning backend |

Python **3.10+** is recommended (the script uses `list[str] | None` union syntax).

---

## Limitations & Roadmap

- **Bounding boxes only** — the `segmentation` field in the output is intentionally left empty (`[]`). CVAT accepts this for detection tasks; for true instance masks you would need a `yolov8m-seg.pt` model and a polygon-extraction step.
- **Single-class-per-box** — multi-label or hierarchical categories are not handled.
- **JPEG images only** — the glob currently targets `*.jpg`; PNG support can be added trivially.

---

## License

This project is released under the **MIT License**. The COCO128 dataset included for demonstration purposes is subject to the [COCO Dataset Terms of Use](https://cocodataset.org/#termsofuse).
