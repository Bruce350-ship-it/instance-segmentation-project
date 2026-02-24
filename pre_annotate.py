"""
pre_annotate.py
===============
YOLOv8-powered pre-annotation pipeline for CVAT instance segmentation.

Workflow
--------
1. Run YOLOv8 inference on a folder of images and save YOLO-format (.txt) labels.
2. Parse those labels and build a COCO-format JSON file (predictions_coco.json).
3. Import the JSON into CVAT as pre-annotations to dramatically speed up manual
   labelling.

Usage
-----
    # Use all defaults (mirrors the original hard-coded behaviour)
    python pre_annotate.py

    # Override paths / model from the command line
    python pre_annotate.py \
        --model   yolov8m.pt \
        --images  coco128/images/train2017 \
        --output  predictions_coco.json
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COCO class list (80 classes — standard MS-COCO order)
# ---------------------------------------------------------------------------
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


# ---------------------------------------------------------------------------
# Helper: locate the YOLO labels folder after inference
# ---------------------------------------------------------------------------
def find_labels_folder(base_run_dir: str = "runs") -> str:
    """Return the path to YOLO's generated labels folder.

    YOLOv8 can produce slightly different sub-paths depending on the version
    and whether *project* / *name* arguments were supplied.  This function
    checks the most common locations in priority order and falls back to a
    recursive glob search.

    Args:
        base_run_dir: Root directory that YOLO writes run artefacts into.

    Returns:
        Absolute-ish path to the labels folder.

    Raises:
        FileNotFoundError: When no labels folder can be located.
    """
    candidates = [
        os.path.join(base_run_dir, "detect", "predict", "labels"),   # default
        os.path.join(base_run_dir, "detect", "preds",   "labels"),   # custom name
        # Edge-case: doubled path produced by some YOLOv8 releases
        os.path.join(base_run_dir, "detect", "runs", "detect", "preds", "labels"),
    ]

    for path in candidates:
        if os.path.isdir(path):
            log.info("Labels folder located at: %s", path)
            return path

    # Last resort — recursive search
    found = glob.glob(os.path.join(base_run_dir, "**", "labels"), recursive=True)
    if found:
        log.info("Labels folder auto-discovered at: %s", found[0])
        return found[0]

    raise FileNotFoundError(
        "Could not find a labels folder anywhere under '%s'.\n"
        "Ensure model.predict() ran successfully with save_txt=True." % base_run_dir
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def run_inference(model_path: str, image_folder: str) -> None:
    """Run YOLOv8 inference and persist YOLO-format label files to disk.

    Args:
        model_path:    Path to the YOLOv8 weights file (e.g. yolov8m.pt).
        image_folder:  Directory containing the input images.
    """
    log.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    log.info("Running inference on: %s", image_folder)
    model.predict(
        source=image_folder,
        save=False,       # skip writing annotated preview images
        save_txt=True,    # write per-image YOLO .txt label files ← required
        save_json=True,   # also write a raw COCO JSON (optional, kept for reference)
    )
    log.info("Inference complete.")


def build_coco_json(
    image_folder: str,
    labels_folder: str,
    output_json: str,
    classes: list[str],
) -> None:
    """Convert YOLO bounding-box labels to a COCO-format JSON file.

    The produced JSON is compatible with CVAT's COCO importer and can be
    uploaded directly as pre-annotations.

    Args:
        image_folder:  Folder containing the original images.
        labels_folder: Folder containing the YOLO-format .txt label files.
        output_json:   Destination path for the resulting COCO JSON.
        classes:       Ordered list of class names (index 0 → category_id 1).
    """
    coco: dict = {
        "images":      [],
        "annotations": [],
        "categories":  [],
    }

    # COCO spec: category IDs are 1-indexed
    for idx, name in enumerate(classes):
        coco["categories"].append({
            "id":            idx + 1,
            "name":          name,
            "supercategory": "",
        })

    annotation_id = 1  # annotation IDs are also 1-indexed in COCO
    image_paths   = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    if not image_paths:
        log.warning("No .jpg images found in '%s'.", image_folder)

    for image_id, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)

        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append({
            "id":        image_id,
            "file_name": filename,
            "width":     width,
            "height":    height,
        })

        label_path = os.path.join(labels_folder, Path(filename).stem + ".txt")
        if not os.path.exists(label_path):
            log.debug("No label file for image '%s' — skipping.", filename)
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                cls, x_center, y_center, w, h = map(float, parts[:5])
                cls = int(cls)

                bbox_w  = w * width
                bbox_h  = h * height
                x_min   = (x_center * width)  - (bbox_w / 2)
                y_min   = (y_center * height) - (bbox_h / 2)

                coco["annotations"].append({
                    "id":           annotation_id,
                    "image_id":     image_id,
                    "category_id":  cls + 1,          # COCO/CVAT: 1-indexed
                    "bbox":         [x_min, y_min, bbox_w, bbox_h],
                    "area":         bbox_w * bbox_h,
                    "segmentation": [],               # required by CVAT importer
                    "iscrowd":      0,
                })
                annotation_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    log.info("COCO JSON saved  → %s", output_json)
    log.info("  Images      : %d", len(coco["images"]))
    log.info("  Annotations : %d", len(coco["annotations"]))
    log.info("  Categories  : %d", len(coco["categories"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 pre-annotation pipeline → COCO JSON for CVAT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="yolov8m.pt",
        help="Path to the YOLOv8 weights file.",
    )
    parser.add_argument(
        "--images", default="coco128/images/train2017",
        help="Folder containing images to annotate.",
    )
    parser.add_argument(
        "--output", default="predictions_coco.json",
        help="Destination path for the generated COCO JSON.",
    )
    parser.add_argument(
        "--runs-dir", default="runs",
        help="Root directory where YOLO writes run artefacts.",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Skip the YOLO inference step and only (re-)build the COCO JSON "
             "from existing label files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        0 on success, 1 on error.
    """
    args = parse_args(argv)

    try:
        if not args.skip_inference:
            run_inference(model_path=args.model, image_folder=args.images)

        labels_folder = find_labels_folder(base_run_dir=args.runs_dir)

        build_coco_json(
            image_folder=args.images,
            labels_folder=labels_folder,
            output_json=args.output,
            classes=COCO_CLASSES,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())