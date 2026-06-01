#!/usr/bin/env python3
"""
infer_visdrone.py — DA-YOLO Inference on VisDrone Aerial Images
================================================================
Runs inference on VisDrone images (1920×1080) using sliding-window tiling
to preserve resolution for small object detection, then merges detections
with cross-patch NMS.

Outputs (saved to --output):
    infer_output/
    ├── visualisations/        ← annotated images (full resolution)
    ├── detections_<img>.json  ← per-image boxes, classes, scores
    ├── summary.json           ← run-level stats + class counts
    └── summary.csv            ← flat CSV of all detections

Usage (from project root):
    # Run on val split (first 10 images):
    python infer_visdrone.py

    # Run on all val images:
    python infer_visdrone.py --source VisDrone_dataset/yolo/images/val --max-images -1

    # Run on test-dev:
    python infer_visdrone.py --source VisDrone_dataset/yolo/images/test-dev

    # Custom weights / thresholds:
    python infer_visdrone.py \\
        --weights runs/da_yolo/visdrone_scratch2/weights/best.pth \\
        --source  VisDrone_dataset/yolo/images/val \\
        --patch   1280 --overlap 0.25 \\
        --conf    0.25 --iou 0.45 \\
        --device  0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo import DetectionModel
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# VisDrone 10-class names (index matches converter output)
VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Class colour palette (BGR) — 10 distinct colours
_CLASS_COLORS_BGR = [
    (220,  20,  60),   # pedestrian   — crimson
    (255, 127,  14),   # people       — orange
    ( 44, 160,  44),   # bicycle      — green
    ( 31, 119, 180),   # car          — blue
    (148, 103, 189),   # van          — purple
    (140,  86,  75),   # truck        — brown
    (227, 119, 194),   # tricycle     — pink
    (127, 127, 127),   # awning-tri   — gray
    (188, 189,  34),   # bus          — olive
    ( 23, 190, 207),   # motor        — cyan
]


# ============================================================================
# Model Loading
# ============================================================================

def load_model(weights: str | Path, device: torch.device) -> DetectionModel:
    """Load DA-YOLO DetectionModel from .pth checkpoint."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    print(f"  Loading       : {weights}")
    ckpt  = torch.load(weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().eval().to(device)

    print(f"  Epoch         : {ckpt.get('epoch', '?')}")
    print(f"  Classes ({model.nc}): {list(model.names.values())}")
    return model


# ============================================================================
# Tiling
# ============================================================================

def compute_tiles(
    img_h: int,
    img_w: int,
    patch_size: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) tile coordinates covering the full image."""
    stride = max(1, int(patch_size * (1.0 - overlap)))
    tiles: list[tuple[int, int, int, int]] = []

    y0 = 0
    while True:
        y1 = min(y0 + patch_size, img_h)
        x0 = 0
        while True:
            x1 = min(x0 + patch_size, img_w)
            tiles.append((x0, y0, x1, y1))
            if x1 == img_w:
                break
            x0 += stride
        if y1 == img_h:
            break
        y0 += stride
    return tiles


# ============================================================================
# Single-patch inference
# ============================================================================

def infer_patch(
    model:      DetectionModel,
    patch_bgr:  np.ndarray,
    patch_size: int,
    conf_thres: float,
    device:     torch.device,
) -> np.ndarray:
    """
    Infer one BGR patch.  Returns (N, 6) float32: [x0,y0,x1,y1, conf, cls_id]
    in *patch* pixel coordinates.
    """
    h, w   = patch_bgr.shape[:2]
    scale  = patch_size / max(h, w)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(patch_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    # BGR → RGB, HWC → CHW, normalise
    img    = canvas[:, :, ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(img, np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)

    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.6, max_det=1000)
    det = det[0]

    if det is None or len(det) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    det = det.cpu().numpy().astype(np.float32)
    det[:, :4] /= scale
    det[:, 0]   = np.clip(det[:, 0], 0, w)
    det[:, 1]   = np.clip(det[:, 1], 0, h)
    det[:, 2]   = np.clip(det[:, 2], 0, w)
    det[:, 3]   = np.clip(det[:, 3], 0, h)
    return det


# ============================================================================
# Cross-patch NMS
# ============================================================================

def cross_patch_nms(dets: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy class-aware NMS over all-patch combined detections."""
    if len(dets) == 0:
        return dets

    order = np.argsort(-dets[:, 4])
    keep  = []

    while len(order):
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break
        rest = order[1:]

        ix0, iy0, ix1, iy1 = dets[idx, :4]
        inter_x0 = np.maximum(ix0, dets[rest, 0])
        inter_y0 = np.maximum(iy0, dets[rest, 1])
        inter_x1 = np.minimum(ix1, dets[rest, 2])
        inter_y1 = np.minimum(iy1, dets[rest, 3])
        inter_w  = np.maximum(0.0, inter_x1 - inter_x0)
        inter_h  = np.maximum(0.0, inter_y1 - inter_y0)
        inter    = inter_w * inter_h

        area_i   = (ix1 - ix0) * (iy1 - iy0)
        area_r   = (dets[rest, 2] - dets[rest, 0]) * (dets[rest, 3] - dets[rest, 1])
        union    = area_i + area_r - inter
        iou      = np.where(union > 0, inter / union, 0.0)

        same_cls = dets[rest, 5] == dets[idx, 5]
        order    = rest[~(same_cls & (iou > iou_thres))]

    return dets[keep]


# ============================================================================
# Direct (non-tiled) inference for images <= patch_size
# ============================================================================

def infer_full_image(
    model:      DetectionModel,
    img_bgr:    np.ndarray,
    patch_size: int,
    conf_thres: float,
    device:     torch.device,
) -> np.ndarray:
    """Infer directly without tiling for smaller images."""
    return infer_patch(model, img_bgr, patch_size, conf_thres, device)


# ============================================================================
# Visualisation
# ============================================================================

def draw_detections(img_bgr: np.ndarray, dets: np.ndarray, names: dict) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    lw   = max(2, int(max(h, w) / 900))
    fs   = max(0.45, max(h, w) / 3500)

    for det in dets:
        x0, y0, x1, y1, conf, cls_id = det
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cls_id  = int(cls_id)
        color   = _CLASS_COLORS_BGR[cls_id % len(_CLASS_COLORS_BGR)]
        # BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        label   = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(vis, (x0, y0), (x1, y1), color_bgr, lw)

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw - 1))
        ly0 = max(y0, th + bl + 2)
        cv2.rectangle(vis, (x0, ly0 - th - bl - 2), (x0 + tw + 4, ly0), color_bgr, -1)
        cv2.putText(vis, label, (x0 + 2, ly0 - bl),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, lw - 1), cv2.LINE_AA)
    return vis


# ============================================================================
# Per-image orchestrator
# ============================================================================

def process_image(
    image_path: Path,
    model:      DetectionModel,
    names:      dict,
    patch_size: int,
    overlap:    float,
    conf_thres: float,
    nms_iou:    float,
    device:     torch.device,
    vis_dir:    Path,
    idx:        int,
    total:      int,
) -> tuple[list[dict], float]:
    """
    Full pipeline for one image.
    Returns (detection_records, inference_time_seconds).
    """
    print(f"\n[{idx}/{total}] {image_path.name}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  WARNING: could not read image — skipping")
        return [], 0.0

    img_h, img_w = img.shape[:2]
    t_start      = time.perf_counter()

    if img_h <= patch_size and img_w <= patch_size:
        # Image fits in one patch — no tiling needed
        dets = infer_full_image(model, img, patch_size, conf_thres, device)
        print(f"  Size {img_w}×{img_h} → single-pass inference")
    else:
        tiles = compute_tiles(img_h, img_w, patch_size, overlap)
        print(f"  Size {img_w}×{img_h} → {len(tiles)} tiles ({patch_size}px, {overlap:.0%} overlap)")

        all_dets: list[np.ndarray] = []
        for t_idx, (x0, y0, x1, y1) in enumerate(tiles, 1):
            print(f"    tile {t_idx}/{len(tiles)} ...", end="\r", flush=True)
            patch = img[y0:y1, x0:x1]
            d     = infer_patch(model, patch, patch_size, conf_thres, device)
            if len(d):
                d[:, 0] += x0;  d[:, 2] += x0
                d[:, 1] += y0;  d[:, 3] += y0
                all_dets.append(d)
        print()

        dets = np.concatenate(all_dets, 0) if all_dets else np.zeros((0, 6), np.float32)
        pre_nms = len(dets)
        dets = cross_patch_nms(dets, iou_thres=nms_iou)
        print(f"  Detections: {pre_nms} → {len(dets)} after NMS")

    elapsed = time.perf_counter() - t_start

    # Save annotated image
    vis = draw_detections(img, dets, names)
    cv2.imwrite(str(vis_dir / image_path.name), vis)

    # Build detection records
    records: list[dict] = []
    for det in dets:
        x0_, y0_, x1_, y1_, conf, cls_id = det
        cls_id = int(cls_id)
        records.append({
            "image":       image_path.name,
            "class_id":    cls_id,
            "class_name":  names.get(cls_id, str(cls_id)),
            "confidence":  round(float(conf), 4),
            "bbox_x0":     int(x0_),
            "bbox_y0":     int(y0_),
            "bbox_x1":     int(x1_),
            "bbox_y1":     int(y1_),
            "bbox_w":      int(x1_ - x0_),
            "bbox_h":      int(y1_ - y0_),
        })

    print(f"  Detections: {len(records)}   time: {elapsed:.2f}s")
    return records, elapsed


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DA-YOLO tiled inference on VisDrone aerial images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",    default="runs/da_yolo/visdrone_scratch2/weights/best.pth")
    p.add_argument("--source",     default="VisDrone_dataset/yolo/images/val",
                   help="Image folder (or single image path)")
    p.add_argument("--output",     default="infer_output/visdrone",
                   help="Output directory")
    p.add_argument("--patch",      type=int,   default=1280,
                   help="Tile size in pixels (match training imgsz)")
    p.add_argument("--overlap",    type=float, default=0.25,
                   help="Fractional tile overlap [0, 0.9]")
    p.add_argument("--conf",       type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--iou",        type=float, default=0.45,
                   help="Cross-patch NMS IoU threshold")
    p.add_argument("--device",     default="",
                   help="Torch device: '' (auto), 'cpu', '0', ...")
    p.add_argument("--max-images", type=int,   default=10,
                   help="Max images to process (-1 = all)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = select_device(args.device)

    out_root = REPO_ROOT / args.output
    vis_dir  = out_root / "visualisations"
    out_root.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    source = REPO_ROOT / args.source
    if source.is_file():
        image_paths = [source]
    else:
        image_paths = sorted(p for p in source.iterdir()
                             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        print(f"No images found in: {source}")
        sys.exit(1)

    print("=" * 70)
    print("  DA-YOLO — VisDrone Inference Pipeline")
    print("=" * 70)
    print(f"  Source       : {source}  ({len(image_paths)} images)")
    print(f"  Weights      : {REPO_ROOT / args.weights}")
    print(f"  Output       : {out_root}")
    print(f"  Tile size    : {args.patch}px   Overlap: {args.overlap:.0%}")
    print(f"  Conf / NMS   : {args.conf} / {args.iou}")
    print(f"  Device       : {device}")
    print("=" * 70)

    model = load_model(REPO_ROOT / args.weights, device)
    names = model.names

    all_records:  list[dict]             = []
    class_counts: dict[str, int]         = defaultdict(int)
    time_per_img: list[float]            = []

    for idx, image_path in enumerate(image_paths, 1):
        records, elapsed = process_image(
            image_path=image_path,
            model=model,
            names=names,
            patch_size=args.patch,
            overlap=args.overlap,
            conf_thres=args.conf,
            nms_iou=args.iou,
            device=device,
            vis_dir=vis_dir,
            idx=idx,
            total=len(image_paths),
        )
        all_records.extend(records)
        time_per_img.append(elapsed)

        for r in records:
            class_counts[r["class_name"]] += 1

        # Per-image JSON
        img_json = out_root / f"detections_{image_path.stem}.json"
        with open(img_json, "w") as f:
            json.dump({"image": image_path.name, "total": len(records), "detections": records}, f, indent=2)

    total_elapsed = sum(time_per_img)
    avg_ms        = (total_elapsed / max(len(image_paths), 1)) * 1000

    # ── Summary JSON ──────────────────────────────────────────────────────
    img_ctr: dict[str, Counter] = defaultdict(Counter)
    for r in all_records:
        img_ctr[r["image"]][r["class_name"]] += 1

    summary = {
        "run_info": {
            "weights":      str(REPO_ROOT / args.weights),
            "source":       str(source),
            "num_images":   len(image_paths),
            "total_dets":   len(all_records),
            "patch_size":   args.patch,
            "overlap":      args.overlap,
            "conf_thres":   args.conf,
            "nms_iou":      args.iou,
            "device":       str(device),
            "total_time_s": round(total_elapsed, 2),
            "avg_ms_per_img": round(avg_ms, 1),
            "fps":          round(1000 / max(avg_ms, 1e-6), 1),
        },
        "class_counts": dict(sorted(class_counts.items(), key=lambda x: -x[1])),
        "per_image_counts": {
            img: {"total": sum(ctr.values()), "by_class": dict(ctr)}
            for img, ctr in img_ctr.items()
        },
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Summary CSV ───────────────────────────────────────────────────────
    fieldnames = ["image", "class_id", "class_name", "confidence",
                  "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "bbox_w", "bbox_h"]
    with open(out_root / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # ── Final report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Inference Complete")
    print("=" * 70)
    print(f"  Images processed  : {len(image_paths)}")
    print(f"  Total detections  : {len(all_records)}")
    print(f"  Avg time / image  : {avg_ms:.1f} ms  ({summary['run_info']['fps']:.1f} FPS)")
    print(f"  Total elapsed     : {total_elapsed:.1f}s")
    print(f"\n  Output: {out_root}")
    print(f"  ├── visualisations/         ({len(image_paths)} annotated images)")
    print(f"  ├── detections_<img>.json   (per-image detection records)")
    print(f"  ├── summary.json            (run info + class counts)")
    print(f"  └── summary.csv             ({len(all_records)} detection rows)")
    print("\n  Class detection counts (sorted by frequency):")
    for cls_name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls_name:<25s} : {cnt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
