#!/usr/bin/env python3
"""
infer_dior.py — DA-YOLO Inference on DIOR-R Remote Sensing Images
=================================================================
Runs inference on DIOR-R val images (800×800) using the best trained checkpoint.
No tiling required — images match the training resolution exactly.

Outputs (saved to --output, default: infer_output/dior):
    infer_output/dior/
    ├── visualisations/           ← annotated images (one per input)
    │     05863.jpg, 05864.jpg, ...
    ├── mosaic.jpg                ← N×N grid of sample annotated images
    ├── class_distribution.png   ← detection frequency bar chart
    ├── detections_<stem>.json   ← per-image JSON: boxes, classes, scores
    ├── summary.json             ← run-level stats + class counts
    └── summary.csv              ← flat CSV of all detections

Usage (from project root):
    # Default: first 20 val images, DIOR-R v1 checkpoint
    python infer_dior.py

    # All val images:
    python infer_dior.py --max-images -1

    # Custom source and confidence:
    python infer_dior.py \\
        --weights runs/da_yolo/dior_scratch3/weights/best.pth \\
        --source  /data/DIOR-R/yolo/images/val \\
        --conf    0.30 --iou 0.45 \\
        --max-images 50

    # Single image:
    python infer_dior.py --source /data/DIOR-R/yolo/images/val/05863.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo import DetectionModel
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# ---------------------------------------------------------------------------
# DIOR-R 20-class metadata
# ---------------------------------------------------------------------------

DIOR_NAMES = {
    0:  "airplane",
    1:  "airport",
    2:  "baseballfield",
    3:  "basketballcourt",
    4:  "bridge",
    5:  "chimney",
    6:  "dam",
    7:  "expressway-service-area",
    8:  "expressway-toll-station",
    9:  "golffield",
    10: "groundtrackfield",
    11: "harbor",
    12: "overpass",
    13: "ship",
    14: "stadium",
    15: "storagetank",
    16: "tenniscourt",
    17: "trainstation",
    18: "vehicle",
    19: "windmill",
}

# 20 visually distinct BGR colors (tab20 palette, converted)
_COLORS_BGR = [
    ( 44, 160, 214),   # 0  airplane          — blue
    (255,  99,  71),   # 1  airport           — tomato
    ( 44, 160,  44),   # 2  baseballfield     — green
    (148, 103, 189),   # 3  basketballcourt   — purple
    (214, 139,   0),   # 4  bridge            — amber
    (227, 119, 194),   # 5  chimney           — pink
    ( 23, 190, 207),   # 6  dam               — cyan
    (188, 189,  34),   # 7  expressway-svc    — yellow-green
    (140,  86,  75),   # 8  expressway-toll   — brown
    (255, 127,  14),   # 9  golffield         — orange
    ( 31, 119, 180),   # 10 groundtrackfield  — dark blue
    ( 77, 119,  42),   # 11 harbor            — dark green
    (220,  20,  60),   # 12 overpass          — crimson
    (127, 127, 127),   # 13 ship              — gray
    (  0, 128, 255),   # 14 stadium           — sky blue
    (255, 165,   0),   # 15 storagetank       — orange-amber
    ( 60, 179, 113),   # 16 tenniscourt       — medium sea green
    (255,  20, 147),   # 17 trainstation      — deep pink
    (100, 149, 237),   # 18 vehicle           — cornflower blue
    (154, 205,  50),   # 19 windmill          — yellow-green
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights: Path, device: torch.device) -> DetectionModel:
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    print(f"  Weights  : {weights}")
    ckpt  = torch.load(weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().eval().to(device)
    print(f"  Epoch    : {ckpt.get('epoch', '?')}")
    print(f"  Classes  : {model.nc}  → {list(model.names.values())}")
    return model


# ---------------------------------------------------------------------------
# Inference (direct — no tiling needed for 800×800 DIOR-R images)
# ---------------------------------------------------------------------------

def infer_image(
    model:      DetectionModel,
    img_bgr:    np.ndarray,
    imgsz:      int,
    conf_thres: float,
    iou_thres:  float,
    device:     torch.device,
) -> np.ndarray:
    """
    Infer a single BGR image.
    Returns (N, 6) float32: [x0, y0, x1, y1, conf, cls_id] in original pixel coords.
    """
    h, w = img_bgr.shape[:2]

    # Letterbox to imgsz × imgsz
    scale = imgsz / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    # Preprocess: BGR→RGB, HWC→CHW, normalise
    tensor = torch.from_numpy(
        np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), np.float32) / 255.0
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)

    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    dets = non_max_suppression(
        pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=500
    )[0]

    if dets is None or len(dets) == 0:
        return np.zeros((0, 6), np.float32)

    dets = dets.cpu().numpy().astype(np.float32)
    # Unscale from letterboxed coords back to original pixel coords
    dets[:, :4] /= scale
    dets[:, 0]  = np.clip(dets[:, 0], 0, w)
    dets[:, 1]  = np.clip(dets[:, 1], 0, h)
    dets[:, 2]  = np.clip(dets[:, 2], 0, w)
    dets[:, 3]  = np.clip(dets[:, 3], 0, h)
    return dets


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_detections(img_bgr: np.ndarray, dets: np.ndarray, names: dict) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the image."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    lw   = max(2, int(max(h, w) / 600))
    fs   = max(0.40, max(h, w) / 2000)

    for det in dets:
        x0, y0, x1, y1, conf, cls_id = det
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cls_id = int(cls_id)
        bgr    = _COLORS_BGR[cls_id % len(_COLORS_BGR)]
        label  = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(vis, (x0, y0), (x1, y1), bgr, lw)

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw - 1))
        ty = max(y0, th + bl + 3)
        cv2.rectangle(vis, (x0, ty - th - bl - 3), (x0 + tw + 4, ty), bgr, -1)
        cv2.putText(vis, label, (x0 + 2, ty - bl),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    max(1, lw - 1), cv2.LINE_AA)
    return vis


def make_mosaic(img_paths: list[Path], vis_dir: Path, grid: int = 3,
                cell: int = 400) -> np.ndarray | None:
    """Stitch a grid×grid mosaic from annotated images saved in vis_dir."""
    n = grid * grid
    chosen = img_paths[:n]
    if not chosen:
        return None

    rows = []
    for r in range(grid):
        row_imgs = []
        for c in range(grid):
            i = r * grid + c
            if i < len(chosen):
                vis_path = vis_dir / chosen[i].name
                img = cv2.imread(str(vis_path))
                if img is None:
                    img = np.zeros((cell, cell, 3), dtype=np.uint8)
            else:
                img = np.zeros((cell, cell, 3), dtype=np.uint8)
            row_imgs.append(cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(row_imgs))

    mosaic = np.vstack(rows)
    # Add thin grey grid lines
    for i in range(1, grid):
        mosaic[:, i * cell - 1: i * cell + 1] = 60
        mosaic[i * cell - 1: i * cell + 1, :] = 60
    return mosaic


def plot_class_distribution(class_counts: dict[str, int], save_path: Path) -> None:
    """Bar chart of detection frequencies per class."""
    if not class_counts:
        return

    # Sort by class ID order (not frequency) for readability
    ordered = [(DIOR_NAMES[i], class_counts.get(DIOR_NAMES[i], 0))
               for i in range(len(DIOR_NAMES))]
    names_ordered = [x[0] for x in ordered]
    counts_ordered = [x[1] for x in ordered]

    colors_rgb = [
        (c[2] / 255, c[1] / 255, c[0] / 255)
        for c in _COLORS_BGR
    ]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(range(len(names_ordered)), counts_ordered,
                  color=colors_rgb, edgecolor="white", linewidth=0.5)

    for bar, cnt in zip(bars, counts_ordered):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(names_ordered)))
    ax.set_xticklabels(names_ordered, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Detection Count", fontsize=11)
    ax.set_title("DA-YOLO Detection Frequency — DIOR-R Val Images",
                 fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-image orchestrator
# ---------------------------------------------------------------------------

def process_image(
    image_path: Path,
    model:      DetectionModel,
    names:      dict,
    imgsz:      int,
    conf_thres: float,
    iou_thres:  float,
    device:     torch.device,
    vis_dir:    Path,
    idx:        int,
    total:      int,
) -> tuple[list[dict], float]:
    """Full pipeline for one image. Returns (detection_records, elapsed_sec)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[{idx}/{total}] WARNING: cannot read {image_path.name} — skipping")
        return [], 0.0

    h, w = img.shape[:2]
    t0   = time.perf_counter()
    dets = infer_image(model, img, imgsz, conf_thres, iou_thres, device)
    elapsed = time.perf_counter() - t0

    # Save annotated image
    vis = draw_detections(img, dets, names)
    cv2.imwrite(str(vis_dir / image_path.name), vis)

    # Build records
    records: list[dict] = []
    for det in dets:
        x0, y0, x1, y1, conf, cls_id = det
        cls_id = int(cls_id)
        records.append({
            "image":      image_path.name,
            "class_id":   cls_id,
            "class_name": names.get(cls_id, str(cls_id)),
            "confidence": round(float(conf), 4),
            "bbox_xyxy":  [int(x0), int(y0), int(x1), int(y1)],
            "bbox_xywh":  [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        })

    print(f"[{idx:>4d}/{total}]  {image_path.name}  "
          f"{w}×{h}  →  {len(records):>3d} detections  {elapsed*1000:.1f}ms")
    return records, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DA-YOLO inference on DIOR-R remote sensing images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",    default="runs/da_yolo/dior_scratch3/weights/best.pth")
    p.add_argument("--source",     default="/data/DIOR-R/yolo/images/val",
                   help="Image directory or single image file")
    p.add_argument("--output",     default="infer_output/dior",
                   help="Root output directory")
    p.add_argument("--imgsz",      type=int,   default=800,
                   help="Model input size (match training imgsz)")
    p.add_argument("--conf",       type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--iou",        type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--device",     default="",
                   help="Torch device: '' (auto), 'cpu', '0', ...")
    p.add_argument("--max-images", type=int,   default=20,
                   help="Max images to process; -1 = all val images")
    p.add_argument("--mosaic-grid", type=int,  default=3,
                   help="Grid size for mosaic (e.g. 3 → 3×3 = 9 images)")
    p.add_argument("--mosaic-cell", type=int,  default=500,
                   help="Cell size (px) in mosaic grid")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = select_device(args.device)

    out_root = REPO_ROOT / args.output
    vis_dir  = out_root / "visualisations"
    out_root.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Discover source images
    source = Path(args.source)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if source.is_file():
        image_paths = [source]
    else:
        image_paths = sorted(
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        print(f"No images found in: {source}"); sys.exit(1)

    print("=" * 72)
    print("  DA-YOLO — DIOR-R Inference Pipeline")
    print("=" * 72)
    print(f"  Source     : {source}  ({len(image_paths)} images)")
    print(f"  Weights    : {REPO_ROOT / args.weights}")
    print(f"  Output     : {out_root}")
    print(f"  Img size   : {args.imgsz}px  (DIOR-R native resolution)")
    print(f"  Conf / NMS : {args.conf} / {args.iou}")
    print(f"  Device     : {device}")
    print("=" * 72)

    model = load_model(REPO_ROOT / args.weights, device)
    names = model.names

    all_records:  list[dict]     = []
    class_counts: Counter        = Counter()
    time_per_img: list[float]    = []

    for idx, image_path in enumerate(image_paths, 1):
        records, elapsed = process_image(
            image_path=image_path, model=model, names=names,
            imgsz=args.imgsz, conf_thres=args.conf, iou_thres=args.iou,
            device=device, vis_dir=vis_dir, idx=idx, total=len(image_paths),
        )
        all_records.extend(records)
        time_per_img.append(elapsed)
        for r in records:
            class_counts[r["class_name"]] += 1

        # Per-image JSON
        img_json = {
            "image":      image_path.name,
            "image_size": list(cv2.imread(str(image_path)).shape[:2][::-1]),
            "model":      str(REPO_ROOT / args.weights),
            "conf_thres": args.conf,
            "iou_thres":  args.iou,
            "num_dets":   len(records),
            "detections": records,
        }
        (out_root / f"detections_{image_path.stem}.json").write_text(
            json.dumps(img_json, indent=2)
        )

    # ── Mosaic visualisation ─────────────────────────────────────────────
    print("\nBuilding mosaic...")
    mosaic = make_mosaic(image_paths, vis_dir,
                         grid=args.mosaic_grid, cell=args.mosaic_cell)
    if mosaic is not None:
        cv2.imwrite(str(out_root / "mosaic.jpg"), mosaic)
        print(f"  Mosaic saved: mosaic.jpg  ({args.mosaic_grid}×{args.mosaic_grid} grid)")

    # ── Class distribution chart ─────────────────────────────────────────
    print("Building class distribution chart...")
    plot_class_distribution(dict(class_counts), out_root / "class_distribution.png")

    # ── Summary JSON ─────────────────────────────────────────────────────
    total_elapsed = sum(time_per_img)
    avg_ms        = total_elapsed / max(len(image_paths), 1) * 1000
    fps           = round(1000 / max(avg_ms, 1e-6), 1)

    img_ctr: dict[str, Counter] = defaultdict(Counter)
    for r in all_records:
        img_ctr[r["image"]][r["class_name"]] += 1

    summary = {
        "run_info": {
            "weights":          str(REPO_ROOT / args.weights),
            "source":           str(source),
            "num_images":       len(image_paths),
            "total_detections": len(all_records),
            "imgsz":            args.imgsz,
            "conf_thres":       args.conf,
            "nms_iou":          args.iou,
            "device":           str(device),
            "total_time_s":     round(total_elapsed, 2),
            "avg_ms_per_img":   round(avg_ms, 1),
            "fps":              fps,
        },
        "class_counts": {
            k: v for k, v in sorted(class_counts.items(), key=lambda x: -x[1])
        },
        "per_image": {
            img: {"total": sum(ctr.values()), "by_class": dict(ctr)}
            for img, ctr in img_ctr.items()
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Summary CSV ───────────────────────────────────────────────────────
    fieldnames = ["image", "class_id", "class_name", "confidence",
                  "x0", "y0", "x1", "y1", "width", "height"]
    with open(out_root / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            bbox = r["bbox_xyxy"]
            writer.writerow({
                "image":      r["image"],
                "class_id":   r["class_id"],
                "class_name": r["class_name"],
                "confidence": r["confidence"],
                "x0": bbox[0], "y0": bbox[1],
                "x1": bbox[2], "y1": bbox[3],
                "width":  r["bbox_xywh"][2],
                "height": r["bbox_xywh"][3],
            })

    # ── Final report ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Inference Complete")
    print("=" * 72)
    print(f"  Images processed   : {len(image_paths)}")
    print(f"  Total detections   : {len(all_records)}")
    print(f"  Avg time / image   : {avg_ms:.1f} ms  ({fps} FPS)")
    print(f"  Total elapsed      : {total_elapsed:.1f}s")
    print(f"\n  Output: {out_root}/")
    print(f"  ├── visualisations/           ({len(image_paths)} annotated images)")
    print(f"  ├── mosaic.jpg                ({args.mosaic_grid}×{args.mosaic_grid} sample grid)")
    print(f"  ├── class_distribution.png   (detection frequency by class)")
    print(f"  ├── detections_<stem>.json   (per-image JSON records)")
    print(f"  ├── summary.json             (run stats + class counts)")
    print(f"  └── summary.csv              ({len(all_records)} detection rows)")
    print("\n  Top-10 detected classes:")
    for cls_name, cnt in class_counts.most_common(10):
        bar = "█" * min(40, int(cnt / max(class_counts.values()) * 40))
        print(f"    {cls_name:<30s} {cnt:>5d}  {bar}")
    print("=" * 72)


if __name__ == "__main__":
    main()
