#!/usr/bin/env python3
"""
evaluate_visdrone_sahi.py — DA-YOLO Tiled (SAHI-style) Evaluation on VisDrone
===============================================================================
Measures mAP on the full VisDrone2019-DET validation split (548 images) using
Slicing Aided Hyper Inference (SAHI):

  • Each 1920×1080 image is sliced into tile_size×tile_size crops (default 640px)
    with fractional overlap (default 25%).
  • Each crop is upscaled to the model's training size (model_size, default 1280px)
    before inference — this 2× upscaling makes tiny 4–8px objects detectable.
  • Detections from all tiles are mapped back to full-image pixel coordinates and
    merged via class-aware NMS.
  • mAP@0.5 and mAP@0.5:0.95 are computed against ground-truth YOLO label files.

Key difference from evaluate_visdrone.py:
  Old script: single forward pass at 1280px → tiny objects shrink below detection
  This script: 640px tiles → each upscaled 2× before model → tiny objects remain
               within the model's anchor range at the P2 detection head.

Outputs (saved to --save-dir):
  eval_results/visdrone_sahi/
  ├── metrics_table.txt       — summary table with SAHI vs baseline comparison
  ├── per_class_ap.txt        — per-class AP (useful to see bicycle/tricycle gain)
  ├── PR_curve.png
  ├── F1_curve.png
  ├── P_curve.png
  ├── R_curve.png
  ├── per_class_ap_bar.png
  └── raw_results.json

Usage:
    # Default (tile=640, model=1280, overlap=0.25):
    python evaluate_visdrone_sahi.py

    # Try larger tiles (less upsampling, faster):
    python evaluate_visdrone_sahi.py --tile-size 800 --model-size 1280

    # Compare different tile sizes:
    python evaluate_visdrone_sahi.py --tile-size 640  --save-dir eval_results/sahi_640
    python evaluate_visdrone_sahi.py --tile-size 1280 --save-dir eval_results/sahi_1280
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo import DetectionModel
from utils.general import non_max_suppression
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device

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
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights: Path, device: torch.device) -> DetectionModel:
    ckpt  = torch.load(weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().eval().to(device)
    print(f"  Epoch      : {ckpt.get('epoch', '?')}")
    print(f"  Classes    : {list(model.names.values())}")
    return model


# ---------------------------------------------------------------------------
# Tiling helpers  (shared with infer_visdrone.py logic)
# ---------------------------------------------------------------------------

def compute_tiles(
    img_h: int, img_w: int, tile_size: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) tile coords covering the full image."""
    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles: list[tuple[int, int, int, int]] = []
    y0 = 0
    while True:
        y1 = min(y0 + tile_size, img_h)
        x0 = 0
        while True:
            x1 = min(x0 + tile_size, img_w)
            tiles.append((x0, y0, x1, y1))
            if x1 == img_w:
                break
            x0 += stride
        if y1 == img_h:
            break
        y0 += stride
    return tiles


def infer_tile(
    model:      DetectionModel,
    tile_bgr:   np.ndarray,
    model_size: int,
    conf_thres: float,
    device:     torch.device,
) -> np.ndarray:
    """
    Infer one BGR tile.  The tile is letterbox-padded to (model_size, model_size).
    Returns (N, 6) float32: [x0, y0, x1, y1, conf, cls_id] in *tile* pixel coords.
    """
    h, w   = tile_bgr.shape[:2]
    scale  = model_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(tile_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((model_size, model_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    img_t = canvas[:, :, ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(img_t, np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)

    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.6, max_det=1000)
    det = det[0]

    if det is None or len(det) == 0:
        return np.zeros((0, 6), np.float32)

    det = det.cpu().numpy().astype(np.float32)
    det[:, :4] /= scale          # back to tile pixel coords
    det[:, 0] = np.clip(det[:, 0], 0, w)
    det[:, 1] = np.clip(det[:, 1], 0, h)
    det[:, 2] = np.clip(det[:, 2], 0, w)
    det[:, 3] = np.clip(det[:, 3], 0, h)
    return det


def cross_patch_nms(dets: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy class-aware NMS over detections from multiple tiles."""
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
        with np.errstate(divide="ignore", invalid="ignore"):
            iou  = np.where(union > 0, inter / union, 0.0)
        same_cls = dets[rest, 5] == dets[idx, 5]
        order    = rest[~(same_cls & (iou > iou_thres))]

    return dets[keep]


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_gt_boxes(lbl_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Load YOLO-format label file and return (M, 5) float32 array:
        [cls, x0, y0, x1, y1]  in full-image pixel coordinates.
    """
    if not lbl_path.exists():
        return np.zeros((0, 5), np.float32)

    boxes = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id            = float(parts[0])
        cx, cy, w, h      = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        cx_px, cy_px      = cx * img_w, cy * img_h
        w_px,  h_px       = w  * img_w, h  * img_h
        x0 = cx_px - w_px / 2
        y0 = cy_px - h_px / 2
        x1 = cx_px + w_px / 2
        y1 = cy_px + h_px / 2
        boxes.append([cls_id, x0, y0, x1, y1])

    return np.array(boxes, np.float32) if boxes else np.zeros((0, 5), np.float32)


# ---------------------------------------------------------------------------
# Prediction–GT matching (vectorised, multi-IoU-threshold)
# ---------------------------------------------------------------------------

def match_predictions(
    pred_boxes: torch.Tensor,   # (N, 4) xyxy
    pred_cls:   torch.Tensor,   # (N,)
    gt:         torch.Tensor,   # (M, 5)  [cls, x0, y0, x1, y1]
    iouv:       torch.Tensor,   # (10,) IoU thresholds
) -> torch.Tensor:
    """
    Returns (N, 10) bool tensor: correct[i, j] = True if prediction i is a TP
    at IoU threshold iouv[j].
    """
    niou    = iouv.numel()
    correct = torch.zeros(pred_boxes.shape[0], niou, dtype=torch.bool, device=iouv.device)

    if gt.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return correct

    iou          = box_iou(gt[:, 1:], pred_boxes)           # (M, N)
    correct_cls  = gt[:, 0:1] == pred_cls.unsqueeze(0)      # (M, N) bool

    for j, thr in enumerate(iouv):
        # Pairs that satisfy both class match and IoU threshold
        m_gt, m_pred = torch.where((iou >= thr) & correct_cls)

        if m_gt.numel() == 0:
            continue

        matches = torch.stack((m_gt, m_pred, iou[m_gt, m_pred]), 1).cpu().numpy()

        # Greedy best-match: highest IoU first, each GT/pred used once
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # unique pred
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # unique GT
        correct[matches[:, 1].astype(int), j] = True

    return correct


# ---------------------------------------------------------------------------
# Core SAHI evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sahi(
    model:      DetectionModel,
    val_img_dir: Path,
    val_lbl_dir: Path,
    tile_size:  int,
    overlap:    float,
    model_size: int,
    conf_thres: float,
    nms_iou:    float,
    nc:         int,
    names:      dict,
    device:     torch.device,
    save_dir:   Path,
) -> dict:
    """Full SAHI evaluation loop. Returns metrics dict."""
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    img_paths = sorted(p for p in val_img_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if not img_paths:
        raise RuntimeError(f"No images found in {val_img_dir}")

    stats:      list[tuple] = []
    total_time: float       = 0.0
    seen:       int         = 0

    tile_counts: list[int] = []

    for img_path in tqdm(img_paths, desc=f"SAHI eval (tile={tile_size}px→{model_size}px)", unit="img"):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        img_h, img_w = img_bgr.shape[:2]
        seen += 1

        # ── Tiled inference ──────────────────────────────────────────────
        tiles    = compute_tiles(img_h, img_w, tile_size, overlap)
        all_dets: list[np.ndarray] = []
        t0 = time.perf_counter()

        for x0, y0, x1, y1 in tiles:
            patch = img_bgr[y0:y1, x0:x1]
            d     = infer_tile(model, patch, model_size, conf_thres, device)
            if len(d):
                d[:, 0] += x0;  d[:, 2] += x0
                d[:, 1] += y0;  d[:, 3] += y0
                all_dets.append(d)

        total_time += time.perf_counter() - t0
        tile_counts.append(len(tiles))

        if all_dets:
            dets = np.concatenate(all_dets, 0)
            dets = cross_patch_nms(dets, nms_iou)
        else:
            dets = np.zeros((0, 6), np.float32)

        # ── Load GT ──────────────────────────────────────────────────────
        lbl_path = val_lbl_dir / img_path.with_suffix(".txt").name
        gt_np    = load_gt_boxes(lbl_path, img_w, img_h)    # (M, 5) [cls,x0,y0,x1,y1]
        gt       = torch.from_numpy(gt_np).to(device)

        # ── Build prediction tensors ──────────────────────────────────────
        if len(dets):
            pred_boxes = torch.from_numpy(dets[:, :4]).to(device)
            pred_conf  = torch.from_numpy(dets[:, 4]).to(device)
            pred_cls   = torch.from_numpy(dets[:, 5]).to(device)
        else:
            pred_boxes = torch.zeros((0, 4), device=device)
            pred_conf  = torch.zeros(0, device=device)
            pred_cls   = torch.zeros(0, device=device)

        # ── Match predictions to GT ───────────────────────────────────────
        correct = torch.zeros(len(dets), niou, dtype=torch.bool, device=device)
        if len(dets) and len(gt):
            correct = match_predictions(pred_boxes, pred_cls, gt, iouv)

        stats.append((
            correct.cpu(),
            pred_conf.cpu(),
            pred_cls.cpu(),
            gt[:, 0].cpu() if len(gt) else torch.zeros(0),
        ))

    # ── Compute mAP ──────────────────────────────────────────────────────
    if not stats:
        raise RuntimeError("No stats collected — check val_img_dir and val_lbl_dir paths.")

    stats_cat = [torch.cat(x, 0).numpy() for x in zip(*stats)]

    if len(stats_cat) and stats_cat[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats_cat, plot=True, save_dir=save_dir, names=names
        )
        ap50    = ap[:, 0]
        ap_avg  = ap.mean(1)
        mp, mr  = float(p.mean()), float(r.mean())
        mf1     = float(f1.mean())
        map50   = float(ap50.mean())
        map5095 = float(ap_avg.mean())
        per_cls = {
            names[int(c)]: {"AP50": float(ap50[i]), "AP5095": float(ap_avg[i])}
            for i, c in enumerate(ap_class)
        }
    else:
        mp = mr = mf1 = map50 = map5095 = 0.0
        per_cls = {}

    avg_tiles   = np.mean(tile_counts)
    avg_ms_img  = total_time / max(seen, 1) * 1000

    return {
        "precision":   mp,
        "recall":      mr,
        "f1":          mf1,
        "mAP50":       map50,
        "mAP5095":     map5095,
        "per_class":   per_cls,
        "num_images":  seen,
        "avg_tiles":   round(avg_tiles, 1),
        "avg_ms_img":  round(avg_ms_img, 1),
        "fps":         round(1000 / max(avg_ms_img, 1e-6), 1),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_table(metrics: dict, baseline: dict | None, tile_size: int,
                model_size: int, overlap: float, save_dir: Path) -> str:
    gain = ""
    if baseline:
        delta = (metrics["mAP50"] - baseline["mAP50"]) * 100
        gain  = f"  (+{delta:+.1f}% vs. {baseline['mAP50']*100:.2f}% baseline)" if delta >= 0 \
                else f"  ({delta:+.1f}% vs. {baseline['mAP50']*100:.2f}% baseline)"

    header = (
        f"  Tile size (crop)   : {tile_size}px  →  model input: {model_size}px  "
        f"(upscale ×{model_size//tile_size:.1f})  overlap: {overlap:.0%}\n"
        f"  Avg tiles/image    : {metrics['avg_tiles']:.1f}\n"
        f"  Images evaluated   : {metrics['num_images']}\n"
    )

    lines = [
        "=" * 72,
        "  DA-YOLO — VisDrone2019-DET SAHI Evaluation",
        "=" * 72,
        header,
        f"  {'Metric':<35s} {'Value':>10s}",
        "-" * 72,
        f"  {'Precision (P)':.<35s} {metrics['precision']:>10.4f}",
        f"  {'Recall (R)':.<35s} {metrics['recall']:>10.4f}",
        f"  {'F1 Score':.<35s} {metrics['f1']:>10.4f}",
        f"  {'mAP @ IoU=0.50':.<35s} {metrics['mAP50']:>10.4f}{gain}",
        f"  {'mAP @ IoU=0.50:0.95':.<35s} {metrics['mAP5095']:>10.4f}",
        f"  {'Avg inference (ms/img, tiled)':.<35s} {metrics['avg_ms_img']:>10.1f}",
        "-" * 72,
        "  Per-class AP @ IoU=0.50:",
        "-" * 72,
        f"  {'Class':<25s} {'AP@0.5':>10s} {'AP@.5:.95':>12s}",
        "-" * 72,
    ]

    if baseline:
        lines.append(f"  {'Class':<25s} {'AP@0.5':>10s} {'AP@.5:.95':>12s} {'Δ AP@.5':>10s}")
        lines.append("-" * 72)
        for cls_name, vals in metrics["per_class"].items():
            base_ap50 = baseline.get("per_class", {}).get(cls_name, {}).get("AP50", None)
            delta_str = f"{(vals['AP50'] - base_ap50)*100:>+9.2f}%" if base_ap50 is not None else ""
            lines.append(
                f"  {cls_name:<25s} {vals['AP50']:>10.4f} {vals['AP5095']:>12.4f} {delta_str}"
            )
    else:
        for cls_name, vals in metrics["per_class"].items():
            lines.append(f"  {cls_name:<25s} {vals['AP50']:>10.4f} {vals['AP5095']:>12.4f}")

    lines.append("=" * 72)
    table = "\n".join(lines)
    (save_dir / "metrics_table.txt").write_text(table + "\n")
    return table


def write_per_class_txt(metrics: dict, save_dir: Path) -> None:
    lines = [f"{'Class':<25s} {'AP@0.5':>10s} {'AP@.5:.95':>12s}", "-" * 50]
    for name, v in metrics["per_class"].items():
        lines.append(f"{name:<25s} {v['AP50']*100:>10.2f} {v['AP5095']*100:>12.2f}")
    lines += ["-" * 50, f"{'mAP':<25s} {metrics['mAP50']*100:>10.2f} {metrics['mAP5095']*100:>12.2f}"]
    (save_dir / "per_class_ap.txt").write_text("\n".join(lines) + "\n")


def plot_per_class_bar(metrics: dict, baseline: dict | None, save_dir: Path) -> None:
    classes = list(metrics["per_class"].keys())
    ap50    = [v["AP50"]   * 100 for v in metrics["per_class"].values()]

    x     = np.arange(len(classes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5))

    bars1 = ax.bar(x - width / 2 if baseline else x, ap50, width,
                   label="SAHI AP@0.5", color="#4C72B0", alpha=0.85)

    if baseline:
        base_ap50 = [baseline.get("per_class", {}).get(c, {}).get("AP50", 0) * 100
                     for c in classes]
        bars2 = ax.bar(x + width / 2, base_ap50, width,
                       label="Baseline AP@0.5", color="#DD8452", alpha=0.65)
        for b1, b2, cls in zip(bars1, bars2, classes):
            delta = b1.get_height() - b2.get_height()
            ax.text(b1.get_x() + b1.get_width() / 2, b1.get_height() + 0.5,
                    f"{delta:+.1f}", ha="center", va="bottom", fontsize=7.5, color="#1a5e1a")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("AP@0.5 (%)", fontsize=12)
    ax.set_title("DA-YOLO — Per-Class AP: SAHI vs Baseline (VisDrone Val)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)
    ax.axhline(np.mean(ap50), color="#4C72B0", linestyle="--", linewidth=1.0, alpha=0.7)
    fig.tight_layout()
    fig.savefig(save_dir / "per_class_ap_bar.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DA-YOLO SAHI (tiled) evaluation on VisDrone2019-DET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",
                   default="runs/da_yolo/visdrone_scratch2/weights/best.pth")
    p.add_argument("--val-img-dir",
                   default="VisDrone_dataset/yolo/images/val")
    p.add_argument("--val-lbl-dir",
                   default="VisDrone_dataset/yolo/labels/val")
    p.add_argument("--tile-size",   type=int,   default=640,
                   help="Crop size from original image (px). Upscaled to model-size before inference.")
    p.add_argument("--model-size",  type=int,   default=1280,
                   help="Model input size (px) — must match training imgsz.")
    p.add_argument("--overlap",     type=float, default=0.25,
                   help="Fractional tile overlap [0, 0.9].")
    p.add_argument("--conf",        type=float, default=0.001,
                   help="Confidence threshold (low for full PR curve).")
    p.add_argument("--nms-iou",     type=float, default=0.55,
                   help="Cross-patch NMS IoU threshold.")
    p.add_argument("--device",      default="",
                   help="Torch device: '' (auto), '0', 'cpu'.")
    p.add_argument("--save-dir",    default="eval_results/visdrone_sahi",
                   help="Directory to save outputs.")
    p.add_argument("--baseline-json",
                   default="eval_results/visdrone_best/raw_results.json",
                   help="Path to baseline raw_results.json for comparison delta.")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    device   = select_device(args.device)
    save_dir = REPO_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline for comparison delta in table
    baseline = None
    baseline_path = REPO_ROOT / args.baseline_json
    if baseline_path.exists():
        with open(baseline_path) as f:
            raw = json.load(f)
        baseline = raw.get("metrics", None)
        if baseline:
            print(f"\n  Baseline loaded: mAP@0.5 = {baseline['mAP50']*100:.2f}%")

    print("\n" + "=" * 70)
    print("  DA-YOLO — SAHI Evaluation")
    print("=" * 70)
    print(f"  Weights     : {args.weights}")
    print(f"  Tile size   : {args.tile_size}px  →  model: {args.model_size}px  "
          f"(×{args.model_size / args.tile_size:.1f} upscale)")
    print(f"  Overlap     : {args.overlap:.0%}")
    print(f"  Conf        : {args.conf}")
    print(f"  NMS IoU     : {args.nms_iou}")
    print(f"  Device      : {device}")
    print(f"  Save dir    : {save_dir}")
    print("=" * 70)

    model = load_model(REPO_ROOT / args.weights, device)
    nc    = model.nc
    names = model.names

    val_img_dir = REPO_ROOT / args.val_img_dir
    val_lbl_dir = REPO_ROOT / args.val_lbl_dir

    metrics = evaluate_sahi(
        model       = model,
        val_img_dir = val_img_dir,
        val_lbl_dir = val_lbl_dir,
        tile_size   = args.tile_size,
        overlap     = args.overlap,
        model_size  = args.model_size,
        conf_thres  = args.conf,
        nms_iou     = args.nms_iou,
        nc          = nc,
        names       = names,
        device      = device,
        save_dir    = save_dir,
    )

    # ── Save outputs ─────────────────────────────────────────────────────
    table = write_table(metrics, baseline, args.tile_size, args.model_size,
                        args.overlap, save_dir)
    print("\n" + table)

    write_per_class_txt(metrics, save_dir)
    plot_per_class_bar(metrics, baseline, save_dir)

    full = {
        "weights":    str(args.weights),
        "tile_size":  args.tile_size,
        "model_size": args.model_size,
        "overlap":    args.overlap,
        "conf":       args.conf,
        "nms_iou":    args.nms_iou,
        "metrics":    metrics,
    }
    (save_dir / "raw_results.json").write_text(json.dumps(full, indent=2))
    print(f"\n  Saved to: {save_dir}\n")


if __name__ == "__main__":
    main()
