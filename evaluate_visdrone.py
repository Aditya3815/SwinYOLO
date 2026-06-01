#!/usr/bin/env python3
"""
evaluate_visdrone.py — DA-YOLO Comprehensive Evaluation on VisDrone
=====================================================================
Computes paper-ready metrics on the VisDrone2019-DET validation split:
  • mAP@0.5 and mAP@0.5:0.95 (COCO-style)
  • Per-class AP@0.5
  • Precision, Recall, F1 at optimal confidence threshold
  • Model complexity: parameters and GFLOPs
  • Inference speed: FPS on GPU / CPU

Outputs (saved to --save-dir):
  eval_results/
  ├── metrics_table.txt       — human-readable summary table
  ├── metrics_table.tex       — LaTeX table ready for paper
  ├── per_class_ap.txt        — per-class AP with GT counts
  ├── PR_curve.png            — precision-recall curve (all classes)
  ├── F1_curve.png            — F1-confidence curve
  ├── P_curve.png             — precision-confidence curve
  ├── R_curve.png             — recall-confidence curve
  ├── per_class_ap_bar.png    — per-class AP bar chart
  ├── training_curves.png     — loss + mAP curves from results.csv
  └── raw_results.json        — machine-readable JSON with all numbers

Usage (from project root):
    python evaluate_visdrone.py

    # Override defaults:
    python evaluate_visdrone.py \\
        --weights runs/da_yolo/visdrone_scratch2/weights/best.pth \\
        --data    data/visdrone.yaml \\
        --imgsz   1280 \\
        --batch   8 \\
        --conf    0.001 \\
        --iou     0.65 \\
        --device  0 \\
        --save-dir eval_results/visdrone_best
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo import DetectionModel
from utils.dataloaders import create_dataloader
from utils.general import (
    check_dataset,
    check_img_size,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    colorstr,
)
from utils.metrics import ap_per_class, box_iou, ConfusionMatrix
from utils.torch_utils import select_device, de_parallel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_bifpn_se_compat(model: DetectionModel) -> None:
    """Inject nn.Identity() for any missing SE attributes in old BiFPNLayer checkpoints."""
    SE_ATTRS = ("se_p4_td", "se_p3_td", "se_p2_out", "se_p3_out", "se_p4_out", "se_p5_out")
    for module in model.modules():
        if type(module).__name__ == "BiFPNLayer":
            for attr in SE_ATTRS:
                if not hasattr(module, attr):
                    setattr(module, attr, torch.nn.Identity())


def load_checkpoint(weights: str | Path, device: torch.device) -> tuple[DetectionModel, dict]:
    """Load DetectionModel from a .pth checkpoint (EMA preferred)."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model: DetectionModel = (ckpt.get("ema") or ckpt["model"]).float().eval()
    _patch_bifpn_se_compat(model)
    model = model.to(device)

    info = {
        "epoch":        ckpt.get("epoch", -1),
        "best_fitness": float(np.asarray(ckpt["best_fitness"]).flat[0]) if "best_fitness" in ckpt else None,
        "date":         ckpt.get("date", "unknown"),
    }
    return model, info


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_gflops(model: DetectionModel, imgsz: int, device: torch.device) -> float:
    """Estimate GFLOPs via a single forward pass profile."""
    try:
        from thop import profile as thop_profile
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        flops, _ = thop_profile(de_parallel(model), inputs=(dummy,), verbose=False)
        return round(flops / 1e9, 2)
    except Exception:
        return float("nan")


def warmup(model: DetectionModel, imgsz: int, device: torch.device, reps: int = 5) -> None:
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    for _ in range(reps):
        with torch.no_grad():
            model(dummy)


def measure_fps(model: DetectionModel, imgsz: int, device: torch.device, reps: int = 100) -> float:
    """Measure single-image inference FPS (includes model forward only, no NMS)."""
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    warmup(model, imgsz, device, reps=10)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return round(reps / elapsed, 1)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      DetectionModel,
    dataloader: torch.utils.data.DataLoader,
    names:      dict,
    nc:         int,
    conf_thres: float,
    iou_thres:  float,
    device:     torch.device,
    save_dir:   Path,
    half:       bool = False,
    verbose:    bool = True,
) -> dict:
    """
    Run inference over the dataloader and accumulate statistics for mAP.
    Returns a dict with all metric tensors and arrays.
    """
    model.half() if half else model.float()

    # IoU thresholds: 0.50, 0.55, …, 0.95  (COCO-style, 10 thresholds)
    iouv  = torch.linspace(0.5, 0.95, 10, device=device)
    niou  = iouv.numel()

    confusion = ConfusionMatrix(nc=nc)

    stats:     list[tuple] = []   # (tp, conf, pred_cls, target_cls) per image
    seen       = 0
    total_time = 0.0

    if verbose:
        print(colorstr("bold", f"\n{'Image':>12s} {'Labels':>10s} {'P':>10s} {'R':>10s} "
                               f"{'mAP@.5':>10s} {'mAP@.5:.95':>12s}"))

    pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
        imgs    = imgs.to(device, non_blocking=True)
        imgs    = imgs.half() if half else imgs.float()
        imgs   /= 255.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape

        # Forward pass timing
        t0 = time.perf_counter()
        out, train_out = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        # NMS
        out = non_max_suppression(
            out,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            labels=[],
            multi_label=True,
            agnostic=False,
            max_det=300,
        )

        for si, pred in enumerate(out):
            labels   = targets[targets[:, 0] == si, 1:]
            nl, npr  = labels.shape[0], pred.shape[0]
            path     = Path(paths[si])
            correct  = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen    += 1

            if npr == 0:
                if nl:
                    stats.append((
                        correct,
                        torch.zeros(0, device=device),
                        torch.zeros(0, device=device),
                        labels[:, 0],
                    ))
                continue

            # Scale predictions to original image coords
            predn = pred.clone()
            scale_boxes(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                # Labels are normalized [0,1] relative to the letterboxed image;
                # scale_boxes expects pixel coords, so denormalize first.
                img_h, img_w = imgs[si].shape[1], imgs[si].shape[2]
                tbox[:, [0, 2]] *= img_w
                tbox[:, [1, 3]] *= img_h
                scale_boxes(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels

                # Evaluate per IoU threshold
                correct = _match_predictions(predn[:, :4], predn[:, 5], labelsn, iouv)
                confusion.process_batch(predn, labelsn)

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels[:, 0].cpu()))

    # Compute metrics
    stats_cat = [torch.cat(x, 0).numpy() for x in zip(*stats)]
    metrics   = {}

    if len(stats_cat) and stats_cat[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats_cat,
            plot=True,
            save_dir=save_dir,
            names=names,
        )
        ap50, ap  = ap[:, 0], ap.mean(1)   # AP@0.5, AP@0.5:0.95
        mp, mr    = p.mean(), r.mean()
        mf1       = f1.mean()
        map50     = ap50.mean()
        map5095   = ap.mean()
    else:
        tp = fp = p = r = f1 = ap50 = ap = torch.zeros(1)
        mp = mr = mf1 = map50 = map5095 = 0.0
        ap_class = []

    metrics.update({
        "precision":   float(mp),
        "recall":      float(mr),
        "f1":          float(mf1),
        "mAP50":       float(map50),
        "mAP5095":     float(map5095),
        "per_class": {
            names[int(c)]: {
                "AP50":   float(ap50[i]),
                "AP5095": float(ap[i]),
            }
            for i, c in enumerate(ap_class)
        },
        "num_images":   seen,
        "infer_ms":     round(total_time / max(seen, 1) * 1000, 2),
    })
    return metrics


def _match_predictions(
    pboxes: torch.Tensor,
    pcls:   torch.Tensor,
    labels: torch.Tensor,
    iouv:   torch.Tensor,
) -> torch.Tensor:
    """Match predictions to ground-truth boxes across IoU thresholds."""
    correct = torch.zeros(pboxes.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], pboxes)
    correct_class = labels[:, 0:1] == pcls

    for i, threshold in enumerate(iouv):
        x = torch.where((iou >= threshold) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_text_table(metrics: dict, ckpt_info: dict, model_info: dict, save_dir: Path) -> str:
    """Write a human-readable metrics table and return it as a string."""
    lines = [
        "=" * 72,
        "  DA-YOLO — VisDrone2019-DET Evaluation Results",
        "=" * 72,
        f"  Checkpoint epoch   : {ckpt_info['epoch']}",
        f"  Best fitness (train): {ckpt_info['best_fitness']:.5f}" if ckpt_info['best_fitness'] else "",
        f"  Parameters         : {model_info['params'] / 1e6:.2f} M",
        f"  GFLOPs             : {model_info['gflops']}",
        f"  Images evaluated   : {metrics['num_images']}",
        "-" * 72,
        f"  {'Metric':<30s} {'Value':>12s}",
        "-" * 72,
        f"  {'Precision (P)':.<30s} {metrics['precision']:>11.4f}",
        f"  {'Recall (R)':.<30s} {metrics['recall']:>11.4f}",
        f"  {'F1 Score':.<30s} {metrics['f1']:>11.4f}",
        f"  {'mAP @ IoU=0.50':.<30s} {metrics['mAP50']:>11.4f}",
        f"  {'mAP @ IoU=0.50:0.95':.<30s} {metrics['mAP5095']:>11.4f}",
        f"  {'Inference speed (ms/img)':.<30s} {metrics['infer_ms']:>11.2f}",
        f"  {'FPS (GPU forward only)':.<30s} {model_info['fps']:>11.1f}",
        "-" * 72,
        "  Per-class AP @ IoU=0.50:",
        "-" * 72,
        f"  {'Class':<25s} {'GT':>6s} {'AP@0.5':>10s} {'AP@.5:.95':>12s}",
        "-" * 72,
    ]
    for cls_name, vals in metrics["per_class"].items():
        lines.append(
            f"  {cls_name:<25s} {'':>6s} {vals['AP50']:>10.4f} {vals['AP5095']:>12.4f}"
        )
    lines.append("=" * 72)
    table = "\n".join(l for l in lines if l is not None)
    (save_dir / "metrics_table.txt").write_text(table + "\n")
    return table


def write_latex_table(metrics: dict, model_info: dict, save_dir: Path) -> None:
    """Write a LaTeX table suitable for direct inclusion in a paper."""
    cls_rows = "\n".join(
        f"        {name.replace('-', r'\text{-}')} & {v['AP50']*100:.1f} & {v['AP5095']*100:.1f} \\\\"
        for name, v in metrics["per_class"].items()
    )
    latex = rf"""% DA-YOLO VisDrone2019-DET Results — auto-generated by evaluate_visdrone.py
\begin{{table}}[t]
\centering
\caption{{DA-YOLO performance on the VisDrone2019-DET validation split.
  P: precision, R: recall, mAP$_{{50}}$: AP at IoU\,=\,0.50,
  mAP$_{{50:95}}$: COCO-style AP averaged over IoU$\in[0.50,0.95]$.}}
\label{{tab:visdrone_results}}
\setlength{{\tabcolsep}}{{6pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Method & P & R & mAP$_{{50}}$ & mAP$_{{50:95}}$ \\
\midrule
DA-YOLO (ours) & {metrics['precision']*100:.1f} & {metrics['recall']*100:.1f} & {metrics['mAP50']*100:.1f} & {metrics['mAP5095']*100:.1f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

% ---- Per-class results ----
\begin{{table}}[t]
\centering
\caption{{Per-class AP on VisDrone2019-DET validation split (DA-YOLO).}}
\label{{tab:visdrone_per_class}}
\begin{{tabular}}{{lrr}}
\toprule
Class & AP$_{{50}}$ (\%) & AP$_{{50:95}}$ (\%) \\
\midrule
{cls_rows}
\midrule
\textbf{{mAP}} & \textbf{{{metrics['mAP50']*100:.1f}}} & \textbf{{{metrics['mAP5095']*100:.1f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (save_dir / "metrics_table.tex").write_text(latex)


def write_per_class_txt(metrics: dict, save_dir: Path) -> None:
    lines = [f"{'Class':<25s} {'AP@0.5':>10s} {'AP@.5:.95':>12s}"]
    lines.append("-" * 50)
    for name, v in metrics["per_class"].items():
        lines.append(f"{name:<25s} {v['AP50']*100:>10.2f} {v['AP5095']*100:>12.2f}")
    lines.append("-" * 50)
    lines.append(f"{'mAP':<25s} {metrics['mAP50']*100:>10.2f} {metrics['mAP5095']*100:>12.2f}")
    (save_dir / "per_class_ap.txt").write_text("\n".join(lines) + "\n")


def plot_per_class_ap(metrics: dict, save_dir: Path) -> None:
    """Bar chart of per-class AP@0.5 and AP@0.5:0.95."""
    classes = list(metrics["per_class"].keys())
    ap50    = [v["AP50"]   * 100 for v in metrics["per_class"].values()]
    ap5095  = [v["AP5095"] * 100 for v in metrics["per_class"].values()]

    x      = np.arange(len(classes))
    width  = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width / 2, ap50,   width, label="AP@0.5",       color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ap5095, width, label="AP@0.5:0.95",  color="#DD8452", alpha=0.85)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Average Precision (%)", fontsize=12)
    ax.set_title("DA-YOLO — Per-Class AP on VisDrone2019-DET Val", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)

    # Value labels on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    # Horizontal mean lines
    ax.axhline(np.mean(ap50),   color="#4C72B0", linestyle="--", linewidth=1.0, alpha=0.7, label=f"mean AP@.5={np.mean(ap50):.1f}")
    ax.axhline(np.mean(ap5095), color="#DD8452", linestyle="--", linewidth=1.0, alpha=0.7, label=f"mean AP@.5:.95={np.mean(ap5095):.1f}")

    fig.tight_layout()
    fig.savefig(save_dir / "per_class_ap_bar.png", dpi=150)
    plt.close(fig)


def plot_training_curves(results_csv: Path, save_dir: Path) -> None:
    """Plot training loss + val mAP from the training results.csv."""
    if not results_csv.exists():
        return
    import csv
    rows = list(csv.DictReader(open(results_csv)))
    if not rows:
        return

    def col(key: str) -> np.ndarray:
        stripped = {k.strip(): v for k, v in rows[0].items()}
        # Find matching column (strip whitespace from header)
        for k in rows[0].keys():
            if k.strip() == key:
                return np.array([float(r[k]) for r in rows])
        return None

    epochs   = np.arange(1, len(rows) + 1)
    box_loss = col("train/box_loss")
    obj_loss = col("train/obj_loss")
    cls_loss = col("train/cls_loss")
    map50    = col("metrics/mAP_0.5")
    map5095  = col("metrics/mAP_0.5:0.95")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Loss curves
    ax = axes[0]
    if box_loss is not None: ax.plot(epochs, box_loss, label="Box loss",  color="#4C72B0")
    if obj_loss is not None: ax.plot(epochs, obj_loss, label="Obj loss",  color="#DD8452")
    if cls_loss is not None: ax.plot(epochs, cls_loss, label="Cls loss",  color="#55A868")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Losses", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # mAP curves
    ax = axes[1]
    if map50   is not None: ax.plot(epochs, map50,   label="mAP@0.5",      color="#4C72B0")
    if map5095 is not None: ax.plot(epochs, map5095, label="mAP@0.5:0.95", color="#DD8452")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP", fontsize=12)
    ax.set_title("Validation mAP", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("DA-YOLO Training on VisDrone2019-DET", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DA-YOLO comprehensive evaluation on VisDrone2019-DET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",  default="runs/da_yolo/visdrone_scratch2/weights/best.pth",
                   help="Path to *.pth checkpoint")
    p.add_argument("--data",     default="data/visdrone.yaml",
                   help="Dataset YAML (must have val: split)")
    p.add_argument("--imgsz",    type=int, default=1280,
                   help="Inference image size (pixels)")
    p.add_argument("--batch",    type=int, default=8,
                   help="Batch size for dataloader")
    p.add_argument("--conf",     type=float, default=0.001,
                   help="Confidence threshold (low value for full PR curve)")
    p.add_argument("--iou",      type=float, default=0.65,
                   help="NMS IoU threshold")
    p.add_argument("--device",   default="",
                   help="Torch device: '' (auto), 'cpu', '0', ...")
    p.add_argument("--workers",  type=int, default=4,
                   help="Dataloader workers")
    p.add_argument("--half",     action="store_true",
                   help="FP16 inference (GPU only)")
    p.add_argument("--save-dir", default="eval_results/visdrone_best",
                   help="Directory to save outputs")
    p.add_argument("--results-csv",
                   default="runs/da_yolo/visdrone_scratch2/results.csv",
                   help="Training results.csv for training curve plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    save_dir = REPO_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    half   = args.half and device.type != "cpu"

    # ── Load model ──────────────────────────────────────────────────────────
    print(colorstr("bold", "\n[1/5] Loading checkpoint..."))
    model, ckpt_info = load_checkpoint(REPO_ROOT / args.weights, device)
    stride = int(model.stride.max())
    imgsz  = check_img_size(args.imgsz, s=stride)
    names  = model.names   # {0: 'pedestrian', ...}
    nc     = model.nc

    print(f"  Epoch          : {ckpt_info['epoch']}")
    print(f"  Classes        : {nc}  → {list(names.values())}")

    # ── Model complexity ────────────────────────────────────────────────────
    print(colorstr("bold", "\n[2/5] Computing model complexity & speed..."))
    n_params = count_parameters(model)
    gflops   = compute_gflops(model, imgsz, device)
    fps      = measure_fps(model, imgsz, device)

    model_info = {
        "params": n_params,
        "gflops": gflops,
        "fps":    fps,
    }
    print(f"  Parameters     : {n_params / 1e6:.2f} M")
    print(f"  GFLOPs         : {gflops}")
    print(f"  FPS (GPU fwd)  : {fps}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    print(colorstr("bold", "\n[3/5] Building dataloader..."))
    data_dict = check_dataset(str(REPO_ROOT / args.data))
    val_path  = data_dict["val"]

    val_loader, dataset = create_dataloader(
        val_path,
        imgsz=imgsz,
        batch_size=args.batch,
        stride=stride,
        single_cls=False,
        pad=0.5,
        rect=True,
        workers=args.workers,
        prefix=colorstr("val: "),
    )
    print(f"  Images         : {len(dataset)}")

    # ── Evaluate ────────────────────────────────────────────────────────────
    print(colorstr("bold", "\n[4/5] Running evaluation..."))
    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        names=names,
        nc=nc,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=device,
        save_dir=save_dir,
        half=half,
        verbose=True,
    )

    # ── Save outputs ─────────────────────────────────────────────────────────
    print(colorstr("bold", "\n[5/5] Saving outputs..."))

    table = write_text_table(metrics, ckpt_info, model_info, save_dir)
    print("\n" + table)

    write_latex_table(metrics, model_info, save_dir)
    write_per_class_txt(metrics, save_dir)
    plot_per_class_ap(metrics, save_dir)
    plot_training_curves(REPO_ROOT / args.results_csv, save_dir)

    # Full JSON dump
    full_results = {
        "checkpoint":  str(REPO_ROOT / args.weights),
        "ckpt_info":   ckpt_info,
        "model_info":  {"params_M": round(n_params / 1e6, 3), "gflops": gflops, "fps": fps},
        "eval_config": {"imgsz": imgsz, "conf": args.conf, "iou": args.iou, "half": half},
        "metrics":     metrics,
    }
    (save_dir / "raw_results.json").write_text(json.dumps(full_results, indent=2))

    print(f"\n{'='*60}")
    print(f"  Results saved to: {save_dir}")
    print(f"  {'File':<35s} Description")
    print(f"  {'-'*55}")
    for fname, desc in [
        ("metrics_table.txt",    "Human-readable summary table"),
        ("metrics_table.tex",    "LaTeX table for paper"),
        ("per_class_ap.txt",     "Per-class AP numbers"),
        ("PR_curve.png",         "Precision-Recall curve"),
        ("F1_curve.png",         "F1-confidence curve"),
        ("per_class_ap_bar.png", "Per-class AP bar chart"),
        ("training_curves.png",  "Loss + mAP training curves"),
        ("raw_results.json",     "Machine-readable JSON"),
    ]:
        print(f"  {fname:<35s} {desc}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
