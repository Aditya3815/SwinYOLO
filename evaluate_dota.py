#!/usr/bin/env python3
"""
evaluate_dota.py — DA-YOLO Comprehensive Evaluation on DOTA 1.5
================================================================
Computes paper-ready metrics on the DOTA 1.5 validation split (3360 images, 16 classes):
  • mAP@0.5 and mAP@0.5:0.95 (COCO-style)
  • Per-class AP@0.5 (all 16 DOTA 1.5 categories)
  • Precision, Recall, F1 at optimal confidence threshold
  • Model complexity: parameters and GFLOPs
  • Inference speed: FPS on GPU

Outputs (saved to --save-dir):
  eval_results/dota_v1/
  ├── metrics_table.txt       — human-readable summary
  ├── metrics_table.tex       — LaTeX table ready for paper
  ├── per_class_ap.txt        — per-class AP with percentages
  ├── PR_curve.png            — precision-recall curve
  ├── F1_curve.png            — F1-confidence curve
  ├── P_curve.png             — precision-confidence curve
  ├── R_curve.png             — recall-confidence curve
  ├── per_class_ap_bar.png    — per-class AP bar chart
  ├── training_curves.png     — loss + mAP curves from results.csv
  └── raw_results.json        — machine-readable JSON

Usage:
    python evaluate_dota.py

    # Override defaults:
    python evaluate_dota.py \\
        --weights runs/da_yolo/dota_scratch2/weights/best.pth \\
        --data    data/dota.yaml \\
        --imgsz   1024 \\
        --batch   4 \\
        --save-dir eval_results/dota_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# DOTA 1.5 — 16 classes (in class-ID order)
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond",
    "tennis-court", "basketball-court", "ground-track-field", "harbor",
    "bridge", "large-vehicle", "small-vehicle", "helicopter",
    "roundabout", "soccer-ball-field", "swimming-pool", "container-crane",
]


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def load_checkpoint(weights: Path, device: torch.device) -> tuple[DetectionModel, dict]:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model: DetectionModel = (ckpt.get("ema") or ckpt["model"]).float().eval().to(device)
    info = {
        "epoch":        ckpt.get("epoch", -1),
        "best_fitness": float(np.asarray(ckpt["best_fitness"]).flat[0]) if "best_fitness" in ckpt else None,
    }
    return model, info


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_gflops(model: DetectionModel, imgsz: int, device: torch.device) -> float:
    try:
        from thop import profile as thop_profile
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        flops, _ = thop_profile(de_parallel(model), inputs=(dummy,), verbose=False)
        return round(flops / 1e9, 2)
    except Exception:
        return float("nan")


def measure_fps(model: DetectionModel, imgsz: int, device: torch.device,
                warmup_reps: int = 10, reps: int = 100) -> float:
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup_reps):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return round(reps / (time.perf_counter() - t0), 1)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, names, nc, conf_thres, iou_thres,
             device, save_dir, half=False) -> dict:
    model.half() if half else model.float()

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    confusion = ConfusionMatrix(nc=nc)

    stats      = []
    seen       = 0
    total_time = 0.0

    pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    for imgs, targets, paths, shapes in pbar:
        imgs    = imgs.to(device, non_blocking=True)
        imgs    = (imgs.half() if half else imgs.float()) / 255.0
        targets = targets.to(device)
        nb, _, h, w = imgs.shape

        t0 = time.perf_counter()
        out, _ = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        out = non_max_suppression(
            out, conf_thres=conf_thres, iou_thres=iou_thres,
            labels=[], multi_label=True, agnostic=False, max_det=500,
        )

        for si, pred in enumerate(out):
            labels  = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen   += 1

            if npr == 0:
                if nl:
                    stats.append((correct, torch.zeros(0, device=device),
                                  torch.zeros(0, device=device), labels[:, 0]))
                continue

            predn = pred.clone()
            scale_boxes(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                img_h, img_w = imgs[si].shape[1], imgs[si].shape[2]
                tbox[:, [0, 2]] *= img_w
                tbox[:, [1, 3]] *= img_h
                scale_boxes(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = _match_predictions(predn[:, :4], predn[:, 5], labelsn, iouv)
                confusion.process_batch(predn, labelsn)

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels[:, 0].cpu()))

    stats_cat = [torch.cat(x, 0).numpy() for x in zip(*stats)]
    metrics   = {}

    if len(stats_cat) and stats_cat[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats_cat, plot=True, save_dir=save_dir, names=names,
        )
        ap50, ap  = ap[:, 0], ap.mean(1)
        mp, mr    = p.mean(), r.mean()
        mf1       = f1.mean()
        map50     = ap50.mean()
        map5095   = ap.mean()
    else:
        tp = fp = p = r = f1 = ap50 = ap = torch.zeros(1)
        mp = mr = mf1 = map50 = map5095 = 0.0
        ap_class = []

    metrics.update({
        "precision":  float(mp),
        "recall":     float(mr),
        "f1":         float(mf1),
        "mAP50":      float(map50),
        "mAP5095":    float(map5095),
        "per_class": {
            names[int(c)]: {"AP50": float(ap50[i]), "AP5095": float(ap[i])}
            for i, c in enumerate(ap_class)
        },
        "num_images": seen,
        "infer_ms":   round(total_time / max(seen, 1) * 1000, 2),
    })
    return metrics


def _match_predictions(pboxes, pcls, labels, iouv):
    correct = torch.zeros(pboxes.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], pboxes)
    correct_class = labels[:, 0:1] == pcls
    for i, thr in enumerate(iouv):
        x = torch.where((iou >= thr) & correct_class)
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

def write_text_table(metrics: dict, ckpt_info: dict, model_info: dict,
                     save_dir: Path) -> str:
    w = 72
    lines = [
        "=" * w,
        "  DA-YOLO — DOTA 1.5 Evaluation",
        "=" * w,
        f"  Dataset            : DOTA 1.5 val split (16 classes, 1024×1024)",
        f"  Checkpoint epoch   : {ckpt_info['epoch']}",
        f"  Parameters         : {model_info['params'] / 1e6:.2f} M",
        f"  GFLOPs             : {model_info['gflops']}",
        f"  Images evaluated   : {metrics['num_images']}",
        "-" * w,
        f"  {'Metric':<35s} {'Value':>10s}",
        "-" * w,
        f"  {'Precision (P)':.<35s} {metrics['precision']:>10.4f}",
        f"  {'Recall (R)':.<35s} {metrics['recall']:>10.4f}",
        f"  {'F1 Score':.<35s} {metrics['f1']:>10.4f}",
        f"  {'mAP @ IoU=0.50':.<35s} {metrics['mAP50']:>10.4f}",
        f"  {'mAP @ IoU=0.50:0.95':.<35s} {metrics['mAP5095']:>10.4f}",
        f"  {'Inference speed (ms/img)':.<35s} {metrics['infer_ms']:>10.2f}",
        f"  {'FPS (GPU forward only)':.<35s} {model_info['fps']:>10.1f}",
        "-" * w,
        "  Per-class AP @ IoU=0.50:",
        "-" * w,
        f"  {'Class':<30s} {'AP@0.5':>10s} {'AP@.5:.95':>12s}",
        "-" * w,
    ]
    for cls_name, vals in metrics["per_class"].items():
        lines.append(
            f"  {cls_name:<30s} {vals['AP50']:>10.4f} {vals['AP5095']:>12.4f}"
        )
    lines.append("-" * w)
    lines.append(f"  {'mAP (mean)':.<30s} {metrics['mAP50']:>10.4f} {metrics['mAP5095']:>12.4f}")
    lines.append("=" * w)
    table = "\n".join(lines)
    (save_dir / "metrics_table.txt").write_text(table + "\n")
    return table


def write_latex_table(metrics: dict, save_dir: Path) -> None:
    cls_rows = "\n".join(
        f"        {name.replace('-', r'\text{-}')} & {v['AP50']*100:.1f} & {v['AP5095']*100:.1f} \\\\"
        for name, v in metrics["per_class"].items()
    )
    latex = rf"""% DA-YOLO DOTA 1.5 Results — auto-generated by evaluate_dota.py
\begin{{table}}[t]
\centering
\caption{{DA-YOLO performance on the DOTA 1.5 validation split (3360 images, 16 classes).
  P: precision, R: recall, mAP$_{{50}}$: AP at IoU=0.50,
  mAP$_{{50:95}}$: COCO-style AP averaged over IoU$\in[0.50,0.95]$.}}
\label{{tab:dota_results}}
\setlength{{\tabcolsep}}{{6pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Method & P & R & mAP$_{{50}}$ (\%) & mAP$_{{50:95}}$ (\%) \\
\midrule
DA-YOLO (ours) & {metrics['precision']*100:.1f} & {metrics['recall']*100:.1f} & {metrics['mAP50']*100:.1f} & {metrics['mAP5095']*100:.1f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

% ---- Per-class results ----
\begin{{table}}[t]
\centering
\caption{{Per-class AP on DOTA 1.5 val (DA-YOLO, HBB).}}
\label{{tab:dota_per_class}}
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
    lines = [f"{'Class':<32s} {'AP@0.5 (%)':>12s} {'AP@.5:.95 (%)':>15s}"]
    lines.append("-" * 62)
    for name, v in metrics["per_class"].items():
        lines.append(f"{name:<32s} {v['AP50']*100:>12.2f} {v['AP5095']*100:>15.2f}")
    lines.append("-" * 62)
    lines.append(f"{'mAP':<32s} {metrics['mAP50']*100:>12.2f} {metrics['mAP5095']*100:>15.2f}")
    (save_dir / "per_class_ap.txt").write_text("\n".join(lines) + "\n")


def plot_per_class_ap(metrics: dict, save_dir: Path) -> None:
    classes = list(metrics["per_class"].keys())
    ap50   = [v["AP50"]   * 100 for v in metrics["per_class"].values()]
    ap5095 = [v["AP5095"] * 100 for v in metrics["per_class"].values()]

    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(16, 5))
    b1 = ax.bar(x - w/2, ap50,   w, label="AP@0.5",      color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + w/2, ap5095, w, label="AP@0.5:0.95", color="#DD8452", alpha=0.85)

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Average Precision (%)", fontsize=11)
    ax.set_title("DA-YOLO — Per-Class AP on DOTA 1.5 Val (16 classes)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)

    ax.axhline(np.mean(ap50),   color="#4C72B0", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(np.mean(ap5095), color="#DD8452", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.tight_layout()
    fig.savefig(save_dir / "per_class_ap_bar.png", dpi=150)
    plt.close(fig)


def plot_training_curves(results_csv: Path, save_dir: Path) -> None:
    if not results_csv.exists():
        return
    import csv
    rows = list(csv.reader(open(results_csv)))
    headers = [h.strip() for h in rows[0]]
    data = {h: [float(r[i]) for r in rows[1:]] for i, h in enumerate(headers)}

    epochs = np.arange(1, len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    for col, label, color in [
        ("train/box_loss", "Box loss", "#4C72B0"),
        ("train/obj_loss", "Obj loss", "#DD8452"),
        ("train/cls_loss", "Cls loss", "#55A868"),
    ]:
        if col in data:
            ax.plot(epochs, data[col], label=label, color=color)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training Losses")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for col, label, color in [
        ("metrics/mAP_0.5",      "mAP@0.5",      "#4C72B0"),
        ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95", "#DD8452"),
    ]:
        if col in data:
            ax.plot(epochs, data[col], label=label, color=color)
    peak_epoch = int(np.argmax(data.get("metrics/mAP_0.5", [0]))) + 1
    peak_val   = max(data.get("metrics/mAP_0.5", [0]))
    ax.axvline(peak_epoch, color="gray", linestyle="--", alpha=0.5, label=f"best ep.{peak_epoch}")
    ax.set(xlabel="Epoch", ylabel="mAP", title=f"Val mAP — peak {peak_val*100:.2f}% @ ep.{peak_epoch}")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("DA-YOLO Training on DOTA 1.5", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DA-YOLO comprehensive evaluation on DOTA 1.5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",  default="runs/da_yolo/dota_scratch2/weights/best.pth")
    p.add_argument("--data",     default="data/dota.yaml")
    p.add_argument("--imgsz",    type=int,   default=1024)
    p.add_argument("--batch",    type=int,   default=4)
    p.add_argument("--conf",     type=float, default=0.001)
    p.add_argument("--iou",      type=float, default=0.65)
    p.add_argument("--device",   default="")
    p.add_argument("--workers",  type=int,   default=4)
    p.add_argument("--half",     action="store_true")
    p.add_argument("--save-dir", default="eval_results/dota_v1")
    p.add_argument("--results-csv",
                   default="runs/da_yolo/dota_scratch2/results.csv")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    save_dir = REPO_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    half   = args.half and device.type != "cpu"

    print(colorstr("bold", "\n[1/5] Loading DOTA 1.5 checkpoint..."))
    model, ckpt_info = load_checkpoint(REPO_ROOT / args.weights, device)
    stride = int(model.stride.max())
    imgsz  = check_img_size(args.imgsz, s=stride)
    names  = model.names
    nc     = model.nc
    print(f"  Epoch   : {ckpt_info['epoch']}")
    print(f"  Classes : {nc}  →  {list(names.values())}")

    print(colorstr("bold", "\n[2/5] Model complexity & speed..."))
    n_params = count_parameters(model)
    gflops   = compute_gflops(model, imgsz, device)
    fps      = measure_fps(model, imgsz, device)
    model_info = {"params": n_params, "gflops": gflops, "fps": fps}
    print(f"  Params  : {n_params/1e6:.2f} M")
    print(f"  GFLOPs  : {gflops}")
    print(f"  FPS     : {fps}")

    print(colorstr("bold", "\n[3/5] Building dataloader (DOTA 1.5 val, 3360 images)..."))
    data_dict = check_dataset(str(REPO_ROOT / args.data))
    val_path  = data_dict["val"]
    val_loader, dataset = create_dataloader(
        val_path, imgsz=imgsz, batch_size=args.batch, stride=stride,
        single_cls=False, pad=0.5, rect=True, workers=args.workers,
        prefix=colorstr("val: "),
    )
    print(f"  Images  : {len(dataset)}")

    print(colorstr("bold", "\n[4/5] Running evaluation..."))
    metrics = evaluate(
        model=model, dataloader=val_loader, names=names, nc=nc,
        conf_thres=args.conf, iou_thres=args.iou, device=device,
        save_dir=save_dir, half=half,
    )

    print(colorstr("bold", "\n[5/5] Saving outputs..."))
    table = write_text_table(metrics, ckpt_info, model_info, save_dir)
    print("\n" + table)

    write_latex_table(metrics, save_dir)
    write_per_class_txt(metrics, save_dir)
    plot_per_class_ap(metrics, save_dir)
    plot_training_curves(REPO_ROOT / args.results_csv, save_dir)

    full = {
        "checkpoint":  str(REPO_ROOT / args.weights),
        "ckpt_info":   ckpt_info,
        "model_info":  {"params_M": round(n_params/1e6, 3), "gflops": gflops, "fps": fps},
        "eval_config": {"imgsz": imgsz, "conf": args.conf, "iou": args.iou, "half": half},
        "metrics":     metrics,
    }
    (save_dir / "raw_results.json").write_text(json.dumps(full, indent=2))

    print(f"\n{'='*60}")
    print(f"  Results saved to: {save_dir}")
    print(f"  {'File':<35s} Description")
    print(f"  {'-'*55}")
    for fname, desc in [
        ("metrics_table.txt",    "Human-readable summary"),
        ("metrics_table.tex",    "LaTeX table for paper"),
        ("per_class_ap.txt",     "Per-class AP (16 classes)"),
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
