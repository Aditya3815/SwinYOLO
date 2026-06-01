#!/usr/bin/env python3
"""
train_da_yolo.py — DA-YOLO Primary Training Launcher
=====================================================
Wraps train.py with DA-YOLO best-practice defaults.
Supports fine-tuning (low lr) and from-scratch (high lr) modes via --mode flag.

Architecture:
  Backbone : YOLOv5s-CSP + DC3SWT (Multi-Scale Deformable Attention @ P5 and PANet levels)
  Neck     : BiFPN (4-level bidirectional FPN, DCNv2 @ P3, SE channel attention)
  Head     : CoordAttMulti (Coordinate Attention per FPN level)
             4-scale Detection Head (P2/P3/P4/P5) for small/tiny objects

Usage:
    # Fine-tune mode (pretrained weights, lr=1e-6)
    python train_da_yolo.py --data data/visdrone.yaml

    # From-scratch mode (lr=1e-4)
    python train_da_yolo.py --data data/visdrone.yaml --mode scratch

    # Custom epochs and batch size
    python train_da_yolo.py --data data/visdrone.yaml --epochs 200 --batch-size 8

    # Large image size (recommended for VisDrone / DOTA)
    python train_da_yolo.py --data data/visdrone.yaml --img-size 1280

    # Resume from checkpoint
    python train_da_yolo.py --data data/visdrone.yaml --resume runs/da_yolo/exp/weights/last.pt

Linux/tmux tip:
    tmux new -s da_yolo
    python train_da_yolo.py --data data/visdrone.yaml
"""

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Mode A — fine-tune (pretrained backbone / transfer learning)
# ---------------------------------------------------------------------------
_FINETUNE = dict(
    cfg             = "models/da_yolo.yaml",
    hyp             = "data/hyps/hyp.da-yolo.yaml",
    weights         = "",
    batch_size      = 4,
    imgsz           = 1024,
    epochs          = 100,
    optimizer       = "AdamW",
    scheduler       = "cosine",
    workers         = 8,
    project         = "runs/da_yolo",
    name            = "exp",
    label_smoothing = 0.1,
    patience        = 30,
    no_augs         = 10,
    seed            = 42,
)

# ---------------------------------------------------------------------------
# Mode B — from-scratch (higher lr, longer patience)
# ---------------------------------------------------------------------------
_SCRATCH = dict(
    cfg             = "models/da_yolo.yaml",
    hyp             = "data/hyps/hyp.visdrone.yaml",    # VisDrone-tuned: lr=1e-4, copy_paste=0.5
    weights         = "",
    batch_size      = 4,
    imgsz           = 1280,
    epochs          = 200,                               # +50 vs v1: rare classes need more examples
    optimizer       = "AdamW",
    scheduler       = "cosine",
    workers         = 8,
    project         = "runs/da_yolo",
    name            = "exp",
    label_smoothing = 0.0,
    patience        = 100,                               # match longer training budget
    no_augs         = 20,
    seed            = 42,
)


def parse_opt():
    parser = argparse.ArgumentParser(
        description="DA-YOLO training launcher — wraps train.py with best-practice defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       required=True, type=str,
                        help="Path to dataset yaml  (e.g. data/visdrone.yaml)")
    parser.add_argument("--mode",       type=str, default="finetune",
                        choices=["finetune", "scratch"],
                        help="'finetune' = lr=1e-6 (pretrained);  'scratch' = lr=1e-4")
    parser.add_argument("--cfg",        type=str,   default=None,
                        help="Override architecture yaml  (default: models/da_yolo.yaml)")
    parser.add_argument("--hyp",        type=str,   default=None,
                        help="Override hyperparameter yaml")
    parser.add_argument("--weights",    type=str,   default=None,
                        help="Pretrained weights path  ('' = train from scratch)")
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--img-size",   type=int,   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--optimizer",  type=str,   default=None)
    parser.add_argument("--scheduler",  type=str,   default=None,
                        choices=["cosine", "linear", "cyclic"])
    parser.add_argument("--workers",    type=int,   default=None)
    parser.add_argument("--project",    type=str,   default=None)
    parser.add_argument("--name",       type=str,   default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--patience",   type=int,   default=None)
    parser.add_argument("--no-augs",    type=int,   default=None,
                        help="Disable mosaic for last N epochs (close-mosaic)")
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--device",     type=str,   default="0",
                        help="CUDA device (e.g. 0, 0,1, or cpu)")
    parser.add_argument("--resume",     type=str,   default="",
                        help="Resume from checkpoint path")
    parser.add_argument("--exist-ok",   action="store_true",
                        help="Do not increment existing project/name directory")
    parser.add_argument("--no-cos-lr",  action="store_true",
                        help="Disable cosine LR annealing")
    return parser.parse_args()


def _resolve(opt, key, defaults):
    """Return CLI override if provided, else mode default."""
    cli_val = getattr(opt, key.replace("-", "_"), None)
    return cli_val if cli_val is not None else defaults[key]


def build_command(opt) -> list:
    d = _SCRATCH if opt.mode == "scratch" else _FINETUNE

    cfg             = opt.cfg        or d.get("cfg", "models/da_yolo.yaml")
    hyp             = opt.hyp        or d["hyp"]
    weights         = opt.weights    if opt.weights is not None else d["weights"]
    batch_size      = opt.batch_size or d["batch_size"]
    imgsz           = opt.img_size   or d["imgsz"]
    epochs          = opt.epochs     or d["epochs"]
    optimizer       = opt.optimizer  or d["optimizer"]
    scheduler       = opt.scheduler  or d["scheduler"]
    workers         = opt.workers    or d["workers"]
    project         = opt.project    or d["project"]
    name            = opt.name       or d["name"]
    label_smoothing = opt.label_smoothing if opt.label_smoothing is not None else d["label_smoothing"]
    patience        = opt.patience   or d["patience"]
    no_augs         = opt.no_augs    or d["no_augs"]
    seed            = opt.seed       or d["seed"]

    cmd = [sys.executable, "train.py"]
    cmd += ["--data",            opt.data]
    cmd += ["--cfg",             cfg]
    cmd += ["--hyp",             hyp]
    cmd += ["--batch-size",      str(batch_size)]
    cmd += ["--img",             str(imgsz)]
    cmd += ["--epochs",          str(epochs)]
    cmd += ["--optimizer",       optimizer]
    cmd += ["--scheduler",       scheduler]
    cmd += ["--workers",         str(workers)]
    cmd += ["--project",         project]
    cmd += ["--name",            name]
    cmd += ["--label-smoothing", str(label_smoothing)]
    cmd += ["--patience",        str(patience)]
    cmd += ["--no-augs",         str(no_augs)]
    cmd += ["--seed",            str(seed)]
    cmd += ["--device",          opt.device]
    cmd += ["--weights",         weights]
    if not opt.no_cos_lr:
        cmd += ["--cos-lr"]
    if opt.resume:
        cmd += ["--resume", opt.resume]
    if opt.exist_ok:
        cmd += ["--exist-ok"]
    return cmd, d


def main():
    opt = parse_opt()
    cmd, defaults = build_command(opt)

    lr_display = "1e-6 (fine-tune)" if opt.mode == "finetune" else "1e-4 (from-scratch)"

    print("=" * 70)
    print("DA-YOLO Training Launcher")
    print("=" * 70)
    print(f"  Mode        : {opt.mode}")
    print(f"  Data        : {opt.data}")
    print(f"  Architecture: {opt.cfg or defaults.get('cfg', 'models/da_yolo.yaml')}")
    print(f"  Epochs      : {opt.epochs or defaults['epochs']}  "
          f"(ValLoss ES patience={opt.patience or defaults['patience']})")
    print(f"  Batch size  : {opt.batch_size or defaults['batch_size']}")
    print(f"  Image size  : {opt.img_size or defaults['imgsz']}px")
    print(f"  Optimizer   : AdamW @ lr={lr_display}")
    print(f"  Scheduler   : {opt.scheduler or defaults['scheduler']} LR")
    print(f"  Device      : {opt.device}")
    print(f"  Close-mosaic: last {opt.no_augs or defaults['no_augs']} epochs")
    print()
    print("DA-YOLO best practices active:")
    print("  AdamW optimizer (required for Transformer blocks)")
    print("  Cosine LR annealing")
    print("  ValLoss early stopping (stable on dense datasets)")
    print("  WIoU geometric loss (down-weights easy anchors)")
    print("  SE channel attention in BiFPN nodes")
    print("  DC3SWT deformable attention (adaptive receptive field)")
    print("  NaN loss guard + gradient clip max_norm=1.0")
    print("=" * 70)
    print("Command:", " ".join(cmd))
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    project = opt.project or defaults["project"]
    name    = opt.name    or defaults["name"]
    log_dir = os.path.join(project, name)
    os.makedirs(log_dir, exist_ok=True)
    system_log = os.path.join(log_dir, "system_output.log")
    print(f"System output → {system_log}")
    print("=" * 70)

    kwargs = dict(check=False, stdout=open(system_log, "w"), stderr=subprocess.STDOUT)
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    result = subprocess.run(cmd, **kwargs)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
