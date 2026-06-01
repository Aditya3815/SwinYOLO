#!/usr/bin/env python3
"""
utils/dota_hbb_converter.py — DOTA 1.5 OBB → YOLO HBB Converter
=================================================================
Converts pre-tiled DOTA 1.5 data (images + OBB labels) into YOLO
axis-aligned bounding box (HBB) format ready for DA-YOLO training.

INPUT  — Mixed directories containing both .png images and .txt OBB labels:
    label format: class_id  x1 y1  x2 y2  x3 y3  x4 y4   (normalized, 4 corners)

OUTPUT — Standard YOLO structure:
    images/
      train/  *.png
      val/    *.png
    labels/
      train/  *.txt   (class_id  cx  cy  w  h  — normalised HBB)
      val/    *.txt

Conversion: OBB quad → HBB
    cx = mean(x1..x4)  →  Not used — we take axis-aligned box instead:
    xmin = min(x1..x4),  xmax = max(x1..x4)
    ymin = min(y1..y4),  ymax = max(y1..y4)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w  = xmax - xmin
    h  = ymax - ymin
    (Values are already normalised 0–1, so no further scaling needed.)

DOTA 1.5 classes (16 total):
    0: plane           8: bridge
    1: ship            9: large-vehicle
    2: storage-tank   10: small-vehicle
    3: baseball-diamond 11: helicopter
    4: tennis-court   12: roundabout
    5: basketball-court 13: soccer-ball-field
    6: ground-track-field 14: swimming-pool
    7: harbor         15: container-crane

Usage:
    python utils/dota_hbb_converter.py \\
        --train-zip  DOTA_dataset/lebel/train/train.zip \\
        --val-zip    DOTA_dataset/lebel/val/val_label.zip \\
        --out        /data/DOTA/yolo

    Or from already-extracted directories:
    python utils/dota_hbb_converter.py \\
        --train-dir  /data/DOTA/raw/train \\
        --val-dir    /data/DOTA/raw/val_label \\
        --out        /data/DOTA/yolo
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# DOTA 1.5 class mapping  (16 classes)
# ---------------------------------------------------------------------------

_CLASS_NAMES: list[str] = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
    "container-crane",
]
NC = len(_CLASS_NAMES)   # 16


# ---------------------------------------------------------------------------
# OBB → HBB conversion
# ---------------------------------------------------------------------------

def obb_line_to_hbb(line: str) -> str | None:
    """
    Parse one OBB label line and return YOLO HBB string, or None on error.
    Input:  class_id  x1 y1  x2 y2  x3 y3  x4 y4
    Output: class_id  cx cy  w  h
    """
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    try:
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
    except ValueError:
        return None

    if cls_id < 0 or cls_id >= NC:
        return None

    xs = coords[0::2]   # x1, x2, x3, x4
    ys = coords[1::2]   # y1, y2, y3, y4

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Clamp to [0, 1]
    xmin = max(0.0, min(1.0, xmin))
    ymin = max(0.0, min(1.0, ymin))
    xmax = max(0.0, min(1.0, xmax))
    ymax = max(0.0, min(1.0, ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w  = xmax - xmin
    h  = ymax - ymin

    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_label_file(src_txt: str, dst_path: Path) -> int:
    """Convert OBB label text to HBB. Returns number of boxes written."""
    lines = [obb_line_to_hbb(ln) for ln in src_txt.splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln is not None]
    dst_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


# ---------------------------------------------------------------------------
# Extraction from zip
# ---------------------------------------------------------------------------

def process_zip(zip_path: Path, split: str, out_root: Path,
                verbose: bool) -> tuple[int, int]:
    """
    Extract a mixed-content DOTA zip (images + OBB labels) into YOLO structure.
    Returns (n_images, n_boxes).
    """
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = 0

    print(f"\n  Processing {split} from: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Build stem → members map
        img_names = {Path(n).stem: n for n in names
                     if n.lower().endswith(".png")}
        txt_names = {Path(n).stem: n for n in names
                     if n.lower().endswith(".txt")}

        stems = set(img_names) | set(txt_names)
        print(f"    Found {len(img_names)} images, {len(txt_names)} label files")

        for i, stem in enumerate(sorted(stems), 1):
            if verbose and i % 500 == 0:
                print(f"    ... {i}/{len(stems)}")

            # Extract / copy image
            if stem in img_names:
                dst_img = img_dir / f"{stem}.png"
                if not dst_img.exists():
                    data = zf.read(img_names[stem])
                    dst_img.write_bytes(data)
                n_images += 1

            # Convert label
            if stem in txt_names:
                dst_lbl = lbl_dir / f"{stem}.txt"
                txt_content = zf.read(txt_names[stem]).decode("utf-8", errors="replace")
                n_boxes += convert_label_file(txt_content, dst_lbl)
            else:
                # Image without label → empty label file
                (lbl_dir / f"{stem}.txt").write_text("")

    print(f"    [{split}]  {n_images} images  {n_boxes} boxes")
    return n_images, n_boxes


def process_dir(src_dir: Path, split: str, out_root: Path,
                verbose: bool) -> tuple[int, int]:
    """
    Process an already-extracted mixed directory (images + OBB labels).
    Returns (n_images, n_boxes).
    """
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = 0

    imgs = sorted(src_dir.glob("*.png"))
    txts = {p.stem: p for p in src_dir.glob("*.txt")}

    print(f"\n  Processing {split} from dir: {src_dir}")
    print(f"    Found {len(imgs)} images, {len(txts)} label files")

    for i, img_path in enumerate(imgs, 1):
        if verbose and i % 500 == 0:
            print(f"    ... {i}/{len(imgs)}")

        stem = img_path.stem
        dst_img = img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        n_images += 1

        dst_lbl = lbl_dir / f"{stem}.txt"
        if stem in txts:
            txt_content = txts[stem].read_text(encoding="utf-8", errors="replace")
            n_boxes += convert_label_file(txt_content, dst_lbl)
        else:
            dst_lbl.write_text("")

    print(f"    [{split}]  {n_images} images  {n_boxes} boxes")
    return n_images, n_boxes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert pre-tiled DOTA 1.5 OBB labels to YOLO HBB format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Source: zip files (preferred)
    p.add_argument("--train-zip", type=str, default=None,
                   help="Path to train.zip (mixed images+OBB labels)")
    p.add_argument("--val-zip", type=str, default=None,
                   help="Path to val_label.zip (mixed images+OBB labels)")
    # Source: pre-extracted directories (alternative to zips)
    p.add_argument("--train-dir", type=str, default=None,
                   help="Extracted train directory (alternative to --train-zip)")
    p.add_argument("--val-dir", type=str, default=None,
                   help="Extracted val directory (alternative to --val-zip)")
    # Output
    p.add_argument("--out", required=True, type=str,
                   help="Output root for YOLO dataset (images/ and labels/ created here)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-batch progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)

    print("=" * 68)
    print("  DOTA 1.5 OBB → YOLO HBB Converter")
    print("=" * 68)
    print(f"  Output  : {out}")
    print(f"  Classes : {NC}")
    for i, name in enumerate(_CLASS_NAMES):
        print(f"    {i:2d}: {name}")
    print("=" * 68)

    total_images = total_boxes = 0

    # --- Train ---
    if args.train_zip:
        n_i, n_b = process_zip(Path(args.train_zip), "train", out,
                               verbose=not args.quiet)
    elif args.train_dir:
        n_i, n_b = process_dir(Path(args.train_dir), "train", out,
                               verbose=not args.quiet)
    else:
        print("\n  [SKIP] No train source provided (--train-zip or --train-dir)")
        n_i = n_b = 0
    total_images += n_i
    total_boxes  += n_b

    # --- Val ---
    if args.val_zip:
        n_i, n_b = process_zip(Path(args.val_zip), "val", out,
                               verbose=not args.quiet)
    elif args.val_dir:
        n_i, n_b = process_dir(Path(args.val_dir), "val", out,
                               verbose=not args.quiet)
    else:
        print("\n  [SKIP] No val source provided (--val-zip or --val-dir)")
        n_i = n_b = 0
    total_images += n_i
    total_boxes  += n_b

    print("\n" + "=" * 68)
    print(f"  Done — {total_images} images / {total_boxes} boxes → {out}")
    print("=" * 68)
    print("\n  Next steps:")
    print(f"  1. Update data/dota.yaml:  path: {out}")
    print("  2. Re-cluster anchors:")
    print(f"     python utils/ciou_kmeans.py \\")
    print(f"         --label-dir {out}/labels/train \\")
    print(f"         --img-size 1024 --n-clusters 12")
    print("  3. Paste anchors into models/da_yolo.yaml, then train:")
    print("     python train_da_yolo.py --data data/dota.yaml \\")
    print("         --mode scratch --img-size 1024 --batch-size 4 \\")
    print("         --name dota_scratch")


if __name__ == "__main__":
    main()
