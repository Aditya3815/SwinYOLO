#!/usr/bin/env python3
"""
utils/visdrone_converter.py — VisDrone2019-DET → YOLO format converter
======================================================================
Converts VisDrone annotation files (CSV, one per image) into YOLO-format
label files (.txt, one per image) and mirrors the directory structure
expected by data/visdrone.yaml.

VisDrone annotation format (per line):
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

YOLO format (per line):
    class_id  cx_norm  cy_norm  w_norm  h_norm

Category mapping (VisDrone category_id → YOLO class_id):
    0  ignored        → SKIP (not an object category)
    1  pedestrian     → 0
    2  people         → 1
    3  bicycle        → 2
    4  car            → 3
    5  van            → 4
    6  truck          → 5
    7  tricycle       → 6
    8  awning-tricycle→ 7
    9  bus            → 8
    10 motor          → 9
    11 others         → SKIP (ambiguous)

Usage:
    python utils/visdrone_converter.py \
        --root /data/VisDrone \
        --out  /data/VisDrone/yolo

    Expected input tree under --root:
        VisDrone2019-DET-train/
            images/        *.jpg
            annotations/   *.txt
        VisDrone2019-DET-val/
            images/
            annotations/
        VisDrone2019-DET-test-dev/
            images/
            annotations/   (may not exist for test-dev; images are copied only)

    Output tree under --out:
        images/
            train/   *.jpg
            val/     *.jpg
            test-dev/*.jpg
        labels/
            train/   *.txt
            val/     *.txt
            test-dev/*.txt   (empty files if no annotations)
"""

import argparse
import shutil
from pathlib import Path


# VisDrone category_id → YOLO class_id (-1 = skip)
_CAT2CLS = {
    0: -1,   # ignored
    1:  0,   # pedestrian
    2:  1,   # people
    3:  2,   # bicycle
    4:  3,   # car
    5:  4,   # van
    6:  5,   # truck
    7:  6,   # tricycle
    8:  7,   # awning-tricycle
    9:  8,   # bus
    10: 9,   # motor
    11: -1,  # others
}

_SPLITS = [
    ("VisDrone2019-DET-train",    "train"),
    ("VisDrone2019-DET-val",      "val"),
    ("VisDrone2019-DET-test-dev", "test-dev"),
]


def convert_annotation(ann_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert one VisDrone annotation file to a list of YOLO label lines."""
    lines = []
    for raw in ann_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) < 6:
            continue
        x1, y1, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        category = int(parts[5])
        cls = _CAT2CLS.get(category, -1)
        if cls < 0 or w <= 0 or h <= 0:
            continue
        # Clamp to image bounds before normalising
        x1 = max(0, x1)
        y1 = max(0, y1)
        w  = min(w,  img_w - x1)
        h  = min(h,  img_h - y1)
        cx = (x1 + w / 2) / img_w
        cy = (y1 + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
    return lines


def get_image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height) without loading the full image."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.width, im.height
    except Exception:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        return img.shape[1], img.shape[0]


def convert_split(src_root: Path, split_name: str, out_root: Path, verbose: bool = True):
    img_src = src_root / "images"
    ann_src = src_root / "annotations"
    img_dst = out_root / "images" / split_name
    lbl_dst = out_root / "labels" / split_name

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    images = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))
    if not images:
        print(f"  [WARN] No images found in {img_src} — skipping split '{split_name}'")
        return

    skipped = 0
    for img_path in images:
        # Copy image
        dst_img = img_dst / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Convert annotation
        ann_path = ann_src / img_path.with_suffix(".txt").name
        lbl_path = lbl_dst / img_path.with_suffix(".txt").name
        if ann_path.exists():
            try:
                w, h = get_image_size(img_path)
                yolo_lines = convert_annotation(ann_path, w, h)
                lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            except Exception as e:
                print(f"  [WARN] Failed to convert {ann_path.name}: {e}")
                lbl_path.write_text("")
                skipped += 1
        else:
            lbl_path.write_text("")  # test-dev has no annotations

    msg = f"  [{split_name}] {len(images)} images"
    if skipped:
        msg += f"  ({skipped} annotation errors written as empty)"
    if verbose:
        print(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone2019-DET annotations to YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, type=str,
                        help="Root directory containing VisDrone split folders")
    parser.add_argument("--out",  required=True, type=str,
                        help="Output directory for YOLO-format dataset")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-split progress output")
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.out)

    print(f"VisDrone → YOLO converter")
    print(f"  Source : {root}")
    print(f"  Output : {out}")
    print()

    converted = 0
    for src_name, split_name in _SPLITS:
        src_dir = root / src_name
        if not src_dir.exists():
            print(f"  [SKIP] {src_name} not found at {src_dir}")
            continue
        print(f"Converting split: {split_name}  ({src_dir.name})")
        convert_split(src_dir, split_name, out, verbose=not args.quiet)
        converted += 1

    print()
    if converted == 0:
        print("No splits converted. Check --root path.")
    else:
        print(f"Done. {converted} split(s) written to: {out}")
        print()
        print("Next steps:")
        print(f"  1. Update data/visdrone.yaml: set  path: {out}")
        print("  2. Re-cluster anchors for VisDrone object sizes:")
        print(f"     python utils/ciou_kmeans.py --label-dir {out}/labels/train \\")
        print("         --img-size 1280 --n-clusters 12")
        print("  3. Train:")
        print("     python train_da_yolo.py --data data/visdrone.yaml \\")
        print("         --hyp data/hyps/hyp.visdrone.yaml --img-size 1280 --mode scratch")


if __name__ == "__main__":
    main()
