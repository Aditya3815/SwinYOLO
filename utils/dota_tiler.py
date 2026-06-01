#!/usr/bin/env python3
"""
utils/dota_tiler.py — DOTA v1.0 HBB → Tiled YOLO Format Converter
===================================================================
DOTA images are up to 4000×4000 pixels — too large for direct training.
This script tiles each image into overlapping patches and re-annotates
bounding boxes relative to each patch.

DOTA v1.0 HBB annotation format (one file per image, space-separated):
    x1 y1 x2 y2 x3 y3 x4 y4 category difficult
    (quad polygon corners, axis-aligned box from min/max of corners)

YOLO output format (per line):
    class_id  cx_norm  cy_norm  w_norm  h_norm

Category mapping (15 classes):
    plane, ship, storage-tank, baseball-diamond, tennis-court,
    basketball-court, ground-track-field, harbor, bridge, large-vehicle,
    small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool

Usage:
    python utils/dota_tiler.py \\
        --root  /data/DOTA \\
        --out   /data/DOTA/yolo \\
        --size  1024 \\
        --stride 824

    Expected input tree under --root:
        train/
            images/    *.png  (or *.jpg)
            labelTxt/  *.txt
        val/
            images/
            labelTxt/
        test/
            images/     (no labels for test)

    Output tree under --out:
        images/
            train/  *.png  (tile patches)
            val/    *.png
            test/   *.png
        labels/
            train/  *.txt  (YOLO labels)
            val/    *.txt
            test/   *.txt  (empty)

    Then update data/dota.yaml: set  path: <--out path>

Notes:
    • Boxes entirely outside a tile are dropped.
    • Boxes partially overlapping a tile are clipped to the tile boundary.
    • Boxes with clipped area < min_visibility fraction of original are dropped
      (default 0.30 — keeps boxes that are at least 30% visible in the patch).
    • Tiles with zero annotations are kept (negative tiles help reduce FP rate).
    • For overlap: stride = size - overlap_px.
      E.g. size=1024, stride=824 gives 200px overlap on each edge.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

# DOTA v1.0 category → YOLO class index
_DOTA_CATEGORIES: dict[str, int] = {
    "plane":               0,
    "ship":                1,
    "storage-tank":        2,
    "baseball-diamond":    3,
    "tennis-court":        4,
    "basketball-court":    5,
    "ground-track-field":  6,
    "harbor":              7,
    "bridge":              8,
    "large-vehicle":       9,
    "small-vehicle":      10,
    "helicopter":         11,
    "roundabout":         12,
    "soccer-ball-field":  13,
    "swimming-pool":      14,
}

# Normalised names that appear in DOTA annotation files (lowercase, stripped)
_DOTA_ALIASES: dict[str, str] = {
    "plane":                "plane",
    "ship":                 "ship",
    "storage-tank":         "storage-tank",
    "storagetank":          "storage-tank",
    "baseball-diamond":     "baseball-diamond",
    "baseballdiamond":      "baseball-diamond",
    "tennis-court":         "tennis-court",
    "tenniscourt":          "tennis-court",
    "basketball-court":     "basketball-court",
    "basketballcourt":      "basketball-court",
    "ground-track-field":   "ground-track-field",
    "groundtrackfield":     "ground-track-field",
    "harbor":               "harbor",
    "bridge":               "bridge",
    "large-vehicle":        "large-vehicle",
    "largevehicle":         "large-vehicle",
    "small-vehicle":        "small-vehicle",
    "smallvehicle":         "small-vehicle",
    "helicopter":           "helicopter",
    "roundabout":           "roundabout",
    "soccer-ball-field":    "soccer-ball-field",
    "soccerballfield":      "soccer-ball-field",
    "swimming-pool":        "swimming-pool",
    "swimmingpool":         "swimming-pool",
}


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_dota_annotation(ann_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Parse a DOTA v1.0 HBB annotation file.
    Returns list of (class_id, x0, y0, x1, y1) in absolute pixel coords.
    """
    if not ann_path.exists():
        return []

    results: list[tuple[int, float, float, float, float]] = []

    for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("imagesource") or line.startswith("gsd"):
            continue

        parts = line.split()
        if len(parts) < 9:
            continue

        try:
            coords = [float(p) for p in parts[:8]]
        except ValueError:
            continue

        xs = coords[0::2]   # [x1, x2, x3, x4]
        ys = coords[1::2]   # [y1, y2, y3, y4]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)

        cat_raw  = parts[8].strip().lower().replace(" ", "-")
        cat_norm = _DOTA_ALIASES.get(cat_raw, None)
        if cat_norm is None:
            continue
        cls_id = _DOTA_CATEGORIES[cat_norm]

        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue

        results.append((cls_id, x0, y0, x1, y1))

    return results


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def compute_tiles(
    img_h: int, img_w: int, tile_size: int, stride: int
) -> list[tuple[int, int, int, int]]:
    """
    Return list of (x0, y0, x1, y1) tile coords covering the full image.
    The last tile in each row/col is snapped to the image boundary.
    """
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


def clip_boxes_to_tile(
    boxes:          list[tuple[int, float, float, float, float]],
    tx0: int, ty0: int, tx1: int, ty1: int,
    min_visibility: float = 0.30,
) -> list[str]:
    """
    Clip absolute-pixel boxes to a tile window.
    Returns YOLO-format label strings (class_id cx cy w h, normalised to tile).

    A box is kept if its intersection area with the tile is >= min_visibility
    fraction of its original area.
    """
    tile_w = tx1 - tx0
    tile_h = ty1 - ty0
    yolo_lines: list[str] = []

    for cls_id, ax0, ay0, ax1, ay1 in boxes:
        orig_area = (ax1 - ax0) * (ay1 - ay0)
        if orig_area <= 0:
            continue

        # Intersect box with tile window
        ix0 = max(ax0, tx0) - tx0
        iy0 = max(ay0, ty0) - ty0
        ix1 = min(ax1, tx1) - tx0
        iy1 = min(ay1, ty1) - ty0

        if ix1 <= ix0 or iy1 <= iy0:
            continue   # box entirely outside tile

        inter_area = (ix1 - ix0) * (iy1 - iy0)
        if inter_area / orig_area < min_visibility:
            continue   # too little of the box is inside the tile

        # Clamp to tile bounds
        ix0 = max(0.0, ix0);  iy0 = max(0.0, iy0)
        ix1 = min(float(tile_w), ix1);  iy1 = min(float(tile_h), iy1)

        if ix1 <= ix0 or iy1 <= iy0:
            continue

        cx = (ix0 + ix1) / 2 / tile_w
        cy = (iy0 + iy1) / 2 / tile_h
        w  = (ix1 - ix0) / tile_w
        h  = (iy1 - iy0) / tile_h

        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def tile_split(
    src_root:       Path,
    split_name:     str,
    out_root:       Path,
    tile_size:      int,
    stride:         int,
    min_visibility: float,
    verbose:        bool,
) -> dict[str, int]:
    """Process one DOTA split. Returns stats dict."""
    img_dir = src_root / "images"
    ann_dir = src_root / "labelTxt"

    out_img_dir = out_root / "images" / split_name
    out_lbl_dir = out_root / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in img_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})

    if not img_paths:
        print(f"  [WARN] No images found in {img_dir} — skipping '{split_name}'")
        return {"images": 0, "tiles": 0, "boxes": 0}

    total_tiles = 0
    total_boxes = 0

    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            if verbose:
                print(f"  [WARN] Cannot read {img_path.name} — skipping")
            continue

        img_h, img_w = img.shape[:2]

        ann_path = ann_dir / img_path.with_suffix(".txt").name
        boxes    = parse_dota_annotation(ann_path)

        tiles = compute_tiles(img_h, img_w, tile_size, stride)

        for t_idx, (tx0, ty0, tx1, ty1) in enumerate(tiles):
            tile_img  = img[ty0:ty1, tx0:tx1]
            stem      = f"{img_path.stem}__{tx0}_{ty0}"
            out_img   = out_img_dir / f"{stem}{img_path.suffix}"
            out_lbl   = out_lbl_dir / f"{stem}.txt"

            # Pad non-square border tiles to tile_size × tile_size
            ph = tile_size - (ty1 - ty0)
            pw = tile_size - (tx1 - tx0)
            if ph > 0 or pw > 0:
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )

            # Write tile image (copy if full-size tile, write otherwise)
            if not out_img.exists():
                cv2.imwrite(str(out_img), tile_img)

            # Write YOLO labels for this tile
            yolo_lines = clip_boxes_to_tile(boxes, tx0, ty0, tx1, ty1, min_visibility)
            out_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            total_boxes += len(yolo_lines)

        total_tiles += len(tiles)

    if verbose:
        print(f"  [{split_name}]  {len(img_paths):5d} images → "
              f"{total_tiles:7d} tiles   {total_boxes:8d} boxes")

    return {"images": len(img_paths), "tiles": total_tiles, "boxes": total_boxes}


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

def write_dota_yaml(out_root: Path, splits_done: list[str]) -> None:
    """Write data/dota.yaml pointing to the tiled dataset."""
    nc    = len(_DOTA_CATEGORIES)
    names = {v: k for k, v in _DOTA_CATEGORIES.items()}

    lines = [
        "# DOTA v1.0 HBB — tiled dataset configuration for DA-YOLO",
        f"# Auto-generated by utils/dota_tiler.py",
        f"# Tile size: see --size / --stride used during tiling",
        "",
        f"path: {out_root}",
        "",
        "train: images/train",
        "val:   images/val",
    ]
    if "test" in splits_done:
        lines.append("test:  images/test")
    lines += [
        "",
        f"nc: {nc}",
        "",
        "names:",
    ]
    for idx in range(nc):
        lines.append(f"  {idx}: {names[idx]}")

    yaml_path = out_root / "dota.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Generated: {yaml_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tile DOTA v1.0 HBB images and convert annotations to YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",   required=True, type=str,
                   help="DOTA dataset root (contains train/, val/, test/ subdirs)")
    p.add_argument("--out",    required=True, type=str,
                   help="Output directory for tiled YOLO dataset")
    p.add_argument("--size",   type=int, default=1024,
                   help="Tile size in pixels (height and width)")
    p.add_argument("--stride", type=int, default=824,
                   help="Stride between tiles (overlap = size - stride pixels)")
    p.add_argument("--min-visibility", type=float, default=0.30,
                   help="Min fraction of a box's area that must fall inside a tile to keep it")
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   help="Splits to process (e.g. train val test)")
    p.add_argument("--quiet",  action="store_true",
                   help="Suppress per-image progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root     = Path(args.root)
    out_root = Path(args.out)
    overlap  = args.size - args.stride

    print("=" * 70)
    print("  DOTA v1.0 HBB Tiler")
    print("=" * 70)
    print(f"  Source     : {root}")
    print(f"  Output     : {out_root}")
    print(f"  Tile size  : {args.size}px")
    print(f"  Stride     : {args.stride}px  (overlap = {overlap}px per edge)")
    print(f"  Min vis.   : {args.min_visibility:.0%}")
    print(f"  Splits     : {args.splits}")
    print("=" * 70)

    splits_processed: list[str] = []
    total: dict[str, int] = {"images": 0, "tiles": 0, "boxes": 0}

    for split_name in args.splits:
        src_dir = root / split_name
        if not src_dir.exists():
            print(f"\n  [SKIP] Split '{split_name}' not found at {src_dir}")
            continue

        print(f"\n  Processing split: {split_name}")
        stats = tile_split(
            src_root       = src_dir,
            split_name     = split_name,
            out_root       = out_root,
            tile_size      = args.size,
            stride         = args.stride,
            min_visibility = args.min_visibility,
            verbose        = not args.quiet,
        )
        splits_processed.append(split_name)
        for k in total:
            total[k] += stats.get(k, 0)

    if not splits_processed:
        print("\n  No splits processed. Check --root path.")
        return

    write_dota_yaml(out_root, splits_processed)

    print("\n" + "=" * 70)
    print(f"  Done — {len(splits_processed)} split(s) written to: {out_root}")
    print(f"  Total original images : {total['images']}")
    print(f"  Total tiles generated : {total['tiles']}")
    print(f"  Total YOLO boxes      : {total['boxes']}")
    print("=" * 70)
    print("\n  Next steps:")
    print(f"  1. Copy or symlink the generated dota.yaml:")
    print(f"     cp {out_root}/dota.yaml data/dota.yaml")
    print(f"  2. Re-cluster anchors for DOTA tile size:")
    print(f"     python utils/ciou_kmeans.py \\")
    print(f"         --label-dir {out_root}/labels/train \\")
    print(f"         --img-size {args.size} \\")
    print(f"         --n-clusters 12")
    print(f"  3. Train:")
    print(f"     python train_da_yolo.py \\")
    print(f"         --data data/dota.yaml \\")
    print(f"         --mode scratch \\")
    print(f"         --img-size {args.size}")


if __name__ == "__main__":
    main()
