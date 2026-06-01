#!/usr/bin/env python3
"""
utils/dior_converter.py — DIOR-R → YOLO HBB Format Converter
=============================================================
Converts DIOR-R dataset (PASCAL VOC XML annotations) to YOLO-format label files.

DIOR-R is a large-scale remote sensing object detection dataset with 20 categories,
23,463 images at 800×800 resolution, ~190,288 instances.
Annotations include both HBB (axis-aligned) and OBB (oriented boxes).
This converter uses the HBB <bndbox> elements — OBB support is future work.

DIOR-R annotation format (PASCAL VOC XML):
    <annotation>
      <filename>00001.jpg</filename>
      <size><width>800</width><height>800</height></size>
      <object>
        <name>airplane</name>
        <difficult>0</difficult>
        <bndbox>
          <xmin>100.0</xmin><ymin>200.0</ymin>
          <xmax>300.0</xmax><ymax>400.0</ymax>
        </bndbox>
      </object>
    </annotation>

YOLO output format:
    class_id  cx_norm  cy_norm  w_norm  h_norm

20 DIOR-R classes:
    airplane, airport, baseballfield, basketballcourt, bridge, chimney,
    dam, expressway-service-area, expressway-toll-station, golffield,
    groundtrackfield, harbor, overpass, ship, stadium, storagetank,
    tenniscourt, trainstation, vehicle, windmill

Dataset structure (expected under --root):
    DIOR-R/
      JPEGImages/              *.jpg  (all images, all splits)
      Annotations/             *.xml  (all annotations, all splits)
      ImageSets/Main/
        train.txt              image stems, one per line (no extension)
        val.txt
        test.txt               (may not exist; labels will be empty)

Output structure under --out:
    images/
      train/   *.jpg
      val/     *.jpg
      test/    *.jpg
    labels/
      train/   *.txt
      val/     *.txt
      test/    *.txt  (empty files if no annotations)

Download:
    Official page:  https://gcheng-nwpu.github.io/#Datasets
    The dataset (~3.5 GB) is hosted on Google Drive and publicly accessible.
    After downloading and extracting, point --root to the DIOR-R folder.

Usage:
    python utils/dior_converter.py \\
        --root /data/DIOR-R \\
        --out  /data/DIOR-R/yolo

    Then update data/dior.yaml: set  path: /data/DIOR-R/yolo
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

# All names are lower-cased and stripped; some annotation files use
# slightly different capitalisation — handled via _normalise_name().
_CLASS_NAMES: list[str] = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "expressway-service-area",
    "expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]
_NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(_CLASS_NAMES)}

# Aliases for annotation variations seen in the wild
_ALIASES: dict[str, str] = {
    "golfield":                 "golffield",
    "golf-field":               "golffield",
    "ground-track-field":       "groundtrackfield",
    "expressway_service_area":  "expressway-service-area",
    "expressway_toll_station":  "expressway-toll-station",
    "baseball-field":           "baseballfield",
    "basketball-court":         "basketballcourt",
    "tennis-court":             "tenniscourt",
    "train-station":            "trainstation",
    "storage-tank":             "storagetank",
}


def _normalise_name(raw: str) -> str | None:
    """Lower-case, strip, then resolve alias. Returns None if unknown."""
    name = raw.strip().lower()
    name = _ALIASES.get(name, name)
    return name if name in _NAME_TO_ID else None


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_xml(xml_path: Path, img_w: int, img_h: int,
              skip_difficult: bool) -> list[str]:
    """
    Parse one DIOR-R XML annotation file.
    Returns list of YOLO-format label strings.
    skip_difficult: if True, objects with <difficult>1</difficult> are dropped.
    """
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        print(f"  [WARN] XML parse error in {xml_path.name}: {e} — skipping")
        return []

    root = tree.getroot()
    lines: list[str] = []

    for obj in root.findall("object"):
        # Skip difficult instances if requested
        diff = obj.findtext("difficult", default="0").strip()
        if skip_difficult and diff == "1":
            continue

        name = obj.findtext("name", default="").strip()
        cls_name = _normalise_name(name)
        if cls_name is None:
            continue
        cls_id = _NAME_TO_ID[cls_name]

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            continue

        # Clamp to image bounds
        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            continue

        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def get_image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height). Tries PIL first, falls back to cv2."""
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


# ---------------------------------------------------------------------------
# Per-split conversion
# ---------------------------------------------------------------------------

def convert_split(
    split_name:     str,
    img_stems:      list[str],
    src_img_dir:    Path,
    src_ann_dir:    Path,
    out_img_dir:    Path,
    out_lbl_dir:    Path,
    skip_difficult: bool,
    verbose:        bool,
) -> tuple[int, int]:
    """Convert one split. Returns (n_images, n_boxes)."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = errors = 0

    for stem in img_stems:
        stem = stem.strip()
        if not stem:
            continue

        # Find image (try jpg, png, tif)
        img_path = None
        for ext in (".jpg", ".png", ".tif", ".tiff", ".jpeg"):
            candidate = src_img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"  [WARN] Image not found for stem '{stem}' — skipping")
            errors += 1
            continue

        # Copy image
        dst_img = out_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Parse annotation
        xml_path = src_ann_dir / f"{stem}.xml"
        lbl_path = out_lbl_dir / f"{stem}.txt"

        if xml_path.exists():
            try:
                w, h = get_image_size(img_path)
                yolo_lines = parse_xml(xml_path, w, h, skip_difficult)
                lbl_path.write_text(
                    "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
                )
                n_boxes += len(yolo_lines)
            except Exception as e:
                print(f"  [WARN] Failed {stem}: {e}")
                lbl_path.write_text("")
                errors += 1
        else:
            lbl_path.write_text("")   # test split has no annotations

        n_images += 1

    if verbose:
        print(f"  [{split_name:<5s}]  {n_images:5d} images  {n_boxes:8d} boxes"
              + (f"  ({errors} errors)" if errors else ""))

    return n_images, n_boxes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert DIOR-R dataset (VOC XML HBB) to YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",  required=True, type=str,
                   help="DIOR-R root directory (contains JPEGImages/, Annotations/, ImageSets/)")
    p.add_argument("--out",   required=True, type=str,
                   help="Output directory for YOLO-format dataset")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Splits to convert (names must match ImageSets/Main/<name>.txt)")
    p.add_argument("--skip-difficult", action="store_true",
                   help="Skip objects marked difficult=1")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-split progress")
    return p.parse_args()


def _resolve_dir(candidates: list) -> Path | None:
    """Return first existing path from candidates list, or None."""
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    out  = Path(args.out)

    # Auto-detect image directory (handles both standard and zip-extracted names)
    img_dir = _resolve_dir([
        root / "JPEGImages",
        root / "JPEGImages-trainval",
        root / "JPEGImages-test",
    ])

    # Auto-detect annotation directory
    # Official download extracts to "Annotations/Horizontal Bounding Boxes/"
    ann_dir = _resolve_dir([
        root / "Annotations" / "Horizontal Bounding Boxes",
        root / "Annotations",
    ])

    # Auto-detect ImageSets (handles both ImageSets/Main/ and direct Main/)
    set_dir = _resolve_dir([
        root / "ImageSets" / "Main",
        root / "Main",
    ])

    print("=" * 68)
    print("  DIOR-R → YOLO HBB Converter")
    print("=" * 68)
    print(f"  Source  : {root}")
    print(f"  Output  : {out}")
    print(f"  Splits  : {args.splits}")
    print(f"  Classes : {len(_CLASS_NAMES)}")
    print("=" * 68)
    print(f"  Images  : {img_dir}")
    print(f"  Annots  : {ann_dir}")
    print(f"  Splits  : {set_dir}")
    print("=" * 68)

    # Validate resolved directories
    missing = []
    if img_dir is None:
        missing.append("JPEGImages/ (or JPEGImages-trainval/)")
    if ann_dir is None:
        missing.append("Annotations/ (or Annotations/Horizontal Bounding Boxes/)")
    if set_dir is None:
        missing.append("ImageSets/Main/ (or Main/)")
    if missing:
        print(f"\n  [ERROR] Missing directories under {root}:")
        for m in missing:
            print(f"    - {m}")
        print(f"\n  Expected (one of these layouts):\n"
              f"    {root}/\n"
              f"      JPEGImages/                           (all images)\n"
              f"      Annotations/Horizontal Bounding Boxes/  (XML files)\n"
              f"      ImageSets/Main/train.txt val.txt\n")
        return

    total_images = total_boxes = 0

    for split_name in args.splits:
        split_list = set_dir / f"{split_name}.txt"
        if not split_list.exists():
            print(f"\n  [SKIP] {split_list} not found — skipping split '{split_name}'")
            continue

        img_stems = split_list.read_text().splitlines()
        img_stems = [s.strip() for s in img_stems if s.strip()]

        if not img_stems:
            print(f"\n  [SKIP] {split_list} is empty — skipping")
            continue

        print(f"\n  Processing split: {split_name}  ({len(img_stems)} images)")
        n_imgs, n_boxes = convert_split(
            split_name     = split_name,
            img_stems      = img_stems,
            src_img_dir    = img_dir,
            src_ann_dir    = ann_dir,
            out_img_dir    = out / "images" / split_name,
            out_lbl_dir    = out / "labels" / split_name,
            skip_difficult = args.skip_difficult,
            verbose        = not args.quiet,
        )
        total_images += n_imgs
        total_boxes  += n_boxes

    print("\n" + "=" * 68)
    print(f"  Done — {total_images} images / {total_boxes} boxes written to: {out}")
    print("=" * 68)
    print("\n  Next steps:")
    print(f"  1. Update data/dior.yaml: set  path: {out}")
    print("  2. Re-cluster anchors for DIOR-R (800px images):")
    print(f"     python utils/ciou_kmeans.py --label-dir {out}/labels/train \\")
    print("         --img-size 800 --n-clusters 12")
    print("  3. Train:")
    print("     python train_da_yolo.py --data data/dior.yaml \\")
    print("         --mode scratch --img-size 800")


if __name__ == "__main__":
    main()
