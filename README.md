<div align="center">
  <h1>🦅 SwinYOLO: Swin-Transformer-Based YOLOv5 for Small-Object Detection</h1>
  <p>
    An enhanced architecture built on top of YOLOv5, explicitly optimized for dense, high-resolution remote sensing imagery and small-object detection.
  </p>
</div>

## 📌 Overview

This repository provides a custom PyTorch implementation of the architecture detailed in the research paper: **"Swin-Transformer-Based YOLOv5 for Small-Object Detection in Remote Sensing Images"** _(Sensors 2023, 23, 3634)_.

While vanilla YOLOv5 excels at generalized object detection, its performance falters on exceptionally dense clusters and small symbols commonly found in aerial photography, drones, and high-resolution maps. SwinYOLO upgrades the traditional CSPDarknet footprint natively, adding a marginal ~0.4M parameters in exchange for robust spatial awareness and tiny-object acuity.

## ✨ Key Architectural Enhancements

SwinYOLO improves upon the baseline YOLOv5s architecture through five fundamental modifications:

1. **CIOU K-Means Anchoring**: Upgraded initial bounding box clustering logic utilizing Complete-IOU (CIOU) instead of standard Euclidean mathematics. This factors aspect-ratio scaling to perfectly map dataset-specific anchors.
2. **C3SW-T (Swin Transformer Blocks)**: Replaces traditional Bottleneck convolution logic deep in the network with modified Swin Transformer operations. _Note: We intentionally excluded LayerNorms here to prevent CNN feature degradation!_
3. **BiFPN Cross-Scale Fusion**: Fully replaces the PANet neck. Features are fused using EfficientDet's fast normalized weighted fusion (leveraging Depthwise Separable Convolutions)—allowing the network to learn the empirical importance of each varying scale directly.
4. **Coordinate Attention (CA)**: Precise spatial positional data is injected adjacent to the detection heads bounding both the X and Y coordinate planes.
5. **4-Tier Detection Head**: Augments the standard 3-scale detection architecture by retaining an ultra-high resolution P2 tracking head exclusively for microscopic symbols.
6. **Deformable Conv v2 (DCNv2)**: Integrated into the BiFPN neck (at the P3 scale) to improve spatial alignment and receptive field adaptability for irregular objects (like rotated vehicles or dense clusters).

---

## 🏗️ Architecture Flow Diagram

```mermaid
graph TD
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000;
    classDef backbone fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    classDef swin fill:#cce5ff,stroke:#007bff,stroke-width:2px,color:#000;
    classDef neck fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#000;
    classDef head fill:#f8d7da,stroke:#dc3545,stroke-width:2px,color:#000;

    A[Input Image<br/>640x640]:::input --> B(Conv Layers P1/P2):::backbone
    B --> C(C3 Modules)

    subgraph CSP-Swin Backbone
        C --> D[C3 Output P3]:::backbone
        D --> E[C3 Output P4]:::backbone
        E --> F[C3SWT Layer P5]:::swin
        F --> G[SPPF]:::backbone
    end

    subgraph Weighted BiFPN Fusion Neck
        G --> H1[BiFPN Layer]:::neck
        E --> H1
        D --> H1
        B --> H1

        H1 --> N1[BiFPN P5]
        H1 --> N2[BiFPN P4]
        H1 --> N3[BiFPN P3 + DCNv2]:::neck
        H1 --> N4[BiFPN P2]
    end

    subgraph Prediction
        N1 --> CA1[CoordAtt]:::head --> HD1((Head P5)):::head
        N2 --> CA2[CoordAtt]:::head --> HD2((Head P4)):::head
        N3 --> CA3[CoordAtt]:::head --> HD3((Head P3)):::head
        N4 --> CA4[CoordAtt]:::head --> HD4((Head P2 Small Obj)):::head
    end
```

---

## 🚀 Quickstart Guide

### 1. Generate Dataset Anchors

Because SwinYOLO relies heavily on tight anchors for dense images, you must generate dataset-specific anchors leveraging CIOU. Update your dataset path in `data/pid.yaml` and run:

```bash
python utils/ciou_kmeans.py --label-dir path/to/your/labels/train --img-size 640 --n-clusters 12
```

_Copy the resultant anchors directly into `models/yolov5s_swint.yaml`._

### 2. Begin Training

Start training directly invoking the newly constructed Swin-Transformer configuration file!

```bash
python train.py \
  --img 640 \
  --batch 32 \
  --epochs 300 \
  --data data/pid.yaml \
  --cfg models/yolov5s_swint.yaml \
  --weights yolov5s.pt
```

_Weights with matching sizes (like early CNN layers) will be seamlessly migrated from standard YOLOv5s._

## 📂 Core Component Locations

If you wish to explore the implemented module topologies:

- `models/yolov5s_swint.yaml` - Master routing configuration unifying the components.
- `models/swin_block.py` - Contains `C3SWT` and the LayerNorm-ablated Swin Transformer.
- `models/bifpn.py` - Contains the `BiFPNLayer` using dynamic parameter scaling and depthwise separated convolutions.
- `models/coord_attention.py` - Contains `CoordAtt` & `CoordAttMulti` global pooling architectures.
- `utils/ciou_kmeans.py` - Anchor generation scripting.
