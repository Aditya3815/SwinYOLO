import argparse
from pathlib import Path

import numpy as np


def ciou_distance(box, anchor):
    """CIOU distance calculation per paper. Centers are aligned, so ρ²(bA, bB) / c² = 0 Returns 1 - CIOU which serves as
    the distance measure.
    """
    w1, h1 = box
    w2, h2 = anchor

    # Intersection and Union
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + 1e-16)

    # Aspect ratio consistency v
    v = (4 / (np.pi**2)) * ((np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2)

    # Trade-off parameter alpha
    with np.errstate(divide="ignore"):
        alpha = v / ((1 - iou) + v + 1e-16)

    # CIOU
    ciou = iou - alpha * v

    # Distance
    return 1 - ciou


def calc_dist_matrix(boxes, anchors):
    """Calculate distance matrix between all boxes and anchors."""
    N = len(boxes)
    K = len(anchors)
    dist_mat = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            dist_mat[i, j] = ciou_distance(boxes[i], anchors[j])
    return dist_mat


def ciou_kmeans(boxes, k, dist_fn=calc_dist_matrix, max_iter=300):
    """K-means using CIOU distance. boxes: numpy array of shape (N, 2) k: number of clusters (e.g., 12 for 4 scales x 3
    anchors).
    """
    num_boxes = boxes.shape[0]

    # Randomly initialize k centroids
    indices = np.random.choice(num_boxes, k, replace=False)
    clusters = boxes[indices].copy()

    prev_clusters = np.zeros_like(clusters)
    best_clusters = None
    min_dist_sum = float("inf")

    for iteration in range(max_iter):
        distances = dist_fn(boxes, clusters)

        # Assign boxes to nearest cluster
        nearest_clusters = np.argmin(distances, axis=1)

        dist_sum = np.sum(np.min(distances, axis=1))
        if dist_sum < min_dist_sum:
            min_dist_sum = dist_sum
            best_clusters = clusters.copy()

        if np.array_equal(clusters, prev_clusters):
            break

        prev_clusters = clusters.copy()
        for i in range(k):
            cluster_boxes = boxes[nearest_clusters == i]
            if len(cluster_boxes) > 0:
                # Update cluster center to the median of the boxes
                clusters[i] = np.median(cluster_boxes, axis=0)

    # Sort clusters by area
    areas = best_clusters[:, 0] * best_clusters[:, 1]
    best_clusters = best_clusters[np.argsort(areas)]

    # Calculate Best Possible Recall (BPR)
    w_out, h_out = best_clusters[:, 0], best_clusters[:, 1]
    w_in, h_in = boxes[:, 0], boxes[:, 1]

    # IoU between all boxes and all anchors
    inter = np.minimum(w_in[:, None], w_out[None, :]) * np.minimum(h_in[:, None], h_out[None, :])
    union = w_in[:, None] * h_in[:, None] + w_out[None, :] * h_out[None, :] - inter
    ious = inter / (union + 1e-16)

    # Max IoU for each box
    max_ious = np.max(ious, axis=1)
    bpr = (max_ious > 0.5).mean()

    print(f"K-means completed. BPR@0.5: {bpr:.4f}")
    return best_clusters


def generate_anchors(label_dir, n=12, img_size=640):
    """Loads labels from text files and clusters them using CIOU."""
    path = Path(label_dir)
    if not path.exists():
        print(f"Error: Path {path} does not exist.")
        return None

    txt_files = list(path.rglob("*.txt"))
    if len(txt_files) == 0:
        print(f"No label files found in {path}")
        return None

    print(f"Loading labels from {len(txt_files)} files...")
    wh_list = []
    for f in txt_files:
        try:
            with open(f) as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        wh_list.append([w, h])
        except Exception:
            pass

    wh = np.array(wh_list)
    if len(wh) == 0:
        print("No valid boxes found.")
        return None

    print(f"Found {len(wh)} boxes. Running CIOU K-means clustering for {n} anchors...")

    # Run CIOU K-means
    anchors_normalized = ciou_kmeans(wh, n)

    # Scale to image size
    anchors = anchors_normalized * img_size
    anchors = np.round(anchors).astype(int)

    print("\nGenerated Anchors (scaled to img_size):")
    # Group into P2, P3, P4, P5 (if n=12)
    group_size = 3
    num_scales = n // group_size

    scales = ["P2/4", "P3/8", "P4/16", "P5/32"]
    yaml_anchors = []

    for i in range(num_scales):
        scale_name = scales[i] if i < len(scales) else f"Scale_{i}"
        scale_anchors = anchors[i * group_size : (i + 1) * group_size]
        flat_anchors = []
        for a in scale_anchors:
            flat_anchors.extend(a.tolist())
        yaml_anchors.append(flat_anchors)
        print(f"  - {flat_anchors}  # {scale_name}")

    return yaml_anchors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=str, required=True, help="Directory containing YOLO label txt files")
    parser.add_argument("--img-size", type=int, default=640, help="Image size anchors are scaled to")
    parser.add_argument("--n-clusters", type=int, default=12, help="Number of anchors (12 for 4 scales)")
    args = parser.parse_args()

    generate_anchors(args.label_dir, args.n_clusters, args.img_size)
