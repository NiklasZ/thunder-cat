import math

import cv2 as cv
import numpy as np
from cv2 import BackgroundSubtractorMOG2
from cv2.typing import MatLike

from configuration import configure_logger

logger = configure_logger()


# Colours for bounding boxes
COLOURS = [
    (0, 0, 255),  # BLUE
    (255, 0, 0),  # RED
    (0, 255, 0),  # GREEN
    (255, 255, 0),  # YELLOW
    (0, 255, 255),  # CYAN
    (255, 0, 255),  # MAGENTA
]


class ClusterBoundingBox:
    def __init__(self, ids: list[int], size: int, x_min: int, y_min: int, x_max: int, y_max: int):
        self.ids = ids  # Ids of clusters in the box
        self.size = size  # Num pixels in cluster
        self.x_min = x_min  # Minimum x-coordinate
        self.y_min = y_min  # Minimum y-coordinate
        self.x_max = x_max  # Maximum x-coordinate
        self.y_max = y_max  # Maximum y-coordinate

    def __repr__(self):
        return f"ClusterBoundingBox({self.size}, {self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"

    def min_distance(self, other: "ClusterBoundingBox") -> float:
        """Calculates Eucliadean minimum distance between a pair of boxes.
        Returns:
            float: min distance
        """
        # Calculate horizontal and vertical distances
        dx = max(0, max(self.x_min, other.x_min) - min(self.x_max, other.x_max))
        dy = max(0, max(self.y_min, other.y_min) - min(self.y_max, other.y_max))
        # Euclidean distance
        return math.hypot(dx, dy)

    def merge(self, other: "ClusterBoundingBox") -> "ClusterBoundingBox":
        """Combines a pair of boxes, creating a new box that spans the area of both boxes.

        Args:
            other (ClusterBoundingBox): box to combine with

        Returns:
            ClusterBoundingBox: combined box
        """
        new_x_min = min(self.x_min, other.x_min)
        new_y_min = min(self.y_min, other.y_min)
        new_x_max = max(self.x_max, other.x_max)
        new_y_max = max(self.y_max, other.y_max)
        return ClusterBoundingBox(
            self.ids + other.ids, self.size + other.size, new_x_min, new_y_min, new_x_max, new_y_max
        )


def merge_cumulatively(clusters: list[ClusterBoundingBox], distance_threshold: float) -> list[ClusterBoundingBox]:
    """Merges all clusters whose bounding boxes are within a certain distance of each other.
    This is the naive O(k^2) complexity approach for k clusters. There are in principle some
    other algorithms like spatial partitioning or similar local neighbourhood checks that can do
    O(k log k) or O(k + C), but they may not be much faster in practice.
    These are also a bit impractical as they usually assume to be working with points, while we are dealing with
    4 point tuples (boxes) and we define new ones as we progress.

    Args:
        clusters (list[ClusterBoundingBox]): clusters to merge
        distance_threshold (float): max distance allowed for merging

    Returns:
        list[ClusterBoundingBox]: merged clusters
    """
    changed = True
    groups = [c for c in clusters]
    while changed:  # Repeat until no more changes
        changed = False
        new_clusters = []
        skip_indices = set()  # clusters that were merged away

        # Iterate over clusters
        for i, cluster1 in enumerate(groups):
            if i in skip_indices:
                continue
            merged_cluster = cluster1
            # Iterate over clusters again
            for j in range(i + 1, len(groups)):
                if j in skip_indices:
                    continue
                cluster2 = clusters[j]
                if merged_cluster.min_distance(cluster2) <= distance_threshold:
                    merged_cluster = merged_cluster.merge(cluster2)
                    skip_indices.add(j)
                    changed = True
            new_clusters.append(merged_cluster)
        groups = new_clusters

    return groups


def bounding_box_clusterer(
    frame: np.ndarray,
    min_initial_cluster_size: int,
    cluster_distance_threshold: float,
    min_final_cluster_size: int,
    min_final_cluster_density: float,
    min_final_bounding_box_length: int,
    pad_bounding_box_px: int,
) -> list[ClusterBoundingBox]:
    """Clusters foreground pixels in frame. Involves 3 steps:
    1. Find connected components and their boundaries.
    2. Construct a bounding box around each component.
    3. Merge boxes that are close enough to each other.

    Args:
        frame (np.ndarray): input frame
        min_initial_cluster_size (int): how large initial cluster should be to be considered for merging.
                                        Used to filter noise.
        cluster_distance_threshold (float): how far clusters are allowed to be to merge them.
        min_final_cluster_size (int): how large clusters have to be post-merging, to be considered.
                                      more noise filtering
        min_final_cluster_density (float): how dense (ratio of foreground pixels to size) bounding boxes have
                                           to be in order to be considered. Another filter parameter.
        min_final_bounding_box_length (int): minimum length of a bounding box along each axis.
        pad_bounding_box (int): most bounding boxes are very tight and tend to trim the edges. This
                                pads each side of the box by the corresponding number of pixels.

    Returns:
        list[ClusterBoundingBox]: merged clusters
    """

    # Get directly connected components and their bounding boxes
    _, _, comp_stats, _ = cv.connectedComponentsWithStats(frame, connectivity=8)

    # Filter out background components
    # NOTE: OpenCV returns the indices in reverse order (i.e x, y while the input is y,x)
    comp_stats = [c for c in comp_stats if frame[c[1], c[0]] != 0]

    # Filter out too small components
    comp_stats = [c for c in comp_stats if c[4] < min_initial_cluster_size]

    cluster_bounding_boxes = [
        ClusterBoundingBox([idx], s, x, y, x + w, y + h) for idx, (x, y, w, h, s) in enumerate(comp_stats)
    ]

    final_clusters = merge_cumulatively(cluster_bounding_boxes, cluster_distance_threshold)

    # Filter out any final clusters with a too low density or overall size
    filtered_clusters = [
        b
        for b in final_clusters
        if b.size / ((b.x_max - b.x_min) * (b.y_max - b.y_min)) >= min_final_cluster_density
        and b.size >= min_final_cluster_size
        and (b.x_max - b.x_min) >= min_final_bounding_box_length
        and (b.y_max - b.y_min) >= min_final_bounding_box_length
    ]

    # Pad bounding boxes on each side a little
    y_lim, x_lim = frame.shape
    for b in final_clusters:
        b.x_min = max(b.x_min - pad_bounding_box_px, 0)
        b.y_min = max(b.y_min - pad_bounding_box_px, 0)
        b.x_max = min(b.x_max + pad_bounding_box_px, x_lim - 1)
        b.y_max = min(b.y_max + pad_bounding_box_px, y_lim - 1)

    return filtered_clusters


def detect_motion(
    frame: MatLike,
    subtractor: BackgroundSubtractorMOG2,
    draw_bounding_boxes: bool,
    mask_pixels_below: int,
    sufficient_motion_thresh: int,
    show_background_sub_output: bool,
    clustering_config: dict,
) -> tuple[MatLike, list[ClusterBoundingBox]]:
    """We detect and bound motion by using background subtraction to identify
    new foreground pixels. These are then clustered together to find areas
    that we can draw bounding boxes around. This is not the most accurate approach
    but it is reasonably fast.

    Args:
        frame (MatLike): input image
        subtractor (BackgroundSubtractorMOG2): background subtractor
        draw_bounding_boxes (bool): whether to draw the bounding boxes
        mask_pixels_below (int): threshold to mask too dim pixels.
        sufficient_motion_thresh (int): how many foreground pixels are needed to bother clustering.
        show_background_sub_output (bool): whether to show the input frame or background difference frame.
                                            used for debugging
        clustering_config (dict): config for the clustering algorithm

    Returns:
        tuple[MatLike, list[ClusterBoundingBox]]: annotated frame and bounding boxes of motion.
    """

    # Get foreground active pixels
    f = subtractor.apply(frame)
    _, f = cv.threshold(f, mask_pixels_below, 255, cv.THRESH_BINARY)

    if show_background_sub_output:
        output_frame = cv.merge((f, f, f))
    else:
        output_frame = frame

    motion_px = np.stack(np.where(f == 255)).T

    clusters = []
    logger.debug(f"Motion pixels: {len(motion_px)} vs. required {sufficient_motion_thresh}")
    # Only run bounding boxing if there are enough active pixels.
    if len(motion_px) > sufficient_motion_thresh:

        # Cluster the pixels and get bounding boxes around them.
        clusters = bounding_box_clusterer(f, **clustering_config)

        if draw_bounding_boxes:
            sizes = [c.size for c in clusters]
            largest_size_indices = np.argsort(sizes)[::-1]

            for c_idx, col in zip(largest_size_indices, COLOURS):
                c = clusters[c_idx]
                cv.rectangle(output_frame, (c.x_min, c.y_min), (c.x_max, c.y_max), col, 2)

            if len(COLOURS) < len(clusters):
                logger.warning(f"Not enough colours for bounding boxes: {len(COLOURS)} vs {len(clusters)}")

    return output_frame, clusters
