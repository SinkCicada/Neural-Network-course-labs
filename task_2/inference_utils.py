"""Shared, training-free sliced inference for the local detection demos.

The detector callback receives a PIL RGB crop and must return boxes in that
crop's pixel coordinates. Predictions are projected to the original image and
merged with class-aware NMS. Full-image predictions preserve larger objects
that can be truncated by cropping. No model, threshold or weights are changed.
"""

import math
import time

import torch
from torchvision.ops import batched_nms, nms


def _positions(length, tile_size, step):
    """Include the last edge without padding or duplicate crop origins."""
    if length <= tile_size:
        return [0]
    last = length - tile_size
    starts = list(range(0, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_tiled(image, predict_fn, tile_size=640, overlap=0.25, merge_iou=0.5,
              max_detections=300, include_full=True, merge_fn=None):
    """Return ``(detections, stats)`` after full-image and sliced inference.

    Each detection has ``class_id``, ``class_name``, ``score``, ``box_xyxy``.
    Extra callback metadata is retained from the surviving candidate. The
    callback is responsible for its own score threshold and per-view NMS.
    An optional merge_fn receives all projected candidates, replacing the
    default hard NMS and limit. It must apply its own output count limit.

    ``tile_count`` counts actual crop calls, excluding the optional full-image
    call. ``total_calls`` includes both. Images fitting in one tile are inferred
    only once: as a full image when include_full=True, otherwise as one crop.
    ``elapsed_ms`` covers every callback, coordinate projection and merge,
    rather than only model-forward time.
    """
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if not 0 <= merge_iou <= 1:
        raise ValueError("merge_iou must be in [0, 1]")
    if not isinstance(max_detections, int) or max_detections < 1:
        raise ValueError("max_detections must be a positive integer")
    image = image.convert("RGB")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")

    start = time.perf_counter()
    step = max(1, int(tile_size * (1 - overlap)))
    xs = _positions(width, tile_size, step)
    ys = _positions(height, tile_size, step)
    candidates = []
    tile_count = 0
    full_calls = 0

    def collect(view, x_offset, y_offset):
        view_width, view_height = view.size
        for prediction in predict_fn(view):
            box = [float(v) for v in prediction["box_xyxy"]]
            score = float(prediction["score"])
            if len(box) != 4 or not all(math.isfinite(v) for v in box + [score]):
                raise ValueError("detector returned non-finite score/coordinates or an invalid box")
            # Clip in crop coordinates first, so padding cannot spill into
            # neighboring image regions when offsets are restored.
            x1 = min(view_width, max(0.0, box[0])) + x_offset
            y1 = min(view_height, max(0.0, box[1])) + y_offset
            x2 = min(view_width, max(0.0, box[2])) + x_offset
            y2 = min(view_height, max(0.0, box[3])) + y_offset
            if x2 <= x1 or y2 <= y1:
                continue
            item = dict(prediction)
            item.update(class_id=int(prediction["class_id"]), score=score,
                        box_xyxy=[x1, y1, x2, y2])
            candidates.append(item)

    if include_full:
        collect(image, 0, 0)
        full_calls = 1
    if not include_full or width > tile_size or height > tile_size:
        for top in ys:
            for left in xs:
                right = min(width, left + tile_size)
                bottom = min(height, top + tile_size)
                collect(image.crop((left, top, right, bottom)), left, top)
                tile_count += 1

    if merge_fn is not None:
        detections = merge_fn(candidates)
    elif candidates:
        boxes = torch.tensor([p["box_xyxy"] for p in candidates], dtype=torch.float32)
        scores = torch.tensor([p["score"] for p in candidates], dtype=torch.float32)
        classes = torch.tensor([p["class_id"] for p in candidates], dtype=torch.int64)
        keep = batched_nms(boxes, scores, classes, merge_iou)[:max_detections].tolist()
        detections = [candidates[index] for index in keep]
    else:
        detections = []

    stats = {
        "tile_count": tile_count,
        "full_image_calls": full_calls,
        "total_calls": tile_count + full_calls,
        "candidate_count": len(candidates),
        "merged_count": len(detections),
        "tile_size": tile_size,
        "overlap": overlap,
        "merge_iou": merge_iou,
        "max_detections": max_detections,
        "include_full": bool(include_full),
        "elapsed_ms": (time.perf_counter() - start) * 1000,
    }
    return detections, stats


CAR_CLASSES = frozenset({"car", "truck", "bus"})
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})


def suppress_vehicle_duplicates(detections, iou_threshold=0.6, enabled=True):
    """Resolve overlapping vehicle labels; preserve other objects and input order.

    Run after per-class / cross-tile merging, in original-image coordinates.
    Category names avoid relying on the different COCO IDs in Torchvision/YOLO.
    The highest scoring candidate survives; this does not correct its class.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("vehicle NMS IoU must be in [0, 1]")
    vehicle_indices = [i for i, item in enumerate(detections)
                       if item["class_name"] in VEHICLE_CLASSES]
    suppressed = []
    if enabled and len(vehicle_indices) > 1:
        boxes = torch.tensor([detections[i]["box_xyxy"] for i in vehicle_indices], dtype=torch.float32)
        scores = torch.tensor([detections[i]["score"] for i in vehicle_indices], dtype=torch.float32)
        kept = set(nms(boxes, scores, iou_threshold).tolist())
        suppressed = [index for local, index in enumerate(vehicle_indices) if local not in kept]
    removed = set(suppressed)
    result = [item for i, item in enumerate(detections) if i not in removed]
    stats = {"enabled": bool(enabled), "iou_threshold": iou_threshold,
             "classes": sorted(VEHICLE_CLASSES), "input_count": len(detections),
             "output_count": len(result), "removed_count": len(suppressed),
             "suppressed_detections": [detections[i] for i in suppressed]}
    return result, stats
