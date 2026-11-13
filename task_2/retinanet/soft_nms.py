"""Linear Soft-NMS, following the book's section 8.2.3 and Bodla et al.

The optional vehicle grouping is an additional project-specific treatment for
car/truck/bus/motorcycle label ambiguity; it is not the book's MDA model.
"""
import torch
from torchvision.ops import box_iou

from inference_utils import VEHICLE_CLASSES


def soft_nms_detections(detections, iou_threshold=0.5, score_threshold=0.35,
                        max_detections=300, vehicle_iou_threshold=None):
    """Decay score by (1 - IoU) above threshold, reselecting the current max.

    Nonvehicle categories are independent. Set vehicle_iou_threshold to group
    the four vehicle categories; None keeps all categories independent.
    No candidate is removed by hard NMS here. Decayed scores below the output
    score threshold are discarded. Input records are never mutated.
    """
    if not 0 <= iou_threshold <= 1 or not 0 <= score_threshold <= 1:
        raise ValueError('Soft-NMS IoU and score thresholds must be in [0, 1]')
    if vehicle_iou_threshold is not None and not 0 <= vehicle_iou_threshold <= 1:
        raise ValueError('vehicle Soft-NMS IoU must be in [0, 1]')
    if max_detections < 1:
        raise ValueError('max_detections must be positive')
    stats = {'method': 'linear', 'iou_threshold': iou_threshold,
             'score_threshold': score_threshold, 'vehicle_iou_threshold': vehicle_iou_threshold,
             'input_count': len(detections), 'output_count': 0, 'decayed_candidate_count': 0}
    if not detections:
        return [], stats
    boxes = torch.tensor([d['box_xyxy'] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError('Soft-NMS received non-finite coordinates/scores')
    valid = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
             & (scores >= score_threshold))
    remaining = torch.where(valid)[0]
    group_ids = {}
    groups = []
    thresholds = []
    for d in detections:
        vehicle = vehicle_iou_threshold is not None and d['class_name'] in VEHICLE_CLASSES
        token = ('vehicle',) if vehicle else ('class', d['class_name'])
        groups.append(group_ids.setdefault(token, len(group_ids)))
        thresholds.append(vehicle_iou_threshold if vehicle else iou_threshold)
    groups = torch.tensor(groups, dtype=torch.int64)
    decayed = set()
    selected = []
    while remaining.numel() and len(selected) < max_detections:
        best_local = int(scores[remaining].argmax())
        best = int(remaining[best_local])
        item = dict(detections[best])
        item['raw_score'] = item.get('raw_score', item['score'])
        item['score'] = float(scores[best])
        selected.append(item)
        remaining = torch.cat((remaining[:best_local], remaining[best_local + 1:]))
        same_group = remaining[groups[remaining] == groups[best]]
        if same_group.numel():
            overlaps = box_iou(boxes[best:best + 1], boxes[same_group])[0].clamp(0, 1)
            attenuate = overlaps > thresholds[best]
            affected = same_group[attenuate]
            scores[affected] *= 1 - overlaps[attenuate]
            decayed.update(affected.tolist())
        remaining = remaining[scores[remaining] >= score_threshold]
    stats['output_count'] = len(selected)
    stats['decayed_candidate_count'] = len(decayed)
    return selected, stats
