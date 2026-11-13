"""Exercise score decay, ranking, grouping and pre-merge candidate retention."""
import copy
import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_utils import run_tiled
from retinanet.soft_nms import soft_nms_detections


def det(score, box, name='car'):
    return {'class_id': 3, 'class_name': name, 'score': score, 'box_xyxy': box}


class SoftNMSTests(unittest.TestCase):
    def test_overlapping_neighbor_survives_with_decayed_score_and_reordering(self):
        candidates = [det(.9, [0, 0, 100, 100]), det(.8, [20, 0, 120, 100]),
                      det(.5, [200, 0, 300, 100])]
        untouched = copy.deepcopy(candidates)
        result, stats = soft_nms_detections(candidates, .5, .2, vehicle_iou_threshold=.5)
        self.assertEqual([r['raw_score'] for r in result], [.9, .5, .8])
        self.assertAlmostEqual(result[-1]['score'], .8 / 3, places=6)
        self.assertEqual(stats['decayed_candidate_count'], 1)
        self.assertEqual(candidates, untouched)

    def test_vehicle_grouping_is_optional_and_other_classes_are_independent(self):
        candidates = [det(.9, [0, 0, 100, 100]), det(.8, [0, 0, 100, 100], 'truck'),
                      det(.7, [0, 0, 100, 100], 'person')]
        self.assertEqual(len(soft_nms_detections(candidates)[0]), 3)
        grouped, _ = soft_nms_detections(candidates, vehicle_iou_threshold=.4)
        self.assertEqual([r['class_name'] for r in grouped], ['car', 'person'])

    def test_final_score_filter_limit_and_empty(self):
        candidates = [det(.9, [0, 0, 100, 100]), det(.8, [20, 0, 120, 100])]
        self.assertEqual(len(soft_nms_detections(candidates, score_threshold=.35)[0]), 1)
        self.assertEqual(len(soft_nms_detections(candidates, max_detections=1)[0]), 1)
        self.assertEqual(soft_nms_detections([])[0], [])

    def test_tiled_merge_gets_candidates_before_hard_nms_or_limit(self):
        candidates = [det(.9, [0, 0, 100, 100]), det(.8, [20, 0, 120, 100])]
        def merge(items):
            self.assertEqual(len(items), 2)
            return soft_nms_detections(items, score_threshold=.2)[0]
        result, _ = run_tiled(Image.new('RGB', (200, 120)), lambda _: candidates,
                             tile_size=640, max_detections=1, merge_fn=merge)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
