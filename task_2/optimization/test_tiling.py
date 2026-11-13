"""Targeted coordinate/merging tests with a deterministic fake detector."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_utils import run_tiled


class TiledInferenceTests(unittest.TestCase):
    def test_offsets_class_aware_merge_and_edge_clipping(self):
        width, height = 900, 600
        pixels = np.arange(width * height, dtype=np.uint32).reshape(height, width)
        encoded = np.stack(((pixels >> 16) & 255, (pixels >> 8) & 255, pixels & 255), axis=2)
        image = Image.fromarray(encoded.astype('uint8'), 'RGB')
        objects = [
            (1, 'person', .90, [350, 250, 390, 290]),
            (2, 'another_class', .80, [350, 250, 390, 290]),
            (1, 'person', .70, [850, 550, 905, 605]),
        ]

        def detector(view):
            rgb = view.getpixel((0, 0))
            origin = (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]
            oy, ox = divmod(origin, width)
            predictions = []
            for cls, name, score, box in objects:
                if (box[0] >= ox and box[1] >= oy
                        and min(width, box[2]) <= ox + view.width
                        and min(height, box[3]) <= oy + view.height):
                    predictions.append(dict(class_id=cls, class_name=name, score=score,
                                            box_xyxy=[box[0]-ox, box[1]-oy, box[2]-ox, box[3]-oy]))
            return predictions

        predictions, stats = run_tiled(image, detector, tile_size=400, overlap=.25)
        self.assertEqual(stats['tile_count'], 6)
        self.assertEqual(stats['total_calls'], 7)
        self.assertEqual(stats['candidate_count'], 12)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(predictions[0]['box_xyxy'], [350, 250, 390, 290])
        self.assertEqual(predictions[1]['class_id'], 2)
        self.assertEqual(predictions[2]['box_xyxy'], [850, 550, 900, 600])
        limited, _ = run_tiled(image, detector, tile_size=400, max_detections=1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0]['score'], .90)

    def test_empty_predictions_and_real_image_crop_count(self):
        calls = []
        def empty(view):
            calls.append(view.size)
            return []
        predictions, stats = run_tiled(Image.new('RGB', (1440, 810)), empty)
        self.assertEqual(predictions, [])
        self.assertEqual(stats['tile_count'], 6)
        self.assertEqual(stats['total_calls'], 7)
        self.assertEqual(calls, [(1440, 810)] + [(640, 640)] * 6)

    def test_small_image_runs_only_once(self):
        for include_full in (True, False):
            calls = []
            def empty(view):
                calls.append(view.size)
                return []
            _, stats = run_tiled(Image.new('RGB', (100, 80)), empty, include_full=include_full)
            self.assertEqual(calls, [(100, 80)])
            self.assertEqual(stats['total_calls'], 1)
            self.assertEqual(stats['tile_count'], 0 if include_full else 1)

    def test_bad_parameters(self):
        image = Image.new('RGB', (50, 50))
        for kwargs in [dict(tile_size=0), dict(overlap=1), dict(merge_iou=-1), dict(max_detections=0)]:
            with self.assertRaises(ValueError):
                run_tiled(image, lambda _: [], **kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
