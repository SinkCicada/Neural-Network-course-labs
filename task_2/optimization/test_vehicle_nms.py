"""Check vehicle-only suppression without changing nested nonvehicle objects."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_utils import suppress_vehicle_duplicates


def detection(name, score, box, class_id=1):
    return {"class_name": name, "class_id": class_id,
            "score": score, "box_xyxy": box}


class VehicleNMSTests(unittest.TestCase):
    def test_overlapping_vehicle_labels_keep_higher_score_for_both_id_schemes(self):
        for car_id, truck_id in ((3, 8), (2, 7)):
            items = [detection("truck", .56, [0, 0, 100, 100], truck_id),
                     detection("car", .69, [0, 0, 100, 100], car_id)]
            result, stats = suppress_vehicle_duplicates(items)
            self.assertEqual(result, [items[1]])
            self.assertEqual(stats["suppressed_detections"], [items[0]])
            self.assertEqual(stats["removed_count"], 1)
            self.assertEqual(len(items), 2)

    def test_preserve_adjacent_cars_and_overlapping_nonvehicles_in_input_order(self):
        items = [detection("car", .6, [0, 0, 100, 100]),
                 detection("person", .8, [0, 0, 100, 100]),
                 detection("backpack", .7, [0, 0, 100, 100]),
                 detection("car", .9, [80, 0, 180, 100])]
        result, stats = suppress_vehicle_duplicates(items)
        self.assertEqual(result, items)
        self.assertEqual(stats["removed_count"], 0)

    def test_disable_and_adjust_iou(self):
        items = [detection("car", .9, [0, 0, 100, 100]),
                 detection("bus", .6, [20, 0, 120, 100])]
        self.assertEqual(suppress_vehicle_duplicates(items, enabled=False)[0], items)
        self.assertEqual(len(suppress_vehicle_duplicates(items, .6)[0]), 1)
        self.assertEqual(suppress_vehicle_duplicates(items, .8)[0], items)

    def test_empty_single_and_nonvehicle_duplicates(self):
        self.assertEqual(suppress_vehicle_duplicates([])[0], [])
        one = [detection("motorcycle", .8, [0, 0, 10, 10])]
        self.assertEqual(suppress_vehicle_duplicates(one)[0], one)
        other = [detection("person", .8, [0, 0, 10, 10]),
                 detection("person", .7, [0, 0, 10, 10])]
        self.assertEqual(suppress_vehicle_duplicates(other)[0], other)


if __name__ == "__main__":
    unittest.main()
