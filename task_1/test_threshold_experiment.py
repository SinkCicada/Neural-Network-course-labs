"""阈值实验校验：先运行 threshold_experiment.py，再运行此文件。"""
import csv
import unittest

import numpy as np
import letter_bp as bp
import threshold_experiment as experiment


def read(name):
    with (experiment.OUT / name).open(encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


class ThresholdTests(unittest.TestCase):
    def test_strict_threshold_and_default(self):
        self.assertEqual(bp.recognize([.91, .09, 0, 0, 0, 0]), "A")
        for score in ([.9, 0, 0, 0, 0, 0], [.91, .1, 0, 0, 0, 0],
                      [.91, .91, 0, 0, 0, 0]):
            self.assertEqual(bp.recognize(score), "?")
        score = [.7, .2, 0, 0, 0, 0]
        self.assertEqual(bp.recognize(score), "?")
        self.assertEqual(bp.recognize(score, .6, .4), "A")

    def test_disjoint_splits_and_exact_noise(self):
        x = np.concatenate([bp.to_vector(bp.PATTERNS[c]) for c in bp.LETTERS])
        before = x.copy()
        records = experiment.make_splits(x)
        self.assertTrue(np.array_equal(before, x))
        old = {tuple(v) for _, _, _, v in bp.make_tests(x)} | {tuple(v) for v in x}
        seen = set()
        for split, k, letter, _, vector in records:
            key = tuple(vector)
            self.assertNotIn(key, old)
            self.assertNotIn(key, seen)
            self.assertEqual(np.count_nonzero(vector != x[bp.LETTERS.index(letter)]), k)
            seen.add(key)
        self.assertEqual(sum(r[0] == "validation" for r in records), 282)
        self.assertEqual(sum(r[0] == "test" for r in records), 288)

    def test_saved_model_reproduces_outputs(self):
        rows = read("split_outputs.csv")
        model = bp.myBP(35, 6, experiment.HIDDEN)
        with np.load(experiment.OUT / "fixed_model.npz") as weights:
            for name in ("w", "b", "v", "c"):
                setattr(model, name, weights[name].copy())
        x = np.array([[int(v) for v in row["column_scan_bits"]] for row in rows])
        saved = np.array([[float(row[c]) for c in bp.LETTERS] for row in rows])
        np.testing.assert_allclose(model.forward(x)[0], saved, atol=1e-12, rtol=0)

    def test_counts_match_raw_predictions(self):
        predictions = read("heldout_predictions.csv")
        for row in read("heldout_comparison.csv"):
            details = [p for p in predictions if p["rule"] == row["rule"]
                       and (row["flipped_pixels"] == "all"
                            or p["flipped_pixels"] == row["flipped_pixels"])]
            correct = rejected = 0
            for p in details:
                value = bp.recognize([float(p[c]) for c in bp.LETTERS],
                                     float(row["on"]), float(row["off"]))
                self.assertEqual(value, p["predicted"])
                correct += value == p["actual"]
                rejected += value == "?"
            self.assertEqual(len(details), int(row["total"]))
            self.assertEqual(correct, int(row["correct"]))
            self.assertEqual(rejected, int(row["rejected"]))
            self.assertEqual(len(details) - correct - rejected, int(row["wrong"]))


if __name__ == "__main__":
    unittest.main()
