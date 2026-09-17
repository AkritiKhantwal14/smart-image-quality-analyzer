import unittest
from src.quality_score import evaluate_quality

class TestQualityScore(unittest.TestCase):
    def test_ideal_metrics(self):
        b_res = {"brightness_score": 120.0, "status": "Optimal Brightness"}
        c_res = {"contrast_score": 60.0, "status": "Good Contrast"}
        s_res = {"sharpness_score": 400.0, "status": "Sharp / In Focus"}
        n_res = {"noise_score": 1.5, "status": "Low Noise"}

        res = evaluate_quality(b_res, c_res, s_res, n_res)
        self.assertEqual(res["overall_score"], 100.0)
        self.assertEqual(res["category"], "Good Quality")

    def test_poor_metrics(self):
        b_res = {"brightness_score": 10.0, "status": "Too Dark"}
        c_res = {"contrast_score": 10.0, "status": "Very Low Contrast"}
        s_res = {"sharpness_score": 20.0, "status": "Blurry / Out of Focus"}
        n_res = {"noise_score": 15.0, "status": "High Noise / Grainy"}

        res = evaluate_quality(b_res, c_res, s_res, n_res)
        self.assertLess(res["overall_score"], 55.0)
        self.assertEqual(res["category"], "Needs Improvement")
        self.assertGreaterEqual(res["overall_score"], 0.0)

    def test_score_range_bounds(self):
        b_res = {"brightness_score": 0.0, "status": "Too Dark"}
        c_res = {"contrast_score": 0.0, "status": "Very Low Contrast"}
        s_res = {"sharpness_score": 0.0, "status": "Blurry"}
        n_res = {"noise_score": 100.0, "status": "High Noise"}

        res = evaluate_quality(b_res, c_res, s_res, n_res)
        self.assertGreaterEqual(res["overall_score"], 0.0)
        self.assertLessEqual(res["overall_score"], 100.0)

if __name__ == "__main__":
    unittest.main()
