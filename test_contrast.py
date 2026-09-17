import unittest
import numpy as np
from src.contrast import calculate_contrast

class TestContrast(unittest.TestCase):
    def test_zero_contrast(self):
        # Uniform intensity array has 0 standard deviation
        flat_img = np.full((100, 100), 100, dtype=np.uint8)
        res = calculate_contrast(flat_img)
        self.assertEqual(res["contrast_score"], 0.0)
        self.assertEqual(res["status"], "Very Low Contrast")

    def test_high_contrast(self):
        # Half 0 and half 255 pixels
        high_img = np.zeros((100, 100), dtype=np.uint8)
        high_img[:, :50] = 255
        res = calculate_contrast(high_img)
        self.assertGreater(res["contrast_score"], 85.0)
        self.assertEqual(res["status"], "High Contrast")

if __name__ == "__main__":
    unittest.main()
