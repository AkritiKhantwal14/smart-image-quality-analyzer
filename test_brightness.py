import unittest
import numpy as np
from src.brightness import calculate_brightness

class TestBrightness(unittest.TestCase):
    def test_dark_image(self):
        # 100x100 array of pure black pixels (intensity 0)
        dark_img = np.zeros((100, 100), dtype=np.uint8)
        res = calculate_brightness(dark_img)
        self.assertEqual(res["brightness_score"], 0.0)
        self.assertEqual(res["status"], "Too Dark")

    def test_bright_image(self):
        # 100x100 array of pure white pixels (intensity 255)
        bright_img = np.full((100, 100), 255, dtype=np.uint8)
        res = calculate_brightness(bright_img)
        self.assertEqual(res["brightness_score"], 255.0)
        self.assertEqual(res["status"], "Too Bright / Overexposed")

    def test_optimal_image(self):
        # 100x100 array with mid-gray pixels (intensity 128)
        opt_img = np.full((100, 100), 128, dtype=np.uint8)
        res = calculate_brightness(opt_img)
        self.assertEqual(res["brightness_score"], 128.0)
        self.assertEqual(res["status"], "Optimal Brightness")

if __name__ == "__main__":
    unittest.main()
