import cv2
import numpy as np

def calculate_sharpness(gray_img):
    """
    Calculate image sharpness / focus level using Laplacian variance.

    The Laplacian operator highlights rapid intensity changes (edges). High variance
    indicates sharp edges; low variance indicates blur or out-of-focus capture.

    Args:
        gray_img (np.ndarray): 2D grayscale image array.

    Returns:
        dict: Contains sharpness_score (float), status (str), and recommendation (str).
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Invalid grayscale image for blur calculation.")

    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    variance = float(laplacian.var())
    score = round(variance, 2)

    if variance < 100.0:
        status = "Blurry / Out of Focus"
        recommendation = "Low edge definition detected. Use a steady camera mount, faster shutter speed, or adjust focus."
    elif 100.0 <= variance < 300.0:
        status = "Moderate Sharpness"
        recommendation = "Acceptable clarity, but minor soft focus or motion blur may be present."
    else:
        status = "Sharp / In Focus"
        recommendation = "High edge crispness and strong detail definition."

    return {
        "sharpness_score": score,
        "status": status,
        "recommendation": recommendation,
    }
