import cv2
import numpy as np

def calculate_noise(gray_img):
    """
    Estimate image noise by measuring standard deviation of high-frequency residuals
    between original grayscale image and a Gaussian-blurred version.

    Args:
        gray_img (np.ndarray): 2D grayscale image array.

    Returns:
        dict: Contains noise_score (float), status (str), and recommendation (str).
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Invalid grayscale image for noise estimation.")

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1.0)
    residual = gray_img.astype(np.float64) - blurred.astype(np.float64)
    noise_std = float(np.std(residual))
    score = round(noise_std, 2)

    if noise_std < 3.0:
        status = "Low Noise"
        recommendation = "Clean image signal with minimal sensor noise or grain."
    elif 3.0 <= noise_std < 7.0:
        status = "Moderate Noise"
        recommendation = "Slight sensor noise visible. Lower camera ISO or apply mild noise filtering if needed."
    else:
        status = "High Noise / Grainy"
        recommendation = "High digital noise detected. Improve light levels to avoid high ISO gain or apply spatial noise reduction."

    return {
        "noise_score": score,
        "status": status,
        "recommendation": recommendation,
    }
