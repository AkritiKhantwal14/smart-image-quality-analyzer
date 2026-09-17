import numpy as np

def calculate_contrast(gray_img):
    """
    Calculate contrast using standard deviation of grayscale intensity.

    Args:
        gray_img (np.ndarray): 2D grayscale image array.

    Returns:
        dict: Contains contrast_score (float), status (str), and recommendation (str).
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Invalid grayscale image for contrast calculation.")

    std_dev = float(np.std(gray_img))
    score = round(std_dev, 2)

    if std_dev < 25.0:
        status = "Very Low Contrast"
        recommendation = "Flat pixel intensity distribution. Consider applying histogram equalization or enhancing contrast."
    elif 25.0 <= std_dev < 45.0:
        status = "Moderate Contrast"
        recommendation = "Acceptable contrast, though darks and highlights could be better separated."
    elif 45.0 <= std_dev <= 85.0:
        status = "Good Contrast"
        recommendation = "Excellent dynamic range with clear separation between light and dark regions."
    else:
        status = "High Contrast"
        recommendation = "High dynamic contrast. Ensure extreme highlights or shadow details are not lost."

    return {
        "contrast_score": score,
        "status": status,
        "recommendation": recommendation,
    }
