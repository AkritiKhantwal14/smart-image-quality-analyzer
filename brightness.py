import numpy as np

def calculate_brightness(gray_img):
    """
    Calculate brightness using the mean grayscale intensity.

    Args:
        gray_img (np.ndarray): 2D grayscale image array.

    Returns:
        dict: Contains brightness_score (float), status (str), and recommendation (str).
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Invalid grayscale image for brightness calculation.")

    mean_intensity = float(np.mean(gray_img))
    score = round(mean_intensity, 2)

    if mean_intensity < 40.0:
        status = "Too Dark"
        recommendation = "Severe underexposure detected. Increase scene illumination or camera exposure/ISO."
    elif 40.0 <= mean_intensity < 70.0:
        status = "Slightly Dark"
        recommendation = "Image is somewhat dark. Consider slightly increasing light or boosting exposure."
    elif 70.0 <= mean_intensity <= 180.0:
        status = "Optimal Brightness"
        recommendation = "Brightness is well-balanced across the dynamic range."
    elif 180.0 < mean_intensity <= 220.0:
        status = "Slightly Bright"
        recommendation = "Image is slightly bright. Check for mild highlight clipping."
    else:
        status = "Too Bright / Overexposed"
        recommendation = "Severe overexposure detected. Reduce light source intensity or decrease camera exposure."

    return {
        "brightness_score": score,
        "status": status,
        "recommendation": recommendation,
    }
