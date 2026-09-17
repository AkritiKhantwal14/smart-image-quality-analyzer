from .image_loader import load_image, get_image_metadata
from .preprocessing import to_grayscale, apply_gaussian_blur, equalize_histogram
from .brightness import calculate_brightness
from .contrast import calculate_contrast
from .blur_detection import calculate_sharpness
from .noise_detection import calculate_noise
from .quality_score import evaluate_quality

def generate_full_report(source):
    """
    Run the end-to-end computer vision image quality analysis pipeline.

    Args:
        source: File path, bytes, BytesIO stream, PIL Image, or UploadedFile.

    Returns:
        dict: Full report containing image_bgr, gray_img, equalized_img, metadata,
              individual metric dicts, overall quality dict, and list of recommendations.
    """
    img_bgr, h, w, c = load_image(source)
    metadata = get_image_metadata(img_bgr)
    gray_img = to_grayscale(img_bgr)
    equalized_img = equalize_histogram(gray_img)

    brightness_res = calculate_brightness(gray_img)
    contrast_res = calculate_contrast(gray_img)
    sharpness_res = calculate_sharpness(gray_img)
    noise_res = calculate_noise(gray_img)

    quality_res = evaluate_quality(brightness_res, contrast_res, sharpness_res, noise_res)

    # Collect recommendations for non-optimal statuses
    recommendations = []
    if brightness_res["status"] != "Optimal Brightness":
        recommendations.append(f"**Brightness**: {brightness_res['recommendation']}")
    if contrast_res["status"] != "Good Contrast":
        recommendations.append(f"**Contrast**: {contrast_res['recommendation']}")
    if sharpness_res["status"] != "Sharp / In Focus":
        recommendations.append(f"**Sharpness**: {sharpness_res['recommendation']}")
    if noise_res["status"] != "Low Noise":
        recommendations.append(f"**Noise**: {noise_res['recommendation']}")

    if not recommendations:
        recommendations.append("All core image quality metrics are within optimal ranges!")

    return {
        "img_bgr": img_bgr,
        "gray_img": gray_img,
        "equalized_img": equalized_img,
        "metadata": metadata,
        "brightness": brightness_res,
        "contrast": contrast_res,
        "sharpness": sharpness_res,
        "noise": noise_res,
        "quality": quality_res,
        "recommendations": recommendations,
    }
