def evaluate_quality(brightness_res, contrast_res, sharpness_res, noise_res):
    """
    Evaluate overall image quality using a transparent rule-based scoring algorithm.

    Starting from a baseline of 100 points, deductions are calculated based on
    heuristic thresholds for brightness, contrast, blur, and noise levels.

    Args:
        brightness_res (dict): Result from brightness calculation module.
        contrast_res (dict): Result from contrast calculation module.
        sharpness_res (dict): Result from sharpness calculation module.
        noise_res (dict): Result from noise calculation module.

    Returns:
        dict: Overall quality score (0-100), category label, and breakdown of deductions.
    """
    b_val = brightness_res.get("brightness_score", 128.0)
    c_val = contrast_res.get("contrast_score", 50.0)
    s_val = sharpness_res.get("sharpness_score", 200.0)
    n_val = noise_res.get("noise_score", 2.0)

    deductions = {
        "brightness": 0.0,
        "contrast": 0.0,
        "sharpness": 0.0,
        "noise": 0.0,
    }

    # 1. Brightness Penalty (ideal range: 70 - 180)
    if b_val < 40.0:
        deductions["brightness"] = min(35.0, (40.0 - b_val) * 0.8)
    elif b_val > 220.0:
        deductions["brightness"] = min(35.0, (b_val - 220.0) * 0.8)
    elif 40.0 <= b_val < 70.0:
        deductions["brightness"] = 10.0
    elif 180.0 < b_val <= 220.0:
        deductions["brightness"] = 10.0

    # 2. Contrast Penalty (ideal range: >= 45.0)
    if c_val < 25.0:
        deductions["contrast"] = min(30.0, (25.0 - c_val) * 1.2)
    elif 25.0 <= c_val < 45.0:
        deductions["contrast"] = 10.0

    # 3. Blur / Sharpness Penalty (ideal range: >= 300.0)
    if s_val < 100.0:
        deductions["sharpness"] = min(35.0, (100.0 - s_val) * 0.35)
    elif 100.0 <= s_val < 300.0:
        deductions["sharpness"] = 10.0

    # 4. Noise Penalty (ideal range: < 3.0)
    if n_val >= 7.0:
        deductions["noise"] = min(25.0, (n_val - 7.0) * 3.0)
    elif 3.0 <= n_val < 7.0:
        deductions["noise"] = 5.0

    total_deduction = sum(deductions.values())
    raw_score = 100.0 - total_deduction
    overall_score = float(max(0.0, min(100.0, round(raw_score, 1))))

    if overall_score >= 80.0:
        category = "Good Quality"
    elif 55.0 <= overall_score < 80.0:
        category = "Average Quality"
    else:
        category = "Needs Improvement"

    # Round deductions for report clarity
    formatted_deductions = {k: round(v, 1) for k, v in deductions.items()}

    return {
        "overall_score": overall_score,
        "category": category,
        "deductions": formatted_deductions,
    }
