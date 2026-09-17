import cv2
import numpy as np

def to_grayscale(img):
    """
    Convert an OpenCV BGR image array to 8-bit single-channel grayscale.

    Args:
        img (np.ndarray): Input image array (BGR or Gray).

    Returns:
        np.ndarray: uint8 2D grayscale image array.
    """
    if img is None:
        raise ValueError("Input image array cannot be None.")

    if len(img.shape) == 2:
        return img.astype(np.uint8)
    
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            return img[:, :, 0].astype(np.uint8)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image dimension for grayscale conversion: {img.shape}")


def apply_gaussian_blur(gray_img, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian filtering to smooth image noise while preserving global structures.

    Args:
        gray_img (np.ndarray): 2D grayscale image.
        kernel_size (tuple): Odd tuple kernel dimensions (default: (5,5)).
        sigma (float): Standard deviation in X and Y directions.

    Returns:
        np.ndarray: Smoothed grayscale image.
    """
    if gray_img is None or len(gray_img.shape) != 2:
        raise ValueError("Gaussian blur requires a 2D grayscale image.")

    return cv2.GaussianBlur(gray_img, kernel_size, sigma)


def equalize_histogram(gray_img):
    """
    Perform global histogram equalization to redistribute pixel intensities
    and enhance overall image contrast.

    Args:
        gray_img (np.ndarray): 2D grayscale image.

    Returns:
        np.ndarray: Equalized 2D grayscale image.
    """
    if gray_img is None or len(gray_img.shape) != 2:
        raise ValueError("Histogram equalization requires a 2D grayscale image.")

    return cv2.equalizeHist(gray_img.astype(np.uint8))
