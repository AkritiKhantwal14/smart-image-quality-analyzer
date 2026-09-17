import cv2
import numpy as np
from PIL import Image
import io

def load_image(source):
    """
    Load and validate image from file path, byte stream, or UploadedFile.

    Args:
        source: File path (str), bytes, BytesIO stream, or Streamlit UploadedFile.

    Returns:
        tuple: (img_bgr, height, width, channels)

    Raises:
        ValueError: If file is invalid, empty, or unreadable as an image.
    """
    if source is None:
        raise ValueError("No image file provided.")

    img_np = None

    try:
        if isinstance(source, (str, tuple)):
            img_np = cv2.imread(str(source))
        elif isinstance(source, (bytes, bytearray)):
            file_bytes = np.frombuffer(source, np.uint8)
            img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif hasattr(source, "read"):
            source.seek(0)
            file_bytes = np.frombuffer(source.read(), np.uint8)
            img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif isinstance(source, np.ndarray):
            img_np = source.copy()
        elif isinstance(source, Image.Image):
            rgb_arr = np.array(source)
            if rgb_arr.ndim == 2:
                img_np = cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
            elif rgb_arr.shape[2] == 4:
                img_np = cv2.cvtColor(rgb_arr, cv2.COLOR_RGBA2BGR)
            else:
                img_np = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported input format.")

    except Exception as e:
        raise ValueError(f"Failed to read image file: {str(e)}")

    if img_np is None or img_np.size == 0 or len(img_np.shape) < 2:
        raise ValueError("The provided file could not be decoded as a valid image.")

    if len(img_np.shape) == 2:
        height, width = img_np.shape
        channels = 1
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        height, width, channels = img_np.shape
        img_bgr = img_np

    return img_bgr, height, width, channels


def get_image_metadata(img_bgr):
    """
    Extract dimensional metadata from an OpenCV BGR image.

    Args:
        img_bgr (np.ndarray): Image array.

    Returns:
        dict: Metadata dictionary containing height, width, channels, total pixels, and aspect ratio.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        raise ValueError("Invalid image array provided for metadata extraction.")

    if len(img_bgr.shape) == 2:
        h, w = img_bgr.shape
        c = 1
    else:
        h, w, c = img_bgr.shape

    return {
        "height": h,
        "width": w,
        "channels": c,
        "total_pixels": h * w,
        "aspect_ratio": round(w / h, 2) if h > 0 else 0.0,
    }
