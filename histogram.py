import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(gray_img):
    """
    Compute 256-bin grayscale intensity histogram.

    Args:
        gray_img (np.ndarray): 2D grayscale image array.

    Returns:
        np.ndarray: 1D array of 256 intensity bin counts.
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Invalid grayscale image for histogram calculation.")

    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    return hist.flatten()


def plot_histogram(gray_img, eq_img=None):
    """
    Generate a styled Matplotlib figure plotting grayscale intensity distribution.

    Args:
        gray_img (np.ndarray): 2D grayscale image array.
        eq_img (np.ndarray, optional): Equalized grayscale image array for comparison.

    Returns:
        matplotlib.figure.Figure: Matplotlib plot figure.
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("Invalid grayscale image for histogram plotting.")

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")

    hist_orig = calculate_histogram(gray_img)
    bins = np.arange(256)

    ax.plot(bins, hist_orig, color="#89b4fa", linewidth=1.8, label="Original Grayscale")
    ax.fill_between(bins, hist_orig, color="#89b4fa", alpha=0.2)

    if eq_img is not None:
        hist_eq = calculate_histogram(eq_img)
        ax.plot(bins, hist_eq, color="#a6e3a1", linewidth=1.8, linestyle="--", label="Histogram Equalized")
        ax.fill_between(bins, hist_eq, color="#a6e3a1", alpha=0.15)

    ax.set_title("Grayscale Pixel Intensity Distribution", color="#cdd6f4", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Pixel Intensity Value (0 - 255)", color="#a6adc8", fontsize=10)
    ax.set_ylabel("Pixel Frequency Count", color="#a6adc8", fontsize=10)
    ax.set_xlim([0, 255])

    ax.tick_params(colors="#a6adc8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#45475a")

    ax.grid(True, linestyle=":", color="#313244", alpha=0.6)
    ax.legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=9)
    plt.tight_layout()

    return fig
