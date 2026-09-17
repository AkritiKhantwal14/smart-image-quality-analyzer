# Smart Image Quality Analyzer 🔍

An original, lightweight academic Computer Vision project for automated image quality assessment, metric evaluation, histogram analysis, and interactive reporting using Python, OpenCV, NumPy, and Streamlit.

---

## 📌 Project Overview
The **Smart Image Quality Analyzer** is a local, rule-based web application that evaluates digital images across key quality metrics:
* **Brightness** (Mean Grayscale Intensity)
* **Contrast** (Standard Deviation of Intensity)
* **Sharpness / Blur** (Laplacian Variance)
* **Noise Level** (Gaussian Residual Standard Deviation)
* **Pixel Intensity Distribution** (Grayscale & Equalized Histograms)

The system computes an overall **Quality Score out of 100**, classifies the image into quality tiers (**Good Quality**, **Average Quality**, **Needs Improvement**), displays transparent penalty deductions, and offers practical recommendations for improvement.

---

## 🎯 Objectives
* Demonstrate fundamental Computer Vision and Digital Image Processing techniques without relying on deep learning models or external cloud APIs.
* Provide an objective, explainable 0–100 quality scoring metric.
* Offer interactive visualization of histogram equalization for contrast enhancement.
* Implement a clean, modular Python codebase with complete unit test coverage.

---

## ⚡ Features
* **Image Input & Validation**: Ingest JPG, JPEG, PNG, and BMP images with dimensional metadata extraction.
* **Deterministic Analysis**:
  * **Brightness**: Identifies under/overexposure.
  * **Contrast**: Measures dynamic range.
  * **Sharpness**: Detects focus and motion blur.
  * **Noise**: Quantifies sensor grain using spatial residuals.
* **Histogram Analysis & Equalization**: Side-by-side comparison of original vs histogram equalized image.
* **Explainable Quality Score**: Rule-based deduction breakdown.
* **Actionable Feedback**: Tailored photographic and lighting recommendations.
* **Preset Sample Images**: Pre-bundled test images for instant demonstration.

---

## 🔬 Computer Vision Concepts Used
1. **Grayscale Conversion**: Reductive luminance mapping $Y = 0.299R + 0.587G + 0.114B$.
2. **Mean Intensity ($\mu$)**: Global brightness metric.
3. **Standard Deviation ($\sigma$)**: Dynamic range and contrast dispersion metric.
4. **Laplacian Variance ($\text{Var}(\Delta I)$)**: High-frequency edge gradient energy for blur detection.
5. **Gaussian Filtering & Residual Noise**: Difference of Gaussian spatial smoothing for zero-mean noise variance estimation.
6. **Histogram Equalization**: Cumulative Distribution Function (CDF) mapping for contrast enhancement.

---

## 🛠️ Tech Stack
* **Python 3.10+**
* **Streamlit**: Web interface framework
* **OpenCV (`opencv-python`)**: Image processing operations
* **NumPy**: Matrix computation and statistical metrics
* **Matplotlib**: Histogram plot rendering
* **Pillow**: Image formatting

---

## 📂 Project Structure
```text
smart-image-quality-analyzer/
│
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies list
├── README.md                  # Comprehensive project documentation
├── statement.md               # Academic project statement
│
├── src/                       # Modular source code package
│   ├── __init__.py            # Package initialization
│   ├── image_loader.py        # Image decoding and metadata extraction
│   ├── preprocessing.py       # Grayscale, Gaussian blur, histogram equalization
│   ├── brightness.py          # Brightness calculation & thresholding
│   ├── contrast.py            # Contrast calculation & thresholding
│   ├── blur_detection.py      # Laplacian variance blur detection
│   ├── noise_detection.py     # Gaussian residual noise estimation
│   ├── histogram.py           # Histogram computation & plot generation
│   ├── quality_score.py       # Rule-based score evaluation (0-100)
│   └── report.py              # Full analysis pipeline orchestration
│
├── tests/                     # Unit test suite
│   ├── test_brightness.py     # Brightness calculation unit tests
│   ├── test_contrast.py       # Contrast calculation unit tests
│   └── test_quality.py        # Quality score bounds and categorization tests
│
└── assets/
    └── sample_images/         # Pre-generated sample images for testing
```

---

## 📥 Installation Instructions

1. Ensure Python 3.8+ is installed on your system.
2. Open terminal in the project directory:
   ```bash
   cd smart-image-quality-analyzer
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

Launch the Streamlit web application with:
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the test suite using Python's standard `unittest` framework:
```bash
python -m unittest discover -s tests -v
```

---

## 🔄 Example Workflow
1. Launch `streamlit run app.py`.
2. Select a preset sample image or upload a custom image (e.g. `photo.jpg`).
3. View the **Overall Quality Score** and **Quality Category**.
4. Inspect individual metrics (**Brightness**, **Contrast**, **Sharpness**, **Noise**).
5. Toggle **Histogram Equalization** to preview contrast enhancement.
6. Review the **Practical Recommendations** section to fix lighting or focus issues.

---

## ⚠️ Limitations
* **Rule-Based Thresholds**: Uses fixed empirical thresholds which may vary depending on aesthetic artistic choices (e.g., intentional low-key silhouette photography).
* **Single Channel Blur**: Blur analysis is performed on grayscale luminance rather than per-color channel.

---

## 🔮 Future Enhancements
* Color cast and white balance deviation measurement.
* Structural Similarity Index (SSIM) reference comparison.
* Spatial noise reduction filtering preview (Median & Bilateral filtering).
