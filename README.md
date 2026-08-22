# Video Processing Program Using OpenCV

A real-time OpenCV Python program that captures video from a webcam and applies various image processing filters and edge detectors dynamically.

## 👤 Author Details
* **Name:** Ahmed Wadan (ID: 221101204)
* **GitHub Repository:** [image_procssing_Project](https://github.com/ahmedwadan56-ops/image_procssing_Project)

---

## 📷 Implemented Filters & Operations

| Filter Category | Operation / Filter | Description |
| :--- | :--- | :--- |
| **Color Spaces** | Grayscale | Converts BGR frame to intensity-only grayscale. |
| **Smoothing** | Gaussian Blur | Reduces Gaussian noise using weighted averaging kernel. |
| | Mean Filters | Arithmetic, Geometric, Harmonic & Contraharmonic filters. |
| | Median Filter | Removes impulse (salt-and-pepper) noise. |
| **Order-Statistic** | Max / Min Filters | Enhances bright (Max) or dark (Min) regions. |
| | Midpoint Filter | Averages maximum and minimum neighborhood values. |
| | Alpha Trimmed Mean | Blends advantages of mean and median filters. |
| **Frequency** | Low Pass / High Pass | Smooths image vs. sharpens high-frequency edges. |
| **Edge Detection** | Sobel X & Sobel Y | Computes horizontal & vertical intensity gradients. |
| | Canny Edge Detector | Detects thin, well-defined edges using multi-stage thresholding. |

---

## 🚀 Setup & Execution

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
