"""
Image preprocessing for scanned exam papers and mobile photos.

Steps: grayscale -> deskew -> denoise -> adaptive threshold -> CLAHE.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

# Minimum absolute skew angle (degrees) before rotation is applied.
_MIN_SKEW_ANGLE = 0.5

# Maximum skew angle (degrees) we attempt to correct.
_MAX_SKEW_ANGLE = 45.0


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image around its center while preserving full content."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    border_value = 255 if len(image.shape) == 2 else (255, 255, 255)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _normalize_line_angle(theta: float) -> float:
    """
    Convert a Hough theta (radians) to a skew angle in degrees.

    Returns an angle in [-45, 45], suitable for deskewing text lines.
    """
    angle = np.degrees(theta) - 90.0
    if angle < -45.0:
        angle += 90.0
    elif angle > 45.0:
        angle -= 90.0
    return angle


def _collect_hough_angles(gray: np.ndarray) -> list[float]:
    """
    Detect skew angles from Hough lines using multiple edge-detection settings.

    Mobile photos often need stronger blur and lower Hough thresholds;
    scanned pages usually work with sharper edges and higher thresholds.
    """
    angles: list[float] = []

    # Resize very large images so Hough detection stays fast and stable.
    height, width = gray.shape[:2]
    scale = 1.0
    working = gray
    if max(height, width) > 2000:
        scale = 2000.0 / max(height, width)
        working = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    configs = [
        # Typical scanned document
        {"blur": (3, 3), "canny": (50, 150), "hough_threshold": 120},
        # Noisy mobile photo
        {"blur": (5, 5), "canny": (30, 120), "hough_threshold": 80},
        # Low-contrast capture
        {"blur": (7, 7), "canny": (20, 100), "hough_threshold": 60},
    ]

    for config in configs:
        blurred = cv2.GaussianBlur(working, config["blur"], 0)
        edges = cv2.Canny(blurred, config["canny"][0], config["canny"][1], apertureSize=3)

        # Close small gaps in text strokes before line detection.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, config["hough_threshold"])
        if lines is None:
            continue

        for line in lines:
            theta = line[0][1]
            angle = _normalize_line_angle(theta)
            if abs(angle) <= _MAX_SKEW_ANGLE:
                angles.append(angle)

        # Probabilistic Hough helps when lines are short or fragmented (mobile photos).
        prob_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, config["hough_threshold"] // 2),
            minLineLength=max(working.shape[1] // 8, 40),
            maxLineGap=20,
        )
        if prob_lines is None:
            continue

        for x1, y1, x2, y2 in prob_lines[:, 0]:
            delta_x = x2 - x1
            delta_y = y2 - y1
            if delta_x == 0:
                continue
            angle = np.degrees(np.arctan2(delta_y, delta_x))
            if abs(angle) <= _MAX_SKEW_ANGLE:
                angles.append(angle)

    return angles


def _detect_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate document skew using the median of detected line angles.

    Median aggregation is robust against outliers from borders, shadows,
    and non-text edges in both scanned and mobile captures.
    """
    angles = _collect_hough_angles(gray)
    if not angles:
        return 0.0

    median_angle = float(np.median(angles))

    # Ignore tiny corrections that mostly add interpolation artifacts.
    if abs(median_angle) < _MIN_SKEW_ANGLE:
        return 0.0

    return median_angle


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Detect skew angle and rotate the grayscale image to straighten text."""
    angle = _detect_skew_angle(gray)
    if abs(angle) < _MIN_SKEW_ANGLE:
        return gray
    return _rotate_image(gray, angle)


def _save_temp_image(image: np.ndarray) -> Path:
    """Persist the processed image to a temporary file and return its path."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.close()
    temp_path = Path(temp_file.name)
    cv2.imwrite(str(temp_path), image)
    return temp_path


def enhance_page_rgb(image: np.ndarray) -> np.ndarray:
    """
    Enhance a color page for layout detection and OCR without binarizing.

    Applies deskew (using grayscale line detection), light denoise, and CLAHE
    on the L channel in LAB space so Surya layout and VLM crops stay sharp.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    angle = _detect_skew_angle(gray)
    if abs(angle) >= _MIN_SKEW_ANGLE:
        image = _rotate_image(image, angle)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    denoised_gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Blend denoised luminance with CLAHE for uneven mobile lighting.
    l_blend = cv2.addWeighted(l_channel, 0.65, denoised_gray, 0.35, 0)
    enhanced = cv2.merge([l_blend, a_channel, b_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def preprocess_image(input_path: str) -> np.ndarray:
    """
    Load and preprocess an exam image for downstream OCR/layout steps.

    Args:
        input_path: Path to a scanned page or mobile photo.

    Returns:
        Preprocessed image as a NumPy array (single-channel, uint8).
        The result is also written to a temporary PNG file on disk.
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not load image from path: {input_path}")

    # a. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # b. Deskew using Hough line angles (robust for scan + mobile input)
    gray = _deskew(gray)

    # c. Denoise while preserving text edges
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # d. Adaptive threshold for uneven lighting (common in phone photos)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # e. Increase local contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(thresholded)

    # f. Save processed output to a temp path for inspection/debugging
    _save_temp_image(processed)

    return processed
