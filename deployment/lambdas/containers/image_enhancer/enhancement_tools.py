"""
Enhancement Tools for Agentic Image Enhancement
=================================================

Modular, individually-callable image operations designed to be invoked
by a vision LLM through tool use. Each operation:

  - Accepts a numpy array (RGB) and parameters
  - Supports optional region-based application (normalized 0-1 coords)
  - Returns the modified image as a numpy array (RGB)

These are the "hands" the LLM uses to manipulate images. The LLM acts
as the "eyes and brain" — deciding which tools to call, in what order,
and with what parameters based on its visual assessment.

Architecture:
    ┌─────────────────┐
    │  Vision LLM     │  ← Sees image, decides what's wrong
    │  (Claude, etc.) │
    └────────┬────────┘
             │ tool_use: enhance_image(operations=[...])
             ▼
    ┌─────────────────┐
    │ Enhancement     │  ← Executes requested operations
    │ Tools (this)    │
    └────────┬────────┘
             │ returns enhanced image
             ▼
    ┌─────────────────┐
    │  Vision LLM     │  ← Compares original vs enhanced
    │  (comparison)   │
    └─────────────────┘

Author: BADGERS Document Processing Pipeline
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union, cast
from pathlib import Path
import logging
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Region:
    """
    Normalized bounding box region (0.0 to 1.0 coordinates).

    This allows the LLM to specify regions using the same coordinate
    system it uses for spatial reasoning — no pixel math needed.

    Example:
        Region(x1=0.0, y1=0.0, x2=0.5, y2=0.5)  → top-left quadrant
        Region(x1=0.0, y1=0.0, x2=1.0, y2=1.0)  → full image (default)
    """

    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coords to pixel coords (x1, y1, x2, y2)."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )

    def is_full_image(self) -> bool:
        """Check if this region covers the entire image (0,0 to 1,1)."""
        return self.x1 <= 0.0 and self.y1 <= 0.0 and self.x2 >= 1.0 and self.y2 >= 1.0


@dataclass
class OperationResult:
    """Result of a single enhancement operation."""

    operation: str
    parameters: Dict[str, Any]
    region: Optional[Region] = None
    notes: str = ""


@dataclass
class EnhancementPlan:
    """
    The LLM's assessment and planned operations.

    This is what gets returned from the assessment step,
    before any operations are executed.
    """

    assessment: str  # LLM's description of image issues
    operations: List[Dict[str, Any]]  # Ordered list of operations to apply
    confidence: float  # 0-1, how confident the LLM is this will help
    skip_enhancement: bool = False  # If True, original is already good enough


# =============================================================================
# Image I/O Helpers
# =============================================================================


def load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Load image from various sources, always returns RGB numpy array.

    Args:
        source: File path, base64 string, or numpy array

    Returns:
        RGB numpy array (uint8)
    """
    if isinstance(source, np.ndarray):
        if len(source.shape) == 2:
            return cast(np.ndarray, cv2.cvtColor(source, cv2.COLOR_GRAY2RGB))
        if source.shape[2] == 4:
            return cast(np.ndarray, cv2.cvtColor(source, cv2.COLOR_RGBA2RGB))
        return cast(np.ndarray, source.copy())

    path = str(source)

    # Handle base64-encoded files
    if path.endswith(".b64"):
        with open(path, "r", encoding="utf-8") as f:
            b64_content = f.read()
        image_bytes = base64.b64decode(b64_content)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode base64 image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Standard image file
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_to_base64(image: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> str:
    """
    Encode numpy RGB image to base64 string for LLM API submission.

    Args:
        image: RGB numpy array
        fmt: 'jpeg' or 'png'
        quality: JPEG quality (1-100)

    Returns:
        Base64-encoded string
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if fmt == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = ".jpg"
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        ext = ".png"

    success, buffer = cv2.imencode(ext, bgr, params)
    if not success:
        raise RuntimeError("Failed to encode image")

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def save_image(image: np.ndarray, path: Union[str, Path], quality: int = 95) -> Path:
    """Save RGB numpy array to file."""
    path = Path(path)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if path.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), bgr)

    return path


# =============================================================================
# Region Application Helper
# =============================================================================


def _apply_to_region(
    image: np.ndarray,
    operation,
    region: Optional[Region] = None,
    **kwargs,
) -> np.ndarray:
    """
    Apply an operation to a specific region, blending edges smoothly.

    If region is None or covers the full image, applies globally.
    Otherwise, applies only within the bounding box with a soft
    feathered blend at the edges to avoid visible seams.

    Args:
        image: Input RGB image
        operation: Callable that takes (image_region, **kwargs) → image_region
        region: Optional normalized region
        **kwargs: Passed to operation

    Returns:
        Modified image
    """
    if region is None or region.is_full_image():
        result: np.ndarray = operation(image, **kwargs)
        return result

    h, w = image.shape[:2]
    x1, y1, x2, y2 = region.to_pixels(w, h)

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    result = image.copy()
    roi = image[y1:y2, x1:x2].copy()

    enhanced_roi = operation(roi, **kwargs)

    # Feathered blend: create a soft mask to avoid hard edges
    mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
    feather = min(20, min(x2 - x1, y2 - y1) // 4)
    if feather > 1:
        for i in range(feather):
            alpha = i / feather
            mask[i, :] *= alpha  # top edge
            mask[-(i + 1), :] *= alpha  # bottom edge
            mask[:, i] *= alpha  # left edge
            mask[:, -(i + 1)] *= alpha  # right edge

    mask_3ch = np.stack([mask] * 3, axis=-1)
    blended: np.ndarray = (enhanced_roi * mask_3ch + roi * (1 - mask_3ch)).astype(
        np.uint8
    )
    result[y1:y2, x1:x2] = blended

    return result


# =============================================================================
# Enhancement Operations
# =============================================================================
# Each operation follows the same signature pattern:
#   op(image: np.ndarray, intensity: float, ...) → np.ndarray
#
# intensity is always 0.0 to 1.0, where:
#   0.0 = no change
#   0.5 = moderate
#   1.0 = maximum effect


def adjust_contrast(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE is superior to global histogram equalization because it operates
    on local regions, preventing over-enhancement of already-bright areas.
    Particularly effective for documents with uneven lighting or fading.

    Args:
        image: RGB numpy array
        intensity: 0.0 (no change) to 1.0 (strong CLAHE)

    Returns:
        Contrast-enhanced image
    """
    if intensity <= 0.01:
        return image

    # Map intensity to CLAHE clip limit (1.0 = subtle, 6.0 = strong)
    clip_limit = 1.0 + (intensity * 5.0)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)

    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def adjust_brightness(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Adjust overall brightness.

    Uses Pillow's ImageEnhance for perceptually smooth brightness adjustment
    rather than raw pixel math (which clips harshly).

    Args:
        image: RGB numpy array
        intensity: 0.0 = darken significantly, 0.5 = no change, 1.0 = brighten significantly

    Returns:
        Brightness-adjusted image
    """
    # Map 0-1 to brightness factor (0.5 to 1.5)
    factor = 0.5 + intensity

    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    result = enhancer.enhance(factor)
    return np.array(result)


def sharpen(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """
    Sharpen via unsharp mask.

    Unsharp masking works by: sharp = original + amount * (original - blurred).
    This enhances edges without amplifying flat-area noise as much as
    simple kernel sharpening would.

    Lower intensity for manuscripts (preserve ink texture),
    higher for printed text or diagrams.

    Args:
        image: RGB numpy array
        intensity: 0.0 (no sharpening) to 1.0 (aggressive)

    Returns:
        Sharpened image
    """
    if intensity <= 0.01:
        return image

    # Map intensity to unsharp mask parameters
    sigma = 1.0 + intensity * 2.0  # blur radius
    amount = 0.5 + intensity * 2.0  # sharpening strength

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Remove noise while preserving edges.

    Uses OpenCV's Non-Local Means Denoising, which is slow but excellent
    at preserving text edges while removing paper grain and scanner noise.

    Warning: at high intensity, fine details (thin strokes, small text)
    may be softened. The LLM should use lower intensity for manuscripts
    with delicate handwriting.

    Args:
        image: RGB numpy array
        intensity: 0.0 (no denoising) to 1.0 (heavy)

    Returns:
        Denoised image
    """
    if intensity <= 0.01:
        return image

    # Map intensity to filter strength (3 = gentle, 15 = heavy)
    h_strength = int(3 + intensity * 12)
    h_color = int(3 + intensity * 12)

    return cv2.fastNlMeansDenoisingColored(image, None, h_strength, h_color, 7, 21)


def deskew(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Detect and correct document rotation/skew.

    Uses Hough line detection to find dominant text line angles,
    then rotates to correct. The intensity parameter controls the
    maximum angle that will be corrected (safety valve against
    false detections).

    Args:
        image: RGB numpy array
        intensity: 0.0-1.0, controls max correction angle (mapped to 2°-15°)

    Returns:
        Deskewed image
    """
    max_angle = 2.0 + intensity * 13.0  # 2° to 15°

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)

    coords = np.column_stack(np.where(gray > 0))
    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def white_balance(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Correct color cast from aging/yellowing using gray-world assumption.

    Historical documents often have a yellow-brown color cast from
    paper aging, foxing, or scanning artifacts. This normalizes the
    color channels to neutral gray, with intensity controlling how
    aggressively the cast is removed.

    Args:
        image: RGB numpy array
        intensity: 0.0 (no correction) to 1.0 (full normalization)

    Returns:
        Color-balanced image
    """
    if intensity <= 0.01:
        return image

    result = image.astype(np.float32)

    avg_r = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    avg_gray = (avg_r + avg_g + avg_b) / 3

    # Blend between original and corrected based on intensity
    corrected = result.copy()
    if avg_r > 0:
        corrected[:, :, 0] = result[:, :, 0] * (avg_gray / avg_r)
    if avg_g > 0:
        corrected[:, :, 1] = result[:, :, 1] * (avg_gray / avg_g)
    if avg_b > 0:
        corrected[:, :, 2] = result[:, :, 2] * (avg_gray / avg_b)

    blended = result * (1 - intensity) + corrected * intensity
    return cast(np.ndarray, np.clip(blended, 0, 255).astype(np.uint8))


def equalize_histogram(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Global histogram equalization on the luminance channel.

    More aggressive than CLAHE — spreads the full tonal range evenly.
    Good for severely faded documents, but can blow out details in
    documents that already have decent contrast. The LLM should prefer
    adjust_contrast (CLAHE) for most cases and reserve this for
    heavily degraded material.

    Args:
        image: RGB numpy array
        intensity: Blend factor (0.0 = original, 1.0 = fully equalized)

    Returns:
        Equalized image
    """
    if intensity <= 0.01:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    equalized_l = cv2.equalizeHist(l_ch)

    # Blend based on intensity
    l_blended = cv2.addWeighted(l_ch, 1 - intensity, equalized_l, intensity, 0)

    lab = cv2.merge([l_blended, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def auto_crop(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Detect and crop to document boundaries.

    Finds the largest contour (assumed to be the document) and crops
    to its bounding rectangle. Useful when document images have large
    borders, table surfaces, or other non-document areas.

    Args:
        image: RGB numpy array
        intensity: Controls padding (0.0 = tight crop, 1.0 = generous padding)

    Returns:
        Cropped image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to find document region
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Get bounding rect of largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add padding based on intensity
    padding = int((1.0 - intensity) * 0.05 * max(image.shape[:2]))
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    # Only crop if it's a meaningful reduction
    crop_area = (x2 - x1) * (y2 - y1)
    orig_area = img_w * img_h
    if crop_area < orig_area * 0.3:  # Don't crop to less than 30% of original
        return image

    return image[y1:y2, x1:x2]


def invert(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Invert image (negative).

    Useful for: dark-background documents, photographic negatives,
    or white-on-dark technical drawings. Intensity controls blend
    between original and inverted.

    Args:
        image: RGB numpy array
        intensity: 0.0 = original, 1.0 = fully inverted

    Returns:
        Inverted/blended image
    """
    inverted = 255 - image
    if intensity >= 0.99:
        return inverted
    return cv2.addWeighted(image, 1 - intensity, inverted, intensity, 0)


def remove_background_stains(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Reduce background stains, foxing, and age spots.

    Uses morphological operations to estimate the background,
    then divides it out. This is specifically designed for historical
    documents where uneven staining obscures text.

    Args:
        image: RGB numpy array
        intensity: 0.0 (no change) to 1.0 (aggressive stain removal)

    Returns:
        Stain-reduced image
    """
    if intensity <= 0.01:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Estimate background using large morphological closing
    kernel_size = int(30 + intensity * 70)  # 30-100 pixels
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Divide out the background (normalize illumination)
    normalized = cv2.divide(gray, background, scale=255)

    # Blend with original based on intensity
    gray_blended = cv2.addWeighted(gray, 1 - intensity, normalized, intensity, 0)

    # Apply to all channels proportionally
    result = image.copy().astype(np.float32)
    gray_f = gray.astype(np.float32)
    gray_f[gray_f == 0] = 1  # avoid division by zero

    scale = gray_blended.astype(np.float32) / gray_f
    scale = np.stack([scale] * 3, axis=-1)

    result = result * scale
    return cast(np.ndarray, np.clip(result, 0, 255).astype(np.uint8))


# =============================================================================
# Operation Registry
# =============================================================================
# Maps operation names (used in tool schemas) to their implementations.


def desaturate(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Reduce color saturation. Useful for removing tinted backgrounds
    (blue security paper, yellowed pages) to produce cleaner grayscale-like output.

    Args:
        image: RGB numpy array
        intensity: 0.0 (full color / no change) to 1.0 (fully desaturated / grayscale)

    Returns:
        Desaturated image (still RGB, but with reduced saturation)
    """
    if intensity <= 0.01:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.0 - intensity
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def threshold(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Apply binary threshold to convert image to black and white.
    Effective for removing watermarks, background patterns, and noise
    that falls below the threshold cutoff.

    The operation also logs an intensity distribution analysis (histogram
    percentiles) so the LLM can evaluate where content vs noise boundaries
    lie and refine the cutoff on subsequent iterations.

    Args:
        image: RGB numpy array
        intensity: 0.0 (low cutoff ~80, keeps more) to 1.0 (high cutoff ~240, aggressive)

    Returns:
        Thresholded image (binary black/white, returned as RGB)
    """
    # Map intensity to threshold value: 0.0 -> 80, 1.0 -> 240
    thresh_value = int(80 + intensity * 160)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute intensity distribution for LLM feedback
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_values = {f"p{p}": int(np.percentile(gray, p)) for p in percentiles}
    mean_val = int(np.mean(gray))
    std_val = int(np.std(gray))

    # Estimate what percentage of pixels will be black vs white at this cutoff
    pct_black = float(np.mean(gray < thresh_value) * 100)
    pct_white = 100.0 - pct_black

    # Store analysis on the function for retrieval by execute_operations
    threshold._last_analysis = {  # type: ignore[attr-defined]
        "cutoff": thresh_value,
        "mean": mean_val,
        "std": std_val,
        "distribution": pct_values,
        "result_pct_black": round(pct_black, 1),
        "result_pct_white": round(pct_white, 1),
        "hint": (
            f"At cutoff {thresh_value}: {pct_black:.1f}% black, {pct_white:.1f}% white. "
            f"Median pixel intensity is {pct_values['p50']}. "
            "Lower intensity = lower cutoff = keeps more content. "
            "Higher intensity = higher cutoff = more aggressive removal."
        ),
    }

    logger.info(
        "Threshold analysis: cutoff=%d, mean=%d, std=%d, distribution=%s",
        thresh_value,
        mean_val,
        std_val,
        pct_values,
    )

    _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

    # Convert back to RGB
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


OPERATIONS = {
    "contrast": adjust_contrast,
    "brightness": adjust_brightness,
    "sharpen": sharpen,
    "denoise": denoise,
    "deskew": deskew,
    "white_balance": white_balance,
    "equalize_histogram": equalize_histogram,
    "auto_crop": auto_crop,
    "invert": invert,
    "remove_stains": remove_background_stains,
    "desaturate": desaturate,
    "threshold": threshold,
}


def execute_operations(
    image: np.ndarray,
    operations: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[OperationResult]]:
    """
    Execute a sequence of operations as specified by the LLM.

    This is the main entry point called by the tool executor.
    The LLM sends a list of operations, each with:
        - op: operation name (key in OPERATIONS dict)
        - intensity: 0.0 to 1.0
        - region: optional {x1, y1, x2, y2} normalized coords

    Operations are applied sequentially (order matters!).

    Args:
        image: Input RGB numpy array
        operations: List of operation dicts from LLM tool call

    Returns:
        Tuple of (enhanced_image, list of OperationResult)
    """
    result = image.copy()
    log: List[OperationResult] = []

    for op_spec in operations:
        op_name = op_spec.get("op", "")
        intensity = float(op_spec.get("intensity", 0.5))
        intensity = max(0.0, min(1.0, intensity))  # clamp

        op_func = OPERATIONS.get(op_name)
        if op_func is None:
            logger.warning("Unknown operation: %s, skipping", op_name)
            continue

        # Parse region if provided
        region = None
        if "region" in op_spec and op_spec["region"] is not None:
            r = op_spec["region"]
            region = Region(
                x1=float(r.get("x1", 0)),
                y1=float(r.get("y1", 0)),
                x2=float(r.get("x2", 1)),
                y2=float(r.get("y2", 1)),
            )

        # Apply operation (region-aware)
        try:
            result = _apply_to_region(
                result, op_func, region=region, intensity=intensity
            )
            notes = f"Applied {op_name} at intensity {intensity:.2f}"

            # Enrich notes with threshold analysis if available
            if op_name == "threshold" and hasattr(threshold, "_last_analysis"):
                analysis = threshold._last_analysis  # type: ignore[attr-defined]
                notes = (
                    f"Applied threshold at intensity {intensity:.2f} "
                    f"(cutoff={analysis['cutoff']}). "
                    f"{analysis['hint']} "
                    f"Distribution: {analysis['distribution']}"
                )

            log.append(
                OperationResult(
                    operation=op_name,
                    parameters={"intensity": intensity},
                    region=region,
                    notes=notes,
                )
            )
            logger.info("Applied: %s (intensity=%.2f)", op_name, intensity)
        except Exception as e:  # noqa: E721  # type: ignore[attr-defined]
            logger.error("Operation %s failed: %s", op_name, e)
            log.append(
                OperationResult(
                    operation=op_name,
                    parameters={"intensity": intensity},
                    region=region,
                    notes=f"FAILED: {e}",
                )
            )

    return result, log


# =============================================================================
# Quick Image Analysis (for initial stats display)
# =============================================================================


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Compute basic image quality metrics.

    These are displayed in the notebook for reference, but the real
    assessment comes from the vision LLM looking at the actual image.
    These metrics are primarily useful for before/after comparison.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 50, 150)

    return {
        "dimensions": f"{w}x{h}",
        "contrast_std": float(gray.std()),
        "sharpness_laplacian": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_brightness": float(gray.mean()),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "edge_density": float(np.sum(edges > 0) / (h * w)),
        "yellowing_index": float((image[:, :, 0].mean() - image[:, :, 2].mean()) / 255),
    }
