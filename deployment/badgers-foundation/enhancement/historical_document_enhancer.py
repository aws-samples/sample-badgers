"""
Historical Document Enhancer for Vision LLM Processing
=======================================================

A comprehensive image enhancement pipeline designed to maximize
extraction quality from historical documents when processed by
vision-enabled LLMs (Claude, GPT-4V, Gemini).

Supports multiple document types:
- Aged manuscripts and parchments
- Annotated documents with handwritten marginalia
- Sheet music with performance annotations
- Technical diagrams with text
- Multi-layer documents (printed + handwritten overlays)

Author: BADGERS Document Processing Pipeline
"""

import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification for preprocessing strategy selection."""

    MANUSCRIPT = auto()  # Handwritten historical documents
    PRINTED_HISTORICAL = auto()  # Old printed books/documents
    ANNOTATED = auto()  # Documents with handwritten annotations over print
    SHEET_MUSIC = auto()  # Musical scores (possibly annotated)
    TECHNICAL_DIAGRAM = auto()  # Diagrams, charts, technical illustrations
    MIXED_MEDIA = auto()  # Complex documents with multiple content types
    UNKNOWN = auto()  # Default/fallback


class EnhancementLevel(Enum):
    """Enhancement intensity levels."""

    MINIMAL = auto()  # Light touch - preserve original character
    MODERATE = auto()  # Balanced enhancement
    AGGRESSIVE = auto()  # Maximum enhancement for severely degraded docs


@dataclass
class EnhancementConfig:
    """Configuration for document enhancement pipeline."""

    # Resolution settings
    target_min_dimension: int = 2000
    target_max_dimension: int = 4000
    target_dpi: int = 300

    # Contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # Denoising
    denoise_strength: int = 10
    denoise_color_strength: int = 10
    denoise_template_window: int = 7
    denoise_search_window: int = 21

    # Sharpening
    sharpen_amount: float = 1.5
    sharpen_kernel_size: int = 3

    # Binarization (adaptive thresholding)
    binary_block_size: int = 11
    binary_constant: int = 2

    # Deskew
    max_skew_angle: float = 10.0
    min_skew_correction: float = 0.5

    # Color annotation preservation
    preserve_annotations: bool = True
    annotation_colors: List[str] = field(
        default_factory=lambda: ["red", "blue", "green"]
    )


@dataclass
class EnhancementResult:
    """Container for enhancement pipeline results."""

    enhanced_image: np.ndarray
    original_shape: Tuple[int, int, int]
    final_shape: Tuple[int, int, int]
    operations_applied: List[str]
    metadata: Dict[str, Any]
    annotation_mask: Optional[np.ndarray] = None
    skew_angle: Optional[float] = None

    def save(self, output_path: Union[str, Path], format: str = "PNG") -> Path:
        """Save enhanced image to file."""
        output_path = Path(output_path)

        # Convert RGB to BGR for OpenCV saving
        if len(self.enhanced_image.shape) == 3:
            save_img = cv2.cvtColor(self.enhanced_image, cv2.COLOR_RGB2BGR)
        else:
            save_img = self.enhanced_image

        cv2.imwrite(str(output_path), save_img)
        logger.info(f"Saved enhanced image to {output_path}")
        return output_path


class HistoricalDocumentEnhancer:
    """
    Main class for enhancing historical documents for vision LLM processing.

    Provides document-type-aware preprocessing pipelines that balance
    enhancement with preservation of original document characteristics.

    Example Usage:
    --------------
    ```python
    enhancer = HistoricalDocumentEnhancer()

    # Auto-detect document type and enhance
    result = enhancer.enhance('old_manuscript.jpg')
    result.save('enhanced_manuscript.png')

    # Specify document type for optimal processing
    result = enhancer.enhance(
        'annotated_score.png',
        document_type=DocumentType.SHEET_MUSIC,
        level=EnhancementLevel.MODERATE
    )

    # Access annotation mask for separate processing
    if result.annotation_mask is not None:
        # Process colored annotations separately
        pass
    ```
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        """
        Initialize the enhancer with optional custom configuration.

        Args:
            config: EnhancementConfig instance. Uses defaults if None.
        """
        self.config = config or EnhancementConfig()
        self._operations_log: List[str] = []

    def enhance(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        document_type: DocumentType = DocumentType.UNKNOWN,
        level: EnhancementLevel = EnhancementLevel.MODERATE,
        preserve_color: bool = True,
    ) -> EnhancementResult:
        """
        Enhance a historical document image for optimal vision LLM processing.

        Args:
            image: Input image (path, numpy array, or PIL Image)
            document_type: Type of document for strategy selection
            level: Enhancement intensity level
            preserve_color: If True, maintain color information

        Returns:
            EnhancementResult containing enhanced image and metadata
        """
        self._operations_log = []

        # Load and normalize image
        img = self._load_image(image)
        original_shape = img.shape

        # Select enhancement strategy based on document type
        strategy = self._select_strategy(document_type, level)

        # Execute enhancement pipeline
        enhanced, metadata = self._execute_pipeline(img, strategy, preserve_color)

        # Extract annotation mask if applicable
        annotation_mask = None
        if self.config.preserve_annotations and document_type in [
            DocumentType.ANNOTATED,
            DocumentType.SHEET_MUSIC,
            DocumentType.MIXED_MEDIA,
        ]:
            annotation_mask = self._extract_annotation_mask(img)

        return EnhancementResult(
            enhanced_image=enhanced,
            original_shape=original_shape,
            final_shape=enhanced.shape,
            operations_applied=self._operations_log.copy(),
            metadata=metadata,
            annotation_mask=annotation_mask,
            skew_angle=metadata.get("skew_angle"),
        )

    def _load_image(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Load image from various sources and normalize to RGB numpy array."""

        if isinstance(image, (str, Path)):
            image_path = str(image)

            # Handle .b64 files (base64-encoded images from pdf-to-images-converter)
            if image_path.endswith(".b64"):
                with open(image_path, "r") as f:
                    b64_content = f.read()
                image_bytes = base64.b64decode(b64_content)
                img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Could not decode base64 image from: {image}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._operations_log.append(f"Loaded base64 image from {image}")
            else:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image from path: {image}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._operations_log.append(f"Loaded image from {image}")

        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
            self._operations_log.append("Loaded from PIL Image")

        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                img = image.copy()
            self._operations_log.append("Loaded from numpy array")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return img

    def _select_strategy(
        self, document_type: DocumentType, level: EnhancementLevel
    ) -> Dict[str, Any]:
        """Select enhancement strategy based on document type and level."""

        # Base strategy
        strategy = {
            "upscale": True,
            "deskew": True,
            "denoise": True,
            "contrast_enhance": True,
            "sharpen": False,
            "binarize": False,
            "color_balance": False,
            "remove_background": False,
        }

        # Adjust for document type
        if document_type == DocumentType.MANUSCRIPT:
            strategy["sharpen"] = True
            strategy["color_balance"] = True

        elif document_type == DocumentType.PRINTED_HISTORICAL:
            strategy["sharpen"] = True
            if level == EnhancementLevel.AGGRESSIVE:
                strategy["binarize"] = True

        elif document_type == DocumentType.ANNOTATED:
            # Preserve color annotations - minimal processing
            strategy["sharpen"] = False
            strategy["binarize"] = False
            strategy["denoise"] = level != EnhancementLevel.MINIMAL

        elif document_type == DocumentType.SHEET_MUSIC:
            # Preserve both print and colored annotations
            strategy["contrast_enhance"] = True
            strategy["binarize"] = False

        elif document_type == DocumentType.TECHNICAL_DIAGRAM:
            strategy["sharpen"] = True
            strategy["contrast_enhance"] = True

        elif document_type == DocumentType.MIXED_MEDIA:
            # Conservative approach for complex documents
            strategy["sharpen"] = level == EnhancementLevel.AGGRESSIVE

        # Adjust for enhancement level
        if level == EnhancementLevel.MINIMAL:
            strategy["denoise"] = False
            strategy["sharpen"] = False

        elif level == EnhancementLevel.AGGRESSIVE:
            strategy["sharpen"] = True
            strategy["denoise"] = True

        return strategy

    def _execute_pipeline(
        self, img: np.ndarray, strategy: Dict[str, Any], preserve_color: bool
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute the enhancement pipeline based on strategy."""

        metadata = {}

        # 1. Upscale if needed
        if strategy["upscale"]:
            img = self._upscale(img)

        # 2. Deskew
        if strategy["deskew"]:
            img, angle = self._deskew(img)
            metadata["skew_angle"] = angle

        # 3. Denoise
        if strategy["denoise"]:
            img = self._denoise(img)

        # 4. Contrast enhancement
        if strategy["contrast_enhance"]:
            img = self._enhance_contrast(img)

        # 5. Color balance
        if strategy["color_balance"]:
            img = self._balance_colors(img)

        # 6. Sharpen
        if strategy["sharpen"]:
            img = self._sharpen(img)

        # 7. Binarize (optional, usually avoided for LLM processing)
        if strategy["binarize"] and not preserve_color:
            img = self._binarize(img)

        return img, metadata

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale image to target resolution for better LLM processing."""

        h, w = img.shape[:2]
        min_dim = min(h, w)
        max_dim = max(h, w)

        # Check if upscaling needed
        if min_dim >= self.config.target_min_dimension:
            self._operations_log.append(f"Upscale skipped (already {w}x{h})")
            return img

        # Calculate scale factor
        scale = self.config.target_min_dimension / min_dim

        # Limit maximum size
        if max_dim * scale > self.config.target_max_dimension:
            scale = self.config.target_max_dimension / max_dim

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Use INTER_CUBIC for upscaling (better for documents)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        self._operations_log.append(f"Upscaled from {w}x{h} to {new_w}x{new_h}")
        return img

    def _deskew(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct document skew."""

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_not(gray)

        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(gray > 0))

        if len(coords) < 100:
            self._operations_log.append("Deskew skipped (insufficient content)")
            return img, 0.0

        # Get rotation angle from minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]

        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only correct significant skew within bounds
        if abs(angle) < self.config.min_skew_correction:
            self._operations_log.append(
                f"Deskew skipped (angle {angle:.2f}° below threshold)"
            )
            return img, angle

        if abs(angle) > self.config.max_skew_angle:
            self._operations_log.append(
                f"Deskew skipped (angle {angle:.2f}° exceeds maximum)"
            )
            return img, angle

        # Perform rotation
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        img = cv2.warpAffine(
            img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        self._operations_log.append(f"Deskewed by {angle:.2f}°")
        return img, angle

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text edges."""

        img = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            self.config.denoise_strength,
            self.config.denoise_color_strength,
            self.config.denoise_template_window,
            self.config.denoise_search_window,
        )

        self._operations_log.append("Applied denoising")
        return img

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement."""

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        l_channel = clahe.apply(l_channel)

        # Merge and convert back
        lab = cv2.merge([l_channel, a_channel, b_channel])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        self._operations_log.append("Applied CLAHE contrast enhancement")
        return img

    def _balance_colors(self, img: np.ndarray) -> np.ndarray:
        """Balance color channels to reduce yellowing/aging artifacts."""

        # Simple white balance using gray world assumption
        result = img.copy().astype(np.float32)

        avg_r = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        avg_gray = (avg_r + avg_g + avg_b) / 3

        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_r), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_b), 0, 255)

        self._operations_log.append("Applied color balance")
        return result.astype(np.uint8)

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Apply unsharp mask sharpening."""

        # Create blurred version
        kernel_size = self.config.sharpen_kernel_size
        blurred = cv2.GaussianBlur(img, (0, 0), kernel_size)

        # Unsharp mask: original + (original - blurred) * amount
        amount = self.config.sharpen_amount
        img = cv2.addWeighted(img, amount, blurred, 1 - amount, 0)

        self._operations_log.append("Applied sharpening")
        return img

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """Convert to binary using adaptive thresholding."""

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.binary_block_size,
            self.config.binary_constant,
        )

        # Convert back to RGB
        img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        self._operations_log.append("Applied adaptive binarization")
        return img

    def _extract_annotation_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Extract mask of colored annotations (red, blue, green marks).

        Useful for separate processing of handwritten annotations
        on historical documents.
        """

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Define color ranges for common annotation colors
        color_ranges = {
            "red": ([0, 50, 50], [10, 255, 255]),
            "red2": ([170, 50, 50], [180, 255, 255]),  # Red wraps around
            "blue": ([100, 50, 50], [130, 255, 255]),
            "green": ([40, 50, 50], [80, 255, 255]),
        }

        # Create combined mask
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        self._operations_log.append("Extracted annotation mask")
        return combined_mask

    # =========================================================================
    # Document-Type-Specific Enhancement Methods
    # =========================================================================

    def enhance_manuscript(
        self,
        image: Union[str, Path, np.ndarray],
        level: EnhancementLevel = EnhancementLevel.MODERATE,
    ) -> EnhancementResult:
        """
        Optimize enhancement for handwritten manuscripts.

        Strategy:
        - Preserve ink variations (important for paleography)
        - Enhance contrast for faded text
        - Reduce paper grain/foxing noise
        - NO binarization (preserves stroke characteristics)
        """
        return self.enhance(image, DocumentType.MANUSCRIPT, level)

    def enhance_annotated_document(
        self,
        image: Union[str, Path, np.ndarray],
        level: EnhancementLevel = EnhancementLevel.MODERATE,
    ) -> EnhancementResult:
        """
        Optimize enhancement for documents with handwritten annotations.

        Strategy:
        - Preserve colored annotations (pencil, ink marks)
        - Balance print visibility with annotation preservation
        - Extract annotation mask for separate analysis
        """
        return self.enhance(image, DocumentType.ANNOTATED, level)

    def enhance_sheet_music(
        self,
        image: Union[str, Path, np.ndarray],
        level: EnhancementLevel = EnhancementLevel.MODERATE,
    ) -> EnhancementResult:
        """
        Optimize enhancement for musical scores.

        Strategy:
        - Preserve musical notation clarity
        - Maintain colored performance annotations
        - Enhance dynamics markings and text
        """
        return self.enhance(image, DocumentType.SHEET_MUSIC, level)

    def enhance_technical_diagram(
        self,
        image: Union[str, Path, np.ndarray],
        level: EnhancementLevel = EnhancementLevel.MODERATE,
    ) -> EnhancementResult:
        """
        Optimize enhancement for technical illustrations and diagrams.

        Strategy:
        - Sharpen line work
        - Enhance labels and captions
        - Maintain diagram-text relationship
        """
        return self.enhance(image, DocumentType.TECHNICAL_DIAGRAM, level)

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def enhance_batch(
        self,
        images: List[Union[str, Path]],
        document_type: DocumentType = DocumentType.UNKNOWN,
        level: EnhancementLevel = EnhancementLevel.MODERATE,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[EnhancementResult]:
        """
        Process multiple images with the same settings.

        Args:
            images: List of image paths
            document_type: Document type for all images
            level: Enhancement level
            output_dir: If provided, save enhanced images here

        Returns:
            List of EnhancementResult objects
        """
        results = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(images):
            logger.info(f"Processing {i+1}/{len(images)}: {img_path}")

            try:
                result = self.enhance(img_path, document_type, level)

                if output_dir:
                    input_path = Path(img_path)
                    output_path = output_dir / f"{input_path.stem}_enhanced.png"
                    result.save(output_path)

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                continue

        return results

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def analyze_image(self, image: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze image characteristics to help select optimal enhancement strategy.

        Returns metrics useful for determining document type and enhancement level.
        """
        img = self._load_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Basic stats
        h, w = img.shape[:2]

        # Contrast analysis
        contrast = gray.std()

        # Noise estimation (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean()

        # Text density estimation (edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)

        # Yellowing detection (compare R and B channels)
        r_mean = img[:, :, 0].mean()
        b_mean = img[:, :, 2].mean()
        yellowing_index = (r_mean - b_mean) / 255

        analysis = {
            "dimensions": (w, h),
            "aspect_ratio": w / h,
            "contrast": float(contrast),
            "sharpness": float(laplacian_var),
            "saturation": float(saturation),
            "edge_density": float(edge_density),
            "yellowing_index": float(yellowing_index),
            "needs_upscale": min(h, w) < self.config.target_min_dimension,
            "likely_degraded": contrast < 40 or laplacian_var < 100,
            "has_color_annotations": saturation > 30,
        }

        # Suggest document type based on analysis
        if analysis["has_color_annotations"] and edge_density > 0.1:
            analysis["suggested_type"] = DocumentType.ANNOTATED
        elif edge_density > 0.15:
            analysis["suggested_type"] = DocumentType.TECHNICAL_DIAGRAM
        elif yellowing_index > 0.1:
            analysis["suggested_type"] = DocumentType.MANUSCRIPT
        else:
            analysis["suggested_type"] = DocumentType.PRINTED_HISTORICAL

        return analysis

    def compare_enhancement(
        self, image: Union[str, Path, np.ndarray], levels: List[EnhancementLevel] = None
    ) -> Dict[str, EnhancementResult]:
        """
        Compare different enhancement levels on the same image.

        Useful for finding optimal settings for a document collection.
        """
        if levels is None:
            levels = [
                EnhancementLevel.MINIMAL,
                EnhancementLevel.MODERATE,
                EnhancementLevel.AGGRESSIVE,
            ]

        results = {}
        for level in levels:
            results[level.name] = self.enhance(image, level=level)

        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def enhance_document(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    document_type: str = "auto",
) -> EnhancementResult:
    """
    Quick enhancement function for single documents.

    Args:
        image_path: Path to input image
        output_path: Path for output (optional)
        document_type: One of 'auto', 'manuscript', 'annotated',
                       'sheet_music', 'diagram', 'printed'

    Returns:
        EnhancementResult
    """
    enhancer = HistoricalDocumentEnhancer()

    type_map = {
        "auto": DocumentType.UNKNOWN,
        "manuscript": DocumentType.MANUSCRIPT,
        "annotated": DocumentType.ANNOTATED,
        "sheet_music": DocumentType.SHEET_MUSIC,
        "diagram": DocumentType.TECHNICAL_DIAGRAM,
        "printed": DocumentType.PRINTED_HISTORICAL,
    }

    doc_type = type_map.get(document_type.lower(), DocumentType.UNKNOWN)

    result = enhancer.enhance(image_path, document_type=doc_type)

    if output_path:
        result.save(output_path)

    return result


def prepare_for_vision_llm(
    image_path: Union[str, Path], max_dimension: int = 4000
) -> np.ndarray:
    """
    Prepare image specifically for vision LLM API consumption.

    Applies moderate enhancement and ensures optimal resolution
    for Claude, GPT-4V, or Gemini vision processing.

    Args:
        image_path: Path to input image
        max_dimension: Maximum dimension (to control token usage)

    Returns:
        Enhanced image as numpy array (RGB)
    """
    config = EnhancementConfig(
        target_min_dimension=2000, target_max_dimension=max_dimension
    )

    enhancer = HistoricalDocumentEnhancer(config)
    result = enhancer.enhance(image_path, level=EnhancementLevel.MODERATE)

    return result.enhanced_image


# =============================================================================
# Example Usage / Main
# =============================================================================

if __name__ == "__main__":
    import sys

    # Example usage
    print("Historical Document Enhancer")
    print("=" * 50)

    # Create enhancer
    enhancer = HistoricalDocumentEnhancer()

    # Example: Analyze an image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        print(f"\nAnalyzing: {image_path}")
        analysis = enhancer.analyze_image(image_path)

        print("\nImage Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

        print(f"\nSuggested document type: {analysis['suggested_type'].name}")

        # Enhance with suggested settings
        result = enhancer.enhance(image_path, document_type=analysis["suggested_type"])

        print("\nOperations applied:")
        for op in result.operations_applied:
            print(f"  - {op}")

        # Save result
        output_path = Path(image_path).stem + "_enhanced.png"
        result.save(output_path)
        print(f"\nEnhanced image saved to: {output_path}")

    else:
        print("\nUsage: python historical_document_enhancer.py <image_path>")
        print("\nExample with code:")
        print(
            """
    from historical_document_enhancer import HistoricalDocumentEnhancer, DocumentType

    enhancer = HistoricalDocumentEnhancer()

    # For the Constitution page
    result = enhancer.enhance_manuscript('constitution-page4.jpg')
    result.save('constitution-enhanced.png')

    # For annotated sheet music
    result = enhancer.enhance_sheet_music('music-score.png')
    result.save('music-enhanced.png')

    # For Washington's annotated draft
    result = enhancer.enhance_annotated_document('washington-draft.png')
    result.save('draft-enhanced.png')
    # Access annotation mask for separate analysis
    if result.annotation_mask is not None:
        cv2.imwrite('annotations-mask.png', result.annotation_mask)
        """
        )
