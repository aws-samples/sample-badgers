"""
Image processing functionality for analyzer system.

Handles image conversion, optimization, and base64 encoding for different analyzer types.
"""

import base64
import hashlib
import io
import logging
import time
from pathlib import Path
from typing import Union, Optional
from PIL import Image


class ImageProcessingError(Exception):
    """Raised when image processing fails."""

    pass


class ImageProcessor:
    """Handles image conversion and optimization."""

    def __init__(
        self,
        max_image_size: int = 20971520,  # 20MB
        max_dimension: int = 2048,
        jpeg_quality: int = 85,
    ):
        """
        Initialize the image processor.

        Args:
            max_image_size: Maximum allowed image size in bytes
            max_dimension: Maximum allowed image dimension in pixels
            jpeg_quality: JPEG compression quality (1-100)
        """
        self.max_image_size = max_image_size
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.logger = logging.getLogger(__name__)

    def image_to_base64(self, image_path: Union[str, bytes, bytearray]) -> str:
        """
        Convert image to base64 string with enhanced error handling.

        Args:
            image_path: File path string or image byte data

        Returns:
            Base64 encoded image string

        Raises:
            ImageProcessingError: If image conversion fails
        """
        try:
            if isinstance(image_path, str):
                self.logger.debug("Converting image file to base64: %s", image_path)
                return self._file_to_base64(image_path)
            elif isinstance(image_path, (bytes, bytearray)):
                self.logger.debug(
                    "Converting byte data to base64, size: %d bytes", len(image_path)
                )
                return self._bytes_to_base64(image_path)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_path)}")
        except Exception as e:
            raise ImageProcessingError(f"Image conversion failed: {e}") from e

    def image_to_base64_with_bytes(
        self, image_path: Union[str, bytes, bytearray]
    ) -> tuple[str, bytes]:
        """
        Convert image to base64 string and also return the optimized image bytes.

        Args:
            image_path: File path string or image byte data

        Returns:
            Tuple of (base64 encoded string, optimized image bytes)

        Raises:
            ImageProcessingError: If image conversion fails
        """
        try:
            if isinstance(image_path, str):
                self.logger.debug(
                    "Converting image file to base64 with bytes: %s", image_path
                )
                return self._file_to_base64_with_bytes(image_path)
            elif isinstance(image_path, (bytes, bytearray)):
                self.logger.debug(
                    "Converting byte data to base64 with bytes, size: %d bytes",
                    len(image_path),
                )
                return self._bytes_to_base64_with_bytes(image_path)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_path)}")
        except Exception as e:
            raise ImageProcessingError(f"Image conversion failed: {e}") from e

    def get_image_dimensions(
        self, image_path: Union[str, bytes, bytearray]
    ) -> tuple[int, int]:
        """
        Get image dimensions (width, height) without full processing.

        Args:
            image_path: File path string or image byte data

        Returns:
            Tuple of (width, height) in pixels

        Raises:
            ImageProcessingError: If image cannot be read
        """
        try:
            if isinstance(image_path, str):
                with open(image_path, "rb") as f:
                    image_data = f.read()
            elif isinstance(image_path, (bytes, bytearray)):
                image_data = (
                    bytes(image_path)
                    if isinstance(image_path, bytearray)
                    else image_path
                )
            else:
                raise ValueError(f"Unsupported image input type: {type(image_path)}")

            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            self.logger.debug("Image dimensions: %dx%d", width, height)
            return width, height

        except Exception as e:
            raise ImageProcessingError(f"Failed to get image dimensions: {e}") from e

    def validate_image(self, image_data: bytes) -> bool:
        """
        Validate image data.

        Args:
            image_data: Raw image bytes

        Returns:
            True if image is valid

        Raises:
            ImageProcessingError: If image is invalid
        """
        if len(image_data) == 0:
            raise ImageProcessingError("Image data is empty")

        if len(image_data) > self.max_image_size:
            raise ImageProcessingError(
                f"Image data too large (>{self.max_image_size} bytes)"
            )

        try:
            # Try to open the image to validate format
            # Note: We don't call verify() here because it corrupts the image object
            # and subsequent operations on the same data will fail
            img = Image.open(io.BytesIO(image_data))
            # Just accessing format is enough to validate the image can be opened
            _ = img.format
            return True
        except Exception as e:
            raise ImageProcessingError(f"Invalid image data: {e}") from e

    def optimize_image(self, image_data: bytes) -> bytes:
        """
        Optimize image for processing (resize, convert format, compress).

        Args:
            image_data: Raw image bytes

        Returns:
            Optimized image bytes

        Raises:
            ImageProcessingError: If optimization fails
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            self.logger.debug(
                "Original image: %s, %s, %s", img.format, img.mode, img.size
            )

            # Convert to RGB if needed
            if img.mode in ["RGBA", "P"]:
                # Create white background for transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if too large
            if max(img.size) > self.max_dimension:
                original_size = img.size
                img.thumbnail(
                    (self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS
                )
                self.logger.debug(
                    "Resized image from %s to %s", original_size, img.size
                )

            # Convert to optimized JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
            optimized_data = buffer.getvalue()

            self.logger.debug(
                "Optimized image: %d -> %d bytes", len(image_data), len(optimized_data)
            )

            return optimized_data

        except Exception as e:
            raise ImageProcessingError(f"Failed to optimize image: {e}") from e

    def get_image_hash(self, image_data: bytes) -> str:
        """
        Generate hash for image data for caching purposes.

        Args:
            image_data: Raw image bytes

        Returns:
            MD5 hash string
        """
        return hashlib.md5(image_data, usedforsecurity=False).hexdigest()

    def _file_to_base64(self, image_path: str) -> str:
        """
        Convert image file to base64 string with validation.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string

        Raises:
            ImageProcessingError: If file processing fails
        """
        path = Path(image_path)

        # Validate file exists and is readable
        if not path.exists():
            raise ImageProcessingError(f"Image file not found: {image_path}")
        if not path.is_file():
            raise ImageProcessingError(f"Path is not a file: {image_path}")
        if path.stat().st_size == 0:
            raise ImageProcessingError(f"Image file is empty: {image_path}")
        if path.stat().st_size > self.max_image_size:
            raise ImageProcessingError(
                f"Image file too large (>{self.max_image_size} bytes): {image_path}"
            )

        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            return self._bytes_to_base64(image_data)
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to read image file {image_path}: {e}"
            ) from e

    def _bytes_to_base64(self, image_data: Union[bytes, bytearray]) -> str:
        """
        Convert image bytes to base64 with format validation and optimization.

        Args:
            image_data: Raw image bytes or bytearray

        Returns:
            Base64 encoded image string

        Raises:
            ImageProcessingError: If conversion fails
        """
        start_time = time.time()

        try:
            if isinstance(image_data, bytearray):
                image_data = bytes(image_data)

            # Validate image data
            self.validate_image(image_data)

            # Check if optimization is needed
            img = Image.open(io.BytesIO(image_data))
            needs_optimization = (
                img.format != "JPEG"
                or img.mode != "RGB"
                or max(img.size) > self.max_dimension
            )

            if needs_optimization:
                image_data = self.optimize_image(image_data)
                self.logger.debug("Image optimized for processing")

            # Encode to base64
            encoded = base64.b64encode(image_data).decode("utf-8")

            processing_time = time.time() - start_time
            self.logger.debug(
                "Image encoded: %d chars in %.2fs", len(encoded), processing_time
            )

            return encoded

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(f"Failed to encode image data: {e}") from e

    def _bytes_to_base64_with_bytes(
        self, image_data: Union[bytes, bytearray]
    ) -> tuple[str, bytes]:
        """
        Convert image bytes to base64 and return both base64 and optimized bytes.

        Args:
            image_data: Raw image bytes or bytearray

        Returns:
            Tuple of (base64 encoded string, optimized image bytes)

        Raises:
            ImageProcessingError: If conversion fails
        """
        start_time = time.time()

        try:
            if isinstance(image_data, bytearray):
                image_data = bytes(image_data)

            # Validate image data
            self.validate_image(image_data)

            # Check if optimization is needed
            img = Image.open(io.BytesIO(image_data))
            needs_optimization = (
                img.format != "JPEG"
                or img.mode != "RGB"
                or max(img.size) > self.max_dimension
            )

            if needs_optimization:
                image_data = self.optimize_image(image_data)
                self.logger.debug("Image optimized for processing")

            # Encode to base64
            encoded = base64.b64encode(image_data).decode("utf-8")

            processing_time = time.time() - start_time
            self.logger.debug(
                "Image encoded: %d chars in %.2fs", len(encoded), processing_time
            )

            return encoded, image_data

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(f"Failed to encode image data: {e}") from e

    def _file_to_base64_with_bytes(self, image_path: str) -> tuple[str, bytes]:
        """
        Convert image file to base64 and return both base64 and optimized bytes.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (base64 encoded string, optimized image bytes)

        Raises:
            ImageProcessingError: If file processing fails
        """
        path = Path(image_path)

        # Validate file exists and is readable
        if not path.exists():
            raise ImageProcessingError(f"Image file not found: {image_path}")
        if not path.is_file():
            raise ImageProcessingError(f"Path is not a file: {image_path}")
        if path.stat().st_size == 0:
            raise ImageProcessingError(f"Image file is empty: {image_path}")
        if path.stat().st_size > self.max_image_size:
            raise ImageProcessingError(
                f"Image file too large (>{self.max_image_size} bytes): {image_path}"
            )

        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            return self._bytes_to_base64_with_bytes(image_data)
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to read image file {image_path}: {e}"
            ) from e
