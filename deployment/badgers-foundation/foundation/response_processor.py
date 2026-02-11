"""
Response processing functionality for analyzer system.

Handles extraction and validation of analysis results from Bedrock responses.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Union


class ResponseProcessingError(Exception):
    """Raised when response processing fails."""

    pass


class ResponseProcessor:
    """Processes and validates Bedrock responses."""

    def __init__(self):
        """Initialize the response processor."""
        self.logger = logging.getLogger(__name__)

    def extract_analysis_result(self, response: Dict[str, Any]) -> str:
        """
        Extract analysis result from Bedrock response.

        Args:
            response: Response dictionary from Bedrock

        Returns:
            Extracted analysis result as string

        Raises:
            ResponseProcessingError: If result extraction fails
        """
        try:
            # Validate response structure
            if not self.validate_response(response):
                raise ResponseProcessingError("Invalid response structure")

            # Extract content from response
            content = response.get("content", [])
            if not content:
                raise ResponseProcessingError("Response contains no content")

            # Find text content
            result_text = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text.strip():
                        result_text += text + "\n"

            if not result_text.strip():
                raise ResponseProcessingError("No text content found in response")

            # Clean and format the result
            cleaned_result = self._clean_response_text(result_text.strip())

            self.logger.info(
                "Extracted analysis result: %d characters", len(cleaned_result)
            )
            return cleaned_result

        except Exception as e:
            if isinstance(e, ResponseProcessingError):
                raise
            raise ResponseProcessingError(
                f"Failed to extract analysis result: {e}"
            ) from e

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate Bedrock response structure.

        Args:
            response: Response dictionary to validate

        Returns:
            True if response is valid

        Raises:
            ResponseProcessingError: If response is invalid
        """
        if not isinstance(response, dict):
            raise ResponseProcessingError("Response must be a dictionary")

        if "content" not in response:
            raise ResponseProcessingError("Response missing 'content' field")

        content = response["content"]
        if not isinstance(content, list):
            raise ResponseProcessingError("Response content must be a list")

        if not content:
            raise ResponseProcessingError("Response content is empty")

        # Validate content structure
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                raise ResponseProcessingError(f"Content item {i} must be a dictionary")

            if "type" not in item:
                raise ResponseProcessingError(f"Content item {i} missing 'type' field")

            if item["type"] == "text" and "text" not in item:
                raise ResponseProcessingError(
                    f"Text content item {i} missing 'text' field"
                )

        return True

    def handle_empty_response(self) -> str:
        """
        Handle cases where Bedrock returns an empty or invalid response.

        Returns:
            Default message for empty responses
        """
        self.logger.warning("Received empty response from Bedrock")
        return "No analysis result could be generated from the provided image."

    def extract_structured_data(
        self, response_text: str, expected_format: Optional[str] = None
    ) -> Union[Dict[str, Any], str]:
        """
        Extract structured data from response text if present.

        Args:
            response_text: Raw response text
            expected_format: Expected format ('json', 'xml', etc.)

        Returns:
            Structured data if found, otherwise original text

        Raises:
            ResponseProcessingError: If structured data extraction fails
        """
        try:
            # Try to extract JSON if present
            if expected_format == "json" or self._contains_json(response_text):
                json_data = self._extract_json(response_text)
                if json_data:
                    self.logger.debug("Extracted JSON data from response")
                    return json_data

            # Try to extract XML if present
            if expected_format == "xml" or self._contains_xml(response_text):
                xml_data = self._extract_xml_content(response_text)
                if xml_data:
                    self.logger.debug("Extracted XML content from response")
                    return xml_data

            # Return original text if no structured data found
            return response_text

        except Exception as e:
            self.logger.warning("Failed to extract structured data: %s", e)
            return response_text

    def _clean_response_text(self, text: str) -> str:
        """
        Clean and format response text.

        Args:
            text: Raw response text

        Returns:
            Cleaned response text
        """
        # Strip markdown code fences (```xml, ```json, etc.)
        text = re.sub(
            r"^```(?:xml|json|html|text)?\s*\n?", "", text, flags=re.IGNORECASE
        )
        text = re.sub(r"\n?```\s*$", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Remove common artifacts
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Trim leading/trailing whitespace
        text = text.strip()

        return text

    def _contains_json(self, text: str) -> bool:
        """Check if text contains JSON data."""
        return "{" in text and "}" in text

    def _contains_xml(self, text: str) -> bool:
        """Check if text contains XML data."""
        return "<" in text and ">" in text

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON data from text.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON data or None if not found
        """
        try:
            # Try to find JSON blocks
            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            matches = re.findall(json_pattern, text, re.DOTALL)

            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

            # Try parsing the entire text as JSON
            return json.loads(text)

        except Exception:
            return None

    def _extract_xml_content(self, text: str) -> Optional[str]:
        """
        Extract XML content from text.

        Args:
            text: Text potentially containing XML

        Returns:
            XML content or None if not found
        """
        try:
            # Look for XML tags
            xml_pattern = r"<[^>]+>.*?</[^>]+>"
            matches = re.findall(xml_pattern, text, re.DOTALL)

            if matches:
                return "\n".join(matches)

            return None

        except Exception:
            return None

    def format_analysis_result(
        self, result: str, analyzer_type: str, include_metadata: bool = False
    ) -> str:
        """
        Format analysis result with optional metadata.

        Args:
            result: Raw analysis result
            analyzer_type: Type of analyzer that produced the result
            include_metadata: Whether to include metadata in output

        Returns:
            Formatted analysis result
        """
        try:
            if include_metadata:
                formatted_result = f"Analysis Type: {analyzer_type}\n"
                formatted_result += f"Result Length: {len(result)} characters\n"
                formatted_result += "-" * 50 + "\n"
                formatted_result += result
            else:
                formatted_result = result

            return formatted_result

        except Exception as e:
            self.logger.warning("Failed to format analysis result: %s", e)
            return result

    def validate_analysis_quality(self, result: str) -> Dict[str, Any]:
        """
        Validate the quality of analysis result.

        Args:
            result: Analysis result to validate

        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics = {
                "length": len(result),
                "has_content": bool(result.strip()),
                "word_count": len(result.split()) if result else 0,
                "line_count": len(result.splitlines()) if result else 0,
                "contains_structured_data": self._contains_json(result)
                or self._contains_xml(result),
                "quality_score": 0.0,
            }

            # Calculate quality score
            score = 0.0
            if quality_metrics["has_content"]:
                score += 0.3
            if quality_metrics["word_count"] > 10:
                score += 0.3
            if quality_metrics["length"] > 50:
                score += 0.2
            if quality_metrics["line_count"] > 1:
                score += 0.1
            if quality_metrics["contains_structured_data"]:
                score += 0.1

            quality_metrics["quality_score"] = min(score, 1.0)

            return quality_metrics

        except Exception as e:
            self.logger.warning("Failed to validate analysis quality: %s", e)
            return {
                "length": 0,
                "has_content": False,
                "word_count": 0,
                "line_count": 0,
                "contains_structured_data": False,
                "quality_score": 0.0,
            }
