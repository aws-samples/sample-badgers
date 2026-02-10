"""
Message chain building functionality for analyzer system.

Handles construction of message chains for Bedrock invocation with examples and target images.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Union, Optional


class MessageChainError(Exception):
    """Raised when message chain building fails."""

    pass


class MessageChainBuilder:
    """Constructs message chains for Bedrock invocation."""

    def __init__(self):
        """Initialize the message chain builder."""
        self.logger = logging.getLogger(__name__)

    def create_message_chain(
        self,
        target_image: str,
        examples: List[str],
        analysis_text: str,
        max_examples: int,
    ) -> List[Dict[str, Any]]:
        """
        Create a complete message chain for Bedrock invocation.

        Args:
            target_image: Base64 encoded target image
            examples: List of base64 encoded example images
            analysis_text: Text describing what type of analysis to perform
            max_examples: Maximum number of examples to include

        Returns:
            List of message dictionaries for Bedrock

        Raises:
            MessageChainError: If message chain creation fails
        """
        try:
            messages = []

            # Add examples to the message chain
            if examples:
                limited_examples = examples[:max_examples]
                self.logger.info(
                    "Adding %d examples (limited from %d) to message chain",
                    len(limited_examples),
                    len(examples),
                )
                self.add_examples_to_chain(messages, limited_examples, analysis_text)
            else:
                self.logger.info("No examples provided for message chain")

            # Add the target image for analysis
            self.add_target_image(messages, target_image, analysis_text)

            self.logger.info("Created message chain with %d messages", len(messages))
            return messages

        except Exception as e:
            raise MessageChainError(f"Failed to create message chain: {e}") from e

    def add_examples_to_chain(
        self, messages: List[Dict[str, Any]], examples: List[str], analysis_text: str
    ) -> None:
        """
        Add example images to the message chain as few-shot examples.

        Args:
            messages: Message list to append to
            examples: List of base64 encoded example images
            analysis_text: Text describing the analysis type

        Raises:
            MessageChainError: If adding examples fails
        """
        try:
            for i, example_image in enumerate(examples):
                if not example_image or not example_image.strip():
                    self.logger.warning("Skipping empty example image %d", i)
                    continue

                # User message with example image
                user_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": example_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Please analyze the {analysis_text} in this image.",
                        },
                    ],
                }

                # Assistant response (placeholder for few-shot learning)
                assistant_message = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"I can see {analysis_text} in this image. Let me analyze it carefully and provide a detailed transcription.",
                        }
                    ],
                }

                messages.extend([user_message, assistant_message])
                self.logger.debug("Added example %d to message chain", i + 1)

        except Exception as e:
            raise MessageChainError(f"Failed to add examples to chain: {e}") from e

    def add_target_image(
        self, messages: List[Dict[str, Any]], image: str, analysis_text: str
    ) -> None:
        """
        Add the target image for analysis to the message chain.

        Args:
            messages: Message list to append to
            image: Base64 encoded target image
            analysis_text: Text describing the analysis type

        Raises:
            MessageChainError: If adding target image fails
        """
        try:
            if not image or not image.strip():
                raise MessageChainError("Target image cannot be empty")

            target_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Please analyze the {analysis_text} in this image and provide a detailed transcription.",
                    },
                ],
            }

            messages.append(target_message)
            self.logger.debug("Added target image to message chain")

        except Exception as e:
            raise MessageChainError(f"Failed to add target image to chain: {e}") from e

    def load_example_images(
        self, examples_path: str, max_examples: int, image_processor
    ) -> List[str]:
        """
        Load example images from directory and convert to base64.

        Args:
            examples_path: Path to directory containing example images
            max_examples: Maximum number of examples to load
            image_processor: ImageProcessor instance for conversion

        Returns:
            List of base64 encoded example images

        Raises:
            MessageChainError: If loading examples fails
        """
        try:
            examples_dir = Path(examples_path)
            if not examples_dir.exists():
                self.logger.warning("Examples directory not found: %s", examples_path)
                return []

            if not examples_dir.is_dir():
                self.logger.warning(
                    "Examples path is not a directory: %s", examples_path
                )
                return []

            # Find image files
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            image_files = []

            for file_path in examples_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)

            if not image_files:
                self.logger.warning("No image files found in: %s", examples_path)
                return []

            # Sort for consistent ordering
            image_files.sort()

            # Limit to max_examples
            image_files = image_files[:max_examples]

            # Convert to base64
            examples = []
            for image_file in image_files:
                try:
                    base64_image = image_processor.image_to_base64(str(image_file))
                    examples.append(base64_image)
                    self.logger.debug("Loaded example image: %s", image_file.name)
                except Exception as e:
                    self.logger.warning(
                        "Failed to load example image %s: %s", image_file, e
                    )
                    continue

            self.logger.info(
                "Loaded %d example images from %s", len(examples), examples_path
            )
            return examples

        except Exception as e:
            raise MessageChainError(f"Failed to load example images: {e}") from e

    def validate_message_chain(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Validate that a message chain is properly formatted.

        Args:
            messages: List of message dictionaries to validate

        Returns:
            True if message chain is valid

        Raises:
            MessageChainError: If message chain is invalid
        """
        if not messages:
            raise MessageChainError("Message chain cannot be empty")

        if not isinstance(messages, list):
            raise MessageChainError("Messages must be a list")

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise MessageChainError(f"Message {i} must be a dictionary")

            if "role" not in message:
                raise MessageChainError(f"Message {i} missing 'role' field")

            if "content" not in message:
                raise MessageChainError(f"Message {i} missing 'content' field")

            if message["role"] not in ["user", "assistant"]:
                raise MessageChainError(
                    f"Message {i} has invalid role: {message['role']}"
                )

            if not isinstance(message["content"], list):
                raise MessageChainError(f"Message {i} content must be a list")

            # Validate content structure
            for j, content_item in enumerate(message["content"]):
                if not isinstance(content_item, dict):
                    raise MessageChainError(
                        f"Message {i} content item {j} must be a dictionary"
                    )

                if "type" not in content_item:
                    raise MessageChainError(
                        f"Message {i} content item {j} missing 'type' field"
                    )

                if content_item["type"] not in ["text", "image"]:
                    raise MessageChainError(
                        f"Message {i} content item {j} has invalid type: {content_item['type']}"
                    )

        return True
