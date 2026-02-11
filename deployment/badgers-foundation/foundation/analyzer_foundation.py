"""
Core analyzer foundation class that orchestrates all foundation components.

This is the main class that analyzer tools will use to perform analysis operations.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

from .configuration_manager import ConfigurationManager
from .prompt_loader import PromptLoader
from .image_processor import ImageProcessor
from .bedrock_client import BedrockClient
from .message_chain_builder import MessageChainBuilder
from .response_processor import ResponseProcessor


def _get_analyzer_config_path() -> str:
    """Get ANALYZER_CONFIG_PATH by temporarily adding config module to path."""
    config_module_path = Path(__file__).parent.parent
    sys.path.insert(0, str(config_module_path))
    try:
        from config.config import (  # pylint: disable=import-outside-toplevel
            ANALYZER_CONFIG_PATH as config_path,
        )

        return config_path
    finally:
        sys.path.pop(0)


ANALYZER_CONFIG_PATH = _get_analyzer_config_path()


class AnalysisError(Exception):
    """Raised when analysis operations fail."""


class AnalyzerFoundation:
    """Core analyzer functionality that all specific analyzers inherit from."""

    def __init__(
        self,
        analyzer_type: str,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the analyzer foundation.

        Args:
            analyzer_type: Type of analyzer (e.g., 'diagram', 'table')
            config_path: Optional path to analyzer config file. If None, tries local manifest first, then falls back to ANALYZER_CONFIG_PATH

        Raises:
            AnalysisError: If initialization fails
        """
        try:
            self.analyzer_type = analyzer_type
            self.logger = logging.getLogger(f"{__name__}.{analyzer_type}")

            # Initialize foundation components
            self.config_manager = ConfigurationManager()
            self.prompt_loader = PromptLoader()
            self.image_processor = ImageProcessor()
            self.bedrock_client = BedrockClient()
            self.message_builder = MessageChainBuilder()
            self.response_processor = ResponseProcessor()

            # Try to load from local manifest first, then fall back to central config
            self.config, self.global_settings = self._load_configuration(
                analyzer_type, config_path
            )

            # Configure components with global settings
            self._configure_components()

            self.logger.info(
                "Initialized %s analyzer with model %s",
                analyzer_type,
                self.config.get("model_id", "unknown"),
            )

        except Exception as e:
            raise AnalysisError(
                f"Failed to initialize {analyzer_type} analyzer: {e}"
            ) from e

    def analyze(
        self,
        image_path: Union[str, bytes, bytearray],
        aws_profile: Optional[str] = None,
        audit_mode: bool = False,
    ) -> Union[str, tuple[str, Optional[str]]]:
        """
        Main analysis method that orchestrates the entire analysis workflow.

        Args:
            image_path: File path string or image byte data
            aws_profile: Optional AWS profile name for Bedrock client
            audit_mode: Whether to include confidence assessment in output

        Returns:
            Analysis result as string, or tuple of (result, thinking_content) if extended thinking enabled

        Raises:
            AnalysisError: If analysis fails
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Starting %s analysis (audit_mode=%s)", self.analyzer_type, audit_mode
            )

            # Step 1: Process the target image (this may resize it)
            target_image_b64, optimized_image_data = self._process_target_image(
                image_path
            )

            # Step 2: Get image dimensions from the optimized image
            image_dimensions = self._get_image_dimensions_from_bytes(
                optimized_image_data
            )

            # Step 3: Load system prompt with dimension placeholders and audit mode
            system_prompt = self._load_system_prompt(image_dimensions, audit_mode)

            # Step 4: Load example images
            example_images = self._load_example_images()

            # Step 5: Build message chain
            messages = self._build_message_chain(target_image_b64, example_images)

            # Step 6: Invoke Bedrock model
            response = self._invoke_bedrock_model(system_prompt, messages, aws_profile)

            # Step 7: Extract thinking content if present
            thinking_content = response.get("thinking")
            if thinking_content:
                self.logger.info(
                    "Extended thinking captured: %d characters", len(thinking_content)
                )

            # Step 8: Process response
            result = self._process_response(response)

            # Step 9: Validate and format result
            final_result = self._finalize_result(result)

            processing_time = time.time() - start_time
            self.logger.info(
                "Analysis completed in %.2fs, result: %d characters",
                processing_time,
                len(final_result),
            )

            # Return tuple if thinking content present, otherwise just result
            if thinking_content:
                return final_result, thinking_content
            return final_result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                "Analysis failed after %.2fs: %s", processing_time, str(e)
            )
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(f"Analysis failed: {e}") from e

    def get_analyzer_info(self) -> Dict[str, Any]:
        """
        Get information about this analyzer instance.

        Returns:
            Dictionary with analyzer information
        """
        return {
            "analyzer_type": self.analyzer_type,
            "name": self.config.get("name", "unknown"),
            "description": self.config.get("description", ""),
            "model_id": self.config.get("model_id", "unknown"),
            "max_examples": self.config.get("max_examples", 0),
            "examples_path": self.config.get("examples_path", ""),
            "prompt_files": self.config.get("prompt_files", []),
        }

    def validate_configuration(self) -> bool:
        """
        Validate that the analyzer configuration is complete and valid.

        Returns:
            True if configuration is valid

        Raises:
            AnalysisError: If configuration is invalid
        """
        try:
            # Validate configuration structure
            self.config_manager.validate_config(
                self.config_manager.load_config(self.config_path)
            )

            # Validate prompt files exist
            prompt_base_path = Path(self.config["prompt_base_path"])
            for prompt_file in self.config["prompt_files"]:
                prompt_path = prompt_base_path / prompt_file
                if not prompt_path.exists():
                    raise AnalysisError(f"Prompt file not found: {prompt_path}")

            # Validate wrapper file exists
            wrapper_path = Path(self.config["wrapper_path"])
            if not wrapper_path.exists():
                raise AnalysisError(f"Wrapper file not found: {wrapper_path}")

            # Validate examples directory exists (optional)
            examples_path = Path(self.config["examples_path"])
            if not examples_path.exists():
                self.logger.warning("Examples directory not found: %s", examples_path)

            return True

        except Exception as e:
            raise AnalysisError(f"Configuration validation failed: {e}") from e

    def _load_configuration(
        self, analyzer_type: str, config_path: Optional[str]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load configuration from local manifest or central config.

        Args:
            analyzer_type: Type of analyzer
            config_path: Optional path to config file

        Returns:
            Tuple of (analyzer_config, global_settings)
        """
        # Try local manifest first - resolve relative to badgers-foundation directory
        # This file is in badgers-foundation/foundation/, so parent is badgers-foundation/
        foundation_dir = Path(__file__).parent.parent
        manifest_path = (
            foundation_dir / "tools" / f"{analyzer_type}_analyzer" / "manifest.json"
        )

        if manifest_path.exists():
            try:
                import json

                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)

                # Handle both single and multi-analyzer manifests
                analyzer_config = None
                if "analyzer" in manifest:
                    # Single analyzer manifest
                    analyzer_config = manifest["analyzer"]
                elif "analyzers" in manifest:
                    # Multi-analyzer manifest (e.g., metadata analyzer)
                    # Try to find the matching analyzer by type
                    if analyzer_type in manifest["analyzers"]:
                        analyzer_config = manifest["analyzers"][analyzer_type]
                    else:
                        # Try to match by name
                        for _, config in manifest["analyzers"].items():
                            if config.get("name") == f"{analyzer_type}_analyzer":
                                analyzer_config = config
                                break

                if analyzer_config:
                    self.logger.info(
                        "Loaded configuration from local manifest: %s", manifest_path
                    )

                    # Paths in manifest are now relative to the tool directory
                    # Manifest is at tools/{analyzer}/manifest.json
                    manifest_dir = manifest_path.parent

                    for path_key in [
                        "prompt_base_path",
                        "prompt_analyzer_prompt_base_path",
                        "examples_path",
                        "wrapper_path",
                    ]:
                        if path_key in analyzer_config:
                            path_value = analyzer_config[path_key]
                            # Make absolute path from manifest directory
                            analyzer_config[path_key] = str(
                                (manifest_dir / path_value).resolve()
                            )

                    # Load global settings from central config or use defaults
                    try:
                        central_config_path = (
                            config_path if config_path else ANALYZER_CONFIG_PATH
                        )
                        global_settings = self.config_manager.get_global_settings(
                            central_config_path
                        )
                    except Exception:
                        # Use default global settings if central config not available
                        global_settings = {
                            "max_tokens": 8000,
                            "temperature": 0.1,
                            "max_image_size": 20971520,
                            "max_dimension": 2048,
                            "jpeg_quality": 85,
                            "cache_enabled": True,
                            "throttle_delay": 1.0,
                            "aws_region": "us-west-2",
                        }

                    return analyzer_config, global_settings
            except Exception as e:
                self.logger.warning(
                    "Failed to load from manifest %s: %s", manifest_path, e
                )

        # Fall back to central config
        self.config_path = config_path if config_path else ANALYZER_CONFIG_PATH
        self.logger.info("Using central analyzer config: %s", self.config_path)

        analyzer_config = self.config_manager.get_analyzer_config(
            analyzer_type, self.config_path
        )
        global_settings = self.config_manager.get_global_settings(self.config_path)

        return analyzer_config, global_settings

    def _configure_components(
        self,
    ) -> None:
        """Configure foundation components with global settings."""
        try:
            # Configure image processor
            if "max_image_size" in self.global_settings:
                self.image_processor.max_image_size = self.global_settings[
                    "max_image_size"
                ]
            if "max_dimension" in self.global_settings:
                self.image_processor.max_dimension = self.global_settings[
                    "max_dimension"
                ]
            if "jpeg_quality" in self.global_settings:
                self.image_processor.jpeg_quality = self.global_settings["jpeg_quality"]

            # Note: PromptLoader doesn't have cache_enabled attribute
            # Cache configuration is handled at the S3/local loader level

            # Configure bedrock client
            if "throttle_delay" in self.global_settings:
                self.bedrock_client.throttle_delay = self.global_settings[
                    "throttle_delay"
                ]
            if "aws_region" in self.global_settings:
                self.bedrock_client.aws_region = self.global_settings["aws_region"]

            self.logger.debug("Configured foundation components with global settings")

        except Exception as e:
            self.logger.warning("Failed to configure components: %s", e)

    def _get_image_dimensions(
        self, image_path: Union[str, bytes, bytearray]
    ) -> Dict[str, int]:
        """Get image dimensions for placeholder replacement."""
        try:
            self.logger.debug("Getting image dimensions")
            width, height = self.image_processor.get_image_dimensions(image_path)
            return {"width": width, "height": height}
        except Exception as e:
            self.logger.warning("Failed to get image dimensions: %s", e)
            return {"width": 0, "height": 0}

    def _get_image_dimensions_from_bytes(self, image_data: bytes) -> Dict[str, int]:
        """Get image dimensions from byte data."""
        try:
            self.logger.debug("Getting image dimensions from bytes")
            width, height = self.image_processor.get_image_dimensions(image_data)
            self.logger.debug(
                "_get_image_dimensions_from_bytes width: %d height: %d", width, height
            )
            return {"width": width, "height": height}
        except Exception as e:
            self.logger.warning("Failed to get image dimensions from bytes: %s", e)
            return {"width": 0, "height": 0}

    def _process_target_image(
        self, image_path: Union[str, bytes, bytearray]
    ) -> tuple[str, bytes]:
        """Process the target image and convert to base64, returning both base64 and optimized bytes."""
        try:
            self.logger.debug("Processing target image")
            base64_str, optimized_bytes = (
                self.image_processor.image_to_base64_with_bytes(image_path)
            )
            return base64_str, optimized_bytes
        except Exception as e:
            raise AnalysisError(f"Failed to process target image: {e}") from e

    def _load_system_prompt(
        self,
        image_dimensions: Optional[Dict[str, int]] = None,
        audit_mode: bool = False,
    ) -> str:
        """Load and combine system prompt files with optional placeholder replacement."""
        try:
            self.logger.debug("Loading system prompt (audit_mode=%s)", audit_mode)

            placeholders: Optional[Dict[str, str]] = None
            if image_dimensions:
                self.logger.debug(
                    "Replacing placeholder image dimensions: width: %d and height: %d",
                    image_dimensions["width"],
                    image_dimensions["height"],
                )
                placeholders = {
                    "PIXEL_WIDTH": str(image_dimensions["width"]),
                    "PIXEL_HEIGHT": str(image_dimensions["height"]),
                }
            return self.prompt_loader.load_system_prompt(
                self.config, placeholders, audit_mode
            )
        except Exception as e:
            raise AnalysisError(f"Failed to load system prompt: {e}") from e

    def _load_example_images(self) -> List[str]:
        """Load example images for few-shot learning."""
        try:
            self.logger.debug("Loading example images")
            return self.message_builder.load_example_images(
                self.config["examples_path"],
                self.config["max_examples"],
                self.image_processor,
            )
        except Exception as e:
            self.logger.warning("Failed to load example images: %s", e)
            return []  # Continue without examples

    def _build_message_chain(
        self, target_image: str, examples: List[str]
    ) -> List[Dict[str, Any]]:
        """Build the message chain for Bedrock invocation."""
        try:
            self.logger.debug("Building message chain")
            return self.message_builder.create_message_chain(
                target_image,
                examples,
                self.config["analysis_text"],
                self.config["max_examples"],
            )
        except Exception as e:
            raise AnalysisError(f"Failed to build message chain: {e}") from e

    def _invoke_bedrock_model(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        aws_profile: Optional[str],
    ) -> Dict[str, Any]:
        """Invoke the Bedrock model with the prepared payload."""
        try:
            self.logger.debug("Invoking Bedrock model")

            # Get model selection config (new format) or fall back to legacy format
            model_id, extended_thinking, budget_tokens, fallback_list = (
                self._get_model_selection()
            )

            # Validate all model IDs
            if model_id:
                self.bedrock_client.validate_model_id(model_id)
            for fallback in fallback_list:
                fb_model_id = (
                    fallback.get("model_id") if isinstance(fallback, dict) else fallback
                )
                if fb_model_id:
                    self.bedrock_client.validate_model_id(fb_model_id)

            if fallback_list:
                self.logger.debug("Fallback models configured: %s", fallback_list)

            # Create payload (in Claude format - will be converted if needed)
            payload = self.bedrock_client.create_anthropic_payload(
                system_prompt,
                messages,
                self.global_settings.get("max_tokens", 4000),
                self.global_settings.get("temperature", 0.1),
            )

            # Get retry config from manifest (default: 3)
            max_retries = self.config.get("max_retries", 3)

            # Invoke model with fallback support and extended thinking
            return self.bedrock_client.invoke_model(
                model_id,
                payload,
                aws_profile,
                fallback_list,
                max_retries,
                extended_thinking,
                budget_tokens,
            )

        except Exception as e:
            raise AnalysisError(f"Failed to invoke Bedrock model: {e}") from e

    def _get_model_selection(self) -> tuple[str, bool, Optional[int], list]:
        """
        Get primary model ID, extended thinking setting, budget tokens, and fallback list from config.

        Supports both new format (model_selections with objects) and legacy format (model_id/fallback_model_id).

        Returns:
            Tuple of (primary_model_id, extended_thinking, budget_tokens, fallback_list)
            fallback_list contains dicts with model_id, extended_thinking, and budget_tokens keys
        """
        # Check for new format: model_selections
        if "model_selections" in self.config:
            selections = self.config["model_selections"]

            # Handle primary model
            primary = selections.get("primary")
            if not primary:
                raise AnalysisError("model_selections.primary is required")

            if isinstance(primary, dict):
                primary_model_id = primary.get("model_id")
                primary_extended_thinking = primary.get("extended_thinking", False)
                primary_budget_tokens = primary.get("budget_tokens")
            else:
                # Legacy: primary is just a string
                primary_model_id = primary
                primary_extended_thinking = False
                primary_budget_tokens = None

            if not primary_model_id:
                raise AnalysisError("model_selections.primary.model_id is required")

            # Handle fallback list
            fallback_list = selections.get("fallback_list", [])
            # Normalize fallback list to always be list of dicts
            normalized_fallbacks = []
            for fb in fallback_list:
                if isinstance(fb, dict):
                    normalized_fallbacks.append(
                        {
                            "model_id": fb.get("model_id"),
                            "extended_thinking": fb.get("extended_thinking", False),
                            "budget_tokens": fb.get("budget_tokens"),
                        }
                    )
                else:
                    # Legacy: fallback is just a string
                    normalized_fallbacks.append(
                        {
                            "model_id": fb,
                            "extended_thinking": False,
                            "budget_tokens": None,
                        }
                    )

            return (
                primary_model_id,
                primary_extended_thinking,
                primary_budget_tokens,
                normalized_fallbacks,
            )

        # Legacy format: model_id + optional fallback_model_id
        model_id = self.config.get("model_id")
        if not model_id:
            raise AnalysisError("model_id or model_selections.primary is required")

        fallback_list = []
        fallback_model_id = self.config.get("fallback_model_id")
        if fallback_model_id:
            fallback_list = [
                {
                    "model_id": fallback_model_id,
                    "extended_thinking": False,
                    "budget_tokens": None,
                }
            ]

        return model_id, False, None, fallback_list

    def _process_response(self, response: Dict[str, Any]) -> str:
        """Process the Bedrock response and extract the result."""
        try:
            self.logger.debug("Processing Bedrock response")
            return self.response_processor.extract_analysis_result(response)
        except Exception as e:
            # Try to handle empty response gracefully
            try:
                return self.response_processor.handle_empty_response()
            except Exception:
                raise AnalysisError(f"Failed to process response: {e}") from e

    def _finalize_result(self, result: str) -> str:
        """Validate and format the final result."""
        try:
            # Validate result quality
            quality_metrics = self.response_processor.validate_analysis_quality(result)

            if quality_metrics["quality_score"] < 0.3:
                self.logger.warning(
                    "Low quality analysis result (score: %.2f)",
                    quality_metrics["quality_score"],
                )

            # Format result if needed
            formatted_result = self.response_processor.format_analysis_result(
                result, self.analyzer_type, include_metadata=False
            )

            return formatted_result

        except Exception as e:
            self.logger.warning("Failed to finalize result: %s", e)
            return result  # Return raw result if formatting fails
