"""
Configuration management for specialist system.

Handles loading, validation, and access to specialist configurations.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


class ConfigurationManager:
    """Manages specialist configuration loading and validation."""

    def __init__(self):
        self._config_cache: Optional[Dict[str, Any]] = None
        self._config_path: Optional[str] = None

    def load_config(self, config_path: str = "") -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to the configuration JSON file

        Returns:
            Dictionary containing the full configuration

        Raises:
            ConfigurationError: If config file cannot be loaded or is invalid
        """
        if self._config_cache is not None and self._config_path == config_path:
            return self._config_cache

        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")

            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.validate_config(config)
            self._config_cache = config
            self._config_path = config_path
            return config

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def get_specialist_config(
        self,
        specialist_type: str,
        config_path: str = "",
    ) -> Dict[str, Any]:
        """
        Get configuration for a specific specialist type.

        Args:
            specialist_type: The type of specialist (e.g., 'diagram', 'table')
            config_path: Path to the configuration JSON file

        Returns:
            Dictionary containing the specialist-specific configuration

        Raises:
            ConfigurationError: If specialist type is not found in configuration
        """
        config = self.load_config(config_path)

        if "specialists" not in config:
            raise ConfigurationError("Configuration missing 'specialists' section")

        if specialist_type not in config["specialists"]:
            available_types = list(config["specialists"].keys())
            raise ConfigurationError(
                f"Specialist type '{specialist_type}' not found. "
                f"Available types: {available_types}"
            )

        return config["specialists"][specialist_type]

    def get_global_settings(self, config_path: str = "") -> Dict[str, Any]:
        """
        Get global settings from configuration.

        Args:
            config_path: Path to the configuration JSON file

        Returns:
            Dictionary containing global settings
        """
        config = self.load_config(config_path)
        return config.get("global_settings", {})

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and required fields.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Check top-level structure
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")

        if "specialists" not in config:
            raise ConfigurationError(
                "Configuration missing required 'specialists' section"
            )

        if not isinstance(config["specialists"], dict):
            raise ConfigurationError("'specialists' section must be a dictionary")

        # Validate each specialist configuration
        for specialist_type, specialist_config in config["specialists"].items():
            self._validate_specialist_config(specialist_type, specialist_config)

        # Validate global settings if present
        if "global_settings" in config:
            self._validate_global_settings(config["global_settings"])

        return True

    def _validate_specialist_config(
        self, specialist_type: str, specialist_config: Dict[str, Any]
    ) -> None:
        """Validate individual specialist configuration."""
        required_fields = [
            "name",
            "description",
            "model_id",
            "prompt_base_path",
            "prompt_files",
            "examples_path",
            "max_examples",
            "analysis_text",
            "wrapper_path",
        ]

        for field in required_fields:
            if field not in specialist_config:
                raise ConfigurationError(
                    f"Specialist '{specialist_type}' missing required field: {field}"
                )

        # Validate field types
        if not isinstance(specialist_config["prompt_files"], list):
            raise ConfigurationError(
                f"Specialist '{specialist_type}': 'prompt_files' must be a list"
            )

        if (
            not isinstance(specialist_config["max_examples"], int)
            or specialist_config["max_examples"] < 0
        ):
            raise ConfigurationError(
                f"Specialist '{specialist_type}': 'max_examples' must be a non-negative integer"
            )

        # Validate pdf_processor specific settings
        if specialist_type == "pdf_processor":
            self._validate_pdf_processor_config(specialist_config)

    def _validate_global_settings(self, global_settings: Dict[str, Any]) -> None:
        """Validate global settings configuration."""
        numeric_fields = [
            "max_tokens",
            "temperature",
            "max_image_size",
            "max_dimension",
            "jpeg_quality",
            "throttle_delay",
        ]

        for field in numeric_fields:
            if field in global_settings:
                value = global_settings[field]
                if not isinstance(value, (int, float)):
                    raise ConfigurationError(
                        f"Global setting '{field}' must be numeric, got {type(value)}"
                    )

        # Validate specific ranges
        if "temperature" in global_settings:
            temp = global_settings["temperature"]
            if not 0 <= temp <= 1:
                raise ConfigurationError("Temperature must be between 0 and 1")

        if "jpeg_quality" in global_settings:
            quality = global_settings["jpeg_quality"]
            if not 1 <= quality <= 100:
                raise ConfigurationError("JPEG quality must be between 1 and 100")

    def _validate_pdf_processor_config(self, config: Dict[str, Any]) -> None:
        """Validate pdf_processor specific configuration settings."""
        # Validate classification confidence threshold
        if "classification_confidence_threshold" in config:
            threshold = config["classification_confidence_threshold"]
            if not isinstance(threshold, (int, float)):
                raise ConfigurationError(
                    "classification_confidence_threshold must be numeric"
                )
            if not 0 <= threshold <= 1:
                raise ConfigurationError(
                    "classification_confidence_threshold must be between 0 and 1"
                )

        # Validate fallback analysis setting
        if "enable_fallback_analysis" in config:
            if not isinstance(config["enable_fallback_analysis"], bool):
                raise ConfigurationError("enable_fallback_analysis must be a boolean")

        # Validate default task timeout
        if "default_task_timeout" in config:
            timeout = config["default_task_timeout"]
            if not isinstance(timeout, (int, float)):
                raise ConfigurationError("default_task_timeout must be numeric")
            if timeout <= 0:
                raise ConfigurationError("default_task_timeout must be positive")

        # Validate tool-specific timeouts
        if "task_timeouts" in config:
            timeouts = config["task_timeouts"]
            if not isinstance(timeouts, dict):
                raise ConfigurationError("task_timeouts must be a dictionary")

            for tool_name, timeout in timeouts.items():
                if not isinstance(timeout, (int, float)):
                    raise ConfigurationError(
                        f"Timeout for tool '{tool_name}' must be numeric"
                    )
                if timeout <= 0:
                    raise ConfigurationError(
                        f"Timeout for tool '{tool_name}' must be positive"
                    )
