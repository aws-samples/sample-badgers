"""
Configuration management for analyzer system.

Handles loading, validation, and access to analyzer configurations.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


class ConfigurationManager:
    """Manages analyzer configuration loading and validation."""

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

    def get_analyzer_config(
        self,
        analyzer_type: str,
        config_path: str = "",
    ) -> Dict[str, Any]:
        """
        Get configuration for a specific analyzer type.

        Args:
            analyzer_type: The type of analyzer (e.g., 'diagram', 'table')
            config_path: Path to the configuration JSON file

        Returns:
            Dictionary containing the analyzer-specific configuration

        Raises:
            ConfigurationError: If analyzer type is not found in configuration
        """
        config = self.load_config(config_path)

        if "analyzers" not in config:
            raise ConfigurationError("Configuration missing 'analyzers' section")

        if analyzer_type not in config["analyzers"]:
            available_types = list(config["analyzers"].keys())
            raise ConfigurationError(
                f"Analyzer type '{analyzer_type}' not found. "
                f"Available types: {available_types}"
            )

        return config["analyzers"][analyzer_type]

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

        if "analyzers" not in config:
            raise ConfigurationError(
                "Configuration missing required 'analyzers' section"
            )

        if not isinstance(config["analyzers"], dict):
            raise ConfigurationError("'analyzers' section must be a dictionary")

        # Validate each analyzer configuration
        for analyzer_type, analyzer_config in config["analyzers"].items():
            self._validate_analyzer_config(analyzer_type, analyzer_config)

        # Validate global settings if present
        if "global_settings" in config:
            self._validate_global_settings(config["global_settings"])

        return True

    def _validate_analyzer_config(
        self, analyzer_type: str, analyzer_config: Dict[str, Any]
    ) -> None:
        """Validate individual analyzer configuration."""
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
            if field not in analyzer_config:
                raise ConfigurationError(
                    f"Analyzer '{analyzer_type}' missing required field: {field}"
                )

        # Validate field types
        if not isinstance(analyzer_config["prompt_files"], list):
            raise ConfigurationError(
                f"Analyzer '{analyzer_type}': 'prompt_files' must be a list"
            )

        if (
            not isinstance(analyzer_config["max_examples"], int)
            or analyzer_config["max_examples"] < 0
        ):
            raise ConfigurationError(
                f"Analyzer '{analyzer_type}': 'max_examples' must be a non-negative integer"
            )

        # Validate pdf_processor specific settings
        if analyzer_type == "pdf_processor":
            self._validate_pdf_processor_config(analyzer_config)

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
